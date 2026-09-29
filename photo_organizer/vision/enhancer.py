"""Adaptive image enhancement with controlled file-size constraint."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

try:
    from PIL import (
        Image,
        ImageEnhance,
        ImageOps,
        ImageStat,
        UnidentifiedImageError,
    )
except ImportError:
    Image = None
    ImageEnhance = None
    ImageOps = None
    ImageStat = None

    class UnidentifiedImageError(Exception):
        pass

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from photo_organizer.core.file_ops import (
    allowed_enhanced_size,
    overwrite_existing_file_from_temp,
    restore_access_modified_times,
)
from photo_organizer.models import EnhancementPlan, ImageQualityMetrics
from photo_organizer.vision.classifier import (
    analysis_image,
    clamp,
    color_saturation_score_from_rgb,
    noise_score,
    sharpness_score,
    text_heaviness_score_from_prepared,
)


def image_quality_metrics(image: "Image.Image") -> ImageQualityMetrics:
    working = analysis_image(image, max_dimension=900)
    gray = ImageOps.grayscale(working)
    stat = ImageStat.Stat(gray)
    text_score, _ = text_heaviness_score_from_prepared(working, gray)

    return ImageQualityMetrics(
        brightness=float(stat.mean[0]),
        contrast=float(stat.stddev[0]),
        sharpness=sharpness_score(gray),
        saturation=color_saturation_score_from_rgb(working),
        noise=noise_score(gray),
        text_score=text_score,
    )


def build_enhancement_plan(metrics: ImageQualityMetrics) -> EnhancementPlan:
    if metrics.text_score >= 0.72:
        return EnhancementPlan(False, f"text-heavy image skipped ({metrics.text_score:.2f})")

    reasons = []
    plan = EnhancementPlan(False, "already good")

    if metrics.brightness < 82:
        plan.gamma = 0.88 if metrics.brightness >= 55 else 0.80
        plan.brightness_factor = 1.03
        reasons.append("underexposed")
    elif metrics.brightness > 190:
        plan.gamma = 1.08
        reasons.append("bright highlights")

    if metrics.contrast < 34:
        plan.autocontrast_cutoff = 0.3
        plan.contrast_factor = 1.06
        reasons.append("low contrast")
    elif metrics.contrast < 44:
        plan.autocontrast_cutoff = 0.15
        plan.contrast_factor = 1.03
        reasons.append("slightly low contrast")

    if metrics.saturation < 30 and metrics.text_score < 0.45:
        plan.color_factor = 1.05
        reasons.append("muted color")
    elif metrics.saturation > 92:
        plan.color_factor = 0.98
        reasons.append("very saturated color")

    if metrics.noise >= 9 and metrics.sharpness < 180:
        plan.denoise = True
        reasons.append("visible noise")

    if metrics.sharpness < 55 and metrics.noise < 14 and metrics.text_score < 0.55:
        plan.sharpness_factor = 1.10
        reasons.append("soft image")
    elif metrics.sharpness < 90 and metrics.noise < 10 and metrics.text_score < 0.45:
        plan.sharpness_factor = 1.05
        reasons.append("slightly soft image")

    if not reasons:
        return plan

    plan.should_enhance = True
    plan.reason = ", ".join(reasons)
    return plan


def apply_gamma(image: "Image.Image", gamma: float) -> "Image.Image":
    if gamma == 1.0:
        return image

    lookup = [int(clamp(((value / 255) ** gamma) * 255, 0, 255)) for value in range(256)]
    if image.mode == "RGBA":
        red, green, blue, alpha = image.split()
        merged = Image.merge(
            "RGB",
            (
                red.point(lookup),
                green.point(lookup),
                blue.point(lookup),
            ),
        )
        merged.putalpha(alpha)
        return merged

    return image.point(lookup * len(image.getbands()))


def denoise_rgb(image: "Image.Image") -> "Image.Image":
    if cv2 is None or np is None:
        return image

    has_alpha = image.mode == "RGBA"
    alpha = image.getchannel("A") if has_alpha else None
    rgb = image.convert("RGB")
    array = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(array, None, 3, 3, 7, 21)
    result = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    if alpha is not None:
        result.putalpha(alpha)
    return result


def enhancement_plan_for_image(image: "Image.Image") -> EnhancementPlan:
    return build_enhancement_plan(image_quality_metrics(image))


def apply_enhancement_plan(
    image: "Image.Image", plan: EnhancementPlan, enhancement_mode: str
) -> "Image.Image":
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )

    if has_alpha:
        working = image.convert("RGBA")
        rgb, alpha = working.convert("RGB"), working.getchannel("A")
    else:
        rgb = image.convert("RGB")
        alpha = None

    if cv2 is not None and np is not None:
        rgb_arr = np.array(rgb)
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 1. Luminance gamma curve
        if plan.gamma != 1.0:
            table = np.array([int(clamp(((v / 255.0) ** plan.gamma) * 255.0, 0, 255)) for v in range(256)]).astype("uint8")
            l = cv2.LUT(l, table)

        # 2. Local adaptive contrast via CLAHE on L (Luminance) channel only
        if plan.contrast_factor > 1.0 or plan.autocontrast_cutoff > 0:
            clip = min(2.4, max(1.2, 1.2 * plan.contrast_factor))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l = clahe.apply(l)

        # 3. Photographic unsharp masking on L channel
        if plan.sharpness_factor > 1.0:
            blurred = cv2.GaussianBlur(l, (0, 0), sigmaX=1.2)
            high_freq = cv2.subtract(l, blurred)
            boost = min(0.45, (plan.sharpness_factor - 1.0) * 3.0)
            l = cv2.addWeighted(l, 1.0, high_freq, boost, 0)

        # 4. Gentle brightness adjustment
        if plan.brightness_factor != 1.0:
            l = np.clip(l.astype(np.float32) * plan.brightness_factor, 0, 255).astype(np.uint8)

        # 5. Safe color vibrance (preserves hue without color distortion)
        if plan.color_factor != 1.0:
            a = np.clip((a.astype(np.float32) - 128.0) * plan.color_factor + 128.0, 0, 255).astype(np.uint8)
            b = np.clip((b.astype(np.float32) - 128.0) * plan.color_factor + 128.0, 0, 255).astype(np.uint8)

        merged_lab = cv2.merge((l, a, b))
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(enhanced_rgb)
    else:
        # Pillow fallback
        rgb = apply_gamma(rgb, plan.gamma)
        if plan.autocontrast_cutoff:
            rgb = ImageOps.autocontrast(rgb, cutoff=plan.autocontrast_cutoff)
        if plan.brightness_factor != 1.0:
            rgb = ImageEnhance.Brightness(rgb).enhance(plan.brightness_factor)
        if plan.color_factor != 1.0:
            rgb = ImageEnhance.Color(rgb).enhance(plan.color_factor)
        if plan.contrast_factor != 1.0:
            rgb = ImageEnhance.Contrast(rgb).enhance(plan.contrast_factor)
        if plan.sharpness_factor != 1.0:
            rgb = ImageEnhance.Sharpness(rgb).enhance(plan.sharpness_factor)
        result = rgb

    if plan.denoise and enhancement_mode == "best":
        result = denoise_rgb(result)

    if alpha is not None:
        result.putalpha(alpha)
    return result


def enhanced_image(
    image: "Image.Image", enhancement_mode: str = "balanced"
) -> Tuple[Optional["Image.Image"], str]:
    plan = enhancement_plan_for_image(image)
    if not plan.should_enhance:
        return None, plan.reason
    return apply_enhancement_plan(image, plan, enhancement_mode), plan.reason


def enhancement_quality_steps(
    format_name: str, enhancement_mode: str
) -> Sequence[Optional[int]]:
    if format_name.upper() not in {"JPEG", "JPG", "WEBP"}:
        return (None,)
    if enhancement_mode == "fast":
        return (95, 88)
    if enhancement_mode == "best":
        return (100, 98, 96, 95, 92, 90, 88, 85)
    return (98, 95, 90, 85)


def save_candidate_bytes(
    image: "Image.Image",
    format_name: str,
    exif: Optional[bytes],
    icc_profile: Optional[bytes],
    jpeg_quality: Optional[int] = None,
    enhancement_mode: str = "balanced",
) -> bytes:
    image_to_save = image
    save_kwargs = {}
    if exif:
        save_kwargs["exif"] = exif
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    normalized_format = format_name.upper()
    if normalized_format in {"JPEG", "JPG"}:
        if image_to_save.mode != "RGB":
            image_to_save = image_to_save.convert("RGB")
        save_kwargs.update(
            {
                "format": "JPEG",
                "quality": jpeg_quality or 95,
                "optimize": enhancement_mode != "fast",
                "progressive": enhancement_mode == "best",
            }
        )
    elif normalized_format == "PNG":
        compress_level = {"fast": 3, "balanced": 6, "best": 9}[enhancement_mode]
        save_kwargs.update(
            {
                "format": "PNG",
                "optimize": enhancement_mode != "fast",
                "compress_level": compress_level,
            }
        )
    elif normalized_format == "WEBP":
        method = {"fast": 2, "balanced": 4, "best": 6}[enhancement_mode]
        save_kwargs.update(
            {"format": "WEBP", "quality": jpeg_quality or 95, "method": method}
        )
    elif normalized_format in {"TIFF", "TIF"}:
        save_kwargs.update({"format": "TIFF", "compression": "tiff_lzw"})
    else:
        save_kwargs["format"] = format_name

    buffer = io.BytesIO()
    image_to_save.save(buffer, **save_kwargs)
    return buffer.getvalue()


def save_candidate(
    image: "Image.Image",
    temp_path: Path,
    format_name: str,
    exif: Optional[bytes],
    icc_profile: Optional[bytes],
    jpeg_quality: Optional[int] = None,
    enhancement_mode: str = "balanced",
) -> None:
    temp_path.write_bytes(
        save_candidate_bytes(
            image,
            format_name,
            exif,
            icc_profile,
            jpeg_quality,
            enhancement_mode,
        )
    )


def enhance_in_place(
    path: Path,
    dry_run: bool,
    max_size_increase_percent: float,
    enhancement_mode: str = "balanced",
) -> Tuple[bool, str]:
    if Image is None:
        return False, "Pillow is not installed"

    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif", ".gif", ".bmp"}:
        return False, f"enhancement skipped for {suffix}"

    try:
        original_stat = path.stat()
        max_allowed_size = allowed_enhanced_size(
            original_stat.st_size, max_size_increase_percent
        )
        with Image.open(path) as image:
            if getattr(image, "is_animated", False):
                return False, "animated image skipped"

            format_name = image.format or path.suffix.lstrip(".").upper()
            exif = image.info.get("exif")
            icc_profile = image.info.get("icc_profile")
            plan = enhancement_plan_for_image(image)
            if not plan.should_enhance:
                return False, plan.reason

            if dry_run:
                return (
                    True,
                    f"would enhance {plan.reason} if output is no more than {max_size_increase_percent:g}% larger",
                )

            candidate = apply_enhancement_plan(image, plan, enhancement_mode)

            # Objective quality guardrails: prevent highlight blowout or shadow crush
            if np is not None:
                orig_arr = np.array(image.convert("RGB"))
                cand_arr = np.array(candidate.convert("RGB"))
                orig_blown = float((orig_arr >= 252).mean())
                cand_blown = float((cand_arr >= 252).mean())
                if cand_blown - orig_blown > 0.025:
                    return False, "enhancement would clip bright highlights"
                orig_crushed = float((orig_arr <= 4).mean())
                cand_crushed = float((cand_arr <= 4).mean())
                if cand_crushed - orig_crushed > 0.025:
                    return False, "enhancement would crush shadow details"

            quality_steps = enhancement_quality_steps(format_name, enhancement_mode)

            best_bytes: Optional[bytes] = None
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.stem}.",
                suffix=path.suffix,
                dir=str(path.parent),
                delete=False,
            ) as temp_handle:
                temp_path = Path(temp_handle.name)

            try:
                for quality in quality_steps:
                    encoded = save_candidate_bytes(
                        candidate,
                        format_name,
                        exif,
                        icc_profile,
                        quality,
                        enhancement_mode,
                    )
                    if len(encoded) <= max_allowed_size:
                        best_bytes = encoded
                        break

                if best_bytes is None:
                    if temp_path.exists():
                        temp_path.unlink()
                    return (
                        False,
                        f"enhanced file would exceed {max_size_increase_percent:g}% size increase",
                    )

                temp_path.write_bytes(best_bytes)
                overwrite_existing_file_from_temp(path, temp_path)
                restore_access_modified_times(path, original_stat)
                return True, f"enhanced: {plan.reason}"
            finally:
                if temp_path.exists():
                    temp_path.unlink()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return False, f"enhancement failed: {exc}"


def enhance_path_for_worker(
    item: Tuple[Path, bool, float, str]
) -> Tuple[Path, bool, str]:
    path, dry_run, max_size_increase_percent, enhancement_mode = item
    enhanced, reason = enhance_in_place(
        path,
        dry_run,
        max_size_increase_percent,
        enhancement_mode,
    )
    return path, enhanced, reason
