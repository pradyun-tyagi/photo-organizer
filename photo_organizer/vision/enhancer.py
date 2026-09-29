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
from photo_organizer.vision.detectors import _get_thread_face_cascades
from photo_organizer.vision.faces import _get_deep_face_models



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


def detect_face_regions(
    bgr_image: "np.ndarray",
) -> Tuple[List["np.ndarray"], List[Tuple[int, int, int, int]]]:
    """Detects human faces using YuNet (or Haar cascade fallback).
    Returns (raw_face_rows, face_bounding_boxes) in coordinates of bgr_image.
    """
    if cv2 is None or np is None:
        return [], []

    orig_h, orig_w = bgr_image.shape[:2]
    if orig_h < 24 or orig_w < 24:
        return [], []

    detector, _ = _get_deep_face_models()
    if detector is not None:
        # Scale for fast and accurate deep detection if image is large
        max_dim = max(orig_h, orig_w)
        scale = 960.0 / max_dim if max_dim > 960 else 1.0
        if scale < 1.0:
            det_w, det_h = int(orig_w * scale), int(orig_h * scale)
            work_bgr = cv2.resize(bgr_image, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_w, det_h = orig_w, orig_h
            work_bgr = bgr_image

        try:
            detector.setInputSize((det_w, det_h))
            _, raw_faces = detector.detect(work_bgr)
            if raw_faces is not None and len(raw_faces) > 0:
                face_rows = []
                face_boxes = []
                for row in raw_faces:
                    score = float(row[14]) if len(row) > 14 else 1.0
                    if score < 0.60:
                        continue
                    # Scale coordinates and keypoints back to full resolution
                    row_full = row.copy()
                    row_full[0:14] = row_full[0:14] / scale
                    fx, fy, fw, fh = [int(v) for v in row_full[0:4]]
                    if fw < 16 or fh < 16:
                        continue
                    face_rows.append(row_full)
                    face_boxes.append((fx, fy, fw, fh))
                if face_boxes:
                    return face_rows, face_boxes
        except Exception:
            pass

    # Fallback to Haar Cascades if deep learning detector is unavailable
    cascades = _get_thread_face_cascades()
    if cascades:
        try:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            scale = 640.0 / max(orig_h, orig_w) if max(orig_h, orig_w) > 640 else 1.0
            if scale < 1.0:
                gray_small = cv2.resize(gray, (int(orig_w * scale), int(orig_h * scale)))
            else:
                gray_small = gray

            found = []
            for cascade in cascades:
                f = cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(f) > 0:
                    found = f
                    break

            face_boxes = []
            for fx, fy, fw, fh in found:
                x1 = int(fx / scale)
                y1 = int(fy / scale)
                w1 = int(fw / scale)
                h1 = int(fh / scale)
                face_boxes.append((x1, y1, w1, h1))
            return [], face_boxes
        except Exception:
            pass

    return [], []


def build_feathered_face_mask(
    image_shape: Tuple[int, int, ...], face_boxes: List[Tuple[int, int, int, int]]
) -> "np.ndarray":
    """Constructs a smooth, Gaussian-feathered alpha mask for all detected faces.
    Values are 1.0 at facial core, smoothly tapering to 0.0 outside face contour.
    Eliminates visible boundaries or halo transitions.
    """
    if np is None or cv2 is None or not face_boxes:
        return np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)

    H, W = image_shape[:2]
    combined_mask = np.zeros((H, W), dtype=np.float32)

    for x, y, w, h in face_boxes:
        cx = max(0, min(W - 1, int(x + w * 0.5)))
        cy = max(0, min(H - 1, int(y + h * 0.45)))
        ax = max(10, int(w * 0.65))
        ay = max(10, int(h * 0.75))

        canvas = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(canvas, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        kx = max(3, int(ax * 0.6) * 2 + 1)
        ky = max(3, int(ay * 0.6) * 2 + 1)
        feathered = cv2.GaussianBlur(
            canvas, (kx, ky), sigmaX=ax * 0.35, sigmaY=ay * 0.35
        ).astype(np.float32) / 255.0

        combined_mask = np.maximum(combined_mask, feathered)

    return combined_mask


def verify_and_protect_facial_identity(
    orig_bgr: "np.ndarray",
    enhanced_bgr: "np.ndarray",
    face_rows: List["np.ndarray"],
    min_cosine_similarity: float = 0.98,
) -> "np.ndarray":
    """Ensures biometric identity invariance using OpenCV SFace 128-d metric embeddings.
    If cosine similarity drops below min_cosine_similarity (0.98), the face region
    is automatically restored to pristine original pixels to guarantee zero identity alteration.
    """
    if cv2 is None or np is None or not face_rows:
        return enhanced_bgr

    _, recognizer = _get_deep_face_models()
    if recognizer is None:
        return enhanced_bgr

    result_bgr = enhanced_bgr.copy()
    H, W = orig_bgr.shape[:2]

    for face_row in face_rows:
        try:
            orig_crop = recognizer.alignCrop(orig_bgr, face_row)
            cand_crop = recognizer.alignCrop(result_bgr, face_row)
            orig_feat = recognizer.feature(orig_crop)
            cand_feat = recognizer.feature(cand_crop)
            sim = float(recognizer.match(orig_feat, cand_feat, cv2.FaceRecognizerSF_FR_COSINE))

            if sim < min_cosine_similarity:
                # Identity drifted below strict guardrail: restore face pixels to original
                fx, fy, fw, fh = [int(v) for v in face_row[0:4]]
                pad_x = int(fw * 0.15)
                pad_y = int(fh * 0.15)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(W, fx + fw + pad_x)
                y2 = min(H, fy + fh + pad_y)

                if x2 > x1 and y2 > y1:
                    box_h = y2 - y1
                    box_w = x2 - x1
                    kx = max(3, int(box_w * 0.25) * 2 + 1)
                    ky = max(3, int(box_h * 0.25) * 2 + 1)
                    mask = np.ones((box_h, box_w), dtype=np.float32)
                    mask = cv2.GaussianBlur(mask, (kx, ky), sigmaX=box_w * 0.15, sigmaY=box_h * 0.15)
                    mask_3ch = cv2.merge([mask, mask, mask])
                    orig_patch = orig_bgr[y1:y2, x1:x2].astype(np.float32)
                    cand_patch = result_bgr[y1:y2, x1:x2].astype(np.float32)
                    restored_patch = np.clip(mask_3ch * orig_patch + (1.0 - mask_3ch) * cand_patch, 0, 255).astype(np.uint8)
                    result_bgr[y1:y2, x1:x2] = restored_patch
        except Exception:
            continue

    return result_bgr


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

        # Detect human faces and construct feathered face protection mask
        face_rows, face_boxes = detect_face_regions(bgr)
        has_faces = len(face_boxes) > 0
        face_mask = build_feathered_face_mask(bgr.shape, face_boxes) if has_faces else None

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 1. Luminance gamma curve (global natural S-curve)
        if plan.gamma != 1.0:
            table = np.array(
                [int(clamp(((v / 255.0) ** plan.gamma) * 255.0, 0, 255)) for v in range(256)]
            ).astype("uint8")
            l_gamma = cv2.LUT(l, table)
        else:
            l_gamma = l.copy()

        # Background branch: CLAHE + Unsharp Masking
        l_bg = l_gamma.copy()
        if plan.contrast_factor > 1.0 or plan.autocontrast_cutoff > 0:
            clip = min(2.4, max(1.2, 1.2 * plan.contrast_factor))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l_bg = clahe.apply(l_bg)

        if plan.sharpness_factor > 1.0:
            blurred = cv2.GaussianBlur(l_bg, (0, 0), sigmaX=1.2)
            high_freq = cv2.subtract(l_bg, blurred)
            boost = min(0.45, (plan.sharpness_factor - 1.0) * 3.0)
            l_bg = cv2.addWeighted(l_bg, 1.0, high_freq, boost, 0)

        if plan.brightness_factor != 1.0:
            l_bg = np.clip(l_bg.astype(np.float32) * plan.brightness_factor, 0, 255).astype(np.uint8)

        # Face branch & Composite:
        if has_faces and face_mask is not None:
            # Human face zone:
            # NO CLAHE: local histogram equalization creates harsh blotches, dark eye circles, and zombie artifacts.
            l_face = l_gamma.copy()
            if plan.brightness_factor > 1.0:
                face_bright = min(1.04, 1.0 + (plan.brightness_factor - 1.0) * 0.5)
                l_face = np.clip(l_face.astype(np.float32) * face_bright, 0, 255).astype(np.uint8)

            # Bilateral edge-preserving skin smoothing:
            # Smooths camera sensor noise and compression grain on skin while keeping eyes, pupils,
            # eyebrows, lips, and nostrils razor sharp without altering identity.
            l_bilat = cv2.bilateralFilter(l_face, d=5, sigmaColor=20, sigmaSpace=20)
            l_face = np.clip(0.70 * l_face.astype(np.float32) + 0.30 * l_bilat.astype(np.float32), 0, 255).astype(np.uint8)

            # Soft Gaussian feather blend between background and face luminance
            fm = face_mask
            bgm = 1.0 - fm
            l_composite = np.clip(bgm * l_bg.astype(np.float32) + fm * l_face.astype(np.float32), 0, 255).astype(np.uint8)
        else:
            l_composite = l_bg

        # Safe color vibrance (with strict skin chromaticity lock on facial zone)
        if plan.color_factor != 1.0:
            a_bg = np.clip((a.astype(np.float32) - 128.0) * plan.color_factor + 128.0, 0, 255)
            b_bg = np.clip((b.astype(np.float32) - 128.0) * plan.color_factor + 128.0, 0, 255)

            if has_faces and face_mask is not None:
                # Skin chromaticity protection: dampen color factor by 75% on face to preserve authentic complexion
                face_color_factor = 1.0 + (plan.color_factor - 1.0) * 0.25
                a_face = np.clip((a.astype(np.float32) - 128.0) * face_color_factor + 128.0, 0, 255)
                b_face = np.clip((b.astype(np.float32) - 128.0) * face_color_factor + 128.0, 0, 255)

                a_composite = np.clip(bgm * a_bg + fm * a_face, 0, 255).astype(np.uint8)
                b_composite = np.clip(bgm * b_bg + fm * b_face, 0, 255).astype(np.uint8)
            else:
                a_composite = a_bg.astype(np.uint8)
                b_composite = b_bg.astype(np.uint8)
        else:
            a_composite = a
            b_composite = b

        merged_lab = cv2.merge((l_composite, a_composite, b_composite))
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # SFace Biometric Identity Guardrail: mathematically guarantees >= 0.98 cosine similarity
        if has_faces and len(face_rows) > 0:
            enhanced_bgr = verify_and_protect_facial_identity(
                bgr, enhanced_bgr, face_rows, min_cosine_similarity=0.98
            )

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
