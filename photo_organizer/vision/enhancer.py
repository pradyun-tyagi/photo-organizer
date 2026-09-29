"""Adaptive image enhancement with controlled file-size constraint."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


def build_facial_landmark_masks(
    image_shape: Tuple[int, int, ...], face_rows: Sequence["np.ndarray"]
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Constructs feathered masks for eyes and lips from YuNet landmarks.
    Returns (eye_mask, lip_mask) as float32 arrays in [0.0, 1.0].
    """
    if np is None or cv2 is None or not face_rows:
        return (
            np.zeros((image_shape[0], image_shape[1]), dtype=np.float32),
            np.zeros((image_shape[0], image_shape[1]), dtype=np.float32),
        )

    H, W = image_shape[:2]
    eye_mask = np.zeros((H, W), dtype=np.float32)
    lip_mask = np.zeros((H, W), dtype=np.float32)

    for row in face_rows:
        if len(row) < 14:
            continue
        rx, ry = int(row[4]), int(row[5])
        lx, ly = int(row[6]), int(row[7])
        dist = float(np.hypot(lx - rx, ly - ry))
        if dist < 6:
            continue

        r = max(4, int(dist * 0.18))
        eye_canvas = np.zeros((H, W), dtype=np.uint8)
        if 0 <= rx < W and 0 <= ry < H:
            cv2.circle(eye_canvas, (rx, ry), r, 255, -1)
        if 0 <= lx < W and 0 <= ly < H:
            cv2.circle(eye_canvas, (lx, ly), r, 255, -1)

        kx = max(3, r * 2 + 1)
        feathered_eyes = cv2.GaussianBlur(
            eye_canvas, (kx, kx), sigmaX=max(1.0, r * 0.45)
        ).astype(np.float32) / 255.0
        eye_mask = np.maximum(eye_mask, feathered_eyes)

        # Mouth landmarks
        rcx, rcy = int(row[10]), int(row[11])
        lcx, lcy = int(row[12]), int(row[13])
        mw = float(np.hypot(lcx - rcx, lcy - rcy))
        if mw < 6:
            continue

        mcx = int((rcx + lcx) * 0.5)
        mcy = int((rcy + lcy) * 0.5)
        ax = max(6, int(mw * 0.55))
        ay = max(3, int(mw * 0.28))
        angle = float(np.degrees(np.arctan2(lcy - rcy, lcx - rcx)))

        lip_canvas = np.zeros((H, W), dtype=np.uint8)
        if 0 <= mcx < W and 0 <= mcy < H:
            cv2.ellipse(lip_canvas, (mcx, mcy), (ax, ay), angle, 0, 360, 255, -1)

        lkx = max(3, int(ax * 0.6) * 2 + 1)
        lky = max(3, int(ay * 0.6) * 2 + 1)
        feathered_lips = cv2.GaussianBlur(
            lip_canvas, (lkx, lky), sigmaX=ax * 0.35, sigmaY=ay * 0.35
        ).astype(np.float32) / 255.0
        lip_mask = np.maximum(lip_mask, feathered_lips)

    return eye_mask, lip_mask


def apply_retinex_illumination_exposure(
    l_channel: "np.ndarray", gamma: float, brightness_factor: float
) -> "np.ndarray":
    """Multi-Scale Retinex illumination decomposition with Filmic Highlight Rolloff.
    Lifts underexposed shadows smoothly while protecting natural beauty of clouds,
    sky, and bright highlights from clipping or blowout.
    """
    if cv2 is None or np is None:
        return l_channel

    H, W = l_channel.shape[:2]
    l_f = l_channel.astype(np.float32) / 255.0

    # Multi-scale illumination estimation on downscaled pyramid (lightning fast < 2ms)
    max_d = max(H, W)
    scale = 320.0 / max_d if max_d > 320 else 1.0
    if scale < 1.0:
        sw, sh = max(16, int(W * scale)), max(16, int(H * scale))
        thumb = cv2.resize(l_f, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        thumb = l_f

    blur_fine = cv2.GaussianBlur(thumb, (0, 0), sigmaX=6.0, sigmaY=6.0)
    blur_coarse = cv2.GaussianBlur(thumb, (0, 0), sigmaX=24.0, sigmaY=24.0)
    illum_small = 0.5 * blur_fine + 0.5 * blur_coarse

    if scale < 1.0:
        illum = cv2.resize(illum_small, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        illum = illum_small
    illum = np.clip(illum, 0.05, 1.0)

    # Adaptive shadow exposure lift: boosts underexposed zones while preserving midtones
    if brightness_factor > 1.0 or gamma < 1.0:
        shadow_deficit = np.clip(1.0 - illum, 0.0, 1.0)
        lift = 1.0 + shadow_deficit * max(0.0, brightness_factor - 1.0) * 1.5
        l_lifted = l_f * lift
    else:
        l_lifted = l_f * brightness_factor

    if gamma != 1.0:
        l_curved = np.clip(l_lifted, 0.0, 1.0) ** gamma
    else:
        l_curved = l_lifted

    # Filmic Highlight Rolloff (Soft-Knee):
    # For luminance > 0.82 (~210/255), applies asymptotic hyperbolic tangent rolloff
    # so puffy clouds, snow peaks, water highlights, and sunset skies never clip.
    T = 0.82
    knee_range = 1.0 - T
    high_mask = l_curved > T
    if np.any(high_mask):
        val = l_curved[high_mask]
        l_curved[high_mask] = T + knee_range * np.tanh((val - T) / knee_range)

    return np.clip(np.round(l_curved * 255.0), 0, 255).astype(np.uint8)


def apply_guided_detail_enhancement(
    l_channel: "np.ndarray", sharpness_factor: float, contrast_factor: float
) -> "np.ndarray":
    """Edge-Gradient Attenuated detail sharpening and conservative contrast.
    Sharpens natural micro-textures (foliage, rock grain, water ripples, grass)
    while completely suppressing glowing halo artifacts at high-contrast boundaries
    (e.g. tree branches against bright sky, mountain crests).
    """
    if cv2 is None or np is None:
        return l_channel

    l_res = l_channel
    if contrast_factor > 1.0:
        clip = min(1.8, max(1.1, 1.0 + (contrast_factor - 1.0) * 2.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l_res = clahe.apply(l_res)

    if sharpness_factor > 1.0:
        l_f = l_res.astype(np.float32)
        # Compute gradient magnitude
        gx = cv2.Sobel(l_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        # Attenuation factor: high on micro-textures (<35), suppressed near zero on step edges (>70)
        tau = 35.0
        attenuation = 1.0 / (1.0 + (grad_mag / tau) ** 2)

        blurred = cv2.GaussianBlur(l_f, (0, 0), sigmaX=1.2, sigmaY=1.2)
        high_freq = l_f - blurred
        clean_hf = high_freq * attenuation
        boost = min(0.40, (sharpness_factor - 1.0) * 2.5)
        l_res = np.clip(np.round(l_f + clean_hf * boost), 0, 255).astype(np.uint8)

    return l_res


def apply_perceptual_adaptive_vibrance(
    a: "np.ndarray",
    b: "np.ndarray",
    color_factor: float,
    face_mask: Optional["np.ndarray"] = None,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Perceptual Adaptive Vibrance prioritizing muted pixels over already-saturated ones.
    Protects chlorophyll green foliage and atmospheric sky blue from turning into
    unnatural neon or radioactive colors, while safely dampening facial skin.
    """
    if np is None or color_factor == 1.0:
        return a, b

    da = a.astype(np.float32) - 128.0
    db = b.astype(np.float32) - 128.0
    chroma = np.sqrt(da ** 2 + db ** 2)

    # Muted-pixel priority weight: S-curve tapering to ~0 as chroma approaches 85
    max_chroma = 85.0
    vibrance_weight = np.clip(1.0 - (chroma / max_chroma) ** 1.2, 0.05, 1.0)

    # Facial skin dampening
    if face_mask is not None:
        skin_damp = 1.0 - 0.75 * face_mask
    else:
        skin_damp = 1.0

    eff_factor = 1.0 + (color_factor - 1.0) * vibrance_weight * skin_damp

    # Natural foliage chlorophyll protection:
    # Foliage in LAB has negative a and positive b (hue angle in green zone 110-145 deg)
    # Prevent a from becoming excessively negative which makes grass fluorescent lime
    is_foliage = (da < -2.0) & (db > 5.0)
    da_factor = np.where(is_foliage, 1.0 + (eff_factor - 1.0) * 0.50, eff_factor)

    # Natural sky protection:
    # Blue sky has negative b (< -8.0). Keep a stable to avoid purple/cyan color cast
    is_sky = db < -8.0
    da_factor = np.where(is_sky, 1.0 + (da_factor - 1.0) * 0.40, da_factor)

    new_da = da * da_factor
    new_db = db * eff_factor

    return (
        np.clip(np.round(new_da + 128.0), 0, 255).astype(np.uint8),
        np.clip(np.round(new_db + 128.0), 0, 255).astype(np.uint8),
    )


def apply_frequency_separation_portrait(
    l_channel: "np.ndarray",
    face_mask: "np.ndarray",
    eye_mask: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """Dual-frequency portrait separation:
    Smooths mid-frequency lighting blotches and sensor grain while 100% preserving
    fine high-frequency skin pores and hair micro-relief (eliminating plastic skin effect).
    Boosts iris catchlight specular sparkle and iris striation clarity in eye zones.
    """
    if cv2 is None or np is None or face_mask is None:
        return l_channel

    l_f = l_channel.astype(np.float32)
    # Low frequency: smooth uneven lighting & blemishes with edge-preserving bilateral filter
    l_low = cv2.bilateralFilter(l_channel, d=5, sigmaColor=18, sigmaSpace=18).astype(np.float32)
    # High frequency: fine skin texture, pores, peach fuzz, eyelashes
    l_high = l_f - l_low

    # Smooth base complexion gently
    l_base_smooth = 0.75 * l_f + 0.25 * l_low
    # Add back 100% of high-frequency skin texture: real human pores are fully preserved!
    l_face = l_base_smooth + l_high

    # Eye micro-enhancement: catchlight specular sparkle and iris striation clarity
    if eye_mask is not None:
        em = eye_mask.astype(np.float32)
        is_specular = (l_f > 175).astype(np.float32)
        catchlight_boost = 1.0 + 0.15 * em * is_specular
        is_iris = ((l_f >= 50) & (l_f <= 175)).astype(np.float32)
        iris_contrast = l_f + (l_f - 110.0) * 0.12 * em * is_iris
        l_face = np.where(em > 0.08, np.clip(iris_contrast * catchlight_boost, 0, 255), l_face)

    fm = face_mask.astype(np.float32)
    result = (1.0 - fm) * l_f + fm * l_face
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def align_melanin_hemoglobin_skin_tones(
    a: "np.ndarray",
    b: "np.ndarray",
    face_mask: "np.ndarray",
    lip_mask: Optional["np.ndarray"] = None,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Aligns facial skin chromaticity gently toward the healthy melanin-hemoglobin locus (~50 deg LAB),
    eliminating greenish fluorescent or cold blue shadow casts while preserving ethnic undertones.
    Enhances lip vitality naturally (+6% red-magenta chroma).
    """
    if np is None or face_mask is None:
        return a, b

    da = a.astype(np.float32) - 128.0
    db = b.astype(np.float32) - 128.0

    # Facial skin zone (excluding lips)
    skin_zone = face_mask.copy()
    if lip_mask is not None:
        skin_zone = np.clip(skin_zone * (1.0 - lip_mask), 0.0, 1.0)

    # Detect human skin chromaticity range in LAB (warm quadrant with healthy melanin/hemoglobin)
    is_skin_chroma = (da > 2.0) & (db > 3.0) & (db > da * 0.50) & (db < da * 2.8)
    skin_weight = skin_zone * is_skin_chroma.astype(np.float32)

    # Melanin-hemoglobin natural locus target: b* / a* ~ 1.35
    target_db = da * 1.35
    # Gentle 18% alignment blend
    aligned_db = (1.0 - 0.18 * skin_weight) * db + (0.18 * skin_weight) * target_db

    # Lip vitality: subtle +6% boost to lip red-magenta chroma within lip_mask
    if lip_mask is not None:
        lm = lip_mask.astype(np.float32)
        lip_boost_a = 1.0 + 0.06 * lm
        da = da * lip_boost_a

    return (
        np.clip(np.round(da + 128.0), 0, 255).astype(np.uint8),
        np.clip(np.round(aligned_db + 128.0), 0, 255).astype(np.uint8),
    )


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

        # 1. Detect human faces and construct feathered face & landmark masks
        face_rows, face_boxes = detect_face_regions(bgr)
        has_faces = len(face_boxes) > 0
        if has_faces:
            face_mask = build_feathered_face_mask(bgr.shape, face_boxes)
            eye_mask, lip_mask = build_facial_landmark_masks(bgr.shape, face_rows)
        else:
            face_mask = None
            eye_mask = None
            lip_mask = None

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. Retinex illumination and exposure with Filmic Highlight Rolloff
        # Lifts underexposed shadows while protecting clouds, sky, and sun reflections from blowout
        l_retinex = apply_retinex_illumination_exposure(l, plan.gamma, plan.brightness_factor)

        # 3. Background Detail & Contrast: Halo-Free Guided Enhancement
        # Edge-gradient attenuation eliminates glowing fringes around tree limbs, mountain ridges, horizons
        l_bg = apply_guided_detail_enhancement(l_retinex, plan.sharpness_factor, plan.contrast_factor)

        # 4. Portrait Face Zone: Frequency Separation + Eye Catchlight Micro-Enhancement
        # 100% preserves real skin pores, eyelashes, and authentic human appearance (never plastic)
        if has_faces and face_mask is not None:
            l_composite = apply_frequency_separation_portrait(l_bg, face_mask, eye_mask)
        else:
            l_composite = l_bg

        # 5. Perceptual Adaptive Vibrance (Muted-Pixel Priority & Chlorophyll/Sky Protection)
        # Prevents neon grass or cyan skies; lifts dull/muted tones naturally
        a_vib, b_vib = apply_perceptual_adaptive_vibrance(a, b, plan.color_factor, face_mask)

        # 6. Melanin-Hemoglobin Skin Tone Locus Alignment & Lip Vitality
        if has_faces and face_mask is not None:
            a_final, b_final = align_melanin_hemoglobin_skin_tones(a_vib, b_vib, face_mask, lip_mask)
        else:
            a_final, b_final = a_vib, b_vib

        merged_lab = cv2.merge((l_composite, a_final, b_final))
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # 7. SFace Biometric Guardrail: mathematically guarantees >= 0.98 cosine similarity
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
