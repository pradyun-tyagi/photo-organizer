"""Burst shot sequence detection and Best Shot quality selector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from photo_organizer.dedupe.hasher import hamming_distance
from photo_organizer.models import BurstGroup, BurstPhoto, PhotoInfo
from photo_organizer.vision.detectors import detect_face
from photo_organizer.vision.faces import _get_deep_face_models


def calculate_burst_photo_metrics(
    path: Path,
) -> Tuple[float, float, float, float]:
    """Calculates comprehensive quality metrics for Golden Frame selection:
    1. Tenengrad gradient energy and high-frequency sharpness
    2. Anisotropic motion blur and camera hand-shake detection
    3. Exposure balance and highlight/shadow clipping avoidance
    4. Facial clarity, eye openness (blink penalty), and smile expression
    Returns (composite_score, sharpness_score, eye_openness, motion_blur).
    """
    if Image is None or cv2 is None or np is None:
        return 50.0, 50.0, 0.5, 0.0

    try:
        with Image.open(path) as image:
            working = ImageOps.exif_transpose(image).convert("RGB")
            working.thumbnail((800, 800), Image.Resampling.BILINEAR)

            arr = np.array(working)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # 1. Tenengrad gradient energy and Laplacian sharpness
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_energy = float(np.mean(gx ** 2 + gy ** 2))
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_score = min(100.0, math.log1p(grad_energy + lap_var) * 9.5)

            # 2. Anisotropic motion blur and hand-shake detection
            vx = float(np.var(gx))
            vy = float(np.var(gy))
            min_v = min(vx, vy) + 1e-4
            max_v = max(vx, vy)
            directional_ratio = max_v / min_v
            motion_blur = 0.0
            if directional_ratio > 3.0 and min_v < 300.0:
                motion_blur = min(30.0, (directional_ratio - 2.5) * 4.0)

            # 3. Exposure balance and highlight/shadow clipping penalties
            mean_lum = float(gray.mean())
            exposure_penalty = abs(mean_lum - 128.0) / 128.0 * 20.0
            blown_ratio = float((gray >= 252).mean())
            crushed_ratio = float((gray <= 4).mean())
            clipping_penalty = (blown_ratio * 50.0) + (crushed_ratio * 30.0)
            exposure_score = max(0.0, 100.0 - exposure_penalty - clipping_penalty)

            # 4. Facial Expression & Open Eyes Scoring
            face_score = 0.0
            eye_openness = 0.5
            detector, _ = _get_deep_face_models()
            faces_detected = False
            if detector is not None:
                orig_h, orig_w = bgr.shape[:2]
                try:
                    detector.setInputSize((orig_w, orig_h))
                    _, raw_faces = detector.detect(bgr)
                    if raw_faces is not None and len(raw_faces) > 0:
                        faces_detected = True
                        face_score = 25.0
                        total_openness = 0.0
                        valid_faces = 0
                        for row in raw_faces[:3]:
                            if len(row) >= 14:
                                rx, ry = int(row[4]), int(row[5])
                                lx, ly = int(row[6]), int(row[7])
                                eye_dist = float(np.hypot(lx - rx, ly - ry))
                                if eye_dist >= 8:
                                    r = max(3, int(eye_dist * 0.15))
                                    r_patch = gray[max(0, ry - r):min(orig_h, ry + r + 1), max(0, rx - r):min(orig_w, rx + r + 1)]
                                    l_patch = gray[max(0, ly - r):min(orig_h, ly + r + 1), max(0, lx - r):min(orig_w, lx + r + 1)]
                                    if r_patch.size > 0 and l_patch.size > 0:
                                        r_std = float(np.std(r_patch))
                                        l_std = float(np.std(l_patch))
                                        avg_eye_std = (r_std + l_std) * 0.5
                                        op = np.clip((avg_eye_std - 10.0) / 25.0, 0.0, 1.0)
                                        total_openness += op
                                        valid_faces += 1

                                    # Smile bonus: mouth width vs eye distance
                                    rcx, rcy = int(row[10]), int(row[11])
                                    lcx, lcy = int(row[12]), int(row[13])
                                    mw = float(np.hypot(lcx - rcx, lcy - rcy))
                                    if mw / eye_dist > 0.85:
                                        face_score += 5.0
                        if valid_faces > 0:
                            eye_openness = float(total_openness / valid_faces)
                            if eye_openness < 0.25:
                                face_score -= 20.0  # Blink penalty
                            else:
                                face_score += eye_openness * 10.0
                except Exception:
                    pass

            if not faces_detected:
                has_face, _ = detect_face(working)
                if has_face:
                    face_score = 15.0

            # Composite weighted quality score
            composite = (sharpness_score * 0.45) + (exposure_score * 0.30) + face_score - motion_blur
            composite = round(max(0.0, min(100.0, composite)), 2)
            return composite, round(sharpness_score, 2), round(eye_openness, 2), round(motion_blur, 2)
    except Exception:
        return 50.0, 50.0, 0.5, 0.0


def calculate_photo_quality_score(path: Path) -> Tuple[float, float]:
    """Calculates a composite quality score (0 - 100) and sharpness score for a photo."""
    score, sharpness, _, _ = calculate_burst_photo_metrics(path)
    return score, sharpness


def detect_burst_groups(
    photos: Sequence[PhotoInfo], max_interval_seconds: float = 3.0
) -> List[BurstGroup]:
    """Finds sequences of rapid-fire photos (bursts) and crowns the Golden Frame."""
    if len(photos) < 2:
        return []

    # Sort photos chronologically
    sorted_photos = sorted(photos, key=lambda p: p.taken_at)
    burst_groups: List[BurstGroup] = []

    current_burst: List[PhotoInfo] = []
    group_counter = 1

    for i in range(len(sorted_photos)):
        p = sorted_photos[i]
        if p.is_video:
            continue

        if not current_burst:
            current_burst.append(p)
            continue

        prev = current_burst[-1]
        dt = (p.taken_at - prev.taken_at).total_seconds()

        # Check timestamp proximity and visual similarity
        is_burst_candidate = False
        if 0.0 <= dt <= max_interval_seconds:
            if p.visual_hash is not None and prev.visual_hash is not None:
                dist = hamming_distance(p.visual_hash, prev.visual_hash)
                if dist <= 8:
                    is_burst_candidate = True
            elif p.aspect_ratio == prev.aspect_ratio:
                is_burst_candidate = True

        if is_burst_candidate:
            current_burst.append(p)
        else:
            if len(current_burst) >= 2:
                burst_groups.append(_build_burst_group(f"burst_{group_counter}", current_burst))
                group_counter += 1
            current_burst = [p]

    if len(current_burst) >= 2:
        burst_groups.append(_build_burst_group(f"burst_{group_counter}", current_burst))

    return burst_groups


def _build_burst_group(group_id: str, photos: List[PhotoInfo]) -> BurstGroup:
    scored_photos: List[BurstPhoto] = []

    for photo in photos:
        score, sharpness, eye_open, motion_blur = calculate_burst_photo_metrics(photo.path)
        scored_photos.append(
            BurstPhoto(
                photo=photo,
                score=score,
                sharpness=sharpness,
                is_best_shot=False,
                is_golden_frame=False,
                eye_openness=eye_open,
                motion_blur=motion_blur,
            )
        )

    # Sort by quality score descending (tie-breaker: resolution pixels, companions, file size, path length)
    scored_photos.sort(
        key=lambda bp: (
            bp.score,
            bp.photo.pixels,
            len(bp.photo.companion_files),
            bp.photo.file_size,
            -len(str(bp.photo.path)),
        ),
        reverse=True,
    )
    best = scored_photos[0]
    best.is_best_shot = True
    best.is_golden_frame = True

    return BurstGroup(
        group_id=group_id,
        photos=scored_photos,
        best_photo=best.photo,
        golden_frame=best.photo,
    )
