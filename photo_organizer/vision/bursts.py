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


def calculate_photo_quality_score(path: Path) -> Tuple[float, float]:
    """Calculates a composite quality score (0 - 100) based on sharpness, exposure, and face clarity."""
    if Image is None or cv2 is None or np is None:
        return 50.0, 50.0

    try:
        with Image.open(path) as image:
            # Resize for fast metric computation
            working = ImageOps.exif_transpose(image).convert("RGB")
            working.thumbnail((800, 800), Image.Resampling.BILINEAR)

            arr = np.array(working)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            # 1. Sharpness via Laplacian variance
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_score = min(100.0, math.log1p(lap_var) * 12.0)

            # 2. Exposure balance (ideal luminance mean ~ 110-150, penalty for extreme blowout/crush)
            mean_lum = float(gray.mean())
            exposure_penalty = abs(mean_lum - 128.0) / 128.0 * 20.0
            blown_ratio = float((gray >= 250).mean())
            crushed_ratio = float((gray <= 5).mean())
            clipping_penalty = (blown_ratio + crushed_ratio) * 40.0
            exposure_score = max(0.0, 100.0 - exposure_penalty - clipping_penalty)

            # 3. Face clarity bonus
            face_score = 0.0
            has_face, _ = detect_face(working)
            if has_face:
                face_score = 25.0

            # Composite weighted score
            composite = (sharpness_score * 0.50) + (exposure_score * 0.35) + face_score
            return round(composite, 2), round(sharpness_score, 2)
    except Exception:
        return 50.0, 50.0


def detect_burst_groups(
    photos: Sequence[PhotoInfo], max_interval_seconds: float = 3.0
) -> List[BurstGroup]:
    """Finds sequences of rapid-fire photos (bursts) and selects the recommended Best Shot."""
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
        score, sharpness = calculate_photo_quality_score(photo.path)
        scored_photos.append(
            BurstPhoto(
                photo=photo,
                score=score,
                sharpness=sharpness,
                is_best_shot=False,
            )
        )

    # Sort by quality score descending
    scored_photos.sort(key=lambda bp: (bp.score, bp.photo.pixels), reverse=True)
    best = scored_photos[0]
    best.is_best_shot = True

    return BurstGroup(
        group_id=group_id,
        photos=scored_photos,
        best_photo=best.photo,
    )
