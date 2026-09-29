"""Local, 100% offline semantic and natural language photo search index."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

from photo_organizer.dedupe.hasher import flattened_pixels
from photo_organizer.models import PhotoInfo


def analyze_photo_semantics(photo: PhotoInfo) -> Dict[str, any]:
    """Extracts dominant color, scene type, and semantic tags for natural language search."""
    tags = set()

    # 1. Temporal attributes
    dt = photo.taken_at
    tags.add(str(dt.year))
    tags.add(dt.strftime("%B").lower())      # e.g. "june"
    tags.add(dt.strftime("%A").lower())      # e.g. "saturday"

    # Season (Northern Hemisphere approximation)
    month = dt.month
    if month in (12, 1, 2):
        tags.add("winter")
    elif month in (3, 4, 5):
        tags.add("spring")
    elif month in (6, 7, 8):
        tags.add("summer")
    else:
        tags.add("autumn")
        tags.add("fall")

    # Time of day
    hour = dt.hour
    if 5 <= hour < 11:
        tags.add("morning")
    elif 11 <= hour < 16:
        tags.add("afternoon")
    elif 16 <= hour < 19:
        tags.add("evening")
        tags.add("golden hour")
    else:
        tags.add("night")

    # 2. Format & Type
    if photo.is_video:
        tags.add("video")
        tags.add("clip")
    else:
        tags.add("photo")
        tags.add("picture")

    if photo.has_live_photo_video:
        tags.add("live photo")

    # 3. Location tags
    if photo.city:
        tags.add(photo.city.lower())
    if photo.country:
        tags.add(photo.country.lower())

    # 4. Filename tokens
    name_tokens = re.split(r"[_\-\s\.]+", photo.path.stem.lower())
    for token in name_tokens:
        if len(token) > 2 and not token.isdigit():
            tags.add(token)

    # 5. Visual color and scene characteristics
    dominant_color = "unknown"
    if Image is not None and not photo.is_video and photo.path.exists():
        try:
            with Image.open(photo.path) as img:
                thumb = img.convert("RGB").resize((40, 40), Image.Resampling.BILINEAR)
                pixels = flattened_pixels(thumb)
                if pixels and np is not None:
                    arr = np.array(pixels)
                    mean_r = float(arr[:, 0].mean())
                    mean_g = float(arr[:, 1].mean())
                    mean_b = float(arr[:, 2].mean())
                    mean_lum = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

                    # Brightness scenes
                    if mean_lum < 40:
                        tags.add("dark")
                        tags.add("night")
                    elif mean_lum > 200:
                        tags.add("bright")
                        tags.add("sunny")

                    # Dominant color determination
                    if mean_lum < 35:
                        dominant_color = "black"
                    elif mean_lum > 220:
                        dominant_color = "white"
                    elif mean_b > mean_r + 20 and mean_b > mean_g + 10:
                        dominant_color = "blue"
                        tags.add("sky")
                        tags.add("water")
                        tags.add("beach")
                    elif mean_g > mean_r + 15 and mean_g > mean_b + 10:
                        dominant_color = "green"
                        tags.add("nature")
                        tags.add("grass")
                        tags.add("garden")
                        tags.add("forest")
                    elif mean_r > mean_g + 30 and mean_g > mean_b:
                        dominant_color = "orange"
                        tags.add("sunset")
                        tags.add("sunrise")
                    elif mean_r > mean_g + 20 and mean_r > mean_b + 20:
                        dominant_color = "red"
                    elif mean_r > 180 and mean_g > 180 and mean_b < 100:
                        dominant_color = "yellow"
                    else:
                        dominant_color = "gray"

                    tags.add(dominant_color)
        except Exception:
            pass

    return {
        "dominant_color": dominant_color,
        "tags": list(tags),
    }


def search_photos(
    query: str, photos: Sequence[PhotoInfo]
) -> List[Tuple[PhotoInfo, float]]:
    """Performs fast offline semantic search matching query terms against extracted attributes.
    Returns list of (PhotoInfo, match_score) sorted by relevance.
    """
    cleaned = query.strip().lower()
    if not cleaned:
        return []

    tokens = [t for t in re.split(r"[\s,]+", cleaned) if t]
    if not tokens:
        return []

    matches: List[Tuple[PhotoInfo, float]] = []

    for photo in photos:
        semantics = analyze_photo_semantics(photo)
        tags = set(semantics["tags"])
        score = 0.0

        for token in tokens:
            # Exact tag match
            if token in tags:
                score += 3.0
            # Partial substring match in tags
            elif any(token in t for t in tags):
                score += 1.5

            # Filename match
            if token in photo.path.name.lower():
                score += 2.0

            # City/Country match
            if photo.city and token == photo.city.lower():
                score += 5.0
            if photo.country and token == photo.country.lower():
                score += 4.0

        if score > 0:
            matches.append((photo, score))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches
