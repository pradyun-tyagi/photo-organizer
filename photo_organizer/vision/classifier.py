"""Text heaviness heuristics and non-relevant / forward classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError
except ImportError:
    Image = None
    ImageFilter = None
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

from photo_organizer.dedupe.hasher import flattened_pixels
from photo_organizer.models import NonRelevantResult, PhotoInfo
from photo_organizer.vision.detectors import detect_face, detect_person


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def analysis_image(image: "Image.Image", max_dimension: int = 900) -> "Image.Image":
    working = ImageOps.exif_transpose(image).convert("RGB")
    working.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    return working


def color_saturation_score_from_rgb(image: "Image.Image") -> float:
    if np is not None:
        array = np.array(image)
        if array.size == 0:
            return 0.0
        return float(np.ptp(array, axis=2).mean())

    pixels = flattened_pixels(image)
    if not pixels:
        return 0.0
    return sum(max(pixel) - min(pixel) for pixel in pixels) / len(pixels)


def color_saturation_score(image: "Image.Image") -> float:
    working = analysis_image(image, max_dimension=600)
    return color_saturation_score_from_rgb(working)


def sharpness_score(gray: "Image.Image") -> float:
    if cv2 is not None and np is not None:
        gray_array = np.array(gray)
        return float(cv2.Laplacian(gray_array, cv2.CV_64F).var())

    if ImageFilter is None or ImageStat is None:
        return 0.0

    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).stddev[0] ** 2)


def noise_score(gray: "Image.Image") -> float:
    if cv2 is None or np is None:
        return 0.0

    gray_array = np.array(gray).astype("float32")
    blurred = cv2.GaussianBlur(gray_array, (3, 3), 0)
    residual = gray_array - blurred
    return float(residual.std())


def text_heaviness_score_from_prepared(
    working: "Image.Image", gray: "Image.Image"
) -> Tuple[float, str]:
    width, height = gray.size
    total = width * height
    if total == 0:
        return 0.0, "empty image"

    if np is not None:
        rgb_array = np.array(working)
        gray_array = np.array(gray)

        light_ratio = float((gray_array >= 225).mean())
        dark_ratio = float((gray_array <= 65).mean())
        low_saturation_ratio = float((np.ptp(rgb_array, axis=2) <= 35).mean())
        
        # Detect uniform background of any color (e.g. colored memes, slides, receipts)
        hist, _ = np.histogram(gray_array, bins=32)
        peak_ratio = float(hist.max() / total)
        dominant_background_ratio = max(light_ratio, dark_ratio, peak_ratio * 0.90)

        if light_ratio >= dark_ratio:
            ink_values = gray_array < 115
            ink_name = "dark text on light background"
        else:
            ink_values = gray_array > 155
            ink_name = "light text on dark background"

        ink_ratio = float(ink_values.mean())
        minimum_ink_per_row = max(2, int(width * 0.01))
        active_rows_mask = ink_values.sum(axis=1) >= minimum_ink_per_row
        active_row_ratio = float(active_rows_mask.mean())
        if active_rows_mask.size:
            starts = np.logical_and(active_rows_mask, ~np.roll(active_rows_mask, 1))
            starts[0] = active_rows_mask[0]
            text_bands = int(starts.sum())
        else:
            text_bands = 0

        edge_width = min(320, width)
        edge_height = max(1, int(height * (edge_width / width)))
        edge_gray = gray.resize((edge_width, edge_height), Image.Resampling.BILINEAR)
        edge_array = np.array(edge_gray).astype("int16")
        if edge_width > 1:
            edge_density = float((np.abs(np.diff(edge_array, axis=1)) > 45).mean())
        else:
            edge_density = 0.0

        background_score = clamp((dominant_background_ratio - 0.25) / 0.55)
        edge_score = clamp((edge_density - 0.015) / 0.10)
        row_score = clamp((active_row_ratio - 0.08) / 0.42)
        band_score = clamp(text_bands / 18)
        saturation_score = clamp((low_saturation_ratio - 0.35) / 0.50)
        ink_score = clamp((ink_ratio - 0.02) / 0.22)

        score = (
            background_score * 0.24
            + edge_score * 0.18
            + row_score * 0.20
            + band_score * 0.16
            + saturation_score * 0.12
            + ink_score * 0.10
        )

        reason = (
            f"{ink_name}; text score {score:.2f}, background {dominant_background_ratio:.2f}, "
            f"rows {active_row_ratio:.2f}, bands {text_bands}, edges {edge_density:.2f}"
        )
        return score, reason

    rgb_pixels = flattened_pixels(working)
    gray_pixels = flattened_pixels(gray)

    light = 0
    dark = 0
    low_saturation = 0
    for red, green, blue in rgb_pixels:
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        if maximum - minimum <= 35:
            low_saturation += 1

    for value in gray_pixels:
        if value >= 225:
            light += 1
        elif value <= 65:
            dark += 1

    light_ratio = light / total
    dark_ratio = dark / total
    low_saturation_ratio = low_saturation / total
    dominant_background_ratio = max(light_ratio, dark_ratio)

    if light_ratio >= dark_ratio:
        ink_values = [value < 115 for value in gray_pixels]
        ink_name = "dark text on light background"
    else:
        ink_values = [value > 155 for value in gray_pixels]
        ink_name = "light text on dark background"

    ink_ratio = sum(ink_values) / total

    active_rows = 0
    text_bands = 0
    in_band = False
    minimum_ink_per_row = max(2, int(width * 0.01))
    for row in range(height):
        start = row * width
        row_has_ink = sum(ink_values[start : start + width]) >= minimum_ink_per_row
        if row_has_ink:
            active_rows += 1
            if not in_band:
                text_bands += 1
                in_band = True
        else:
            in_band = False

    active_row_ratio = active_rows / height

    edge_width = min(320, width)
    edge_height = max(1, int(height * (edge_width / width)))
    edge_gray = gray.resize((edge_width, edge_height), Image.Resampling.BILINEAR)
    edge_pixels = flattened_pixels(edge_gray)
    transitions = 0
    comparisons = 0
    for row in range(edge_height):
        for column in range(edge_width - 1):
            comparisons += 1
            if (
                abs(
                    edge_pixels[row * edge_width + column]
                    - edge_pixels[row * edge_width + column + 1]
                )
                > 45
            ):
                transitions += 1
    edge_density = transitions / comparisons if comparisons else 0.0

    background_score = clamp((dominant_background_ratio - 0.25) / 0.55)
    edge_score = clamp((edge_density - 0.015) / 0.10)
    row_score = clamp((active_row_ratio - 0.08) / 0.42)
    band_score = clamp(text_bands / 18)
    saturation_score = clamp((low_saturation_ratio - 0.35) / 0.50)
    ink_score = clamp((ink_ratio - 0.02) / 0.22)

    score = (
        background_score * 0.24
        + edge_score * 0.18
        + row_score * 0.20
        + band_score * 0.16
        + saturation_score * 0.12
        + ink_score * 0.10
    )

    reason = (
        f"{ink_name}; text score {score:.2f}, background {dominant_background_ratio:.2f}, "
        f"rows {active_row_ratio:.2f}, bands {text_bands}, edges {edge_density:.2f}"
    )
    return score, reason


def text_heaviness_score(image: "Image.Image") -> Tuple[float, str]:
    working = analysis_image(image)
    gray = ImageOps.grayscale(working)
    return text_heaviness_score_from_prepared(working, gray)


def classify_non_relevant(
    path: Path, likely_camera_photo: bool = False
) -> NonRelevantResult:
    if Image is None:
        return NonRelevantResult(False, "Pillow is not installed", 0.0, False, False)

    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        return NonRelevantResult(False, f"{suffix} screening skipped", 0.0, False, False)

    try:
        with Image.open(path) as image:
            if getattr(image, "is_animated", False):
                return NonRelevantResult(
                    False, "animated image screening skipped", 0.0, False, False
                )

            # 1. Check for tiny icons, stickers, thumbnails
            orig_w, orig_h = image.size
            if max(orig_w, orig_h) < 96 or (orig_w * orig_h < 10000):
                return NonRelevantResult(
                    True,
                    f"tiny thumbnail or sticker ({orig_w}x{orig_h}); discarded as non-relevant",
                    text_score=0.0,
                    face_detected=False,
                    face_detector_available=True,
                )

            working = analysis_image(image, max_dimension=700)
            gray = ImageOps.grayscale(working)

            # 2. Check for blank, pitch dark pocket photos or solid color swatches
            stat = ImageStat.Stat(gray)
            mean_lum = float(stat.mean[0])
            std_lum = float(stat.stddev[0])

            # Pocket dark accidental shot
            if mean_lum < 10.0 and std_lum < 6.0:
                return NonRelevantResult(
                    True,
                    f"pocket dark or blank image (brightness {mean_lum:.1f}, std {std_lum:.1f}); discarded",
                    text_score=0.0,
                    face_detected=False,
                    face_detector_available=True,
                )

            # Flat uniform color swatch (only if no camera metadata)
            if not likely_camera_photo and std_lum < 2.0:
                return NonRelevantResult(
                    True,
                    f"solid uniform color graphic (std {std_lum:.1f}); discarded",
                    text_score=0.0,
                    face_detected=False,
                    face_detector_available=True,
                )

            # 3. Check for extreme blur / camera shake on non-camera files
            if not likely_camera_photo:
                sharpness = sharpness_score(gray)
                if sharpness < 8.0 and std_lum < 24.0:
                    return NonRelevantResult(
                        False,
                        f"extreme blur / motion shake (sharpness {sharpness:.1f}); moved to review",
                        text_score=0.0,
                        face_detected=False,
                        face_detector_available=True,
                        is_maybe_non_relevant=True,
                    )

            # 4. Text heaviness and meme/document screening
            text_score, text_reason = text_heaviness_score_from_prepared(working, gray)

            min_screening_threshold = 0.84 if likely_camera_photo else 0.70
            if text_score < min_screening_threshold:
                reason = text_reason
                if likely_camera_photo:
                    reason = f"camera metadata present; kept unless very text-heavy; {text_reason}"
                return NonRelevantResult(
                    False,
                    reason,
                    text_score,
                    face_detected=False,
                    face_detector_available=True,
                    is_maybe_non_relevant=False,
                    person_detected=False,
                    person_detector_available=True,
                )

            face_detected, face_detector_available = False, False
            person_detected, person_detector_available = False, False
            try:
                face_detected, face_detector_available = detect_face(image)
                if not face_detected:
                    person_detected, person_detector_available = detect_person(image)
                else:
                    person_detector_available = True
            except Exception:
                face_detected, face_detector_available = False, False
                person_detected, person_detector_available = False, False

            # SAFEGUARD: Any human face or person detected keeps the image relevant
            if face_detected or person_detected:
                reason = (
                    "person detected; kept as relevant "
                    f"despite text score {text_score:.2f}"
                )
                return NonRelevantResult(
                    False,
                    reason,
                    text_score,
                    face_detected,
                    face_detector_available,
                    person_detected=person_detected,
                    person_detector_available=person_detector_available,
                )

            detector_available = face_detector_available or person_detector_available
            if likely_camera_photo:
                non_relevant_threshold = 0.92
                maybe_threshold = 0.84
            elif detector_available:
                non_relevant_threshold = 0.84
                maybe_threshold = 0.70
            else:
                non_relevant_threshold = 0.88
                maybe_threshold = 0.78

            is_non_relevant = text_score >= non_relevant_threshold
            if is_non_relevant:
                detector_note = (
                    "no person detected"
                    if detector_available
                    else "person detector unavailable; very strict text threshold used"
                )
                return NonRelevantResult(
                    True,
                    f"text-heavy image, {detector_note}; {text_reason}",
                    text_score,
                    face_detected,
                    face_detector_available,
                    person_detected=person_detected,
                    person_detector_available=person_detector_available,
                )

            is_maybe = text_score >= maybe_threshold
            if is_maybe:
                reason = (
                    f"borderline text-heavy image; {text_reason}; "
                    "moved to maybe-non-relevant for review"
                )
            else:
                reason = text_reason
                if likely_camera_photo:
                    reason = f"camera metadata present; kept unless very text-heavy; {text_reason}"
                elif not detector_available:
                    reason = (
                        f"kept; person detector unavailable and text score {text_score:.2f}"
                    )

            return NonRelevantResult(
                False,
                reason,
                text_score,
                face_detected,
                face_detector_available,
                is_maybe_non_relevant=is_maybe,
                person_detected=person_detected,
                person_detector_available=person_detector_available,
            )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return NonRelevantResult(False, f"screening failed: {exc}", 0.0, False, False)


def classify_photo_for_screening(
    item: Tuple[PhotoInfo, bool]
) -> Tuple[PhotoInfo, NonRelevantResult]:
    photo, likely_camera_photo = item
    import sys
    root_mod = sys.modules.get("photo_organizer")
    func = getattr(root_mod, "classify_non_relevant", classify_non_relevant)
    return photo, func(photo.path, likely_camera_photo)

