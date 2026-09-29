"""Hashing utilities: SHA-256, difference hash (dHash), and DCT-based perceptual hash (pHash)."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:
    Image = None
    ImageOps = None

    class UnidentifiedImageError(Exception):
        pass

try:
    import numpy as np
except ImportError:
    np = None

from photo_organizer.config import EXIF_DATE_TAGS
from photo_organizer.core.dates import parse_exif_datetime


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flattened_pixels(image: "Image.Image") -> List[object]:
    get_flattened_data = getattr(image, "get_flattened_data", None)
    if get_flattened_data is not None:
        return list(get_flattened_data())
    return list(image.getdata())


def compute_dhash(image: "Image.Image") -> int:
    """Computes a 64-bit difference hash (dHash) comparing adjacent horizontal pixels."""
    gray = ImageOps.grayscale(image)
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = flattened_pixels(small)
    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits = (bits << 1) | int(left > right)
    return bits


_DCT_MATRIX_CACHE: Optional["np.ndarray"] = None


def _get_dct_matrix(n: int = 32) -> "np.ndarray":
    global _DCT_MATRIX_CACHE
    if _DCT_MATRIX_CACHE is not None:
        return _DCT_MATRIX_CACHE
    d = np.zeros((n, n), dtype="float32")
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            d[k, i] = alpha * np.cos(np.pi * (2 * i + 1) * k / (2.0 * n))
    _DCT_MATRIX_CACHE = d
    return _DCT_MATRIX_CACHE


def compute_phash(image: "Image.Image") -> Optional[int]:
    """Computes a 64-bit DCT perceptual hash (pHash) robust to scaling and minor color edits."""
    if np is None or Image is None:
        return None
    try:
        gray = ImageOps.grayscale(image)
        small = gray.resize((32, 32), Image.Resampling.LANCZOS)
        arr = np.array(small, dtype="float32")
        dct_mat = _get_dct_matrix(32)
        dct = dct_mat @ arr @ dct_mat.T
        low_freq = dct[:8, :8]
        flat = low_freq.flatten()
        # Exclude DC component (0,0) from median calculation
        median = float(np.median(flat[1:]))
        bits = 0
        for val in flat:
            bits = (bits << 1) | int(val > median)
        return bits
    except Exception:
        return None


def hamming_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def visual_hash_and_size(path: Path) -> Tuple[int, int, Optional[int], Optional[str]]:
    if Image is None:
        return 0, 0, None, None
    try:
        with Image.open(path) as image:
            width, height = image.size
            format_name = image.format
            if getattr(image, "is_animated", False):
                return width, height, None, format_name

            if format_name == "JPEG":
                try:
                    image.draft("L", (128, 128))
                except Exception:
                    pass

            bits = compute_dhash(image)
            return width, height, bits, format_name
    except (OSError, UnidentifiedImageError, ValueError):
        return 0, 0, None, None


def image_metadata(
    path: Path,
) -> Tuple[int, int, Optional[int], Optional[str], Optional[datetime]]:
    if Image is None:
        return 0, 0, None, None, None
    try:
        with Image.open(path) as image:
            width, height = image.size
            format_name = image.format

            exif_date = None
            exif = image.getexif()
            if exif:
                for tag in EXIF_DATE_TAGS:
                    exif_date = parse_exif_datetime(exif.get(tag))
                    if exif_date is not None:
                        break

            if getattr(image, "is_animated", False):
                return width, height, None, format_name, exif_date

            if format_name == "JPEG":
                try:
                    image.draft("L", (128, 128))
                except Exception:
                    pass

            bits = compute_dhash(image)
            return width, height, bits, format_name, exif_date
    except (OSError, UnidentifiedImageError, ValueError):
        return 0, 0, None, None, None


def full_image_inspection(
    path: Path, compute_phash_flag: bool = True
) -> Tuple[int, int, Optional[int], Optional[int], Optional[str], Optional[datetime]]:
    """Extracts width, height, dHash, pHash, format_name, and EXIF date with fast decoding."""
    if Image is None:
        return 0, 0, None, None, None, None
    try:
        with Image.open(path) as image:
            width, height = image.size
            format_name = image.format

            exif_date = None
            exif = image.getexif()
            if exif:
                for tag in EXIF_DATE_TAGS:
                    exif_date = parse_exif_datetime(exif.get(tag))
                    if exif_date is not None:
                        break

            if getattr(image, "is_animated", False):
                return width, height, None, None, format_name, exif_date

            # Accelerate JPEG decoding if possible
            if format_name == "JPEG":
                try:
                    image.draft("L", (128, 128))
                except Exception:
                    pass

            d_hash = compute_dhash(image)
            p_hash = compute_phash(image) if compute_phash_flag else None
            return width, height, d_hash, p_hash, format_name, exif_date
    except (OSError, UnidentifiedImageError, ValueError):
        try:
            from photo_organizer.core.raw_heif import extract_raw_preview, is_raw_file

            if is_raw_file(path):
                raw_img = extract_raw_preview(path)
                if raw_img is not None:
                    width, height = raw_img.size
                    d_hash = compute_dhash(raw_img)
                    p_hash = compute_phash(raw_img) if compute_phash_flag else None
                    return width, height, d_hash, p_hash, "RAW", None
        except Exception:
            pass
        return 0, 0, None, None, None, None


def compute_phash_for_path(path: Path) -> Optional[int]:
    """Computes pHash on demand for duplicate candidate verification."""
    if Image is None:
        return None
    try:
        with Image.open(path) as image:
            if image.format == "JPEG":
                try:
                    image.draft("L", (128, 128))
                except Exception:
                    pass
            return compute_phash(image)
    except Exception:
        try:
            from photo_organizer.core.raw_heif import extract_raw_preview, is_raw_file

            if is_raw_file(path):
                raw_img = extract_raw_preview(path)
                if raw_img is not None:
                    return compute_phash(raw_img)
        except Exception:
            pass
        return None
