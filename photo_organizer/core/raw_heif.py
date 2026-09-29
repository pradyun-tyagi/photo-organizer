"""HEIC/HEIF and Camera RAW decoding and Live Photo companion handling."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None

    class UnidentifiedImageError(Exception):
        pass

# Register pillow-heif if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef"}


def is_raw_file(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def is_heif_file(path: Path) -> bool:
    return path.suffix.lower() in {".heic", ".heif"}


def extract_raw_preview(path: Path) -> Optional["Image.Image"]:
    """Extracts the embedded full-resolution or medium JPEG thumbnail from a RAW file."""
    if Image is None:
        return None

    try:
        # 1. First attempt native Pillow (works for DNG and some TIFF-based RAWs)
        try:
            with Image.open(path) as img:
                return img.copy()
        except Exception:
            pass

        # 2. Extract embedded JPEG by scanning container stream for JPEG SOI marker (0xFFD8FFE0 or 0xFFD8FFE1)
        with path.open("rb") as f:
            data = f.read(10 * 1024 * 1024)  # Read first 10MB where preview JPEG resides

        start = data.find(b"\xff\xd8\xff")
        if start != -1:
            end = data.rfind(b"\xff\xd9")
            if end != -1 and end > start:
                jpeg_bytes = data[start : end + 2]
                return Image.open(io.BytesIO(jpeg_bytes))
    except Exception:
        return None

    return None


def get_live_photo_video_path(photo_path: Path) -> Optional[Path]:
    """Finds accompanying Live Photo video (.MOV or .MP4) with the same stem."""
    parent = photo_path.parent
    stem = photo_path.stem

    for ext in (".mov", ".MOV", ".mp4", ".MP4"):
        candidate = parent / f"{stem}{ext}"
        if candidate.exists() and candidate.resolve() != photo_path.resolve():
            return candidate

    return None
