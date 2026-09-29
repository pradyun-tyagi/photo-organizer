"""HEIC/HEIF and Camera RAW decoding and Live Photo companion handling."""

from __future__ import annotations

import io
import struct
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


def parse_exif_thumbnail(data: bytes) -> Optional[bytes]:
    """Extracts raw embedded JPEG thumbnail from EXIF APP1 IFD1 segment in < 0.2ms.
    Avoids loading or decompressing the main image bitmap.
    """
    if not data.startswith(b"\xff\xd8"):
        return None
    pos = 2
    data_len = len(data)
    while pos < data_len - 4:
        if data[pos] != 0xFF:
            break
        marker = (data[pos] << 8) | data[pos + 1]
        length = (data[pos + 2] << 8) | data[pos + 3]
        if marker == 0xFFE1:  # APP1 EXIF
            app1 = data[pos + 4 : pos + 2 + length]
            if app1.startswith(b"Exif\x00\x00"):
                tiff = app1[6:]
                if len(tiff) < 8:
                    return None
                endian = "<" if tiff[:2] == b"II" else ">"
                try:
                    tag_mark = struct.unpack(endian + "H", tiff[2:4])[0]
                    if tag_mark != 0x002A:
                        return None
                    ifd0_offset = struct.unpack(endian + "I", tiff[4:8])[0]
                    if ifd0_offset >= len(tiff):
                        return None
                    num_entries = struct.unpack(endian + "H", tiff[ifd0_offset : ifd0_offset + 2])[0]
                    ifd1_ptr_offset = ifd0_offset + 2 + num_entries * 12
                    if ifd1_ptr_offset + 4 > len(tiff):
                        return None
                    ifd1_offset = struct.unpack(endian + "I", tiff[ifd1_ptr_offset : ifd1_ptr_offset + 4])[0]
                    if ifd1_offset == 0 or ifd1_offset >= len(tiff):
                        return None
                    num_entries_1 = struct.unpack(endian + "H", tiff[ifd1_offset : ifd1_offset + 2])[0]
                    thumb_offset = None
                    thumb_len = None
                    for i in range(num_entries_1):
                        entry_pos = ifd1_offset + 2 + i * 12
                        entry = tiff[entry_pos : entry_pos + 12]
                        if len(entry) < 12:
                            break
                        tag = struct.unpack(endian + "H", entry[:2])[0]
                        val = struct.unpack(endian + "I", entry[8:12])[0]
                        if tag == 0x0201:  # JpegIFOffset
                            thumb_offset = val
                        elif tag == 0x0202:  # JpegIFByteCount
                            thumb_len = val
                    if thumb_offset and thumb_len and thumb_offset + thumb_len <= len(tiff):
                        thumb = tiff[thumb_offset : thumb_offset + thumb_len]
                        if thumb.startswith(b"\xff\xd8"):
                            return thumb
                except Exception:
                    return None
            return None
        elif (marker & 0xFF00) == 0xFF00 and marker not in (0xFFE0, 0xFFE1, 0xFFE2, 0xFFFE):
            break
        pos += 2 + length
    return None


def extract_fast_thumbnail(path: Path) -> Optional[bytes]:
    """Fast extraction of embedded thumbnail without decompressing the main image.
    Works for JPEG files (via EXIF APP1) and RAW files (via container stream) in < 1ms.
    """
    if is_raw_file(path):
        raw_img = extract_raw_preview(path)
        if raw_img is not None:
            buf = io.BytesIO()
            resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR if Image else 2)
            raw_img.thumbnail((400, 400), resample)
            raw_img.convert("RGB").save(buf, format="JPEG", quality=75)
            return buf.getvalue()
        return None

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        try:
            with open(path, "rb") as f:
                head = f.read(131072)  # Read first 128KB where EXIF header resides
            thumb = parse_exif_thumbnail(head)
            if thumb and len(thumb) >= 1024:
                return thumb
        except Exception:
            pass

    return None

