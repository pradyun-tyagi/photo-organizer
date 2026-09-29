"""Date parsing utilities from EXIF, video metadata, filenames, and filesystems."""

from __future__ import annotations

import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None

    class UnidentifiedImageError(Exception):
        pass

from photo_organizer.config import EXIF_DATE_TAGS, FILENAME_DATE_PATTERNS, VIDEO_EXTENSIONS


def parse_exif_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    if not isinstance(value, str):
        return None

    value = value.strip().replace("\x00", "")
    candidates = (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y:%m:%d",
        "%Y-%m-%d",
    )
    for fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def date_from_exif(path: Path) -> Optional[datetime]:
    if Image is None:
        return None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return None
            for tag in EXIF_DATE_TAGS:
                taken_at = parse_exif_datetime(exif.get(tag))
                if taken_at is not None:
                    return taken_at
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    return None


def date_from_filename(path: Path) -> Optional[datetime]:
    stem = path.stem
    for pattern in FILENAME_DATE_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        groups = match.groupdict()
        try:
            return datetime(
                int(groups["year"]),
                int(groups["month"]),
                int(groups["day"]),
                int(groups.get("hour") or 0),
                int(groups.get("minute") or 0),
                int(groups.get("second") or 0),
            )
        except (TypeError, ValueError):
            continue
    return None


def date_from_filesystem(path: Path) -> datetime:
    stat = path.stat()
    created = getattr(stat, "st_birthtime", None)
    timestamp = created if created is not None else stat.st_mtime
    return datetime.fromtimestamp(timestamp)


def date_from_quicktime_video(path: Path) -> Optional[datetime]:
    """Extract creation time from MP4 / MOV mvhd atom if present."""
    try:
        with path.open("rb") as f:
            # Search within first 1MB for the mvhd atom
            chunk = f.read(1024 * 1024)
            mvhd_pos = chunk.find(b"mvhd")
            if mvhd_pos != -1 and mvhd_pos + 8 <= len(chunk):
                version = chunk[mvhd_pos + 4]
                if version == 0 and mvhd_pos + 12 <= len(chunk):
                    # 32-bit creation time (seconds since 1904-01-01)
                    creation_secs = struct.unpack(">I", chunk[mvhd_pos + 8 : mvhd_pos + 12])[0]
                    if creation_secs > 0:
                        base = datetime(1904, 1, 1)
                        # Quick sanity check for valid range (1970 - 2040)
                        dt = base + timedelta(seconds=creation_secs)
                        if 1970 <= dt.year <= 2040:
                            return dt
                elif version == 1 and mvhd_pos + 16 <= len(chunk):
                    # 64-bit creation time
                    creation_secs = struct.unpack(">Q", chunk[mvhd_pos + 8 : mvhd_pos + 16])[0]
                    if creation_secs > 0:
                        base = datetime(1904, 1, 1)
                        dt = base + timedelta(seconds=creation_secs)
                        if 1970 <= dt.year <= 2040:
                            return dt
    except Exception:
        pass
    return None


def taken_date(path: Path) -> Tuple[datetime, str]:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        video_date = date_from_quicktime_video(path)
        if video_date is not None:
            return video_date, "video_metadata"

    exif_date = date_from_exif(path)
    if exif_date is not None:
        return exif_date, "exif"

    filename_date = date_from_filename(path)
    if filename_date is not None:
        return filename_date, "filename"

    return date_from_filesystem(path), "filesystem"
