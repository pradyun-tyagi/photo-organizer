"""EXIF metadata back-filling for images lacking native EXIF (WhatsApp, scans, etc.)."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None

    class UnidentifiedImageError(Exception):
        pass

from photo_organizer.core.file_ops import overwrite_existing_file_from_temp, restore_access_modified_times
from photo_organizer.models import PhotoInfo


def format_exif_datetime(dt: datetime) -> str:
    """Formats datetime as standard EXIF string: 'YYYY:MM:DD HH:MM:SS'."""
    return dt.strftime("%Y:%m:%d %H:%M:%S")


def backfill_exif_datetime(
    image_path: Path, dt: datetime, dry_run: bool = False
) -> Tuple[bool, str]:
    """Writes DateTimeOriginal and DateTime into image EXIF metadata atomically."""
    if Image is None:
        return False, "Pillow is not installed"

    suffix = image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".tiff", ".tif", ".webp"}:
        return False, f"EXIF writing not supported for {suffix}"

    if not image_path.exists():
        return False, "File does not exist"

    if dry_run:
        return True, f"would write {format_exif_datetime(dt)} to EXIF"

    exif_str = format_exif_datetime(dt)
    original_stat = image_path.stat()

    try:
        with Image.open(image_path) as image:
            exif = image.getexif()

            # EXIF tags:
            # 0x0132 = DateTime
            # 0x9003 = DateTimeOriginal
            # 0x9004 = DateTimeDigitized
            exif[0x0132] = exif_str
            exif[0x9003] = exif_str
            exif[0x9004] = exif_str

            save_kwargs = {"exif": exif}
            fmt = image.format or "JPEG"
            if fmt == "JPEG":
                save_kwargs["quality"] = 98
                save_kwargs["optimize"] = True

            with tempfile.NamedTemporaryFile(
                prefix=f".exif_{image_path.stem}.",
                suffix=image_path.suffix,
                dir=str(image_path.parent),
                delete=False,
            ) as temp_handle:
                temp_path = Path(temp_handle.name)

            try:
                image.save(temp_path, format=fmt, **save_kwargs)
                overwrite_existing_file_from_temp(image_path, temp_path)
                restore_access_modified_times(image_path, original_stat)
                return True, f"wrote EXIF date {exif_str}"
            finally:
                if temp_path.exists():
                    temp_path.unlink()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return False, f"failed to write EXIF: {exc}"


def backfill_library_exif_dates(
    photos: Sequence[PhotoInfo], dry_run: bool = False
) -> Dict[str, int]:
    """Batch updates EXIF dates for all photos whose date was resolved from filename or filesystem."""
    results = {"total_candidates": 0, "updated": 0, "skipped": 0, "failed": 0}

    for photo in photos:
        if photo.date_source == "exif" or photo.is_video:
            results["skipped"] += 1
            continue

        results["total_candidates"] += 1
        success, _ = backfill_exif_datetime(photo.path, photo.taken_at, dry_run=dry_run)
        if success:
            results["updated"] += 1
            if not dry_run:
                photo.date_source = "exif"
        else:
            results["failed"] += 1

    return results
