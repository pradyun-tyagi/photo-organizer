"""Filesystem operations: safe moves, collisions, in-place writes, and directory cleanup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set, Tuple

from photo_organizer.config import DEFAULT_VIDEO_FOLDER
from photo_organizer.core.discovery import is_video_file


def destination_for_photo(
    base: Path,
    photo,
    video_folder_name: Optional[str] = DEFAULT_VIDEO_FOLDER,
) -> Path:
    is_vid = getattr(photo, "is_video", False) or is_video_file(photo.path)
    if is_vid and video_folder_name:
        if base.name == video_folder_name:
            return base / photo.year / photo.month / photo.path.name
        return base / video_folder_name / photo.year / photo.month / photo.path.name
    return base / photo.year / photo.month / photo.path.name


def unique_destination(destination: Path, source: Path) -> Path:
    """Find a non-colliding destination path by appending numeric suffixes if necessary."""
    if not destination.exists() or destination.resolve() == source.resolve():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}__{counter}{suffix}"
        if not candidate.exists() or candidate.resolve() == source.resolve():
            return candidate
        counter += 1


def move_file(
    source: Path,
    destination: Path,
    dry_run: bool,
    journal=None,
    run_id: Optional[str] = None,
) -> Tuple[bool, Path]:
    """Moves a single file safely with collision prevention and transaction journaling."""
    destination = unique_destination(destination, source)
    if source.resolve() == destination.resolve():
        return False, source

    if dry_run:
        print(f"DRY RUN move: {source} -> {destination}")
        return True, destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    print(f"Moved: {source} -> {destination}")

    if journal and run_id:
        journal.record_action(
            run_id=run_id,
            action_type="MOVE",
            source_path=str(source),
            dest_path=str(destination),
        )

    return True, destination


def move_file_with_companions(
    source: Path,
    destination: Path,
    companions: List[Path],
    dry_run: bool,
    journal=None,
    run_id: Optional[str] = None,
) -> Tuple[bool, Path, int]:
    """Moves an image and its paired companion files (Live Photo video, XMP) atomically."""
    moved, final_dest = move_file(source, destination, dry_run, journal=journal, run_id=run_id)
    if not moved:
        return False, source, 0

    companions_moved = 0
    if not companions:
        return True, final_dest, companions_moved

    dest_stem = final_dest.stem
    dest_dir = final_dest.parent

    for companion in companions:
        if not companion.exists():
            continue
        companion_dest = dest_dir / f"{dest_stem}{companion.suffix}"
        companion_moved, _ = move_file(
            companion,
            companion_dest,
            dry_run,
            journal=journal,
            run_id=run_id,
        )
        if companion_moved:
            companions_moved += 1

    return True, final_dest, companions_moved


def restore_access_modified_times(path: Path, original_stat: os.stat_result) -> None:
    atime_ns = getattr(original_stat, "st_atime_ns", None)
    mtime_ns = getattr(original_stat, "st_mtime_ns", None)
    if atime_ns is not None and mtime_ns is not None:
        os.utime(path, ns=(atime_ns, mtime_ns))
    else:
        os.utime(path, (original_stat.st_atime, original_stat.st_mtime))


def overwrite_existing_file_from_temp(path: Path, temp_path: Path) -> None:
    """Overwrites existing file in-place, preserving OS filesystem created timestamp."""
    with temp_path.open("rb") as src, path.open("r+b") as dst:
        dst.seek(0)
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
        dst.truncate()
        dst.flush()
        os.fsync(dst.fileno())


def remove_empty_dirs(
    root: Path,
    dry_run: bool,
    journal=None,
    run_id: Optional[str] = None,
) -> Tuple[int, int]:
    """Removes directories that became empty after moves, bottom-up."""
    removed = 0
    warnings = 0
    root = root.resolve()
    dry_run_removals: Set[Path] = set()

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        path = Path(dirpath)
        try:
            resolved = path.resolve()
        except OSError:
            warnings += 1
            continue

        if resolved == root:
            continue

        try:
            if dry_run:
                entries = list(path.iterdir())
                is_empty_after_cleanup = all(
                    entry.is_dir() and entry.resolve() in dry_run_removals
                    for entry in entries
                )
                if not is_empty_after_cleanup:
                    continue
                dry_run_removals.add(resolved)
                print(f"DRY RUN remove empty folder: {path}")
            else:
                path.rmdir()
                print(f"Removed empty folder: {path}")
                if journal and run_id:
                    journal.record_action(
                        run_id=run_id,
                        action_type="RMDIR",
                        source_path=str(path),
                        dest_path="",
                    )
            removed += 1
        except OSError:
            pass

    return removed, warnings


def allowed_enhanced_size(original_size: int, max_size_increase_percent: float) -> int:
    return int(original_size * (1 + max_size_increase_percent / 100))
