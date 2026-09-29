"""Directory discovery and companion file pairing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Set, Union

from photo_organizer.config import (
    COMPANION_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def is_inside(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False


def discover_images(
    root: Path,
    excluded_dirs: Union[Path, Sequence[Path]],
    include_videos: bool = False,
) -> List[Path]:
    """Finds all eligible image and media files, skipping excluded review folders."""
    if isinstance(excluded_dirs, Path):
        excluded = [excluded_dirs]
    else:
        excluded = list(excluded_dirs)
    excluded = [path.resolve() for path in excluded]

    discovered: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current_path = Path(dirpath)
        current_resolved = current_path.resolve()
        if any(
            current_resolved == excluded_dir or is_inside(current_resolved, excluded_dir)
            for excluded_dir in excluded
        ):
            dirnames[:] = []
            continue

        kept_dirnames = []
        for dirname in dirnames:
            candidate = (current_path / dirname).resolve()
            if any(
                candidate == excluded_dir or is_inside(candidate, excluded_dir)
                for excluded_dir in excluded
            ):
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        for filename in filenames:
            path = current_path / filename
            if is_image_file(path) or (include_videos and is_video_file(path)):
                discovered.append(path)

    return sorted(discovered)


_DIR_FILE_NAMES_CACHE: Dict[Path, Set[str]] = {}


def clear_dir_cache() -> None:
    _DIR_FILE_NAMES_CACHE.clear()


def get_dir_files_set(parent: Path) -> Set[str]:
    resolved = parent.resolve()
    cached = _DIR_FILE_NAMES_CACHE.get(resolved)
    if cached is None:
        try:
            cached = set(os.listdir(resolved))
        except OSError:
            cached = set()
        _DIR_FILE_NAMES_CACHE[resolved] = cached
    return cached


def find_companion_files(image_path: Path) -> List[Path]:
    """Finds companion sidecars (Live Photo video, XMP, AAE) with matching stem using cached directory lookups."""
    companions: List[Path] = []
    parent = image_path.parent
    dir_files = get_dir_files_set(parent)
    stem = image_path.stem

    for ext in COMPANION_EXTENSIONS:
        for candidate_ext in (ext, ext.upper()):
            filename = f"{stem}{candidate_ext}"
            if filename in dir_files:
                candidate = parent / filename
                if candidate != image_path and candidate.is_file():
                    if candidate not in companions:
                        companions.append(candidate)

    return companions
