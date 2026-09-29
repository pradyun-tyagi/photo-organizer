"""Duplicate candidate matching, LSH band indexing, and keeper selection."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

from photo_organizer.config import VIDEO_EXTENSIONS
from photo_organizer.core.cache import PhotoCache
from photo_organizer.core.dates import date_from_filename, date_from_filesystem, date_from_quicktime_video
from photo_organizer.core.discovery import find_companion_files, is_video_file
from photo_organizer.dedupe.dsu import DisjointSet
from photo_organizer.dedupe.hasher import (
    compute_phash_for_path,
    full_image_inspection,
    hamming_distance,
    sha256_file,
)
from photo_organizer.models import PhotoInfo

T = TypeVar("T")
U = TypeVar("U")


def parallel_map(
    items: Sequence[T],
    worker_count: int,
    function: Callable[[T], U],
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[U]:
    total = len(items)
    if total == 0:
        return []

    if worker_count <= 1 or total <= 1:
        results = []
        for idx, item in enumerate(items):
            results.append(function(item))
            if on_progress and ((idx + 1) % 50 == 0 or (idx + 1) == total):
                on_progress(idx + 1, total)
        return results

    results = [None] * total  # type: ignore
    completed = 0
    update_interval = max(10, total // 100)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {executor.submit(function, item): idx for idx, item in enumerate(items)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if on_progress and (completed % update_interval == 0 or completed == total):
                on_progress(completed, total)
    return results


def inspect_photo(
    path: Path,
    cache: Optional[PhotoCache] = None,
    preloaded_cache: Optional[Dict[str, Tuple[PhotoInfo, int, int]]] = None,
) -> PhotoInfo:
    stat = path.stat()
    file_size = stat.st_size
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))

    resolved_str = str(path.resolve())
    if preloaded_cache is not None and resolved_str in preloaded_cache:
        cached_info, c_size, c_mtime = preloaded_cache[resolved_str]
        if c_size == file_size and c_mtime == mtime_ns:
            cached_info.companion_files = find_companion_files(path)
            return cached_info

    if cache is not None:
        cached = cache.get(path, file_size, mtime_ns)
        if cached is not None:
            cached.companion_files = find_companion_files(path)
            return cached

    width, height, visual_hash, phash, format_name, exif_date = full_image_inspection(path, compute_phash_flag=True)
    if exif_date is not None:
        date, source = exif_date, "exif"
    elif path.suffix.lower() in VIDEO_EXTENSIONS:
        video_date = date_from_quicktime_video(path)
        if video_date is not None:
            date, source = video_date, "video_metadata"
        else:
            filename_date = date_from_filename(path)
            if filename_date is not None:
                date, source = filename_date, "filename"
            else:
                date, source = date_from_filesystem(path), "filesystem"
    else:
        filename_date = date_from_filename(path)
        if filename_date is not None:
            date, source = filename_date, "filename"
        else:
            date, source = date_from_filesystem(path), "filesystem"

    if format_name is None and path.suffix:
        format_name = path.suffix.lstrip(".").upper()

    companions = find_companion_files(path)

    # Extract GPS and Location if EXIF present
    lat, lon, city, country = None, None, None, None
    try:
        from photo_organizer.core.geo import extract_gps_from_exif, reverse_geocode_offline
        from PIL import Image as _PILImg
        if _PILImg is not None and not is_video_file(path):
            with _PILImg.open(path) as _img:
                _exif = _img.getexif()
                if _exif:
                    coords = extract_gps_from_exif(_exif)
                    if coords:
                        lat, lon = coords
                        city, country = reverse_geocode_offline(lat, lon)
    except Exception:
        pass

    photo = PhotoInfo(
        path=path,
        file_size=file_size,
        sha256=None,
        taken_at=date,
        date_source=source,
        width=width,
        height=height,
        visual_hash=visual_hash,
        phash=phash,
        format_name=format_name,
        companion_files=companions,
        is_video=is_video_file(path),
        latitude=lat,
        longitude=lon,
        city=city,
        country=country,
    )

    if cache is not None:
        cache.set(photo, mtime_ns)

    return photo


def hash_photo_for_exact_duplicates(photo: PhotoInfo) -> Tuple[PhotoInfo, str]:
    return photo, sha256_file(photo.path)


def populate_hashes_for_exact_candidates(
    photos: Sequence[PhotoInfo],
    worker_count: int,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    by_size: Dict[int, List[PhotoInfo]] = {}
    for photo in photos:
        by_size.setdefault(photo.file_size, []).append(photo)

    candidates = [
        photo
        for same_size_photos in by_size.values()
        if len(same_size_photos) > 1
        for photo in same_size_photos
        if not photo.sha256
    ]
    for photo, digest in parallel_map(
        candidates,
        worker_count,
        hash_photo_for_exact_duplicates,
        on_progress=on_progress,
    ):
        photo.sha256 = digest


def aspect_ratio_close(left: PhotoInfo, right: PhotoInfo, tolerance: float = 0.06) -> bool:
    left_ratio = left.aspect_ratio
    right_ratio = right.aspect_ratio
    if left_ratio is None or right_ratio is None:
        return True
    diff1 = abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)
    if diff1 <= tolerance:
        return True
    # Check 90-degree rotated aspect ratio (e.g. portrait vs landscape rotated copies)
    diff2 = abs(left_ratio - (1.0 / right_ratio)) / max(left_ratio, 1.0 / right_ratio)
    return diff2 <= tolerance


def aspect_bucket(photo: PhotoInfo) -> Optional[int]:
    ratio = photo.aspect_ratio
    if ratio is None:
        return None
    return round(ratio * 50)


def visual_hash_bands(visual_hash: int) -> Iterable[Tuple[int, int]]:
    for band_index in range(8):
        yield band_index, (visual_hash >> (band_index * 8)) & 0xFF


def visual_candidate_pairs(
    photos: Sequence[PhotoInfo], visual_threshold: int
) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    band_index: Dict[Tuple[int, int], List[int]] = {}
    for index, photo in enumerate(photos):
        if photo.visual_hash is None:
            continue
        for band_index_num, band_val in visual_hash_bands(photo.visual_hash):
            key = (band_index_num, band_val)
            for other_index in band_index.get(key, []):
                if aspect_ratio_close(photos[other_index], photo):
                    pairs.add((other_index, index))
            band_index.setdefault(key, []).append(index)
    return pairs


def duplicate_groups(photos: Sequence[PhotoInfo], visual_threshold: int) -> List[List[PhotoInfo]]:
    dsu = DisjointSet(len(photos))

    # 1. Exact SHA-256 matches
    exact_groups: Dict[str, List[int]] = {}
    for index, photo in enumerate(photos):
        if not photo.sha256:
            continue
        exact_groups.setdefault(photo.sha256, []).append(index)
    for indexes in exact_groups.values():
        first = indexes[0]
        for index in indexes[1:]:
            dsu.union(first, index)

    # 2. Exact visual hash matches (Hamming distance 0)
    vhash_groups: Dict[int, List[int]] = {}
    for index, photo in enumerate(photos):
        if photo.visual_hash is not None:
            vhash_groups.setdefault(photo.visual_hash, []).append(index)
    for vhash, indexes in vhash_groups.items():
        if len(indexes) > 1:
            first = indexes[0]
            for index in indexes[1:]:
                if aspect_ratio_close(photos[first], photos[index]):
                    dsu.union(first, index)

    # 3. Exact pHash matches (Hamming distance 0)
    phash_groups: Dict[int, List[int]] = {}
    for index, photo in enumerate(photos):
        if photo.phash is not None:
            phash_groups.setdefault(photo.phash, []).append(index)
    for phash_val, indexes in phash_groups.items():
        if len(indexes) > 1:
            first = indexes[0]
            for index in indexes[1:]:
                if aspect_ratio_close(photos[first], photos[index]):
                    dsu.union(first, index)

    # 4. Multi-index inverted candidate matching for near-duplicates
    if visual_threshold > 0:
        band_index: Dict[Tuple[int, int], List[int]] = {}
        for index, photo in enumerate(photos):
            if photo.visual_hash is None:
                continue
            for band_number in range(8):
                band_val = (photo.visual_hash >> (band_number * 8)) & 0xFF
                key = (band_number, band_val)
                for other_index in band_index.get(key, []):
                    if dsu.find(other_index) == dsu.find(index):
                        continue
                    if not aspect_ratio_close(photos[other_index], photo):
                        continue
                    dist = hamming_distance(photos[other_index].visual_hash, photo.visual_hash)
                    if dist <= visual_threshold:
                        if photos[other_index].phash is not None and photo.phash is not None:
                            p_dist = hamming_distance(photos[other_index].phash, photo.phash)
                            if p_dist > 12:
                                continue
                        dsu.union(other_index, index)
                band_index.setdefault(key, []).append(index)

        # Also match by pHash bands for recompressed or slightly cropped images
        pband_index: Dict[Tuple[int, int], List[int]] = {}
        for index, photo in enumerate(photos):
            if photo.phash is None:
                continue
            for band_number in range(8):
                band_val = (photo.phash >> (band_number * 8)) & 0xFF
                key = (band_number, band_val)
                for other_index in pband_index.get(key, []):
                    if dsu.find(other_index) == dsu.find(index):
                        continue
                    if not aspect_ratio_close(photos[other_index], photo):
                        continue
                    p_dist = hamming_distance(photos[other_index].phash, photo.phash)
                    if p_dist <= min(8, visual_threshold + 2):
                        dsu.union(other_index, index)
                pband_index.setdefault(key, []).append(index)

    return [[photos[index] for index in indexes] for indexes in dsu.groups()]


def keeper_for_group(group: Sequence[PhotoInfo]) -> PhotoInfo:
    return max(
        group,
        key=lambda photo: (
            photo.pixels,
            len(photo.companion_files),  # Prefer keeping Live Photos / companion sets
            photo.file_size,
            photo.taken_at,
            -len(str(photo.path)),
        ),
    )


def duplicate_paths_to_move(
    photos: Sequence[PhotoInfo], visual_threshold: int
) -> Dict[Path, PhotoInfo]:
    duplicates: Dict[Path, PhotoInfo] = {}
    for group in duplicate_groups(photos, visual_threshold):
        keeper = keeper_for_group(group)
        for photo in group:
            if photo.path != keeper.path:
                duplicates[photo.path] = photo
    return duplicates

