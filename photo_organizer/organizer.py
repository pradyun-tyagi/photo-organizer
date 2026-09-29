"""High-level organizer workflow and scan coordinator."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from photo_organizer.config import (
    DEFAULT_MAX_SIZE_INCREASE,
    DEFAULT_MAYBE_NON_RELEVANT_FOLDER,
    DEFAULT_NON_RELEVANT_FOLDER,
    DEFAULT_TO_DELETE_FOLDER,
    DEFAULT_VIDEO_FOLDER,
    DEFAULT_VISUAL_THRESHOLD,
)
from photo_organizer.core.cache import PhotoCache
from photo_organizer.core.discovery import discover_images, is_image_file, is_video_file
from photo_organizer.core.file_ops import (
    destination_for_photo,
    move_file,
    move_file_with_companions,
    remove_empty_dirs,
)
from photo_organizer.core.journal import Journal
from photo_organizer.dedupe.matcher import (
    duplicate_groups,
    duplicate_paths_to_move,
    inspect_photo,
    keeper_for_group,
    parallel_map,
    populate_hashes_for_exact_candidates,
)
from photo_organizer.core.flashbacks import get_flashback_cards
from photo_organizer.models import ActionSummary, NonRelevantResult, PhotoInfo
from photo_organizer.vision.bursts import detect_burst_groups
from photo_organizer.vision.classifier import classify_photo_for_screening
from photo_organizer.vision.enhancer import enhance_path_for_worker
from photo_organizer.vision.faces import cluster_people_albums


def get_default_workers() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count))


def scan_library(
    root: Path,
    visual_threshold: int = DEFAULT_VISUAL_THRESHOLD,
    to_delete_folder_name: str = DEFAULT_TO_DELETE_FOLDER,
    non_relevant_folder_name: str = DEFAULT_NON_RELEVANT_FOLDER,
    maybe_non_relevant_folder_name: str = DEFAULT_MAYBE_NON_RELEVANT_FOLDER,
    video_folder_name: str = DEFAULT_VIDEO_FOLDER,
    skip_non_relevant: bool = False,
    include_videos: bool = True,
    worker_count: int = 0,
    enable_cache: bool = True,
    progress_callback: Optional[Callable[[str, float, Dict], None]] = None,
) -> Dict:
    """Scans and analyzes a photo library without modifying any files on disk."""
    root = root.expanduser().resolve()
    to_delete_dir = root / to_delete_folder_name
    non_relevant_dir = root / non_relevant_folder_name
    maybe_non_relevant_dir = root / maybe_non_relevant_folder_name

    if worker_count <= 0:
        worker_count = get_default_workers()

    cache = PhotoCache(root_dir=root) if enable_cache else None

    if progress_callback:
        progress_callback("Discovering files...", 0.02, {})

    image_paths = discover_images(
        root,
        [to_delete_dir, non_relevant_dir, maybe_non_relevant_dir],
        include_videos=include_videos,
    )
    total_images = len(image_paths)

    if progress_callback:
        progress_callback(
            f"Discovered {total_images:,} images. Inspecting metadata...",
            0.05,
            {"total": total_images},
        )

    preloaded_cache = cache.load_cache_bulk(image_paths) if cache else None

    def _inspect_progress(completed: int, total: int):
        pct = 0.05 + 0.35 * (completed / total)
        if progress_callback:
            progress_callback(
                f"Inspecting metadata ({completed:,} / {total:,})...",
                pct,
                {"completed": completed, "total": total},
            )

    def _inspect_with_cache(path: Path) -> PhotoInfo:
        return inspect_photo(path, cache=None, preloaded_cache=preloaded_cache)

    photos = parallel_map(
        image_paths,
        worker_count,
        _inspect_with_cache,
        on_progress=_inspect_progress,
    )

    if cache is not None:
        new_items = []
        for photo in photos:
            resolved_str = str(photo.path.resolve())
            if not preloaded_cache or resolved_str not in preloaded_cache:
                try:
                    stat = photo.path.stat()
                    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                    new_items.append((photo, mtime_ns))
                except OSError:
                    pass
        if new_items:
            cache.set_bulk(new_items)

    def _hash_progress(completed: int, total: int):
        pct = 0.40 + 0.10 * (completed / total)
        if progress_callback:
            progress_callback(
                f"Checking exact duplicates ({completed:,} / {total:,})...",
                pct,
                {"completed": completed, "total": total},
            )

    populate_hashes_for_exact_candidates(
        photos,
        worker_count,
        on_progress=_hash_progress,
    )

    if progress_callback:
        progress_callback("Grouping exact and visual duplicates...", 0.52, {"scanned": len(photos)})

    groups = duplicate_groups(photos, visual_threshold)
    duplicate_clusters = []
    duplicate_path_set = set()

    for idx, group in enumerate(groups):
        keeper = keeper_for_group(group)
        cluster_info = {
            "group_id": idx,
            "keeper_path": str(keeper.path),
            "members": [
                {
                    "path": str(p.path),
                    "relative_path": str(p.path.relative_to(root)),
                    "file_size": p.file_size,
                    "width": p.width,
                    "height": p.height,
                    "taken_at": p.taken_at.isoformat(),
                    "date_source": p.date_source,
                    "is_keeper": p.path == keeper.path,
                    "companion_count": len(p.companion_files),
                }
                for p in group
            ],
        }
        duplicate_clusters.append(cluster_info)
        for p in group:
            if p.path != keeper.path:
                duplicate_path_set.add(p.path)

    kept_photos = [p for p in photos if p.path not in duplicate_path_set]
    non_relevant_items = []
    maybe_non_relevant_items = []

    if not skip_non_relevant:
        kept_images = [p for p in kept_photos if is_image_file(p.path)]
        total_kept = len(kept_images)
        if total_kept > 0:
            if progress_callback:
                progress_callback(
                    f"Screening text forwards ({total_kept:,} photos)...",
                    0.55,
                    {"kept_count": total_kept},
                )

            def _screen_progress(completed: int, total: int):
                pct = 0.55 + 0.40 * (completed / total)
                if progress_callback:
                    progress_callback(
                        f"Screening text forwards ({completed:,} / {total:,})...",
                        pct,
                        {"completed": completed, "total": total},
                    )

            screening_inputs = [(p, p.date_source == "exif") for p in kept_images]
            screening_results = parallel_map(
                screening_inputs,
                worker_count,
                classify_photo_for_screening,
                on_progress=_screen_progress,
            )

            for photo, result in screening_results:
                if not result.is_non_relevant and not result.is_maybe_non_relevant:
                    continue

                item_data = {
                    "path": str(photo.path),
                    "relative_path": str(photo.path.relative_to(root)),
                    "text_score": result.text_score,
                    "reason": result.reason,
                    "face_detected": result.face_detected,
                    "person_detected": result.person_detected,
                    "file_size": photo.file_size,
                    "width": photo.width,
                    "height": photo.height,
                }
                if result.is_non_relevant:
                    non_relevant_items.append(item_data)
                elif result.is_maybe_non_relevant:
                    maybe_non_relevant_items.append(item_data)

    if progress_callback:
        progress_callback("Generating albums, map pins, and flashback memories...", 0.95, {})

    # 1. Geographic photos for Map
    geo_photos = [
        {
            "path": str(p.path),
            "lat": p.latitude,
            "lon": p.longitude,
            "city": p.city,
            "country": p.country,
            "taken_at": p.taken_at.strftime("%Y-%m-%d %H:%M"),
        }
        for p in photos
        if p.latitude is not None and p.longitude is not None
    ]

    # 2. Burst groups and best shot selector
    bursts = detect_burst_groups(photos)
    burst_groups_data = [
        {
            "group_id": b.group_id,
            "best_path": str(b.best_photo.path),
            "photos": [
                {
                    "path": str(bp.photo.path),
                    "filename": bp.photo.path.name,
                    "score": bp.score,
                    "sharpness": bp.sharpness,
                    "is_best": bp.is_best_shot,
                }
                for bp in b.photos
            ],
        }
        for b in bursts
    ]

    # 3. People Albums face clustering
    people_clusters = cluster_people_albums(photos[:400])
    people_albums_data = [
        {
            "person_id": pa.person_id,
            "name": pa.name,
            "face_count": pa.face_count,
            "thumbnail": pa.representative_crop_data,
            "photo_paths": pa.photo_paths,
        }
        for pa in people_clusters
    ]

    # 4. Flashback memories ("On This Day")
    flashback_cards = get_flashback_cards(photos)

    # 5. EXIF backfill candidates
    exif_backfill_count = len([p for p in photos if p.date_source != "exif" and not p.is_video])

    if progress_callback:
        progress_callback("Scan completed successfully.", 1.0, {"ready": True})

    scan_result = {
        "root": str(root),
        "total_scanned": len(photos),
        "photos": photos,
        "duplicate_clusters": duplicate_clusters,
        "duplicate_count": len(duplicate_path_set),
        "non_relevant_items": non_relevant_items,
        "maybe_non_relevant_items": maybe_non_relevant_items,
        "geo_photos": geo_photos,
        "burst_groups": burst_groups_data,
        "people_albums": people_albums_data,
        "flashbacks": flashback_cards,
        "exif_backfill_count": exif_backfill_count,
    }

    if cache is not None:
        try:
            cache.save_studio_state({
                "root": str(root),
                "total_scanned": len(photos),
                "people_albums": people_albums_data,
                "flashbacks": flashback_cards,
                "geo_photos": geo_photos,
                "burst_groups": burst_groups_data,
                "exif_backfill_count": exif_backfill_count,
            })
        except Exception:
            pass

    return scan_result


def organize(
    root: Path,
    dry_run: bool = False,
    visual_threshold: int = DEFAULT_VISUAL_THRESHOLD,
    to_delete_folder_name: str = DEFAULT_TO_DELETE_FOLDER,
    skip_enhance: bool = False,
    non_relevant_folder_name: str = DEFAULT_NON_RELEVANT_FOLDER,
    maybe_non_relevant_folder_name: str = DEFAULT_MAYBE_NON_RELEVANT_FOLDER,
    video_folder_name: str = DEFAULT_VIDEO_FOLDER,
    skip_non_relevant: bool = False,
    include_videos: bool = True,
    max_size_increase_percent: float = DEFAULT_MAX_SIZE_INCREASE,
    worker_count: int = 0,
    enhancement_mode: str = "balanced",
    enable_cache: bool = True,
    journal: Optional[Journal] = None,
    progress_callback: Optional[Callable[[str, float, Dict], None]] = None,
    custom_keeper_map: Optional[Dict[str, str]] = None,
    custom_triage_decisions: Optional[Dict[str, str]] = None,
) -> ActionSummary:
    root = root.expanduser().resolve()
    to_delete_dir = root / to_delete_folder_name
    non_relevant_dir = root / non_relevant_folder_name
    maybe_non_relevant_dir = root / maybe_non_relevant_folder_name
    summary = ActionSummary()

    if worker_count <= 0:
        worker_count = get_default_workers()

    if journal is None and not dry_run:
        journal = Journal(root_dir=root)

    run_id = journal.start_run(root, dry_run) if (journal is not None and not dry_run) else None
    cache = PhotoCache(root_dir=root) if enable_cache else None

    if progress_callback:
        progress_callback("Discovering images...", 0.02, {})

    image_paths = discover_images(
        root,
        [to_delete_dir, non_relevant_dir, maybe_non_relevant_dir],
        include_videos=include_videos,
    )
    total_images = len(image_paths)

    preloaded_cache = cache.load_cache_bulk(image_paths) if cache else None

    def _inspect_progress(completed: int, total: int):
        pct = 0.05 + 0.20 * (completed / total)
        if progress_callback:
            progress_callback(
                f"Inspecting metadata ({completed:,} / {total:,})...",
                pct,
                {"completed": completed, "total": total},
            )

    def _inspect_with_cache(path: Path) -> PhotoInfo:
        return inspect_photo(path, cache=None, preloaded_cache=preloaded_cache)

    photos = parallel_map(
        image_paths,
        worker_count,
        _inspect_with_cache,
        on_progress=_inspect_progress,
    )

    if cache is not None:
        new_items = []
        for photo in photos:
            resolved_str = str(photo.path.resolve())
            if not preloaded_cache or resolved_str not in preloaded_cache:
                try:
                    stat = photo.path.stat()
                    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
                    new_items.append((photo, mtime_ns))
                except OSError:
                    pass
        if new_items:
            cache.set_bulk(new_items)

    populate_hashes_for_exact_candidates(photos, worker_count)
    summary.scanned = len(photos)

    if progress_callback:
        progress_callback("Analyzing duplicates...", 0.28, {"scanned": summary.scanned})

    duplicates: Dict[Path, PhotoInfo] = {}
    if custom_keeper_map:
        groups = duplicate_groups(photos, visual_threshold)
        for group in groups:
            keeper_str = None
            for p in group:
                if str(p.path) in custom_keeper_map:
                    keeper_str = custom_keeper_map[str(p.path)]
                    break
            if keeper_str == "__KEEP_ALL__":
                continue
            keeper = next((p for p in group if str(p.path) == keeper_str), None) or keeper_for_group(group)
            for p in group:
                if p.path != keeper.path:
                    duplicates[p.path] = p
    else:
        duplicates = duplicate_paths_to_move(photos, visual_threshold)

    duplicate_path_set = set(duplicates)
    if custom_triage_decisions:
        for p in photos:
            p_str = str(p.path)
            p_res = str(p.path.resolve())
            if custom_triage_decisions.get(p_str) == "DELETE" or custom_triage_decisions.get(p_res) == "DELETE":
                duplicate_path_set.add(p.path)

    # 1. Move duplicates into to-be-deleted
    for photo in photos:
        if photo.path not in duplicate_path_set:
            continue
        destination = destination_for_photo(to_delete_dir, photo, video_folder_name=video_folder_name)
        companions = [c for c in photo.companion_files if not is_video_file(c)] if is_image_file(photo.path) else photo.companion_files
        try:
            moved, dest_final, companions_moved = move_file_with_companions(
                photo.path,
                destination,
                companions,
                dry_run,
                journal=journal,
                run_id=run_id,
            )
            if moved:
                summary.duplicates_moved += 1
                summary.companions_moved += companions_moved
                summary.moved_paths_map[str(photo.path.resolve())] = str(dest_final.resolve())
        except OSError as exc:
            summary.warnings += 1
            print(f"WARNING could not move duplicate {photo.path}: {exc}", file=sys.stderr)

    # 2. Screen non-relevant and maybe-non-relevant
    kept_photos = [photo for photo in photos if photo.path not in duplicate_path_set]
    relevant_photos = []

    if skip_non_relevant:
        relevant_photos = kept_photos
    else:
        kept_images = [photo for photo in kept_photos if is_image_file(photo.path)]
        non_images = [photo for photo in kept_photos if not is_image_file(photo.path)]
        relevant_photos.extend(non_images)
        total_kept = len(kept_images)
        if total_kept > 0:
            if progress_callback:
                progress_callback(
                    f"Screening text forwards ({total_kept:,} photos)...",
                    0.35,
                    {"kept_count": total_kept},
                )

            def _screen_progress(completed: int, total: int):
                pct = 0.35 + 0.30 * (completed / total)
                if progress_callback:
                    progress_callback(
                        f"Screening text forwards ({completed:,} / {total:,})...",
                        pct,
                        {"completed": completed, "total": total},
                    )

            screening_inputs = [
                (photo, photo.date_source == "exif") for photo in kept_images
            ]
            screening_results = parallel_map(
                screening_inputs,
                worker_count,
                classify_photo_for_screening,
                on_progress=_screen_progress,
            )
            for photo, result in screening_results:
                path_str = str(photo.path)
                resolved_str = str(photo.path.resolve())
                if custom_triage_decisions:
                    decision = custom_triage_decisions.get(path_str) or custom_triage_decisions.get(resolved_str)
                    if decision == "KEEP":
                        relevant_photos.append(photo)
                        continue
                    elif decision == "DISCARD":
                        result.is_non_relevant = True
                        result.is_maybe_non_relevant = False

                if not result.is_non_relevant and not result.is_maybe_non_relevant:
                    relevant_photos.append(photo)
                    continue

                if result.is_non_relevant:
                    destination = destination_for_photo(non_relevant_dir, photo, video_folder_name=video_folder_name)
                    folder_label = "Non-relevant"
                else:
                    destination = destination_for_photo(maybe_non_relevant_dir, photo, video_folder_name=video_folder_name)
                    folder_label = "Maybe non-relevant"

                companions = [c for c in photo.companion_files if not is_video_file(c)] if is_image_file(photo.path) else photo.companion_files
                try:
                    moved, dest_final, companions_moved = move_file_with_companions(
                        photo.path,
                        destination,
                        companions,
                        dry_run,
                        journal=journal,
                        run_id=run_id,
                    )
                    if moved:
                        if result.is_non_relevant:
                            summary.non_relevant_moved += 1
                        else:
                            summary.maybe_non_relevant_moved += 1
                        summary.companions_moved += companions_moved
                        summary.moved_paths_map[str(photo.path.resolve())] = str(dest_final.resolve())
                        print(f"{folder_label}: {photo.path} ({result.reason})")
                except OSError as exc:
                    summary.warnings += 1
                    print(
                        f"WARNING could not move screened image {photo.path}: {exc}",
                        file=sys.stderr,
                    )
                    relevant_photos.append(photo)

    # 3. Organize relevant photos into YYYY/YYYY-MM/ and videos into video/YYYY/YYYY-MM/
    if progress_callback:
        progress_callback("Organizing files by capture date...", 0.68, {})

    updated_paths: List[Path] = []
    for photo in relevant_photos:
        destination = destination_for_photo(root, photo, video_folder_name=video_folder_name)
        companions = [c for c in photo.companion_files if not is_video_file(c)] if is_image_file(photo.path) else photo.companion_files
        try:
            moved, final_path, companions_moved = move_file_with_companions(
                photo.path,
                destination,
                companions,
                dry_run,
                journal=journal,
                run_id=run_id,
            )
            if moved:
                summary.organized_moved += 1
                summary.companions_moved += companions_moved
                summary.moved_paths_map[str(photo.path.resolve())] = str(final_path.resolve())
                updated_paths.append(photo.path if dry_run else final_path)
            else:
                summary.already_organized += 1
                summary.moved_paths_map[str(photo.path.resolve())] = str(final_path.resolve())
                updated_paths.append(final_path)
        except OSError as exc:
            summary.warnings += 1
            print(f"WARNING could not organize {photo.path}: {exc}", file=sys.stderr)

    # 4. Enhance kept pictures
    if skip_enhance:
        removed, warnings = remove_empty_dirs(root, dry_run, journal=journal, run_id=run_id)
        summary.empty_dirs_removed += removed
        summary.warnings += warnings
        if run_id:
            journal.complete_run(run_id, summary.__dict__)
        if progress_callback:
            progress_callback("Organization complete.", 1.0, {"summary": summary.__dict__})
        return summary

    images_to_enhance = [p for p in updated_paths if is_image_file(p)]
    total_to_enhance = len(images_to_enhance)
    if progress_callback:
        progress_callback(f"Enhancing photos ({total_to_enhance:,} photos)...", 0.75, {})

    def _enhance_progress(completed: int, total: int):
        pct = 0.75 + 0.20 * (completed / total)
        if progress_callback:
            progress_callback(
                f"Enhancing photos ({completed:,} / {total:,})...",
                pct,
                {"completed": completed, "total": total},
            )

    enhancement_inputs = [
        (path, dry_run, max_size_increase_percent, enhancement_mode)
        for path in images_to_enhance
    ]
    enhancement_results = parallel_map(
        enhancement_inputs,
        worker_count,
        enhance_path_for_worker,
        on_progress=_enhance_progress,
    )
    for path, enhanced, reason in enhancement_results:
        if enhanced:
            summary.enhanced += 1
            print(f"Enhanced: {path} ({reason})")
            if journal and run_id:
                journal.record_action(
                    run_id=run_id,
                    action_type="ENHANCE",
                    source_path=str(path),
                    dest_path=str(path),
                    details=reason,
                )
        else:
            summary.enhancement_skipped += 1
            print(f"Skipped enhancement: {path} ({reason})")

    # 5. Clean up empty source directories
    removed, warnings = remove_empty_dirs(root, dry_run, journal=journal, run_id=run_id)
    summary.empty_dirs_removed += removed
    summary.warnings += warnings

    if run_id:
        journal.complete_run(run_id, summary.__dict__)

    if cache is not None and not dry_run and summary.moved_paths_map:
        try:
            cached_state = cache.load_studio_state()
            if cached_state:
                for pa in cached_state.get("people_albums", []):
                    pa["photo_paths"] = [
                        summary.moved_paths_map.get(str(Path(p).resolve()), p)
                        for p in pa.get("photo_paths", [])
                    ]
                for fb in cached_state.get("flashbacks", []):
                    for item in fb.get("photos", []):
                        if "path" in item:
                            item["path"] = summary.moved_paths_map.get(str(Path(item["path"]).resolve()), item["path"])
                for gp in cached_state.get("geo_photos", []):
                    if "path" in gp:
                        gp["path"] = summary.moved_paths_map.get(str(Path(gp["path"]).resolve()), gp["path"])
                cached_state["last_summary"] = summary.__dict__
                cache.save_studio_state(cached_state)
        except Exception:
            pass

    if progress_callback:
        progress_callback("All tasks complete.", 1.0, {"summary": summary.__dict__})

    return summary
