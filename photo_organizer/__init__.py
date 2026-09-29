from __future__ import annotations

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from photo_organizer.config import (
    DEFAULT_MAX_SIZE_INCREASE,
    DEFAULT_MAYBE_NON_RELEVANT_FOLDER,
    DEFAULT_NON_RELEVANT_FOLDER,
    DEFAULT_TO_DELETE_FOLDER,
    DEFAULT_VISUAL_THRESHOLD,
    EXIF_DATE_TAGS,
    FILENAME_DATE_PATTERNS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from photo_organizer.core.cache import PhotoCache
from photo_organizer.core.dates import (
    date_from_exif,
    date_from_filename,
    date_from_filesystem,
    date_from_quicktime_video,
    parse_exif_datetime,
    taken_date,
)
from photo_organizer.core.discovery import (
    discover_images,
    is_image_file,
    is_inside,
    is_video_file,
)
from photo_organizer.core.file_ops import (
    allowed_enhanced_size,
    destination_for_photo,
    move_file,
    move_file_with_companions,
    overwrite_existing_file_from_temp,
    remove_empty_dirs,
    restore_access_modified_times,
    unique_destination,
)
from photo_organizer.core.journal import Journal
from photo_organizer.dedupe.dsu import DisjointSet
from photo_organizer.dedupe.hasher import (
    compute_dhash,
    compute_phash,
    flattened_pixels,
    hamming_distance,
    image_metadata,
    sha256_file,
    visual_hash_and_size,
)
from photo_organizer.dedupe.matcher import (
    aspect_bucket,
    aspect_ratio_close,
    duplicate_groups,
    duplicate_paths_to_move,
    inspect_photo,
    keeper_for_group,
    parallel_map,
    populate_hashes_for_exact_candidates,
    visual_candidate_pairs,
    visual_hash_bands,
)
from photo_organizer.models import (
    ActionSummary,
    EnhancementPlan,
    ImageQualityMetrics,
    NonRelevantResult,
    PhotoInfo,
)
from photo_organizer.organizer import organize, scan_library
from photo_organizer.ui.cli import main, parse_args, print_summary
from photo_organizer.vision.classifier import (
    analysis_image,
    clamp,
    classify_non_relevant,
    classify_photo_for_screening,
    color_saturation_score,
    color_saturation_score_from_rgb,
    noise_score,
    sharpness_score,
    text_heaviness_score,
    text_heaviness_score_from_prepared,
)
from photo_organizer.vision.detectors import FACE_CASCADES, detect_face, detect_person
from photo_organizer.vision.enhancer import (
    apply_enhancement_plan,
    apply_gamma,
    build_enhancement_plan,
    denoise_rgb,
    enhance_in_place,
    enhance_path_for_worker,
    enhanced_image,
    enhancement_plan_for_image,
    enhancement_quality_steps,
    image_quality_metrics,
    save_candidate,
    save_candidate_bytes,
)
from photo_organizer.models import BurstGroup, BurstPhoto, PersonCluster
from photo_organizer.core.geo import extract_gps_from_exif, reverse_geocode_offline
from photo_organizer.core.exif_writer import backfill_exif_datetime, backfill_library_exif_dates
from photo_organizer.core.raw_heif import extract_raw_preview, is_raw_file, is_heif_file, get_live_photo_video_path
from photo_organizer.vision.bursts import detect_burst_groups, calculate_photo_quality_score
from photo_organizer.vision.faces import cluster_people_albums, extract_face_crops_and_features
from photo_organizer.vision.semantic import search_photos, analyze_photo_semantics
from photo_organizer.core.flashbacks import get_on_this_day_memories, get_flashback_cards

__all__ = [
    "ActionSummary",
    "DisjointSet",
    "EnhancementPlan",
    "ImageQualityMetrics",
    "Journal",
    "NonRelevantResult",
    "PhotoCache",
    "PhotoInfo",
    "allowed_enhanced_size",
    "analysis_image",
    "apply_enhancement_plan",
    "apply_gamma",
    "aspect_bucket",
    "aspect_ratio_close",
    "build_enhancement_plan",
    "clamp",
    "classify_non_relevant",
    "classify_photo_for_screening",
    "color_saturation_score",
    "color_saturation_score_from_rgb",
    "compute_dhash",
    "compute_phash",
    "date_from_exif",
    "date_from_filename",
    "date_from_filesystem",
    "date_from_quicktime_video",
    "denoise_rgb",
    "destination_for_photo",
    "detect_face",
    "detect_person",
    "discover_images",
    "duplicate_groups",
    "duplicate_paths_to_move",
    "enhance_in_place",
    "enhance_path_for_worker",
    "enhanced_image",
    "enhancement_plan_for_image",
    "enhancement_quality_steps",
    "flattened_pixels",
    "hamming_distance",
    "image_metadata",
    "image_quality_metrics",
    "inspect_photo",
    "is_image_file",
    "is_inside",
    "is_video_file",
    "keeper_for_group",
    "main",
    "move_file",
    "move_file_with_companions",
    "noise_score",
    "organize",
    "overwrite_existing_file_from_temp",
    "parallel_map",
    "parse_args",
    "parse_exif_datetime",
    "populate_hashes_for_exact_candidates",
    "print_summary",
    "remove_empty_dirs",
    "restore_access_modified_times",
    "save_candidate",
    "save_candidate_bytes",
    "scan_library",
    "sha256_file",
    "sharpness_score",
    "taken_date",
    "text_heaviness_score",
    "text_heaviness_score_from_prepared",
    "BurstGroup",
    "BurstPhoto",
    "PersonCluster",
    "extract_gps_from_exif",
    "reverse_geocode_offline",
    "backfill_exif_datetime",
    "backfill_library_exif_dates",
    "extract_raw_preview",
    "is_raw_file",
    "is_heif_file",
    "get_live_photo_video_path",
    "detect_burst_groups",
    "calculate_photo_quality_score",
    "cluster_people_albums",
    "extract_face_crops_and_features",
    "search_photos",
    "analyze_photo_semantics",
    "get_on_this_day_memories",
    "get_flashback_cards",
    "unique_destination",
    "visual_candidate_pairs",
    "visual_hash_and_size",
    "visual_hash_bands",
]
