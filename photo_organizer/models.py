"""Domain models and data structures for Photo Organizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PhotoInfo:
    path: Path
    file_size: int = 0
    sha256: Optional[str] = None
    taken_at: datetime = field(default_factory=datetime.now)
    date_source: str = "exif"
    width: int = 0
    height: int = 0
    visual_hash: Optional[int] = None  # dHash
    phash: Optional[int] = None        # pHash (DCT)
    format_name: Optional[str] = None
    companion_files: List[Path] = field(default_factory=list)
    is_video: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: Optional[str] = None
    country: Optional[str] = None
    dominant_color: Optional[str] = None

    @property
    def year(self) -> str:
        return f"{self.taken_at.year:04d}"

    @property
    def month(self) -> str:
        return f"{self.taken_at.year:04d}-{self.taken_at.month:02d}"

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> Optional[float]:
        if not self.width or not self.height:
            return None
        return self.width / self.height

    @property
    def has_live_photo_video(self) -> bool:
        return any(c.suffix.lower() in {".mov", ".mp4"} for c in self.companion_files)


@dataclass
class BurstPhoto:
    photo: PhotoInfo
    score: float
    sharpness: float
    is_best_shot: bool = False
    is_golden_frame: bool = False
    eye_openness: float = 0.0
    motion_blur: float = 0.0


@dataclass
class BurstGroup:
    group_id: str
    photos: List[BurstPhoto]
    best_photo: PhotoInfo
    golden_frame: Optional[PhotoInfo] = None


@dataclass
class PersonCluster:
    person_id: str
    name: str
    face_count: int
    representative_crop_data: Optional[str] = None  # base64 data URI
    photo_paths: List[str] = field(default_factory=list)


@dataclass
class ActionSummary:
    scanned: int = 0
    duplicates_moved: int = 0
    burst_duplicates_moved: int = 0
    non_relevant_moved: int = 0
    maybe_non_relevant_moved: int = 0
    organized_moved: int = 0
    already_organized: int = 0
    exif_backfilled: int = 0
    enhanced: int = 0
    enhancement_skipped: int = 0
    empty_dirs_removed: int = 0
    companions_moved: int = 0
    warnings: int = 0
    moved_paths_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class NonRelevantResult:
    is_non_relevant: bool
    reason: str
    text_score: float
    face_detected: bool
    face_detector_available: bool
    is_maybe_non_relevant: bool = False
    person_detected: bool = False
    person_detector_available: bool = False


@dataclass
class ImageQualityMetrics:
    brightness: float
    contrast: float
    sharpness: float
    saturation: float
    noise: float
    text_score: float
    haze_score: float = 0.0


@dataclass
class EnhancementPlan:
    should_enhance: bool
    reason: str
    gamma: float = 1.0
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    color_factor: float = 1.0
    sharpness_factor: float = 1.0
    autocontrast_cutoff: float = 0.0
    denoise: bool = False
    dehaze: bool = False
    dehaze_strength: float = 0.0
    shadow_chroma_denoise: bool = False


@dataclass
class JournalRecord:
    run_id: str
    action_type: str  # 'MOVE', 'ENHANCE', 'RMDIR'
    source_path: str
    dest_path: str
    timestamp: str
    details: Optional[str] = None
    id: Optional[int] = None
