"""Configuration constants and defaults for Photo Organizer."""

from __future__ import annotations

import re

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".rw2",
    ".pef",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
}

# Companion sidecars that should move atomically with main images (e.g. Live Photos, iOS edits, Lightroom)
COMPANION_EXTENSIONS = {
    ".mov",
    ".mp4",
    ".aae",
    ".xmp",
}

ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

EXIF_DATE_TAGS = (
    36867,  # DateTimeOriginal
    36868,  # DateTimeDigitized
    306,    # DateTime
)

FILENAME_DATE_PATTERNS = (
    re.compile(
        r"(?P<year>19\d\d|20\d\d)"
        r"[-_\.]?"
        r"(?P<month>0[1-9]|1[0-2])"
        r"[-_\.]?"
        r"(?P<day>0[1-9]|[12]\d|3[01])"
        r"(?:[T\s_-]?"
        r"(?P<hour>[01]\d|2[0-3])"
        r"[:_\.-]?"
        r"(?P<minute>[0-5]\d)"
        r"(?::?|\.-|_)?"
        r"(?P<second>[0-5]\d)?)?"
    ),
    re.compile(
        r"(?P<day>0[1-9]|[12]\d|3[01])"
        r"[-_\.]"
        r"(?P<month>0[1-9]|1[0-2])"
        r"[-_\.]"
        r"(?P<year>19\d\d|20\d\d)"
    ),
)

DEFAULT_VISUAL_THRESHOLD = 5
DEFAULT_MAX_SIZE_INCREASE = 20.0
DEFAULT_TO_DELETE_FOLDER = "to-be-deleted"
DEFAULT_NON_RELEVANT_FOLDER = "non-relevant"
DEFAULT_MAYBE_NON_RELEVANT_FOLDER = "maybe-non-relevant"
DEFAULT_VIDEO_FOLDER = "video"

JOURNAL_DB_NAME = ".photo_organizer_journal.db"
CACHE_DB_NAME = ".photo_cache.sqlite"
