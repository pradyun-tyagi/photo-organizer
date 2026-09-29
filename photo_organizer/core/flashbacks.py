"""'On This Day' Flashbacks and Anniversary Memories generator."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional, Sequence

from photo_organizer.models import PhotoInfo


def get_on_this_day_memories(
    photos: Sequence[PhotoInfo], reference_date: Optional[date] = None
) -> Dict[int, List[PhotoInfo]]:
    """Groups photos taken on the same month and day in prior years.
    Returns mapping: years_ago -> [PhotoInfo, ...] sorted chronologically.
    """
    if reference_date is None:
        reference_date = date.today()

    target_month = reference_date.month
    target_day = reference_date.day
    current_year = reference_date.year

    memories_by_year: Dict[int, List[PhotoInfo]] = {}

    for photo in photos:
        p_dt = photo.taken_at
        if p_dt.month == target_month and p_dt.day == target_day:
            years_ago = current_year - p_dt.year
            if years_ago > 0:
                memories_by_year.setdefault(years_ago, []).append(photo)

    for years_ago in memories_by_year:
        memories_by_year[years_ago].sort(key=lambda p: p.taken_at)

    return dict(sorted(memories_by_year.items()))


def get_flashback_cards(
    photos: Sequence[PhotoInfo], reference_date: Optional[date] = None
) -> List[Dict]:
    """Formats flashbacks into structured cards for Web UI display."""
    memories = get_on_this_day_memories(photos, reference_date)
    cards = []

    for years_ago, year_photos in memories.items():
        year_val = (reference_date or date.today()).year - years_ago
        cards.append({
            "years_ago": years_ago,
            "year": year_val,
            "title": f"{years_ago} {'Year' if years_ago == 1 else 'Years'} Ago ({year_val})",
            "photo_count": len(year_photos),
            "photos": [
                {
                    "path": str(p.path),
                    "filename": p.path.name,
                    "taken_at": p.taken_at.strftime("%I:%M %p"),
                    "city": p.city,
                    "country": p.country,
                }
                for p in year_photos
            ],
            "cover_photo": str(year_photos[0].path) if year_photos else None,
        })

    return cards
