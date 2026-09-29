"""Persistent metadata and hash cache for instant re-scans with WAL mode and bulk operations."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from photo_organizer.config import CACHE_DB_NAME
from photo_organizer.models import PhotoInfo


class PhotoCache:
    def __init__(self, db_path: Optional[Path] = None, root_dir: Optional[Path] = None) -> None:
        if db_path is not None:
            self.db_path = db_path
        elif root_dir is not None:
            self.db_path = root_dir / CACHE_DB_NAME
        else:
            self.db_path = Path.cwd() / CACHE_DB_NAME

        self._local = threading.local()
        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._create_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_cache (
                    file_path TEXT PRIMARY KEY,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    sha256 TEXT,
                    taken_at TEXT NOT NULL,
                    date_source TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    visual_hash TEXT,
                    phash TEXT,
                    format_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    city TEXT,
                    country TEXT
                )
                """
            )
            # Safe schema migrations for existing databases
            for col, col_type in (
                ("latitude", "REAL"),
                ("longitude", "REAL"),
                ("city", "TEXT"),
                ("country", "TEXT"),
            ):
                try:
                    conn.execute(f"ALTER TABLE file_cache ADD COLUMN {col} {col_type};")
                except sqlite3.OperationalError:
                    pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_lookup ON file_cache(file_path, file_size, mtime_ns);")
            conn.execute("CREATE TABLE IF NOT EXISTS studio_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            conn.commit()
        finally:
            conn.close()

    def _row_to_photo_info(self, row: sqlite3.Row, path: Path) -> PhotoInfo:
        v_hash = int(row["visual_hash"], 16) if row["visual_hash"] else None
        p_hash = int(row["phash"], 16) if row["phash"] else None
        lat = row["latitude"] if "latitude" in row.keys() else None
        lon = row["longitude"] if "longitude" in row.keys() else None
        city = row["city"] if "city" in row.keys() else None
        country = row["country"] if "country" in row.keys() else None

        return PhotoInfo(
            path=path,
            file_size=row["file_size"],
            sha256=row["sha256"],
            taken_at=datetime.fromisoformat(row["taken_at"]),
            date_source=row["date_source"],
            width=row["width"],
            height=row["height"],
            visual_hash=v_hash,
            phash=p_hash,
            format_name=row["format_name"],
            latitude=lat,
            longitude=lon,
            city=city,
            country=country,
        )

    def get(self, path: Path, file_size: int, mtime_ns: int) -> Optional[PhotoInfo]:
        resolved = str(path.resolve())
        conn = self._create_connection()
        try:
            cursor = conn.execute(
                """
                SELECT file_size, sha256, taken_at, date_source, width, height, visual_hash, phash, format_name,
                       latitude, longitude, city, country
                FROM file_cache
                WHERE file_path = ? AND file_size = ? AND mtime_ns = ?
                """,
                (resolved, file_size, mtime_ns),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_photo_info(row, path)
        finally:
            conn.close()

    def load_cache_bulk(self, paths: Sequence[Path]) -> Dict[str, Tuple[PhotoInfo, int, int]]:
        """Preloads cached photo metadata into memory for fast O(1) lookup during scanning."""
        if not paths:
            return {}

        cache_map: Dict[str, Tuple[PhotoInfo, int, int]] = {}
        conn = self._create_connection()
        chunk_size = 900
        resolved_dict = {str(p.resolve()): p for p in paths}
        resolved_keys = list(resolved_dict.keys())

        try:
            for i in range(0, len(resolved_keys), chunk_size):
                chunk = resolved_keys[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"""
                    SELECT file_path, file_size, mtime_ns, sha256, taken_at, date_source,
                           width, height, visual_hash, phash, format_name,
                           latitude, longitude, city, country
                    FROM file_cache
                    WHERE file_path IN ({placeholders})
                    """,
                    chunk,
                )
                for row in cursor.fetchall():
                    fp = row["file_path"]
                    original_path = resolved_dict.get(fp, Path(fp))
                    info = self._row_to_photo_info(row, original_path)
                    cache_map[fp] = (info, row["file_size"], row["mtime_ns"])
        finally:
            conn.close()

        return cache_map

    def set(self, photo: PhotoInfo, mtime_ns: int) -> None:
        resolved = str(photo.path.resolve())
        v_hash_hex = hex(photo.visual_hash) if photo.visual_hash is not None else None
        p_hash_hex = hex(photo.phash) if photo.phash is not None else None

        conn = self._create_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_cache (
                    file_path, file_size, mtime_ns, sha256, taken_at, date_source,
                    width, height, visual_hash, phash, format_name,
                    latitude, longitude, city, country
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved,
                    photo.file_size,
                    mtime_ns,
                    photo.sha256,
                    photo.taken_at.isoformat(),
                    photo.date_source,
                    photo.width,
                    photo.height,
                    v_hash_hex,
                    p_hash_hex,
                    photo.format_name,
                    photo.latitude,
                    photo.longitude,
                    photo.city,
                    photo.country,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def set_bulk(self, items: Sequence[Tuple[PhotoInfo, int]]) -> None:
        """Saves a batch of PhotoInfo items and mtime_ns in a single fast transaction."""
        if not items:
            return

        rows = []
        for photo, mtime_ns in items:
            resolved = str(photo.path.resolve())
            v_hash_hex = hex(photo.visual_hash) if photo.visual_hash is not None else None
            p_hash_hex = hex(photo.phash) if photo.phash is not None else None
            rows.append(
                (
                    resolved,
                    photo.file_size,
                    mtime_ns,
                    photo.sha256,
                    photo.taken_at.isoformat(),
                    photo.date_source,
                    photo.width,
                    photo.height,
                    v_hash_hex,
                    p_hash_hex,
                    photo.format_name,
                    photo.latitude,
                    photo.longitude,
                    photo.city,
                    photo.country,
                )
            )

        conn = self._create_connection()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO file_cache (
                    file_path, file_size, mtime_ns, sha256, taken_at, date_source,
                    width, height, visual_hash, phash, format_name,
                    latitude, longitude, city, country
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def set_kv(self, key: str, value: str) -> None:
        """Stores a key-value pair in the persistent cache."""
        conn = self._create_connection()
        try:
            conn.execute("INSERT OR REPLACE INTO studio_kv (key, value) VALUES (?, ?);", (key, value))
            conn.commit()
        finally:
            conn.close()

    def get_kv(self, key: str) -> Optional[str]:
        """Retrieves a value by key from the persistent cache."""
        conn = self._create_connection()
        try:
            cursor = conn.execute("SELECT value FROM studio_kv WHERE key = ?;", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None
        finally:
            conn.close()

    def save_studio_state(self, state_dict: Dict) -> None:
        """Persists the studio state (People albums, Flashbacks, Map pins, Summary) as JSON."""
        clean_state = {}
        for k, v in state_dict.items():
            if k in ("photos", "all_photos", "duplicate_clusters_raw"):
                continue
            clean_state[k] = v
        self.set_kv("studio_state", json.dumps(clean_state, default=str))

    def load_studio_state(self) -> Optional[Dict]:
        """Loads previously saved studio state for instant viewing without re-scanning."""
        raw = self.get_kv("studio_state")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def close(self) -> None:
        pass

