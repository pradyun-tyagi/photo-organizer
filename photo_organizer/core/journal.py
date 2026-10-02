"""Transaction ledger and one-click undo mechanism using SQLite."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from photo_organizer.config import (
    JOURNAL_DB_NAME,
    expand_personal_path,
    serialize_path_portable,
)


class Journal:
    def __init__(self, db_path: Optional[Path] = None, root_dir: Optional[Path] = None) -> None:
        if db_path is not None:
            self.db_path = db_path
        elif root_dir is not None:
            self.db_path = root_dir / JOURNAL_DB_NAME
        else:
            self.db_path = Path.cwd() / JOURNAL_DB_NAME

        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    root_dir TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    summary_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    dest_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def start_run(self, root_dir: Path, dry_run: bool) -> str:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute(
                """
                INSERT INTO runs (run_id, started_at, root_dir, dry_run)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, datetime.now().isoformat(), serialize_path_portable(root_dir), int(dry_run)),
            )
            conn.commit()
        finally:
            conn.close()
        return run_id

    def record_action(
        self,
        run_id: str,
        action_type: str,
        source_path: str,
        dest_path: str,
        details: Optional[str] = None,
    ) -> None:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute(
                """
                INSERT INTO actions (run_id, action_type, source_path, dest_path, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    action_type,
                    serialize_path_portable(source_path),
                    serialize_path_portable(dest_path),
                    datetime.now().isoformat(),
                    details,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def complete_run(self, run_id: str, summary: Dict[str, Any]) -> None:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute(
                """
                UPDATE runs
                SET completed_at = ?, summary_json = ?
                WHERE run_id = ?
                """,
                (datetime.now().isoformat(), json.dumps(summary), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def list_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT run_id, started_at, completed_at, root_dir, dry_run, summary_json
                FROM runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def undo_run(self, run_id: Optional[str] = None) -> Tuple[int, int, List[str]]:
        """Reverses all file moves in reverse chronological order for the given run."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            if run_id is None:
                cursor = conn.execute(
                    """
                    SELECT run_id FROM runs
                    WHERE dry_run = 0 AND completed_at IS NOT NULL
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if not row:
                    return 0, 0, ["No completed non-dry-run found to undo."]
                run_id = row["run_id"]

            cursor = conn.execute(
                """
                SELECT id, action_type, source_path, dest_path
                FROM actions
                WHERE run_id = ? AND action_type = 'MOVE'
                ORDER BY id DESC
                """,
                (run_id,),
            )
            moves = cursor.fetchall()
        finally:
            conn.close()

        restored = 0
        failed = 0
        errors: List[str] = []

        for move in moves:
            current_loc = expand_personal_path(move["dest_path"])
            orig_loc = expand_personal_path(move["source_path"])

            if not current_loc.exists():
                failed += 1
                errors.append(f"File missing at destination, cannot restore: {current_loc}")
                continue

            try:
                orig_loc.parent.mkdir(parents=True, exist_ok=True)
                current_loc.rename(orig_loc)
                restored += 1
            except OSError as exc:
                failed += 1
                errors.append(f"Failed to restore {current_loc} -> {orig_loc}: {exc}")

        # Mark run as undone
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute(
                "UPDATE runs SET summary_json = json_set(COALESCE(summary_json, '{}'), '$.undone', 1) WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
        finally:
            conn.close()

        return restored, failed, errors
