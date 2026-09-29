"""Command-line interface and argument parser for Photo Organizer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from photo_organizer.config import (
    DEFAULT_MAX_SIZE_INCREASE,
    DEFAULT_MAYBE_NON_RELEVANT_FOLDER,
    DEFAULT_NON_RELEVANT_FOLDER,
    DEFAULT_TO_DELETE_FOLDER,
    DEFAULT_VIDEO_FOLDER,
    DEFAULT_VISUAL_THRESHOLD,
)
from photo_organizer.core.journal import Journal
from photo_organizer.models import ActionSummary
from photo_organizer.organizer import organize


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count))


def print_summary(summary: ActionSummary) -> None:
    print("")
    print("Summary")
    print(f"  scanned: {summary.scanned}")
    print(f"  duplicates moved to to-be-deleted: {summary.duplicates_moved}")
    if summary.burst_duplicates_moved > 0:
        print(f"  burst duplicates moved to to-be-deleted: {summary.burst_duplicates_moved}")
    print(f"  non-relevant moved: {summary.non_relevant_moved}")
    print(f"  maybe non-relevant moved: {summary.maybe_non_relevant_moved}")
    print(f"  organized moves: {summary.organized_moved}")
    print(f"  already organized: {summary.already_organized}")
    print(f"  exif metadata backfilled: {summary.exif_backfilled}")
    print(f"  enhanced or enhancement dry-run candidates: {summary.enhanced}")
    print(f"  enhancement skipped: {summary.enhancement_skipped}")
    print(f"  empty folders removed: {summary.empty_dirs_removed}")
    if summary.companions_moved > 0:
        print(f"  companion files (Live Photos/XMP) moved: {summary.companions_moved}")
    print(f"  warnings: {summary.warnings}")


def pillow_required(skip_enhance: bool, skip_non_relevant: bool) -> bool:
    return not skip_enhance or not skip_non_relevant


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Organize pictures into YYYY/YYYY-MM folders, move duplicates into "
            "to-be-deleted, screen text-heavy forwards into review folders, "
            "and enhance kept pictures with a controlled size increase."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        type=Path,
        help="Folder containing pictures (optional when using --undo or --ui).",
    )
    parser.add_argument(
        "--ui",
        "--serve",
        action="store_true",
        help="Launch the interactive local Web UI studio in your browser.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind Web UI to. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number to run Web UI on. Default: 8080",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="Undo the last completed non-dry-run organization and restore files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview moves and enhancement attempts without changing files.",
    )
    parser.add_argument(
        "--skip-enhance",
        action="store_true",
        help="Only organize and move duplicates; do not rewrite kept pictures.",
    )
    parser.add_argument(
        "--max-size-increase",
        type=float,
        default=DEFAULT_MAX_SIZE_INCREASE,
        help="Maximum enhancement size increase percentage. Default: 20.",
    )
    parser.add_argument(
        "--enhancement-mode",
        choices=("fast", "balanced", "best"),
        default="balanced",
        help=(
            "Enhancement encode strategy. fast is quickest, best tries more "
            "quality candidates, balanced is the default."
        ),
    )
    parser.add_argument(
        "--skip-non-relevant",
        action="store_true",
        help="Do not screen or move text-forward/text-only images.",
    )
    parser.add_argument(
        "--skip-exif-backfill",
        action="store_true",
        help="Skip writing missing EXIF DateTimeOriginal metadata to image headers.",
    )
    parser.add_argument(
        "--skip-burst-curation",
        action="store_true",
        help="Do not auto-curate burst shots (by default, non-golden burst frames are moved to to-be-deleted).",
    )
    parser.add_argument(
        "--visual-threshold",
        type=int,
        default=DEFAULT_VISUAL_THRESHOLD,
        help=(
            "Perceptual duplicate sensitivity from 0 to 64. Lower is stricter; "
            "default is 5."
        ),
    )
    parser.add_argument(
        "--to-delete-folder",
        default=DEFAULT_TO_DELETE_FOLDER,
        help='Folder name for moved duplicates. Default: "to-be-deleted".',
    )
    parser.add_argument(
        "--non-relevant-folder",
        default=DEFAULT_NON_RELEVANT_FOLDER,
        help='Folder name for high-confidence text-only images. Default: "non-relevant".',
    )
    parser.add_argument(
        "--maybe-non-relevant-folder",
        default=DEFAULT_MAYBE_NON_RELEVANT_FOLDER,
        help=(
            "Folder name for borderline text-heavy images that need review. "
            'Default: "maybe-non-relevant".'
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help=(
            "Number of worker threads for scanning, screening, and enhancement. "
            f"Default: {default_worker_count()}."
        ),
    )
    parser.add_argument(
        "--video-folder",
        default=DEFAULT_VIDEO_FOLDER,
        help='Folder name for organized videos under the root folder. Default: "video".',
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        default=True,
        help="Organize video files into the video folder hierarchy. Default: True.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_false",
        dest="include_videos",
        help="Skip organizing video files.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the persistent metadata and hash cache.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # 1. Handle Web UI launch
    if args.ui:
        import uvicorn
        from photo_organizer.ui.web.server import create_app

        root = args.folder.expanduser().resolve() if args.folder else None
        app = create_app(default_root=root)
        print(f"Starting Photo Organizer Web Studio at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    # 2. Handle Undo
    if args.undo:
        root = args.folder.expanduser().resolve() if args.folder else Path.cwd().resolve()
        journal = Journal(root_dir=root)
        restored, failed, errors = journal.undo_run()
        print(f"Undo completed: {restored} files restored, {failed} failed.")
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        return 0 if failed == 0 else 1

    # 3. Standard CLI run
    if not args.folder:
        print("ERROR: folder is required. Run with --help for usage.", file=sys.stderr)
        return 2

    root = args.folder.expanduser()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: folder does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    if not 0 <= args.visual_threshold <= 64:
        print("ERROR: --visual-threshold must be between 0 and 64", file=sys.stderr)
        return 2

    if args.max_size_increase < 0:
        print("ERROR: --max-size-increase must be 0 or greater", file=sys.stderr)
        return 2

    if args.workers < 1:
        print("ERROR: --workers must be 1 or greater", file=sys.stderr)
        return 2

    try:
        from PIL import Image
    except ImportError:
        Image = None

    if pillow_required(args.skip_enhance, args.skip_non_relevant) and Image is None:
        print(
            "ERROR: Pillow is required for enhancement and non-relevant screening.\n"
            "Install it with: python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    summary = organize(
        root=root,
        dry_run=args.dry_run,
        visual_threshold=args.visual_threshold,
        to_delete_folder_name=args.to_delete_folder,
        skip_enhance=args.skip_enhance,
        non_relevant_folder_name=args.non_relevant_folder,
        maybe_non_relevant_folder_name=args.maybe_non_relevant_folder,
        video_folder_name=args.video_folder,
        skip_non_relevant=args.skip_non_relevant,
        skip_exif_backfill=args.skip_exif_backfill,
        auto_curate_bursts=not args.skip_burst_curation,
        include_videos=args.include_videos,
        max_size_increase_percent=args.max_size_increase,
        worker_count=args.workers,
        enhancement_mode=args.enhancement_mode,
        enable_cache=not args.no_cache,
    )
    print_summary(summary)
    return 0 if summary.warnings == 0 else 1
