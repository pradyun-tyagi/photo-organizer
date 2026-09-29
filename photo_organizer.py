#!/usr/bin/env python3
"""Organize photos by capture date, move duplicates/text forwards aside, and enhance.

The default run is active: files are moved and eligible images are enhanced in
place. Use --dry-run to preview actions without changing files, or --ui to launch
the interactive Web Studio.
"""

from __future__ import annotations

import sys
from photo_organizer import *
from photo_organizer.ui.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
