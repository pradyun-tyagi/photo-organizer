"""CLI entrypoint when running `python -m photo_organizer`."""

from __future__ import annotations

import sys
from photo_organizer.ui.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
