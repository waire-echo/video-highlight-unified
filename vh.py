#!/usr/bin/env python3
"""Repository entry point, also usable before installing the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from video_highlight.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
