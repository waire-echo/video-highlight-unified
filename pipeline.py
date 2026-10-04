#!/usr/bin/env python3
"""Run the project directly without installing its lightweight CLI package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from video_highlight.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
