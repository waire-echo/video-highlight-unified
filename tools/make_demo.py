#!/usr/bin/env python3
"""Generate two synthetic videos for the CPU baseline, without competition data."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1] / 'datasets/local/demo/videos'
root.mkdir(parents=True, exist_ok=True)
for name in ('wide', 'portrait'):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=160x90:rate=10', '-t', '2', '-c:v', 'mpeg4', str(root / (name + '.mp4'))], check=True)
print(root)
