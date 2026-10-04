#!/usr/bin/env python3
"""Compare a migrated temporal reproduction without treating old results as GT."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--original', type=Path, required=True)
parser.add_argument('--reproduced', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()


def rows(path):
    values = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    result = {str(item['video_id']):item for item in values}
    if len(result) != len(values):
        raise ValueError('Duplicate video IDs')
    return result


original, reproduced = rows(args.original), rows(args.reproduced)
if original.keys() != reproduced.keys():
    raise ValueError('Video IDs differ')
fields = ('targetRatioWH', 'segments_sec', 'segments_frame')
different = sum(any(original[vid].get(key) != reproduced[vid].get(key) for key in fields) for vid in original)
report = {'date':'2026-10-04', 'balanced':{'videos':len(original),'different_temporal_rows':different},
          'note':'Reproduction equivalence, not highlight accuracy. GPU inference not rerun.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
raise SystemExit(1 if different else 0)
