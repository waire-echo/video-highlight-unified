#!/usr/bin/env python3
"""Build the input index from video files and their original JSONL configurations."""
import argparse
import json
from pathlib import Path


def build_index(video_dir, config_dir):
    videos = {path.stem for path in video_dir.glob('*.mp4') if not path.name.startswith('._')}
    configs = {path.stem: path for path in config_dir.glob('*.jsonl') if not path.name.startswith('._')}
    if not videos or videos != set(configs):
        raise ValueError(f'Video/config IDs differ: missing configs={sorted(videos-set(configs))[:10]}, missing videos={sorted(set(configs)-videos)[:10]}')
    rows = []
    if any(not video_id.isdecimal() for video_id in videos):
        raise ValueError('Competition video IDs must be numeric filenames')
    for video_id in sorted(videos, key=lambda value: (0, int(value)) if value.isdigit() else (1, value)):
        entries = [json.loads(line) for line in configs[video_id].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if len(entries) != 1 or str(entries[0].get('video_id', video_id)) != video_id:
            raise ValueError('Expected one configuration for video ' + video_id)
        ratio = entries[0].get('targetRatioWH')
        if ratio not in ([16, 9], [9, 16]):
            raise ValueError('Unsupported target ratio for ' + video_id)
        rows.append({'video_id': video_id, 'targetRatioWH': ratio})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = build_index(args.videos, args.configs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'videos': len(rows), 'index': str(args.output)}, ensure_ascii=False))
