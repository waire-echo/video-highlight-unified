"""Compare fresh GPU inference in original slow and relay configurations."""
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'src'))
from highlight_pipeline.config import load_config, write_resolved

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--video-ids', required=True, help='Comma-separated small-sample IDs')
options = parser.parse_args()
config = load_config(options.config)
resolved = write_resolved(config)
os.environ['HIGHLIGHT_PIPELINE_CONFIG'] = str(resolved)
stage = Path(config['run_root'])
spatial = repo / 'src/highlight_pipeline/spatial'
runtime = repo / 'src/highlight_pipeline/runtime'
sys.path.insert(0, str(spatial))
sys.path.insert(0, str(runtime))
import run_stage2 as pipeline
from resource_tuning import capture_resources, choose_plan


def box_iou(left, right, ratio):
    a = [left[0], left[1], left[0] + left[2], left[1] + left[2] * ratio[1] / ratio[0]]
    b = [right[0], right[1], right[0] + right[2], right[1] + right[2] * ratio[1] / ratio[0]]
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def compare(left, right):
    assert [str(row['video_id']) for row in left] == [str(row['video_id']) for row in right]
    exact = total = 0
    ious = []
    maximum = 0
    changes = []
    for a, b in zip(left, right):
        assert a['targetRatioWH'] == b['targetRatioWH']
        assert [item['frame'] for item in a['predictions']] == [item['frame'] for item in b['predictions']]
        count = 0
        for x, y in zip(a['predictions'], b['predictions']):
            total += 1
            same = x['bboxes'] == y['bboxes']
            exact += same
            count += not same
            ious.append(box_iou(x['bboxes'], y['bboxes'], a['targetRatioWH']))
            maximum = max(maximum, *(abs(p - q) for p, q in zip(x['bboxes'], y['bboxes'])))
        if count:
            changes.append({'video_id': str(a['video_id']), 'changed_frames': count})
    return {'frames': total, 'identical_boxes': exact, 'identical_fraction': exact / max(total, 1),
            'mean_crop_iou': sum(ious) / max(len(ious), 1), 'minimum_crop_iou': min(ious) if ious else 1,
            'maximum_coordinate_delta_pixels': maximum, 'changed_videos': changes,
            'frame_ids_and_ratios_identical': True}


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = pipeline.LOG_DIR / ('fresh_relay_comparison_' + stamp)
    root.mkdir(parents=True)
    source = Path(config.get('temporal_handoff') or Path(config['temporal_run_dir']) / 'export_grounded/temporal_handoff.jsonl')
    source_rows, member, input_sha = pipeline.load_source(source)
    pipeline.ensure_work_state(input_sha, member, False)
    baseline, _, _ = pipeline.prepare_input(source_rows, None)
    by_id = {str(row['video_id']): row for row in baseline}
    ids = [value.strip() for value in options.video_ids.split(',') if value.strip()]
    if not ids or len(ids) != len(set(ids)) or any(value not in by_id for value in ids):
        raise ValueError('Comparison IDs must be unique IDs in the input index')
    subset = [by_id[video_id] for video_id in ids]
    pipeline.write_jsonl(root / 'center.jsonl', subset)
    original = pipeline.commands()
    timings = {}
    environment = os.environ.copy()
    plan = choose_plan(capture_resources())
    environment.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                       YOLO_CONFIG_DIR=str(stage / 'settings'), YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    environment.update(HIGHLIGHT_TRACKING_WORKERS=str(plan.tracking_workers),
                       HIGHLIGHT_TRACKING_THREADS=str(plan.threads_per_tracking_worker),
                       HIGHLIGHT_DYNAMIC_SCHEDULING='1', HIGHLIGHT_BATCH_PREFETCH=str(plan.batch_prefetch),
                       HIGHLIGHT_FRAME_PREFETCH=str(plan.frame_prefetch))
    for mode in ['slow', 'relay']:
        folder = root / mode
        folder.mkdir()
        start = time.monotonic()
        for index, (label, command) in enumerate(original):
            c = command.copy()
            mappings = {
                '--base-predictions': root / 'center.jsonl',
                '--output': folder / ['yolo.jsonl', 'pose.jsonl', 'final.jsonl'][index],
                '--cache': folder / ['yolo_cache.jsonl', 'pose_cache.jsonl', 'tracking_cache.jsonl'][index],
                '--report': folder / 'tracking_report.json',
                '--zip-output': folder / 'tracking.zip',
            }
            if '--input' in c:
                mappings['--input'] = folder / ('yolo.jsonl' if index == 1 else 'pose.jsonl')
            for key, value in mappings.items():
                if key in c:
                    c[c.index(key) + 1] = str(value)
            if mode == 'relay':
                if '--batch-size' in c:
                    c[c.index('--batch-size') + 1] = str(plan.yolo_batch_size if index == 0 else plan.pose_batch_size)
                    c.insert(2, str(runtime / 'stage2_prefetch_entry.py'))
                else:
                    c[2] = str(runtime / 'stage2_parallel_tracking.py')
            with (folder / (str(index) + '.log')).open('wb') as output:
                subprocess.run(c, stdout=output, stderr=subprocess.STDOUT, env=environment, check=True)
        timings[mode] = round(time.monotonic() - start, 3)
    comparisons = {
        step: compare(pipeline.load_jsonl(root / 'slow' / (step + '.jsonl')),
                      pipeline.load_jsonl(root / 'relay' / (step + '.jsonl')))
        for step in ['yolo', 'pose', 'final']
    }
    final = comparisons['final']
    report = {'status': 'COMPLETE', 'fresh_model_inference': True, 'cache_reused': False,
              'video_ids': ids, 'input_frames': sum(len(row['predictions']) for row in subset),
              'timings_seconds': timings, 'comparison': comparisons,
              'relay_plan': plan.to_dict(),
              'conclusion': 'identical_sampled_outputs' if final['identical_fraction'] == 1 else 'sampled_outputs_differ',
              'full_dataset_bitwise_equivalence_proven': False,
              'note': 'Engineering equivalence check, not ground truth accuracy or competition score.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (pipeline.LOG_DIR / 'fresh_relay_comparison_latest.json').write_text(json.dumps({'folder': str(root), 'report': report}, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
