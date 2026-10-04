"""Run Stage 2 with a hardware-aware plan, ordered prefetch and safe sharding."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

RUNTIME = Path(__file__).resolve().parent
SPATIAL = RUNTIME.parent / 'spatial'
sys.path.insert(0, str(SPATIAL))
import run_stage2 as pipeline

from resource_tuning import ResourceMonitor, capture_resources, choose_plan


def selected_profile() -> str:
    if '--profile' not in sys.argv:
        return os.environ.get('HIGHLIGHT_PROFILE', 'auto')
    index = sys.argv.index('--profile')
    try:
        value = sys.argv[index + 1]
    except IndexError as error:
        raise ValueError('--profile requires auto or compat') from error
    del sys.argv[index:index + 2]
    return value


profile = selected_profile()
snapshot = capture_resources()
plan = choose_plan(snapshot, profile)
os.environ.update({
    'HIGHLIGHT_BATCH_PREFETCH': str(plan.batch_prefetch),
    'HIGHLIGHT_FRAME_PREFETCH': str(plan.frame_prefetch),
    'HIGHLIGHT_TRACKING_WORKERS': str(plan.tracking_workers),
    'HIGHLIGHT_TRACKING_THREADS': str(plan.threads_per_tracking_worker),
    'HIGHLIGHT_DYNAMIC_SCHEDULING': '1' if profile == 'auto' else '0',
    'OMP_NUM_THREADS': str(max(1, min(12, snapshot.cpu_count))),
    'MKL_NUM_THREADS': str(max(1, min(12, snapshot.cpu_count))),
})

original_commands = pipeline.commands


def tuned_commands():
    commands = original_commands()
    for stage_name, command in commands:
        if '--batch-size' in command:
            index = command.index('--batch-size') + 1
            command[index] = str(plan.yolo_batch_size if stage_name == 'YOLO 主体定位' else plan.pose_batch_size)
            if profile == 'auto':
                command.insert(2, str(RUNTIME / 'stage2_prefetch_entry.py'))
        elif stage_name == 'BoT-SORT 主体锁定' and plan.tracking_workers > 1:
            command[2] = str(RUNTIME / 'stage2_parallel_tracking.py')
    return commands


pipeline.commands = tuned_commands
pipeline.LOG_DIR.mkdir(parents=True, exist_ok=True)
(pipeline.LOG_DIR / 'adaptive_plan.json').write_text(json.dumps({
    'resources': snapshot.to_dict(), 'plan': plan.to_dict(),
    'semantic_parameters_changed': bool(pipeline.CONFIG.get('spatial_parameters')),
    'frame_order_preserved': True,
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
pipeline.start_live_log()
try:
    with ResourceMonitor(pipeline.LOG_DIR / 'resource_usage.jsonl'):
        pipeline.main()
except Exception as error:
    pipeline.set_status('FAILED', '异常退出', f'{type(error).__name__}: {error}')
    (pipeline.LOG_DIR / 'adaptive_failure.json').write_text(json.dumps({
        'error_type': type(error).__name__, 'error': str(error), 'plan': plan.to_dict(),
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    traceback.print_exc()
    raise SystemExit(1)
