"""Unified CLI for temporal inference, spatial inference and project tools."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from .config import load_config, write_resolved

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]


def run(command, config, *, dry_run=False, extra_env=None):
    command = [str(item) for item in command]
    print(json.dumps({'command': command}, ensure_ascii=False), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true',
                       YOLO_CONFIG_DIR=str(Path(config['run_root']) / 'settings'))
    environment.update(extra_env or {})
    subprocess.run(command, env=environment, check=True)


def handoff_path(config, supplied=None):
    return Path(supplied or config.get('temporal_handoff') or
                Path(config['temporal_run_dir']) / 'export_grounded/temporal_handoff.jsonl')


def spatial_command(config, resolved, supplied=None, *, profile=None, smoke=None, check=False, prepare=False):
    command = [config['python_executable'], '-B', PACKAGE / 'runtime/adaptive_spatial.py',
               handoff_path(config, supplied), '--profile', profile or config['profile']]
    if smoke:
        command.extend(['--smoke-videos', smoke])
    if check:
        command.append('--check-only')
    if prepare:
        command.append('--prepare-only')
    return command, {'HIGHLIGHT_PIPELINE_CONFIG': str(resolved)}


def server_command(config):
    model = Path(config['temporal_model'])
    executable = Path(config['temporal_python']).parent / ('vllm.exe' if os.name == 'nt' else 'vllm')
    media_root = config.get('media_root') or os.path.commonpath([config['video_dir'], config['temporal_run_dir']])
    if Path(media_root) == Path(Path(media_root).anchor):
        raise ValueError('Set media_root to a directory containing videos and the temporal run')
    for path in (config['video_dir'], config['temporal_run_dir']):
        if not Path(path).is_relative_to(Path(media_root)):
            raise ValueError('media_root does not contain ' + path)
    return [str(executable), 'serve', str(model), '--served-model-name', 'Video-ORA-4B',
            '--host', '127.0.0.1', '--port', str(config['model_port']), '--dtype', 'bfloat16',
            '--tensor-parallel-size', '1', '--max-model-len', '16384', '--max-num-seqs', '1',
            '--max-num-batched-tokens', '8192', '--enforce-eager', '--gpu-memory-utilization', '0.85',
            '--mm-processor-cache-gb', '0', '--structured-outputs-config',
            '{"backend":"xgrammar","disable_any_whitespace":true}', '--allowed-local-media-path', media_root,
            '--media-io-kwargs', '{"video":{"num_frames":-1}}', '--mm-processor-kwargs',
            '{"max_frames":128,"size":{"shortest_edge":65536,"longest_edge":25690112}}',
            '--limit-mm-per-prompt', '{"image":0,"video":1}']


def server_ready(config):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{config['model_port']}/v1/models", timeout=3) as response:
            return any(item['id'] == 'Video-ORA-4B' for item in json.load(response)['data'])
    except (OSError, ValueError, KeyError):
        return False


def start_server(config, dry_run=False):
    command = server_command(config)
    if dry_run:
        print(json.dumps({'server_command': command}, ensure_ascii=False))
        return False
    if server_ready(config):
        print('Existing local Video-ORA service is ready', flush=True)
        return False
    root = Path(config['run_root'])
    log = root / 'logs/model_server.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       VLLM_VIDEO_LOADER_BACKEND='opencv')
    with log.open('ab', buffering=0) as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=os.name != 'nt')
    (root / 'server.json').write_text(json.dumps({'pid': process.pid, 'command': command,
                                                  'model': config['temporal_model']}, indent=2))
    deadline = time.monotonic() + 900
    while not server_ready(config):
        if process.poll() is not None:
            raise RuntimeError('Model service exited; see ' + str(log))
        if time.monotonic() > deadline:
            raise TimeoutError('Model service startup timed out; inspect ' + str(log))
        time.sleep(2)
    print('Local model service ready', flush=True)
    return True


def stop_server(config):
    path = Path(config['run_root']) / 'server.json'
    if not path.is_file():
        print('No model service owned by this run')
        return
    state = json.loads(path.read_text())
    process = Path('/proc') / str(state['pid'])
    if os.name != 'nt' and process.exists():
        command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'vllm' not in command or state['model'] not in command:
            raise RuntimeError('Recorded server PID belongs to another process; refusing to stop it')
        os.kill(state['pid'], signal.SIGTERM)
        for _ in range(60):
            if not process.exists() or 'State:\tZ' in (process / 'status').read_text():
                break
            time.sleep(1)
        else:
            raise RuntimeError('Model service did not stop normally')
    elif os.name == 'nt':
        raise RuntimeError('Automatic model-service stopping is supported on Linux')
    path.rename(path.with_name('server.stopped.json'))
    print('Owned model service stopped', flush=True)


def temporal(config, *, dry_run=False, smoke=0, plan=False):
    folder = PACKAGE / 'temporal'
    output = Path(config['temporal_run_dir'])
    base = [config['temporal_python'], '-B']
    command = [*base, folder / 'run_temporal.py', '--index', config['test_index'], '--videos', config['video_dir'],
               '--output', output, '--model-revision', config['model_revision'],
               '--base-url', f"http://127.0.0.1:{config['model_port']}/v1"]
    if smoke:
        command.extend(['--smoke', str(smoke)])
    if plan:
        command.append('--plan')
    run(command, config, dry_run=dry_run)
    if plan or smoke:
        return
    run([*base, folder / 'repair_grounding.py', '--run', output, '--videos', config['video_dir'],
         '--base-url', f"http://127.0.0.1:{config['model_port']}/v1"], config, dry_run=dry_run)
    run([*base, folder / 'summarize_run.py', '--run', output], config, dry_run=dry_run)
    for variant in ('grounded', 'recall'):
        run([*base, folder / 'export_submission.py', '--index', config['test_index'], '--videos', config['video_dir'],
             '--run', output, '--variant', variant, '--temporal-only'], config, dry_run=dry_run)
    run([*base, folder / 'validate_export.py', '--run', output, '--index', config['test_index']], config, dry_run=dry_run)
    if not dry_run:
        (output / 'temporal.complete').write_text(datetime.now(timezone.utc).isoformat() + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Video highlight temporal + spatial pipeline')
    parser.add_argument('--config', type=Path, default=REPO / 'configs/pipeline.local.json')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without inference')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('serve', help='Start the offline local Video-ORA service')
    sub.add_parser('stop-server', help='Stop only the service started by this run')
    for action in ('temporal', 'all'):
        command = sub.add_parser(action)
        command.add_argument('--smoke', type=int, default=0, help='Temporal smoke count; skips export and spatial stage')
        command.add_argument('--plan', action='store_true', help='Metadata only; no model inference')
    for action in ('spatial', 'check', 'prepare'):
        command = sub.add_parser(action)
        command.add_argument('--handoff', type=Path)
        command.add_argument('--profile', choices=('auto', 'compat'))
        command.add_argument('--smoke-videos', help='Comma-separated IDs; no official ZIP is produced')
    sub.add_parser('status', help='Read the run status and adaptive plan')
    validate = sub.add_parser('validate')
    validate.add_argument('--prediction', type=Path)
    preview = sub.add_parser('preview')
    preview.add_argument('--video-id', default=None)
    preview.add_argument('--random', type=int, default=3)
    preview.add_argument('--every', type=int, default=3)
    verify = sub.add_parser('verify-relay', help='Fresh slow/relay small-sample comparison')
    verify.add_argument('--video-ids', required=True, help='Small set of representative IDs')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        resolved = write_resolved(config) if not args.dry_run else args.config.resolve()
        if args.action == 'serve':
            start_server(config, args.dry_run)
        elif args.action == 'stop-server':
            if not args.dry_run:
                stop_server(config)
        elif args.action in ('all', 'temporal'):
            owned = False
            try:
                if not args.plan:
                    owned = start_server(config, args.dry_run)
                temporal(config, dry_run=args.dry_run, smoke=args.smoke, plan=args.plan)
            finally:
                if owned:
                    stop_server(config)
            if args.action == 'all' and not args.plan and not args.smoke:
                command, env = spatial_command(config, resolved)
                run(command, config, dry_run=args.dry_run, extra_env=env)
        elif args.action in ('spatial', 'check', 'prepare'):
            command, env = spatial_command(config, resolved, args.handoff, profile=args.profile,
                                           smoke=args.smoke_videos, check=args.action == 'check', prepare=args.action == 'prepare')
            run(command, config, dry_run=args.dry_run, extra_env=env)
        elif args.action == 'status':
            for name in ('status.json', 'adaptive_plan.json', 'smoke_report.json'):
                path = Path(config['log_dir']) / name
                if path.exists():
                    print(name + '\n' + path.read_text(encoding='utf-8'))
            work = Path(config['work_dir'])
            counts = {}
            for name in ('02_yolo_cache.jsonl', '03_pose_cache.jsonl', '04_botsort_cache.jsonl'):
                path = work / name
                ids = set()
                if path.exists():
                    for line in path.read_text(encoding='utf-8').splitlines():
                        try:
                            ids.add(str(json.loads(line)['video_id']))
                        except (ValueError, KeyError):
                            pass
                counts[name] = len(ids)
            parallel = sorted((work / 'parallel_tracking').glob('*/status.json'),
                              key=lambda path: path.stat().st_mtime, reverse=True)
            if parallel:
                print('parallel_tracking\n' + parallel[0].read_text(encoding='utf-8'))
                tracking_ids = set()
                for cache in parallel[0].parent.glob('part_*/cache.jsonl'):
                    for line in cache.read_text(encoding='utf-8').splitlines():
                        try:
                            tracking_ids.add(str(json.loads(line)['video_id']))
                        except (ValueError, KeyError):
                            pass
                counts['tracking_shard_cached_nonempty_videos'] = len(tracking_ids)
            print(json.dumps({'cached_videos': counts}, ensure_ascii=False))
        elif args.action == 'validate':
            prediction = args.prediction or Path(config['output_dir']) / 'predictions_spatial.jsonl'
            command = [config['python_executable'], '-B', PACKAGE / 'spatial/validate_submission.py', prediction,
                       '--video-dir', config['video_dir'], '--index', config['test_index'],
                       '--report', Path(config['log_dir']) / 'independent_validation.json']
            baseline = Path(config['work_dir']) / '01_center.jsonl'
            if baseline.is_file():
                command.extend(['--baseline', baseline])
            run(command, config, dry_run=args.dry_run)
        elif args.action == 'preview':
            command = [config['python_executable'], REPO / 'tools/preview_submission.py',
                       '--pred', Path(config['output_dir']) / 'predictions_spatial.jsonl', '--video-dir', config['video_dir'],
                       '--out-dir', Path(config['run_root']) / 'previews', '--mode', 'side-by-side', '--every', str(args.every)]
            command.extend(['--video-id', args.video_id] if args.video_id else ['--random', str(args.random), '--seed', '20261002'])
            run(command, config, dry_run=args.dry_run)
        elif args.action == 'verify-relay':
            run([config['python_executable'], '-B', REPO / 'tools/verify_relay_equivalence.py',
                 '--config', resolved, '--video-ids', args.video_ids], config, dry_run=args.dry_run)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
