"""Track independent videos concurrently, then restore the original row order."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

runtime = Path(__file__).resolve().parent
spatial = runtime.parent / 'spatial'
sys.path.insert(0, str(spatial))
tracking = importlib.import_module('generate_botsort_identity_lock')
from resource_tuning import capture_resources, concurrency_limit


def replace_argument(command, name, value):
    command[command.index(name) + 1] = str(value)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    original_args = sys.argv[1:]
    args = tracking.parse_args()
    rows = tracking.load_jsonl(args.input)
    if args.only_video or args.max_videos:
        from stage2_prefetch import install
        install(tracking)
        tracking.generate(args)
        return
    ids = [str(row['video_id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate tracking input IDs')
    requested_workers = max(1, int(os.environ.get('HIGHLIGHT_TRACKING_WORKERS', '1')))
    workers = min(requested_workers, len(rows))
    dynamic = os.environ.get('HIGHLIGHT_DYNAMIC_SCHEDULING') == '1'
    shard_count = min(len(rows), workers * 2 if dynamic else workers)
    shards = [[] for _ in range(shard_count)]
    weights = [0] * shard_count
    jobs = []
    for row in rows:
        hint = tracking.load_hint(args.hints_dir / (str(row['video_id']) + '.json'))
        stride = (args.group_track_stride if tracking.target_count(hint) > 1 else args.single_track_stride) or args.track_stride
        jobs.append((len(row.get('predictions', [])) / stride, row))
    for weight, row in sorted(jobs, key=lambda job: job[0], reverse=True):
        index = min(range(shard_count), key=lambda worker: weights[worker])
        shards[index].append(row)
        weights[index] += weight
    positions = {video_id: index for index, video_id in enumerate(ids)}
    global_cache = tracking.load_cache(args.cache)
    input_digest = hashlib.sha256(args.input.read_bytes()).hexdigest()[:12]
    # Reuse completed videos even if a later resource plan chooses a different shard count.
    if not args.reset_cache:
        for previous in (args.output.parent / 'parallel_tracking').glob(f'{input_digest}_*/part_*/cache.jsonl'):
            for line in previous.read_text(encoding='utf-8').splitlines():
                try:
                    row = json.loads(line)
                    global_cache[str(row['video_id'])] = row
                except (ValueError, KeyError):
                    pass
    folder = args.output.parent / 'parallel_tracking' / f'{input_digest}_{shard_count}'
    folder.mkdir(parents=True, exist_ok=True)
    children = []
    logs = []
    outputs = []
    status_path = folder / 'status.json'
    pending = []
    try:
        for index, shard in enumerate(shards):
            shard.sort(key=lambda row: positions[str(row['video_id'])])
            part = folder / f'part_{index}'
            part.mkdir(exist_ok=True)
            source = part / 'input.jsonl'
            cache = part / 'cache.jsonl'
            output = part / 'predictions.jsonl'
            report = part / 'report.json'
            tracking.write_jsonl(source, shard)
            if not cache.exists():
                tracking.write_jsonl(cache, [global_cache[str(row['video_id'])] for row in shard if str(row['video_id']) in global_cache])
            command = [sys.executable, '-B', str(runtime / 'stage2_prefetch_entry.py'),
                       str(spatial / 'generate_botsort_identity_lock.py'), *original_args]
            for name, value in [('--input', source), ('--cache', cache), ('--output', output),
                                ('--report', report), ('--zip-output', part / 'preview.zip')]:
                replace_argument(command, name, value)
            environment = os.environ.copy()
            threads = os.environ.get('HIGHLIGHT_TRACKING_THREADS', '4')
            environment['OMP_NUM_THREADS'] = threads
            environment['MKL_NUM_THREADS'] = threads
            environment['PYTHONUNBUFFERED'] = '1'
            log = (part / 'console.log').open('ab', buffering=0)
            logs.append(log)
            pending.append((command, environment, log))
            outputs.append((output, report, cache))
            print(f'Tracking worker {index}: {len(shard)} videos, approximately {weights[index]:.0f} samples', flush=True)
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'Tracking worker failed: {codes}; inspect {folder}')
            running = sum(code is None for code in codes)
            snapshot = capture_resources() if dynamic else None
            limit = concurrency_limit(snapshot, workers) if snapshot else workers
            while pending and running < limit:
                command, environment, log = pending.pop(0)
                children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                 env=environment, cwd=runtime))
                running += 1
            codes = [child.poll() for child in children]
            atomic_json(status_path, {'workers': [{'pid': child.pid, 'return_code': code}
                                                  for child, code in zip(children, codes)],
                                      'pending_shards': len(pending), 'concurrency_limit': limit,
                                      'resources': snapshot.to_dict() if snapshot else None,
                                      'updated_unix': time.time()})
            failures = [code for code in codes if code not in (None, 0)]
            if failures:
                raise RuntimeError(f'Tracking worker failed: {codes}; inspect {folder}')
            if not pending and all(code == 0 for code in codes):
                break
            time.sleep(5)
        final = {}
        reports = {}
        merged_cache = dict(global_cache)
        for output, report, cache in outputs:
            for row in tracking.load_jsonl(output):
                video_id = str(row['video_id'])
                if video_id in final:
                    raise ValueError('Duplicate worker output: ' + video_id)
                final[video_id] = row
            reports.update({str(row['video_id']): row for row in json.loads(report.read_text())})
            merged_cache.update(tracking.load_cache(cache))
        if set(final) != set(ids):
            raise ValueError('Worker outputs do not cover the input')
        for original in rows:
            row = final[str(original['video_id'])]
            if row['targetRatioWH'] != original['targetRatioWH']:
                raise ValueError('Worker changed the target ratio')
            if [item['frame'] for item in row['predictions']] != [item['frame'] for item in original['predictions']]:
                raise ValueError('Worker changed the input frame IDs')
            if [item['bboxes'][2] for item in row['predictions']] != [item['bboxes'][2] for item in original['predictions']]:
                raise ValueError('Worker changed crop widths')
        tracking.write_jsonl(args.output, [final[video_id] for video_id in ids])
        args.report.write_text(json.dumps([reports[video_id] for video_id in ids if video_id in reports],
                                          ensure_ascii=False, indent=2), encoding='utf-8')
        tracking.write_jsonl(args.cache, [merged_cache[video_id] for video_id in ids if video_id in merged_cache])
        if args.zip_output:
            with zipfile.ZipFile(args.zip_output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(args.output, arcname=args.output.name)
        atomic_json(status_path, {'status': 'COMPLETE', 'videos': len(rows), 'workers': workers,
                                  'frame_ids_and_crop_widths_unchanged': True, 'updated_unix': time.time()})
        print(f'Parallel tracking merged {len(rows)} videos in original order', flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
