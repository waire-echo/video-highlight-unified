"""Build portable, compact inputs from the original ORA records. No re-selection."""
import argparse
from pathlib import Path
import json
import numpy as np
from common import load_rows, read_json, write_json, sha, validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--ora-directory', required=True)
    p.add_argument('--yolo-cache', required=True)
    p.add_argument('--yolo-run-validation', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source, out = Path(a.ora_directory), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    baseline = load_rows(a.baseline)
    import zipfile,hashlib
    validation=read_json(a.yolo_run_validation)
    with zipfile.ZipFile(a.baseline) as z:
        raw=z.read('predictions.jsonl')
    if validation.get('predictions_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('YOLO run validation does not match chosen baseline payload')
    manifest = {'baseline_sha256': sha(a.baseline), 'videos': {},
                'time_source': 'original decoded frame PTS; normalized to zero',
                'uncertainty_cache_note': 'Matching YOLO-only run cache; baseline JSONL SHA256 matched its run validation.',
                'yolo_run_validation': validation}
    timestamps = {}
    cache_arrays = {}
    for row in baseline:
        vid = str(row['video_id'])
        record = read_json(source / 'records' / (vid + '.json'))
        stamp = read_json(source / 'audit/timestamps' / (vid + '.json'))
        if stamp['source_sha256'] != record['source_sha256']:
            raise ValueError('Timestamp source fingerprint mismatch')
        ts = np.asarray(stamp['times'], dtype=np.float64)
        if len(ts) == 0 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError(f'Invalid original timestamps: {vid}')
        timestamps[vid] = ts
        manifest['videos'][vid] = {
            'width': record['width'], 'height': record['height'], 'fps': record['fps'],
            'frame_count': len(ts), 'duration': record['duration'],
            'source_sha256': record['source_sha256'], 'targetRatioWH': row['targetRatioWH'],
            'subject_hint': record['main_subject'],
            'events': [{k: e[k] for k in ('query','start','end','main_subject','grounded_segment','selection') if k in e}
                       for e in record['events']]}
    for row in load_rows(a.yolo_cache):
        vid = str(row['video_id'])
        if vid not in manifest['videos']:
            continue
        observations = {}
        for group in row.get('groups', []):
            for item in group.get('detections', []):
                observations[int(item['frame'])] = float(item.get('confidence', 0)) if item.get('detected') else 0.0
        cache_arrays[vid] = np.array(sorted(observations.items()), dtype=float).reshape(-1, 2)
    np.savez_compressed(out/'timestamps.npz', **timestamps)
    np.savez_compressed(out/'yolo_evidence.npz', **cache_arrays)
    manifest['timestamps_sha256'] = sha(out/'timestamps.npz')
    manifest['yolo_evidence_sha256'] = sha(out/'yolo_evidence.npz')
    manifest['yolo_cache_source_sha256'] = sha(a.yolo_cache)
    write_json(out/'manifest.json', manifest)
    print(json.dumps(validate(baseline, baseline, manifest) | {'manifest': str(out/'manifest.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
