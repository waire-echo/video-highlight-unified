"""Experiment 1: camera smoothing + center fallback; exact baseline temporal mask."""
import argparse
from pathlib import Path
import copy
import numpy as np
from common import load_rows, read_json, write_json, sha, digest, export, groups, center_of, box_from_center
from camera import smooth_camera


def cache_reliability(frames, ts, observations, cfg):
    # Never call unobserved frames a failed detection. Existing cache can have gaps.
    result = np.full(len(frames), cfg['unknown_evidence_reliability'], float)
    if observations is None or len(observations) == 0:
        return result
    observed_frames = observations[:, 0].astype(int)
    known_times = ts[observed_frames]
    for i, f in enumerate(frames):
        k = int(np.searchsorted(known_times, ts[f]))
        candidates = [j for j in (k-1, k) if 0 <= j < len(known_times)]
        j = min(candidates, key=lambda k: abs(known_times[k]-ts[f]))
        if abs(known_times[j]-ts[f]) <= cfg['cache_nearest_seconds']:
            c = observations[j, 1]
            result[i] = np.clip((c-cfg['low_detection_confidence']) /
                                (cfg['high_detection_confidence']-cfg['low_detection_confidence']), 0, 1)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', default='inputs')
    p.add_argument('--baseline', default='inputs/baseline.zip')
    p.add_argument('--shots', default='output/shots')
    p.add_argument('--config', default='config/default.json')
    p.add_argument('--output', default='output/E1_camera')
    a = p.parse_args()
    cfg = read_json(a.config)
    root = Path(a.inputs)
    manifest = read_json(root/'manifest.json')
    if sha(a.baseline) != manifest['baseline_sha256']:
        raise ValueError('Wrong baseline; rebuild inputs explicitly')
    if sha(root/'timestamps.npz') != manifest['timestamps_sha256'] or sha(root/'yolo_evidence.npz') != manifest['yolo_evidence_sha256']:
        raise ValueError('Input cache fingerprint mismatch')
    ts_cache = np.load(root/'timestamps.npz')
    evidence = np.load(root/'yolo_evidence.npz')
    base = load_rows(a.baseline)
    rows = copy.deepcopy(base)
    report = {'baseline_sha256': sha(a.baseline), 'config': cfg, 'groups': {}, 'stage': 'E1_camera'}
    for row in rows:
        vid = str(row['video_id'])
        meta = manifest['videos'][vid]
        ts = ts_cache[vid]
        frame_map = {p['frame']: p for p in row['predictions']}
        if not frame_map:
            continue
        shot = read_json(Path(a.shots)/(vid+'.json'))
        if shot['source_sha256'] != meta['source_sha256']:
            raise ValueError('Wrong shot cache')
        reports = []
        for fs in groups(list(frame_map), shot['cuts']):
            boxes = [frame_map[f]['bboxes'] for f in fs]
            widths = {b[2] for b in boxes}
            if len(widths) != 1:
                raise ValueError('Experiment requires constant width within a selected segment')
            w = boxes[0][2]
            centers = [center_of(b, row['targetRatioWH']) for b in boxes]
            # Geometry-only uncertainty is not a calibrated model probability.
            q = cache_reliability(fs, ts, evidence[vid] if vid in evidence else None, cfg['stage1'])
            # Hints can be wrong (a scene-labelled video can contain a dog/person).
            # Only actual baseline detection evidence determines uncertainty here.
            h = w*row['targetRatioWH'][1]/row['targetRatioWH'][0]
            path, diag = smooth_camera(ts[fs], centers, q, (meta['width'],meta['height']), (w,h), cfg['camera'])
            for f, c in zip(fs, path):
                frame_map[f]['bboxes'] = box_from_center(c,w,row['targetRatioWH'],meta['width'],meta['height'])
            reports.append({'first': fs[0], 'last': fs[-1], **diag})
        report['groups'][vid] = reports
    summary = export(a.output, rows, base, manifest)
    report['summary'] = summary
    report['output_jsonl_sha256'] = sha(Path(a.output).with_suffix('.jsonl'))
    write_json(Path(a.output).with_suffix('.report.json'), report)
    print({k:v for k,v in summary.items() if k!='changes_by_video'})


if __name__ == '__main__':
    main()
