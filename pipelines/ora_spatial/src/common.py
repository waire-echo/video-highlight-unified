"""Strict submission contracts; frame IDs always refer to the original video."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def load_rows(path):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            if z.namelist() != ['predictions.jsonl'] or z.testzip() is not None:
                raise ValueError('Expected one valid root predictions.jsonl')
            raw = z.read('predictions.jsonl').decode('utf-8-sig')
    else:
        raw = path.read_text(encoding='utf-8-sig')
    rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
    if len({str(r['video_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate video IDs')
    return rows


def box_from_center(center, w, ratio, width, height):
    h = w * ratio[1] / ratio[0]
    if not (0 < w <= width and 0 < h <= height + 1e-8):
        raise ValueError('Crop cannot fit source')
    return [min(max(0, round(center[0] - w / 2)), math.floor(width - w)),
            min(max(0, round(center[1] - h / 2)), math.floor(height - h + 1e-8)), w]


def center_of(box, ratio):
    x, y, w = box
    return [x + w / 2, y + w * ratio[1] / ratio[0] / 2]


def validate(rows, baseline, manifest, preserve_width=True):
    if len(rows) != len(baseline):
        raise ValueError('Video count changed')
    changed = 0
    changed_widths = 0
    total = 0
    video_changes = {}
    for r, b in zip(rows, baseline):
        vid = str(b['video_id'])
        if str(r['video_id']) != vid or r['targetRatioWH'] != b['targetRatioWH']:
            raise ValueError('ID/order/ratio changed')
        m = manifest['videos'][vid]
        frames = [p['frame'] for p in r['predictions']]
        if frames != [p['frame'] for p in b['predictions']] or frames != sorted(set(frames)):
            raise ValueError(f'Temporal mask changed: {vid}')
        n = 0
        for p, old in zip(r['predictions'], b['predictions']):
            f = p['frame']
            if type(f) is not int or not 0 <= f < m['frame_count']:
                raise ValueError(f'Bad frame: {vid}/{f}')
            box = p['bboxes']
            if len(box) != 3 or any(type(x) not in (float, int) or not math.isfinite(x) for x in box):
                raise ValueError(f'Bad box: {vid}/{f}')
            x, y, w = box
            h = w * r['targetRatioWH'][1] / r['targetRatioWH'][0]
            if not (x >= 0 and y >= 0 and w > 0 and x + w <= m['width'] + 1e-7
                    and y + h <= m['height'] + 1e-7):
                raise ValueError(f'Out of bounds: {vid}/{f}')
            if preserve_width and w != old['bboxes'][2]:
                raise ValueError(f'Width changed: {vid}/{f}')
            changed_widths += w != old['bboxes'][2]
            n += box != old['bboxes']
        changed += n
        total += len(frames)
        video_changes[vid] = n
    return {'videos': len(rows), 'frames': total, 'changed_boxes': changed,
            'changed_videos': sum(v > 0 for v in video_changes.values()),
            'changes_by_video': video_changes,
            'empty_video_ids': [r['video_id'] for r in rows if not r['predictions']],
            'temporal_mask_unchanged': True, 'widths_unchanged': changed_widths==0,
            'changed_widths': changed_widths, 'width_lock_enforced': preserve_width}


def export(path, rows, baseline, manifest, preserve_width=True):
    report = validate(rows, baseline, manifest,preserve_width=preserve_width)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
                  for r in rows).encode('utf-8')
    path.with_suffix('.jsonl').write_bytes(raw)
    temp = path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo('predictions.jsonl', date_time=(2026, 10, 8, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, raw)
    os.replace(temp, path.with_suffix('.zip'))
    validate(load_rows(path.with_suffix('.zip')), baseline, manifest,preserve_width=preserve_width)
    report['zip_sha256'] = sha(path.with_suffix('.zip'))
    write_json(path.with_suffix('.validation.json'), report)
    return report


def groups(frames, cuts=()):
    cuts = set(cuts)
    result = []
    for f in frames:
        if not result or f != result[-1][-1] + 1 or f in cuts:
            result.append([f])
        else:
            result[-1].append(f)
    return result


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
