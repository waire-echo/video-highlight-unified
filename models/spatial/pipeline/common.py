import hashlib
import importlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as fp:
        json.dump(value, fp, ensure_ascii=False, separators=(',', ':'))
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def entrypoint(spec):
    name, attribute = spec.split(':')
    return getattr(importlib.import_module(name), attribute)


def artifact(root, config, key):
    path = root / config['assets'][key]['path']
    assert path.is_file(), f'Missing packaged asset: {path}'
    return path


def validate_stage1(row):
    assert row['valid'] and row['video_id'] and len(row['targetRatioWH']) == 2
    assert len(row['segments_frame']) > 0
    previous = -1
    for a, b in row['segments_frame']:
        assert type(a) is int and type(b) is int
        assert previous < a <= b < row['frame_count']
        previous = b


def validate_stage2(row, stage1, width, height):
    expected = [f for a, b in stage1['segments_frame'] for f in range(a, b + 1)]
    assert row['video_id'] == stage1['video_id']
    assert row['targetRatioWH'] == stage1['targetRatioWH']
    assert [x['frame'] for x in row['predictions']] == expected
    rw, rh = row['targetRatioWH']
    for pred in row['predictions']:
        box = pred['bboxes']
        assert len(box) == 3 and all(type(v) is int for v in box)
        x, y, w = box
        assert x >= 0 and y >= 0 and w > 0
        assert x + w <= width and y + w * rh / rw <= height + 1e-6
