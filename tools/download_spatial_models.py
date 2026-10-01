#!/usr/bin/env python3
"""Download the two pinned official YOLO11 weights and verify every byte."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REVISION = '8b8ac7d1fae7468f85dbf89670dd66f41f485aab'
MODELS = {
    'yolo11m.pt': (40684120, 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'),
    'yolo11m-pose.pt': (42459307, '29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2'),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    receipts = []
    for name, (size, expected) in MODELS.items():
        path = args.output / name
        url = endpoint + '/Ultralytics/YOLO11/resolve/' + REVISION + '/' + name
        if not (path.exists() and path.stat().st_size == size and sha(path) == expected):
            temporary = path.with_suffix('.pt.download')
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
            if temporary.stat().st_size != size or sha(temporary) != expected:
                raise ValueError('Model size/SHA256 differs from pinned official asset: ' + name)
            if path.exists():
                raise FileExistsError('Existing model differs; preserve it under another name before replacing: ' + str(path))
            temporary.replace(path)
        receipts.append({'name': name, 'revision': REVISION, 'bytes': size, 'sha256': expected, 'source': url})
        print('Verified ' + name, flush=True)
    (args.output / 'spatial_model_provenance.json').write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
