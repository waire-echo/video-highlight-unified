"""TransNetV2 on original frames, never concatenated selected frames."""
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from common import read_json, write_json, sha, digest


def decode_small(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {path}')
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(bgr, (48, 27)), cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.asarray(frames, dtype=np.uint8)


def boundaries(scores, threshold):
    # One boundary at the maximum in each transition run, plus its end for fades.
    active = np.flatnonzero(scores >= threshold)
    cuts = {0}
    for g in np.split(active, np.flatnonzero(np.diff(active) > 1) + 1):
        if len(g):
            cuts.add(int(g[np.argmax(scores[g])]))
            if len(g) > 3:
                cuts.add(int(g[-1]) + 1)
    return sorted(c for c in cuts if c < len(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video-dir', required=True)
    p.add_argument('--inputs', default='inputs')
    p.add_argument('--weights', default='models/transnetv2-pytorch-weights.pth')
    p.add_argument('--output', default='output/shots')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--ids', default='')
    a = p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor'))
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2()
    model.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True))
    model.to(a.device).eval()
    manifest = read_json(Path(a.inputs)/'manifest.json')
    signature = digest({'weights': sha(a.weights), 'code': sha(__file__),
                        'vendor': sha(Path(__file__).resolve().parents[1]/'vendor/transnetv2_pytorch.py'),
                        'threshold': a.threshold})
    selected = set(a.ids.split(',')) if a.ids else None
    for i, (vid, meta) in enumerate(manifest['videos'].items(), 1):
        if selected is not None and vid not in selected:
            continue
        out = Path(a.output)/(vid+'.json')
        if out.exists():
            prev = read_json(out)
            if prev.get('signature') == signature and prev.get('source_sha256') == meta['source_sha256']:
                continue
        video = Path(a.video_dir)/(vid+'.mp4')
        if sha(video) != meta['source_sha256']:
            raise ValueError(f'Source video changed: {vid}')
        frames = decode_small(video)
        if len(frames) != meta['frame_count']:
            raise ValueError(f'Decode count mismatch: {vid}: {len(frames)} != {meta["frame_count"]}')
        scores = []
        with torch.inference_mode():
            for start in range(0, len(frames), 50):
                ids = np.clip(np.arange(start-25, start+75), 0, len(frames)-1)
                logits, _ = model(torch.from_numpy(frames[ids][None]).to(a.device))
                scores.extend(torch.sigmoid(logits[0, 25:75, 0]).cpu().tolist())
        scores = np.array(scores[:len(frames)])
        write_json(out, {'video_id': vid, 'cuts': boundaries(scores, a.threshold),
                         'signature': signature, 'source_sha256': meta['source_sha256'],
                         'frame_count': len(frames), 'parameters': sum(p.numel() for p in model.parameters())})
        print(f'[{i}/{len(manifest["videos"])}] {vid} cuts={len(boundaries(scores,a.threshold))}', flush=True)


if __name__ == '__main__':
    main()
