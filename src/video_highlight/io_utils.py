import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2


IndexItem = Tuple[str, Tuple[int, int]]


def load_index(index_path: str) -> List[IndexItem]:
    """Load either the official array index or a directory of per-video jsonl files."""
    path = Path(index_path)
    if path.is_dir():
        items = []
        files = sorted(path.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as f:
                line = f.readline().strip()
            if not line:
                continue
            obj = json.loads(line)
            tr = obj.get("targetRatioWH", [16, 9])
            items.append((file_path.stem, (int(tr[0]), int(tr[1]))))
        return items

    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    if text[0] == "[":
        raw_items = json.loads(text)
    else:
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]

    items = []
    for obj in raw_items:
        vid = str(obj["video_id"])
        tr = obj.get("targetRatioWH", [16, 9])
        items.append((vid, (int(tr[0]), int(tr[1]))))
    return items


def write_index(items: Sequence[IndexItem], out_path: str) -> None:
    data = [
        {"video_id": vid, "targetRatioWH": [int(target[0]), int(target[1])]}
        for vid, target in items
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def video_path_for(video_dir: str, video_id: str) -> str:
    return os.path.join(video_dir, f"{video_id}.mp4")


def video_meta(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return 0, 0.0, 0, 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n_frames, fps, width, height


def extract_frames(video_path: str, frame_ids: Iterable[int]) -> Dict[int, object]:
    """Read requested frames with sequential decoding."""
    frame_ids = sorted(set(int(f) for f in frame_ids if int(f) >= 0))
    if not frame_ids:
        return {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {}

    out = {}
    start = frame_ids[0]
    end = frame_ids[-1]
    want = set(frame_ids)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            out[idx] = frame
        idx += 1
    cap.release()
    return out


def iter_sampled_frames(video_path: str, step_frames: int):
    """Yield (frame_index, frame) every step_frames."""
    step_frames = max(1, int(step_frames))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step_frames == 0:
            yield idx, frame
        idx += 1
    cap.release()


def write_jsonl(records: Iterable[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
