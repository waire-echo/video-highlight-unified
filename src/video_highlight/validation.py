import json
import math
import os
from typing import Dict, List, Sequence, Tuple

from .io_utils import load_index, video_meta, video_path_for


def _iter_submission(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc


def _load_submission(path: str) -> List[dict]:
    return list(_iter_submission(path))


def validate_submission_for_items(submission_path: str, index_items: Sequence[Tuple[str, Tuple[int, int]]], video_dir: str = "") -> Tuple[bool, List[str]]:
    errors = []
    expected_ids = [vid for vid, _target in index_items]
    expected_targets = {vid: target for vid, target in index_items}
    records = _iter_submission(submission_path)

    seen_ids = []
    seen = set()
    for rec in records:
        if not isinstance(rec, dict):
            errors.append("submission line must be an object")
            continue
        vid = str(rec.get("video_id", ""))
        seen_ids.append(vid)
        if vid in seen:
            errors.append(f"duplicate video_id line: {vid}")
        seen.add(vid)
        if vid not in expected_targets:
            continue
        target = tuple(rec.get("targetRatioWH", []))
        if target != tuple(expected_targets[vid]):
            errors.append(f"{vid}: targetRatioWH mismatch, got {target}, expected {expected_targets[vid]}")
            continue
        preds = rec.get("predictions")
        if not isinstance(preds, list):
            errors.append(f"{vid}: predictions must be a list")
            continue

        n_frames = fps = width = height = None
        if video_dir:
            video_path = video_path_for(video_dir, vid)
            if os.path.exists(video_path):
                n_frames, fps, width, height = video_meta(video_path)
                if n_frames <= 0 or width <= 0 or height <= 0:
                    errors.append(f"{vid}: source video is unreadable")

        frame_seen = set()
        last_frame = -1
        target_w, target_h = expected_targets[vid]
        for i, pred in enumerate(preds):
            if not isinstance(pred, dict):
                errors.append(f"{vid}: prediction {i} must be an object")
                continue
            frame = pred.get("frame")
            bbox = pred.get("bboxes")
            if type(frame) is not int:
                errors.append(f"{vid}: prediction {i} frame must be int")
                continue
            if frame in frame_seen:
                errors.append(f"{vid}: duplicate frame {frame}")
            frame_seen.add(frame)
            if frame < last_frame:
                errors.append(f"{vid}: frames are not sorted near {frame}")
            last_frame = frame
            if n_frames is not None and (frame < 0 or frame >= n_frames):
                errors.append(f"{vid}: frame {frame} out of range 0..{n_frames - 1}")
            if not isinstance(bbox, list) or len(bbox) != 3:
                errors.append(f"{vid}: prediction {i} bboxes must be [x,y,w]")
                continue
            if not all(type(v) is int for v in bbox):
                errors.append(f"{vid}: prediction {i} bboxes values must be ints")
                continue
            if any(math.isnan(float(v)) for v in bbox):
                errors.append(f"{vid}: prediction {i} contains NaN")
                continue
            x, y, w = bbox
            if x < 0 or y < 0 or w <= 0:
                errors.append(f"{vid}: invalid bbox {bbox}")
                continue
            if width and height:
                h = int(round(w * target_h / float(target_w)))
                if h <= 0 or x + w > width or y + h > height:
                    errors.append(f"{vid}: bbox {bbox} -> h={h} exceeds {width}x{height}")

    key = lambda value: (0, int(value), value) if value.isdigit() else (1, 0, value)
    missing = sorted(set(expected_ids) - seen, key=key)
    extra = sorted(seen - set(expected_ids), key=key)
    if missing:
        errors.append(f"missing video_id(s): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"extra video_id(s): {extra[:10]}{'...' if len(extra) > 10 else ''}")
    return len(errors) == 0, errors


def validate_submission(submission_path: str, index_path: str, video_dir: str = "") -> Tuple[bool, List[str]]:
    index = load_index(index_path)
    return validate_submission_for_items(submission_path, index, video_dir)
