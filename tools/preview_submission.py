#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview a highlight re-framing submission.

The script reads a competition JSONL file, opens the corresponding source
videos, and writes small MP4 previews for selected video ids.

Modes:
  crop         only the submitted crop, concatenated over predicted frames
  overlay      source frames with the submitted crop box drawn on top
  side-by-side source+box on the left, cropped output on the right
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Record = Dict[str, object]


def load_submission(path: Path) -> Dict[str, Record]:
    records: Dict[str, Record] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vid = str(obj["video_id"])
            records[vid] = obj
    return records


def find_video_dir(root: Path) -> Path:
    candidates = []
    for video in root.rglob("0.mp4"):
        if video.parent.name.lower() == "video":
            candidates.append(video.parent)
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-find a video directory. Pass --video-dir explicitly."
        )
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def parse_video_ids(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out or None


def parse_canvas_size(value: str) -> Tuple[int, int]:
    text = value.lower().replace("*", "x").replace(",", "x")
    parts = [p.strip() for p in text.split("x") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, e.g. 1280x720")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Canvas size must be integers") from exc
    return even(w), even(h)


def selected_records(
    records: Dict[str, Record],
    video_ids: Optional[List[str]],
    first: int,
    random_count: int,
    seed: Optional[int],
) -> List[Tuple[str, Record]]:
    if video_ids:
        missing = [vid for vid in video_ids if vid not in records]
        if missing:
            raise KeyError(f"video_id not found in submission: {missing}")
        return [(vid, records[vid]) for vid in video_ids]
    ordered = sorted(records.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
    if random_count > 0:
        rng = random.Random(seed)
        count = min(random_count, len(ordered))
        sampled = rng.sample(ordered, count)
        return sorted(sampled, key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
    if first > 0:
        ordered = ordered[:first]
    return ordered


def even(value: int) -> int:
    return max(2, value - (value % 2))


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    width = even(int(round(w * height / float(h))))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def target_size(target: Tuple[int, int], height: int) -> Tuple[int, int]:
    tw, th = target
    width = even(int(round(height * tw / float(th))))
    return width, even(height)


def bbox_to_crop(
    bbox: Iterable[object], target: Tuple[int, int], frame_w: int, frame_h: int
) -> Tuple[int, int, int, int, bool]:
    raw = list(bbox)
    if len(raw) != 3:
        raise ValueError(f"bboxes must be [x,y,w], got {raw}")
    tw, th = target
    x = int(round(float(raw[0])))
    y = int(round(float(raw[1])))
    w = int(round(float(raw[2])))
    changed = False

    if w <= 0:
        w = 1
        changed = True
    h = int(round(w * th / float(tw)))
    if h <= 0:
        h = 1
        changed = True

    max_w_by_h = int(math.floor(frame_h * tw / float(th)))
    max_w = max(1, min(frame_w, max_w_by_h))
    if w > max_w:
        w = max_w
        h = int(round(w * th / float(tw)))
        changed = True

    if x < 0:
        x = 0
        changed = True
    if y < 0:
        y = 0
        changed = True
    if x + w > frame_w:
        x = max(0, frame_w - w)
        changed = True
    if y + h > frame_h:
        y = max(0, frame_h - h)
        changed = True

    return x, y, w, h, changed


def draw_overlay(
    frame: np.ndarray,
    crop: Tuple[int, int, int, int],
    video_id: str,
    frame_id: int,
    changed: bool,
) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = crop
    color = (0, 255, 0) if not changed else (0, 165, 255)
    thickness = max(2, int(round(min(frame.shape[:2]) / 240)))
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, thickness)
    label = f"id={video_id} frame={frame_id} bbox=[{x},{y},{w}]"
    if changed:
        label += " clamped"
    cv2.putText(
        out,
        label,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        label,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_side_by_side(
    overlay: np.ndarray, crop_img: np.ndarray, target: Tuple[int, int], height: int
) -> np.ndarray:
    left = resize_to_height(overlay, height)
    crop_w, crop_h = target_size(target, height)
    right = cv2.resize(crop_img, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
    gap = 12
    canvas = np.zeros((height, left.shape[1] + gap + right.shape[1], 3), dtype=np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] + gap :] = right
    cv2.putText(
        canvas,
        "source + submitted box",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "cropped highlight output",
        (left.shape[1] + gap + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def fit_to_canvas(img: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Letterbox an image into a fixed-size canvas for concatenated previews."""
    h, w = img.shape[:2]
    scale = min(canvas_w / float(w), canvas_h / float(h))
    new_w = even(int(round(w * scale)))
    new_h = even(int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_title_card(
    video_id: str,
    target: Tuple[int, int],
    frame_summary: str,
    canvas_w: int,
    canvas_h: int,
    note: str = "",
) -> np.ndarray:
    card = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    lines = [
        f"video_id: {video_id}",
        f"targetRatioWH: [{target[0]}, {target[1]}]",
        f"predicted frames: {frame_summary}",
    ]
    if note:
        lines.append(note)
    y = canvas_h // 2 - 58
    for i, line in enumerate(lines):
        scale = 1.1 if i == 0 else 0.72
        thickness = 2
        cv2.putText(
            card,
            line,
            (56, y + i * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return card


def build_preview_frame(
    frame: np.ndarray,
    pred: Record,
    target: Tuple[int, int],
    video_id: str,
    frame_id: int,
    frame_w: int,
    frame_h: int,
    mode: str,
    preview_height: int,
) -> Tuple[Optional[np.ndarray], bool]:
    x, y, w, h, changed = bbox_to_crop(pred["bboxes"], target, frame_w, frame_h)
    crop_img = frame[y : y + h, x : x + w]
    if crop_img.size == 0:
        return None, changed

    overlay = draw_overlay(frame, (x, y, w, h), video_id, frame_id, changed)
    if mode == "crop":
        out_w, out_h = target_size(target, preview_height)
        preview = cv2.resize(crop_img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    elif mode == "overlay":
        preview = resize_to_height(overlay, preview_height)
    elif mode == "side-by-side":
        preview = make_side_by_side(overlay, crop_img, target, preview_height)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return preview, changed


def summarize_frames(frames: List[int]) -> str:
    if not frames:
        return "none"
    spans = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        spans.append((start, prev))
        start = prev = f
    spans.append((start, prev))
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in spans[:8]]
    if len(spans) > 8:
        parts.append("...")
    return ", ".join(parts)


def prediction_frames(record: Record, max_frames: int, every: int) -> Tuple[Dict[int, Record], List[int]]:
    preds = record.get("predictions", [])
    if not isinstance(preds, list) or not preds:
        return {}, []
    pred_by_frame: Dict[int, Record] = {}
    for pred in preds:
        frame_id = int(pred["frame"])
        if frame_id not in pred_by_frame:
            pred_by_frame[frame_id] = pred
    frames = sorted(pred_by_frame)
    if every > 1:
        frames = frames[::every]
    if max_frames > 0:
        frames = frames[:max_frames]
    return pred_by_frame, frames


def write_preview(
    record: Record,
    video_dir: Path,
    out_dir: Path,
    mode: str,
    preview_height: int,
    playback_fps: Optional[float],
    max_frames: int,
    every: int,
) -> Optional[Path]:
    video_id = str(record["video_id"])
    target_raw = record.get("targetRatioWH", [16, 9])
    target = (int(target_raw[0]), int(target_raw[1]))
    pred_by_frame, frames = prediction_frames(record, max_frames, every)
    if not frames:
        print(f"[skip] video {video_id}: no predictions")
        return None

    video_path = video_dir / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] video {video_id}: cannot open {video_path}")
        return None

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = float(playback_fps or source_fps)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    out_path = out_dir / f"{video_id}_{mode}.mp4"
    written = 0
    clamped = 0

    need = set(frames)
    current = frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current)
    last = frames[-1]
    while current <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if current in need:
            pred = pred_by_frame[current]
            preview, changed = build_preview_frame(
                frame,
                pred,
                target,
                video_id,
                current,
                frame_w,
                frame_h,
                mode,
                preview_height,
            )
            if preview is None:
                current += 1
                continue
            if changed:
                clamped += 1

            if writer is None:
                out_dir.mkdir(parents=True, exist_ok=True)
                h0, w0 = preview.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w0, h0))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video writer: {out_path}")
            writer.write(preview)
            written += 1
        current += 1

    cap.release()
    if writer is not None:
        writer.release()
    if written == 0:
        print(f"[skip] video {video_id}: no preview frames written")
        return None

    print(
        f"[ok] video {video_id}: {written} frames, fps={fps:.2f}, "
        f"segments={summarize_frames(frames)}, clamped={clamped} -> {out_path}"
    )
    return out_path


def write_concat_preview(
    selected: List[Tuple[str, Record]],
    video_dir: Path,
    out_dir: Path,
    mode: str,
    preview_height: int,
    playback_fps: float,
    max_frames: int,
    every: int,
    concat_out: str,
    canvas_w: int,
    canvas_h: int,
    title_seconds: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(concat_out)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, playback_fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {out_path}")

    title_frames = max(1, int(round(playback_fps * title_seconds)))
    total_written = 0
    try:
        for _, record in selected:
            video_id = str(record["video_id"])
            target_raw = record.get("targetRatioWH", [16, 9])
            target = (int(target_raw[0]), int(target_raw[1]))
            pred_by_frame, frames = prediction_frames(record, max_frames, every)
            summary = summarize_frames(frames)
            if not frames:
                card = make_title_card(
                    video_id,
                    target,
                    summary,
                    canvas_w,
                    canvas_h,
                    note="no predictions, skipped",
                )
                for _ in range(title_frames):
                    writer.write(card)
                    total_written += 1
                print(f"[skip] video {video_id}: no predictions")
                continue

            video_path = video_dir / f"{video_id}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                card = make_title_card(
                    video_id,
                    target,
                    summary,
                    canvas_w,
                    canvas_h,
                    note=f"cannot open {video_path.name}",
                )
                for _ in range(title_frames):
                    writer.write(card)
                    total_written += 1
                print(f"[skip] video {video_id}: cannot open {video_path}")
                continue

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            card = make_title_card(video_id, target, summary, canvas_w, canvas_h)
            for _ in range(title_frames):
                writer.write(card)
                total_written += 1

            need = set(frames)
            current = frames[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            last = frames[-1]
            written = 0
            clamped = 0
            while current <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                if current in need:
                    pred = pred_by_frame[current]
                    preview, changed = build_preview_frame(
                        frame,
                        pred,
                        target,
                        video_id,
                        current,
                        frame_w,
                        frame_h,
                        mode,
                        preview_height,
                    )
                    if preview is not None:
                        writer.write(fit_to_canvas(preview, canvas_w, canvas_h))
                        total_written += 1
                        written += 1
                        if changed:
                            clamped += 1
                current += 1
            cap.release()
            print(
                f"[ok] video {video_id}: {written} frames, "
                f"segments={summary}, clamped={clamped}"
            )
    finally:
        writer.release()

    print(
        f"Done concat: {len(selected)} videos / {total_written} preview frames "
        f"-> {out_path}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render preview videos from a submission JSONL.")
    parser.add_argument("--pred", required=True, help="Predictions JSONL to preview")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--out-dir", default="previews")
    parser.add_argument("--video-id", default=None, help="Comma-separated ids, e.g. 0,1,25")
    parser.add_argument("--first", type=int, default=0, help="Preview the first N records if --video-id is omitted.")
    parser.add_argument("--random", type=int, default=0, help="Preview N random videos, e.g. --random 10.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --random.")
    parser.add_argument("--all", action="store_true", help="Preview all records explicitly. This is also the default.")
    parser.add_argument(
        "--mode",
        choices=["crop", "overlay", "side-by-side"],
        default="side-by-side",
    )
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument("--playback-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per preview; 0 means no limit.")
    parser.add_argument("--every", type=int, default=1, help="Use every Nth predicted frame for a lighter preview.")
    parser.add_argument("--concat", action="store_true", help="Write one continuous preview video instead of separate files.")
    parser.add_argument("--concat-out", default="all_preview.mp4", help="Output filename for --concat.")
    parser.add_argument("--concat-size", type=parse_canvas_size, default=parse_canvas_size("1280x720"))
    parser.add_argument("--title-seconds", type=float, default=1.0, help="Title card duration before each clip in --concat.")
    args = parser.parse_args()

    root = Path.cwd()
    pred_path = Path(args.pred)
    if not pred_path.is_absolute():
        pred_path = root / pred_path
    video_dir = Path(args.video_dir) if args.video_dir else find_video_dir(root)
    if not video_dir.is_absolute():
        video_dir = root / video_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    records = load_submission(pred_path)
    ids = parse_video_ids(args.video_id)
    first = 0 if args.all else args.first
    random_count = 0 if args.all else args.random
    selected = selected_records(records, ids, first, random_count, args.seed)
    if not selected:
        raise SystemExit("No records selected. Use --video-id or --first.")

    print(f"Submission: {pred_path}")
    print(f"Video dir : {video_dir}")
    print(f"Out dir   : {out_dir}")
    print(f"Mode      : {args.mode}")

    if args.concat:
        canvas_w, canvas_h = args.concat_size
        write_concat_preview(
            selected,
            video_dir,
            out_dir,
            args.mode,
            even(args.preview_height),
            float(args.playback_fps or 24.0),
            args.max_frames,
            max(1, args.every),
            args.concat_out,
            canvas_w,
            canvas_h,
            max(0.0, args.title_seconds),
        )
    else:
        for _, record in selected:
            write_preview(
                record,
                video_dir,
                out_dir,
                args.mode,
                even(args.preview_height),
                args.playback_fps,
                args.max_frames,
                max(1, args.every),
            )


if __name__ == "__main__":
    main()
