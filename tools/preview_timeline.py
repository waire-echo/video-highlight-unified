#!/usr/bin/env python3
"""Render full video timelines for a submission or two synchronized submissions."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

from preview_submission import bbox_to_crop

WIDTH, HEIGHT, PANEL_WIDTH = 1536, 640, 512


def load_predictions(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    result = {str(row["video_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate video IDs: {path}")
    return result


def label(canvas, text, x, y, color=(230, 230, 230)):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1, cv2.LINE_AA)


def panel(canvas, image, index):
    if image is None:
        label(canvas, "NOT SELECTED", index * PANEL_WIDTH + 145, 320, (150, 150, 150))
        return
    height, width = image.shape[:2]
    scale = min(496 / width, 526 / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    x = index * PANEL_WIDTH + (PANEL_WIDTH - new_width) // 2
    y = 70 + (526 - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def crop(frame, prediction, target):
    if prediction is None:
        return None, None, False
    height, width = frame.shape[:2]
    x, y, crop_width, crop_height, changed = bbox_to_crop(prediction["bboxes"], target, width, height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + crop_width - 1, y + crop_height - 1), (0, 230, 80), max(2, width // 400))
    return overlay, frame[y:y + crop_height, x:x + crop_width], changed


def render(video_id, primary, comparison, args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError(f"Unsafe video filename: {video_id}")
    if comparison and primary["targetRatioWH"] != comparison["targetRatioWH"]:
        raise ValueError(f"Video {video_id}: target ratios differ")
    target = tuple(primary["targetRatioWH"])
    predictions = {int(p["frame"]): p for p in primary["predictions"]}
    others = {int(p["frame"]): p for p in comparison["predictions"]} if comparison else {}
    cap = cv2.VideoCapture(str(args.videos / f"{video_id}.mp4"))
    process = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video {video_id}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(source_fps) or source_fps <= 0 or source_frames <= 0:
            raise ValueError(f"Invalid source metadata: {video_id}")
        indices = predictions.keys() | others.keys()
        if indices and (min(indices) < 0 or max(indices) >= source_frames):
            raise ValueError(f"Prediction frame outside source: {video_id}")
        output = args.out / f"{video_id}_timeline.mp4"
        command = ["ffmpeg", "-v", "error", "-y" if args.overwrite else "-n", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(args.fps),
                   "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                   "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        frame_id = output_frames = clamped = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if output_frames / args.fps < (frame_id + 1) / source_fps:
                canvas = np.full((HEIGHT, WIDTH, 3), 18, np.uint8)
                label(canvas, "ORIGINAL - FULL TIMELINE", 16, 30)
                panel(canvas, frame, 0)
                overlay, selected, changed = crop(frame, predictions.get(frame_id), target)
                clamped += int(changed)
                if comparison is None:
                    label(canvas, "SELECTED FRAMES + BOX", 528, 30)
                    label(canvas, "CROPPED HIGHLIGHT", 1040, 30)
                    panel(canvas, overlay, 1)
                    panel(canvas, selected, 2)
                else:
                    label(canvas, args.label_a, 528, 30)
                    label(canvas, args.label_b, 1040, 30)
                    _, other_crop, other_changed = crop(frame, others.get(frame_id), target)
                    clamped += int(other_changed)
                    panel(canvas, selected, 1)
                    panel(canvas, other_crop, 2)
                label(canvas, f"Video {video_id} | {frame_id / source_fps:.2f}s / {source_frames / source_fps:.2f}s | frame {frame_id}", 16, 628)
                while output_frames / args.fps < (frame_id + 1) / source_fps:
                    process.stdin.write(canvas.tobytes())
                    output_frames += 1
            frame_id += 1
    finally:
        cap.release()
        if process is not None:
            try:
                process.stdin.close()
            finally:
                code = process.wait()
    expected = math.ceil(source_frames / source_fps * args.fps - 1e-8)
    if code or frame_id != source_frames or abs(output_frames - expected) > 1:
        raise RuntimeError(f"Video {video_id}: encoder={code}, decoded={frame_id}/{source_frames}, output={output_frames}/{expected}")
    return {"video_id": video_id, "source_frames": frame_id, "source_fps": source_fps,
            "output_frames": output_frames, "duration_sec": output_frames / args.fps,
            "selected_frames": len(predictions), "comparison_selected_frames": len(others) if comparison else None,
            "clamped_render_samples": clamped, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--ids", help="Comma-separated IDs; default is every ID in the first submission")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--label-a", default="SUBMISSION A - CROPPED")
    parser.add_argument("--label-b", default="SUBMISSION B - CROPPED")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if len(args.label_a) > 35 or len(args.label_b) > 35:
        parser.error("Panel labels must be at most 35 characters")
    first = load_predictions(args.prediction)
    second = load_predictions(args.compare) if args.compare else None
    ids = [value.strip() for value in args.ids.split(",") if value.strip()] if args.ids else list(first)
    if not ids or len(set(ids)) != len(ids):
        parser.error("Select one or more unique video IDs")
    for video_id in ids:
        if video_id not in first or (second is not None and video_id not in second):
            parser.error(f"Missing submission row: {video_id}")
        if (args.out / f"{video_id}_timeline.mp4").exists() and not args.overwrite:
            parser.error(f"Existing preview for {video_id}; choose a new output directory or --overwrite")
    args.out.mkdir(parents=True, exist_ok=True)
    results = [render(video_id, first[video_id], second[video_id] if second is not None else None, args) for video_id in ids]
    summary = {"videos": results, "prediction": str(args.prediction), "comparison": str(args.compare) if args.compare else None,
               "note": "Silent diagnostic previews. Source-FPS timing is for display; not official PTS evaluation or GT."}
    (args.out / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
