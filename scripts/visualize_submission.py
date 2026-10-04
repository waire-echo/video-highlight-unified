#!/usr/bin/env python3
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from video_highlight.io_utils import video_meta, video_path_for


def load_record(submission_path, video_id):
    with open(submission_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if str(obj.get("video_id")) == str(video_id):
                return obj
    raise ValueError(f"video_id {video_id} not found in {submission_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw predicted crop boxes for quick visual inspection.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--sample-step", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    record = load_record(args.submission, args.video_id)
    target_w, target_h = record["targetRatioWH"]
    by_frame = {int(p["frame"]): p["bboxes"] for p in record.get("predictions", [])}
    frames = sorted(by_frame)[:: max(1, args.sample_step)][: args.max_images]
    if not frames:
        print("no predicted frames to visualize")
        return

    video_path = video_path_for(args.video_dir, args.video_id)
    n_frames, _fps, width, height = video_meta(video_path)
    cap = cv2.VideoCapture(video_path)
    saved = 0
    for frame_id in frames:
        if frame_id < 0 or frame_id >= n_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        x, y, w = by_frame[frame_id]
        h = int(round(w * target_h / float(target_w)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"id={args.video_id} frame={frame_id} box=[{x},{y},{w}]",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        out_path = os.path.join(args.out_dir, f"{args.video_id}_{frame_id:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
    cap.release()
    print(f"saved {saved} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
