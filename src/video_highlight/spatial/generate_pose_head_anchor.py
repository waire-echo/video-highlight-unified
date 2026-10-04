#!/usr/bin/env python3
"""Refine 16:9 person crops with YOLO pose head and shoulder keypoints.

The input is the current best submission, including its temporal trimming.
All video IDs, target ratios, frame IDs, and crop widths are preserved.  Only
the y coordinate of 16:9 crops in videos hinted as ``person`` may change.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any

from generate_highlight_yolo_center import (
    contiguous_groups,
    interpolate_centers,
    load_hint,
    load_jsonl,
    median_smooth,
    place_fixed_crop,
    read_frame_batches,
    read_video_meta,
    sampled_ids,
)


FACE_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, eyes, ears
SHOULDER_KEYPOINTS = (5, 6)
HIP_KEYPOINTS = (11, 12)


def crop_center(box: list[int], ratio: list[int]) -> tuple[float, float]:
    x, y, width = (int(value) for value in box)
    height = width * ratio[1] / ratio[0]
    return x + width / 2.0, y + height / 2.0


def visible_values(xy: Any, conf: Any, indices: tuple[int, ...], axis: int) -> list[float]:
    values: list[float] = []
    for index in indices:
        if float(conf[index]) >= 0.25:
            values.append(float(xy[index][axis]))
    return values


def candidate_head_anchor(
    result: Any,
    index: int,
    guide: tuple[float, float],
    crop_height: float,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any] | None:
    box = [float(value) for value in result.boxes.xyxy[index].detach().cpu().tolist()]
    confidence = float(result.boxes.conf[index].detach().cpu().item())
    xy = result.keypoints.xy[index].detach().cpu().tolist()
    kp_conf_tensor = result.keypoints.conf
    if kp_conf_tensor is None:
        return None
    kp_conf = kp_conf_tensor[index].detach().cpu().tolist()
    face_y = visible_values(xy, kp_conf, FACE_KEYPOINTS, axis=1)
    face_x = visible_values(xy, kp_conf, FACE_KEYPOINTS, axis=0)
    shoulder_y = visible_values(xy, kp_conf, SHOULDER_KEYPOINTS, axis=1)
    torso_x = visible_values(xy, kp_conf, SHOULDER_KEYPOINTS + HIP_KEYPOINTS, axis=0)
    if not face_y or not shoulder_y:
        return None

    face_mid_y = statistics.median(face_y)
    shoulder_mid_y = statistics.median(shoulder_y)
    if shoulder_mid_y <= face_mid_y:
        return None
    head_top = face_mid_y - 0.45 * (shoulder_mid_y - face_mid_y)
    # Five percent headroom: crop_top = head_top - 0.05 * crop_height.
    anchor_y = head_top + 0.45 * crop_height
    anchor_y = min(max(anchor_y, 0.0), float(frame_height))
    anchor_x = guide[0]

    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_score = min(1.0, math.sqrt(box_area / max(1.0, frame_width * frame_height)))
    face_mid_x = statistics.median(face_x) if face_x else (x1 + x2) / 2.0
    torso_mid_x = statistics.median(torso_x) if torso_x else face_mid_x
    if crop_height >= 0.9 * frame_height:
        distance = abs(torso_mid_x - guide[0]) / max(1.0, frame_width)
    else:
        distance = math.hypot(face_mid_x - guide[0], anchor_y - guide[1]) / max(
            1.0, math.hypot(frame_width, frame_height)
        )
    proximity = max(0.0, 1.0 - distance)
    score = 0.55 * confidence + 0.20 * area_score + 0.25 * proximity
    return {
        "center": [anchor_x, anchor_y],
        "box": box,
        "confidence": confidence,
        "face_y": face_mid_y,
        "shoulder_y": shoulder_mid_y,
        "torso_x": torso_mid_x,
        "selection_score": score,
    }


def detect_pose_group(
    model: Any,
    video_path: Path,
    group: list[int],
    prediction_by_frame: dict[int, dict],
    ratio: list[int],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[dict]:
    keyframes = sampled_ids(group, args.stride)
    detections: list[dict] = []
    crop_height = prediction_by_frame[group[0]]["bboxes"][2] * ratio[1] / ratio[0]
    for batch_ids, batch_frames in read_frame_batches(video_path, keyframes, args.batch_size):
        results = model.predict(
            source=batch_frames,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            verbose=False,
        )
        if len(results) != len(batch_ids):
            raise RuntimeError(f"Pose model returned {len(results)} results for {len(batch_ids)} frames")
        for frame, result in zip(batch_ids, results):
            guide = crop_center(prediction_by_frame[frame]["bboxes"], ratio)
            candidates: list[dict[str, Any]] = []
            if result.boxes is not None and result.keypoints is not None:
                for index in range(len(result.boxes)):
                    candidate = candidate_head_anchor(
                        result, index, guide, crop_height, width, height
                    )
                    if candidate is not None:
                        candidates.append(candidate)
            if not candidates:
                detections.append({"frame": frame, "detected": False})
                continue
            selected = max(candidates, key=lambda item: item["selection_score"])
            selected["frame"] = frame
            selected["detected"] = True
            detections.append(selected)
    return detections


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {str(row["video_id"]): row for row in load_jsonl(path) if "video_id" in row}


def generate(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.ERROR)
    rows = load_jsonl(args.input)
    cached = {} if args.reset_cache else load_cache(args.cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache_handle = args.cache.open("w" if args.reset_cache else "a", encoding="utf-8", newline="\n")
    model = YOLO(args.model)
    output_rows: list[dict] = []
    target_videos = changed_frames = sampled = detected = 0

    try:
        for position, row in enumerate(rows, 1):
            video_id = str(row["video_id"])
            hint = load_hint(args.hints_dir / f"{video_id}.json")
            subject_hint = str(hint.get("subject_hint", hint.get("main_subject", ""))).lower()
            requested_ratio = (16, 9) if args.target_ratio == "16x9" else (9, 16)
            is_target = tuple(row["targetRatioWH"]) == requested_ratio and subject_hint == "person"
            if args.only_video and video_id != args.only_video:
                is_target = False
            if not is_target or not row["predictions"]:
                output_rows.append(row)
                continue

            target_videos += 1
            video_path = args.video_dir / f"{video_id}.mp4"
            frame_count, width, height = read_video_meta(video_path)
            predictions = row["predictions"]
            prediction_by_frame = {int(item["frame"]): item for item in predictions}
            frames = list(prediction_by_frame)
            groups = contiguous_groups(frames)
            cache_row = cached.get(video_id)
            expected_ranges = [(group[0], group[-1]) for group in groups]
            if cache_row is None:
                cached_groups = []
                for group in groups:
                    group_detections = detect_pose_group(
                        model,
                        video_path,
                        group,
                        prediction_by_frame,
                        row["targetRatioWH"],
                        width,
                        height,
                        args,
                    )
                    cached_groups.append(
                        {"first": group[0], "last": group[-1], "detections": group_detections}
                    )
                cache_row = {"video_id": video_id, "groups": cached_groups}
                cache_handle.write(json.dumps(cache_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                cache_handle.flush()
            else:
                cached_ranges = [
                    (group_cache["first"], group_cache["last"])
                    for group_cache in cache_row.get("groups", [])
                ]
                if cached_ranges != expected_ranges:
                    raise ValueError(f"Stale pose cache for video {video_id}; use --reset-cache")

            centers_by_frame: dict[int, tuple[float, float]] = {}
            for group, group_cache in zip(groups, cache_row["groups"]):
                crop_width = int(predictions[0]["bboxes"][2])
                crop_height = crop_width * row["targetRatioWH"][1] / row["targetRatioWH"][0]
                group_detections = []
                for cached_detection in group_cache["detections"]:
                    item = dict(cached_detection)
                    if item.get("detected") and "face_y" in item:
                        center = list(item["center"])
                        if tuple(row["targetRatioWH"]) == (16, 9):
                            center[1] = float(item["face_y"]) + (
                                0.5 - args.face_line_fraction
                            ) * crop_height
                        elif "torso_x" in item:
                            center[0] = float(item["torso_x"])
                        item["center"] = center
                    group_detections.append(item)
                sampled += len(group_detections)
                detected += sum(bool(item.get("detected")) for item in group_detections)
                centers = interpolate_centers(group, group_detections)
                centers_by_frame.update(median_smooth(group, centers, args.smooth_window))

            output_predictions = []
            ratio = row["targetRatioWH"]
            for prediction in predictions:
                frame = int(prediction["frame"])
                original_box = [int(value) for value in prediction["bboxes"]]
                pose_center = centers_by_frame.get(frame)
                if pose_center is None:
                    box = original_box
                else:
                    old_center = crop_center(original_box, ratio)
                    crop_height = original_box[2] * ratio[1] / ratio[0]
                    if tuple(ratio) == (16, 9):
                        max_shift = args.max_shift_fraction * crop_height
                        limited_x = old_center[0]
                        limited_y = min(
                            max(pose_center[1], old_center[1] - max_shift),
                            old_center[1] + max_shift,
                        )
                        if args.upward_only:
                            limited_y = min(limited_y, old_center[1])
                    else:
                        max_shift = args.max_shift_fraction * original_box[2]
                        limited_x = min(
                            max(pose_center[0], old_center[0] - max_shift),
                            old_center[0] + max_shift,
                        )
                        limited_y = old_center[1]
                    box = place_fixed_crop(
                        (limited_x, limited_y),
                        original_box,
                        ratio,
                        width,
                        height,
                        args.pose_blend,
                    )
                derived_height = box[2] * ratio[1] / ratio[0]
                if box[0] < 0 or box[1] < 0 or box[0] + box[2] > width or box[1] + derived_height > height:
                    raise ValueError(f"Invalid pose box for video {video_id} frame {frame}: {box}")
                if box != original_box:
                    changed_frames += 1
                output_predictions.append({"frame": frame, "bboxes": box})
            output_rows.append(
                {"video_id": video_id, "targetRatioWH": ratio, "predictions": output_predictions}
            )
            print(
                f"[{position:03d}/{len(rows)}] id={video_id} frames={len(frames)} "
                f"groups={len(groups)} cache={'hit' if video_id in cached else 'new'}"
            )
    finally:
        cache_handle.close()

    if len(output_rows) != len(rows):
        raise ValueError("Output row count changed")
    for before, after in zip(rows, output_rows):
        if before["video_id"] != after["video_id"] or before["targetRatioWH"] != after["targetRatioWH"]:
            raise ValueError("Index fields changed")
        if [p["frame"] for p in before["predictions"]] != [p["frame"] for p in after["predictions"]]:
            raise ValueError(f"Frame IDs changed for video {before['video_id']}")
        if [p["bboxes"][2] for p in before["predictions"]] != [p["bboxes"][2] for p in after["predictions"]]:
            raise ValueError(f"Crop widths changed for video {before['video_id']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    rate = detected / sampled if sampled else 0.0
    print(
        f"Done: target_videos={target_videos} changed_frames={changed_frames} "
        f"pose_keyframe_rate={rate:.2%} output={args.output} cache={args.cache}"
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    aic_dir = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "predictions_highlight_yolo11m_headroom05_trim025.jsonl")
    parser.add_argument("--video-dir", type=Path, default=aic_dir / "video")
    parser.add_argument("--hints-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=here / "predictions_highlight_pose_head_trim025.jsonl")
    parser.add_argument("--cache", type=Path, default=here / "yolo11m_pose_head_keyframes.jsonl")
    parser.add_argument("--model", default=str(here / "models" / "yolo11m-pose.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--pose-blend", type=float, default=1.0)
    parser.add_argument("--target-ratio", choices=("16x9", "9x16"), default="16x9")
    parser.add_argument(
        "--face-line-fraction",
        type=float,
        default=0.30,
        help="Place the median visible face keypoint at this fraction of crop height.",
    )
    parser.add_argument("--max-shift-fraction", type=float, default=0.35)
    parser.add_argument(
        "--upward-only",
        action="store_true",
        help="Apply pose corrections only when they move the crop window upward.",
    )
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--only-video", default="")
    args = parser.parse_args()
    if args.stride <= 0 or args.batch_size <= 0 or args.imgsz <= 0:
        parser.error("stride, batch size, and image size must be positive")
    if not 0 <= args.pose_blend <= 1:
        parser.error("pose blend must be between 0 and 1")
    if not 0 < args.face_line_fraction < 0.5:
        parser.error("face line fraction must be between 0 and 0.5")
    if args.max_shift_fraction <= 0:
        parser.error("max shift fraction must be positive")
    return args


if __name__ == "__main__":
    generate(parse_args())
