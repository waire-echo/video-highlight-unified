#!/usr/bin/env python3
"""Move the 35.1-point highlight-center crops toward YOLO-detected subjects.

This is a controlled spatial ablation:
  * video rows and predicted frame IDs are copied unchanged from the supplied
    baseline JSONL;
  * every prediction keeps exactly the same width ``w``;
  * only ``x`` and ``y`` may change;
  * if no allowed subject is detected in a highlight segment, the original
    centered boxes are retained.

YOLO is run on every Nth highlight frame. Selected subject centers are
interpolated to every frame and median-smoothed before crop placement.
Completed per-video keyframe detections are cached so post-processing can be
rerun without another model pass.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any

import cv2


ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


def allowed_classes(subject_hint: str) -> set[str]:
    hint = (subject_hint or "").strip().lower()
    if hint == "person":
        return {"person"}
    if hint == "dog":
        return {"dog"}
    if hint == "cat":
        return {"cat"}
    if hint == "animal":
        return set(ANIMAL_CLASSES)
    return {"person"} | set(ANIMAL_CLASSES)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def load_hint(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def contiguous_groups(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    groups: list[list[int]] = [[frames[0]]]
    for frame in frames[1:]:
        if frame == groups[-1][-1] + 1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def sampled_ids(group: list[int], stride: int) -> list[int]:
    ids = list(range(group[0], group[-1] + 1, stride))
    if ids[-1] != group[-1]:
        ids.append(group[-1])
    return ids


def read_frame_batches(
    video_path: Path,
    sample_ids: list[int],
    batch_size: int,
):
    """Decode a monotonically increasing sample sequence and yield small batches."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_ids[0])
    wanted_index = 0
    current = sample_ids[0]
    batch_frames: list[Any] = []
    batch_ids: list[int] = []
    last = sample_ids[-1]
    retried_from_start = False
    try:
        while current <= last and wanted_index < len(sample_ids):
            ok, frame = cap.read()
            if not ok:
                # Some MP4 files report the correct frame count and decode
                # fully from frame zero, but an OpenCV seek can land on a bad
                # keyframe and stop early. Resume the remaining sample IDs by
                # reopening from frame zero once.
                if not retried_from_start and sample_ids[0] > 0:
                    cap.release()
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot reopen video: {video_path}")
                    current = 0
                    retried_from_start = True
                    continue
                raise RuntimeError(f"Decode failed at frame {current}: {video_path}")
            if current == sample_ids[wanted_index]:
                batch_frames.append(frame)
                batch_ids.append(current)
                wanted_index += 1
                if len(batch_frames) >= batch_size:
                    yield batch_ids, batch_frames
                    batch_frames, batch_ids = [], []
            current += 1
        if wanted_index != len(sample_ids):
            raise RuntimeError(
                f"Decoded only {wanted_index}/{len(sample_ids)} requested frames: {video_path}"
            )
        if batch_frames:
            yield batch_ids, batch_frames
    finally:
        cap.release()


def choose_candidate(
    result: Any,
    names: dict[int, str],
    allowed: set[str],
    width: int,
    height: int,
    previous_center: tuple[float, float] | None,
) -> dict | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    diagonal = max(1.0, math.hypot(width, height))
    frame_center = (width / 2.0, height / 2.0)
    candidates: list[dict] = []
    for box, confidence, class_id in zip(xyxy, confidences, classes):
        class_name = str(names[int(class_id)]).lower()
        if class_name not in allowed:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width <= 0 or box_height <= 0:
            continue
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        area_score = min(1.0, math.sqrt((box_width * box_height) / max(1.0, width * height)))
        if previous_center is None:
            distance = math.hypot(center[0] - frame_center[0], center[1] - frame_center[1]) / diagonal
            centrality = max(0.0, 1.0 - distance * 2.0)
            selection_score = 0.60 * float(confidence) + 0.25 * area_score + 0.15 * centrality
        else:
            distance = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) / diagonal
            proximity = math.exp(-5.0 * distance)
            selection_score = 0.55 * float(confidence) + 0.20 * area_score + 0.25 * proximity
        candidates.append(
            {
                "center": [center[0], center[1]],
                "box": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "class_name": class_name,
                "selection_score": float(selection_score),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["selection_score"])


def detect_group(
    model: Any,
    video_path: Path,
    group: list[int],
    allowed: set[str],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[dict]:
    keyframe_ids = sampled_ids(group, args.stride)
    detections: list[dict] = []
    previous_center: tuple[float, float] | None = None
    for batch_ids, batch_frames in read_frame_batches(video_path, keyframe_ids, args.batch_size):
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
            raise RuntimeError(f"Model returned {len(results)} results for {len(batch_ids)} frames")
        for frame_id, result in zip(batch_ids, results):
            selected = choose_candidate(
                result,
                model.names,
                allowed,
                width,
                height,
                previous_center,
            )
            if selected is None:
                detections.append({"frame": frame_id, "detected": False})
                continue
            previous_center = (selected["center"][0], selected["center"][1])
            selected["frame"] = frame_id
            selected["detected"] = True
            detections.append(selected)
    return detections


def interpolate_centers(group: list[int], detections: list[dict]) -> dict[int, tuple[float, float]]:
    known = [item for item in detections if item.get("detected")]
    if not known:
        return {}
    known.sort(key=lambda item: item["frame"])
    known_frames = [int(item["frame"]) for item in known]
    centers: dict[int, tuple[float, float]] = {}
    for frame in group:
        position = bisect.bisect_left(known_frames, frame)
        if position == 0:
            center = known[0]["center"]
        elif position >= len(known):
            center = known[-1]["center"]
        else:
            left = known[position - 1]
            right = known[position]
            span = right["frame"] - left["frame"]
            weight = (frame - left["frame"]) / span if span > 0 else 0.0
            center = [
                left["center"][0] + (right["center"][0] - left["center"][0]) * weight,
                left["center"][1] + (right["center"][1] - left["center"][1]) * weight,
            ]
        centers[frame] = (float(center[0]), float(center[1]))
    return centers


def median_smooth(
    group: list[int],
    centers: dict[int, tuple[float, float]],
    window: int,
) -> dict[int, tuple[float, float]]:
    if not centers or window <= 1:
        return centers
    radius = window // 2
    values = [centers[frame] for frame in group]
    smoothed: dict[int, tuple[float, float]] = {}
    for index, frame in enumerate(group):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        xs = [value[0] for value in values[start:end]]
        ys = [value[1] for value in values[start:end]]
        smoothed[frame] = (float(statistics.median(xs)), float(statistics.median(ys)))
    return smoothed


def adjust_person_16x9_centers(
    detections: list[dict],
    target_ratio: list[int],
    crop_width: int,
    anchor_limit: float,
) -> list[dict]:
    """Move tall-person anchors from the full-body center toward head/torso.

    For a short 16:9 crop cut from a portrait source, the center of a tall
    full-body detection often lands on the chest or abdomen.  Keep the normal
    box center for smaller people, but cap the vertical anchor distance from
    the detected box top as a fraction of the crop height.
    """
    if tuple(target_ratio) != (16, 9) or anchor_limit <= 0:
        return detections
    crop_height = crop_width * target_ratio[1] / target_ratio[0]
    adjusted: list[dict] = []
    for detection in detections:
        item = dict(detection)
        if item.get("detected") and item.get("class_name") == "person" and item.get("box"):
            x1, y1, x2, y2 = (float(value) for value in item["box"])
            box_height = max(0.0, y2 - y1)
            center = list(item["center"])
            center[1] = y1 + min(box_height / 2.0, anchor_limit * crop_height)
            item["center"] = center
        adjusted.append(item)
    return adjusted


def place_fixed_crop(
    detected_center: tuple[float, float],
    original_box: list[int],
    target_ratio: list[int],
    width: int,
    height: int,
    blend: float,
) -> list[int]:
    original_x, original_y, crop_width = (int(value) for value in original_box)
    target_w, target_h = target_ratio
    crop_height = crop_width * target_h / target_w
    original_center = (original_x + crop_width / 2.0, original_y + crop_height / 2.0)
    center_x = original_center[0] + blend * (detected_center[0] - original_center[0])
    center_y = original_center[1] + blend * (detected_center[1] - original_center[1])
    max_x = max(0, width - crop_width)
    max_y = max(0, math.floor(height - crop_height))
    x = min(max(0, int(round(center_x - crop_width / 2.0))), max_x)
    y = min(max(0, int(round(center_y - crop_height / 2.0))), max_y)
    return [x, y, crop_width]


def scale_crop_about_center(
    original_box: list[int],
    target_ratio: list[int],
    width: int,
    height: int,
    scale: float,
) -> list[int]:
    """Scale a crop while preserving its center, ratio, and image bounds."""
    original_x, original_y, original_width = (int(value) for value in original_box)
    target_w, target_h = target_ratio
    original_height = original_width * target_h / target_w
    center_x = original_x + original_width / 2.0
    center_y = original_y + original_height / 2.0
    crop_width = max(1, math.floor(original_width * scale))
    crop_height = crop_width * target_h / target_w
    max_x = max(0, width - crop_width)
    max_y = max(0, math.floor(height - crop_height))
    x = min(max(0, int(round(center_x - crop_width / 2.0))), max_x)
    y = min(max(0, int(round(center_y - crop_height / 2.0))), max_y)
    return [x, y, crop_width]


def width_scale_for_ratio(target_ratio: list[int], args: argparse.Namespace) -> float:
    ratio = tuple(int(value) for value in target_ratio)
    if ratio == (16, 9):
        return args.width_scale_16x9
    if ratio == (9, 16):
        return args.width_scale_9x16
    raise ValueError(f"Unsupported target ratio: {target_ratio!r}")


def read_video_meta(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return frame_count, width, height


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    cached: dict[str, dict] = {}
    for row in load_jsonl(path):
        if "video_id" in row:
            cached[str(row["video_id"])] = row
    return cached


def validate_against_base(
    base_rows: list[dict], output_rows: list[dict], args: argparse.Namespace
) -> dict[str, int]:
    if len(base_rows) != len(output_rows):
        raise ValueError(f"Row count changed: {len(base_rows)} -> {len(output_rows)}")
    changed = 0
    total = 0
    for base, output in zip(base_rows, output_rows):
        if str(base["video_id"]) != str(output["video_id"]):
            raise ValueError("video_id order changed")
        base_predictions = base["predictions"]
        output_predictions = output["predictions"]
        if len(base_predictions) != len(output_predictions):
            raise ValueError(f"Prediction count changed for video {base['video_id']}")
        scale = width_scale_for_ratio(base["targetRatioWH"], args)
        for before, after in zip(base_predictions, output_predictions):
            total += 1
            if before["frame"] != after["frame"]:
                raise ValueError(f"Frame changed for video {base['video_id']}")
            expected_width = max(1, math.floor(before["bboxes"][2] * scale))
            if expected_width != after["bboxes"][2]:
                raise ValueError(
                    f"Unexpected width for video {base['video_id']} frame {before['frame']}: "
                    f"{after['bboxes'][2]} != {expected_width}"
                )
            if before["bboxes"] != after["bboxes"]:
                changed += 1
    return {"total": total, "changed": changed}


def generate(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    # Ultralytics 8.4 emits the same half-precision deprecation warning for
    # every batch. Keep the run log readable; this does not change inference.
    LOGGER.setLevel(logging.ERROR)

    base_rows = load_jsonl(args.base_predictions)
    if args.max_videos > 0:
        base_rows = base_rows[: args.max_videos]
    cached = {} if args.reset_cache else load_cache(args.cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache_mode = "w" if args.reset_cache else "a"
    cache_handle = args.cache.open(cache_mode, encoding="utf-8", newline="\n")
    model = YOLO(args.model)
    output_rows: list[dict] = []
    videos_with_detections = 0
    detected_keyframes = 0
    sampled_keyframes = 0

    try:
        for position, base in enumerate(base_rows, 1):
            video_id = str(base["video_id"])
            video_path = args.video_dir / f"{video_id}.mp4"
            hint_path = args.hints_dir / f"{video_id}.json"
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing video: {video_path}")
            if not hint_path.is_file():
                raise FileNotFoundError(f"Missing hint: {hint_path}")
            frame_count, width, height = read_video_meta(video_path)
            hint = load_hint(hint_path)
            subject_hint = str(hint.get("subject_hint", hint.get("main_subject", "")))
            allowed = allowed_classes(subject_hint)
            predictions = base.get("predictions", [])
            frames = [int(item["frame"]) for item in predictions]
            if frames != sorted(set(frames)):
                raise ValueError(f"Base frames are duplicated or unsorted for video {video_id}")
            if frames and (frames[0] < 0 or frames[-1] >= frame_count):
                raise ValueError(f"Base frame is out of bounds for video {video_id}")
            groups = contiguous_groups(frames)

            cache_row = cached.get(video_id)
            if cache_row is None:
                cached_groups: list[dict] = []
                for group in groups:
                    detections = detect_group(
                        model,
                        video_path,
                        group,
                        allowed,
                        width,
                        height,
                        args,
                    )
                    cached_groups.append(
                        {"first": group[0], "last": group[-1], "detections": detections}
                    )
                cache_row = {
                    "video_id": video_id,
                    "subject_hint": subject_hint,
                    "allowed_classes": sorted(allowed),
                    "groups": cached_groups,
                }
                cache_handle.write(json.dumps(cache_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                cache_handle.flush()
            else:
                cached_groups = cache_row.get("groups", [])
                expected_ranges = [(group[0], group[-1]) for group in groups]
                cached_ranges = [(group["first"], group["last"]) for group in cached_groups]
                if cached_ranges != expected_ranges:
                    raise ValueError(f"Stale cache ranges for video {video_id}; rerun with --reset-cache")

            centers_by_frame: dict[int, tuple[float, float]] = {}
            video_has_detection = False
            for group, group_cache in zip(groups, cache_row.get("groups", [])):
                detections = group_cache.get("detections", [])
                sampled_keyframes += len(detections)
                detected_count = sum(bool(item.get("detected")) for item in detections)
                detected_keyframes += detected_count
                video_has_detection = video_has_detection or detected_count > 0
                crop_width = int(predictions[0]["bboxes"][2]) if predictions else 1
                detections = adjust_person_16x9_centers(
                    detections,
                    base["targetRatioWH"],
                    crop_width,
                    args.person_16x9_anchor_limit,
                )
                centers = interpolate_centers(group, detections)
                centers = median_smooth(group, centers, args.smooth_window)
                centers_by_frame.update(centers)
            if video_has_detection:
                videos_with_detections += 1

            output_predictions: list[dict] = []
            width_scale = width_scale_for_ratio(base["targetRatioWH"], args)
            for prediction in predictions:
                frame = int(prediction["frame"])
                original_box = [int(value) for value in prediction["bboxes"]]
                working_box = scale_crop_about_center(
                    original_box,
                    base["targetRatioWH"],
                    width,
                    height,
                    width_scale,
                )
                center = centers_by_frame.get(frame)
                if center is None:
                    box = working_box
                else:
                    box = place_fixed_crop(
                        center,
                        working_box,
                        base["targetRatioWH"],
                        width,
                        height,
                        args.center_blend,
                    )
                derived_height = box[2] * base["targetRatioWH"][1] / base["targetRatioWH"][0]
                if box[0] < 0 or box[1] < 0 or box[2] <= 0:
                    raise ValueError(f"Invalid box for video {video_id} frame {frame}: {box}")
                if box[0] + box[2] > width or box[1] + derived_height > height:
                    raise ValueError(f"Out-of-bounds box for video {video_id} frame {frame}: {box}")
                output_predictions.append({"frame": frame, "bboxes": box})

            output_rows.append(
                {
                    "video_id": video_id,
                    "targetRatioWH": base["targetRatioWH"],
                    "predictions": output_predictions,
                }
            )
            print(
                f"[{position:03d}/{len(base_rows)}] id={video_id} hint={subject_hint} "
                f"frames={len(frames)} groups={len(groups)} "
                f"detected={'yes' if video_has_detection else 'no'} "
                f"cache={'hit' if video_id in cached else 'new'}"
            )
    finally:
        cache_handle.close()

    stats = validate_against_base(base_rows, output_rows, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    detection_rate = detected_keyframes / sampled_keyframes if sampled_keyframes else 0.0
    print(
        f"Done: videos={len(output_rows)} frames={stats['total']} changed_boxes={stats['changed']} "
        f"videos_with_detection={videos_with_detections} "
        f"keyframe_detection_rate={detection_rate:.2%} output={args.output} cache={args.cache}"
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    aic_dir = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path, default=here / "predictions_highlight_center.jsonl")
    parser.add_argument("--video-dir", type=Path, default=aic_dir / "video")
    parser.add_argument("--hints-dir", type=Path, default=aic_dir / "highlights_json_174_simple_hints")
    parser.add_argument("--output", type=Path, default=here / "predictions_highlight_yolo_center.jsonl")
    parser.add_argument("--cache", type=Path, default=here / "yolo11s_center_keyframes.jsonl")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--center-blend", type=float, default=1.0)
    parser.add_argument("--width-scale-16x9", type=float, default=1.0)
    parser.add_argument("--width-scale-9x16", type=float, default=1.0)
    parser.add_argument(
        "--person-16x9-anchor-limit",
        type=float,
        default=0.0,
        help=(
            "For 16:9 crops, cap a tall person's vertical center distance from "
            "the detection top to this multiple of crop height; 0 disables it."
        ),
    )
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0, help="Process only the first N videos for testing.")
    args = parser.parse_args()
    if args.stride <= 0 or args.batch_size <= 0 or args.imgsz <= 0:
        parser.error("stride, batch-size and imgsz must be positive")
    if not (0 < args.width_scale_16x9 <= 1 and 0 < args.width_scale_9x16 <= 1):
        parser.error("width scales must be in the interval (0, 1]")
    if args.person_16x9_anchor_limit < 0:
        parser.error("person 16x9 anchor limit must be non-negative")
    if args.smooth_window <= 0 or args.smooth_window % 2 == 0:
        parser.error("--smooth-window must be a positive odd integer")
    if not 0.0 <= args.center_blend <= 1.0:
        parser.error("--center-blend must be between 0 and 1")
    return args


if __name__ == "__main__":
    generate(parse_args())
