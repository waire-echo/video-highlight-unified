#!/usr/bin/env python3
"""Stabilize highlight crop x coordinates with BoT-SORT track identities.

The temporal predictions, crop widths, and y coordinates are copied exactly
from the input submission.  Only x may change.  Detailed English hints decide
whether the target is one subject or a multi-subject group.  Single targets
use one globally selected track ID; groups use the union center of the best
two/three persistent tracks.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import random
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any

import cv2

from generate_highlight_yolo_center import (
    ANIMAL_CLASSES,
    contiguous_groups,
    load_hint,
    load_jsonl,
    read_video_meta,
    sampled_ids,
)


def allowed_names(hint: dict[str, Any]) -> set[str]:
    description = " ".join(
        str(hint.get(key, "")) for key in ("subject_hint", "description", "action_description")
    ).lower()
    main = str(hint.get("main_subject", "")).lower()
    if main == "person":
        return {"person"}
    if "dog" in description or "collie" in description or "puppy" in description:
        return {"dog"}
    if "cat" in description or "kitten" in description:
        return {"cat"}
    if main in {"animal", "dog", "cat"}:
        return set(ANIMAL_CLASSES)
    return {"person"} | set(ANIMAL_CLASSES)


def target_count(hint: dict[str, Any]) -> int:
    """Infer how many instances of the main subject should define the crop."""
    description = " ".join(
        str(hint.get(key, "")) for key in ("subject_hint", "description", "action_description")
    ).lower()
    main = str(hint.get("main_subject", "")).lower()
    if main == "person":
        if any(token in description for token in ("three men", "three women", "three people")):
            return 3
        if any(
            token in description
            for token in (
                "two men",
                "two women",
                "two people",
                "two hikers",
                "two children",
                "two kids",
                "couple",
                "pair of people",
            )
        ):
            return 2
        if any(token in description for token in ("group of", "group walking", "multiple people")):
            return 3
        return 1
    if main == "animal":
        if any(token in description for token in ("three dogs", "three cats", "three animals")):
            return 3
        if re.search(r"\btwo\b.{0,24}\b(dogs?|cats?|animals?|puppies|kittens)\b", description) or any(
            token in description
            for token in ("two dogs", "two cats", "two animals", "pair of dogs", "pair of cats")
        ):
            return 2
        if any(
            token in description
            for token in ("multiple dogs", "multiple cats", "group of dogs", "group of small dogs")
        ):
            return 3
    return 1


def reset_tracker(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            tracker.reset()
    if predictor is not None and hasattr(predictor, "vid_path"):
        predictor.vid_path = [None for _ in getattr(predictor, "vid_path", [None])]


def model_class_ids(model: Any, names: set[str]) -> list[int]:
    mapping = model.names
    items = mapping.items() if isinstance(mapping, dict) else enumerate(mapping)
    return [int(index) for index, name in items if str(name).lower() in names]


def read_sampled_frames(video_path: Path, frame_ids: list[int]):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    wanted = 0
    current = frame_ids[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current)
    sequential_fallback = False
    try:
        while wanted < len(frame_ids):
            ok, frame = cap.read()
            if not ok:
                if sequential_fallback:
                    raise RuntimeError(f"Decode failed at frame {current}: {video_path}")
                cap.release()
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot reopen video: {video_path}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current = 0
                sequential_fallback = True
                continue
            if current == frame_ids[wanted]:
                yield current, frame
                wanted += 1
            current += 1
    finally:
        cap.release()


def detect_track_group(
    model: Any,
    video_path: Path,
    group: list[int],
    class_ids: list[int],
    width: int,
    height: int,
    prediction_by_frame: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    track_stride: int,
) -> list[dict[str, Any]]:
    reset_tracker(model)
    frame_ids = sampled_ids(group, track_stride)
    samples: list[dict[str, Any]] = []
    for frame_id, image in read_sampled_frames(video_path, frame_ids):
        result = model.track(
            source=image,
            persist=True,
            tracker=str(args.tracker),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            classes=class_ids,
            verbose=False,
        )[0]
        observations: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) and boxes.id is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            track_ids = boxes.id.detach().cpu().tolist()
            for box, confidence, class_id, track_id in zip(
                xyxy, confidences, classes, track_ids
            ):
                x1, y1, x2, y2 = (float(value) for value in box)
                if x2 <= x1 or y2 <= y1:
                    continue
                observations.append(
                    {
                        "track_id": int(track_id),
                        "class_id": int(class_id),
                        "class_name": str(model.names[int(class_id)]).lower(),
                        "confidence": float(confidence),
                        "box": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    }
                )
        base_box = prediction_by_frame[frame_id]["bboxes"]
        base_center_x = float(base_box[0]) + float(base_box[2]) / 2.0
        samples.append(
            {
                "frame": frame_id,
                "base_center_x": base_center_x,
                "observations": observations,
            }
        )
    return samples


def track_ranking(
    samples: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    by_id: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for sample in samples:
        for observation in sample["observations"]:
            by_id.setdefault(int(observation["track_id"]), []).append((sample, observation))
    ranking: list[dict[str, Any]] = []
    sample_count = max(1, len(samples))
    frame_area = max(1.0, float(width * height))
    for track_id, pairs in by_id.items():
        coverage = len(pairs) / sample_count
        confidence = statistics.fmean(observation["confidence"] for _, observation in pairs)
        proximity = statistics.fmean(
            math.exp(
                -4.0
                * abs(observation["center"][0] - sample["base_center_x"])
                / max(1.0, width)
            )
            for sample, observation in pairs
        )
        area = statistics.fmean(
            min(
                1.0,
                math.sqrt(
                    max(0.0, observation["box"][2] - observation["box"][0])
                    * max(0.0, observation["box"][3] - observation["box"][1])
                    / frame_area
                ),
            )
            for _, observation in pairs
        )
        score = 0.45 * coverage + 0.25 * proximity + 0.20 * confidence + 0.10 * area
        ranking.append(
            {
                "track_id": track_id,
                "score": score,
                "coverage": coverage,
                "confidence": confidence,
                "proximity": proximity,
                "area": area,
                "observations": len(pairs),
            }
        )
    ranking.sort(key=lambda item: item["score"], reverse=True)
    return ranking


def interpolate_scalar(points: list[tuple[int, float]], frames: list[int]) -> dict[int, float]:
    points = sorted(points)
    known_frames = [frame for frame, _ in points]
    output: dict[int, float] = {}
    for frame in frames:
        position = bisect.bisect_left(known_frames, frame)
        if position == 0:
            value = points[0][1]
        elif position >= len(points):
            value = points[-1][1]
        else:
            left_frame, left_value = points[position - 1]
            right_frame, right_value = points[position]
            weight = (frame - left_frame) / max(1, right_frame - left_frame)
            value = left_value + (right_value - left_value) * weight
        output[frame] = float(value)
    return output


def selected_center_series(
    group: list[int],
    samples: list[dict[str, Any]],
    selected_ids: list[int],
    smooth_window: int,
) -> dict[int, float]:
    by_id: dict[int, list[tuple[int, dict[str, Any]]]] = {track_id: [] for track_id in selected_ids}
    for sample in samples:
        frame = int(sample["frame"])
        for observation in sample["observations"]:
            track_id = int(observation["track_id"])
            if track_id in by_id:
                by_id[track_id].append((frame, observation))
    tracks: list[dict[str, dict[int, float]]] = []
    for track_id in selected_ids:
        observations = by_id[track_id]
        if not observations:
            continue
        tracks.append(
            {
                "x1": interpolate_scalar(
                    [(frame, item["box"][0]) for frame, item in observations], group
                ),
                "x2": interpolate_scalar(
                    [(frame, item["box"][2]) for frame, item in observations], group
                ),
            }
        )
    if not tracks:
        return {}
    raw = {
        frame: (min(track["x1"][frame] for track in tracks) + max(track["x2"][frame] for track in tracks))
        / 2.0
        for frame in group
    }
    if smooth_window <= 1:
        return raw
    radius = smooth_window // 2
    values = [raw[frame] for frame in group]
    return {
        frame: float(
            statistics.median(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        )
        for index, frame in enumerate(group)
    }


def confident_single_gate(
    group: list[int],
    samples: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    proposed: dict[int, float],
    prediction_by_frame: dict[int, dict[str, Any]],
    min_coverage: float,
    min_score_margin: float,
    max_median_shift_fraction: float,
    min_jump_pixels: float,
    min_jump_fraction: float,
    max_step_ratio: float,
    gate_mode: str,
    min_jerk: float,
    max_jerk_ratio: float,
    jerk_min_coverage: float,
    jerk_min_score_margin: float,
    jerk_max_shift_fraction: float,
) -> tuple[bool, str, dict[str, float]]:
    """Apply a single-track correction only when switching evidence is strong."""
    if not ranking:
        return False, "no_tracks", {}
    best = ranking[0]
    if best["coverage"] < max(0.55, min_coverage):
        return False, "low_coverage", {"coverage": float(best["coverage"])}
    score_margin = best["score"] - ranking[1]["score"] if len(ranking) > 1 else 1.0
    if score_margin < min_score_margin:
        return False, "ambiguous_track", {"score_margin": float(score_margin)}
    if not proposed:
        return False, "no_center_series", {}

    crop_width = float(prediction_by_frame[group[0]]["bboxes"][2])
    baseline = [
        float(prediction_by_frame[frame]["bboxes"][0])
        + float(prediction_by_frame[frame]["bboxes"][2]) / 2.0
        for frame in group
    ]
    candidate = [float(proposed[frame]) for frame in group]
    median_shift_fraction = statistics.median(
        abs(after - before) for before, after in zip(baseline, candidate)
    ) / max(1.0, crop_width)
    if median_shift_fraction > max_median_shift_fraction:
        return False, "large_shift", {"median_shift_fraction": median_shift_fraction}

    baseline_steps = [abs(baseline[index] - baseline[index - 1]) for index in range(1, len(group))]
    candidate_steps = [abs(candidate[index] - candidate[index - 1]) for index in range(1, len(group))]
    baseline_max_step = max(baseline_steps, default=0.0)
    candidate_max_step = max(candidate_steps, default=0.0)
    baseline_jerk = statistics.fmean(
        abs(baseline[index] - 2.0 * baseline[index - 1] + baseline[index - 2])
        for index in range(2, len(group))
    ) if len(group) >= 3 else 0.0
    candidate_jerk = statistics.fmean(
        abs(candidate[index] - 2.0 * candidate[index - 1] + candidate[index - 2])
        for index in range(2, len(group))
    ) if len(group) >= 3 else 0.0

    nearest_sequence: list[int] = []
    nearest_counts: dict[int, int] = {}
    for sample in samples:
        observations = sample["observations"]
        if not observations:
            continue
        nearest = min(
            observations,
            key=lambda item: abs(float(item["center"][0]) - float(sample["base_center_x"])),
        )
        track_id = int(nearest["track_id"])
        nearest_sequence.append(track_id)
        nearest_counts[track_id] = nearest_counts.get(track_id, 0) + 1
    transitions = sum(
        nearest_sequence[index] != nearest_sequence[index - 1]
        for index in range(1, len(nearest_sequence))
    )
    shares = sorted(
        (count / max(1, len(nearest_sequence)) for count in nearest_counts.values()), reverse=True
    )
    second_share = shares[1] if len(shares) > 1 else 0.0

    sharp_jump_fixed = (
        baseline_max_step >= max(min_jump_pixels, min_jump_fraction * crop_width)
        and candidate_max_step <= max_step_ratio * baseline_max_step
    )
    repeated_switch_fixed = (
        transitions >= 2
        and second_share >= 0.15
        and baseline_jerk > 0.10
        and candidate_jerk <= 0.80 * baseline_jerk
        and candidate_max_step <= baseline_max_step
    )
    global_jerk_fixed = (
        best["coverage"] >= jerk_min_coverage
        and score_margin >= jerk_min_score_margin
        and median_shift_fraction <= jerk_max_shift_fraction
        and baseline_jerk >= min_jerk
        and candidate_jerk <= max_jerk_ratio * baseline_jerk
        and candidate_max_step <= baseline_max_step
    )
    metrics = {
        "coverage": float(best["coverage"]),
        "score_margin": float(score_margin),
        "median_shift_fraction": float(median_shift_fraction),
        "baseline_max_step": float(baseline_max_step),
        "candidate_max_step": float(candidate_max_step),
        "baseline_jerk": float(baseline_jerk),
        "candidate_jerk": float(candidate_jerk),
        "nearest_transitions": float(transitions),
        "second_nearest_share": float(second_share),
    }
    if gate_mode in {"jump", "jump-or-jerk"} and sharp_jump_fixed:
        return True, "sharp_jump_fixed", metrics
    if gate_mode in {"jump", "jump-or-jerk"} and repeated_switch_fixed:
        return True, "repeated_switch_fixed", metrics
    if gate_mode in {"jerk", "jump-or-jerk"} and global_jerk_fixed:
        return True, "global_jerk_fixed", metrics
    return False, "no_switch_evidence", metrics


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(row["video_id"]): row for row in load_jsonl(path) if "video_id" in row}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def generate(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import numpy as np
    import torch

    LOGGER.setLevel(logging.ERROR)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    rows = load_jsonl(args.input)
    cached = {} if args.reset_cache else load_cache(args.cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache_handle = args.cache.open("w" if args.reset_cache else "a", encoding="utf-8", newline="\n")
    model = YOLO(args.model)
    output_rows: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    changed_frames = target_videos = unresolved_groups = 0

    try:
        for position, row in enumerate(rows, 1):
            video_id = str(row["video_id"])
            hint = load_hint(args.hints_dir / f"{video_id}.json")
            count = target_count(hint)
            tracking_stride = (
                args.group_track_stride if count > 1 else args.single_track_stride
            ) or args.track_stride
            should_apply = args.apply_mode in {"all", "confident"} or count > 1
            if args.only_video and video_id != args.only_video:
                should_apply = False
            if args.max_videos and position > args.max_videos:
                should_apply = False
            predictions = row.get("predictions", [])
            if not should_apply or not predictions:
                output_rows.append(row)
                continue

            target_videos += 1
            video_path = args.video_dir / f"{video_id}.mp4"
            frame_count, width, height = read_video_meta(video_path)
            prediction_by_frame = {int(item["frame"]): item for item in predictions}
            frames = list(prediction_by_frame)
            if frames != sorted(set(frames)) or frames[-1] >= frame_count:
                raise ValueError(f"Invalid input frames for video {video_id}")
            groups = contiguous_groups(frames)
            names = allowed_names(hint)
            class_ids = model_class_ids(model, names)
            cache_row = cached.get(video_id)
            expected_ranges = [(group[0], group[-1]) for group in groups]
            if cache_row is None:
                cached_groups = []
                for group in groups:
                    samples = detect_track_group(
                        model,
                        video_path,
                        group,
                        class_ids,
                        width,
                        height,
                        prediction_by_frame,
                        args,
                        tracking_stride,
                    )
                    cached_groups.append({"first": group[0], "last": group[-1], "samples": samples})
                cache_row = {
                    "video_id": video_id,
                    "description": hint.get("description", hint.get("subject_hint", "")),
                    "main_subject": hint.get("main_subject", ""),
                    "allowed_names": sorted(names),
                    "track_stride": tracking_stride,
                    "groups": cached_groups,
                }
                cache_handle.write(json.dumps(cache_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                cache_handle.flush()
            else:
                cached_ranges = [
                    (int(group_cache["first"]), int(group_cache["last"]))
                    for group_cache in cache_row.get("groups", [])
                ]
                if cached_ranges != expected_ranges:
                    raise ValueError(f"Stale BoT-SORT cache for video {video_id}; use --reset-cache")
                cached_stride = cache_row.get("track_stride")
                if cached_stride is not None and int(cached_stride) != tracking_stride:
                    raise ValueError(
                        f"Stale BoT-SORT stride for video {video_id}: "
                        f"{cached_stride} != {tracking_stride}; use --reset-cache"
                    )

            centers: dict[int, float] = {}
            group_reports: list[dict[str, Any]] = []
            for group, group_cache in zip(groups, cache_row["groups"]):
                samples = group_cache["samples"]
                ranking = track_ranking(samples, width, height)
                eligible = [item for item in ranking if item["coverage"] >= args.min_coverage]
                selected = eligible[:count]
                if not selected:
                    unresolved_groups += 1
                    group_reports.append(
                        {"first": group[0], "last": group[-1], "selected_ids": [], "ranking": ranking[:5]}
                    )
                    continue
                selected_ids = [int(item["track_id"]) for item in selected]
                proposed = selected_center_series(group, samples, selected_ids, args.smooth_window)
                apply_group = True
                reason = "multi_subject_group" if count > 1 else "all_mode"
                gate_metrics: dict[str, float] = {}
                if count == 1 and args.apply_mode == "confident":
                    apply_group, reason, gate_metrics = confident_single_gate(
                        group,
                        samples,
                        ranking,
                        proposed,
                        prediction_by_frame,
                        args.min_coverage,
                        args.min_score_margin,
                        args.max_median_shift_fraction,
                        args.min_jump_pixels,
                        args.min_jump_fraction,
                        args.max_step_ratio,
                        args.gate_mode,
                        args.min_jerk,
                        args.max_jerk_ratio,
                        args.jerk_min_coverage,
                        args.jerk_min_score_margin,
                        args.jerk_max_shift_fraction,
                    )
                if apply_group:
                    centers.update(proposed)
                group_reports.append(
                    {
                        "first": group[0],
                        "last": group[-1],
                        "selected_ids": selected_ids,
                        "applied": apply_group,
                        "reason": reason,
                        "gate_metrics": gate_metrics,
                        "ranking": ranking[:5],
                    }
                )

            output_predictions: list[dict[str, Any]] = []
            video_changed = 0
            for prediction in predictions:
                frame = int(prediction["frame"])
                original = [int(value) for value in prediction["bboxes"]]
                crop_width = original[2]
                center_x = centers.get(frame)
                if center_x is None or crop_width >= width:
                    box = original
                else:
                    base_center_x = original[0] + crop_width / 2.0
                    blended_center = base_center_x + args.center_blend * (center_x - base_center_x)
                    x = min(max(0, int(round(blended_center - crop_width / 2.0))), width - crop_width)
                    box = [x, original[1], crop_width]
                if box != original:
                    changed_frames += 1
                    video_changed += 1
                output_predictions.append({"frame": frame, "bboxes": box})
            output_rows.append(
                {
                    "video_id": row["video_id"],
                    "targetRatioWH": row["targetRatioWH"],
                    "predictions": output_predictions,
                }
            )
            report.append(
                {
                    "video_id": video_id,
                    "description": hint.get("description", hint.get("subject_hint", "")),
                    "target_count": count,
                    "changed_frames": video_changed,
                    "groups": group_reports,
                }
            )
            print(
                f"[{position}/{len(rows)}] {video_id}: target_count={count} "
                f"changed={video_changed} groups={len(groups)}"
            )
    finally:
        cache_handle.close()

    write_jsonl(args.output, output_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.zip_output:
        args.zip_output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(args.output, arcname=args.output.name)
    print(
        f"Done: target_videos={target_videos} changed_frames={changed_frames} "
        f"unresolved_groups={unresolved_groups} output={args.output} zip={args.zip_output}"
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    aic_dir = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=here / "predictions_detailed_query_boundary_retain975.jsonl"
    )
    parser.add_argument("--video-dir", type=Path, default=aic_dir / "video")
    parser.add_argument(
        "--hints-dir", type=Path, required=True
    )
    parser.add_argument("--model", type=Path, default=here / "models" / "yolo11m.pt")
    parser.add_argument("--tracker", type=Path, default=here / "botsort_reid.yaml")
    parser.add_argument("--output", type=Path, default=here / "predictions_botsort_group_lock.jsonl")
    parser.add_argument("--cache", type=Path, default=here / "botsort_identity_cache.jsonl")
    parser.add_argument("--report", type=Path, default=here / "botsort_group_lock_report.json")
    parser.add_argument("--zip-output", type=Path, default=here / "submission_botsort_group_lock.zip")
    parser.add_argument(
        "--apply-mode", choices=("group-only", "confident", "all"), default="group-only"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--track-stride", type=int, default=2)
    parser.add_argument("--group-track-stride", type=int, default=0)
    parser.add_argument("--single-track-stride", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--center-blend", type=float, default=1.0)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--min-score-margin", type=float, default=0.06)
    parser.add_argument("--max-median-shift-fraction", type=float, default=0.25)
    parser.add_argument("--min-jump-pixels", type=float, default=8.0)
    parser.add_argument("--min-jump-fraction", type=float, default=0.04)
    parser.add_argument("--max-step-ratio", type=float, default=0.75)
    parser.add_argument(
        "--gate-mode", choices=("jump", "jerk", "jump-or-jerk"), default="jump"
    )
    parser.add_argument("--min-jerk", type=float, default=0.15)
    parser.add_argument("--max-jerk-ratio", type=float, default=0.70)
    parser.add_argument("--jerk-min-coverage", type=float, default=0.75)
    parser.add_argument("--jerk-min-score-margin", type=float, default=0.10)
    parser.add_argument("--jerk-max-shift-fraction", type=float, default=0.12)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--only-video", default="")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        args.track_stride < 1
        or args.group_track_stride < 0
        or args.single_track_stride < 0
        or args.smooth_window < 1
    ):
        parser.error("track strides and smooth window must be non-negative/positive")
    bounded = (
        args.center_blend,
        args.min_coverage,
        args.min_score_margin,
        args.max_median_shift_fraction,
        args.min_jump_fraction,
        args.max_step_ratio,
        args.max_jerk_ratio,
        args.jerk_min_coverage,
        args.jerk_min_score_margin,
        args.jerk_max_shift_fraction,
    )
    if any(not 0.0 <= value <= 1.0 for value in bounded):
        parser.error("blend, coverage, score margin, and shift fraction must be in [0, 1]")
    if args.min_jump_pixels < 0 or args.min_jerk < 0:
        parser.error("minimum jump pixels and jerk must be non-negative")
    return args


if __name__ == "__main__":
    generate(parse_args())
