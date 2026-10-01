#!/usr/bin/env python3
"""Validate an AIC highlight JSONL and optionally compare it with a baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def video_meta(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return frame_count, width, height


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--video-dir", type=Path, default=root / "video")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--index", type=Path, required=True, help="Original test index JSON")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.submission)
    index_rows = json.loads(args.index.read_text(encoding="utf-8-sig"))
    expected_ratios = {str(row["video_id"]): row["targetRatioWH"] for row in index_rows}
    if len(expected_ratios) != len(index_rows):
        raise ValueError("Duplicate video IDs in test index")
    if len(rows) != len(expected_ratios):
        raise ValueError(f"Expected {len(expected_ratios)} rows, found {len(rows)}")
    ids = [str(row.get("video_id")) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate video_id rows")
    if set(ids) != set(expected_ratios):
        raise ValueError("Output video IDs differ from the original test index")

    total_frames = total_segments = 0
    invalid: list[str] = []
    for row in rows:
        video_id = str(row["video_id"])
        ratio = row.get("targetRatioWH")
        if ratio not in ([16, 9], [9, 16]):
            invalid.append(f"{video_id}: invalid targetRatioWH={ratio}")
            continue
        if ratio != expected_ratios[video_id]:
            invalid.append(f"{video_id}: ratio differs from original test index")
        frame_count, width, height = video_meta(args.video_dir / f"{video_id}.mp4")
        predictions = row.get("predictions")
        if not isinstance(predictions, list):
            invalid.append(f"{video_id}: predictions is not a list")
            continue
        frames = [int(item.get("frame", -1)) for item in predictions]
        if frames != sorted(set(frames)):
            invalid.append(f"{video_id}: frames are duplicated or unsorted")
        if frames:
            total_segments += 1 + sum(
                frames[index] != frames[index - 1] + 1 for index in range(1, len(frames))
            )
        for item in predictions:
            frame = item.get("frame")
            box = item.get("bboxes")
            if type(frame) is not int or not 0 <= frame < frame_count:
                invalid.append(f"{video_id}: invalid frame={frame}")
                continue
            if (
                not isinstance(box, list)
                or len(box) != 3
                or any(type(value) is not int for value in box)
            ):
                invalid.append(f"{video_id}/{frame}: invalid bboxes={box}")
                continue
            x, y, crop_width = box
            crop_height = crop_width * ratio[1] / ratio[0]
            if (
                x < 0
                or y < 0
                or crop_width <= 0
                or x + crop_width > width
                or y + crop_height > height + 1e-9
                or not all(math.isfinite(value) for value in box)
            ):
                invalid.append(f"{video_id}/{frame}: out-of-bounds bboxes={box}")
        total_frames += len(predictions)

    comparison: dict[str, Any] | None = None
    if args.baseline:
        baseline_rows = load_jsonl(args.baseline)
        baseline = {str(row["video_id"]): row for row in baseline_rows}
        if len(baseline) != len(baseline_rows) or set(baseline) != set(ids):
            raise ValueError("Baseline video IDs differ or are duplicated")
        added = dropped = changed_boxes = changed_videos = 0
        for row in rows:
            video_id = str(row["video_id"])
            if row["targetRatioWH"] != baseline[video_id]["targetRatioWH"]:
                invalid.append(f"{video_id}: ratio differs from temporal baseline")
            before = {
                int(item["frame"]): item["bboxes"]
                for item in baseline[video_id].get("predictions", [])
            }
            after = {
                int(item["frame"]): item["bboxes"] for item in row.get("predictions", [])
            }
            local_added = len(after.keys() - before.keys())
            local_dropped = len(before.keys() - after.keys())
            local_changed = sum(
                before[frame] != after[frame] for frame in before.keys() & after.keys()
            )
            if local_added or local_dropped or local_changed:
                changed_videos += 1
            added += local_added
            dropped += local_dropped
            changed_boxes += local_changed
        comparison = {
            "changed_videos": changed_videos,
            "added_frames": added,
            "dropped_frames": dropped,
            "changed_common_boxes": changed_boxes,
        }
        if added or dropped:
            invalid.append(f"Temporal frame set changed: added={added}, dropped={dropped}")

    payload = {
        "submission": str(args.submission),
        "sha256": sha256(args.submission),
        "videos": len(rows),
        "frames": total_frames,
        "segments": total_segments,
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:20],
        "comparison": comparison,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
