"""Validation for the temporal-only handoff consumed by the YOLO stage.

The handoff intentionally contains time ranges and optional semantic hints
only.  Spatial predictions are rejected so the Stage-1 artifact cannot be
mistaken for a competition submission or silently couple to a YOLO backend.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "temporal-handoff/v1"
REQUIRED_FIELDS = {
    "schema_version",
    "video_id",
    "targetRatioWH",
    "fps",
    "frame_count",
    "duration_sec",
    "segments_sec",
    "segments_frame",
    "subject_hint",
    "action_description",
    "source",
    "valid",
    "fallback_used",
}
FORBIDDEN_SPATIAL_FIELDS = {
    "bbox",
    "bboxes",
    "box",
    "boxes",
    "crop_box",
    "crop_boxes",
    "crop_center",
    "frame_prediction",
    "frame_predictions",
    "prediction",
    "predictions",
    "spatial",
    "spatial_records",
    "spatial_summary",
}


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _find_forbidden_fields(value, path: str = "$") -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.lower() in FORBIDDEN_SPATIAL_FIELDS:
                yield child_path
            yield from _find_forbidden_fields(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _find_forbidden_fields(item, f"{path}[{index}]")


def _expected_targets(
    expected_items: Optional[Sequence[Tuple[str, Tuple[int, int]]] | Mapping[str, Sequence[int]]],
) -> Optional[dict[str, Tuple[int, int]]]:
    if expected_items is None:
        return None
    if isinstance(expected_items, Mapping):
        source = expected_items.items()
    else:
        source = expected_items
    result: dict[str, Tuple[int, int]] = {}
    for video_id, target in source:
        result[str(video_id)] = (int(target[0]), int(target[1]))
    return result


def _validate_status(record: dict, label: str, errors: list[str]) -> None:
    if not isinstance(record.get("valid"), bool):
        errors.append(f"{label}: valid must be a boolean")
    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"{label}: source must be a non-empty string")
    fallback = record.get("fallback_used")
    if fallback is not None and (not isinstance(fallback, str) or not fallback.strip()):
        errors.append(f"{label}: fallback_used must be null or a non-empty string")
    for key in ("subject_hint", "action_description"):
        value = record.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{label}: {key} must be null or a non-empty string")
    if "errors" in record:
        value = record["errors"]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{label}: errors must be a list of strings when present")


def _validate_target(record: dict, label: str, errors: list[str]) -> None:
    target = record.get("targetRatioWH")
    if (
        not isinstance(target, list)
        or len(target) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in target)
    ):
        errors.append(f"{label}: targetRatioWH must contain two positive integers")


def _validate_invalid_record(record: dict, label: str, errors: list[str]) -> None:
    if record.get("segments_sec") != [] or record.get("segments_frame") != []:
        errors.append(f"{label}: invalid records must use empty temporal segments")
    for key in ("fps", "frame_count", "duration_sec"):
        value = record.get(key)
        if value is not None and not _is_number(value):
            errors.append(f"{label}: {key} must be finite or null for an invalid record")
    if not record.get("errors"):
        errors.append(f"{label}: invalid records must explain the failure in errors")


def _validate_segments(record: dict, label: str, errors: list[str]) -> None:
    fps = record.get("fps")
    frame_count = record.get("frame_count")
    duration = record.get("duration_sec")
    if not _is_number(fps) or float(fps) <= 0:
        errors.append(f"{label}: fps must be finite and positive")
        return
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count <= 0
    ):
        errors.append(f"{label}: frame_count must be a positive integer")
        return
    if not _is_number(duration) or float(duration) <= 0:
        errors.append(f"{label}: duration_sec must be finite and positive")
        return

    segments_sec = record.get("segments_sec")
    segments_frame = record.get("segments_frame")
    if not isinstance(segments_sec, list) or not isinstance(segments_frame, list):
        errors.append(f"{label}: segments_sec and segments_frame must be lists")
        return
    if not segments_sec:
        errors.append(f"{label}: a valid record must contain at least one segment")
        return
    if len(segments_sec) != len(segments_frame):
        errors.append(f"{label}: second/frame segment counts differ")
        return

    previous_sec_end = -math.inf
    previous_frame_end = -1
    for index, (second_pair, frame_pair) in enumerate(zip(segments_sec, segments_frame)):
        prefix = f"{label}: segment {index}"
        if (
            not isinstance(second_pair, list)
            or len(second_pair) != 2
            or not all(_is_number(item) for item in second_pair)
        ):
            errors.append(f"{prefix}: seconds must be [finite_start, finite_end]")
            continue
        start_sec, end_sec = map(float, second_pair)
        if start_sec < 0 or end_sec <= start_sec or end_sec > float(duration) + 1e-6:
            errors.append(f"{prefix}: seconds are outside [0, duration] or empty")
        if start_sec <= previous_sec_end + 1e-9:
            errors.append(f"{prefix}: second ranges overlap, touch, or are unsorted")
        previous_sec_end = end_sec

        if (
            not isinstance(frame_pair, list)
            or len(frame_pair) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in frame_pair)
        ):
            errors.append(f"{prefix}: frames must be [integer_start, integer_end]")
            continue
        start_frame, end_frame = frame_pair
        if start_frame < 0 or end_frame < start_frame or end_frame >= frame_count:
            errors.append(f"{prefix}: closed frame range is outside the video")
        if start_frame <= previous_frame_end:
            errors.append(f"{prefix}: frame ranges overlap, duplicate, or are unsorted")
        previous_frame_end = end_frame

        expected_start = max(0, min(int(round(start_sec * float(fps))), frame_count - 1))
        expected_end = max(0, min(int(round(end_sec * float(fps))), frame_count - 1))
        if [start_frame, end_frame] != [expected_start, expected_end]:
            errors.append(
                f"{prefix}: frame range does not match rounded seconds "
                f"([{expected_start}, {expected_end}] expected)"
            )


def validate_temporal_handoff_records(
    records: Sequence[dict],
    expected_items: Optional[Sequence[Tuple[str, Tuple[int, int]]] | Mapping[str, Sequence[int]]] = None,
    *,
    require_all_expected: bool = True,
    reject_unexpected: bool = True,
) -> tuple[bool, list[str]]:
    """Validate in-memory ``temporal-handoff/v1`` records."""

    errors: list[str] = []
    if not isinstance(records, (list, tuple)):
        return False, ["handoff records must be a sequence"]
    targets = _expected_targets(expected_items)
    seen: set[str] = set()

    for row_index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            errors.append(f"line {row_index}: record must be a JSON object")
            continue
        video_id = str(record.get("video_id", "")).strip()
        label = f"line {row_index}" + (f" video={video_id}" if video_id else "")
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            errors.append(f"{label}: missing required field(s): {', '.join(missing)}")
        if record.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{label}: schema_version must be {SCHEMA_VERSION!r}")
        if not video_id:
            errors.append(f"{label}: video_id must be non-empty")
        elif video_id in seen:
            errors.append(f"{label}: duplicate video_id")
        seen.add(video_id)

        forbidden = sorted(set(_find_forbidden_fields(record)))
        if forbidden:
            errors.append(
                f"{label}: spatial/submission field(s) are forbidden: {', '.join(forbidden)}"
            )
        _validate_target(record, label, errors)
        _validate_status(record, label, errors)

        if targets is not None and video_id:
            if video_id not in targets:
                if reject_unexpected:
                    errors.append(f"{label}: video_id is not present in the expected index")
            else:
                target = record.get("targetRatioWH")
                if isinstance(target, list) and tuple(target) != targets[video_id]:
                    errors.append(
                        f"{label}: targetRatioWH mismatch; expected {list(targets[video_id])}"
                    )

        if record.get("valid") is True:
            _validate_segments(record, label, errors)
        else:
            _validate_invalid_record(record, label, errors)

    if targets is not None and require_all_expected:
        missing_ids = sorted(set(targets) - seen, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))
        if missing_ids:
            errors.append(f"missing expected video_id(s): {', '.join(missing_ids)}")
    return not errors, errors


def load_temporal_handoff(path: str | Path) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"cannot read {path}: {exc}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: record must be a JSON object")
            continue
        records.append(value)
    return records, errors


def validate_temporal_handoff_file(
    path: str | Path,
    expected_items: Optional[Sequence[Tuple[str, Tuple[int, int]]] | Mapping[str, Sequence[int]]] = None,
    *,
    require_all_expected: bool = True,
    reject_unexpected: bool = True,
) -> tuple[bool, list[str]]:
    records, read_errors = load_temporal_handoff(path)
    ok, validation_errors = validate_temporal_handoff_records(
        records,
        expected_items,
        require_all_expected=require_all_expected,
        reject_unexpected=reject_unexpected,
    )
    errors = read_errors + validation_errors
    return ok and not read_errors, errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a temporal-handoff/v1 JSONL file")
    parser.add_argument("handoff")
    args = parser.parse_args(argv)
    ok, errors = validate_temporal_handoff_file(args.handoff)
    if ok:
        print("temporal handoff validation: OK")
        return 0
    print("temporal handoff validation: FAILED")
    for error in errors:
        print(f"  - {error}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORBIDDEN_SPATIAL_FIELDS",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "load_temporal_handoff",
    "validate_temporal_handoff_file",
    "validate_temporal_handoff_records",
]
