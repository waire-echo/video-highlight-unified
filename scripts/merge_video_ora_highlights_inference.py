#!/usr/bin/env python3
"""Merge highlights-only per-window inference shards without rerunning a model."""

from __future__ import annotations

import argparse
import hashlib
from decimal import Decimal
import json
from pathlib import Path
from typing import Any


UNION_FIELDS = (
    "model",
    "adapter",
    "adapter_sha256",
    "materialized_root",
    "manifest_sha256",
    "video_root",
    "generation",
    "prompt",
    "assistant_answers_passed_to_model",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def interval_union(
    intervals: list[list[float]],
) -> list[list[float]]:
    ordered = sorted(
        [
            [float(start), float(end)]
            for start, end in intervals
            if Decimal(str(end)) > Decimal(str(start))
        ],
        key=lambda item: (item[0], item[1]),
    )
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or Decimal(str(start)) > Decimal(str(merged[-1][1])):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def merge(args: argparse.Namespace) -> dict[str, Any]:
    runs = [path.resolve() for path in args.runs]
    if not runs:
        raise ValueError("At least one run is required")
    plans = [read_json(run / "plan.json") for run in runs]
    reference = plans[0]
    for run, plan in zip(runs[1:], plans[1:]):
        mismatch = [
            key for key in UNION_FIELDS if plan.get(key) != reference.get(key)
        ]
        if mismatch:
            raise ValueError(f"Run {run} differs on plan fields: {mismatch}")

    source_index = read_json(runs[0] / "source_index.json")
    expected_windows = {
        window["window_id"]: {
            **window,
            "video_id": source["video_id"],
            "duration": source["duration"],
        }
        for source in source_index
        for window in source["windows"]
    }

    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    conflicts: list[str] = []
    for run in runs:
        record_dir = run / "records"
        if not record_dir.is_dir():
            raise FileNotFoundError(record_dir)
        for path in sorted(record_dir.glob("*.json")):
            record = read_json(path)
            window_id = str(record["window_id"])
            if window_id not in expected_windows:
                raise ValueError(f"Unknown window ID in {path}: {window_id}")
            previous = selected.get(window_id)
            if previous:
                previous_path, previous_record = previous
                for key in (
                    "request_sha256",
                    "window_video_sha256",
                    "raw_json_valid",
                    "window_start_sec",
                    "window_end_sec",
                ):
                    if record.get(key) != previous_record.get(key):
                        conflicts.append(
                            f"{window_id}: {previous_path} vs {path} ({key})"
                        )
                if previous_record.get("raw_json_valid") and not record.get(
                    "raw_json_valid"
                ):
                    continue
                if record.get("raw_json_valid") and not previous_record.get(
                    "raw_json_valid"
                ):
                    selected[window_id] = (path, record)
                continue
            selected[window_id] = (path, record)
    if conflicts:
        raise ValueError(
            "Conflicting duplicate window records: " + "; ".join(conflicts[:8])
        )

    missing = sorted(set(expected_windows) - set(selected))
    if missing and not getattr(args, "allow_missing", False):
        raise ValueError(
            f"Missing {len(missing)} windows; first={missing[:8]} last={missing[-8:]}"
        )

    highlights_by_source: dict[str, list[list[float]]] = {
        str(source["video_id"]): [] for source in source_index
    }
    failures = []
    window_rows = []
    for window_id, (path, record) in sorted(selected.items()):
        expected = expected_windows[window_id]
        if (
            record["source_video_id"] != expected["video_id"]
            or record["window_start_sec"] != expected["window_start_sec"]
            or record["window_end_sec"] != expected["window_end_sec"]
        ):
            raise ValueError(f"Window identity mismatch: {window_id}")
        source_id = str(record["source_video_id"])
        item = {
            "window_id": window_id,
            "source_video_id": source_id,
            "window_start_sec": record["window_start_sec"],
            "window_end_sec": record["window_end_sec"],
            "raw_json_valid": bool(record.get("raw_json_valid")),
            "record_path": str(path),
        }
        if not item["raw_json_valid"]:
            failures.append(
                {
                    "video_id": source_id,
                    "window_id": window_id,
                    "error": record.get("validation_error"),
                    "record_path": str(path),
                }
            )
        else:
            intervals = [
                [float(interval["start"]), float(interval["end"])]
                for interval in record.get("source_highlights", [])
            ]
            highlights_by_source[source_id].extend(intervals)
            item["source_highlights"] = intervals
        window_rows.append(item)

    highlight_rows = []
    for source in source_index:
        source_id = str(source["video_id"])
        intervals = interval_union(highlights_by_source[source_id])
        highlight_rows.append(
            {
                "video_id": source_id,
                "duration": source["duration"],
                "highlights": [
                    {"start": start, "end": end}
                    for start, end in intervals
                ],
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "highlights.jsonl", highlight_rows)
    write_jsonl(output / "windows.jsonl", window_rows)
    write_jsonl(output / "failures.jsonl", failures)
    summary = {
        "schema": "video-ora-highlights-merged/v1",
        "state": (
            "incomplete_with_failures"
            if missing
            else ("complete_with_failures" if failures else "complete")
        ),
        "runs": [str(run) for run in runs],
        "source_videos": len(source_index),
        "total_windows": len(expected_windows),
        "included_windows": len(selected),
        "missing_windows": len(missing),
        "valid_windows": len(selected) - len(failures),
        "failed_windows": len(failures),
        "source_videos_with_highlights": sum(
            bool(row["highlights"]) for row in highlight_rows
        ),
        "total_merged_highlights": sum(
            len(row["highlights"]) for row in highlight_rows
        ),
        "failures": failures,
        "missing_window_ids": missing,
        "artifacts_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                             for name in ("highlights.jsonl", "windows.jsonl", "failures.jsonl")},
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(merge(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
