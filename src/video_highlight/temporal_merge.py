"""Merge independently generated temporal handoff shards."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .temporal_handoff_validation import validate_temporal_handoff_records


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            records.append(value)
    return records


def _video_sort_key(record: Mapping) -> tuple[int, object]:
    video_id = str(record.get("video_id", ""))
    try:
        return (0, int(video_id))
    except ValueError:
        return (1, video_id)


def _index_unique(records: Iterable[dict], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        video_id = str(record.get("video_id", "")).strip()
        if not video_id:
            raise ValueError(f"{label}: row is missing video_id")
        if video_id in indexed:
            raise ValueError(f"{label}: duplicate video_id {video_id}")
        indexed[video_id] = record
    return indexed


def load_temporal_shards(shard_paths: Sequence[Path]) -> tuple[list[dict], list[dict]]:
    """Load handoff shards and their adjacent ``.raw.jsonl`` diagnostics."""

    handoff_rows: list[dict] = []
    raw_rows: list[dict] = []
    if not shard_paths:
        raise ValueError("at least one shard is required")
    for value in shard_paths:
        shard = Path(value)
        raw = Path(str(shard) + ".raw.jsonl")
        if not shard.is_file():
            raise FileNotFoundError(str(shard))
        if not raw.is_file():
            raise FileNotFoundError(str(raw))
        handoff_rows.extend(_read_jsonl(shard))
        raw_rows.extend(_read_jsonl(raw))

    handoff_by_id = _index_unique(handoff_rows, "handoff shards")
    raw_by_id = _index_unique(raw_rows, "raw shards")
    if set(handoff_by_id) != set(raw_by_id):
        only_handoff = sorted(set(handoff_by_id) - set(raw_by_id))
        only_raw = sorted(set(raw_by_id) - set(handoff_by_id))
        raise ValueError(
            "handoff/raw video IDs differ: "
            f"handoff_only={only_handoff} raw_only={only_raw}"
        )
    handoff = sorted(handoff_by_id.values(), key=_video_sort_key)
    raw = [raw_by_id[str(record["video_id"])] for record in handoff]
    return handoff, raw


def validate_merged_handoff(
    records: Sequence[dict], expected_ids: Sequence[str] | None = None
) -> None:
    if expected_ids is not None:
        actual = {str(record.get("video_id", "")) for record in records}
        expected = {str(video_id) for video_id in expected_ids}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(
                "merged temporal handoff ID mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
    ok, errors = validate_temporal_handoff_records(records)
    if not ok:
        raise ValueError("merged temporal handoff is invalid: " + "; ".join(errors[:20]))


def atomic_write_jsonl(path: Path, records: Sequence[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()


def atomic_write_json(path: Path, value: Mapping) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    try:
        partial.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()


def merge_summary(records: Sequence[dict], shard_paths: Sequence[Path]) -> dict:
    valid = [record for record in records if record.get("valid")]
    invalid = [record for record in records if not record.get("valid")]
    selected_seconds = sum(
        max(0.0, float(end) - float(start))
        for record in valid
        for start, end in record.get("segments_sec", [])
    )
    return {
        "schema_version": "temporal-merge-summary/v1",
        "source_shards": [str(Path(path).resolve()) for path in shard_paths],
        "videos_written": len(records),
        "valid_videos": len(valid),
        "invalid_videos": len(invalid),
        "invalid_video_ids": [str(record["video_id"]) for record in invalid],
        "segments_total": sum(len(record.get("segments_sec", [])) for record in valid),
        "selected_seconds_total": selected_seconds,
        "contains_spatial_output": False,
    }


__all__ = [
    "atomic_write_json",
    "atomic_write_jsonl",
    "load_temporal_shards",
    "merge_summary",
    "validate_merged_handoff",
]
