#!/usr/bin/env python3
"""Merge parallel temporal-only runner outputs into one validated delivery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from package_temporal_hints import package_handoff_records
from video_highlight.temporal_merge import (
    atomic_write_json,
    atomic_write_jsonl,
    load_temporal_shards,
    merge_summary,
    validate_merged_handoff,
)


def parse_expected_ids(value: str) -> list[str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "," not in text and "-" in text:
        start_text, end_text = text.split("-", 1)
        start, end = int(start_text), int(end_text)
        if start < 0 or end < start:
            raise ValueError("expected ID range must satisfy 0 <= start <= end")
        return [str(value) for value in range(start, end + 1)]
    ids = [item.strip() for item in text.split(",") if item.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("expected IDs contain duplicates")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge handoff shards plus their adjacent .raw.jsonl files."
    )
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expected-ids",
        default="0-173",
        help="Comma list or inclusive numeric range; empty disables completeness check",
    )
    parser.add_argument("--zip-out", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> dict:
    expected_ids = parse_expected_ids(args.expected_ids)
    records, raw_records = load_temporal_shards(args.shards)
    validate_merged_handoff(records, expected_ids)

    output = Path(args.out)
    raw_output = Path(str(output) + ".raw.jsonl")
    summary_output = Path(str(output) + ".summary.json")
    state_output = Path(str(output) + ".state.json")
    zip_output = Path(args.zip_out) if args.zip_out else output.with_name("temporal_hints.zip")

    atomic_write_jsonl(output, records)
    atomic_write_jsonl(raw_output, raw_records)
    summary = merge_summary(records, args.shards)
    package_summary = package_handoff_records(records, zip_output)
    summary["artifacts"] = {
        "handoff": str(output.resolve()),
        "raw": str(raw_output.resolve()),
        "summary": str(summary_output.resolve()),
        "state": str(state_output.resolve()),
        "legacy_zip": str(zip_output.resolve()),
    }
    summary["legacy_zip"] = package_summary
    state = {
        "schema_version": "temporal-merge-state/v1",
        "source_shards": summary["source_shards"],
        "completed_video_ids": [str(record["video_id"]) for record in records],
        "valid_video_ids": [
            str(record["video_id"]) for record in records if record.get("valid")
        ],
        "error_video_ids": [
            str(record["video_id"]) for record in records if not record.get("valid")
        ],
        "artifacts": summary["artifacts"],
    }
    atomic_write_json(summary_output, summary)
    atomic_write_json(state_output, state)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
