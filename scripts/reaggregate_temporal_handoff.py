#!/usr/bin/env python3
"""Generate temporal A/B variants from an existing window-score raw JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from package_temporal_hints import package_handoff_records
from video_highlight.temporal_merge import atomic_write_json, atomic_write_jsonl
from video_highlight.temporal_reaggregation import (
    PROFILES,
    reaggregate_records,
    summarize_reaggregation,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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
            rows.append(value)
    return rows


def _parse_profiles(value: str) -> list[str]:
    names = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not names:
        raise ValueError("at least one profile is required")
    unknown = sorted(set(names) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}; available={sorted(PROFILES)}")
    return list(dict.fromkeys(names))


def _prediction_sets_from_zip(path: Path) -> Dict[str, set[int]]:
    with zipfile.ZipFile(path) as archive:
        jsonl_names = [name for name in archive.namelist() if name.lower().endswith(".jsonl")]
        if len(jsonl_names) != 1:
            raise ValueError(
                f"reference submission must contain exactly one JSONL, found {jsonl_names}"
            )
        rows = [
            json.loads(line)
            for line in archive.read(jsonl_names[0]).decode("utf-8-sig").splitlines()
            if line.strip()
        ]
    return {
        str(row.get("video_id")): {
            int(item["frame"])
            for item in row.get("predictions") or []
            if isinstance(item, Mapping) and "frame" in item
        }
        for row in rows
    }


def _handoff_frame_sets(records: Iterable[Mapping[str, Any]]) -> Dict[str, set[int]]:
    result: Dict[str, set[int]] = {}
    for row in records:
        frames: set[int] = set()
        for value in row.get("segments_frame") or []:
            if not isinstance(value, Sequence) or len(value) < 2:
                continue
            start, end = sorted((int(value[0]), int(value[1])))
            frames.update(range(start, end + 1))
        result[str(row.get("video_id"))] = frames
    return result


def _reference_agreement(
    records: Sequence[Mapping[str, Any]], reference: Mapping[str, set[int]]
) -> Dict[str, Any]:
    predicted = _handoff_frame_sets(records)
    if set(predicted) != set(reference):
        raise ValueError("reference submission IDs do not match handoff IDs")
    intersections = 0
    unions = 0
    per_video = []
    for video_id in predicted:
        current = predicted[video_id]
        baseline = reference[video_id]
        intersection = len(current & baseline)
        union = len(current | baseline)
        intersections += intersection
        unions += union
        per_video.append(intersection / union if union else 1.0)
    return {
        # This is deliberately labelled agreement, never accuracy or ground truth.
        "note": "weak reference agreement only; the reference is not ground truth",
        "global_frame_iou": intersections / unions if unions else 1.0,
        "per_video_iou_mean": statistics.mean(per_video) if per_video else 0.0,
        "per_video_iou_median": statistics.median(per_video) if per_video else 0.0,
        "videos_iou_below_0_3": sum(value < 0.3 for value in per_video),
        "videos_iou_below_0_5": sum(value < 0.5 for value in per_video),
        "videos_zero_overlap": sum(value <= 1e-12 for value in per_video),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-aggregate saved temporal window curves without loading Qwen or YOLO."
    )
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        default="balanced,conservative,precision",
        help=f"Comma-separated names from: {', '.join(sorted(PROFILES))}",
    )
    parser.add_argument(
        "--reference-submission-zip",
        type=Path,
        default=None,
        help="Optional old submission used only for weak time-axis agreement diagnostics",
    )
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    handoff_path = Path(args.handoff)
    raw_path = Path(args.raw) if args.raw else Path(str(handoff_path) + ".raw.jsonl")
    handoff = _read_jsonl(handoff_path)
    raw = _read_jsonl(raw_path)
    names = _parse_profiles(args.profiles)
    reference = (
        _prediction_sets_from_zip(Path(args.reference_submission_zip))
        if args.reference_submission_zip
        else None
    )
    root = Path(args.out_root)
    summaries: Dict[str, Any] = {}
    for name in names:
        profile = PROFILES[name]
        records, raw_records = reaggregate_records(handoff, raw, profile)
        output_dir = root / name
        output = output_dir / "temporal_handoff.jsonl"
        raw_output = output_dir / "temporal_handoff.jsonl.raw.jsonl"
        summary_output = output_dir / "temporal_handoff.jsonl.summary.json"
        state_output = output_dir / "temporal_handoff.jsonl.state.json"
        zip_output = output_dir / "temporal_hints.zip"
        atomic_write_jsonl(output, records)
        atomic_write_jsonl(raw_output, raw_records)
        summary = summarize_reaggregation(records, profile)
        summary["artifacts"] = {
            "handoff": str(output.resolve()),
            "raw": str(raw_output.resolve()),
            "summary": str(summary_output.resolve()),
            "state": str(state_output.resolve()),
            "legacy_zip": str(zip_output.resolve()),
        }
        summary["legacy_zip"] = package_handoff_records(records, zip_output)
        if reference is not None:
            summary["reference_agreement"] = _reference_agreement(records, reference)
        state = {
            "schema_version": "temporal-reaggregation-state/v1",
            "profile": profile.to_dict(),
            "source_handoff": str(handoff_path.resolve()),
            "source_raw": str(raw_path.resolve()),
            "completed_video_ids": [str(row["video_id"]) for row in records],
            "valid_video_ids": [str(row["video_id"]) for row in records if row.get("valid")],
            "error_video_ids": [str(row["video_id"]) for row in records if not row.get("valid")],
            "artifacts": summary["artifacts"],
        }
        atomic_write_json(summary_output, summary)
        atomic_write_json(state_output, state)
        summaries[name] = summary
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        root / "comparison_summary.json",
        {
            "schema_version": "temporal-reaggregation-comparison/v1",
            "source_handoff": str(handoff_path.resolve()),
            "source_raw": str(raw_path.resolve()),
            "profiles": summaries,
        },
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = run(args)
    print(json.dumps(summaries, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
