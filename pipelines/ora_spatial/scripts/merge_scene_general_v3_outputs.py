#!/usr/bin/env python3
"""Merge disjoint scene_general_v3 runs into one validated 426-video result."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile


ROOT = Path(os.environ.get("ORA_SPATIAL_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / "src"))
from common import export, load_rows, read_json, sha, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "inputs" / "baseline.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "inputs" / "manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_rows(args.baseline)
    manifest = read_json(args.manifest)
    rows_by_id = {}
    sources = []
    for run in args.run:
        result = run / "SPATIAL_temporal_v3.jsonl"
        status_path = run / "status.json"
        status = read_json(status_path)
        if status.get("status") != "complete":
            raise RuntimeError(f"run is not complete: {run}")
        rows = load_rows(result)
        for row in rows:
            video_id = str(row["video_id"])
            if video_id in rows_by_id:
                raise ValueError(f"duplicate video ID across runs: {video_id}")
            rows_by_id[video_id] = row
        sources.append({
            "run": str(run),
            "videos": len(rows),
            "jsonl_sha256": sha(result),
            "frozen_sha256": sha(run / "frozen.json"),
            "bundle_sha256": status.get("bundle_sha256"),
        })

    expected = [str(row["video_id"]) for row in baseline]
    missing = [video_id for video_id in expected if video_id not in rows_by_id]
    extra = sorted(set(rows_by_id) - set(expected))
    if missing or extra:
        raise ValueError(f"coverage mismatch: missing={missing} extra={extra}")

    merged = [rows_by_id[video_id] for video_id in expected]
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "SPATIAL_temporal_v3_full_426"
    validation = export(stem, merged, baseline, manifest, preserve_width=False)
    report = {
        "pipeline": "scene_general_v3",
        "videos": len(merged),
        "sources": sources,
        "validation": validation,
        "jsonl_sha256": sha(stem.with_suffix(".jsonl")),
        "submission_zip_sha256": sha(stem.with_suffix(".zip")),
    }
    write_json(args.output / "merge_report.json", report)

    bundle = args.output / "scene_general_v3_full_426_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (
            stem.with_suffix(".jsonl"),
            stem.with_suffix(".zip"),
            stem.with_suffix(".validation.json"),
            args.output / "merge_report.json",
        ):
            archive.write(path, path.name)
    report["result_bundle_sha256"] = sha(bundle)
    write_json(args.output / "merge_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
