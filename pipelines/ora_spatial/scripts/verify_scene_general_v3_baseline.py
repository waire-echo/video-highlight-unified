#!/usr/bin/env python3
"""Verify the immutable scene_general_v3 baseline before a GPU run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance" / "scene_general_v3_baseline_20261010.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, expected: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing: {path}")
        return
    actual = sha256(path)
    if actual != expected:
        failures.append(f"sha256 mismatch: {path} expected={expected} actual={actual}")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--record", type=Path, default=PROVENANCE)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--check-environment", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    record_path = args.record if args.record.is_absolute() else root / args.record
    record = json.loads(record_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked: list[str] = []

    for relative, expected in record["source_files"].items():
        path = root / relative
        check_file(path, expected, failures)
        checked.append(str(path))

    external = record["external_inputs"]
    for relative, key in (
        ("inputs/baseline.zip", "baseline_zip_sha256"),
        ("inputs/manifest.json", "manifest_json_sha256"),
        ("inputs/timestamps.npz", "timestamps_npz_sha256"),
        ("output/parameter_ledger.json", "parameter_ledger_sha256"),
    ):
        path = root / relative
        check_file(path, external[key], failures)
        checked.append(str(path))

    shots = root / "output" / "shots"
    shot_count = len(list(shots.glob("*.json"))) if shots.is_dir() else 0
    if shot_count != 426:
        failures.append(f"expected 426 shot files, found {shot_count}: {shots}")

    if args.model_dir:
        model_record = json.loads((root / record["model"]["provenance_file"]).read_text(encoding="utf-8"))
        for item in model_record["weight_files"]:
            path = args.model_dir / item["filename"]
            check_file(path, item["sha256"], failures)
            if path.is_file() and path.stat().st_size != item["size_bytes"]:
                failures.append(f"size mismatch: {path}")
            checked.append(str(path))

    if args.check_environment:
        expected_packages = {
            "torch": record["runtime"]["torch"],
            "vllm": record["runtime"]["vllm"],
            "transformers": record["runtime"]["transformers"],
            "numpy": record["runtime"]["numpy"],
            "opencv-python-headless": record["runtime"]["opencv_python_headless"],
            "scipy": record["runtime"]["scipy"],
            "requests": record["runtime"]["requests"],
            "Pillow": record["runtime"]["pillow"],
            "safetensors": record["runtime"]["safetensors"],
            "qwen-vl-utils": record["runtime"]["qwen_vl_utils"],
        }
        for package, expected in expected_packages.items():
            actual = package_version(package)
            if actual != expected:
                failures.append(f"package mismatch: {package} expected={expected} actual={actual}")
        expected_python = record["runtime"]["python"].split(".")[:2]
        actual_python = [str(sys.version_info.major), str(sys.version_info.minor)]
        if actual_python != expected_python:
            failures.append(f"Python mismatch: expected={'.'.join(expected_python)} actual={'.'.join(actual_python)}")

    result = {
        "baseline_id": record["baseline_id"],
        "source_commit": record["source_commit"],
        "checked_files": len(checked),
        "shot_files": shot_count,
        "model_checked": bool(args.model_dir),
        "environment_checked": args.check_environment,
        "pass": not failures,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
