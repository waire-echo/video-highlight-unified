#!/usr/bin/env python3
"""Run frozen v7E Stage 2 from this pipeline's handoff without package fallback.

The teammate release package is an approved 41.09-point reproduction. Its
normal Stage 2 runner validates the supplied Stage 1 file but then also checks
the output against the package's historical Stage 1 fixture. That behavior is
useful for reproduction and unsafe for a live submission because the current
temporal answer must control the selected frames.

This wrapper imports the frozen package's own Stage 2 module, detector assets,
and validation functions, but passes the current handoff directly and writes
an independent run. It never calls the package's Stage 1 model or reads its
reference timestamps as answers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(package: Path, name: str):
    sys.path.insert(0, str(package))
    return importlib.import_module(name)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def build_empty_spatial_outputs(
    stage1: dict,
    source_size: list[int],
    run_fingerprint: str,
) -> tuple[dict, dict, dict]:
    """Build package-compatible files for a valid no-highlight video."""
    if stage1.get("valid") is not False or stage1.get("segments_frame") != []:
        raise ValueError("empty-output helper requires valid=false and no segments")
    if len(stage1.get("targetRatioWH", [])) != 2:
        raise ValueError("empty Stage 1 row has no target ratio")
    if int(stage1.get("frame_count", 0)) <= 0:
        raise ValueError("empty Stage 1 row has no positive frame_count")
    if len(source_size) != 2 or any(int(value) <= 0 for value in source_size):
        raise ValueError("empty Stage 2 row has invalid source size")
    stage2 = {
        "video_id": str(stage1["video_id"]),
        "targetRatioWH": [int(value) for value in stage1["targetRatioWH"]],
        "predictions": [],
    }
    stage1_bytes = json.dumps(
        stage1, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    stage2_bytes = json.dumps(
        stage2, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    done = {
        "video_id": str(stage1["video_id"]),
        "source_sha256": stage1["source_sha256"],
        "source_size": [int(value) for value in source_size],
        "run_fingerprint": run_fingerprint,
        "file_sha256": {
            "stage1.json": hashlib.sha256(stage1_bytes).hexdigest(),
            "stage2.json": hashlib.sha256(stage2_bytes).hexdigest(),
        },
    }
    return stage1, stage2, done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-file", type=Path, required=True)
    parser.add_argument("--gpus", default="2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    package = args.package.resolve()
    videos_dir = args.videos.resolve()
    stage1_file = args.stage1_file.resolve()
    output = args.output.resolve()
    if not (package / "run.py").is_file():
        raise FileNotFoundError(package / "run.py")
    if not stage1_file.is_file():
        raise FileNotFoundError(stage1_file)

    # Import the package's own code and assets; do not reimplement YOLO logic.
    common = load_module(package, "pipeline.common")
    config = read_json(package / "configs/v7e_b_4109.json")
    for item in config["assets"].values():
        path = package / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"packaged asset changed: {path}")
    stage1_rows = read_jsonl(stage1_file)
    by_id = {str(row["video_id"]): row for row in stage1_rows}
    if len(stage1_rows) != 426 or len(by_id) != 426:
        raise ValueError("Stage 1 handoff must contain 426 unique videos")
    expected_ratios = {
        str(row["video_id"]): row["targetRatioWH"] for row in stage1_rows
    }
    if any(ratio not in ([16, 9], [9, 16]) for ratio in expected_ratios.values()):
        raise ValueError("Stage 1 handoff contains an invalid ratio")

    paths = sorted(
        (
            path
            for path in videos_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda path: int(path.stem),
    )
    if len(paths) != 426 or {path.stem for path in paths} != set(by_id):
        raise ValueError("video directory and handoff IDs differ")
    metadata_rows = read_json(package / "reference/source_metadata426.json")
    metadata_by_hash = {
        str(row["source_sha256"]): row for row in metadata_rows
    }
    videos = []
    for path in paths:
        video_id = path.stem
        row = by_id[video_id]
        source_hash = sha256_file(path)
        if source_hash != row["source_sha256"]:
            raise ValueError(f"source video hash mismatch: {video_id}")
        metadata = metadata_by_hash.get(source_hash)
        if metadata is None:
            raise ValueError(f"no target ratio metadata for {video_id}")
        if row["targetRatioWH"] != metadata["target_ratio"]:
            raise ValueError(f"target ratio mismatch: {video_id}")
        actual_meta = metadata
        videos.append(
            {
                "video_id": video_id,
                "path": str(path),
                "sha256": source_hash,
                "ratio": row["targetRatioWH"],
                "source_metadata": actual_meta,
            }
        )

    run_key = common.fingerprint(
        {
            "config": config,
            "code": common.fingerprint(
                {
                    str(path.relative_to(package)): sha256_file(path)
                    for folder in ("pipeline", "modules", "vendor")
                    for path in sorted((package / folder).rglob("*.py"))
                }
            ),
            "videos": videos,
            "stage": "stage2",
            "stage1_sha256": sha256_file(stage1_file),
        }
    )
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        if not (output / "CONFIG.json").is_file():
            raise FileNotFoundError(output / "CONFIG.json")
        if read_json(output / "CONFIG.json")["run_fingerprint"] != run_key:
            raise ValueError("existing spatial output belongs to different inputs")
        if (output / "COMPLETE.json").exists():
            raise FileExistsError("spatial output already complete")
    else:
        output.mkdir(parents=True)
        common.write(
            output / "CONFIG.json",
            {
                "config": config,
                "code_fingerprint": {
                    str(path.relative_to(package)): sha256_file(path)
                    for folder in ("pipeline", "modules", "vendor")
                    for path in sorted((package / folder).rglob("*.py"))
                },
                "videos": videos,
                "stage": "stage2",
                "run_fingerprint": run_key,
                "hardware_gpus": args.gpus.split(","),
                "stage1_file": str(stage1_file),
                "stage1_sha256": sha256_file(stage1_file),
                "source_reference_role": "metadata only; timestamps come from stage1_file",
            },
        )

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("invalid --gpus")
    worker_videos = [
        video for video in videos if video["source_metadata"]["frame_count"]
    ]
    shards = [[] for _ in gpu_ids]
    for index, video in enumerate(worker_videos):
        shards[index % len(shards)].append(video)
    jobs = []
    logs = []
    try:
        first_by_id = {str(row["video_id"]): row for row in stage1_rows}
        for video in videos:
            first = first_by_id[video["video_id"]]
            if first.get("segments_frame") == [] and first.get("valid") is False:
                folder = output / "per_video" / video["video_id"]
                folder.mkdir(parents=True, exist_ok=True)
                size = video["source_metadata"]["source_size"]
                first, second, done = build_empty_spatial_outputs(
                    first, size, run_key
                )
                common.write(folder / "stage1.json", first)
                common.write(folder / "stage2.json", second)
                common.write(folder / "DONE.json", done)
        for index, shard in enumerate(shards):
            if not shard:
                continue
            plan_path = output / f"worker_{index:02d}_plan.json"
            common.write(
                plan_path,
                {
                    "config": config,
                    "videos": shard,
                    "output": str(output),
                    "stage": "stage2",
                    "stage1_file": str(stage1_file),
                    "resume": args.resume,
                    "code_fingerprint": read_json(output / "CONFIG.json")[
                        "code_fingerprint"
                    ],
                    "run_fingerprint": run_key,
                },
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[index]
            log = (output / f"worker_{index:02d}_{int(time.time())}.log").open(
                "x", encoding="utf-8"
            )
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    str(package / "run.py"),
                    "--worker-plan",
                    str(plan_path),
                ],
                cwd=package,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            jobs.append(process)
            print(
                f"Worker {index}: GPU {gpu_ids[index]}, {len(shard)} videos, "
                f"log={log.name}",
                flush=True,
            )
        while any(process.poll() is None for process in jobs):
            if any(process.poll() not in (None, 0) for process in jobs):
                raise RuntimeError("spatial worker failed; inspect worker log")
            common.write(
                output / "status.json",
                {
                    "state": "RUNNING",
                    "completed": len(
                        list((output / "per_video").glob("*/DONE.json"))
                    ),
                    "videos": len(videos),
                },
            )
            time.sleep(2)
        if any(process.returncode != 0 for process in jobs):
            raise RuntimeError("spatial worker failed; inspect worker log")
    except BaseException:
        for process in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise
    finally:
        for log in logs:
            log.close()

    stage1_rows = read_jsonl(stage1_file)
    first_by_id = {str(row["video_id"]): row for row in stage1_rows}
    first_rows, second_rows = [], []
    empty_ids = set()
    for video in videos:
        folder = output / "per_video" / video["video_id"]
        first = read_json(folder / "stage1.json")
        second = read_json(folder / "stage2.json")
        size = read_json(folder / "DONE.json")["source_size"]
        if first != first_by_id[video["video_id"]]:
            raise ValueError(
                f"Stage 2 did not preserve the supplied Stage 1 row: {video['video_id']}"
            )
        if (first["source_sha256"], first["targetRatioWH"]) != (
            video["sha256"],
            video["ratio"],
        ):
            raise ValueError(f"Stage 1 identity mismatch: {video['video_id']}")
        if first.get("segments_frame") == [] and first.get("valid") is False:
            if (first["video_id"], first["targetRatioWH"], first["frame_count"]) != (
                video["video_id"],
                video["ratio"],
                video["source_metadata"]["frame_count"],
            ):
                raise ValueError(
                    f"empty Stage 1 identity mismatch: {video['video_id']}"
                )
            empty_ids.add(str(video["video_id"]))
        else:
            common.validate_stage1(first)
        common.validate_stage2(second, first, *size)
        first_rows.append(first)
        second_rows.append(second)
    for name, values in (
        ("stage1.jsonl", first_rows),
        ("predictions_spatial.jsonl", second_rows),
    ):
        path = output / name
        if path.exists():
            path.unlink()
        with path.open("x", encoding="utf-8") as handle:
            for row in values:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )
    with __import__("zipfile").ZipFile(
        output / "submission.zip",
        "x",
        compression=__import__("zipfile").ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        info = __import__("zipfile").ZipInfo(
            "predictions_spatial.jsonl", date_time=(2026, 10, 23, 0, 0, 0)
        )
        info.compress_type = __import__("zipfile").ZIP_DEFLATED
        archive.writestr(
            info,
            (output / "predictions_spatial.jsonl").read_bytes(),
            compresslevel=6,
        )
    with __import__("zipfile").ZipFile(output / "submission.zip") as archive:
        if (
            archive.namelist() != ["predictions_spatial.jsonl"]
            or archive.testzip() is not None
        ):
            raise ValueError("spatial ZIP verification failed")
    result = {
        "status": "COMPLETE_AND_STRUCTURALLY_VALID",
        "videos": len(videos),
        "empty_highlights": len(empty_ids),
        "groups": sum(len(row["segments_frame"]) for row in first_rows),
        "frames": sum(
            last - first + 1
            for row in first_rows
            for first, last in row["segments_frame"]
        ),
        "run_fingerprint": run_key,
        "stage1_sha256": sha256_file(output / "stage1.jsonl"),
        "spatial_sha256": sha256_file(output / "predictions_spatial.jsonl"),
        "submission_zip_sha256": sha256_file(output / "submission.zip"),
        "stage1_source": str(stage1_file),
        "package_reference_timestamps_used": False,
    }
    common.write(output / "COMPLETE.json", result)
    common.write(output / "status.json", {"state": "COMPLETE", **result})
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
