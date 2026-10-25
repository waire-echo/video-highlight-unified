#!/usr/bin/env python3
"""Run spatial detection and cropping from the current temporal handoff.

The wrapper imports the spatial module, detector assets, and validators from
the configured package. Selected frames come from the supplied handoff, and
outputs are written to an independent run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_highlight.submission_recovery import SPATIAL_FAILURE_POLICY, center_box, repair_spatial_row, runtime_policy


def python_code_fingerprint(package: Path, folder_names: tuple[str, ...]):
    return {
        str(path.relative_to(package)): sha256_file(path)
        for folder in folder_names
        for path in sorted((package / folder).rglob("*.py"))
    }


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


def save_video(common, folder, first, second, size, run_key, *, state="ok", reason=None, repaired_frames=0):
    common.validate_stage2(second, first, *size)
    common.write(folder / "stage1.json", first)
    common.write(folder / "stage2.json", second)
    common.write(folder / "DONE.json", {
        "video_id": str(first["video_id"]), "source_sha256": first["source_sha256"],
        "source_size": size, "run_fingerprint": run_key,
        "state": state, "reason": reason, "repaired_frames": repaired_frames,
        "file_sha256": {name: sha256_file(folder / name) for name in ("stage1.json", "stage2.json")}})


def stop_worker(process):
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def completed_video(folder, video, run_key):
    done_path = folder / "DONE.json"
    if not done_path.is_file():
        return False
    try:
        done = read_json(done_path)
    except (ValueError, OSError):
        return False
    if done.get("run_fingerprint") != run_key or done.get("source_sha256") != video["sha256"]:
        raise ValueError("Completed video belongs to different input identity")
    return all((folder / name).is_file() and sha256_file(folder / name) == done.get("file_sha256", {}).get(name)
               for name in ("stage1.json", "stage2.json"))


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
        "state": stage1.get("temporal_status", "no_highlights"),
        "reason": stage1.get("fallback_reason"),
        "file_sha256": {
            "stage1.json": hashlib.sha256(stage1_bytes).hexdigest(),
            "stage2.json": hashlib.sha256(stage2_bytes).hexdigest(),
        },
    }
    return stage1, stage2, done


def run_worker(plan_path: Path) -> int:
    """Stage 2 only: no historical Stage 1 models, answers or video table."""
    plan = read_json(plan_path)
    runtime_policy("spatial_failure_policy", plan.get("failure_policy"))
    package = Path(plan["package"])
    output = Path(plan["output"])
    os.environ.update(YOLO_AUTOINSTALL="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      YOLO_CONFIG_DIR=str(output / ".runtime/ultralytics"),
                      MPLCONFIGDIR=str(output / ".runtime/matplotlib"))
    common = load_module(package, "pipeline.common")
    actual_code = python_code_fingerprint(package, ("pipeline", "modules", "vendor"))
    if common.fingerprint(actual_code) != plan["code_fingerprint"]:
        raise ValueError("Spatial worker source code changed after planning")
    import cv2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 requires a supported NVIDIA GPU")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    config = plan["config"]
    stage2 = common.entrypoint(config["stage2"]["entrypoint"])
    first_by_id = {str(row["video_id"]): row for row in read_jsonl(Path(plan["stage1_file"]))}
    for video in plan["videos"]:
        folder = output / "per_video" / video["video_id"]
        folder.mkdir(parents=True, exist_ok=True)
        done_path = folder / "DONE.json"
        if plan["resume"] and completed_video(folder, video, plan["run_fingerprint"]):
            continue
        first = first_by_id[video["video_id"]]
        cap = cv2.VideoCapture(video["path"])
        try:
            size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != first["frame_count"]:
                raise ValueError("Decoded source frame count differs from the prepared input")
        finally:
            cap.release()
        common.validate_stage1(first)
        if sha256_file(Path(video["path"])) != video["sha256"]:
            raise ValueError("Source changed during Stage 2")
        context = {"root": package, "config": config, "video_dir": folder,
                   "resume": plan["resume"], "code_fingerprint": plan["code_fingerprint"],
                   "historical_detector_profile": {}}
        status_path = Path(plan.get("worker_status", output / "worker_status.json"))
        common.write(status_path, {"state": "RUNNING", "video_id": video["video_id"],
                                   "started_monotonic": time.monotonic()})
        state, reason, repaired = "ok", None, []
        try:
            second = stage2(context, video, first, size)
            fixed, repaired = repair_spatial_row(second, first, size)
            if repaired or second.get("predictions") != fixed["predictions"]:
                common.write(folder / "stage2.raw.json", second)
                state, reason = "repaired", "invalid_or_missing_spatial_boxes"
            second = fixed
            common.validate_stage2(second, first, *size)
        except Exception as error:
            state, reason = "fallback", f"{type(error).__name__}: {error}"
            second, repaired = repair_spatial_row({}, first, size)
            common.write(folder / "ERROR.json", {"state": state, "error": reason})
        save_video(common, folder, first, second, size, plan["run_fingerprint"],
                   state=state, reason=reason, repaired_frames=len(repaired))
        if state != "ok":
            print(json.dumps({"event": "VIDEO_RECOVERED", "video_id": video["video_id"],
                              "state": state, "error": reason, "policy": SPATIAL_FAILURE_POLICY}), flush=True)
        common.write(status_path, {"state": "IDLE", "video_id": video["video_id"]})
        print(json.dumps({"event": "VIDEO_COMPLETE", "video_id": video["video_id"]}), flush=True)
    return 0


def main() -> int:
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; stopping owned workers")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--package-config", default="configs/v7e_b_4109.json")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage1-file", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--failure-policy", choices=("error", "center"), default=SPATIAL_FAILURE_POLICY,
                        help=argparse.SUPPRESS)
    parser.add_argument("--video-timeout-sec", type=float, default=1800)
    args = parser.parse_args()
    args.failure_policy = runtime_policy("spatial_failure_policy", args.failure_policy)
    if args.worker_plan:
        return run_worker(args.worker_plan)
    if not all((args.package, args.videos, args.output, args.stage1_file)):
        parser.error("--package, --videos, --output and --stage1-file are required")

    package = args.package.resolve()
    videos_dir = args.videos.resolve()
    stage1_file = args.stage1_file.resolve()
    output = args.output.resolve()
    os.environ.update(YOLO_AUTOINSTALL="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      YOLO_CONFIG_DIR=str(output / ".runtime/ultralytics"),
                      MPLCONFIGDIR=str(output / ".runtime/matplotlib"))
    if not (package / "modules/stage2_b.py").is_file():
        raise FileNotFoundError(package / "modules/stage2_b.py")
    if not stage1_file.is_file():
        raise FileNotFoundError(stage1_file)

    # Import the package's own code and assets; do not reimplement YOLO logic.
    common = load_module(package, "pipeline.common")
    config = read_json(package / args.package_config)
    if args.video_timeout_sec <= 0:
        parser.error("--video-timeout-sec must be positive")
    for item in [config["assets"]["yolo"]]:
        path = package / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"packaged asset changed: {path}")
    stage1_rows = read_jsonl(stage1_file)
    by_id = {str(row["video_id"]): row for row in stage1_rows}
    if not stage1_rows or len(by_id) != len(stage1_rows):
        raise ValueError("Stage 1 handoff must contain unique videos")
    expected_ratios = {
        str(row["video_id"]): row["targetRatioWH"] for row in stage1_rows
    }
    if any(len(ratio) != 2 or any(type(x) is not int or x <= 0 for x in ratio) for ratio in expected_ratios.values()):
        raise ValueError("Stage 1 handoff contains an invalid ratio")

    paths = sorted(
        (
            path
            for path in videos_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda path: (not path.stem.isdecimal(), int(path.stem) if path.stem.isdecimal() else path.stem),
    )
    if len(paths) != len(stage1_rows) or {path.stem for path in paths} != set(by_id):
        raise ValueError("video directory and handoff IDs differ")
    if not args.source_metadata:
        raise ValueError("--source-metadata must describe current inputs, not historical package videos")
    metadata_rows = read_json(args.source_metadata)
    metadata_by_id = {str(row["video_id"]): row for row in metadata_rows}
    if len(metadata_by_id) != len(metadata_rows) or set(metadata_by_id) != set(by_id):
        raise ValueError("source metadata IDs differ from the handoff")
    videos = []
    for path in paths:
        video_id = path.stem
        row = by_id[video_id]
        source_hash = sha256_file(path)
        if source_hash != row["source_sha256"]:
            raise ValueError(f"source video hash mismatch: {video_id}")
        metadata = metadata_by_id[video_id]
        if metadata["source_sha256"] != source_hash:
            raise ValueError(f"source metadata hash mismatch: {video_id}")
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

    package_code = python_code_fingerprint(
        package, ("pipeline", "modules", "vendor")
    )
    package_code_fingerprint = common.fingerprint(package_code)
    run_key = common.fingerprint(
        {
            "config": config,
            "code": package_code_fingerprint,
            "videos": videos,
            "stage": "stage2",
            "stage1_sha256": sha256_file(stage1_file),
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
            "recovery_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/video_highlight/submission_recovery.py"),
            "failure_policy": args.failure_policy,
            "video_timeout_sec": args.video_timeout_sec,
        }
    )
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        if not (output / "CONFIG.json").is_file():
            raise FileNotFoundError(output / "CONFIG.json")
        if read_json(output / "CONFIG.json")["run_fingerprint"] != run_key:
            raise ValueError("existing spatial output belongs to different inputs")
    else:
        output.mkdir(parents=True)
        common.write(
            output / "CONFIG.json",
            {
                "config": config,
                "code_fingerprint": package_code,
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
    worker_videos = [video for video in videos if by_id[video["video_id"]].get("segments_frame")
                     and not completed_video(output / "per_video" / video["video_id"], video, run_key)]
    if worker_videos:
        # Deployment failures must not be disguised as all-video fallbacks.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        import cv2
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(gpu_ids):
            raise RuntimeError("No usable CUDA GPU; refusing an all-empty/fallback deployment")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        detector_probe = YOLO(str(package / config["assets"]["yolo"]["path"]))
        del detector_probe
        common.entrypoint(config["stage2"]["entrypoint"])
    shards = [[] for _ in gpu_ids]
    for index, video in enumerate(worker_videos):
        shards[index % len(shards)].append(video)
    jobs = []
    logs = []
    worker_statuses = []
    failures = []
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
            worker_status = output / f"worker_{index:02d}_status.json"
            common.write(worker_status, {"state": "STARTING", "started_monotonic": time.monotonic()})
            common.write(
                plan_path,
                {
                    "package": str(package),
                    "config": config,
                    "videos": shard,
                    "output": str(output),
                    "stage": "stage2",
                    "stage1_file": str(stage1_file),
                    "resume": args.resume,
                    "code_fingerprint": package_code_fingerprint,
                    "run_fingerprint": run_key,
                    "failure_policy": args.failure_policy,
                    "worker_status": str(worker_status),
                },
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[index]
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            log = (output / f"worker_{index:02d}_{int(time.time())}.log").open(
                "x", encoding="utf-8"
            )
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    str(Path(__file__).resolve()),
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
            worker_statuses.append(worker_status)
            print(
                f"Worker {index}: GPU {gpu_ids[index]}, {len(shard)} videos, "
                f"log={log.name}",
                flush=True,
            )
        while any(process.poll() is None for process in jobs):
            for process, path in zip(jobs, worker_statuses):
                status = read_json(path)
                if process.poll() is None and status.get("state") in ("STARTING", "RUNNING") and (
                        time.monotonic() - status["started_monotonic"] > args.video_timeout_sec):
                    stop_worker(process)
                    failures.append({"worker": str(path), "reason": "video_timeout", "video_id": status.get("video_id")})
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
        failures.extend({"worker": str(path), "return_code": process.returncode}
                        for process, path in zip(jobs, worker_statuses) if process.returncode != 0)
        if failures:
            print(json.dumps({"warning": "spatial_worker_failed_recovering", "worker_failures": failures,
                              "policy": SPATIAL_FAILURE_POLICY}), flush=True)
        for video in videos:
            folder = output / "per_video" / video["video_id"]
            if not completed_video(folder, video, run_key):
                folder.mkdir(parents=True, exist_ok=True)
                if (folder / "stage2.json").is_file():
                    os.replace(folder / "stage2.json", folder / f"stage2.before-recovery-{uuid.uuid4().hex}.json")
                first = by_id[video["video_id"]]
                second, repaired = repair_spatial_row({}, first, video["source_metadata"]["source_size"])
                save_video(common, folder, first, second, video["source_metadata"]["source_size"], run_key,
                           state="fallback", reason="worker_crashed_or_timed_out", repaired_frames=len(repaired))
        common.write(output / "RECOVERY.json", {"worker_failures": failures,
                                                "policy": args.failure_policy,
                                                "videos": [read_json(output / "per_video" / v["video_id"] / "DONE.json")
                                                           for v in videos]})
    except BaseException as error:
        for process in jobs:
            stop_worker(process)
        common.write(output / "status.json", {"state": "FAILED", "error": str(error),
                                               "completed": len(list((output / "per_video").glob("*/DONE.json"))),
                                               "videos": len(videos)})
        raise
    finally:
        for log in logs:
            log.close()

    return export_spatial(common, videos, stage1_file, output, run_key, args.failure_policy)


def export_spatial(common, videos, stage1_file, output, run_key, failure_policy):
    """Finalize one video at a time; interrupted ZIPs never block a resume."""
    runtime_policy("spatial_failure_policy", failure_policy)
    first_by_id = {str(row["video_id"]): row for row in read_jsonl(stage1_file)}
    empty_ids = set()
    frames, groups, terminal = 0, 0, []
    temporaries = [output / f".{name}.{uuid.uuid4().hex}.tmp" for name in ("stage1", "spatial")]
    with temporaries[0].open("x", encoding="utf-8") as first_out, temporaries[1].open("x", encoding="utf-8") as second_out:
        for video in videos:
            folder = output / "per_video" / video["video_id"]
            if not completed_video(folder, video, run_key):
                raise ValueError(f"Spatial checkpoint incomplete or changed: {video['video_id']}")
            first, second = read_json(folder / "stage1.json"), read_json(folder / "stage2.json")
            done = read_json(folder / "DONE.json")
            size = done["source_size"]
            if (first != first_by_id[video["video_id"]] or first["source_sha256"] != video["sha256"]
                    or first["targetRatioWH"] != video["ratio"]
                    or first["frame_count"] != video["source_metadata"]["frame_count"]
                    or size != video["source_metadata"]["source_size"]):
                raise ValueError(f"Spatial output changed temporal/source identity: {video['video_id']}")
            if first.get("segments_frame") == [] and first.get("valid") is False:
                empty_ids.add(str(video["video_id"]))
            else:
                common.validate_stage1(first)
            fixed, repaired = repair_spatial_row(second, first, size)
            if fixed != second:
                common.write(folder / f"stage2.before-export-{uuid.uuid4().hex}.json", second)
                save_video(common, folder, first, fixed, size, run_key, state="repaired",
                           reason="final_spatial_validation", repaired_frames=len(repaired))
                second, done = fixed, read_json(folder / "DONE.json")
            common.validate_stage2(second, first, *size)
            first_out.write(json.dumps(first, ensure_ascii=False, separators=(",", ":")) + "\n")
            second_out.write(json.dumps(second, ensure_ascii=False, separators=(",", ":")) + "\n")
            frames += len(second["predictions"])
            groups += len(first["segments_frame"])
            terminal.append({"video_id": video["video_id"], "state": done.get("state", "no_highlights" if not first["segments_frame"] else "ok"),
                             "reason": done.get("reason"), "repaired_frames": done.get("repaired_frames", 0)})
    for temporary, name in zip(temporaries, ("stage1.jsonl", "predictions_spatial.jsonl")):
        os.replace(temporary, output / name)
    temporary_zip = output / f".submission.{uuid.uuid4().hex}.zip"
    with ZipFile(
        temporary_zip,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        info = ZipInfo(
            "predictions_spatial.jsonl", date_time=(2026, 10, 23, 0, 0, 0)
        )
        info.compress_type = ZIP_DEFLATED
        with archive.open(info, "w") as zipped, (output / "predictions_spatial.jsonl").open("rb") as source:
            shutil.copyfileobj(source, zipped)
    with ZipFile(temporary_zip) as archive:
        if (
            archive.namelist() != ["predictions_spatial.jsonl"]
            or archive.testzip() is not None
        ):
            raise ValueError("spatial ZIP verification failed")
    os.replace(temporary_zip, output / "submission.zip")
    result = {
        "status": "COMPLETE_AND_STRUCTURALLY_VALID",
        "videos": len(videos),
        "empty_highlights": len(empty_ids),
        "groups": groups,
        "frames": frames,
        "run_fingerprint": run_key,
        "stage1_sha256": sha256_file(output / "stage1.jsonl"),
        "spatial_sha256": sha256_file(output / "predictions_spatial.jsonl"),
        "submission_zip_sha256": sha256_file(output / "submission.zip"),
        "stage1_source": str(stage1_file),
        "package_reference_timestamps_used": False,
        "fallback_videos": sum(row["state"] == "fallback" for row in terminal),
        "repaired_videos": sum(row["state"] == "repaired" for row in terminal),
    }
    common.write(output / "TERMINAL_STATES.json", {"videos": terminal})
    common.write(output / "COMPLETE.json", result)
    common.write(output / "status.json", {"state": "COMPLETE", **result})
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
