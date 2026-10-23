#!/usr/bin/env python3
"""Run the frozen Video-ORA, lenient repair, spatial, and submission pipeline.

This entry point orchestrates existing, already-tested components. It never
trains, retries model inference with different settings, or treats a failed
window as an empty prediction. Every stage is bound to the same 426-video
materialized dataset and LoRA adapter through SHA256 receipts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_highlight.validation import validate_submission


SERVER_ROOT = Path("/media/kubevolume/teacher1/video-highlight")
DEFAULT_MODEL = (
    SERVER_ROOT
    / "models/Video-ORA-4B/01850297d5ab2adaaf130f700ed7cec52993d956"
)
DEFAULT_ADAPTER = (
    SERVER_ROOT
    / "runtime/sft426-doubao237-continue-adapter118-lr3e5-world3-v13-e1-20261023"
    / "trainer/checkpoint-228"
)
DEFAULT_ADAPTER_SHA256 = (
    "37b5d75def0c4c9061efc405495c3d5bcc6ace4b60c509657a68c0d90678c790"
)
DEFAULT_MATERIALIZED = (
    SERVER_ROOT
    / "datasets/sft/highlights-20261022"
    / "data-highlights-window32s-v1style-w8-20261022"
)
DEFAULT_VIDEOS = SERVER_ROOT / "datasets/aic426/videos"
DEFAULT_PYTHON = (
    SERVER_ROOT
    / "envs/vh-sft-capacity-tf519-20261021/bin/python"
)
DEFAULT_SPATIAL_PYTHON = (
    SERVER_ROOT
    / "envs/vh-spatial-ora426-20261021/bin/python"
)
DEFAULT_SPATIAL_PACKAGE = (
    SERVER_ROOT
    / "development/event-review-20261019-v1/repro-packages"
    / "V7E_B_4109_REPRO_20261022_v1"
)
DEFAULT_PIPELINE_DIR = SERVER_ROOT / "runtime/submission-pipeline-new-20261023"
DEFAULT_INDEX = SERVER_ROOT / "datasets/aic426/test_index.normalized.json"
STAGES = (
    "infer",
    "merge",
    "lenient",
    "handoff",
    "spatial",
    "submission",
)
SPATIAL_RUNNER_SOURCE = REPO / "scripts/run_v7e_stage2_submission.py"


class PipelineError(RuntimeError):
    """The run cannot continue without violating its input contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PipelineError(f"{path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise PipelineError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stage_dir(run_root: Path, name: str) -> Path:
    return run_root / f"{STAGES.index(name) + 1:02d}-{name}"


def command_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(SRC)
            + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def run_command(
    *,
    name: str,
    command: list[str],
    output: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    command_dir = (
        output
        / "commands"
        / (
            f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            f"-{os.getpid()}"
        )
    )
    command_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = command_dir / "stdout.log"
    started = time.monotonic()
    with stdout_path.open("x", encoding="utf-8", newline="\n") as output_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output_handle.write(line)
            output_handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    receipt = {
        "stage": name,
        "command": command,
        "cwd": str(REPO),
        "return_code": return_code,
        "seconds": time.monotonic() - started,
        "stdout_log": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
    }
    atomic_json(command_dir / "COMMAND.json", receipt)
    atomic_json(output / "COMMAND.json", receipt)
    if return_code:
        raise PipelineError(
            f"{name} failed with exit code {return_code}; see {stdout_path}"
        )
    return receipt


def verify_paths(args: argparse.Namespace) -> None:
    required = {
        "model": args.model,
        "adapter": args.adapter,
        "materialized": args.materialized,
        "videos": args.videos,
        "index": args.index,
        "spatial_package": args.spatial_package,
        "python": args.python,
        "spatial_python": args.spatial_python,
        "spatial_runner": SPATIAL_RUNNER_SOURCE,
    }
    missing = [
        f"{name}: {path}" for name, path in required.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    if not hasattr(args, "video_max_token_num"):
        args.video_max_token_num = 196
    if not hasattr(args, "video_min_token_num"):
        args.video_min_token_num = 128
    if not hasattr(args, "spatial_arm"):
        args.spatial_arm = "B"
    seed = getattr(args, "seed_strict_merged", None)
    if seed is not None:
        seed_required = {
            "summary.json",
            "windows.jsonl",
            "highlights.jsonl",
            "failures.jsonl",
        }
        seed_missing = sorted(
            str(seed / name)
            for name in seed_required
            if not (seed / name).is_file()
        )
        if seed_missing:
            raise FileNotFoundError("\n".join(seed_missing))
    if args.index.is_dir():
        raise PipelineError("--index must be a single JSON/JSONL index file")
    model_revision = read_json(args.model / "config.json").get("_commit_hash")
    if (
        args.model.name != "01850297d5ab2adaaf130f700ed7cec52993d956"
        or model_revision
        not in (None, "01850297d5ab2adaaf130f700ed7cec52993d956")
    ):
        raise PipelineError(
            f"unexpected base model revision: {model_revision!r}"
        )
    adapter_sha = sha256_file(args.adapter / "adapter_model.safetensors")
    if adapter_sha != args.expected_adapter_sha256:
        raise PipelineError(
            f"unexpected adapter SHA256: {adapter_sha}; "
            f"expected {args.expected_adapter_sha256}"
        )

    manifest = read_json(args.materialized / "manifest.json")
    if manifest.get("answer_contract") != "highlights-only/v1":
        raise PipelineError("unexpected materialized answer contract")
    if manifest.get("source_counts") != {"train": 383, "val": 43}:
        raise PipelineError("materialized source split is not 383/43")
    if manifest.get("training_data_ready") is not True:
        raise PipelineError("materialized windows are not ready")
    if manifest.get("training_started") is not False:
        raise PipelineError("materialized manifest has training_started=true")
    if manifest["samples"].keys() != {"train", "val"}:
        raise PipelineError("unexpected materialized sample splits")
    total_windows = sum(len(manifest["samples"][split]) for split in ("train", "val"))
    if total_windows != 523:
        raise PipelineError(f"expected 523 test windows, found {total_windows}")

    index_rows = read_json(args.index) if args.index.suffix == ".json" else read_jsonl(args.index)
    index_ids = {str(row["video_id"]) for row in index_rows}
    if len(index_rows) != 426 or index_ids != {str(value) for value in range(426)}:
        raise PipelineError("test index must contain exactly video IDs 0..425")
    expected_ratios = {
        str(row["video_id"]): row["targetRatioWH"] for row in index_rows
    }
    if any(ratio not in ([16, 9], [9, 16]) for ratio in expected_ratios.values()):
        raise PipelineError("test index contains an unsupported target ratio")
    if len(list(args.videos.glob("*.mp4"))) != 426:
        raise PipelineError("video directory must contain 426 MP4 files")


def build_pipeline_id(args: argparse.Namespace) -> str:
    return canonical_sha256(pipeline_contract(args))[:16]


def pipeline_contract(args: argparse.Namespace) -> dict[str, Any]:
    model_config = read_json(args.model / "config.json")
    seed = getattr(args, "seed_strict_merged", None)
    video_max_token_num = getattr(args, "video_max_token_num", 196)
    video_min_token_num = getattr(args, "video_min_token_num", 128)
    spatial_arm = getattr(args, "spatial_arm", "B")
    return {
        "schema": "video-ora-submission-pipeline/v1",
        "model": str(args.model.resolve()),
        "model_revision": model_config.get("_commit_hash"),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "materialized": str(args.materialized.resolve()),
        "materialized_manifest_sha256": sha256_file(
            args.materialized / "manifest.json"
        ),
        "videos": str(args.videos.resolve()),
        "index": str(args.index.resolve()),
        "test_index_sha256": sha256_file(args.index),
        "video_names": sorted(path.name for path in args.videos.glob("*.mp4")),
        "inference": {
            "fps": args.fps,
            "max_frames": args.max_frames,
            "max_tokens": args.inference_max_tokens,
            "video_max_token_num": video_max_token_num,
            "video_min_token_num": video_min_token_num,
            "temperature": 0,
            "seed": 42,
        },
        "lenient_end_tolerance_sec": args.end_tolerance_sec,
        "strict_seed": (
            {
                "path": str(seed.resolve()),
                "summary_sha256": sha256_file(seed / "summary.json"),
                "windows_sha256": sha256_file(seed / "windows.jsonl"),
                "highlights_sha256": sha256_file(seed / "highlights.jsonl"),
                "failures_sha256": sha256_file(seed / "failures.jsonl"),
            }
            if seed is not None
            else None
        ),
        "spatial_package": str(args.spatial_package.resolve()),
        "spatial_package_manifest_sha256": sha256_file(
            args.spatial_package / "RELEASE_MANIFEST.json"
        ),
        "spatial_generator_sha256": sha256_file(
            args.spatial_package / "vendor/yolo_center.py"
        ),
        "spatial_weights_sha256": sha256_file(
            args.spatial_package / "assets/yolo11m.pt"
        ),
        "spatial_runner": str(SPATIAL_RUNNER_SOURCE.resolve()),
        "spatial_runner_sha256": sha256_file(SPATIAL_RUNNER_SOURCE),
        "pipeline_runner": str(Path(__file__).resolve()),
        "pipeline_runner_sha256": sha256_file(Path(__file__).resolve()),
        "spatial_arm": spatial_arm,
        "training": False,
    }


def legacy_pipeline_id(args: argparse.Namespace) -> str:
    """Reproduce the pre-fix ID so no-op dry-runs remain resumable."""
    model_config = read_json(args.model / "config.json")
    seed = getattr(args, "seed_strict_merged", None)
    return canonical_sha256(
        {
            "schema": "video-ora-submission-pipeline/v1",
            "model_revision": model_config.get("_commit_hash"),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
            "materialized_manifest_sha256": sha256_file(
                args.materialized / "manifest.json"
            ),
            "test_index_sha256": sha256_file(args.index),
            "video_names": sorted(path.name for path in args.videos.glob("*.mp4")),
            "inference": {
                "fps": args.fps,
                "max_frames": args.max_frames,
                "max_tokens": args.inference_max_tokens,
                "video_max_token_num": args.video_max_token_num,
                "video_min_token_num": args.video_min_token_num,
            },
            "lenient_end_tolerance_sec": args.end_tolerance_sec,
            "strict_seed": (
                {
                    "path": str(seed.resolve()),
                    "summary_sha256": sha256_file(seed / "summary.json"),
                    "windows_sha256": sha256_file(seed / "windows.jsonl"),
                    "highlights_sha256": sha256_file(seed / "highlights.jsonl"),
                    "failures_sha256": sha256_file(seed / "failures.jsonl"),
                }
                if seed is not None
                else None
            ),
            "spatial_package_manifest_sha256": sha256_file(
                args.spatial_package / "RELEASE_MANIFEST.json"
            ),
            "spatial_generator_sha256": sha256_file(
                args.spatial_package / "vendor/yolo_center.py"
            ),
            "spatial_weights_sha256": sha256_file(
                args.spatial_package / "assets/yolo11m.pt"
            ),
            "spatial_runner_sha256": sha256_file(SPATIAL_RUNNER_SOURCE),
            "pipeline_runner_sha256": sha256_file(Path(__file__).resolve()),
        }
    )[:16]


def prepare_run(
    args: argparse.Namespace,
    pipeline_id: str,
) -> tuple[Path, dict[str, Any], bool]:
    args.pipeline_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.pipeline_dir / f"run-{pipeline_id}"
    lock = args.pipeline_dir / "CURRENT.json"
    legacy_run = args.pipeline_dir / f"run-{legacy_pipeline_id(args)}"
    if not run_root.exists() and legacy_run.is_dir():
        legacy_contract = read_json(legacy_run / "PIPELINE.json")
        current_contract = pipeline_contract(args)
        ignored_fields = {"created_at"}
        differences = {
            key
            for key in set(legacy_contract) | set(current_contract)
            if key not in ignored_fields
            and legacy_contract.get(key) != current_contract.get(key)
        }
        if not differences:
            run_root = legacy_run
    if lock.exists():
        current = read_json(lock)
        existing = Path(str(current.get("run_root", ""))).resolve()
        if existing != run_root.resolve():
            print(
                "Inputs changed; preserving the previous run and creating a "
                f"new history entry: previous={existing}",
                flush=True,
            )
    contract = {
        **pipeline_contract(args),
        "schema": "video-ora-submission-pipeline-run/v2",
        "pipeline_id": pipeline_id,
        "created_at": utc_now(),
    }
    contract_path = run_root / "PIPELINE.json"
    resumed = contract_path.exists()
    if resumed:
        stored = read_json(contract_path)
        differences = {
            key
            for key in set(stored) | set(contract)
            if key != "created_at" and stored.get(key) != contract.get(key)
        }
        if differences:
            raise PipelineError(
                "existing PIPELINE.json differs from current inputs: "
                + ", ".join(sorted(differences))
            )
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        atomic_json(contract_path, contract)
    atomic_json(
        lock,
        {
            "pipeline_id": pipeline_id,
            "run_root": str(run_root),
            "updated_at": utc_now(),
        },
    )
    return run_root, contract, resumed


def update_state(
    run_root: Path,
    *,
    stage: str,
    state: str,
    detail: str,
    artifact: Path | None = None,
) -> None:
    value = {
        "schema": "video-ora-submission-pipeline-state/v1",
        "state": state,
        "stage": stage,
        "detail": detail,
        "updated_at": utc_now(),
        "pid": os.getpid(),
    }
    if artifact is not None:
        value["artifact"] = str(artifact)
        if artifact.is_file():
            value["artifact_sha256"] = sha256_file(artifact)
    atomic_json(run_root / "STATE.json", value)


def stage_inference(
    args: argparse.Namespace,
    run_root: Path,
    *,
    execute: bool,
) -> Path:
    stage = stage_dir(run_root, "infer")
    output = stage / "run"
    if not execute:
        return output
    if (stage / "COMMAND.json").exists():
        command_receipt = read_json(stage / "COMMAND.json")
        if command_receipt.get("return_code") != 0:
            raise PipelineError("previous inference command failed")
    else:
        command = [
            str(args.python),
            "-B",
            "-u",
            str(REPO / "scripts/infer_video_ora_highlights_adapter_426.py"),
            "--model",
            str(args.model),
            "--adapter",
            str(args.adapter),
            "--materialized-root",
            str(args.materialized),
            "--video-root",
            str(args.videos),
            "--output",
            str(output),
            "--visible-devices",
            args.inference_gpus,
            "--expect-gpus",
            str(len(args.inference_gpus.split(","))),
            "--max-frames",
            str(args.max_frames),
            "--max-tokens",
            str(args.inference_max_tokens),
            "--per-gpu-gib",
            str(args.per_gpu_gib),
            "--resume",
        ]
        environment = command_env()
        environment.update(
            {
                "FPS": str(args.fps),
                "FPS_MAX_FRAMES": str(args.max_frames),
                "VIDEO_MAX_TOKEN_NUM": str(args.video_max_token_num),
                "VIDEO_MIN_TOKEN_NUM": str(args.video_min_token_num),
            }
        )
        run_command(
            name="infer",
            command=command,
            output=stage,
            environment=environment,
        )
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise PipelineError(
            f"inference did not reach a complete summary: {summary_path}"
        )
    return summary_path


def stage_merge(
    args: argparse.Namespace,
    run_root: Path,
    *,
    execute: bool,
) -> Path:
    stage = stage_dir(run_root, "merge")
    output = stage / "merged"
    if not execute:
        return output
    inference_root = stage_dir(run_root, "infer") / "run"
    if (stage / "merged/summary.json").is_file():
        retry_receipt = stage_dir(run_root, "merge") / "02b-retry" / "RETRY.json"
        if retry_receipt.is_file():
            apply_retry_highlights(
                run_root,
                read_json(retry_receipt)["merged_highlights"],
            )
        return output / "summary.json"
    if args.seed_strict_merged is not None:
        seed = args.seed_strict_merged.resolve()
        required = {
            "summary.json",
            "windows.jsonl",
            "highlights.jsonl",
            "failures.jsonl",
        }
        strict_summary = read_json(seed / "summary.json")
        failures = read_jsonl(seed / "failures.jsonl")
        if (
            strict_summary.get("source_videos") != 426
            or strict_summary.get("total_windows") != 523
            or strict_summary.get("valid_windows") + len(failures) != 523
        ):
            raise PipelineError("strict seed is not the expected 426/523/509/14 run")
        if len(read_jsonl(seed / "windows.jsonl")) != 523:
            raise PipelineError("strict seed windows.jsonl must contain 523 rows")
        if len(read_jsonl(seed / "highlights.jsonl")) != 426:
            raise PipelineError("strict seed highlights.jsonl must contain 426 rows")
        output.mkdir(parents=True, exist_ok=False)
        for name in sorted(required):
            shutil.copyfile(seed / name, output / name)
        atomic_json(
            stage / "SEED.json",
            {
                "schema": "video-ora-submission-strict-seed/v1",
                "state": "IMPORTED",
                "source": str(seed),
                "files": {
                    name: {
                        "bytes": (output / name).stat().st_size,
                        "sha256": sha256_file(output / name),
                    }
                    for name in sorted(required)
                },
                "inference_not_rerun": True,
            },
        )
        if failures:
            retry_failure_windows(args, run_root, seed, failures, execute=True)
            apply_retry_highlights(
                run_root,
                read_json(stage / "02b-retry" / "RETRY.json")["merged_highlights"],
            )
        return output / "summary.json"
    command = [
        str(args.python),
        "-B",
        str(REPO / "scripts/merge_video_ora_highlights_inference.py"),
        "--runs",
        str(inference_root),
        "--output",
        str(output),
    ]
    run_command(
        name="merge",
        command=command,
        output=stage,
        environment=command_env(),
    )
    return output / "summary.json"


def retry_failure_windows(
    args: argparse.Namespace,
    run_root: Path,
    seed: Path,
    failures: list[dict[str, Any]],
    *,
    execute: bool,
) -> dict[str, Any] | None:
    """Rerun only windows that had no strict response, never fabricate output."""
    if not failures:
        return None
    retry = run_root / "02b-retry"
    receipt_path = retry / "RETRY.json"
    if not execute:
        return receipt_path
    if receipt_path.is_file():
        return read_json(receipt_path)

    materialized = args.materialized.resolve()
    samples = []
    for split in ("train", "val"):
        samples.extend(
            (int(index), split, sample)
            for index, sample in enumerate(
                read_json(materialized / "manifest.json")["samples"][split]
            )
        )
    gpu_ids = [
        item.strip() for item in args.inference_gpus.split(",") if item.strip()
    ]
    retry.mkdir(parents=True, exist_ok=False)
    rows = []
    for shard_index, gpu in enumerate(gpu_ids):
        assigned = failures[shard_index::len(gpu_ids)]
        if not assigned:
            continue
        indexes = []
        for failure in assigned:
            window_id = str(failure["window_id"])
            for index, split, sample in samples:
                if (
                    f"{split}_{sample['source_video_id']}_w"
                    f"{int(sample['window_index']):04d}"
                    == window_id
                ):
                    indexes.append(index)
                    break
            else:
                raise PipelineError(f"retry window is absent from materialized data: {window_id}")
        if len(set(indexes)) != len(indexes):
            raise PipelineError("retry window indexes are not unique")
        shard = retry / f"gpu-{gpu}"
        command = [
            str(args.python),
            "-B",
            "-u",
            str(REPO / "scripts/infer_video_ora_highlights_adapter_426.py"),
            "--model",
            str(args.model),
            "--adapter",
            str(args.adapter),
            "--materialized-root",
            str(args.materialized),
            "--video-root",
            str(args.videos),
            "--output",
            str(shard),
            "--visible-devices",
            gpu,
            "--expect-gpus",
            "1",
            "--max-frames",
            str(args.max_frames),
            "--max-tokens",
            str(args.inference_max_tokens),
            "--per-gpu-gib",
            str(args.per_gpu_gib),
        ]
        if len(indexes) == 1:
            command.extend(["--start-index", str(indexes[0]), "--end-index", str(indexes[0] + 1)])
        else:
            start = min(indexes)
            stop = max(indexes) + 1
            if stop - start > 32:
                raise PipelineError(
                    "retry indexes span too many windows; use explicit per-window retry"
                )
            command.extend(["--start-index", str(start), "--end-index", str(stop)])
        environment = command_env()
        environment.update(
            {
                "FPS": str(args.fps),
                "FPS_MAX_FRAMES": str(args.max_frames),
                "VIDEO_MAX_TOKEN_NUM": str(args.video_max_token_num),
                "VIDEO_MIN_TOKEN_NUM": str(args.video_min_token_num),
                "CUDA_VISIBLE_DEVICES": gpu,
            }
        )
        run_command(
            name="retry",
            command=command,
            output=retry,
            environment=environment,
        )
        for record_path in (shard / "records").glob("*.json"):
            rows.append(read_json(record_path))
    wanted = {str(item["window_id"]) for item in failures}
    if {str(row["window_id"]) for row in rows} != wanted:
        raise PipelineError("retry output does not cover exactly the failed windows")
    if any(not row.get("raw_json_valid") for row in rows):
        raise PipelineError("one or more retried windows still have no strict answer")
    strict = seed
    highlights = {str(row["video_id"]): row for row in read_jsonl(strict / "highlights.jsonl")}
    for row in rows:
        video_id = str(row["source_video_id"])
        highlights[video_id]["highlights"].extend(row["source_highlights"])
    final = []
    for row in highlights.values():
        intervals = sorted(
            ([float(x["start"]), float(x["end"])] for x in row["highlights"]),
            key=lambda item: (item[0], item[1]),
        )
        merged: list[list[float]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        final.append({**row, "highlights": [{"start": a, "end": b} for a, b in merged]})
    atomic_json(
        receipt_path,
        {
            "schema": "video-ora-submission-strict-retry/v1",
            "state": "COMPLETE",
            "parent_strict": str(strict),
            "failed_windows": sorted(wanted),
            "retried_windows": len(rows),
            "source_highlights": {
                str(row["window_id"]): row["source_highlights"] for row in rows
            },
            "merged_highlights": final,
        },
    )
    return read_json(receipt_path)


def apply_retry_highlights(run_root: Path, rows: list[dict[str, Any]]) -> None:
    strict_path = stage_dir(run_root, "merge") / "merged" / "highlights.jsonl"
    current = read_jsonl(strict_path)
    current_by_id = {str(row["video_id"]): row for row in current}
    if len(current) != 426 or len(current_by_id) != 426:
        raise PipelineError("strict merge highlights must contain 426 unique videos")
    for row in rows:
        video_id = str(row["video_id"])
        if video_id not in current_by_id:
            raise PipelineError(f"retry result has unknown source video: {video_id}")
        current_by_id[video_id]["highlights"] = row["highlights"]
    temporary = strict_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in current),
        encoding="utf-8",
    )
    os.replace(temporary, strict_path)


def stage_lenient(
    args: argparse.Namespace,
    run_root: Path,
    *,
    execute: bool,
) -> Path:
    stage = stage_dir(run_root, "lenient")
    strict = stage_dir(run_root, "merge") / "merged"
    output = stage / "merged"
    if not execute:
        return output
    if (output / "summary.json").is_file():
        summary = read_json(output / "summary.json")
        if summary.get("unresolved_windows"):
            raise PipelineError("previous lenient merge has unresolved windows")
        return output / "summary.json"
    command = [
        str(args.python),
        "-B",
        str(REPO / "scripts/merge_video_ora_highlights_lenient_diagnostic.py"),
        "--strict-merged",
        str(strict),
        "--output",
        str(output),
        "--end-tolerance-sec",
        str(args.end_tolerance_sec),
    ]
    run_command(
        name="lenient",
        command=command,
        output=stage,
        environment=command_env(),
    )
    summary = read_json(output / "summary.json")
    if summary.get("unresolved_windows"):
        raise PipelineError(
            f"{summary['unresolved_windows']} windows remain unresolved after "
            "lenient repair"
        )
    if len(read_jsonl(output / "highlights.jsonl")) != 426:
        raise PipelineError("lenient highlights must contain 426 source videos")
    return output / "summary.json"


def stage_handoff(
    args: argparse.Namespace,
    run_root: Path,
    *,
    execute: bool,
) -> Path:
    stage = stage_dir(run_root, "handoff")
    output = stage / "temporal_handoff.jsonl"
    if not execute:
        return output
    receipt_path = stage / "HANDOFF.json"
    if output.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (
            receipt.get("sha256") == sha256_file(output)
            and receipt.get("lenient_highlights_sha256")
            == sha256_file(stage_dir(run_root, "lenient") / "merged/highlights.jsonl")
        ):
            return output
    reference_handoff = args.spatial_package / "reference/stage1_expected.jsonl"
    if not reference_handoff.is_file():
        raise FileNotFoundError(reference_handoff)
    reference_metadata = args.spatial_package / "reference/source_metadata426.json"
    if not reference_metadata.is_file():
        raise FileNotFoundError(reference_metadata)
    source_rows = read_jsonl(stage_dir(run_root, "lenient") / "merged/highlights.jsonl")
    reference_rows = read_jsonl(reference_handoff)
    reference_by_id = {str(row["video_id"]): row for row in reference_rows}
    if len(reference_rows) != 426 or len(reference_by_id) != 426:
        raise PipelineError("spatial reference handoff must contain 426 rows")
    source_by_id = {str(row["video_id"]): row for row in source_rows}
    if len(source_rows) != 426 or set(source_by_id) != set(reference_by_id):
        raise PipelineError("lenient and spatial handoff IDs differ")
    metadata_rows = read_json(reference_metadata)
    metadata_by_hash = {
        str(row["source_sha256"]): row for row in metadata_rows
    }

    rows: list[dict[str, Any]] = []
    for video_id in sorted(source_by_id, key=int):
        source = source_by_id[video_id]
        base = reference_by_id[video_id]
        duration = float(base["duration_sec"])
        metadata = metadata_by_hash.get(str(base["source_sha256"]))
        if metadata is None:
            raise PipelineError(
                f"no target-ratio metadata for source hash {video_id}"
            )
        if base["targetRatioWH"] != metadata["target_ratio"]:
            raise PipelineError(f"target ratio metadata mismatch: {video_id}")
        intervals = [
            [float(item["start"]), float(item["end"])]
            for item in source["highlights"]
        ]
        if any(
            start < 0
            or start >= end
            or end > duration
            for start, end in intervals
        ):
            raise PipelineError(f"invalid lenient interval for video {video_id}")
        frame_count = int(base["frame_count"])
        segments_frame = []
        for start, end in intervals:
            source_duration = Decimal(str(duration))
            source_start = Decimal(str(start))
            source_end = Decimal(str(end))
            first = min(
                frame_count - 1,
                max(
                    0,
                    int(
                        (
                            source_start
                            * Decimal(frame_count)
                            / source_duration
                        ).to_integral_value(rounding="ROUND_CEILING")
                    ),
                ),
            )
            last = min(
                frame_count - 1,
                max(
                    0,
                    int(
                        (
                            source_end
                            * Decimal(frame_count)
                            / source_duration
                        ).to_integral_value(rounding="ROUND_FLOOR")
                    )
                    - 1,
                ),
            )
            if last >= first:
                segments_frame.append([first, last])
        rows.append(
            {
                "video_id": video_id,
                "targetRatioWH": base["targetRatioWH"],
                "segments_sec": intervals,
                "segments_frame": segments_frame,
                "frame_index_convention": "inclusive",
                "time_interval_convention": "start_inclusive_end_exclusive",
                "frame_count": frame_count,
                "duration_sec": duration,
                "source_sha256": base["source_sha256"],
                "valid": bool(segments_frame),
                "subject_hint": "",
                "action_description": "",
                "semantic_status": "Video-ORA adapter118 lenient pipeline",
                "event_quality_threshold": None,
                "checkpoint_sha256": (
                    sha256_file(args.adapter / "adapter_model.safetensors")
                ),
                "adapter": str(args.adapter.resolve()),
                "fallback_reason": None if segments_frame else "no_highlights",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    atomic_json(
        receipt_path,
        {
            "schema": "video-ora-submission-handoff/v1",
            "path": str(output),
            "sha256": sha256_file(output),
            "videos": len(rows),
            "videos_with_highlights": sum(bool(row["segments_frame"]) for row in rows),
            "frames": sum(
                last - first + 1
                for row in rows
                for first, last in row["segments_frame"]
            ),
            "groups": sum(len(row["segments_frame"]) for row in rows),
            "reference": str(reference_handoff),
            "reference_sha256": sha256_file(reference_handoff),
            "ratio_metadata": str(reference_metadata),
            "ratio_metadata_sha256": sha256_file(reference_metadata),
            "reference_usage": (
                "frame metadata, duration, source hash, and target ratio only"
            ),
            "reference_timestamps_used": False,
            "lenient_highlights_sha256": sha256_file(
                stage_dir(run_root, "lenient") / "merged/highlights.jsonl"
            ),
        },
    )
    return output


def stage_spatial(
    args: argparse.Namespace,
    run_root: Path,
    *,
    execute: bool,
) -> Path:
    stage = stage_dir(run_root, "spatial")
    output = stage / "predictions.jsonl"
    if not execute:
        return output
    if output.is_file() and (stage / "COMPLETE.json").is_file():
        receipt = read_json(stage / "COMPLETE.json")
        if receipt.get("predictions_sha256") == sha256_file(output):
            return output
    handoff = stage_dir(run_root, "handoff") / "temporal_handoff.jsonl"
    receipt_path = stage_dir(run_root, "handoff") / "HANDOFF.json"
    receipt = read_json(receipt_path)
    if len(read_jsonl(handoff)) != 426:
        raise PipelineError("spatial handoff must contain 426 rows")
    source_videos = args.videos
    package = args.spatial_package
    if not (package / "run.py").is_file():
        raise FileNotFoundError(package / "run.py")
    if args.spatial_arm != "B":
        raise PipelineError(
            "only the frozen v7E Stage 2 B arm is supported by this submission path"
        )

    stage.mkdir(parents=True, exist_ok=True)
    spatial_input = stage / "input"
    spatial_input.mkdir(parents=True, exist_ok=True)
    target = spatial_input / "temporal_handoff.jsonl"
    if target.exists() and sha256_file(target) != sha256_file(handoff):
        raise PipelineError("spatial input directory already contains different data")
    if not target.exists():
        shutil.copyfile(handoff, target)

    work = stage / "work"
    if (work / "COMPLETE.json").exists():
        raise PipelineError(
            "spatial work is complete but its predictions were not finalized"
        )
    config = {
        "input": str(target),
        "input_sha256": sha256_file(target),
        "videos": str(source_videos),
        "package": str(package),
        "arm": args.spatial_arm,
        "gpu": args.spatial_gpu,
        "stage": "stage2",
        "stride": 3,
        "imgsz": 640,
        "conf": 0.15,
        "iou": 0.5,
        "training": False,
    }
    config_path = stage / "CONFIG.json"
    if config_path.exists():
        if read_json(config_path) != config:
            raise PipelineError("existing spatial CONFIG.json differs")
    else:
        atomic_json(config_path, config)

    command = [
        str(args.spatial_python),
        "-B",
        "-u",
        str(SPATIAL_RUNNER_SOURCE),
        "--package",
        str(package),
        "--videos",
        str(source_videos),
        "--output",
        str(work),
        "--stage1-file",
        str(target),
        "--gpus",
        str(args.spatial_gpu),
        "--resume",
    ]
    environment = command_env()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.spatial_gpu)
    run_command(
        name="spatial",
        command=command,
        output=stage,
        environment=environment,
    )

    candidates = sorted(work.rglob("predictions_spatial.jsonl"))
    if len(candidates) != 1:
        raise PipelineError(
            f"expected one spatial predictions_spatial.jsonl, found {candidates}"
        )
    shutil.copyfile(candidates[0], output)
    spatial_complete = read_json(work / "COMPLETE.json")
    if (
        spatial_complete.get("videos") != 426
        or spatial_complete.get("spatial_sha256")
        != sha256_file(candidates[0])
    ):
        raise PipelineError("spatial COMPLETE.json does not match its output")
    atomic_json(
        stage / "COMPLETE.json",
        {
            "schema": "video-ora-submission-spatial/v1",
            "state": "COMPLETE",
            "predictions": str(output),
            "predictions_sha256": sha256_file(output),
            "videos": spatial_complete["videos"],
            "frames": spatial_complete["frames"],
            "groups": spatial_complete["groups"],
            "spatial_run_fingerprint": spatial_complete["run_fingerprint"],
            "source_work": str(work),
            "arm": args.spatial_arm,
        },
    )
    return output


def normalize_submission_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": str(row["video_id"]),
        "targetRatioWH": [int(value) for value in row["targetRatioWH"]],
        "predictions": [
            {
                "frame": int(item["frame"]),
                "bboxes": [int(value) for value in item["bboxes"]],
            }
            for item in row["predictions"]
        ],
    }


def stage_submission(
    args: argparse.Namespace,
    run_root: Path,
    *,
    execute: bool,
) -> Path:
    output_dir = stage_dir(run_root, "submission")
    output = output_dir / "predictions_spatial.jsonl"
    if not execute:
        return output
    spatial = stage_dir(run_root, "spatial") / "predictions.jsonl"
    rows = read_jsonl(spatial)
    if len(rows) != 426:
        raise PipelineError(f"spatial output has {len(rows)} rows, expected 426")
    normalized = [normalize_submission_row(row) for row in rows]
    ids = [row["video_id"] for row in normalized]
    if ids != [str(value) for value in range(426)]:
        raise PipelineError("submission video IDs must be exactly 0..425 in order")
    expected_ratio = {
        str(row["video_id"]): row["targetRatioWH"] for row in read_json(args.index)
    }
    for row in normalized:
        if row["targetRatioWH"] != expected_ratio[row["video_id"]]:
            raise PipelineError(
                f"target ratio changed for video {row['video_id']}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    ok, errors = validate_submission(str(output), str(args.index), str(args.videos))
    if not ok:
        raise PipelineError("submission validation failed: " + "; ".join(errors[:12]))

    submission_alias = output_dir / "submission.jsonl"
    if submission_alias.exists():
        submission_alias.unlink()
    os.link(output, submission_alias)

    archive = output_dir / "submission.zip"
    if archive.exists():
        archive.unlink()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.write(output, arcname="predictions_spatial.jsonl")
    with ZipFile(archive) as handle:
        if (
            handle.namelist() != ["predictions_spatial.jsonl"]
            or handle.testzip() is not None
            or sha256_file(output) != hashlib.sha256(
                handle.read("predictions_spatial.jsonl")
            ).hexdigest()
            or sha256_file(output) != sha256_file(submission_alias)
        ):
            raise PipelineError("submission ZIP verification failed")

    frames = sum(len(row["predictions"]) for row in normalized)
    nonempty = sum(bool(row["predictions"]) for row in normalized)
    report = {
        "schema": "video-ora-submission-final/v1",
        "state": "READY_TO_SUBMIT",
        "created_at": utc_now(),
        "pipeline_id": read_json(run_root / "PIPELINE.json")["pipeline_id"],
        "predictions_spatial_jsonl": str(output),
        "predictions_spatial_jsonl_sha256": sha256_file(output),
        "submission_jsonl_alias": str(submission_alias),
        "submission_zip": str(archive),
        "submission_zip_sha256": sha256_file(archive),
        "videos": len(normalized),
        "videos_with_predictions": nonempty,
        "rounded_spatial_frames": frames,
        "stage1_handoff_sha256": read_json(
            stage_dir(run_root, "handoff") / "HANDOFF.json"
        )["sha256"],
        "lenient_merge_summary_sha256": sha256_file(
            stage_dir(run_root, "lenient") / "merged/summary.json"
        ),
        "training": False,
        "website_submission_performed": False,
    }
    atomic_json(output_dir / "FINAL.json", report)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--expected-adapter-sha256",
        default=DEFAULT_ADAPTER_SHA256,
        help="Freeze the exact LoRA adapter weights used by this submission",
    )
    parser.add_argument("--materialized", type=Path, default=DEFAULT_MATERIALIZED)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--spatial-python", type=Path, default=DEFAULT_SPATIAL_PYTHON
    )
    parser.add_argument("--spatial-package", type=Path, default=DEFAULT_SPATIAL_PACKAGE)
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument(
        "--seed-strict-merged",
        type=Path,
        help="Import an exact retained strict merge instead of rerunning inference",
    )
    parser.add_argument("--inference-gpus", default="2,3,4,5")
    parser.add_argument("--spatial-gpu", default="2")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--video-max-token-num", type=int, default=196)
    parser.add_argument("--video-min-token-num", type=int, default=128)
    parser.add_argument("--inference-max-tokens", type=int, default=512)
    parser.add_argument("--per-gpu-gib", type=int, default=8)
    parser.add_argument("--end-tolerance-sec", type=float, default=0.5)
    parser.add_argument("--spatial-arm", choices=("B",), default="B")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-existing", action="store_true")
    parser.add_argument("--start-stage", choices=STAGES, default="infer")
    parser.add_argument("--stop-after", choices=STAGES, default="submission")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if args.execute_existing and not args.dry_run:
        parser.error("--execute-existing is only a dry-run planning option")
    if not args.dry_run:
        args.execute = True
    if not 0 <= args.end_tolerance_sec <= 1:
        parser.error("--end-tolerance-sec must be between 0 and 1")
    if args.fps <= 0 or args.max_frames <= 0:
        parser.error("fps and max-frames must be positive")
    verify_paths(args)
    pipeline_id = build_pipeline_id(args)
    run_root, contract, resumed = prepare_run(args, pipeline_id)
    print(
        json.dumps(
            {
                "pipeline_id": pipeline_id,
                "run_root": str(run_root),
                "resumed": resumed,
                "execute": args.execute,
                "start_stage": args.start_stage,
                "stop_after": args.stop_after,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        planned_stages = []
        for stage in STAGES[
            STAGES.index(args.start_stage) :
            STAGES.index(args.stop_after) + 1
        ]:
            existing = None
            if stage == "infer":
                existing = stage_dir(run_root, "infer") / "run/summary.json"
            elif stage == "merge":
                existing = stage_dir(run_root, "merge") / "merged/summary.json"
            elif stage == "lenient":
                existing = stage_dir(run_root, "lenient") / "merged/summary.json"
            elif stage == "handoff":
                existing = stage_dir(run_root, "handoff") / "temporal_handoff.jsonl"
            elif stage == "spatial":
                existing = stage_dir(run_root, "spatial") / "predictions.jsonl"
            elif stage == "submission":
                existing = stage_dir(run_root, "submission") / "submission.jsonl"
            planned_stages.append(
                {
                    "stage": stage,
                    "existing_artifact": str(existing) if existing else None,
                    "existing": bool(existing and existing.exists()),
                }
            )
        atomic_json(
            run_root / "DRY_RUN.json",
            {
                **contract,
                "dry_run": True,
                "execute_existing": args.execute_existing,
                "planned_stages": planned_stages,
            },
        )
        return 0

    selected = list(STAGES)
    begin = selected.index(args.start_stage)
    end = selected.index(args.stop_after) + 1
    if begin >= end:
        parser.error("--start-stage must not be after --stop-after")
    try:
        for stage in selected[begin:end]:
            update_state(run_root, stage=stage, state="running", detail="started")
            if stage == "infer":
                artifact = stage_inference(args, run_root, execute=True)
            elif stage == "merge":
                artifact = stage_merge(args, run_root, execute=True)
            elif stage == "lenient":
                artifact = stage_lenient(args, run_root, execute=True)
            elif stage == "handoff":
                artifact = stage_handoff(args, run_root, execute=True)
            elif stage == "spatial":
                artifact = stage_spatial(args, run_root, execute=True)
            elif stage == "submission":
                artifact = stage_submission(args, run_root, execute=True)
            else:
                raise AssertionError(stage)
            update_state(
                run_root,
                stage=stage,
                state="complete",
                detail="stage completed",
                artifact=artifact,
            )
        final_report = stage_dir(run_root, "submission") / "FINAL.json"
        if end == len(STAGES):
            report = read_json(final_report)
            update_state(
                run_root,
                stage="submission",
                state="READY_TO_SUBMIT",
                detail="submission validated; website upload not performed",
                artifact=final_report,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as error:
        current_stage = locals().get("stage", args.start_stage)
        update_state(
            run_root,
            stage=current_stage,
            state="FAILED",
            detail=f"{type(error).__name__}: {error}",
        )
        atomic_json(
            run_root / "FAILURE.json",
            {
                "pipeline_id": pipeline_id,
                "stage": current_stage,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
