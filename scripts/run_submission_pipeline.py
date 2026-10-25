#!/usr/bin/env python3
"""Run the frozen Video-ORA, lenient repair, spatial, and submission pipeline.

This entry point is inference only. Normal model settings remain frozen;
bounded retries and explicit degraded states are recorded separately. Every stage is bound to the current
input dataset and LoRA adapter through SHA256 receipts.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from datetime import datetime, timezone
from math import isfinite
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_highlight.validation import validate_submission
from video_highlight.submission_inputs import (
    discover_input, executable, frame_spans, index_rows, prepare_inputs,
    prepared_metadata, resolve_config, run_lock, sort_id,
)
from video_highlight.submission_recovery import (
    RUNTIME_POLICIES, SPATIAL_FAILURE_POLICY, TIME_FAILURE_POLICY, TIME_RECOVERY_POLICY,
    repair_spatial_row, runtime_policy,
)


DEFAULT_MODEL = REPO / "models/ora"
DEFAULT_ADAPTER = REPO / "models/adapter"
DEFAULT_MATERIALIZED = None
DEFAULT_VIDEOS = REPO / "input/video"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_SPATIAL_PYTHON = Path(sys.executable)
DEFAULT_SPATIAL_PACKAGE = REPO / "models/spatial"
DEFAULT_PIPELINE_DIR = REPO / "outputs/submission"
DEFAULT_INDEX = REPO / "input/jsonl"
STAGES = (
    "infer",
    "merge",
    "lenient",
    "handoff",
    "spatial",
    "submission",
)
SPATIAL_RUNNER_SOURCE = REPO / "scripts/run_spatial_submission.py"


class PipelineError(RuntimeError):
    """The run cannot continue without violating its input contract."""


class WindowTimeoutError(TimeoutError):
    """The active inference worker must be reaped before continuing."""


def expected_ids(args) -> list[str]:
    return [row["video_id"] for row in index_rows(args.index)]


def expected_window_count(args) -> int:
    manifest = read_json(args.materialized / "manifest.json")
    return sum(len(samples) for samples in manifest["samples"].values())


def load_request_plan(args):
    spec = importlib.util.spec_from_file_location("submission_request_plan", REPO / "scripts/infer_video_ora_highlights.py")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    return inference.load_plan(args.materialized, args.videos)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return stream_sha256(handle)


def stream_sha256(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _weight_sha256(path: str, size: int, modified_ns: int) -> str:
    return sha256_file(Path(path))


def weight_hashes(model: Path) -> dict:
    paths = set(model.glob("*.safetensors"))
    for index in model.glob("*.safetensors.index.json"):
        for name in set(read_json(index)["weight_map"].values()):
            path = model / name
            if not path.resolve().is_relative_to(model.resolve()):
                raise PipelineError("Model shard path escapes the model directory")
            paths.add(path)
    return {str(path.relative_to(model)): _weight_sha256(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(paths)}


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
            "PYTHONPATH": str(SRC),
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
    timeout_sec: float | None = None,
    progress_path: Path | None = None,
    window_timeout_sec: float = 0,
) -> dict[str, Any]:
    if timeout_sec is None:
        timeout_sec = float(os.environ.get("VH_COMMAND_TIMEOUT_SEC", "86400"))
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
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
        try:
            with stdout_path.open("r", encoding="utf-8", errors="replace") as reader:
                while process.poll() is None:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    if timeout_sec and time.monotonic() - started > timeout_sec:
                        raise TimeoutError(f"{name} exceeded {timeout_sec:g}s; see {stdout_path}")
                    if progress_path and progress_path.is_file() and window_timeout_sec:
                        progress = read_json(progress_path)
                        active_started = progress.get("active_started_monotonic")
                        if active_started and time.monotonic() - active_started > window_timeout_sec:
                            raise WindowTimeoutError(f"Window {progress.get('active_window_id')} exceeded {window_timeout_sec:g}s")
                    time.sleep(0.1)
                print(reader.read(), end="", flush=True)
            return_code = process.wait()
        except BaseException as error:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            receipt = {"stage": name, "command": command, "return_code": process.poll(),
                       "state": "INTERRUPTED", "error": f"{type(error).__name__}: {error}",
                       "seconds": time.monotonic() - started, "stdout_log": str(stdout_path)}
            atomic_json(command_dir / "COMMAND.json", receipt)
            atomic_json(output / "COMMAND.json", receipt)
            raise
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
    if args.spatial_arm in ("B_RGBRank_DP_v1", "B_RGBRank_MissingEvidenceNative_v1"):
        rgb_backend_contract(args)
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
    revision = getattr(args, "expected_model_revision", None)
    if revision and model_revision not in (None, revision):
        raise PipelineError(
            f"unexpected base model revision: {model_revision!r}"
        )
    adapter_sha = sha256_file(args.adapter / "adapter_model.safetensors")
    if args.expected_adapter_sha256 and adapter_sha != args.expected_adapter_sha256:
        raise PipelineError(
            f"unexpected adapter SHA256: {adapter_sha}; "
            f"expected {args.expected_adapter_sha256}"
        )

    manifest = read_json(args.materialized / "manifest.json")
    if manifest.get("answer_contract") != "highlights-only/v1":
        raise PipelineError("unexpected materialized answer contract")
    if not expected_window_count(args):
        raise PipelineError("materialized input contains no windows")
    ids = set(expected_ids(args))
    sample_ids = {str(sample["source_video_id"]) for samples in manifest["samples"].values() for sample in samples}
    if ids != sample_ids or sum(manifest["source_counts"].values()) != len(ids):
        raise PipelineError("materialized source IDs differ from the input index")
    if {path.stem for path in args.videos.glob("*.mp4") if not path.name.startswith("._")} != ids:
        raise PipelineError("video directory IDs differ from the input index")
    if not (args.materialized / "source_metadata.json").is_file():
        raise PipelineError("Missing current source metadata; prepare the input with --input-root first")
    metadata = prepared_metadata(args.materialized)
    if len(metadata) != len(ids) or {str(row["video_id"]) for row in metadata} != ids:
        raise PipelineError("Prepared source metadata IDs differ from the input index")
    for row in metadata:
        if sha256_file(args.videos / (str(row["video_id"]) + ".mp4")) != row["source_sha256"]:
            raise PipelineError(f"Source video content changed: {row['video_id']}")
    for name, expected in manifest.get("prepared_files", {}).items():
        if sha256_file(args.materialized / name) != expected:
            raise PipelineError(f"Prepared inference input changed: {name}")


def build_pipeline_id(args: argparse.Namespace) -> str:
    return canonical_sha256(pipeline_contract(args))[:16]


def rgb_backend_contract(args):
    if getattr(args, "spatial_arm", "B") not in ("B_RGBRank_DP_v1", "B_RGBRank_MissingEvidenceNative_v1"):return None
    from video_highlight.stage2_rgb_assets import asset_contract
    folder=Path(args.rgb_rank_assets)
    return {"assets":asset_contract(folder),"asset_root":str(folder.resolve())}


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
        "model_weights_sha256": weight_hashes(args.model),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": (sha256_file(args.adapter / "adapter_config.json")
                                   if (args.adapter / "adapter_config.json").is_file() else None),
        "materialized": str(args.materialized.resolve()),
        "materialized_manifest_sha256": sha256_file(
            args.materialized / "manifest.json"
        ),
        "videos": str(args.videos.resolve()),
        "index": str(args.index.resolve()),
        "test_index_sha256": sha256_file(args.index),
        "video_names": sorted(path.name for path in args.videos.glob("*.mp4")),
        "inference": {
            "video_reader_backend": "decord",
            "fps": args.fps,
            "max_frames": args.max_frames,
            "max_tokens": args.inference_max_tokens,
            "video_max_token_num": video_max_token_num,
            "video_min_token_num": video_min_token_num,
            "temperature": 0,
            "seed": 42,
        },
        "unresolved_policy": TIME_FAILURE_POLICY,
        "recovery": {"recovery_policy": TIME_RECOVERY_POLICY,
                     "spatial_failure_policy": SPATIAL_FAILURE_POLICY,
                     **{key: getattr(args, key, default) for key, default in (
            ("recovery_attempts", 0),
            ("window_timeout_sec", 0), ("command_timeout_sec", 86400),
            ("spatial_video_timeout_sec", 1800))}},
        "model_processor_sha256": {str(path.relative_to(args.model)): sha256_file(path)
                                    for path in sorted([*args.model.iterdir(), *(args.model / "chat_templates").rglob("*")])
                                    if path.is_file() and path.suffix in (".json", ".model", ".txt", ".tiktoken", ".vocab", ".jinja", ".jinja2")},
        "source_files_sha256": {
            str(path.relative_to(REPO)): sha256_file(path)
            for path in [REPO / "scripts/infer_video_ora_highlights.py",
                         REPO / "scripts/merge_video_ora_highlights_inference.py",
                         REPO / "scripts/merge_video_ora_highlights_lenient_diagnostic.py",
                         *sorted((SRC / "video_highlight").glob("submission_*.py")),
                         *[SRC / "video_highlight" / name for name in
                           ("highlight_output.py", "timing.py", "validation.py", "io_utils.py",
                             "stage2_rgb_assets.py", "stage2_rgb_rank.py", "rgb_rank_core.py", "rgb_vision_compat.py")]]
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
        "spatial_python_sha256": {
            str(path.relative_to(args.spatial_package)): sha256_file(path)
            for folder in ("pipeline", "modules", "vendor")
            for path in sorted((args.spatial_package / folder).rglob("*.py"))
        },
        "spatial_config_sha256": (sha256_file(args.spatial_package / "configs/v7e_b_4109.json")
                                   if (args.spatial_package / "configs/v7e_b_4109.json").is_file() else None),
        "spatial_runner": str(SPATIAL_RUNNER_SOURCE.resolve()),
        "spatial_runner_sha256": sha256_file(SPATIAL_RUNNER_SOURCE),
        "pipeline_runner": str(Path(__file__).resolve()),
        "pipeline_runner_sha256": sha256_file(Path(__file__).resolve()),
        "spatial_arm": spatial_arm,
        "rgb_rank_backend": rgb_backend_contract(args),
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
    existing_run = getattr(args, "existing_run_root", None)
    run_root = (
        existing_run.resolve()
        if existing_run is not None
        else args.pipeline_dir / f"run-{pipeline_id}"
    )
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
    if not (output / "summary.json").is_file():
        command = [
            str(args.python),
            "-B",
            "-u",
            str(REPO / "scripts/infer_video_ora_highlights.py"),
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
            "--window-timeout-sec", str(getattr(args, "window_timeout_sec", 0)),
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
        run_inference_command(args, command, stage, output, environment, name="infer")
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise PipelineError(
            f"inference did not reach a complete summary: {summary_path}"
        )
    for name, expected in read_json(summary_path).get("record_files_sha256", {}).items():
        if sha256_file(output / "records" / name) != expected:
            raise PipelineError(f"Inference checkpoint was modified: {name}")
    return summary_path


def run_inference_command(args, command, stage, output, environment, *, name):
    """A crashed active request is one failed window, not a lost whole shard.

    Initialization failures have no active-window marker and remain fatal.
    The failed request is saved, the owned worker is reaped, and a fresh
    process resumes the other windows. The later recovery queue has one
    separately frozen retry budget.
    """
    deadline = time.monotonic() + getattr(args, "command_timeout_sec", 86400)
    handled = set()
    while True:
        atomic_json(output / "status.json", {"state": "STARTING"})
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Inference shard exhausted its wall-clock budget")
            run_command(name=name, command=command, output=stage, environment=environment,
                        timeout_sec=remaining, progress_path=output / "status.json",
                        window_timeout_sec=getattr(args, "window_timeout_sec", 0))
            return
        except (PipelineError, WindowTimeoutError) as error:
            status = read_json(output / "status.json")
            window_id = status.get("active_window_id")
            if not window_id or window_id in handled:
                raise
            handled.add(window_id)
            _, windows, _, _ = load_request_plan(args)
            item = next((w for w in windows if w["window_id"] == window_id), None)
            if item is None:
                raise PipelineError("Worker active window is outside the current input") from error
            record_path = output / "records" / f"{window_id}.json"
            if record_path.is_file() and read_json(record_path).get("raw_json_valid"):
                continue  # Died after publishing this success; resume the rest.
            record = read_json(record_path) if record_path.is_file() else dict(item)
            request_hash = hashlib.sha256(json.dumps(
                {"messages": item["messages"], "video": item["window_video"]},
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            record.update(request_sha256=request_hash, raw_json_valid=False,
                          validation_error=f"worker_failed: {type(error).__name__}: {error}",
                          assistant_answer_passed_to_model=False)
            for folder in ("records", "raw", "failed"):
                atomic_json(output / folder / f"{window_id}.json", record)
            atomic_json(output / "WORKER_RECOVERY.json", {"failed_windows": sorted(handled),
                                                          "policy": "reap_worker_preserve_success_resume_remaining"})


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
        retry_path = run_root / "02b-retry/RETRY.json"
        if retry_path.is_file() and "window_results" in read_json(retry_path):
            receipt = read_json(retry_path)
            if read_json(output / "summary.json").get("recovery_receipt_sha256") != canonical_sha256(receipt):
                apply_retry_records(output, receipt)
        verify_cached_artifacts(output)
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
            strict_summary.get("source_videos") != len(expected_ids(args))
            or strict_summary.get("total_windows") != expected_window_count(args)
            or strict_summary.get("valid_windows") + len(failures) != expected_window_count(args)
        ):
            raise PipelineError("strict seed counts differ from the current input")
        if len(read_jsonl(seed / "windows.jsonl")) != expected_window_count(args):
            raise PipelineError("strict seed window count differs from current input")
        if {str(row['video_id']) for row in read_jsonl(seed / "highlights.jsonl")} != set(expected_ids(args)):
            raise PipelineError("strict seed source IDs differ from current input")
        sample_ids = {f"{split}_{sample['source_video_id']}_w{int(sample['window_index']):04d}"
                      for split, samples in read_json(args.materialized / "manifest.json")["samples"].items()
                      for sample in samples}
        if {str(row["window_id"]) for row in read_jsonl(seed / "windows.jsonl")} != sample_ids:
            raise PipelineError("Strict seed window IDs differ; counts alone cannot identify the current input")
        _, _, current_windows, _ = load_request_plan(args)
        for window in read_jsonl(seed / "windows.jsonl"):
            record = read_json(Path(window["record_path"]))
            expected = current_windows[str(window["window_id"])]
            keys = ("source_video_id", "window_start_sec", "window_end_sec", "window_video_sha256", "messages")
            if any(record.get(key) != expected[key] for key in keys):
                raise PipelineError(f"Strict seed request/video differs: {window['window_id']}")
        source_runs = strict_summary.get("runs", [])
        if not source_runs:
            raise PipelineError("Strict seed lacks inference-plan provenance")
        for source_run in source_runs:
            plan = read_json(Path(source_run) / "plan.json")
            if plan.get("adapter_sha256") != sha256_file(args.adapter / "adapter_model.safetensors"):
                raise PipelineError("Strict seed used a different or unidentified Adapter")
        temporary_output = output.with_name(f".importing-{uuid.uuid4().hex}")
        temporary_output.mkdir(parents=True, exist_ok=False)
        for name in sorted(required):
            shutil.copyfile(seed / name, temporary_output / name)
        verify_cached_artifacts(temporary_output)
        os.replace(temporary_output, output)
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


def verify_cached_artifacts(output):
    for name, expected in read_json(output / "summary.json").get("artifacts_sha256", {}).items():
        if not (output / name).is_file() or sha256_file(output / name) != expected:
            raise PipelineError(f"Cached aggregate was modified: {output / name}")


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
    offset = 0
    for split, split_samples in read_json(materialized / "manifest.json")["samples"].items():
        samples.extend(
            (offset + index, split, sample)
            for index, sample in enumerate(split_samples)
        )
        offset += len(split_samples)
    gpu_ids = [
        item.strip() for item in args.inference_gpus.split(",") if item.strip()
    ]
    retry.mkdir(parents=True, exist_ok=True)
    rows = []
    record_paths = {}
    for shard_index, gpu in enumerate(gpu_ids):
        assigned = failures[shard_index::len(gpu_ids)]
        if not assigned:
            continue
        window_ids = []
        for failure in assigned:
            window_id = str(failure["window_id"])
            for index, split, sample in samples:
                if (
                    f"{split}_{sample['source_video_id']}_w"
                    f"{int(sample['window_index']):04d}"
                    == window_id
                ):
                    window_ids.append(window_id)
                    break
            else:
                raise PipelineError(f"retry window is absent from materialized data: {window_id}")
        if len(set(window_ids)) != len(window_ids):
            raise PipelineError("retry window indexes are not unique")
        shard = retry / f"gpu-{gpu}"
        command = [
            str(args.python),
            "-B",
            "-u",
            str(REPO / "scripts/infer_video_ora_highlights.py"),
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
        command.extend(["--resume", "--window-timeout-sec", str(getattr(args, "window_timeout_sec", 0)),
                        "--window-ids", *window_ids])
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
        if not (shard / "summary.json").is_file():
            run_inference_command(args, command, retry, shard, environment, name="retry")
        for record_path in (shard / "records").glob("*.json"):
            row = read_json(record_path)
            rows.append(row)
            record_paths[str(row["window_id"])] = str(record_path)
    wanted = {str(item["window_id"]) for item in failures}
    if {str(row["window_id"]) for row in rows} != wanted:
        raise PipelineError("retry output does not cover exactly the failed windows")
    strict = seed
    highlights = {str(row["video_id"]): row for row in read_jsonl(strict / "highlights.jsonl")}
    for row in rows:
        if not row.get("raw_json_valid"):
            continue
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
                str(row["window_id"]): row.get("source_highlights", []) for row in rows
            },
            "merged_highlights": final,
            "window_results": rows,
            "record_paths": record_paths,
            "still_failed_windows": sum(not row.get("raw_json_valid") for row in rows),
        },
    )
    return read_json(receipt_path)


def apply_retry_records(strict: Path, receipt: dict) -> None:
    """Rebuild every strict aggregate from the new per-window terminal states."""
    windows = read_jsonl(strict / "windows.jsonl")
    replacements = {str(row["window_id"]): row for row in receipt["window_results"]}
    if set(replacements) - {str(w["window_id"]) for w in windows}:
        raise PipelineError("Retry contains an unknown window")
    failures, intervals = [], {}
    for window in windows:
        window_id = str(window["window_id"])
        row = replacements.get(window_id)
        if row:
            if (str(row["source_video_id"]), row["window_start_sec"], row["window_end_sec"]) != (
                str(window["source_video_id"]), window["window_start_sec"], window["window_end_sec"]):
                raise PipelineError("Retry changed window identity")
            window.update(raw_json_valid=bool(row.get("raw_json_valid")),
                          record_path=receipt["record_paths"][window_id],
                          source_highlights=[[x["start"], x["end"]] for x in row.get("source_highlights", [])]
                          if row.get("raw_json_valid") else [])
        video_id = str(window["source_video_id"])
        if window["raw_json_valid"]:
            intervals.setdefault(video_id, []).extend(window.get("source_highlights", []))
        else:
            record = read_json(Path(window["record_path"]))
            failures.append({"video_id": video_id, "window_id": window_id,
                             "record_path": window["record_path"], "error": record.get("validation_error")})
    highlights = read_jsonl(strict / "highlights.jsonl")
    for row in highlights:
        merged = []
        for a, b in sorted(intervals.get(str(row["video_id"]), [])):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        row["highlights"] = [{"start": a, "end": b} for a, b in merged]
    for name, values in (("windows.jsonl", windows), ("failures.jsonl", failures), ("highlights.jsonl", highlights)):
        temporary = strict / f".{name}.tmp"
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
        os.replace(temporary, strict / name)
    summary = read_json(strict / "summary.json")
    summary.update(state="complete_with_failures" if failures else "complete", failures=failures,
                   valid_windows=len(windows) - len(failures), failed_windows=len(failures),
                   source_videos_with_highlights=sum(bool(r["highlights"]) for r in highlights),
                   total_merged_highlights=sum(len(r["highlights"]) for r in highlights),
                   recovery_receipt_sha256=canonical_sha256(receipt))
    summary["artifacts_sha256"] = {name: sha256_file(strict / name) for name in
                                   ("windows.jsonl", "failures.jsonl", "highlights.jsonl")}
    atomic_json(strict / "summary.json", summary)


def apply_retry_highlights(run_root: Path, rows: list[dict[str, Any]]) -> None:
    """Legacy video-only API; the formal recovery path uses apply_retry_records."""
    strict_path = stage_dir(run_root, "merge") / "merged" / "highlights.jsonl"
    current = read_jsonl(strict_path)
    current_by_id = {str(row["video_id"]): row for row in current}
    if not current or len(current_by_id) != len(current):
        raise PipelineError("strict merge highlights must contain unique videos")
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
    fallback_path = stage / "FALLBACKS.json"
    if not execute:
        return output
    command = [
            str(getattr(args, "python", sys.executable)), "-B",
            str(REPO / "scripts/merge_video_ora_highlights_lenient_diagnostic.py"),
            "--strict-merged", str(strict), "--output", str(output),
            "--end-tolerance-sec", str(getattr(args, "end_tolerance_sec", 0.5)),
            "--recovery-policy", TIME_RECOVERY_POLICY,
        ]
    if (output / "summary.json").is_file():
        old_inputs = read_json(output / "summary.json").get("strict_inputs_sha256")
        if old_inputs and old_inputs != {name: sha256_file(strict / name) for name in old_inputs}:
            os.replace(output, output.with_name(f"merged-before-recovery-{uuid.uuid4().hex}"))
    if not (output / "summary.json").is_file():
        run_command(name="lenient", command=command, output=stage, environment=command_env())
    verify_cached_artifacts(output)
    summary = read_json(output / "summary.json")
    if summary.get("unresolved_windows") and getattr(args, "recovery_attempts", 0):
        receipt_path = run_root / "02b-retry/RETRY.json"
        already_applied = (receipt_path.is_file() and
                           read_json(strict / "summary.json").get("recovery_receipt_sha256") ==
                           canonical_sha256(read_json(receipt_path)))
        if not already_applied:
            receipt = retry_failure_windows(args, run_root, strict, summary["unresolved"], execute=True)
            apply_retry_records(strict, receipt)
            os.replace(output, output.with_name(f"merged-before-retry-{uuid.uuid4().hex}"))
            run_command(name="lenient-after-retry", command=command, output=stage, environment=command_env())
            verify_cached_artifacts(output)
            summary = read_json(output / "summary.json")
    if summary.get("unresolved_windows"):
        unresolved = summary.get("unresolved", [])
        if not unresolved:
            raise PipelineError("unresolved count has no window details")
        source_rows = read_jsonl(output / "highlights.jsonl")
        source_by_id = {
            str(row["video_id"]): row for row in source_rows
        }
        fallbacks = []
        for item in unresolved:
            video_id = str(item["video_id"])
            if video_id not in source_by_id:
                raise PipelineError(
                    f"unresolved fallback has unknown source video: {video_id}"
                )
            fallbacks.append(
                {
                    "video_id": video_id,
                    "window_id": str(item["window_id"]),
                    "error": str(item.get("error", "")),
                    "lenient_error": str(item.get("lenient_error", "")),
                    "fallback": "drop_failed_window_only",
                    "source_highlights_preserved": source_by_id[video_id][
                        "highlights"
                    ],
                }
            )
        atomic_json(
            fallback_path,
            {
                "schema": "video-ora-submission-fallback/v1",
                "state": "APPLIED",
                "policy": "drop_unresolved_window_preserve_other_windows",
                "source_video_count": len({row["video_id"] for row in fallbacks}),
                "fallbacks": fallbacks,
            },
        )
        print(
            json.dumps(
                {
                    "warnings": "unresolved windows explicitly dropped",
                    "fallback_count": len(fallbacks),
                    "fallback_receipt": str(fallback_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if {str(row["video_id"]) for row in read_jsonl(output / "highlights.jsonl")} != set(expected_ids(args)):
        raise PipelineError("lenient source IDs differ from current input")
    windows = read_jsonl(output / "windows.jsonl") if (output / "windows.jsonl").is_file() else []
    terminal = []
    unresolved_ids = {str(item["video_id"]) for item in summary.get("unresolved", [])}
    for row in read_jsonl(output / "highlights.jsonl"):
        video_id = str(row["video_id"])
        own = [w for w in windows if str(w["source_video_id"]) == video_id]
        if video_id in unresolved_ids:
            state = "partial" if row["highlights"] else "inference_failed"
        elif any(w.get("recovery_status") == "partial" for w in own):
            state = "partial"
        elif any(w.get("accepted_by_lenient") for w in own):
            state = "repaired" if row["highlights"] else "no_highlights"
        else:
            state = "ok" if row["highlights"] else "no_highlights"
        terminal.append({"video_id": video_id, "state": state,
                         "unresolved_windows": [item["window_id"] for item in summary.get("unresolved", [])
                                                if str(item["video_id"]) == video_id]})
    atomic_json(stage / "TERMINAL_STATES.json", {"videos": terminal, "input_coverage": len(terminal),
                                               "policy": TIME_FAILURE_POLICY})
    return output / "summary.json"


def recompute_lenient_to_backup(
    args: argparse.Namespace,
    run_root: Path,
) -> Path:
    """Recompute the lenient merge in a new directory without erasing evidence."""
    stage = stage_dir(run_root, "lenient")
    strict = stage_dir(run_root, "merge") / "merged"
    previous = stage / "merged"
    if not (previous / "summary.json").is_file():
        return stage_lenient(args, run_root, execute=True)
    output = stage / f"merged-recompute-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
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
        "--recovery-policy", TIME_RECOVERY_POLICY,
    ]
    run_command(
        name="lenient-recompute",
        command=command,
        output=stage,
        environment=command_env(),
    )
    summary = read_json(output / "summary.json")
    if {str(row["video_id"]) for row in read_jsonl(output / "highlights.jsonl")} != set(expected_ids(args)):
        raise PipelineError("recomputed lenient source IDs differ from current input")
    old = stage / f"merged-before-recompute-{uuid.uuid4().hex}"
    os.replace(previous, old)
    os.replace(output, previous)
    atomic_json(
        stage / "RECOMPUTE.json",
        {
            "schema": "video-ora-submission-lenient-recompute/v1",
            "state": "COMPLETE",
            "previous": str(old),
            "active": str(previous),
            "previous_summary_sha256": sha256_file(old / "summary.json"),
            "active_summary_sha256": sha256_file(
                previous / "summary.json"
            ),
            "active_highlights_sha256": sha256_file(
                previous / "highlights.jsonl"
            ),
        },
    )
    return stage_lenient(args, run_root, execute=True)


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
            and receipt.get("source_metadata_sha256") == sha256_file(args.materialized / "source_metadata.json")
            and receipt.get("temporal_terminal_sha256") == (
                sha256_file(stage_dir(run_root, "lenient") / "TERMINAL_STATES.json")
                if (stage_dir(run_root, "lenient") / "TERMINAL_STATES.json").is_file() else None)
        ):
            return output
    source_metadata_path = args.materialized / "source_metadata.json"
    terminal_path = stage_dir(run_root, "lenient") / "TERMINAL_STATES.json"
    temporal_states = ({str(row["video_id"]): row["state"] for row in read_json(terminal_path)["videos"]}
                       if terminal_path.is_file() else {})
    source_rows = read_jsonl(stage_dir(run_root, "lenient") / "merged/highlights.jsonl")
    metadata_rows = prepared_metadata(args.materialized)
    metadata_by_id = {str(row["video_id"]): row for row in metadata_rows}
    ratios = {row["video_id"]: row["targetRatioWH"] for row in index_rows(args.index)}
    source_by_id = {str(row["video_id"]): row for row in source_rows}
    if (len(source_by_id) != len(source_rows) or len(metadata_by_id) != len(metadata_rows)
            or set(source_by_id) != set(ratios) or set(metadata_by_id) != set(ratios)):
        raise PipelineError("input metadata and temporal result IDs differ")

    rows: list[dict[str, Any]] = []
    for video_id in sorted(source_by_id, key=sort_id):
        source = source_by_id[video_id]
        base = metadata_by_id[video_id]
        duration = float(base["duration_sec"])
        if base["target_ratio"] != ratios[video_id]:
            raise PipelineError(f"target ratio metadata mismatch: {video_id}")
        if sha256_file(args.videos / f"{video_id}.mp4") != base["source_sha256"]:
            raise PipelineError(f"source video changed: {video_id}")
        intervals = [
            [float(item["start"]), float(item["end"])]
            for item in source["highlights"]
        ]
        if any(
            not isfinite(start) or not isfinite(end) or start < 0
            or start >= end
            or end > duration
            for start, end in intervals
        ):
            raise PipelineError(f"invalid lenient interval for video {video_id}")
        frame_count = int(base["frame_count"])
        segments_frame = frame_spans(intervals, base["pts_sec"])
        rows.append(
            {
                "video_id": video_id,
                "targetRatioWH": ratios[video_id],
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
                "semantic_status": "Video-ORA configured-adapter inference pipeline",
                "event_quality_threshold": None,
                "checkpoint_sha256": (
                    sha256_file(args.adapter / "adapter_model.safetensors")
                ),
                "adapter": str(args.adapter.resolve()),
                "temporal_status": temporal_states.get(video_id, "ok" if intervals else "no_highlights"),
                "fallback_reason": (None if segments_frame else
                                    temporal_states.get(video_id) if temporal_states.get(video_id) in ("inference_failed", "partial") else
                                    "no_frame_pts_in_highlight" if intervals else "no_highlights"),
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
            "source_metadata": str(source_metadata_path),
            "source_metadata_sha256": sha256_file(source_metadata_path),
            "temporal_terminal_sha256": sha256_file(terminal_path) if terminal_path.is_file() else None,
            "frame_mapping": "decoded source PTS, half-open time intervals",
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
        if (receipt.get("predictions_sha256") == sha256_file(output)
                and receipt.get("handoff_sha256") == sha256_file(stage_dir(run_root, "handoff") / "temporal_handoff.jsonl")):
            return output
    handoff = stage_dir(run_root, "handoff") / "temporal_handoff.jsonl"
    receipt_path = stage_dir(run_root, "handoff") / "HANDOFF.json"
    receipt = read_json(receipt_path)
    if {str(row["video_id"]) for row in read_jsonl(handoff)} != set(expected_ids(args)):
        raise PipelineError("spatial handoff IDs differ from current input")
    source_videos = args.videos
    package = args.spatial_package
    if not (package / "modules/stage2_b.py").is_file():
        raise FileNotFoundError(package / "modules/stage2_b.py")
    if args.spatial_arm not in ("B", "B_RGBRank_DP_v1", "B_RGBRank_MissingEvidenceNative_v1"):
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
    config = {
        "input": str(target),
        "input_sha256": sha256_file(target),
        "videos": str(source_videos),
        "package": str(package),
        "arm": args.spatial_arm,
        "rgb_rank_backend": rgb_backend_contract(args),
        "gpu": args.spatial_gpu,
        "stage": "stage2",
        "stride": 3,
        "imgsz": 640,
        "conf": 0.15,
        "iou": 0.5,
        "training": False,
        "failure_policy": SPATIAL_FAILURE_POLICY,
        "video_timeout_sec": getattr(args, "spatial_video_timeout_sec", 1800),
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
        "--source-metadata",
        str(args.materialized / "source_metadata.json"),
        "--resume",
        "--failure-policy", SPATIAL_FAILURE_POLICY,
        "--video-timeout-sec", str(getattr(args, "spatial_video_timeout_sec", 1800)),
    ]
    command += ["--arm", args.spatial_arm]
    if args.spatial_arm in ("B_RGBRank_DP_v1", "B_RGBRank_MissingEvidenceNative_v1"):
        command += ["--rgb-rank-assets", str(args.rgb_rank_assets)]
    environment = command_env()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.spatial_gpu)
    if not (work / "COMPLETE.json").is_file():
        run_command(name="spatial", command=command, output=stage, environment=environment)

    candidates = sorted(work.rglob("predictions_spatial.jsonl"))
    if len(candidates) != 1:
        raise PipelineError(
            f"expected one spatial predictions_spatial.jsonl, found {candidates}"
        )
    shutil.copyfile(candidates[0], output)
    spatial_complete = read_json(work / "COMPLETE.json")
    if (
        spatial_complete.get("videos") != len(expected_ids(args))
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
            "handoff_sha256": sha256_file(handoff),
            "videos": spatial_complete["videos"],
            "frames": spatial_complete["frames"],
            "groups": spatial_complete["groups"],
            "spatial_run_fingerprint": spatial_complete["run_fingerprint"],
            "source_work": str(work),
            "arm": args.spatial_arm,
        "rgb_rank_backend": rgb_backend_contract(args),
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
    expected = index_rows(args.index)
    expected_set = {r["video_id"] for r in expected}
    handoff = {str(row["video_id"]): row for row in read_jsonl(stage_dir(run_root, "handoff") / "temporal_handoff.jsonl")}
    metadata = {str(row["video_id"]): row for row in read_json(args.materialized / "source_metadata.json")}
    if set(handoff) != expected_set or set(metadata) != expected_set:
        raise PipelineError("Final export metadata differs from the input index")
    offsets, duplicates = {}, set()
    with spatial.open("r", encoding="utf-8-sig") as reader:
        while True:
            offset = reader.tell()
            line = reader.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = str(row.get("video_id", ""))
            if video_id not in expected_set:
                raise PipelineError(f"Unknown spatial video identity: {video_id!r}")
            if video_id in offsets:
                duplicates.add(video_id)
            offsets[video_id] = offset
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    frames, nonempty, recoveries = 0, 0, []
    with temporary.open("x", encoding="utf-8") as writer, spatial.open("r", encoding="utf-8-sig") as reader:
        for item in expected:
            video_id = item["video_id"]
            missing = video_id not in offsets or video_id in duplicates
            if missing:
                row = {}
            else:
                reader.seek(offsets[video_id])
                row = json.loads(reader.readline())
            first = handoff[video_id]
            if first["targetRatioWH"] != item["targetRatioWH"]:
                raise PipelineError("Temporal handoff ratio differs from the input index")
            normalized, repaired = repair_spatial_row(row, first, metadata[video_id]["source_size"])
            if repaired or missing or row.get("targetRatioWH") != item["targetRatioWH"]:
                recoveries.append({"video_id": video_id, "state": "fallback" if missing else "repaired",
                                   "repaired_frames": len(repaired),
                                   "reason": "missing_or_duplicate_row" if missing else "invalid_spatial_fields"})
            writer.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            frames += len(normalized["predictions"])
            nonempty += bool(normalized["predictions"])
    ok, errors = validate_submission(str(temporary), str(args.index), str(args.videos))
    if not ok:
        raise PipelineError("submission validation failed: " + "; ".join(errors[:12]))
    os.replace(temporary, output)
    atomic_json(output_dir / "REPAIRS.json", {"videos": recoveries, "policy": SPATIAL_FAILURE_POLICY})

    submission_alias = output_dir / "submission.jsonl"
    temporary_alias = output_dir / f".submission.{uuid.uuid4().hex}.jsonl"
    shutil.copyfile(output, temporary_alias)
    os.replace(temporary_alias, submission_alias)

    archive = output_dir / "submission.zip"
    temporary_archive = output_dir / f".submission.{uuid.uuid4().hex}.zip"
    with ZipFile(temporary_archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.write(output, arcname="predictions_spatial.jsonl")
    with ZipFile(temporary_archive) as handle:
        with handle.open("predictions_spatial.jsonl") as zipped:
            member_sha256 = stream_sha256(zipped)
        if (
            handle.namelist() != ["predictions_spatial.jsonl"]
            or handle.testzip() is not None
            or sha256_file(output) != member_sha256
            or sha256_file(output) != sha256_file(submission_alias)
        ):
            raise PipelineError("submission ZIP verification failed")
    os.replace(temporary_archive, archive)
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
        "videos": len(expected),
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
        "final_spatial_repaired_videos": len(recoveries),
        "temporal_terminal_states": str(stage_dir(run_root, "lenient") / "TERMINAL_STATES.json"),
        "spatial_recovery_receipt": str(stage_dir(run_root, "spatial") / "work/RECOVERY.json"),
    }
    atomic_json(output_dir / "FINAL.json", report)
    return output


def main(argv=None) -> int:
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; stopping owned pipeline children")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Experiment config; relative paths resolve against this file")
    parser.add_argument("--input-root", type=Path, help="New input folder containing video(s)/ and jsonl/ or test_index.json")
    parser.add_argument("--prompt", type=Path, default=REPO / "prompts/ora/whole-video-highlights-v1/prompt.txt")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--window-seconds", type=float, default=32.0)
    parser.add_argument("--window-stride", type=float, default=24.0)
    # Accept old commands without exposing a switch that can turn recovery off.
    parser.add_argument("--unresolved-policy", choices=("error", "drop_windows"), default=TIME_FAILURE_POLICY,
                        help=argparse.SUPPRESS)
    parser.add_argument("--recovery-policy", choices=("diagnostic", "intersection"), default=TIME_RECOVERY_POLICY,
                        help=argparse.SUPPRESS)
    parser.add_argument("--recovery-attempts", type=int, choices=(0, 1), default=1)
    parser.add_argument("--window-timeout-sec", type=float, default=600)
    parser.add_argument("--command-timeout-sec", type=float, default=86400)
    parser.add_argument("--spatial-video-timeout-sec", type=float, default=1800)
    parser.add_argument("--spatial-failure-policy", choices=("error", "center"), default=SPATIAL_FAILURE_POLICY,
                        help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare new inputs on CPU; never load models")
    parser.add_argument("--expected-model-revision")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--expected-adapter-sha256",
        default=None,
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
    parser.add_argument("--pipeline-dir", "--output", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument(
        "--seed-strict-merged",
        type=Path,
        help="Import an exact retained strict merge instead of rerunning inference",
    )
    parser.add_argument(
        "--existing-run-root",
        type=Path,
        help="Explicitly resume an existing run after a code-contract update",
    )
    parser.add_argument("--inference-gpus", default="0")
    parser.add_argument("--spatial-gpu", default="0")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--video-max-token-num", type=int, default=196)
    parser.add_argument("--video-min-token-num", type=int, default=128)
    parser.add_argument("--inference-max-tokens", type=int, default=512)
    parser.add_argument("--per-gpu-gib", type=int, default=8)
    parser.add_argument("--end-tolerance-sec", type=float, default=0.5)
    parser.add_argument("--rgb-rank-assets", type=Path, default=REPO / "models/spatial_rgb_rank")
    parser.add_argument("--spatial-arm", choices=("B", "B_RGBRank_DP_v1", "B_RGBRank_MissingEvidenceNative_v1"), default="B")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-existing", action="store_true")
    parser.add_argument(
        "--recompute-lenient",
        action="store_true",
        help="Recompute lenient merge into a new directory and resume",
    )
    parser.add_argument("--start-stage", choices=STAGES, default="infer")
    parser.add_argument("--stop-after", choices=STAGES, default="submission")
    config_arg, _ = parser.parse_known_args(argv)
    if config_arg.config:
        parser.set_defaults(**resolve_config(config_arg.config))
    args = parser.parse_args(argv)
    for name in RUNTIME_POLICIES:
        setattr(args, name, runtime_policy(name, getattr(args, name)))
    for name in ("window_timeout_sec", "command_timeout_sec", "spatial_video_timeout_sec"):
        if not isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    os.environ["VH_COMMAND_TIMEOUT_SEC"] = str(args.command_timeout_sec)
    if STAGES.index(args.start_stage) > STAGES.index(args.stop_after):
        parser.error("--start-stage must not be after --stop-after")
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    if not (isfinite(args.fps) and args.fps > 0 and isfinite(args.window_seconds)
            and isfinite(args.window_stride) and args.max_frames > 0
            and 0 <= args.end_tolerance_sec <= 1):
        parser.error("Invalid/nonfinite sampling or tolerance parameter")
    for name in ("model", "adapter", "videos", "index", "spatial_package", "rgb_rank_assets", "pipeline_dir", "prompt", "materialized", "existing_run_root", "seed_strict_merged"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    if args.input_root:
        args.videos, args.index = discover_input(Path(args.input_root))
    if not (0 < args.window_stride <= args.window_seconds) or args.window_seconds * args.fps > args.max_frames:
        parser.error("Require 0 < stride <= window size and window size * fps <= max frames")
    for name in ("python", "spatial_python", "ffmpeg", "ffprobe"):
        setattr(args, name, executable(getattr(args, name)))
    if args.dry_run and args.materialized is None:
        rows = index_rows(args.index)
        print(json.dumps({"dry_run": True, "videos": len(rows), "input": str(args.videos),
                          "model": str(args.model), "adapter": str(args.adapter),
                          "spatial_package": str(args.spatial_package), "pipeline_dir": str(args.pipeline_dir),
                          "stages": ["prepare", *STAGES], "gpu_started": False,
                          "recovery_policies": dict(RUNTIME_POLICIES)}, ensure_ascii=False, indent=2))
        return 0
    if args.materialized is None:
        args.materialized = prepare_inputs(args)
        args.index = args.materialized / "test_index.json"
    if args.prepare_only:
        print(json.dumps({"prepared_inputs": str(args.materialized), "gpu_started": False}, indent=2))
        return 0
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
    if args.existing_run_root is not None:
        if not (args.existing_run_root / "PIPELINE.json").is_file():
            parser.error("--existing-run-root has no PIPELINE.json")
        if not (
            args.existing_run_root
            / stage_dir(Path("."), "lenient")
            / "merged/summary.json"
        ).is_file():
            parser.error(
                "--existing-run-root has no completed lenient merge"
            )
        if args.start_stage == "infer":
            parser.error(
                "--existing-run-root cannot start from infer"
            )
        args.pipeline_dir = args.existing_run_root.parent
    verify_paths(args)
    pipeline_id = build_pipeline_id(args)
    if args.existing_run_root is not None:
        run_root = args.existing_run_root.resolve()
        contract = read_json(run_root / "PIPELINE.json")
        current = pipeline_contract(args)
        differences = [key for key, value in current.items() if contract.get(key) != value]
        if differences:
            raise PipelineError("Existing run inputs/code changed; use a new output run: " + ", ".join(differences))
        resumed = True
        atomic_json(args.pipeline_dir / "CURRENT.json", {"pipeline_id": contract["pipeline_id"],
                    "run_root": str(run_root), "updated_at": utc_now()})
    else:
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

    with run_lock(run_root):
        return execute_stages(args, run_root, pipeline_id)


def execute_stages(args, run_root: Path, pipeline_id: str) -> int:
    selected = list(STAGES)
    begin = selected.index(args.start_stage)
    end = selected.index(args.stop_after) + 1
    if begin >= end:
        raise PipelineError("--start-stage must not be after --stop-after")
    try:
        if args.recompute_lenient and args.start_stage != "lenient":
            raise PipelineError(
                "--recompute-lenient requires --start-stage lenient"
            )
        if args.recompute_lenient:
            update_state(
                run_root,
                stage="lenient",
                state="running",
                detail="recomputing from preserved strict failures",
            )
            artifact = recompute_lenient_to_backup(args, run_root)
            update_state(
                run_root,
                stage="lenient",
                state="complete",
                detail="lenient recompute completed",
                artifact=artifact,
            )
            begin += 1
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
    except BaseException as error:
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
