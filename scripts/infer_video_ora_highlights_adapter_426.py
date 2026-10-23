#!/usr/bin/env python3
"""Run a frozen highlights-only LoRA adapter over every materialized test window.

This is an inference-only entry point. It removes assistant messages before
generation, maps window-local seconds back to the source-video timeline, merges
overlaps, and writes every raw response, parse result, and failure separately.
It never trains, edits source annotations, or silently converts a failed parse
into an empty highlight answer.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
from typing import Any


EMPTY_THINK = re.compile(r"\A\s*<think>\s*</think>\s*", re.IGNORECASE)
REQUIRED_ROW_KEYS = {"messages", "videos"}
WINDOW_LOCKS = {}
REQUIRED_SAMPLE_KEYS = {
    "source_video_id",
    "source_video",
    "source_duration_sec",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def acquire_window_claim(failed_dir: Path, window_id: str) -> Path | None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from video_highlight.submission_inputs import run_lock
    lock = run_lock(failed_dir / ".window-locks" / window_id)
    try:
        lock.__enter__()
    except RuntimeError:
        return None
    claim = failed_dir / f".{window_id}.claim"
    try:
        # The OS lock, not this receipt, establishes ownership. A killed process
        # releases its lock automatically, so its stale receipt is recoverable.
        with claim.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "created_at": utc_now(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except BaseException:
        lock.__exit__(None, None, None)
        raise
    WINDOW_LOCKS[claim.resolve()] = lock
    return claim


def release_window_claim(claim: Path | None) -> None:
    if claim is not None:
        try:
            claim.unlink(missing_ok=True)
        finally:
            lock = WINDOW_LOCKS.pop(claim.resolve(), None)
            if lock is not None:
                lock.__exit__(None, None, None)


def interval_union(intervals: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def normalized_highlights(
    content: str, duration: Decimal
) -> tuple[list[dict[str, float]], bool, dict[str, Any]]:
    body, think_removed = strip_exact_empty_think(content)
    parsed = json.loads(body)
    if not isinstance(parsed, dict) or set(parsed) != {"highlights"}:
        raise ValueError("Root object must contain only highlights")
    highlights = parsed["highlights"]
    if not isinstance(highlights, list):
        raise ValueError("highlights must be an array")
    intervals: list[list[float]] = []
    for index, item in enumerate(highlights):
        if not isinstance(item, dict) or set(item) != {"start", "end"}:
            raise ValueError(
                f"highlights[{index}] must contain only start/end"
            )
        start = item["start"]
        end = item["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(
                f"highlights[{index}] start/end must be numbers"
            )
        try:
            dec_start = Decimal(str(start))
            dec_end = Decimal(str(end))
        except InvalidOperation as error:
            raise ValueError(
                f"highlights[{index}] is not a decimal number"
            ) from error
        if not dec_start.is_finite() or not dec_end.is_finite():
            raise ValueError(f"highlights[{index}] must be finite")
        if dec_start < 0 or dec_start >= dec_end or dec_end > duration:
            raise ValueError(
                f"highlights[{index}] is outside the window: "
                f"start={dec_start}, end={dec_end}, duration={duration}"
            )
        intervals.append([float(dec_start), float(dec_end)])
    merged = interval_union(intervals)
    audit = {
        "raw_interval_count": len(intervals),
        "merged_interval_count": len(merged),
        "duplicate_or_touching_intervals_merged": len(merged) < len(intervals),
        "stripped_exact_empty_think": think_removed,
    }
    return (
        [{"start": start, "end": end} for start, end in merged],
        think_removed,
        audit,
    )


def strip_exact_empty_think(content: str) -> tuple[str, bool]:
    match = EMPTY_THINK.match(content)
    if match is None:
        return content, False
    return content[match.end() :], True


def validate_highlights(
    content: str, duration: Decimal
) -> tuple[list[dict[str, float]], bool]:
    result, think_removed, _ = normalized_highlights(content, duration)
    return result, think_removed


def map_to_source(
    highlights: list[dict[str, float]],
    *,
    window_start: Decimal,
    source_duration: Decimal,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    intervals: list[list[float]] = []
    audit: list[dict[str, Any]] = []
    for index, item in enumerate(highlights):
        start = window_start + Decimal(str(item["start"]))
        end = window_start + Decimal(str(item["end"]))
        if start < 0 or end > source_duration:
            raise ValueError(
                f"mapped highlight[{index}] is outside source video: "
                f"start={start}, end={end}, duration={source_duration}"
            )
        intervals.append([float(start), float(end)])
        audit.append(
            {
                "window_start_sec": float(window_start),
                "local_start_sec": item["start"],
                "local_end_sec": item["end"],
                "source_start_sec": float(start),
                "source_end_sec": float(end),
            }
        )
    return intervals, audit


def load_plan(
    materialized_root: Path, video_root: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest_path = materialized_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("answer_contract") != "highlights-only/v1":
        raise ValueError("unexpected answer contract")
    samples_by_split = manifest.get("samples")
    if not isinstance(samples_by_split, dict):
        raise ValueError("manifest.samples is missing")
    if not samples_by_split or set(samples_by_split) - {"train", "val", "test"}:
        raise ValueError("unexpected inference sample splits")
    rows_by_window: dict[str, dict[str, Any]] = {}
    source_rows: dict[str, list[dict[str, Any]]] = {}
    windows: list[dict[str, Any]] = []
    for split in samples_by_split:
        rows = read_jsonl(materialized_root / f"{split}.windows.jsonl")
        samples = samples_by_split.get(split)
        if not isinstance(samples, list) or len(rows) != len(samples):
            raise ValueError(f"{split}: row/sample count mismatch")
        expected_rows = int(
            manifest["source_files"][f"{split}_windows"]["rows"]
        )
        if len(rows) != expected_rows:
            raise ValueError(f"{split}: manifest row count mismatch")
        for row, sample in zip(rows, samples):
            if not isinstance(row, dict) or set(row) != REQUIRED_ROW_KEYS:
                raise ValueError(f"{split}: unexpected materialized row schema")
            if not isinstance(sample, dict) or not REQUIRED_SAMPLE_KEYS <= set(
                sample
            ):
                raise ValueError(f"{split}: incomplete manifest sample")
            messages = row["messages"]
            if (
                not isinstance(messages, list)
                or not messages
                or not any(message.get("role") == "user" for message in messages)
            ):
                raise ValueError(f"{split}: expected a user video request")
            videos = row["videos"]
            if not isinstance(videos, list) or len(videos) != 1:
                raise ValueError(f"{split}: expected exactly one window video")
            window_video = Path(videos[0])
            if not window_video.is_absolute():
                window_video = materialized_root / window_video
            if not window_video.is_file():
                raise FileNotFoundError(window_video)
            source_id = str(sample["source_video_id"])
            source_video = video_root / Path(str(sample["source_video"])).name
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            start = Decimal(str(sample["window_start_sec"]))
            end = Decimal(str(sample["window_end_sec"]))
            window_duration = Decimal(str(sample["window_duration_sec"]))
            source_duration = Decimal(str(sample["source_duration_sec"]))
            if (
                start < 0
                or start >= end
                or end > source_duration + Decimal("0.001")
                or abs((end - start) - window_duration)
                > Decimal("0.001")
            ):
                raise ValueError(
                    f"{split}: invalid window endpoints: {source_id} "
                    f"{start}-{end}"
                )
            if source_id in source_rows and source_rows[source_id][
                0
            ]["source_duration_sec"] != sample["source_duration_sec"]:
                raise ValueError(
                    f"{split}: source duration changes across windows: {source_id}"
                )
            window_id = (
                f"{split}_{source_id}_w{int(sample['window_index']):04d}"
            )
            if window_id in rows_by_window:
                raise ValueError(f"duplicate window id: {window_id}")
            request_messages = [
                deepcopy(message)
                for message in messages
                if message.get("role") != "assistant"
            ]
            if not request_messages:
                raise ValueError(f"{split}: no user message remains")
            item = {
                "window_id": window_id,
                "split": split,
                "source_video_id": source_id,
                "source_video": str(source_video),
                "source_duration_sec": float(source_duration),
                "window_index": int(sample["window_index"]),
                "window_start_sec": float(start),
                "window_end_sec": float(end),
                "window_duration_sec": float(window_duration),
                "window_video": str(window_video.resolve()),
                "window_video_sha256": sha256_file(window_video),
                "messages": request_messages,
            }
            windows.append(item)
            rows_by_window[window_id] = item
            source_rows.setdefault(source_id, []).append(item)
    expected_source_ids = set(map(str, manifest.get("source_ids", source_rows)))
    expected_count = sum(manifest.get("source_counts", {}).values())
    if not expected_source_ids or len(source_rows) != expected_count or set(source_rows) != expected_source_ids:
        missing = sorted(expected_source_ids - set(source_rows))
        extra = sorted(set(source_rows) - expected_source_ids)
        raise ValueError(
            f"source coverage mismatch: got={len(source_rows)} "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    for source_id, source_windows in source_rows.items():
        source_windows.sort(key=lambda item: item["window_start_sec"])
        for previous, current in zip(source_windows, source_windows[1:]):
            if current["window_start_sec"] <= previous["window_start_sec"]:
                raise ValueError(
                    f"duplicate/non-increasing window start: {source_id}"
                )
    source_index = []
    for source_id in sorted(source_rows, key=lambda value: (not value.isdecimal(), int(value) if value.isdecimal() else value)):
        source_windows = source_rows[source_id]
        source_index.append(
            {
                "video_id": source_id,
                "source_video": source_windows[0]["source_video"],
                "duration": source_windows[0]["source_duration_sec"],
                "windows": [
                    {
                        "window_id": item["window_id"],
                        "split": item["split"],
                        "window_index": item["window_index"],
                        "window_start_sec": item["window_start_sec"],
                        "window_end_sec": item["window_end_sec"],
                    }
                    for item in source_windows
                ],
            }
        )
    return manifest, windows, rows_by_window, source_index


def build_engine(
    *,
    model_path: str,
    adapter: Path,
    visible_devices: str,
    expect_gpus: int,
    single_gpu: bool,
    max_frames: int,
    max_tokens: int,
    per_gpu_gib: int,
):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=visible_devices,
        FPS=os.environ.get("FPS", "2"),
        FPS_MAX_FRAMES=str(max_frames),
        VIDEO_MAX_TOKEN_NUM=os.environ.get("VIDEO_MAX_TOKEN_NUM", "196"),
        VIDEO_MIN_TOKEN_NUM=os.environ.get("VIDEO_MIN_TOKEN_NUM", "128"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        FORCE_QWENVL_VIDEO_READER="decord",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from swift.infer_engine import RequestConfig, TransformersEngine
    from swift.model import get_model_processor
    from swift.template import get_template

    visible_count = torch.cuda.device_count()
    if visible_count != expect_gpus:
        raise ValueError(
            f"Expected {expect_gpus} visible GPUs, found {visible_count}"
        )
    if single_gpu:
        if expect_gpus != 1:
            raise ValueError("single-gpu requires exactly one visible GPU")
        model, processor = get_model_processor(
            model_path,
            model_type="qwen3_5",
            download_model=False,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_impl="sdpa",
        )
    else:
        model, processor = get_model_processor(
            model_path,
            model_type="qwen3_5",
            download_model=False,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            attn_impl="sdpa",
            model_kwargs={
                "max_memory": {
                    index: f"{per_gpu_gib}GiB"
                    for index in range(expect_gpus)
                }
            },
        )
    model = PeftModel.from_pretrained(
        model, str(adapter), is_trainable=False
    )
    model.eval()
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(model)
    if set(saved) != set(actual) or not all(
        torch.equal(
            actual[key].detach().cpu().to(saved[key].dtype), saved[key]
        )
        for key in saved
    ):
        raise RuntimeError("Exact adapter reload verification failed")
    adapter_tensor_count = len(saved)
    del saved, actual
    engine = TransformersEngine(
        model,
        template=get_template(
            processor,
            template_type="qwen3_5",
            enable_thinking=False,
        ),
    )
    return (
        model,
        engine,
        RequestConfig(
            max_tokens=max_tokens,
            temperature=0,
            seed=42,
            stream=False,
        ),
        adapter_tensor_count,
        {
            "torch": torch.__version__,
            "cuda_device_count": visible_count,
            "visible_devices": visible_devices,
            "device_map": getattr(model, "hf_device_map", None),
        },
    )


def infer_all(
    *,
    model_path: str,
    adapter: Path,
    materialized_root: Path,
    video_root: Path,
    output: Path,
    visible_devices: str,
    expect_gpus: int,
    single_gpu: bool,
    max_frames: int,
    max_tokens: int,
    per_gpu_gib: int,
    reverse: bool,
    start_index: int,
    end_index: int | None,
    resume: bool,
    window_ids: list[str] | None = None,
    window_timeout_sec: float = 0,
) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from video_highlight.submission_recovery import inference_deadline
    adapter_hash = sha256_file(adapter / "adapter_model.safetensors")
    manifest_hash = sha256_file(materialized_root / "manifest.json")
    if output.exists() and not output.is_dir():
        raise FileExistsError(output)
    if output.is_dir() and not resume:
        existing = list(output.iterdir())
        if existing:
            raise FileExistsError(
                f"Output directory is not empty: {output} "
                f"entries={[item.name for item in existing[:8]]}"
            )
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw"
    record_dir = output / "records"
    failed_dir = output / "failed"
    for path in (raw_dir, record_dir, failed_dir):
        path.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    status_path = output / "status.json"
    if plan_path.exists() and not resume:
        raise FileExistsError(plan_path)
    manifest, windows, _, source_index = load_plan(
        materialized_root, video_root
    )
    if not 0 <= start_index < len(windows):
        raise ValueError(
            f"start-index must be within 0..{len(windows) - 1}"
        )
    stop_index = len(windows) if end_index is None else end_index
    if not start_index < stop_index <= len(windows):
        raise ValueError(
            f"end-index must be within {start_index + 1}..{len(windows)}"
        )
    selected_windows = windows[start_index:stop_index]
    if window_ids is not None:
        wanted = set(window_ids)
        if len(wanted) != len(window_ids) or not wanted or wanted - {w["window_id"] for w in windows}:
            raise ValueError("window-ids must be a nonempty, unique subset of the input windows")
        selected_windows = [w for w in windows if w["window_id"] in wanted]
    selected_ids = {w["window_id"] for w in selected_windows}
    selected_sources = [dict(source, windows=[w for w in source["windows"] if w["window_id"] in selected_ids])
                        for source in source_index]
    selected_sources = [source for source in selected_sources if source["windows"]]
    plan = {
        "schema": "video-ora-adapter426-inference/v1",
        "created_at": utc_now(),
        "model": model_path,
        "adapter": str(adapter),
        "adapter_sha256": adapter_hash,
        "materialized_root": str(materialized_root),
        "manifest_sha256": manifest_hash,
        "video_root": str(video_root),
        "windows": len(windows),
        "selected_window_count": len(selected_windows),
        "selected_window_ids": [w["window_id"] for w in selected_windows],
        "window_timeout_sec": window_timeout_sec,
        "start_index": start_index,
        "end_index": stop_index,
        "start_window_id": selected_windows[0]["window_id"],
        "end_window_id": selected_windows[-1]["window_id"],
        "source_videos": len(source_index),
        "splits": {
            "train_windows": sum(w["split"] == "train" for w in windows),
            "val_windows": sum(w["split"] == "val" for w in windows),
        },
        "prompt": windows[0]["messages"],
        "assistant_answers_passed_to_model": False,
        "generation": {
            "video_reader_backend": "decord",
            "fps": float(os.environ.get("FPS", "2")),
            "max_frames": max_frames,
            "video_max_token_num": int(os.environ.get("VIDEO_MAX_TOKEN_NUM", "196")),
            "video_min_token_num": int(os.environ.get("VIDEO_MIN_TOKEN_NUM", "128")),
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "thinking": False,
        },
        "ordered_min_first": not reverse,
        "single_gpu_loading": single_gpu,
        "reuse_policy": "Existing per-window raw/record files are reused only "
        "when their request hash and adapter hash match.",
        "concurrency_policy": "Each window is claimed with an exclusive "
        "OS-owned per-window lock plus pid-hostname receipt. A crash releases "
        "the OS lock; a stale receipt does not block recovery. Live locks are never stolen.",
    }
    if not plan_path.exists():
        write_json(plan_path, plan)
    elif json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        stable_keys = [
            "model",
            "adapter",
            "adapter_sha256",
            "materialized_root",
            "manifest_sha256",
            "video_root",
            "windows",
            "selected_window_count",
            "selected_window_ids",
            "window_timeout_sec",
            "start_index",
            "end_index",
            "start_window_id",
            "end_window_id",
            "source_videos",
            "splits",
            "prompt",
            "assistant_answers_passed_to_model",
            "generation",
            "ordered_min_first",
            "single_gpu_loading",
        ]
        for key in stable_keys:
            if previous.get(key) != plan.get(key):
                raise ValueError(f"resume plan mismatch: {key}")
    write_json(output / "source_index.json", source_index)
    write_json(status_path, {"state": "LOADING_MODEL", "updated_at": utc_now()})
    model, engine, request_config, adapter_tensor_count, environment = build_engine(
        model_path=model_path,
        adapter=adapter,
        visible_devices=visible_devices,
        expect_gpus=expect_gpus,
        single_gpu=single_gpu,
        max_frames=max_frames,
        max_tokens=max_tokens,
        per_gpu_gib=per_gpu_gib,
    )
    import torch
    from swift.infer_engine import InferRequest
    if torch.cuda.is_available():
        for index in range(expect_gpus):
            torch.cuda.reset_peak_memory_stats(index)
    plan["environment"] = environment
    plan["adapter_tensor_count_exact"] = adapter_tensor_count
    write_json(plan_path, plan)
    write_json(
        status_path,
        {
            "state": "running",
            "updated_at": utc_now(),
            "total_windows": len(windows),
            "assigned_windows": len(selected_windows),
            "completed_windows": 0,
            "failed_windows": 0,
        },
    )
    completed = 0
    failed = 0
    started_run = time.monotonic()
    ordered_windows = (
        list(reversed(selected_windows)) if reverse else selected_windows
    )
    for item in ordered_windows:
        raw_path = raw_dir / f"{item['window_id']}.json"
        record_path = record_dir / f"{item['window_id']}.json"
        request_hash = hashlib.sha256(
            json.dumps(
                {"messages": item["messages"], "video": item["window_video"]},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if resume and raw_path.is_file() and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                record.get("request_sha256") == request_hash
                and record.get("window_video_sha256")
                == item["window_video_sha256"]
                and isinstance(record.get("raw_json_valid"), bool)
            ):
                completed += 1
                failed += int(not record["raw_json_valid"])
                continue
        claim = acquire_window_claim(failed_dir, item["window_id"])
        if claim is None:
            continue
        started = time.monotonic()
        record: dict[str, Any] = {
            **item,
            "request_sha256": request_hash,
            "raw_json_valid": False,
            "assistant_answer_passed_to_model": False,
        }
        write_json(status_path, {"state": "running", "active_window_id": item["window_id"],
                                 "active_started_monotonic": time.monotonic(), "updated_at": utc_now()})
        try:
            with inference_deadline(window_timeout_sec):
                response = engine.infer(
                [
                    InferRequest(
                        messages=deepcopy(item["messages"]),
                        videos=[item["window_video"]],
                        chat_template_kwargs={"enable_thinking": False},
                    )
                ],
                request_config,
                )[0]
            choice = response.choices[0]
            raw_content = choice.message.content
            record["raw_response"] = asdict(response)
            record["finish_reason"] = choice.finish_reason
            if choice.finish_reason != "stop":
                raise ValueError(
                    f"Generation did not finish normally: "
                    f"{choice.finish_reason}"
                )
            highlights, think_removed, normalization_audit = normalized_highlights(
                raw_content,
                Decimal(str(item["window_duration_sec"])),
            )
            intervals, interval_audit = map_to_source(
                highlights,
                window_start=Decimal(str(item["window_start_sec"])),
                source_duration=Decimal(str(item["source_duration_sec"])),
            )
            record.update(
                raw_json_valid=True,
                stripped_empty_think=think_removed,
                window_highlights=highlights,
                source_highlights=[
                    {"start": start, "end": end}
                    for start, end in intervals
                ],
                interval_audit=interval_audit,
                highlight_normalization=normalization_audit,
            )
        except Exception as error:
            record.update(
                raw_json_valid=False,
                validation_error=f"{type(error).__name__}: {error}",
                validation_traceback=traceback.format_exc(),
            )
            write_json(record_path, record)
            write_json(
                failed_dir / f"{item['window_id']}.json",
                record,
            )
            failed += 1
        else:
            write_json(record_path, record)
            failure_path = failed_dir / f"{item['window_id']}.json"
            if failure_path.exists():
                failure_path.unlink()
            completed += 1
        finally:
            record["seconds"] = time.monotonic() - started
            write_json(raw_path, record)
            release_window_claim(claim)
        write_json(
            status_path,
            {
                "state": "running",
                "updated_at": utc_now(),
                "total_windows": len(windows),
                "assigned_windows": len(selected_windows),
                "processed_windows": completed + failed,
                "completed_windows": completed,
                "failed_windows": failed,
                "last_window_id": item["window_id"],
                "last_window_valid": record["raw_json_valid"],
                "elapsed_seconds": time.monotonic() - started_run,
            },
        )
        print(
            json.dumps(
                {
                    "window_id": item["window_id"],
                    "valid": record["raw_json_valid"],
                    "processed": completed + failed,
                    "total": len(windows),
                    "seconds": round(record["seconds"], 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    records_by_source: dict[str, list[dict[str, Any]]] = {
        item["video_id"]: [] for item in selected_sources
    }
    for path in record_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["window_id"] not in selected_ids:
            raise ValueError(f"record outside assigned window set: {record['window_id']}")
        video_id = str(record["source_video_id"])
        if video_id not in records_by_source:
            raise ValueError(f"record has unknown source video: {video_id}")
        records_by_source[video_id].append(record)
    by_source: dict[str, list[list[float]]] = {
        item["video_id"]: [] for item in selected_sources
    }
    failures = []
    for source_id in by_source:
        source_windows = sorted(
            records_by_source[source_id],
            key=lambda record: record["window_start_sec"],
        )
        expected_source = next(
            item for item in selected_sources if item["video_id"] == source_id
        )
        if [
            window["window_id"] for window in expected_source["windows"]
        ] != [record["window_id"] for record in source_windows]:
            raise ValueError(f"missing or mismatched window records: {source_id}")
        for record in source_windows:
            if not record.get("raw_json_valid"):
                failures.append(
                    {
                        "video_id": source_id,
                        "window_id": record["window_id"],
                        "error": record.get("validation_error"),
                    }
                )
                continue
            by_source[source_id].extend(
                [
                    [interval["start"], interval["end"]]
                    for interval in record["source_highlights"]
                ]
            )
    highlight_rows = []
    for item in selected_sources:
        intervals = interval_union(by_source[item["video_id"]])
        highlight_rows.append(
            {
                "video_id": item["video_id"],
                "duration": item["duration"],
                "highlights": [
                    {"start": start, "end": end}
                    for start, end in intervals
                ],
            }
        )
    for name, rows in (
        ("highlights.jsonl", highlight_rows),
        ("failures.jsonl", failures),
    ):
        path = output / name
        if path.exists():
            if not resume:
                raise FileExistsError(path)
            path.unlink()
        write_jsonl(path, rows)
    peak = {}
    if torch.cuda.is_available():
        for index in range(expect_gpus):
            peak[index] = {
                "allocated_gib": torch.cuda.max_memory_allocated(index)
                / 1024**3,
                "reserved_gib": torch.cuda.max_memory_reserved(index)
                / 1024**3,
            }
    summary = {
        "state": "complete" if not failures else "partial_with_failures",
        "updated_at": utc_now(),
        "model": model_path,
        "adapter": str(adapter),
        "adapter_sha256": plan["adapter_sha256"],
        "assistant_answers_passed_to_model": False,
        "source_videos": len(selected_sources),
        "windows": len(windows),
        "assigned_windows": len(selected_windows),
        "valid_windows": sum(
            bool(record.get("raw_json_valid"))
            for records in records_by_source.values()
            for record in records
        ),
        "failed_windows": sum(
            not record.get("raw_json_valid")
            for records in records_by_source.values()
            for record in records
        ),
        "windows_with_nonempty_highlights": sum(
            bool(record.get("window_highlights"))
            for records in records_by_source.values()
            for record in records
            if record.get("raw_json_valid")
        ),
        "source_videos_with_highlights": sum(
            bool(row["highlights"]) for row in highlight_rows
        ),
        "total_merged_highlights": sum(
            len(row["highlights"]) for row in highlight_rows
        ),
        "peak_memory": peak,
        "elapsed_seconds": time.monotonic() - started_run,
        "failures": failures,
        "record_files_sha256": {path.name: sha256_file(path) for path in sorted(record_dir.glob("*.json"))},
    }
    write_json(output / "summary.json", summary)
    write_json(
        status_path,
        {
            "state": summary["state"],
            "updated_at": utc_now(),
            "total_windows": len(windows),
            "assigned_windows": len(selected_windows),
            "processed_windows": completed + failed,
            "completed_windows": completed,
            "failed_windows": failed,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--expect-gpus", type=int, required=True)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--per-gpu-gib", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--window-ids", nargs="+", help="Exact sparse window IDs; never widen to an index range")
    parser.add_argument("--window-timeout-sec", type=float, default=0)
    args = parser.parse_args()
    if not 1 <= args.expect_gpus <= 4:
        raise ValueError("expect-gpus must be between 1 and 4")
    visible = [item.strip() for item in args.visible_devices.split(",")]
    if len(visible) != args.expect_gpus:
        raise ValueError("visible-devices count must match expect-gpus")
    summary = infer_all(
        model_path=args.model,
        adapter=args.adapter,
        materialized_root=args.materialized_root,
        video_root=args.video_root,
        output=args.output,
        visible_devices=args.visible_devices,
        expect_gpus=args.expect_gpus,
        single_gpu=args.single_gpu,
        max_frames=args.max_frames,
        max_tokens=args.max_tokens,
        per_gpu_gib=args.per_gpu_gib,
        reverse=args.reverse,
        start_index=args.start_index,
        end_index=args.end_index,
        resume=args.resume,
        window_ids=args.window_ids,
        window_timeout_sec=args.window_timeout_sec,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
