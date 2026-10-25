"""Fault injection: model calls are simulated, never competition-score evidence."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from video_highlight.submission_recovery import center_box, recover_intervals, repair_spatial_row
from test_submission_portability import example, args_for, put, put_rows
from video_highlight import submission_inputs as inputs
import infer_video_ora_highlights as inference
import merge_video_ora_highlights_inference as merger
import run_submission_pipeline as pipeline
import run_spatial_submission as spatial


@pytest.mark.parametrize("excess", [0.04, 5, 50])
def test_end_overflow_intersection_not_a_larger_tolerance(excess):
    raw = json.dumps({"highlights": [{"start": 27, "end": 32 + excess}]})
    obj, spans, audit = recover_intervals(raw, 32)
    assert spans == [[27, 32]]
    assert audit["time_changes"][0]["clipped_seconds"] == pytest.approx(excess)


def test_bad_entries_isolated_and_numeric_strings_auditable():
    raw = '{"highlights":[{"start":-5,"end":8},{"start":20,"end":10},' \
          '{"start":NaN,"end":8},{"start":9},{"start":36,"end":40},' \
          '{"start":"10","end":"12"}]}'
    _, spans, audit = recover_intervals(raw, 32)
    assert spans == [[0, 8], [10, 12]]
    assert len(audit["discarded_entries"]) == 4
    assert audit["status"] == "partial"


@pytest.mark.parametrize("raw", ["", "No answer; 10 to 12 seconds", '{"highlights":[{"start":10}]}',
                                '{"highlights":[{"start":20,"end":10}]}',
                                '{"highlights":[{"start":36,"end":40}]}'])
def test_invalid_answer_never_becomes_model_no_highlights(raw):
    with pytest.raises(ValueError):
        recover_intervals(raw, 32)


def test_explicit_empty_and_truncated_complete_entries_distinguished():
    assert recover_intervals('```json\n{"highlights":[]}\n```', 32)[2]["explicit_no_highlights"]
    _, spans, audit = recover_intervals('{"highlights":[{"start":1,"end":2},{"start":3', 32)
    assert spans == [[1, 2]] and audit["truncated"] and audit["status"] == "partial"
    with pytest.raises(ValueError):
        recover_intervals('{"highlights":[{"start":1,"end":2', 32)


@pytest.mark.parametrize("ratio", [[16, 9], [9, 16], [1, 1]])
def test_center_crop_matches_official_three_value_protocol(ratio):
    x, y, width = center_box([96, 64], ratio)
    height = round(width * ratio[1] / ratio[0])
    assert width > 0 and height > 0 and x >= 0 and y >= 0
    assert x + width <= 96 and y + height <= 64
    assert width == 96 or round((width + 1) * ratio[1] / ratio[0]) > 64


def test_spatial_repair_freezes_frames_and_keeps_valid_boxes():
    first = {"video_id": "x", "targetRatioWH": [16, 9], "frame_count": 10, "segments_frame": [[2, 4]]}
    good = {"video_id": "x", "targetRatioWH": [16, 9], "predictions": [
        {"frame": frame, "bboxes": [1, 2, 32]} for frame in (2, 3, 4)]}
    repaired, frames = repair_spatial_row(good, first, [96, 64])
    assert repaired == good and frames == []
    good["predictions"][1]["bboxes"] = [999, -1, 500]
    good["predictions"].extend([{"frame": 2, "bboxes": [0, 0, 1]}, {"frame": 8, "bboxes": [0, 0, 1]}])
    repaired, frames = repair_spatial_row(good, first, [96, 64])
    assert [p["frame"] for p in repaired["predictions"]] == [2, 3, 4]
    assert frames == [2, 3]
    assert repaired["predictions"][2]["bboxes"] == [1, 2, 32]


@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message
    finish_reason: str = "stop"


@dataclass
class Response:
    choices: list


def fake_engine(monkeypatch, content='{"highlights":[{"start":0,"end":0.5}]}'):
    engine = SimpleNamespace(infer=lambda *a: [Response([Choice(Message(content))])])
    monkeypatch.setattr(inference, "build_engine", lambda **kw: (None, engine, None, 1, {"simulated": True}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "swift.infer_engine", SimpleNamespace(InferRequest=lambda **kw: kw))


def strict_with_sparse_failures(example, *, train_val=False):
    args = args_for(example)
    args.materialized = inputs.prepare_inputs(args)
    args.index = args.materialized / "test_index.json"
    args.python = Path(sys.executable)
    args.inference_gpus, args.inference_max_tokens, args.per_gpu_gib = "0", 512, 8
    args.end_tolerance_sec, args.recovery_policy = 0.5, "intersection"
    args.recovery_attempts, args.unresolved_policy = 1, "drop_windows"
    if train_val:
        manifest = inputs.read_json(args.materialized / "manifest.json")
        requests = pipeline.read_jsonl(args.materialized / "test.windows.jsonl")
        samples = manifest["samples"]["test"]
        manifest.update(samples={"train": samples[:2], "val": samples[2:]},
                        source_counts={"train": 1, "val": 1},
                        source_files={"train_windows": {"rows": 2}, "val_windows": {"rows": 2}})
        put(args.materialized / "manifest.json", manifest)
        put_rows(args.materialized / "train.windows.jsonl", requests[:2])
        put_rows(args.materialized / "val.windows.jsonl", requests[2:])
    _, windows, _, sources = inference.load_plan(args.materialized, args.videos)
    root = example.root / "run"
    original = root / "01-infer/run"
    put(original / "plan.json", {})
    put(original / "source_index.json", sources)
    wanted = {windows[i]["window_id"] for i in (0, 2)}
    for w in windows:
        valid = w["window_id"] not in wanted
        put(original / "records" / (w["window_id"] + ".json"), {
            **w, "raw_json_valid": valid, "validation_error": None if valid else "simulated OOM",
            "source_highlights": [{"start": w["window_start_sec"], "end": w["window_start_sec"] + 0.5}] if valid else []})
    strict = root / "02-merge/merged"
    merger.merge(argparse.Namespace(runs=[original], output=strict, allow_missing=False))
    return args, root, strict, wanted


@pytest.mark.parametrize("retry_content", ['{"highlights":[{"start":0,"end":0.5}]}',
                                          '{"highlights":[{"start":0,"end":50}]}'])
@pytest.mark.parametrize("train_val", [False, True])
def test_real_sparse_retry_and_aggregate_state_closure(example, monkeypatch, retry_content, train_val):
    args, root, strict, wanted = strict_with_sparse_failures(example, train_val=train_val)
    before_retry_summary = inputs.read_json(strict / "summary.json")
    fake_engine(monkeypatch, retry_content)
    original_command = pipeline.run_command
    assignments = []

    def command(*, name, command, output, environment=None, **budget):
        if name != "retry":
            return original_command(name=name, command=command, output=output, environment=environment, **budget)
        assert "--start-index" not in command and "--end-index" not in command
        selected = command[command.index("--window-ids") + 1:]
        assignments.append(selected)
        shard = Path(command[command.index("--output") + 1])
        inference.infer_all(model_path=str(args.model), adapter=args.adapter, materialized_root=args.materialized,
            video_root=args.videos, output=shard, visible_devices="0", expect_gpus=1, single_gpu=True,
            max_frames=64, max_tokens=512, per_gpu_gib=8, reverse=False, start_index=0, end_index=None,
            resume=True, window_ids=selected)
        return {}

    monkeypatch.setattr(pipeline, "run_command", command)
    pipeline.stage_lenient(args, root, execute=True)
    assert set(assignments[0]) == wanted and len(assignments) == 1
    receipt = inputs.read_json(root / "02b-retry/RETRY.json")
    assert set(receipt["record_paths"]) == wanted
    updated = pipeline.read_jsonl(strict / "windows.jsonl")
    assert {w["window_id"] for w in updated} >= wanted
    assert all("02b-retry" in w["record_path"] for w in updated if w["window_id"] in wanted)
    summary = inputs.read_json(root / "03-lenient/merged/summary.json")
    assert summary["unresolved_windows"] == 0
    assert summary["valid_windows_after_lenient"] == 4
    if "50" not in retry_content:
        assert pipeline.read_jsonl(strict / "failures.jsonl") == []
        assert inputs.read_json(strict / "summary.json")["valid_windows"] == 4
    pipeline.stage_lenient(args, root, execute=True)
    assert len(assignments) == 1  # Receipt closes the budget across resumes.
    if "50" not in retry_content:
        # Simulate interruption after writing new windows but before publishing
        # their summary. The controller replays the receipt, not model calls.
        put(strict / "summary.json", before_retry_summary)
        pipeline.stage_merge(args, root, execute=True)
        pipeline.stage_lenient(args, root, execute=True)
        assert len(assignments) == 1 and pipeline.read_jsonl(strict / "failures.jsonl") == []


def test_unrecoverable_retry_has_one_budget_and_separate_failed_state(example, monkeypatch):
    args, root, strict, wanted = strict_with_sparse_failures(example)
    fake_engine(monkeypatch, "not JSON")
    original = pipeline.run_command
    calls = []
    def command(*, name, command, output, environment=None, **budget):
        if name != "retry":
            return original(name=name, command=command, output=output, environment=environment, **budget)
        calls.append(name)
        selected = command[command.index("--window-ids") + 1:]
        inference.infer_all(model_path=str(args.model), adapter=args.adapter, materialized_root=args.materialized,
            video_root=args.videos, output=Path(command[command.index("--output") + 1]), visible_devices="0",
            expect_gpus=1, single_gpu=True, max_frames=64, max_tokens=512, per_gpu_gib=8, reverse=False,
            start_index=0, end_index=None, resume=True, window_ids=selected)
        return {}
    monkeypatch.setattr(pipeline, "run_command", command)
    pipeline.stage_lenient(args, root, execute=True)
    pipeline.stage_lenient(args, root, execute=True)
    assert calls == ["retry"]
    states = inputs.read_json(root / "03-lenient/TERMINAL_STATES.json")["videos"]
    assert all(row["state"] == "partial" for row in states)
    assert inputs.read_json(root / "03-lenient/FALLBACKS.json")["source_video_count"] == 2


def test_wall_clock_timeout_reaps_owned_process_and_records_failure(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        pipeline.run_command(name="hang", command=[sys.executable, "-u", "-c", "import time; print('started'); time.sleep(20)"],
                             output=tmp_path, timeout_sec=0.3)
    receipt = inputs.read_json(tmp_path / "COMMAND.json")
    assert receipt["return_code"] is not None and receipt["state"] == "INTERRUPTED"
    assert time.monotonic() - started < 5


def test_killed_process_claim_does_not_block_resume(tmp_path):
    script_dir = Path(inference.__file__).parent
    code = ("import sys,time; from pathlib import Path; "
            f"sys.path.insert(0,{str(script_dir)!r}); import infer_video_ora_highlights as m; "
            f"print(m.acquire_window_claim(Path({str(tmp_path)!r}),'test_x_w0000'),flush=True); time.sleep(20)")
    process = subprocess.Popen([sys.executable, "-B", "-u", "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert ".claim" in process.stdout.readline()
        assert inference.acquire_window_claim(tmp_path, "test_x_w0000") is None
    finally:
        process.terminate()
        process.wait(timeout=5)
    claim = inference.acquire_window_claim(tmp_path, "test_x_w0000")
    assert claim is not None
    inference.release_window_claim(claim)


@pytest.mark.parametrize("failure_policy", [None, "error", "center"])
def test_worker_model_error_falls_back_only_failed_video(example, monkeypatch, failure_policy):
    args = args_for(example)
    materialized = inputs.prepare_inputs(args)
    metadata = inputs.prepared_metadata(materialized)
    firsts = [{"video_id": row["video_id"], "source_sha256": row["source_sha256"],
               "targetRatioWH": row["target_ratio"], "frame_count": row["frame_count"],
               "valid": True, "segments_frame": [[0, 2]]} for row in metadata]
    stage1 = example.root / "stage1.jsonl"
    put_rows(stage1, firsts)
    def stage2(ctx, video, first, size):
        if video["video_id"] == "81":
            raise RuntimeError("simulated CUDA OOM")
        return {"video_id": first["video_id"], "targetRatioWH": first["targetRatioWH"],
                "predictions": [{"frame": f, "bboxes": [0, 0, 32]} for f in range(3)]}
    common = SimpleNamespace(fingerprint=inputs.identity, write=put, entrypoint=lambda spec: stage2,
                             validate_stage1=lambda row: None, validate_stage2=lambda *args: None)
    monkeypatch.setattr(spatial, "load_module", lambda *args: common)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                                            set_num_threads=lambda count: None))
    output = example.root / "spatial work"
    plan = example.root / "worker.json"
    put(plan, {"package": str(example.spatial), "config": inputs.read_json(example.spatial / "configs/v7e_b_4109.json"),
               "code_fingerprint": inputs.identity(spatial.python_code_fingerprint(example.spatial, ("pipeline", "modules", "vendor"))),
               "output": str(output), "stage1_file": str(stage1), "resume": True,
               "run_fingerprint": "test", **({"failure_policy": failure_policy} if failure_policy is not None else {}),
               "videos": [{"video_id": r["video_id"], "path": str(args.videos / (r["video_id"] + ".mp4")),
                           "sha256": r["source_sha256"]} for r in metadata]})
    assert spatial.run_worker(plan) == 0
    assert inputs.read_json(output / "per_video/81/DONE.json")["state"] == "fallback"
    assert "simulated CUDA OOM" in inputs.read_json(output / "per_video/81/ERROR.json")["error"]
    assert inputs.read_json(output / "per_video/scene_A/DONE.json")["state"] == "ok"
    for first in firsts:
        row = inputs.read_json(output / "per_video" / first["video_id"] / "stage2.json")
        assert [p["frame"] for p in row["predictions"]] == [0, 1, 2]


@pytest.mark.parametrize("failure_policy", [None, "error", "center"])
def test_final_repair_missing_video_bad_box_and_stale_zip(example, failure_policy):
    args = args_for(example)
    args.materialized = inputs.prepare_inputs(args)
    args.index = args.materialized / "test_index.json"
    if failure_policy is not None:
        args.spatial_failure_policy = failure_policy
    metadata = inputs.prepared_metadata(args.materialized)
    root = example.root / "run"
    put(root / "PIPELINE.json", {"pipeline_id": "cpu-test"})
    firsts = [{"video_id": row["video_id"], "source_sha256": row["source_sha256"], "targetRatioWH": row["target_ratio"],
               "frame_count": row["frame_count"], "segments_frame": [[1, 3]], "valid": True} for row in metadata]
    put_rows(root / "04-handoff/temporal_handoff.jsonl", firsts)
    put(root / "04-handoff/HANDOFF.json", {"sha256": "fixture"})
    put(root / "03-lenient/merged/summary.json", {})
    put_rows(root / "05-spatial/predictions.jsonl", [{"video_id": "81", "targetRatioWH": [16, 9],
              "predictions": [{"frame": 1, "bboxes": [0, 0, 32]}, {"frame": 2, "bboxes": [999, 0, 5]}]}])
    old_zip = root / "06-submission/submission.zip"
    old_zip.parent.mkdir(parents=True)
    old_zip.write_bytes(b"interrupted ZIP")
    result = pipeline.stage_submission(args, root, execute=True)
    rows = pipeline.read_jsonl(result)
    assert [r["video_id"] for r in rows] == ["81", "scene_A"]
    assert all([p["frame"] for p in r["predictions"]] == [1, 2, 3] for r in rows)
    assert rows[0]["predictions"][0]["bboxes"] == [0, 0, 32]
    with ZipFile(old_zip) as archive:
        assert archive.testzip() is None
    assert pipeline.stage_submission(args, root, execute=True) == result
    assert inputs.read_json(root / "06-submission/REPAIRS.json")["policy"] == "center"
    firsts[0]["targetRatioWH"] = [1, 1]
    put_rows(root / "04-handoff/temporal_handoff.jsonl", firsts)
    with pytest.raises(pipeline.PipelineError, match="ratio differs"):
        pipeline.stage_submission(args, root, execute=True)


@pytest.mark.parametrize("failure_policy", ["error", "center"])
def test_completed_spatial_reexport_needs_no_model_and_replaces_broken_zip(example, failure_policy):
    args = args_for(example)
    materialized = inputs.prepare_inputs(args)
    metadata = inputs.prepared_metadata(materialized)
    firsts = [{"video_id": row["video_id"], "source_sha256": row["source_sha256"], "targetRatioWH": row["target_ratio"],
               "frame_count": row["frame_count"], "segments_frame": [[1, 2]], "valid": True} for row in metadata]
    stage1, output = example.root / "handoff.jsonl", example.root / "completed spatial output"
    put_rows(stage1, firsts)
    common = SimpleNamespace(write=put, validate_stage1=lambda row: None, validate_stage2=lambda *args: None)
    videos = []
    for row, first in zip(metadata, firsts):
        folder = output / "per_video" / row["video_id"]
        second, repaired = repair_spatial_row({}, first, row["source_size"])
        if first["video_id"] == "81":
            second["predictions"][0]["bboxes"] = [999, 0, 5]
        spatial.save_video(common, folder, first, second, row["source_size"], "test", state="ok")
        videos.append({"video_id": row["video_id"], "sha256": row["source_sha256"], "ratio": row["target_ratio"], "source_metadata": row})
    (output / "submission.zip").write_bytes(b"truncated ZIP from interrupted attempt")
    assert spatial.export_spatial(common, videos, stage1, output, "test", failure_policy) == 0
    assert inputs.read_json(output / "per_video/81/DONE.json")["state"] == "repaired"
    assert inputs.read_json(output / "per_video/scene_A/DONE.json")["state"] == "ok"
    with ZipFile(output / "submission.zip") as archive:
        assert archive.testzip() is None and archive.namelist() == ["predictions_spatial.jsonl"]
    (output / "COMPLETE.json").unlink()
    assert spatial.export_spatial(common, videos, stage1, output, "test", failure_policy) == 0
    assert inputs.read_json(output / "COMPLETE.json")["frames"] == 4


def test_inference_deployment_error_is_not_all_empty(example, monkeypatch):
    args = args_for(example)
    def failed(**kwargs):
        raise pipeline.PipelineError("Missing CUDA dependency during model initialization")
    monkeypatch.setattr(pipeline, "run_command", failed)
    output = example.root / "failed init"
    with pytest.raises(pipeline.PipelineError, match="Missing CUDA"):
        pipeline.run_inference_command(args, [], example.root / "stage", output, {}, name="infer")
    assert not (output / "summary.json").exists()
    assert not (output / "records").exists()


def test_all_failed_video_not_labeled_no_highlights_through_handoff(example):
    args, root, strict, wanted = strict_with_sparse_failures(example)
    original = root / "01-infer/run"
    records = list((original / "records").glob("*.json"))
    for path in records:
        value = inputs.read_json(path)
        if value["source_video_id"] == "81":
            value.update(raw_json_valid=False, source_highlights=[], validation_error="simulated inference failure")
            put(path, value)
    merger.merge(argparse.Namespace(runs=[original], output=strict, allow_missing=False))
    args.recovery_attempts = 0
    pipeline.stage_lenient(args, root, execute=True)
    states = {r["video_id"]: r["state"] for r in inputs.read_json(root / "03-lenient/TERMINAL_STATES.json")["videos"]}
    assert states == {"81": "inference_failed", "scene_A": "partial"}
    handoff = pipeline.read_jsonl(pipeline.stage_handoff(args, root, execute=True))
    assert handoff[0]["segments_frame"] == []
    assert handoff[0]["temporal_status"] == handoff[0]["fallback_reason"] == "inference_failed"
    _, _, done = spatial.build_empty_spatial_outputs(handoff[0], [96, 64], "test")
    assert done["state"] == "inference_failed"


@pytest.mark.parametrize("tamper", [False, "adapter", "request"])
def test_seed_import_checks_provenance_and_request_not_just_counts(example, tamper):
    args, root, strict, wanted = strict_with_sparse_failures(example)
    original = root / "01-infer/run"
    put(original / "plan.json", {"adapter_sha256": "wrong" if tamper == "adapter" else inputs.digest(args.adapter / "adapter_model.safetensors")})
    if tamper == "request":
        path = next((original / "records").glob("*.json"))
        value = inputs.read_json(path)
        value["window_video_sha256"] = "wrong-current-video"
        put(path, value)
    args.seed_strict_merged = strict
    target = example.root / "imported run"
    if tamper:
        with pytest.raises(pipeline.PipelineError, match="seed"):
            pipeline.stage_merge(args, target, execute=True)
        assert not (target / "02-merge/merged/summary.json").exists()
    else:
        result = pipeline.stage_merge(args, target, execute=True)
        assert inputs.read_json(result)["total_windows"] == 4
        assert inputs.digest(target / "02-merge/merged/windows.jsonl") == inputs.digest(strict / "windows.jsonl")


def test_hung_request_process_is_reaped_and_remaining_windows_resume(example):
    args = args_for(example)
    args.materialized = inputs.prepare_inputs(args)
    args.command_timeout_sec, args.window_timeout_sec = 15, 0.3
    output, stage = example.root / "inference", example.root / "controller"
    code = f"""
import sys,time
from pathlib import Path
sys.path.insert(0, {str(Path(inference.__file__).parent)!r})
import infer_video_ora_highlights as m
out = Path({str(output)!r})
_, windows, _, _ = m.load_plan(Path({str(args.materialized)!r}), Path({str(args.videos)!r}))
for index, item in enumerate(windows):
    path = out / 'records' / (item['window_id'] + '.json')
    if path.is_file():
        continue
    m.write_json(out / 'status.json', {{'active_window_id': item['window_id'], 'active_started_monotonic': time.monotonic()}})
    if index == 0:
        time.sleep(20)
    m.write_json(path, {{**item, 'raw_json_valid': True, 'source_highlights': []}})
m.write_json(out / 'status.json', {{'state': 'complete'}})
m.write_json(out / 'summary.json', {{'state': 'partial_with_failures'}})
"""
    pipeline.run_inference_command(args, [sys.executable, "-B", "-u", "-c", code], stage, output,
                                   pipeline.command_env(), name="infer")
    records = [inputs.read_json(path) for path in (output / "records").glob("*.json")]
    assert len(records) == 4 and sum(r["raw_json_valid"] for r in records) == 3
    failed = next(r for r in records if not r["raw_json_valid"])
    assert "WindowTimeoutError" in failed["validation_error"]
    assert failed["request_sha256"] and not failed["assistant_answer_passed_to_model"]
    commands = [inputs.read_json(path) for path in (stage / "commands").glob("*/COMMAND.json")]
    assert len(commands) == 2 and commands[0]["return_code"] is not None


@pytest.mark.parametrize("failure_policy", [None, "error"])
def test_spatial_worker_process_crash_preserves_other_worker_result(example, monkeypatch, failure_policy):
    args = args_for(example)
    materialized = inputs.prepare_inputs(args)
    metadata = inputs.prepared_metadata(materialized)
    firsts = [{"video_id": row["video_id"], "source_sha256": row["source_sha256"], "targetRatioWH": row["target_ratio"],
               "frame_count": row["frame_count"], "segments_frame": [[1, 3]], "valid": True} for row in metadata]
    stage1, output = example.root / "handoff.jsonl", example.root / "spatial-worker-crash"
    put_rows(stage1, firsts)
    common = SimpleNamespace(write=put, fingerprint=inputs.identity, entrypoint=lambda spec: object(),
                             validate_stage1=lambda row: None, validate_stage2=lambda *args: None)
    monkeypatch.setattr(spatial, "load_module", lambda *args: common)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: object()))
    actual_popen = subprocess.Popen
    def popen(command, **options):
        plan = inputs.read_json(Path(command[-1]))
        video = plan["videos"][0]
        if video["video_id"] == "scene_A":
            first = firsts[1]
            second = {"video_id": "scene_A", "targetRatioWH": [9, 16],
                      "predictions": [{"frame": f, "bboxes": [0, 0, 32]} for f in range(1, 4)]}
            spatial.save_video(common, output / "per_video/scene_A", first, second, metadata[1]["source_size"], plan["run_fingerprint"])
        return actual_popen([sys.executable, "-B", "-c", "raise SystemExit(" + ("7" if video["video_id"] == "81" else "0") + ")"], **options)
    monkeypatch.setattr(spatial.subprocess, "Popen", popen)
    argv = [str(Path(spatial.__file__)), "--package", str(example.spatial),
        "--videos", str(args.videos), "--output", str(output), "--stage1-file", str(stage1),
        "--source-metadata", str(materialized / "source_metadata.json"), "--gpus", "0,1", "--resume"]
    if failure_policy is not None:
        argv.extend(["--failure-policy", failure_policy])
    monkeypatch.setattr(sys, "argv", argv)
    assert spatial.main() == 0
    result = pipeline.read_jsonl(output / "predictions_spatial.jsonl")
    assert len(result) == 2 and all([p["frame"] for p in row["predictions"]] == [1, 2, 3] for row in result)
    assert all(p["bboxes"] == [0, 0, 32] for p in result[1]["predictions"])
    assert inputs.read_json(output / "per_video/81/DONE.json")["state"] == "fallback"
    assert inputs.read_json(output / "per_video/scene_A/DONE.json")["state"] == "ok"
    assert inputs.read_json(output / "RECOVERY.json")["policy"] == "center"
    assert pipeline.validate_submission(str(output / "predictions_spatial.jsonl"), str(materialized / "test_index.json"), str(args.videos))[0]
