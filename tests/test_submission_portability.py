"""CPU/media engineering tests; dummy weights/models are never quality evidence."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from video_highlight import submission_inputs as inputs
import infer_video_ora_highlights_adapter_426 as inference
import package_submission as packaging
import run_submission_pipeline as pipeline
import run_v7e_stage2_submission as spatial_runner


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def put_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def example(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Real CPU media test requires FFmpeg")
    data = tmp_path / "new test set"
    (data / "video").mkdir(parents=True)
    subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-n",
                    "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=8:duration=3",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
                    str(data / "video/81.mp4")], check=True)
    shutil.copyfile(data / "video/81.mp4", data / "video/scene_A.mp4")
    for video_id, ratio in (("81", [16, 9]), ("scene_A", [9, 16])):
        put_rows(data / "jsonl" / f"{video_id}.jsonl", [{"video_id": video_id, "targetRatioWH": ratio}])
    model = tmp_path / "model"
    put(model / "config.json", {"model_type": "dummy-cpu-fixture"})
    for name in ("tokenizer_config.json", "tokenizer.json", "processor_config.json"):
        put(model / name, {})
    (model / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (model / "chat_templates").mkdir()
    (model / "chat_templates/default.jinja").write_text("{{ messages }}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"dummy-base-for-packaging-tests-not-a-model")
    adapter = tmp_path / "adapter"
    put(adapter / "adapter_config.json", {"base_model_name_or_path": "/old/server/model"})
    (adapter / "adapter_model.safetensors").write_bytes(b"dummy-adapter")
    package = tmp_path / "spatial"
    for file in packaging.SPATIAL_CODE:
        path = package / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Dummy CPU test fixture\n", encoding="utf-8")
    weights = package / "assets/yolo11m.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"dummy-yolo")
    put(package / "configs/v7e_b_4109.json", {"schema": "video-highlight-pipeline/v1",
        "pipeline_version": "cpu-fixture", "stage2": {"entrypoint": "modules.stage2_b:run"},
        "assets": {"yolo": {"path": "assets/yolo11m.pt", "sha256": inputs.digest(weights)}}})
    put(package / "RELEASE_MANIFEST.json", {})
    config = tmp_path / "config" / "experiment.json"
    put(config, {"schema": "video-highlight-submission-config/v1", "input_root": "../new test set",
                 "model": "../model", "adapter": "../adapter", "spatial_package": "../spatial",
                 "pipeline_dir": "../runs", "window_seconds": 2.0, "window_stride": 1.0})
    return SimpleNamespace(root=tmp_path, data=data, model=model, adapter=adapter, spatial=package, config=config)


def args_for(example):
    options = inputs.resolve_config(example.config)
    options.update(videos=example.data / "video", index=example.data / "jsonl", fps=2.0,
                   max_frames=64, video_min_token_num=128, video_max_token_num=196,
                   prompt=pipeline.REPO / "prompts/ora/whole-video-highlights-v1/prompt.txt",
                   ffmpeg=inputs.executable("ffmpeg"), ffprobe=inputs.executable("ffprobe"))
    return argparse.Namespace(**options)


def test_config_is_relative_to_its_file_not_cwd(example, monkeypatch):
    elsewhere = example.root / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    value = inputs.resolve_config(example.config)
    assert value["model"] == example.model
    assert value["input_root"] == example.data


def test_dynamic_input_counts_and_missing_annotations(example):
    args = args_for(example)
    first = inputs.prepare_inputs(args)
    manifest = inputs.read_json(first / "manifest.json")
    assert manifest["source_counts"] == {"test": 2}
    assert manifest["source_ids"] == ["81", "scene_A"]
    assert len(manifest["samples"]["test"]) == 4
    _, windows, _, sources = inference.load_plan(first, args.videos)
    assert len(sources) == 2 and len(windows) == 4
    assert all(message["role"] == "user" for item in windows for message in item["messages"])
    assert all(not Path(row["videos"][0]).is_absolute() for row in inference.read_jsonl(first / "test.windows.jsonl"))
    assert inputs.prepare_inputs(args) == first
    put_rows(args.index / "scene_A.jsonl", [{"video_id": "scene_A", "targetRatioWH": [16, 9]}])
    assert inputs.prepare_inputs(args) != first


def test_prepared_clip_tampering_is_rejected(example):
    args = args_for(example)
    output = inputs.prepare_inputs(args)
    next((output / "clips").glob("*.mp4")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        inputs.prepare_inputs(args)


@pytest.mark.parametrize("rotation_metadata", [{"tags": {"rotate": "90"}},
                                               {"side_data_list": [{"rotation": -90}]}])
def test_probe_preserves_rotation_checks_without_new_ffprobe_sections(monkeypatch, rotation_metadata):
    def probe(command, **kwargs):
        assert "-show_streams" in command and "-show_entries" not in command
        return json.dumps({"streams": [{"width": 96, "height": 64, **rotation_metadata}]})

    monkeypatch.setattr(inputs.subprocess, "check_output", probe)
    with pytest.raises(ValueError, match="Rotated source"):
        inputs.probe_video(Path("rotated.mp4"), Path("ffprobe"))


def test_probe_rejects_real_rotated_media_with_available_ffprobe(example):
    rotated = example.data / "rotated.mp4"
    # FFmpeg 8 no longer writes rotation from the legacy metadata option;
    # older FFmpeg versions do not yet implement display_rotation.
    modern = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-display_rotation:v:0", "90",
                             "-i", str(example.data / "video/81.mp4"), "-c", "copy", str(rotated)],
                            capture_output=True)
    if modern.returncode:
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-y", "-i", str(example.data / "video/81.mp4"),
                        "-c", "copy", "-metadata:s:v:0", "rotate=90", str(rotated)], check=True)
    with pytest.raises(ValueError, match="Rotated source"):
        inputs.probe_video(rotated, inputs.executable("ffprobe"))


def test_dry_run_does_not_prepare_or_start_a_gpu(example):
    assert pipeline.main(["--config", str(example.config), "--dry-run"]) == 0
    assert not (example.root / "runs").exists()


@pytest.mark.parametrize("source", ["default", "config", "cli"])
def test_formal_runtime_cannot_enable_strict_mode(example, capsys, source):
    old_policies = {"unresolved_policy": "error", "recovery_policy": "diagnostic",
                    "spatial_failure_policy": "error"}
    argv = ["--config", str(example.config), "--dry-run"]
    if source == "config":
        put(example.config, {**inputs.read_json(example.config), **old_policies})
    elif source == "cli":
        for name, value in old_policies.items():
            argv.extend(["--" + name.replace("_", "-"), value])
    assert pipeline.main(argv) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["recovery_policies"] == {
        "unresolved_policy": "drop_windows", "recovery_policy": "intersection",
        "spatial_failure_policy": "center"}
    if source != "default":
        warnings = [json.loads(line) for line in captured.err.splitlines()]
        assert {item["option"] for item in warnings} == set(old_policies)
        assert all(item["warning"] == "strict_runtime_disabled" for item in warnings)
    assert not (example.root / "runs").exists()


@pytest.mark.parametrize("runner", [pipeline, spatial_runner])
def test_help_does_not_advertise_strict_runtime_switches(runner, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [runner.__file__, "--help"])
    with pytest.raises(SystemExit) as result:
        runner.main()
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--unresolved-policy", "--recovery-policy", "--spatial-failure-policy", "--failure-policy"):
        assert flag not in help_text


def test_pts_mapping_handles_variable_intervals_and_half_open_end():
    pts = [0.0, 0.03, 0.08, 0.2, 0.25, 0.8]
    assert inputs.frame_spans([[0.03, 0.25]], pts) == [[1, 3]]
    assert inputs.frame_spans([[0.031, 0.08]], pts) == []
    assert inputs.window_spans(7, 4, 3) == [(0.0, 4.0), (3.0, 7)]
    with pytest.raises(ValueError):
        inputs.window_spans(7, 4, 5)


def test_run_lock_is_exclusive_and_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="Another process"):
        with inputs.run_lock(tmp_path):
            with inputs.run_lock(tmp_path):
                pass
    with inputs.run_lock(tmp_path):
        pass


def test_interrupted_fallback_receipt_is_recoverable_even_with_old_strict_option(example, monkeypatch):
    args = args_for(example)
    args.unresolved_policy = "error"
    run = example.root / "run"
    merged = run / "03-lenient/merged"
    put(merged / "summary.json", {"unresolved_windows": 2, "unresolved": [
        {"video_id": "81", "window_id": "a"}, {"video_id": "81", "window_id": "b"}]})
    put_rows(merged / "highlights.jsonl", [{"video_id": key, "highlights": []} for key in ("81", "scene_A")])
    original_atomic_json = pipeline.atomic_json

    def interrupted(path, value):
        if path.name == "FALLBACKS.json":
            raise OSError("simulated interruption while writing recovery receipt")
        return original_atomic_json(path, value)

    monkeypatch.setattr(pipeline, "atomic_json", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        pipeline.stage_lenient(args, run, execute=True)
    monkeypatch.setattr(pipeline, "atomic_json", original_atomic_json)
    pipeline.stage_lenient(args, run, execute=True)
    receipt = inputs.read_json(run / "03-lenient/FALLBACKS.json")
    assert receipt["source_video_count"] == 1
    assert len(receipt["fallbacks"]) == 2
    assert inputs.read_json(run / "03-lenient/TERMINAL_STATES.json")["policy"] == "drop_windows"


def test_worker_fingerprint_matches_parent_without_importing_stage1(example, monkeypatch):
    args = args_for(example)
    materialized = inputs.prepare_inputs(args)
    metadata = inputs.prepared_metadata(materialized)
    output = example.root / "spatial run"
    first = {"video_id": "81", "source_sha256": metadata[0]["source_sha256"], "targetRatioWH": [16, 9],
             "frame_count": metadata[0]["frame_count"], "valid": True, "segments_frame": [[0, 2]]}
    stage1 = example.root / "stage1.jsonl"
    put_rows(stage1, [first])
    code = spatial_runner.python_code_fingerprint(example.spatial, ("pipeline", "modules", "vendor"))
    common = SimpleNamespace(fingerprint=inputs.identity, write=put, entrypoint=lambda spec: fake_stage2,
                             validate_stage1=lambda row: None, validate_stage2=lambda *args: None)
    observed = []

    def fake_stage2(ctx, video, row, size):
        observed.append(ctx)
        return {"video_id": row["video_id"], "targetRatioWH": row["targetRatioWH"], "predictions": []}

    monkeypatch.setattr(spatial_runner, "load_module", lambda *args: common)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                                            set_num_threads=lambda count: None))
    plan = example.root / "plan.json"
    value = {"package": str(example.spatial), "code_fingerprint": inputs.identity(code),
             "config": inputs.read_json(example.spatial / "configs/v7e_b_4109.json"), "output": str(output),
             "stage1_file": str(stage1), "resume": True, "run_fingerprint": "fixture",
             "videos": [{"video_id": "81", "path": str(args.videos / "81.mp4"),
                         "sha256": metadata[0]["source_sha256"], "ratio": [16, 9]}]}
    put(plan, value)
    assert spatial_runner.run_worker(plan) == 0
    assert observed[0]["historical_detector_profile"] == {}
    assert isinstance(observed[0]["code_fingerprint"], str)
    assert (output / "per_video/81/DONE.json").is_file()
    value["code_fingerprint"] = "changed"
    put(plan, value)
    with pytest.raises(ValueError, match="source code changed"):
        spatial_runner.run_worker(plan)


def test_relocated_package_runs_cpu_preparation_without_source_checkout(example):
    # These bytes are explicitly dummy weights; this checks packaging, not inference.
    put(example.config, {**inputs.read_json(example.config), "unresolved_policy": "error",
                        "recovery_policy": "diagnostic", "spatial_failure_policy": "error"})
    (example.spatial / "reference").mkdir()
    (example.spatial / "reference/old_test_answers.jsonl").write_text("private", encoding="utf-8")
    release = example.root / "release"
    result = packaging.build_package(example.config, release)
    assert result["state"] == "VERIFIED"
    assert not (release / "models/spatial/reference").exists()
    assert not (release / ".env").exists()
    assert not (release / "models/spatial/run.py").exists()
    release_config = inputs.read_json(release / "configs/submission.json")
    assert not {"unresolved_policy", "recovery_policy", "spatial_failure_policy"} & release_config.keys()
    assert inputs.read_json(release / "models/adapter/adapter_config.json")["base_model_name_or_path"] == "../ora"
    assert (release / "models/ora/chat_template.jinja").read_text(encoding="utf-8") == "{{ messages }}"
    assert (release / "models/ora/chat_templates/default.jinja").is_file()
    moved = example.root / "a completely different location" / "moved package"
    moved.parent.mkdir()
    shutil.move(str(release), str(moved))
    elsewhere = example.root / "work"
    elsewhere.mkdir()
    clean_environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    process = subprocess.run([sys.executable, "-I", "-B", str(moved / "scripts/run_submission_pipeline.py"),
                              "--config", str(moved / "configs/submission.json"),
                              "--input-root", str(example.data), "--prepare-only"],
                             cwd=elsewhere, env=clean_environment, text=True, encoding="utf-8", capture_output=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout)["gpu_started"] is False
    assert packaging.verify_package(moved)["state"] == "VERIFIED"
    (moved / "models/adapter/adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        packaging.verify_package(moved)


def test_packager_refuses_missing_weights(example):
    (example.model / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="weights must be included"):
        packaging.build_package(example.config, example.root / "incomplete")
    assert not (example.root / "incomplete").exists()


def test_synthetic_e2e_and_resume_skip_finished_inference(example, monkeypatch):
    """Real videos + real merges/validator/ZIP; simulated inference/spatial models."""
    original_command = pipeline.run_command
    calls = []

    def command(name, command, output, environment=None, **budgets):
        if name == "infer":
            calls.append(name)
            materialized = Path(command[command.index("--materialized-root") + 1])
            target = output / "run"
            _, windows, _, sources = inference.load_plan(materialized, example.data / "video")
            put(target / "plan.json", {"assistant_answers_passed_to_model": False})
            put(target / "source_index.json", sources)
            for item in windows:
                put(target / "records" / (item["window_id"] + ".json"), {
                    "window_id": item["window_id"], "source_video_id": item["source_video_id"],
                    "window_start_sec": item["window_start_sec"], "window_end_sec": item["window_end_sec"],
                    "raw_json_valid": True, "source_highlights": [] if item["source_video_id"] == "scene_A" else [
                        {"start": item["window_start_sec"], "end": item["window_end_sec"]}]})
            put(target / "summary.json", {"state": "complete"})
            return {}
        if name == "spatial":
            calls.append(name)
            assert command[command.index("--failure-policy") + 1] == "center"
            work = Path(command[command.index("--output") + 1])
            rows = pipeline.read_jsonl(Path(command[command.index("--stage1-file") + 1]))
            predictions = [{"video_id": row["video_id"], "targetRatioWH": row["targetRatioWH"],
                            "predictions": [{"frame": frame, "bboxes": [0, 0, 32]} for a, b in row["segments_frame"]
                                            for frame in range(a, b + 1)]} for row in rows]
            put_rows(work / "predictions_spatial.jsonl", predictions)
            put(work / "COMPLETE.json", {"videos": len(rows), "spatial_sha256": inputs.digest(work / "predictions_spatial.jsonl"),
                "frames": sum(len(row["predictions"]) for row in predictions), "groups": 1, "run_fingerprint": "cpu-simulated"})
            return {}
        return original_command(name=name, command=command, output=output, environment=environment, **budgets)

    monkeypatch.setattr(pipeline, "run_command", command)
    argv = ["--config", str(example.config), "--unresolved-policy", "error",
            "--recovery-policy", "diagnostic", "--spatial-failure-policy", "error"]
    assert pipeline.main(argv) == 0
    current = inputs.read_json(example.root / "runs/CURRENT.json")
    run = Path(current["run_root"])
    report = inputs.read_json(run / "06-submission/FINAL.json")
    assert report["videos"] == 2 and report["videos_with_predictions"] == 1
    assert report["website_submission_performed"] is False
    assert pipeline.main(argv) == 0
    assert calls == ["infer", "spatial"]
    handoff = pipeline.read_jsonl(run / "04-handoff/temporal_handoff.jsonl")
    assert handoff[0]["segments_frame"] == [[0, 23]]
    assert handoff[1]["segments_frame"] == []
