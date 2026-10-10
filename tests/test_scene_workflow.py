"""Version boundaries, interruption recovery and exact original-frame I/O contracts."""
import importlib.util
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from video_highlight.workflow import runner
from video_highlight.workflow.contracts import fingerprint, read_json, run_lock, sha256, write_json
from video_highlight.workflow.handoff import build_spatial_input, common, validate_result
from video_highlight.workflow.io_optimized import CheckpointExporter, SequentialFrameReader
from video_highlight.workflow.registry import REPO, release, resolve

SCENE = REPO / "pipelines/ora_spatial"


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def handoff_fixture(tmp_path):
    timestamps = np.array([0, .05, .11, .20, .24, .35])
    np.savez(tmp_path / "timestamps.npz", v=timestamps, empty=timestamps)
    meta = {"frame_count": 6, "width": 120, "height": 90, "source_sha256": "source"}
    write_json(tmp_path / "manifest.json", {"videos": {"v": meta, "empty": meta}})
    handoff = {"schema_version": "temporal-handoff/v1", "video_id": "v", "targetRatioWH": [16, 9],
               "valid": True, "frame_count": 6, "duration_sec": .4, "timestamp_mapping": "original_frame_pts",
               "frame_index_convention": "inclusive", "segments_frame": [[1, 2], [5, 5]], "segments_sec": [[.05, .20], [.35, .4]]}
    empty = {**handoff, "video_id": "empty", "segments_frame": [], "segments_sec": []}
    masks = [{"video_id": vid, "targetRatioWH": [16, 9], "frame_count": 6, "frame_index_base": 0,
              "highlight_frames": frames} for vid, frames in [("v", [1, 2, 5]), ("empty", [])]]
    jsonl(tmp_path / "handoff.jsonl", [handoff, empty])
    jsonl(tmp_path / "frames.jsonl", masks)
    return handoff, empty, masks


def build_fixture(tmp_path):
    return build_spatial_input(tmp_path / "handoff.jsonl", tmp_path / "frames.jsonl", tmp_path / "manifest.json",
                               tmp_path / "timestamps.npz", tmp_path / "adapted", ["v", "empty"], SCENE)


def test_vfr_handoff_keeps_exact_frames_and_empty_video(tmp_path):
    handoff_fixture(tmp_path)
    report = build_fixture(tmp_path)
    rows = common(SCENE).load_rows(tmp_path / "adapted/baseline.zip")
    assert [p["frame"] for p in rows[0]["predictions"]] == [1, 2, 5]
    assert rows[1]["predictions"] == []
    assert report["frames"] == 3 and report["temporal_mask_unchanged"]
    validation = validate_result(SCENE, tmp_path / "adapted/baseline.zip", tmp_path / "adapted/manifest.json",
                                 tmp_path / "adapted/baseline.jsonl", ["v", "empty"])
    assert validation["invalid_count"] == 0


@pytest.mark.parametrize("mutation", ["mask", "fps", "pts", "duplicate", "failed", "ratio", "bounds", "bool-frame"])
def test_handoff_rejects_hidden_frame_or_pts_changes(tmp_path, mutation):
    handoff, empty, masks = handoff_fixture(tmp_path)
    if mutation == "mask":
        masks[0]["highlight_frames"] = [1, 2]
    elif mutation == "fps":
        handoff["timestamp_mapping"] = "constant_fps"
    elif mutation == "pts":
        handoff["segments_sec"][0][0] = .04
    elif mutation == "duplicate":
        masks.append(masks[0])
    elif mutation == "failed":
        handoff["valid"] = False
    elif mutation == "ratio":
        masks[0]["targetRatioWH"] = [9, 16]
    elif mutation == "bounds":
        masks[0]["highlight_frames"] = [1, 2, 6]
    else:
        masks[0]["highlight_frames"] = [True, 2, 5]
    jsonl(tmp_path / "handoff.jsonl", [handoff, empty])
    jsonl(tmp_path / "frames.jsonl", masks)
    with pytest.raises(ValueError):
        build_fixture(tmp_path)


def make_video(path, count=60):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 80))
    assert writer.isOpened()
    random = np.random.default_rng(5)
    for _ in range(count):
        writer.write(random.integers(0, 256, (80, 128, 3), dtype=np.uint8))
    writer.release()


def original_reader():
    spec = importlib.util.spec_from_file_location("scene_ora_tools_test", SCENE / "src/ora_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_frames


def test_sequential_decoder_matches_original_pixels_resize_and_backwards_reads(tmp_path):
    video = tmp_path / "video.mp4"
    make_video(video)
    reader = SequentialFrameReader(max_bytes=4 << 20)
    original = original_reader()
    try:
        for frames in [[0, 3, 10], [12, 20, 29], [31, 48, 59], [3, 12], [1]]:
            expected, actual = original(video, frames, 70), reader(video, frames, 70)
            assert actual.keys() == expected.keys()
            assert all(np.array_equal(actual[f], expected[f]) for f in frames)
        assert reader.metrics["decoded_frames"] == 62  # 60 forward + restart through original frame 1
        assert reader.metrics["decoder_opens"] == 2
        # Callers can annotate returned images without corrupting the cache.
        old = reader(video, [1], 70)[1]
        reader(video, [1], 70)[1][:] = 0
        assert np.array_equal(reader(video, [1], 70)[1], old)
    finally:
        reader.close()


def test_decoder_cache_is_bounded_and_source_changes_are_rejected(tmp_path):
    video = tmp_path / "video.mp4"
    make_video(video, 10)
    reader = SequentialFrameReader(max_bytes=128 * 80 * 3 * 2)
    try:
        reader(video, [0, 1, 2, 3])
        assert reader.metrics["cache_peak_bytes"] <= reader.max_bytes
        with video.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="changed"):
            reader(video, [4])
    finally:
        reader.close()


def test_checkpoint_policy_keeps_final_jsonl_and_zip_bytes(tmp_path):
    api = common(SCENE)
    rows = [{"video_id": str(i), "targetRatioWH": [16, 9],
             "predictions": [{"frame": 0, "bboxes": [0, 0, 100]}]} for i in range(7)]
    metadata = {"videos": {str(i): {"width": 120, "height": 90, "frame_count": 1} for i in range(7)}}
    optimized = CheckpointExporter(api, interval=3)
    for n in range(1, 8):
        optimized(tmp_path / "new/partial", rows[:n], rows[:n], metadata, False)
    optimized(tmp_path / "new/final", rows, rows, metadata, False)
    api.export(tmp_path / "old/final", rows, rows, metadata, False)
    for suffix in [".jsonl", ".zip", ".validation.json"]:
        assert (tmp_path / ("new/final" + suffix)).read_bytes() == (tmp_path / ("old/final" + suffix)).read_bytes()
    assert len(list((tmp_path / "new/checkpoints").glob("*.json"))) == 7
    assert optimized.metrics["partial_full_exports"] == 2


def test_checkpoint_checks_new_video_before_skip(tmp_path):
    api = common(SCENE)
    old = [{"video_id": "v", "targetRatioWH": [16, 9], "predictions": [{"frame": 0, "bboxes": [0, 0, 100]}]}]
    bad = [{**old[0], "predictions": [{"frame": 1, "bboxes": [0, 0, 100]}]}]
    metadata = {"videos": {"v": {"width": 120, "height": 90, "frame_count": 2}}}
    with pytest.raises(ValueError, match="mask"):
        CheckpointExporter(api)(tmp_path / "partial", bad, old, metadata, False)


def test_release_guards_baseline_sources():
    value = release("scene-general-v3-compat-20261010")
    assert value["mode"] == "compat" and value["input_modes"] == ["baseline"]
    assert value["source_files"]["experiments/scene_general_v3/temporal.py"] == sha256(SCENE / "experiments/scene_general_v3/temporal.py")
    with pytest.raises(ValueError):
        release("../scene-general-v3-compat-20261010")


def config_fixture(tmp_path, version="scene-general-v3-compat-20261010", mode="baseline"):
    data = tmp_path / "data"
    (data / "inputs").mkdir(parents=True)
    write_json(data / "inputs/manifest.json", {"videos": {"0": {}, "1": {}}})
    config = {"schema_version": "scene-workflow/v1", "version": version, "input_mode": mode, "ids": "all",
              "paths": {"data_root": str(data), "video_dir": str(tmp_path / "videos"), "runtime_root": str(tmp_path / "runtime"),
                        "run_root": str(tmp_path / "run")}}
    path = tmp_path / "config.json"
    write_json(path, config)
    return path, config


def test_interpreter_symlink_and_planning_have_no_run_side_effects(tmp_path):
    path, config = config_fixture(tmp_path)
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to("/usr/bin/python3")
    config["paths"]["client_python"] = str(executable)
    write_json(path, config)
    resolved, descriptor = resolve(path)
    assert resolved["paths"]["client_python"] == str(executable)
    plan = runner.make_plan(resolved, descriptor)
    assert plan["resources"]["max_num_seqs"] == 1
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mutation", ["order", "duplicate", "unknown", "unsafe-root", "compat-handoff", "unknown-key"])
def test_configuration_rejects_ambiguous_or_unsafe_version_use(tmp_path, mutation):
    path, config = config_fixture(tmp_path)
    if mutation == "order":
        config["ids"] = ["1", "0"]
    elif mutation == "duplicate":
        config["ids"] = ["0", "0"]
    elif mutation == "unknown":
        config["version"] = "missing-version"
    elif mutation == "unsafe-root":
        config["paths"]["run_root"] = config["paths"]["data_root"]
    elif mutation == "compat-handoff":
        config["input_mode"] = "handoff"
    else:
        config["fps"] = 10
    write_json(path, config)
    with pytest.raises(ValueError):
        resolve(path)


def test_duplicate_runner_is_blocked(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(ValueError, match="owns"):
            with run_lock(tmp_path):
                pass


def fake_workflow(tmp_path, monkeypatch):
    path, config = config_fixture(tmp_path, version="scene-general-v3-unified-v1")
    resolved, descriptor = resolve(path)
    plan = runner.make_plan(resolved, descriptor)
    plan["stages"] = [{"stage": "first"}, {"stage": "second"}]
    monkeypatch.setattr(runner, "frozen_inputs", lambda plan: {"plan_sha256": fingerprint(plan), "files": {}, "weights": {}})
    monkeypatch.setattr(runner, "prepare_source", lambda *args: None)
    return plan


def test_failed_stage_resume_reuses_completed_stage_and_records_error(tmp_path, monkeypatch):
    plan = fake_workflow(tmp_path, monkeypatch)
    calls = []
    def action(stage, plan, descriptor, state, env):
        calls.append(stage["stage"])
        if calls == ["first", "second"]:
            raise RuntimeError("model unavailable")
        path = state.root / (stage["stage"] + ".json")
        write_json(path, {"ok": True})
        return [path]
    monkeypatch.setattr(runner, "stage_action", action)
    with pytest.raises(RuntimeError):
        runner.execute(plan)
    root = tmp_path / "run"
    assert read_json(root / "status.json")["status"] == "failed"
    assert json.loads((root / "errors/failed_items.jsonl").read_text())["stage"] == "second"
    assert runner.execute(plan, resume=True)["status"] == "complete"
    assert calls == ["first", "second", "second"]
    events = [json.loads(line)["event"] for line in (root / "events.jsonl").read_text().splitlines()]
    assert "stage_reused" in events


def test_resume_rejects_changed_outputs_without_overwriting_evidence(tmp_path, monkeypatch):
    plan = fake_workflow(tmp_path, monkeypatch)
    def action(stage, plan, descriptor, state, env):
        path = state.root / (stage["stage"] + ".json")
        write_json(path, {"ok": True})
        return [path]
    monkeypatch.setattr(runner, "stage_action", action)
    runner.execute(plan)
    path = tmp_path / "run/first.json"
    path.write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        runner.execute(plan, resume=True)
    assert path.read_text() == "changed"


def test_resume_rejects_changed_version_before_any_stage(tmp_path, monkeypatch):
    plan = fake_workflow(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "stage_action", lambda *args: [])
    runner.execute(plan)
    status = (tmp_path / "run/status.json").read_bytes()
    plan["version"] = "scene-general-v3-io-v1"
    with pytest.raises(ValueError, match="changed"):
        runner.execute(plan, resume=True)
    assert (tmp_path / "run/status.json").read_bytes() == status


def test_process_cleanup_does_not_touch_already_exited_process():
    process = subprocess.Popen(["/bin/true"], start_new_session=True)
    process.wait()
    runner.stop_process(process)
    assert process.returncode == 0


@pytest.mark.parametrize("invalid_handoff", [False, True])
def test_full_stage_machine_uses_exact_adapter_and_blocks_failed_handoff(tmp_path, monkeypatch, invalid_handoff):
    original, empty, masks = handoff_fixture(tmp_path)
    if invalid_handoff:
        original["valid"] = False
        jsonl(tmp_path / "handoff.jsonl", [original, empty])
    data = tmp_path / "data"
    (data / "inputs").mkdir(parents=True)
    (data / "output/shots").mkdir(parents=True)
    import shutil
    shutil.copyfile(tmp_path / "manifest.json", data / "inputs/manifest.json")
    shutil.copyfile(tmp_path / "timestamps.npz", data / "inputs/timestamps.npz")
    write_json(data / "output/parameter_ledger.json", {"pass": True, "total_upper_bound": 0})
    write_json(tmp_path / "index.json", [{"video_id": row["video_id"], "targetRatioWH": row["targetRatioWH"]} for row in [original, empty]])
    config = {"schema_version": "scene-workflow/v1", "version": "scene-general-v3-unified-v1", "input_mode": "temporal", "ids": "all",
              "paths": {"data_root": str(data), "video_dir": str(tmp_path / "videos"), "runtime_root": str(tmp_path / "runtime"),
                        "run_root": str(tmp_path / "run"), "test_index": str(tmp_path / "index.json")}}
    path = tmp_path / "config.json"
    write_json(path, config)
    resolved, descriptor = resolve(path)
    plan = runner.make_plan(resolved, descriptor)
    monkeypatch.setattr(runner, "frozen_inputs", lambda plan: {"plan_sha256": fingerprint(plan), "files": {}, "weights": {}})
    calls = []
    @contextmanager
    def model(*args):
        yield
    monkeypatch.setattr(runner, "managed_model", model)
    def command(argv, state, env):
        calls.append(state.state["stage"])
        if state.state["stage"] == "temporal":
            handoff, frames = runner.temporal_files(state.root)
            handoff.parent.mkdir(parents=True)
            shutil.copyfile(tmp_path / "handoff.jsonl", handoff)
            shutil.copyfile(tmp_path / "frames.jsonl", frames)
            (state.root / "temporal/temporal.complete").write_text("complete")
        elif "video_highlight.workflow.scene_entry" in argv:
            out = state.root / "spatial"
            out.mkdir()
            for suffix in [".jsonl", ".zip", ".validation.json"]:
                shutil.copyfile(state.root / ("handoff/baseline" + suffix), out / ("SPATIAL_temporal_v3" + suffix))
            write_json(out / "report.json", {})
            write_json(out / "frozen.json", {})
            shutil.copyfile(out / "SPATIAL_temporal_v3.zip", out / "results.zip")
            write_json(out / "status.json", {"status": "complete", "requested_ids": ["v", "empty"], "completed_ids": ["v", "empty"]})
    monkeypatch.setattr(runner, "run_command", command)
    if invalid_handoff:
        with pytest.raises(ValueError, match="invalid"):
            runner.execute(plan)
        assert calls == ["temporal"]
        assert not (tmp_path / "run/final/submission.zip").exists()
    else:
        result = runner.execute(plan)
        assert result["status"] == "complete" and result["completed_ids"] == ["v", "empty"]
        artifacts = read_json(tmp_path / "run/final/artifacts.json")
        assert artifacts["submission"] == "submission.zip"
        assert artifacts["artifacts"]["submission.zip"]["sha256"] == sha256(tmp_path / "run/handoff/baseline.zip")
        before = calls.copy()
        runner.execute(plan, resume=True)
        assert calls == before
