import ast
import importlib.util
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_highlight.temporal import TemporalResult


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_temporal_frontend.py"
SPEC = importlib.util.spec_from_file_location("run_temporal_frontend", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def _write_inputs(tmp_path, ids=("1", "2")):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    index = []
    for video_id in ids:
        (video_dir / f"{video_id}.mp4").write_bytes(b"stub")
        index.append({"video_id": video_id, "targetRatioWH": [16, 9]})
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path, video_dir


def _argv(index_path, video_dir, output, *extra):
    return [
        "--index",
        str(index_path),
        "--video-dir",
        str(video_dir),
        "--out",
        str(output),
        "--temporal-mode",
        "motion",
        *extra,
    ]


class FakeDetector:
    def __init__(self, fail_ids=()):
        self.fail_ids = {str(item) for item in fail_ids}
        self.detected = []
        self.closed = False

    def detect(self, video_path):
        video_id = Path(video_path).stem
        self.detected.append(video_id)
        if video_id in self.fail_ids:
            raise RuntimeError(f"synthetic failure {video_id}")
        return TemporalResult(
            [(1.0, 2.0)],
            subject_hint="person",
            source="window_score_curve",
            valid=True,
            errors=["first response invalid; retry recovered"],
            duration_sec=10.0,
            action_description="person jumps",
            window_records=[
                {
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                    "valid": True,
                    "action": 3,
                    "action_description": "person jumps",
                    "fused_score": 0.9,
                    "attempts": 1,
                }
            ],
        )

    def close(self):
        self.closed = True


def test_runner_source_never_imports_spatial_or_yolo_module():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("spatial" in name.lower() or "yolo" in name.lower() for name in imports)


def test_video_ids_are_one_comma_separated_value():
    assert runner.parse_video_ids("76,100,107,121,154") == [
        "76",
        "100",
        "107",
        "121",
        "154",
    ]
    assert runner.parse_video_ids("76,76,100") == ["76", "100"]
    with pytest.raises(ValueError, match="empty IDs"):
        runner.parse_video_ids("76,,100")


def test_temporal_parser_rejects_structured_reze_otsu_before_runtime(tmp_path):
    index_path, video_dir = _write_inputs(tmp_path, ids=("1",))
    output = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit):
        runner.parse_args(
            _argv(
                index_path,
                video_dir,
                output,
                "--temporal-mode",
                "window_score",
                "--window-semantic-mode",
                "structured",
                "--window-aggregation",
                "reze_otsu",
            )
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows E-drive policy")
def test_qwen_runtime_path_guard_rejects_c_drive():
    assert runner._require_e_absolute(r"E:\AI\tmp\out.jsonl", "output").startswith(
        "E:\\"
    )
    with pytest.raises(ValueError, match="absolute E"):
        runner._require_e_absolute(r"C:\Users\example\out.jsonl", "output")


@pytest.mark.skipif(os.name != "nt", reason="Windows E-drive policy")
def test_qwen_preflight_redirects_all_runtime_caches_to_e(tmp_path, monkeypatch):
    if runner.ntpath.splitdrive(str(tmp_path))[0].upper() != "E:":
        pytest.skip("run pytest with an E-drive --basetemp to exercise the no-C policy")
    model_path = tmp_path / "model"
    model_path.mkdir()
    output = tmp_path / "result" / "temporal_handoff.jsonl"
    args = SimpleNamespace(
        temporal_mode="window_score",
        temporal_model_path=str(model_path),
        window_cache_root=str(tmp_path / "cache"),
        window_log_root=str(tmp_path / "logs"),
        out=str(output),
        zip_out="",
    )
    old_pycache_prefix = runner.sys.pycache_prefix
    old_tempdir = runner.tempfile.tempdir
    monkeypatch.setattr(runner.sys, "pycache_prefix", old_pycache_prefix)
    monkeypatch.setattr(runner.tempfile, "tempdir", old_tempdir)

    runner._prepare_runtime(args)

    for key in (
        "TEMP",
        "TMP",
        "TMPDIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "PYTHONPYCACHEPREFIX",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
    ):
        assert runner.ntpath.splitdrive(os.environ[key])[0].upper() == "E:"
    assert runner.ntpath.splitdrive(runner.sys.pycache_prefix)[0].upper() == "E:"
    assert runner.ntpath.splitdrive(runner.tempfile.tempdir)[0].upper() == "E:"
    assert runner.ntpath.splitdrive(args.out)[0].upper() == "E:"
    assert runner.ntpath.splitdrive(args.zip_out)[0].upper() == "E:"


def test_temporal_only_checkpoint_resume_retry_and_legacy_zip(tmp_path, monkeypatch):
    index_path, video_dir = _write_inputs(tmp_path)
    output = tmp_path / "temporal_handoff.jsonl"
    monkeypatch.setattr(runner, "video_meta", lambda _path: (100, 10.0, 640, 360))

    built = []

    def first_builder(_args):
        detector = FakeDetector(fail_ids={"1"})
        built.append(detector)
        return detector

    monkeypatch.setattr(runner, "build_temporal_detector", first_builder)
    assert runner.main(_argv(index_path, video_dir, output, "--video-ids", "1,2")) == 0
    assert len(built) == 1
    assert built[0].detected == ["1", "2"]
    assert built[0].closed

    handoff = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["video_id"] for record in handoff] == ["1", "2"]
    assert handoff[0]["valid"] is False
    assert handoff[1]["segments_sec"] == [[1.0, 2.0]]
    assert handoff[1]["segments_frame"] == [[10, 20]]
    assert handoff[1]["action_description"] == "person jumps"
    assert handoff[1]["errors"] == []
    assert handoff[1]["warnings"] == ["first response invalid; retry recovered"]
    assert "predictions" not in handoff[1]
    assert "spatial" not in handoff[1]

    raw_path = Path(str(output) + ".raw.jsonl")
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert raw[1]["window_records"][0]["action"] == 3
    assert raw[1]["window_records"][0]["action_description"] == "person jumps"
    assert raw[1]["errors"] == ["first response invalid; retry recovered"]
    assert Path(str(output) + ".summary.json").exists()
    assert Path(str(output) + ".state.json").exists()
    zip_path = tmp_path / "temporal_hints.zip"
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["1.json", "2.json"]
        assert json.loads(archive.read("1.json"))["segments"] == []
        assert json.loads(archive.read("2.json"))["segments"] == [[1.0, 2.0]]

    retry_detectors = []

    def retry_builder(_args):
        detector = FakeDetector()
        retry_detectors.append(detector)
        return detector

    monkeypatch.setattr(runner, "build_temporal_detector", retry_builder)
    assert runner.main(
        _argv(index_path, video_dir, output, "--video-ids", "1,2", "--retry-errors")
    ) == 0
    assert len(retry_detectors) == 1
    assert retry_detectors[0].detected == ["1"]
    assert retry_detectors[0].closed
    state = json.loads(Path(str(output) + ".state.json").read_text(encoding="utf-8"))
    assert state["error_video_ids"] == []
    assert state["valid_video_ids"] == ["1", "2"]

    def should_not_build(_args):
        raise AssertionError("detector must not be built when no videos are pending")

    monkeypatch.setattr(runner, "build_temporal_detector", should_not_build)
    assert runner.main(_argv(index_path, video_dir, output, "--video-ids", "1,2")) == 0


def test_fingerprint_conflict_refuses_mixed_parameters_and_overwrite_allows_new_run(
    tmp_path, monkeypatch
):
    index_path, video_dir = _write_inputs(tmp_path, ids=("1",))
    output = tmp_path / "temporal_handoff.jsonl"
    monkeypatch.setattr(runner, "video_meta", lambda _path: (100, 10.0, 640, 360))
    detectors = []

    def builder(_args):
        detector = FakeDetector()
        detectors.append(detector)
        return detector

    monkeypatch.setattr(runner, "build_temporal_detector", builder)
    runner.main(_argv(index_path, video_dir, output))
    with pytest.raises(RuntimeError, match="fingerprint conflicts"):
        runner.main(_argv(index_path, video_dir, output, "--window-sec", "5"))
    assert len(detectors) == 1

    runner.main(
        _argv(index_path, video_dir, output, "--window-sec", "5", "--overwrite")
    )
    assert len(detectors) == 2
    assert all(detector.closed for detector in detectors)


def test_missing_video_is_isolated_without_building_fake_spatial_output(tmp_path, monkeypatch):
    index_path, video_dir = _write_inputs(tmp_path, ids=("1", "2"))
    (video_dir / "1.mp4").unlink()
    output = tmp_path / "temporal_handoff.jsonl"
    monkeypatch.setattr(runner, "video_meta", lambda _path: (100, 10.0, 640, 360))
    detector = FakeDetector()
    monkeypatch.setattr(runner, "build_temporal_detector", lambda _args: detector)

    runner.main(_argv(index_path, video_dir, output))

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0]["source"] == "input_error"
    assert records[0]["valid"] is False
    assert records[0]["segments_frame"] == []
    assert records[1]["valid"] is True
    assert detector.detected == ["2"]
    assert detector.closed


def test_resume_recovers_uncommitted_row_between_atomic_files(tmp_path, monkeypatch):
    index_path, video_dir = _write_inputs(tmp_path, ids=("1", "2", "3"))
    output = tmp_path / "temporal_handoff.jsonl"
    monkeypatch.setattr(runner, "video_meta", lambda _path: (100, 10.0, 640, 360))
    first = FakeDetector()
    monkeypatch.setattr(runner, "build_temporal_detector", lambda _args: first)
    runner.main(_argv(index_path, video_dir, output, "--video-ids", "1,2"))

    # Simulate interruption after handoff replacement but before raw/state.
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    uncommitted = dict(records[-1])
    uncommitted["video_id"] = "3"
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(uncommitted) + "\n")

    resumed = FakeDetector()
    monkeypatch.setattr(runner, "build_temporal_detector", lambda _args: resumed)
    runner.main(_argv(index_path, video_dir, output, "--video-ids", "3"))

    assert resumed.detected == ["3"]
    final_records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["video_id"] for record in final_records] == ["1", "2", "3"]
