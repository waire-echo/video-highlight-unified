"""Exercise the real worker dispatch with stand-in models and a stand-in decoder."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arm, expected", [
    ("B_RGBRank_MissingEvidenceNative_v1", True),
    ("B_RGBRank_DP_v1", False),
    ("B", None),
])
def test_worker_selects_expected_rgb_evidence_mode(tmp_path, monkeypatch, arm, expected):
    runner = load("rgb_dispatch_runner", ROOT / "scripts/run_spatial_submission.py")
    package = ROOT / "models/spatial"
    common = load("rgb_dispatch_common", package / "pipeline/common.py")
    source = tmp_path / "synthetic-source.bin"
    source.write_bytes(b"unit test; decoding is replaced")
    digest = runner.sha256_file(source)
    first = dict(video_id="sample", source_sha256=digest, targetRatioWH=[1, 1],
                 segments_frame=[[0, 1]], frame_count=2, valid=True)
    handoff = tmp_path / "stage1.jsonl"
    handoff.write_text(json.dumps(first) + "\n", encoding="utf-8")
    calls = []

    def baseline(context, video, first, size):
        return dict(video_id="sample", targetRatioWH=[1, 1],
                    predictions=[dict(frame=i, bboxes=[16, 0, 64]) for i in [0, 1]])

    def refine(context, video, first, size, result, assets):
        calls.append(context["rgb_missing_evidence"])
        return result

    class Capture:
        def __init__(self, path): pass
        def isOpened(self): return True
        def release(self): pass
        def get(self, key): return {1: 96, 2: 64, 3: 2}[key]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True), set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FRAME_COUNT=3,
        VideoCapture=Capture, setNumThreads=lambda n: None))
    monkeypatch.setitem(sys.modules, "video_highlight.stage2_rgb_rank", SimpleNamespace(refine=refine))
    monkeypatch.setattr(runner, "load_module", lambda *args: common)
    monkeypatch.setattr(common, "entrypoint", lambda spec: baseline)
    output = tmp_path / "output"
    code = runner.python_code_fingerprint(package, ("pipeline", "modules", "vendor"))
    plan = dict(package=str(package), output=str(output), code_fingerprint=common.fingerprint(code),
                config=common.read(package / "configs/v7e_b_4109.json"), arm=arm,
                stage1_file=str(handoff), resume=False, run_fingerprint="current-attempt",
                videos=[dict(video_id="sample", path=str(source), sha256=digest, ratio=[1, 1])],
                rgb_rank_assets=str(tmp_path / "assets"), rgb_code={
                    name: runner.sha256_file(ROOT / "src/video_highlight" / name)
                    for name in ["stage2_rgb_assets.py", "stage2_rgb_rank.py", "rgb_rank_core.py", "rgb_vision_compat.py"]})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert runner.run_worker(plan_path) == 0
    assert calls == ([] if expected is None else [expected])
    assert common.read(output / "per_video/sample/DONE.json")["state"] == "ok"
