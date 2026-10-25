import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_spatial_submission.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_spatial_submission", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_empty_spatial_outputs_are_contract_complete():
    stage1 = {
        "video_id": "7",
        "source_sha256": "a" * 64,
        "targetRatioWH": [9, 16],
        "segments_frame": [],
        "frame_count": 300,
        "valid": False,
    }
    first, second, done = MODULE.build_empty_spatial_outputs(
        stage1, [720, 1280], "fingerprint"
    )
    assert first == stage1
    assert second == {
        "video_id": "7",
        "targetRatioWH": [9, 16],
        "predictions": [],
    }
    assert done["source_size"] == [720, 1280]
    assert done["run_fingerprint"] == "fingerprint"
    assert set(done["file_sha256"]) == {"stage1.json", "stage2.json"}


def test_empty_spatial_outputs_reject_nonempty_stage1():
    stage1 = {
        "video_id": "7",
        "source_sha256": "a" * 64,
        "targetRatioWH": [9, 16],
        "segments_frame": [[1, 3]],
        "frame_count": 300,
        "valid": True,
    }
    try:
        MODULE.build_empty_spatial_outputs(stage1, [720, 1280], "fingerprint")
    except ValueError as error:
        assert "valid=false" in str(error)
    else:
        raise AssertionError("nonempty Stage 1 row must not use the empty helper")
