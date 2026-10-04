import importlib.util
import json
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_temporal_hints.py"
SPEC = importlib.util.spec_from_file_location("package_temporal_hints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_infer_main_subject_uses_yolo_friendly_groups():
    assert MODULE.infer_main_subject("white dog running toward camera") == "animal"
    assert MODULE.infer_main_subject("person in red holding a ball") == "sports ball"
    assert MODULE.infer_main_subject("woman turning toward camera") == "person"
    assert MODULE.infer_main_subject("blue car crossing finish") == "vehicle"
    assert MODULE.infer_main_subject("moving text") == "other"


def test_package_records_matches_legacy_zip_shape_and_marks_missing(tmp_path):
    item_dir = tmp_path / "items"
    item_dir.mkdir()
    raw = {
        "video_id": "0",
        "valid": True,
        "source": "vlm",
        "fallback_used": None,
        "temporal_segments_sec": [[1.0, 2.5]],
        "subject_hint": "white dog jumping",
    }
    (item_dir / "0.jsonl.raw.jsonl").write_text(json.dumps(raw), encoding="utf-8")
    output_zip = tmp_path / "hints.zip"

    summary = MODULE.package_records(item_dir, output_zip, 0, 1)

    assert summary["entries"] == 2
    assert summary["valid_vlm_entries"] == 1
    assert summary["placeholder_video_ids"] == ["1"]
    with zipfile.ZipFile(output_zip) as archive:
        assert archive.namelist() == ["0.json", "1.json"]
        first = json.loads(archive.read("0.json"))
        missing = json.loads(archive.read("1.json"))
    assert first == {
        "segments": [[1.0, 2.5]],
        "subject_hint": "white dog jumping",
        "main_subject": "animal",
        "confidence": 0.0,
    }
    assert missing == {
        "segments": [],
        "subject_hint": "",
        "main_subject": "other",
        "confidence": 0.0,
    }


def test_package_records_accepts_valid_vlm_retry_without_fallback(tmp_path):
    item_dir = tmp_path / "items"
    item_dir.mkdir()
    raw = {
        "video_id": "71",
        "valid": True,
        "source": "vlm_retry",
        "fallback_used": None,
        "temporal_segments_sec": [[0.0, 10.667]],
        "subject_hint": "person walking",
    }
    (item_dir / "71.jsonl.raw.jsonl").write_text(json.dumps(raw), encoding="utf-8")

    summary = MODULE.package_records(item_dir, tmp_path / "retry.zip", 71, 71)

    assert summary["valid_vlm_entries"] == 1
    assert summary["placeholder_video_ids"] == []


def test_package_handoff_records_writes_window_score_and_error_placeholder(tmp_path):
    records = [
        {
            "schema_version": "temporal-handoff/v1",
            "video_id": "76",
            "valid": True,
            "segments_sec": [[8.75, 12.75]],
            "subject_hint": "woman riding a bicycle",
        },
        {
            "schema_version": "temporal-handoff/v1",
            "video_id": "100",
            "valid": False,
            "segments_sec": [],
            "subject_hint": None,
        },
    ]
    output_zip = tmp_path / "temporal_hints.zip"

    summary = MODULE.package_handoff_records(records, output_zip)

    assert summary["entries"] == 2
    assert summary["valid_entries"] == 1
    assert summary["placeholder_video_ids"] == ["100"]
    assert not (tmp_path / "temporal_hints.zip.partial").exists()
    with zipfile.ZipFile(output_zip) as archive:
        assert archive.namelist() == ["76.json", "100.json"]
        valid = json.loads(archive.read("76.json"))
        invalid = json.loads(archive.read("100.json"))
    assert valid["segments"] == [[8.75, 12.75]]
    assert valid["main_subject"] == "person"
    assert invalid == {
        "segments": [],
        "subject_hint": "",
        "main_subject": "other",
        "confidence": 0.0,
    }
