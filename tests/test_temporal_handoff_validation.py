import json

from video_highlight.temporal_handoff_validation import (
    validate_temporal_handoff_file,
    validate_temporal_handoff_records,
)


def _valid_record(video_id="76"):
    return {
        "schema_version": "temporal-handoff/v1",
        "video_id": video_id,
        "targetRatioWH": [16, 9],
        "fps": 25.0,
        "frame_count": 390,
        "duration_sec": 15.6,
        "segments_sec": [[8.76, 12.76]],
        "segments_frame": [[219, 319]],
        "subject_hint": "person",
        "action_description": "person completes a jump",
        "source": "window_score_curve",
        "valid": True,
        "fallback_used": None,
        "errors": [],
    }


def test_accepts_valid_closed_frame_handoff():
    ok, errors = validate_temporal_handoff_records(
        [_valid_record()], {"76": [16, 9]}
    )
    assert ok
    assert errors == []


def test_rejects_submission_and_spatial_fields_even_when_nested():
    record = _valid_record()
    record["diagnostics"] = {
        "predictions": [{"frame": 219, "bboxes": [0, 0, 100]}],
        "spatial": [],
    }
    ok, errors = validate_temporal_handoff_records([record])
    assert not ok
    assert any("forbidden" in error for error in errors)
    assert any("predictions" in error and "bboxes" in error for error in errors)


def test_rejects_duplicate_ids_bad_bounds_and_second_frame_mismatch():
    first = _valid_record()
    first["segments_frame"] = [[220, 319]]
    second = _valid_record()
    second["segments_sec"] = [[15.0, 16.0]]
    second["segments_frame"] = [[375, 389]]
    ok, errors = validate_temporal_handoff_records([first, second])
    assert not ok
    assert any("does not match rounded seconds" in error for error in errors)
    assert any("outside [0, duration]" in error for error in errors)
    assert any("duplicate video_id" in error for error in errors)


def test_invalid_error_record_may_have_unknown_video_metadata():
    record = _valid_record("missing")
    record.update(
        {
            "fps": None,
            "frame_count": None,
            "duration_sec": None,
            "segments_sec": [],
            "segments_frame": [],
            "subject_hint": None,
            "action_description": None,
            "source": "input_error",
            "valid": False,
            "errors": ["missing video"],
        }
    )
    ok, errors = validate_temporal_handoff_records([record])
    assert ok
    assert errors == []


def test_expected_index_checks_missing_unexpected_and_target_ratio():
    record = _valid_record("100")
    record["targetRatioWH"] = [9, 16]
    ok, errors = validate_temporal_handoff_records(
        [record], {"76": [16, 9], "100": [16, 9]}
    )
    assert not ok
    assert any("targetRatioWH mismatch" in error for error in errors)
    assert any("missing expected video_id(s): 76" in error for error in errors)


def test_file_loader_reports_bad_json_without_crashing(tmp_path):
    path = tmp_path / "handoff.jsonl"
    path.write_text(
        json.dumps(_valid_record(), ensure_ascii=False) + "\n{broken\n",
        encoding="utf-8",
    )
    ok, errors = validate_temporal_handoff_file(path)
    assert not ok
    assert any("invalid JSON" in error for error in errors)
