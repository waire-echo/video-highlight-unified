import json

import pytest

from video_highlight.temporal_merge import (
    load_temporal_shards,
    validate_merged_handoff,
)


def _record(video_id: str) -> dict:
    return {
        "schema_version": "temporal-handoff/v1",
        "video_id": video_id,
        "targetRatioWH": [16, 9],
        "fps": 10.0,
        "frame_count": 100,
        "duration_sec": 10.0,
        "segments_sec": [[1.0, 2.0]],
        "segments_frame": [[10, 20]],
        "subject_hint": "person",
        "action_description": "person jumps",
        "source": "window_score_curve",
        "valid": True,
        "fallback_used": None,
        "errors": [],
    }


def _write_shard(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    raw = [{"video_id": record["video_id"], "window_records": []} for record in records]
    (path.parent / (path.name + ".raw.jsonl")).write_text(
        "".join(json.dumps(record) + "\n" for record in raw), encoding="utf-8"
    )


def test_load_temporal_shards_sorts_numeric_ids_and_aligns_raw(tmp_path):
    first = tmp_path / "worker0.jsonl"
    second = tmp_path / "worker1.jsonl"
    _write_shard(first, [_record("10"), _record("1")])
    _write_shard(second, [_record("2"), _record("0")])

    records, raw = load_temporal_shards([first, second])

    assert [record["video_id"] for record in records] == ["0", "1", "2", "10"]
    assert [record["video_id"] for record in raw] == ["0", "1", "2", "10"]


def test_load_temporal_shards_rejects_duplicate_and_mismatched_raw(tmp_path):
    first = tmp_path / "worker0.jsonl"
    second = tmp_path / "worker1.jsonl"
    _write_shard(first, [_record("0")])
    _write_shard(second, [_record("0")])
    with pytest.raises(ValueError, match="duplicate video_id 0"):
        load_temporal_shards([first, second])

    _write_shard(second, [_record("1")])
    (tmp_path / "worker1.jsonl.raw.jsonl").write_text(
        json.dumps({"video_id": "2"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="handoff/raw video IDs differ"):
        load_temporal_shards([first, second])


def test_validate_merged_handoff_requires_all_expected_ids():
    with pytest.raises(ValueError, match="ID mismatch"):
        validate_merged_handoff([_record("0")], ["0", "1"])

    validate_merged_handoff([_record("0"), _record("1")], ["0", "1"])
