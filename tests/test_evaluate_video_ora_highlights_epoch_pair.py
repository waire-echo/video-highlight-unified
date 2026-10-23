from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_video_ora_highlights_epoch_pair as module  # noqa: E402


def test_metrics_use_union_and_overlap_once():
    actual = [{"start": 0.0, "end": 2.0}, {"start": 1.0, "end": 3.0}]
    reference = [{"start": 2.0, "end": 4.0}]
    metrics = module.temporal_metrics(actual, reference)
    assert metrics["temporal_iou"] == pytest.approx(1 / 4)
    assert metrics["recall"] == pytest.approx(1 / 2)


def test_exact_empty_think_prefix_is_recorded_and_removed():
    content = '<think>\n</think>\n{"highlights":[{"start":1,"end":2}]}'
    normalized, removed = module.strip_exact_empty_think(content)
    assert removed is True
    assert normalized.startswith("{")
    parsed = module.validate_generation(content, "stop", Decimal("10"))
    assert parsed["stripped_empty_think"] is True
    assert parsed["highlights"] == [{"start": 1.0, "end": 2.0}]


def test_nonempty_think_is_not_silently_accepted():
    content = '<think>reasoning</think>{"highlights":[]}'
    with pytest.raises(json.JSONDecodeError):
        module.validate_generation(content, "stop", Decimal("10"))


def test_highlight_contract_rejects_event_fields_and_out_of_bounds():
    with pytest.raises(ValueError, match="only highlights"):
        module.validate_highlights('{"events":[]}', Decimal("10"))
    with pytest.raises(ValueError, match="only start/end"):
        module.validate_highlights(
            '{"highlights":[{"start":1,"end":2,"decision":"retain"}]}',
            Decimal("10"),
        )
    with pytest.raises(ValueError, match="outside"):
        module.validate_highlights(
            '{"highlights":[{"start":9,"end":11}]}',
            Decimal("10"),
        )


def test_highlight_contract_merges_before_counting():
    content = json.dumps({
        "highlights": [
            {"start": float(index), "end": float(index + 1)}
            for index in range(32)
        ]
    })
    assert module.validate_highlights(content, Decimal("32")) == [
        {"start": 0.0, "end": 32.0}
    ]


def test_highlight_contract_keeps_twenty_disjoint_intervals():
    content = json.dumps({
        "highlights": [
            {"start": float(index * 2), "end": float(index * 2 + 1)}
            for index in range(20)
        ]
    })
    assert len(module.validate_highlights(content, Decimal("40"))) == 20


def test_truncated_generation_is_invalid_even_if_json_is_complete():
    with pytest.raises(ValueError, match="did not finish normally"):
        module.validate_generation(
            '{"highlights":[]}', "length", Decimal("10")
        )


def test_window_duration_uses_metadata_endpoints():
    metadata = {"window_start_sec": 24.0, "window_end_sec": 54.433333}
    assert module.window_duration(metadata) == Decimal("30.433333")


def test_temporal_metrics_keep_all_duration_in_false_retain():
    metrics = module.temporal_metrics(
        [{"start": 0.0, "end": 4.0}],
        [{"start": 2.0, "end": 4.0}],
    )
    assert metrics["false_retain_duration_fraction"] == pytest.approx(0.5)


def test_source_intervals_add_window_offset():
    record = {
        "window": {"window_start_sec": 24.0, "window_end_sec": 54.0},
        "highlights": [{"start": 1.0, "end": 3.0}],
        "reference_highlights": [{"start": 2.0, "end": 5.0}],
    }
    assert module.source_intervals(record, prediction=True) == [[25.0, 27.0]]
    assert module.source_intervals(record, prediction=False) == [[26.0, 29.0]]


def test_materialized_reference_uses_manifest_media():
    metadata = {
        "media": {
            "window.mp4": {"path": "/real/data/window.mp4", "sha256": "abc"}
        }
    }
    assert (
        module.materialized_reference(metadata, "data/window.mp4")
        == "/real/data/window.mp4"
    )


def test_materialized_reference_does_not_guess_ambiguous_media():
    metadata = {
        "media": {
            "a/window.mp4": {"path": "/real/a/window.mp4"},
            "b/window.mp4": {"path": "/real/b/window.mp4"},
        }
    }
    assert (
        module.materialized_reference(metadata, "offsets/window.mp4")
        == "offsets/window.mp4"
    )


def test_row_window_metadata_accepts_training_protocol_fields():
    row = {
        "window_start_sec": 24.0,
        "window_end_sec": 54.0,
        "group": "video-1",
        "source_video": "videos/1.mp4",
    }
    metadata = module.row_window_metadata(row)
    assert metadata["window_start_sec"] == 24.0
    assert metadata["group"] == "video-1"


def test_row_window_metadata_rejects_missing_timing():
    with pytest.raises(ValueError, match="refusing to guess timing"):
        module.row_window_metadata({"group": "video-1"})


def test_attach_manifest_metadata_requires_exact_same_order(tmp_path):
    video = tmp_path / "window.mp4"
    answer = {"highlights": [{"start": 1.0, "end": 2.0}]}
    row = {
        "messages": [
            {"role": "user", "content": "<video>\nPrompt"},
            {
                "role": "assistant",
                "content": json.dumps(answer, separators=(",", ":")),
            },
        ],
        "videos": [str(video)],
    }
    metadata = {
        "group": "video-1",
        "window_start_sec": 24.0,
        "window_end_sec": 56.0,
        "window_highlights": answer["highlights"],
        "window_video": str(video),
    }
    attached = module.attach_manifest_metadata([row], [metadata])
    assert attached[0]["window_metadata"] == metadata
    assert "window_metadata" not in row


def test_attach_manifest_metadata_rejects_answer_drift(tmp_path):
    video = tmp_path / "window.mp4"
    row = {
        "messages": [
            {"role": "user", "content": "<video>\nPrompt"},
            {
                "role": "assistant",
                "content": '{"highlights":[{"start":1.0,"end":2.0}]}',
            },
        ],
        "videos": [str(video)],
    }
    metadata = {
        "group": "video-1",
        "window_start_sec": 0.0,
        "window_end_sec": 10.0,
        "window_highlights": [{"start": 1.0, "end": 3.0}],
        "window_video": str(video),
    }
    with pytest.raises(ValueError, match="answer and window metadata differ"):
        module.attach_manifest_metadata([row], [metadata])


def test_attach_manifest_metadata_rejects_video_name_drift(tmp_path):
    row = {
        "messages": [
            {"role": "user", "content": "<video>\nPrompt"},
            {
                "role": "assistant",
                "content": '{"highlights":[{"start":1.0,"end":2.0}]}',
            },
        ],
        "videos": [str(tmp_path / "row.mp4")],
    }
    metadata = {
        "group": "video-1",
        "window_start_sec": 0.0,
        "window_end_sec": 10.0,
        "window_highlights": [{"start": 1.0, "end": 2.0}],
        "window_video": str(tmp_path / "metadata.mp4"),
    }
    with pytest.raises(ValueError, match="video names differ"):
        module.attach_manifest_metadata([row], [metadata])
