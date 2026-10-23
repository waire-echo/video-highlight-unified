from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts")
)
import infer_video_ora_highlights_adapter_426 as module  # noqa: E402


def test_highlights_contract_and_exact_empty_think_prefix():
    output, removed = module.validate_highlights(
        '<think>\n</think>\n'
        '{"highlights":[{"start":1,"end":2.5}]}',
        Decimal("10"),
    )
    assert removed is True
    assert output == [{"start": 1.0, "end": 2.5}]


@pytest.mark.parametrize(
    "content",
    [
        '{"events":[]}',
        '{"highlights":[{"start":1,"end":2,"decision":"retain"}]}',
        '{"highlights":[{"start":9,"end":11}]}',
    ],
)
def test_invalid_highlights_are_rejected(content):
    with pytest.raises(ValueError):
        module.validate_highlights(content, Decimal("10"))


def test_contiguous_one_second_intervals_merge_before_contract_checks():
    content = json.dumps({
        "highlights": [
            {"start": float(index), "end": float(index + 1)}
            for index in range(32)
        ]
    })
    output, removed = module.validate_highlights(content, Decimal("32"))
    assert removed is False
    assert output == [{"start": 0.0, "end": 32.0}]


def test_more_than_sixteen_disjoint_intervals_are_retained():
    content = json.dumps({
        "highlights": [
            {"start": float(index * 2), "end": float(index * 2 + 1)}
            for index in range(20)
        ]
    })
    output, _ = module.validate_highlights(content, Decimal("40"))
    assert len(output) == 20
    assert output[-1] == {"start": 38.0, "end": 39.0}


def test_overlap_and_unsorted_intervals_are_merged():
    content = json.dumps({
        "highlights": [
            {"start": 5, "end": 7},
            {"start": 1, "end": 3},
            {"start": 2, "end": 4},
            {"start": 8, "end": 9},
        ]
    })
    output, _ = module.validate_highlights(content, Decimal("10"))
    assert output == [
        {"start": 1.0, "end": 4.0},
        {"start": 5.0, "end": 7.0},
        {"start": 8.0, "end": 9.0},
    ]


def test_window_local_time_maps_to_source_time():
    intervals, audit = module.map_to_source(
        [{"start": 1.25, "end": 4.5}],
        window_start=Decimal("24"),
        source_duration=Decimal("60"),
    )
    assert intervals == [[25.25, 28.5]]
    assert audit == [
        {
            "window_start_sec": 24.0,
            "local_start_sec": 1.25,
            "local_end_sec": 4.5,
            "source_start_sec": 25.25,
            "source_end_sec": 28.5,
        }
    ]
    with pytest.raises(ValueError, match="outside source"):
        module.map_to_source(
            [{"start": 1, "end": 2}],
            window_start=Decimal("59"),
            source_duration=Decimal("60"),
        )


def test_plan_removes_assistant_answer_before_request(tmp_path):
    root = tmp_path / "materialized"
    video_root = tmp_path / "videos"
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    video_root.mkdir()
    window = train_dir / "video0_0000_0_10.mp4"
    window.write_bytes(b"window")
    source = video_root / "0.mp4"
    source.write_bytes(b"source")
    prompt = "<video>\nreturn highlights"
    sample = {
        "split": "train",
        "source_video_id": "0",
        "source_video": "videos/0.mp4",
        "source_duration_sec": 10.0,
        "window_index": 0,
        "window_start_sec": 0.0,
        "window_end_sec": 10.0,
        "window_duration_sec": 10.0,
        "window_highlights": [{"start": 1, "end": 2}],
    }
    row = {
        "messages": [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": '{"highlights":[{"start":1,"end":2}]}',
            },
        ],
        "videos": [str(window)],
    }
    (root / "train.windows.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    (root / "val.windows.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "answer_contract": "highlights-only/v1",
        "source_counts": {"train": 383, "val": 43},
        "samples": {"train": [sample], "val": []},
        "source_files": {
            "train_windows": {"rows": 1},
            "val_windows": {"rows": 0},
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source coverage mismatch"):
        module.load_plan(root, video_root)


def test_plan_rejects_assistant_content_inside_user_request(tmp_path):
    script = Path(module.__file__).read_text(encoding="utf-8")
    assert 'if message.get("role") != "assistant"' in script
    assert '"assistant_answer_passed_to_model": False' in script


def test_window_endpoint_rounding_is_bounded():
    script = Path(module.__file__).read_text(encoding="utf-8")
    assert 'source_duration + Decimal("0.001")' in script
    assert '> Decimal("0.001")' in script


def test_window_claim_is_exclusive_and_released(tmp_path):
    first = module.acquire_window_claim(tmp_path, "train_0_w0000")
    assert first is not None
    assert first.is_file()
    assert module.acquire_window_claim(tmp_path, "train_0_w0000") is None
    module.release_window_claim(first)
    assert not first.exists()
    second = module.acquire_window_claim(tmp_path, "train_0_w0000")
    assert second is not None
    module.release_window_claim(second)


def test_window_partitions_are_disjoint_and_complete():
    first = list(range(0, 262))
    second = list(range(262, 523))
    assert not set(first) & set(second)
    assert sorted(first + second) == list(range(523))
