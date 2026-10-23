import json
from argparse import Namespace
from pathlib import Path

import pytest

from merge_video_ora_highlights_lenient_diagnostic import merge


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_lenient_diagnostic_preserves_strict_and_repairs_end_overrun(
    tmp_path,
):
    strict = tmp_path / "strict"
    strict.mkdir()
    record_path = tmp_path / "record.json"
    write_json(
        record_path,
        {
            "window_id": "w1",
            "source_video_id": "1",
            "window_start_sec": 10.0,
            "window_end_sec": 20.0,
            "window_duration_sec": 10.0,
            "source_duration_sec": 30.0,
            "raw_json_valid": False,
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>\n\n</think>\n\n"
                                '{"highlights":[{"start":2.0,"end":10.2}]}'
                            )
                        }
                    }
                ]
            },
        },
    )
    write_jsonl(
        strict / "highlights.jsonl",
        [
            {
                "video_id": "1",
                "duration": 30.0,
                "highlights": [{"start": 1.0, "end": 2.0}],
            }
        ],
    )
    write_jsonl(
        strict / "windows.jsonl",
        [
            {
                "window_id": "w0",
                "source_video_id": "1",
                "window_start_sec": 0.0,
                "window_end_sec": 10.0,
                "raw_json_valid": True,
                "source_highlights": [[1.0, 2.0]],
            },
            {
                "window_id": "w1",
                "source_video_id": "1",
                "window_start_sec": 10.0,
                "window_end_sec": 20.0,
                "raw_json_valid": False,
            },
        ],
    )
    write_jsonl(
        strict / "failures.jsonl",
        [
            {
                "video_id": "1",
                "window_id": "w1",
                "error": "end outside window",
                "record_path": str(record_path),
            }
        ],
    )

    output = tmp_path / "lenient"
    summary = merge(
        Namespace(
            strict_merged=strict,
            output=output,
            end_tolerance_sec=0.5,
        )
    )

    assert summary["strict_valid_windows_preserved"] == 1
    assert summary["lenient_repaired_windows"] == 1
    assert summary["valid_windows_after_lenient"] == 2
    assert summary["unresolved_windows"] == 0
    highlights = [
        json.loads(line)
        for line in (output / "highlights.jsonl").read_text().splitlines()
    ]
    assert highlights[0]["highlights"] == [
        {"start": 1.0, "end": 2.0},
        {"start": 12.0, "end": 20.0},
    ]
    repair = json.loads(
        (output / "lenient_repairs.jsonl").read_text().strip()
    )
    assert repair["normalization"]["policy"] == "diagnostic_lenient/v2"
    assert repair["normalization"]["time_changes"][0]["reason"] == (
        "clipped_to_video_end"
    )
    assert repair["lenient_window_highlights"] == [[2.0, 10.0]]


def test_lenient_diagnostic_keeps_oom_without_response(tmp_path):
    strict = tmp_path / "strict"
    strict.mkdir()
    record_path = tmp_path / "record.json"
    write_json(
        record_path,
        {
            "window_id": "w1",
            "source_video_id": "1",
            "window_start_sec": 0.0,
            "window_end_sec": 10.0,
            "window_duration_sec": 10.0,
            "source_duration_sec": 10.0,
            "raw_json_valid": False,
        },
    )
    write_jsonl(
        strict / "highlights.jsonl",
        [{"video_id": "1", "duration": 10.0, "highlights": []}],
    )
    write_jsonl(
        strict / "windows.jsonl",
        [
            {
                "window_id": "w1",
                "source_video_id": "1",
                "window_start_sec": 0.0,
                "window_end_sec": 10.0,
                "raw_json_valid": False,
            }
        ],
    )
    write_jsonl(
        strict / "failures.jsonl",
        [
            {
                "video_id": "1",
                "window_id": "w1",
                "error": "OutOfMemoryError",
                "record_path": str(record_path),
            }
        ],
    )

    output = tmp_path / "lenient"
    summary = merge(
        Namespace(
            strict_merged=strict,
            output=output,
            end_tolerance_sec=0.5,
        )
    )

    assert summary["lenient_repaired_windows"] == 0
    assert summary["unresolved_windows"] == 1
    assert summary["state"] == "complete_with_failures"
    assert summary["unresolved"][0]["window_id"] == "w1"


def test_lenient_diagnostic_refuses_to_overwrite_strict(tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        merge(
            Namespace(
                strict_merged=tmp_path,
                output=tmp_path,
                end_tolerance_sec=0.5,
            )
        )


def test_lenient_source_mapping_absorbs_decimal_roundoff(tmp_path):
    strict = tmp_path / "strict"
    strict.mkdir()
    record_path = tmp_path / "record.json"
    write_json(
        record_path,
        {
            "window_id": "w1",
            "source_video_id": "1",
            "window_start_sec": 24.0,
            "window_end_sec": 35.28,
            "window_duration_sec": 11.280000000000001,
            "source_duration_sec": 35.28,
            "raw_json_valid": False,
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"highlights":[{"start":0.0,"end":11.3}]}'
                            )
                        }
                    }
                ]
            },
        },
    )
    write_jsonl(
        strict / "highlights.jsonl",
        [{"video_id": "1", "duration": 35.28, "highlights": []}],
    )
    write_jsonl(
        strict / "windows.jsonl",
        [
            {
                "window_id": "w1",
                "source_video_id": "1",
                "window_start_sec": 24.0,
                "window_end_sec": 35.28,
                "raw_json_valid": False,
            }
        ],
    )
    write_jsonl(
        strict / "failures.jsonl",
        [
            {
                "video_id": "1",
                "window_id": "w1",
                "error": "outside window",
                "record_path": str(record_path),
            }
        ],
    )

    output = tmp_path / "lenient"
    summary = merge(
        Namespace(
            strict_merged=strict,
            output=output,
            end_tolerance_sec=0.5,
        )
    )

    assert summary["lenient_repaired_windows"] == 1
    highlights = json.loads(
        (output / "highlights.jsonl").read_text().strip()
    )
    assert highlights["highlights"] == [{"start": 24.0, "end": 35.28}]
