import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_video_ora_highlights_inference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "merge_video_ora_highlights_inference", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_run(root, name, windows, values):
    run = root / name
    plan = {
        "model": "model",
        "adapter": "adapter",
        "adapter_sha256": "adapter-sha",
        "materialized_root": "data",
        "manifest_sha256": "manifest-sha",
        "video_root": "video",
        "generation": {"fps": 2, "max_frames": 64},
        "prompt": [{"role": "user", "content": "prompt"}],
        "assistant_answers_passed_to_model": False,
    }
    write_json(run / "plan.json", plan)
    write_json(
        run / "source_index.json",
        [
            {
                "video_id": "0",
                "duration": 20.0,
                "source_video": "/video/0.mp4",
                "windows": [
                    {
                        "window_id": "train_0_w0000",
                        "split": "train",
                        "window_index": 0,
                        "window_start_sec": 0.0,
                        "window_end_sec": 12.0,
                    },
                    {
                        "window_id": "train_0_w0001",
                        "split": "train",
                        "window_index": 1,
                        "window_start_sec": 8.0,
                        "window_end_sec": 20.0,
                    },
                ],
            }
        ],
    )
    for index, valid in windows:
        window_id = f"train_0_w{index:04d}"
        start = 0.0 if index == 0 else 8.0
        end = 12.0 if index == 0 else 20.0
        record = {
            "window_id": window_id,
            "source_video_id": "0",
            "window_start_sec": start,
            "window_end_sec": end,
            "request_sha256": f"request-{index}",
            "window_video_sha256": f"video-{index}",
            "raw_json_valid": valid,
            "source_highlights": values.get(index, []),
            "validation_error": None if valid else "bad output",
        }
        write_json(run / "records" / f"{window_id}.json", record)
    return run


def test_merge_deduplicates_and_unions_overlapping_windows(tmp_path):
    first = write_run(
        tmp_path,
        "first",
        [(0, True), (1, True)],
        {0: [{"start": 1.0, "end": 4.0}], 1: [{"start": 10.0, "end": 12.0}]},
    )
    second = write_run(
        tmp_path,
        "second",
        [(1, True)],
        {1: [{"start": 10.0, "end": 12.0}]},
    )
    output = tmp_path / "merged"
    summary = MODULE.merge(
        type(
            "Args",
            (),
            {"runs": [first, second], "output": output},
        )
    )
    assert summary["state"] == "complete"
    assert summary["total_windows"] == 2
    assert summary["valid_windows"] == 2
    rows = [
        json.loads(line)
        for line in (output / "highlights.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {
            "video_id": "0",
            "duration": 20.0,
            "highlights": [
                {"start": 1.0, "end": 4.0},
                {"start": 10.0, "end": 12.0},
            ],
        }
    ]


def test_merge_rejects_missing_windows(tmp_path):
    first = write_run(tmp_path, "first", [(0, True)], {})
    output = tmp_path / "merged"
    try:
        MODULE.merge(
            type(
                "Args",
                (),
                {"runs": [first], "output": output},
            )
        )
    except ValueError as error:
        assert "Missing 1 windows" in str(error)
    else:
        raise AssertionError("Expected missing-window failure")
