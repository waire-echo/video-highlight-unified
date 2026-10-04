import importlib.util
import json
import os
from types import SimpleNamespace

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_PATH = os.path.join(ROOT, "scripts", "run_modified_baseline.py")
SPEC = importlib.util.spec_from_file_location("run_modified_baseline", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _write_config(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _required_args(config_path):
    return [
        "--config",
        config_path,
        "--index",
        "index.json",
        "--video-dir",
        "videos",
        "--out",
        "out.jsonl",
    ]


def test_repository_default_config_allows_automatic_attention():
    args = runner.parse_args(_required_args(os.path.join(ROOT, 'configs', 'default.json')))
    assert args.window_attn_implementation is None


def test_config_values_are_loaded_and_cli_explicit_values_win(tmp_path):
    config_path = _write_config(
        tmp_path / "config.json",
        {
            "temporal_mode": "full",
            "min_segment_sec": 7.0,
            "short_video_full_sec": 12.0,
        },
    )
    args = runner.parse_args(
        _required_args(config_path)
        + ["--temporal-mode", "motion", "--min-segment-sec", "5"]
    )
    assert args.temporal_mode == "motion"
    assert args.min_segment_sec == 5.0
    assert args.short_video_full_sec == 12.0


def test_unknown_config_key_is_rejected(tmp_path):
    config_path = _write_config(tmp_path / "config.json", {"not_a_real_option": 1})
    with pytest.raises(SystemExit):
        runner.parse_args(_required_args(config_path))


def test_spatial_defaults_remain_unchanged(tmp_path):
    config_path = _write_config(tmp_path / "config.json", {})
    args = runner.parse_args(_required_args(config_path))
    assert args.spatial_backend == "yolo"
    assert args.yolo_model == "yolov8n.pt"
    assert args.yolo_conf == 0.25
    assert args.crop_stride == 15
    assert args.smooth_window == 7


def test_qwen_video_budget_defaults_are_loaded(tmp_path):
    config_path = _write_config(tmp_path / "config.json", {})
    args = runner.parse_args(_required_args(config_path))
    assert args.vlm_max_video_frames == 64
    assert args.vlm_load_in_4bit is False
    assert args.vlm_max_frame_pixels == 360 * 420
    assert args.vlm_total_video_pixels == 4096 * 28 * 28
    assert args.max_new_tokens == 128


def test_qwen_4bit_config_can_be_enabled_and_disabled(tmp_path):
    config_path = _write_config(tmp_path / "config.json", {"vlm_load_in_4bit": True})
    enabled = runner.parse_args(_required_args(config_path))
    disabled = runner.parse_args(
        _required_args(config_path) + ["--no-vlm-load-in-4bit"]
    )
    assert enabled.vlm_load_in_4bit is True
    assert disabled.vlm_load_in_4bit is False


def test_window_score_mode_and_coarse_fine_defaults_are_available(tmp_path):
    config_path = _write_config(tmp_path / "config.json", {})
    args = runner.parse_args(
        _required_args(config_path) + ["--temporal-mode", "window_score"]
    )
    assert args.temporal_mode == "window_score"
    assert (args.window_coarse_sec, args.window_coarse_stride_sec, args.window_coarse_fps) == (
        2.0,
        1.0,
        2.0,
    )
    assert (args.window_fine_sec, args.window_fine_stride_sec, args.window_fine_fps) == (
        2.0,
        0.5,
        4.0,
    )
    assert args.window_cache_root.startswith("E:\\AI\\")
    assert args.window_log_root.startswith("E:\\AI\\")
    assert args.window_absolute_threshold == 0.45
    assert args.window_growth_threshold == 0.40
    assert args.window_semantic_mode == "structured"
    assert args.window_aggregation == "current"


def test_reze_window_switches_are_explicit_and_configurable(tmp_path):
    config_path = _write_config(
        tmp_path / "config.json",
        {
            "window_semantic_mode": "reze_yesno",
            "window_aggregation": "reze_otsu",
        },
    )
    args = runner.parse_args(
        _required_args(config_path) + ["--temporal-mode", "window_score"]
    )
    assert args.window_semantic_mode == "reze_yesno"
    assert args.window_aggregation == "reze_otsu"


def test_structured_mode_rejects_reze_only_aggregation_before_model_load(tmp_path):
    config_path = _write_config(tmp_path / "config.json", {})
    with pytest.raises(SystemExit):
        runner.parse_args(
            _required_args(config_path)
            + [
                "--temporal-mode",
                "window_score",
                "--window-semantic-mode",
                "structured",
                "--window-aggregation",
                "reze_otsu",
            ]
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows E-drive guard")
def test_window_score_runtime_rejects_c_drive_path():
    assert runner._require_e_absolute(r"E:\AI\tmp\out.jsonl", "output").startswith(
        "E:\\"
    )
    with pytest.raises(ValueError, match="absolute E"):
        runner._require_e_absolute(r"C:\Users\example\out.jsonl", "output")


def test_boolean_cli_can_override_true_config_values(tmp_path):
    config_path = _write_config(
        tmp_path / "config.json", {"save_raw": True, "validate": True}
    )
    args = runner.parse_args(
        _required_args(config_path) + ["--no-save-raw", "--no-validate"]
    )
    assert args.save_raw is False
    assert args.validate is False


def test_summary_counts_template_response_without_marking_invalid_vlm_success():
    args = SimpleNamespace(
        out="out.jsonl",
        config="",
        temporal_mode="vlm",
        temporal_model_path="model",
        vlm_load_in_4bit=True,
        detect_fps=1.0,
        vlm_max_video_frames=64,
        vlm_max_frame_pixels=360 * 420,
        vlm_total_video_pixels=4096 * 28 * 28,
        max_new_tokens=128,
        dtype="float16",
        device_map="auto",
    )
    records = [{"video_id": "0", "predictions": []}]
    raw_records = [
        {
            "source": "vlm",
            "valid": False,
            "attempts": 2,
            "fallback_used": None,
            "coverage_ratio": 0.0,
            "vlm_raw_responses": ['{"segments": [[10, 20]]}'],
            "errors": ["outside video duration"],
            "corrections": [],
        }
    ]
    summary = runner._build_summary(args, records, raw_records, total_predictions=0)
    assert summary["vlm_first_success_count"] == 0
    assert summary["vlm_settings"]["load_in_4bit"] is True
    assert summary["vlm_settings"]["max_video_frames"] == 64
    assert summary["fixed_start_10_sec_count"] == 1
    assert summary["fixed_end_20_sec_count"] == 1


def test_window_summary_does_not_treat_many_first_attempt_windows_as_retry():
    args = SimpleNamespace(
        out="out.jsonl",
        config="",
        temporal_mode="window_score",
        temporal_model_path=r"E:\models\qwen",
    )
    records = [{"video_id": "76", "predictions": [{"frame": 1, "bboxes": [0, 0, 1]}]}]
    raw_records = [
        {
            "source": "window_score_curve",
            "valid": True,
            "attempts": 17,
            "coverage_ratio": 0.2,
            "window_records": [
                {"stage": "coarse", "valid": True, "attempts": 1}
                for _ in range(17)
            ],
        }
    ]
    summary = runner._build_summary(args, records, raw_records, 1)
    assert summary["videos_with_retry"] == 0
