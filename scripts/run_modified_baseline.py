#!/usr/bin/env python3
import argparse
import inspect
import json
import math
import ntpath
import os
import statistics
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from video_highlight.geometry import (
    absolute_center_to_box,
    compute_crop_size,
    densify_boxes,
    keyframes_for_segment,
    merge_segments,
    sec_segments_to_frames,
    smooth_boxes,
)
from video_highlight.io_utils import extract_frames, load_index, video_meta, video_path_for, write_jsonl
from video_highlight.spatial_legacy import build_spatial_detector
from video_highlight.temporal import build_temporal_detector, parse_temporal_response
from video_highlight.validation import validate_submission_for_items


DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "default.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modified baseline: VLM/motion temporal localization + YOLO spatial detection + interpolation."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="JSON config file")
    parser.add_argument("--index", default="", help="test_index.json or directory with per-video jsonl files")
    parser.add_argument("--video-dir", default="", help="Directory containing <video_id>.mp4")
    parser.add_argument("--out", default="", help="Output submit.jsonl")
    parser.add_argument("--num-videos", type=int, default=0, help="Process first N videos; 0 means all")

    parser.add_argument(
        "--temporal-mode",
        choices=["motion", "middle", "full", "vlm", "window_score", "pgl_sum"],
        default="motion",
    )
    parser.add_argument(
        "--temporal-model-path",
        default="",
        help="Local Qwen checkpoint for --temporal-mode vlm/window_score",
    )
    parser.add_argument("--detect-fps", type=float, default=1.0)
    parser.add_argument("--top-k-segments", type=int, default=1)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--min-segment-sec", type=float, default=4.0, help="Minimum final temporal segment length")
    parser.add_argument(
        "--motion-min-segment-sec",
        type=float,
        default=1.0,
        help="Minimum segment length used by the motion detector",
    )
    parser.add_argument("--padding-sec", type=float, default=0.25, help="Motion detector window padding")
    parser.add_argument("--segment-padding-sec", type=float, default=3.0, help="Final temporal segment padding")
    parser.add_argument("--vlm-retries", type=int, default=1)
    parser.add_argument(
        "--vlm-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the temporal VLM with bitsandbytes 4-bit inference quantization",
    )
    parser.add_argument("--vlm-max-video-frames", type=int, default=64)
    parser.add_argument("--vlm-max-frame-pixels", type=int, default=360 * 420)
    parser.add_argument("--vlm-total-video-pixels", type=int, default=4096 * 28 * 28)
    parser.add_argument("--temporal-fallback", choices=["motion", "full", "middle", "none"], default="motion")
    parser.add_argument("--short-video-full-sec", type=float, default=15.0)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.30)
    parser.add_argument("--max-coverage-sec", type=float, default=30.0)
    parser.add_argument("--max-temporal-segments", type=int, default=3)
    parser.add_argument("--subject-hint", default="", help="Optional global subject hint, e.g. person/ball/car")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--pgl-checkpoint-path",
        default=r"E:\AI\models\mrhisum_pgl_sum\checkpoints\pgl_sum1_best_f1.pkl",
    )
    parser.add_argument(
        "--pgl-feature-model-dir", default=r"E:\AI\models\youtube8m"
    )
    parser.add_argument(
        "--pgl-feature-cache-dir", default=r"E:\AI\cache\youtube8m_features"
    )
    parser.add_argument("--pgl-sample-fps", type=float, default=1.0)
    parser.add_argument("--pgl-boundary-grid-sec", type=float, default=0.25)
    parser.add_argument("--pgl-device", default="auto")
    parser.add_argument("--pgl-hints-jsonl", default="")

    # Experimental Qwen window-score temporal frontend.  These names are kept
    # separate from the legacy motion/VLM window settings so both pipelines can
    # be reproduced without silently changing one another.
    parser.add_argument("--window-coarse-sec", type=float, default=2.0)
    parser.add_argument("--window-coarse-stride-sec", type=float, default=1.0)
    parser.add_argument("--window-coarse-fps", type=float, default=2.0)
    parser.add_argument("--window-fine-sec", type=float, default=2.0)
    parser.add_argument("--window-fine-stride-sec", type=float, default=0.5)
    parser.add_argument("--window-fine-fps", type=float, default=4.0)
    parser.add_argument("--window-max-fine-regions", type=int, default=6)
    parser.add_argument("--window-score-retries", type=int, default=1)
    parser.add_argument("--window-max-new-tokens", type=int, default=192)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument(
        "--window-attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument(
        "--window-compact-output",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--window-curve-step-sec", type=float, default=0.5)
    parser.add_argument("--window-smoothing-points", type=int, default=3)
    parser.add_argument("--window-relative-top-fraction", type=float, default=0.30)
    parser.add_argument("--window-absolute-threshold", type=float, default=0.45)
    parser.add_argument("--window-growth-threshold", type=float, default=0.40)
    parser.add_argument("--window-coarse-candidate-threshold", type=float, default=0.45)
    parser.add_argument("--window-merge-gap-sec", type=float, default=1.0)
    parser.add_argument("--window-boundary-expand-sec", type=float, default=0.75)
    parser.add_argument("--window-max-boundary-expand-sec", type=float, default=1.5)
    parser.add_argument("--window-max-segments", type=int, default=12)
    parser.add_argument("--window-max-coverage-ratio", type=float, default=0.60)
    parser.add_argument("--window-max-coverage-sec", type=float, default=30.0)
    parser.add_argument("--window-short-full-sec", type=float, default=4.0)
    parser.add_argument("--window-audio-sample-rate", type=int, default=8000)
    parser.add_argument("--window-ffmpeg-path", default="ffmpeg")
    parser.add_argument(
        "--window-cache-root",
        default=r"E:\AI\tmp\video_highlight\window_score",
    )
    parser.add_argument(
        "--window-log-root",
        default=r"E:\AI\logs\video_highlight",
    )
    parser.add_argument(
        "--window-semantic-mode",
        choices=["structured", "global_event", "reze_yesno"],
        default="structured",
        help="Structured scoring, global-event conditioning, or opt-in REZE Yes/No confidence",
    )
    parser.add_argument("--global-event-max-events", type=int, default=3)
    parser.add_argument("--global-event-long-video-sec", type=float, default=60.0)
    parser.add_argument("--global-event-overview-chunk-sec", type=float, default=30.0)
    parser.add_argument("--global-event-overview-max-frames", type=int, default=64)
    parser.add_argument("--global-event-seed-threshold", type=float, default=0.55)
    parser.add_argument("--global-event-growth-threshold", type=float, default=0.40)
    parser.add_argument("--global-event-valley-threshold", type=float, default=0.35)
    parser.add_argument("--global-event-min-prominence", type=float, default=0.10)
    parser.add_argument("--global-event-padding-sec", type=float, default=0.50)
    parser.add_argument("--global-event-merge-gap-sec", type=float, default=0.25)
    parser.add_argument("--global-event-max-coverage-ratio", type=float, default=0.75)
    parser.add_argument(
        "--window-aggregation",
        choices=["current", "reze_otsu"],
        default="current",
        help="Temporal curve aggregation; reze_otsu is experimental",
    )

    parser.add_argument("--spatial-backend", choices=["yolo", "center", "motion"], default="yolo")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-ball-conf", type=float, default=0.15)
    parser.add_argument("--spatial-scoring", choices=["legacy", "enhanced"], default="enhanced")
    parser.add_argument(
        "--spatial-stateful",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--spatial-continuity-weight", type=float, default=0.30)
    parser.add_argument("--spatial-max-missed-keyframes", type=int, default=1)
    parser.add_argument("--spatial-person-ball-weight", type=float, default=0.20)
    parser.add_argument("--max-raw-spatial-candidates", type=int, default=5)
    parser.add_argument("--crop-stride", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=7)

    parser.add_argument(
        "--save-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write debug raw JSONL next to output",
    )
    parser.add_argument("--max-raw-spatial", type=int, default=24)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _load_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> dict:
    if not config_path:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        parser.error(f"config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON config {config_path}: {exc}")
    if not isinstance(config, dict):
        parser.error(f"config must contain a JSON object: {config_path}")

    actions = {action.dest: action for action in parser._actions if action.dest not in {"help", "config"}}
    unknown = sorted(set(config) - set(actions))
    if unknown:
        parser.error(f"unknown config key(s): {', '.join(unknown)}")

    normalized = {}
    for key, value in config.items():
        action = actions[key]
        if value is None and action.default is None:
            normalized[key] = None
            continue
        if isinstance(
            action,
            (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction),
        ):
            if not isinstance(value, bool):
                parser.error(f"config key {key!r} must be a boolean")
            converted = value
        elif action.type is not None:
            try:
                converted = action.type(value)
            except (TypeError, ValueError):
                parser.error(f"invalid value for config key {key!r}: {value!r}")
        else:
            converted = value
        if action.choices is not None and converted not in action.choices:
            parser.error(
                f"invalid value for config key {key!r}: {converted!r}; "
                f"choose from {', '.join(str(choice) for choice in action.choices)}"
            )
        normalized[key] = converted
    return normalized


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = _build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _unknown = config_parser.parse_known_args(argv)
    config_defaults = _load_config_defaults(parser, config_args.config)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    for required_name in ("index", "video_dir", "out"):
        if not getattr(args, required_name):
            parser.error(f"--{required_name.replace('_', '-')} is required (CLI or config)")
    if (
        args.window_semantic_mode in {"structured", "global_event"}
        and args.window_aggregation != "current"
    ):
        parser.error(
            "structured/global_event semantic modes require --window-aggregation current"
        )
    return args


def _require_e_absolute(path: str, label: str, *, must_exist: bool = False) -> str:
    """Validate a window-score runtime path before importing the model stack."""

    raw = os.path.expanduser(str(path))
    if os.name == "nt" and not ntpath.isabs(raw):
        raise ValueError(f"{label} must be an absolute E: path for window_score: {path}")
    value = os.path.abspath(raw)
    drive, _tail = ntpath.splitdrive(value)
    # ``os.path.abspath`` on Windows is sufficient for normal CLI paths; the
    # ntpath check also keeps the guard testable with synthetic E:\\ strings.
    if os.name == "nt" and (drive.upper() != "E:" or not ntpath.isabs(value)):
        raise ValueError(f"{label} must be an absolute E: path for window_score: {path}")
    if must_exist and not os.path.exists(path):
        raise ValueError(f"{label} does not exist: {path}")
    return value


def _prepare_window_score_runtime(args) -> None:
    """Make model-backed temporal modes obey the E-drive runtime policy."""

    temporal_mode = getattr(args, "temporal_mode", "motion")
    if temporal_mode not in {"window_score", "pgl_sum"} or os.name != "nt":
        return
    if temporal_mode == "window_score":
        _require_e_absolute(args.temporal_model_path, "temporal model path", must_exist=True)
    else:
        args.pgl_checkpoint_path = _require_e_absolute(
            args.pgl_checkpoint_path, "PGL-SUM checkpoint", must_exist=True
        )
        args.pgl_feature_model_dir = _require_e_absolute(
            args.pgl_feature_model_dir,
            "YouTube-8M feature model directory",
            must_exist=True,
        )
        args.pgl_feature_cache_dir = _require_e_absolute(
            args.pgl_feature_cache_dir, "YouTube-8M feature cache"
        )
        if args.pgl_hints_jsonl:
            args.pgl_hints_jsonl = _require_e_absolute(
                args.pgl_hints_jsonl, "cached semantic hints", must_exist=True
            )
    cache_root = _require_e_absolute(args.window_cache_root, "window cache root")
    output_path = _require_e_absolute(args.out, "output path")
    log_root = _require_e_absolute(
        getattr(args, "window_log_root", r"E:\AI\logs\video_highlight"),
        "window log root",
    )
    directories = [cache_root, log_root]
    if temporal_mode == "pgl_sum":
        directories.append(args.pgl_feature_cache_dir)
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    ai_root = r"E:\AI"
    cache_root_env = os.path.join(ai_root, "cache", "huggingface")
    temp_root = os.path.join(ai_root, "tmp")
    runtime_env = {
        "HF_HOME": cache_root_env,
        "HF_HUB_CACHE": os.path.join(cache_root_env, "hub"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(cache_root_env, "hub"),
        "HF_ASSETS_CACHE": os.path.join(cache_root_env, "assets"),
        "HF_DATASETS_CACHE": os.path.join(cache_root_env, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(cache_root_env, "transformers"),
        "XDG_CACHE_HOME": os.path.join(ai_root, "cache"),
        "TORCH_HOME": os.path.join(ai_root, "cache", "torch"),
        "TORCH_EXTENSIONS_DIR": os.path.join(ai_root, "cache", "torch_extensions"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(ai_root, "cache", "torchinductor"),
        "TRITON_CACHE_DIR": os.path.join(ai_root, "cache", "triton"),
        "CUDA_CACHE_PATH": os.path.join(ai_root, "cache", "cuda"),
        "PIP_CACHE_DIR": os.path.join(ai_root, "cache", "pip"),
        "UV_CACHE_DIR": os.path.join(ai_root, "cache", "uv"),
        "PYTHONPYCACHEPREFIX": os.path.join(ai_root, "cache", "pycache"),
        "TEMP": temp_root,
        "TMP": temp_root,
        "TMPDIR": temp_root,
        "QWEN_VIDEO_COMPAT_DIR": os.path.join(temp_root, "qwen_video_compat"),
        "VIDEO_HIGHLIGHT_LOG_DIR": log_root,
    }
    for key, value in runtime_env.items():
        os.makedirs(value, exist_ok=True)
        os.environ[key] = value
    os.environ["VIDEO_HIGHLIGHT_SET_TEMP"] = "1"
    # Keep the normalised absolute path for subsequent output creation.
    args.out = output_path


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _json_ready(value):
    """Convert optional window-score diagnostics into JSON-safe values."""
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _coverage_ratio(segments, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    clean = []
    for item in segments or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start, end = sorted((start, end))
        start = max(0.0, min(start, duration_sec))
        end = max(0.0, min(end, duration_sec))
        if end > start:
            clean.append((start, end))
    if not clean:
        return 0.0
    clean.sort()
    merged = [list(clean[0])]
    for start, end in clean[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return min(1.0, sum(end - start for start, end in merged) / duration_sec)


def _call_spatial_lifecycle(detector, method_name: str, *args):
    method = getattr(detector, method_name, None)
    if callable(method):
        return method(*args)
    return None


def _detect_spatial_center(detector, frame, subject_hint, frame_id):
    method = detector.detect_center
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "frame_id" in parameters:
        return method(frame, subject_hint=subject_hint, frame_id=frame_id)
    return method(frame, subject_hint=subject_hint)


def process_video(args, vid: str, target):
    video_path = video_path_for(args.video_dir, vid)
    if not os.path.exists(video_path):
        return {
            "record": {"video_id": vid, "targetRatioWH": [int(target[0]), int(target[1])], "predictions": []},
            "raw": {
                "video_id": vid,
                "error": f"missing video: {video_path}",
                "temporal_mode_requested": args.temporal_mode,
                "temporal_source_used": "input_error",
                "source": "input_error",
                "valid": False,
                "validation_errors": [f"missing video: {video_path}"],
                "errors": [f"missing video: {video_path}"],
                "vlm_attempts": 0,
                "attempts": 0,
                "fallback_used": None,
                "duration_sec": None,
                "coverage_ratio": 0.0,
                "vlm_raw_responses": [],
                "segments_before_normalization": [],
                "segments_after_normalization": [],
                "corrections": [],
                "temporal_segments_sec": [],
                "temporal_segments_frame": [],
                "subject_hint": None,
                "action_description": None,
                "temporal_raw": None,
                "num_predictions": 0,
            },
        }

    n_frames, fps, width, height = video_meta(video_path)
    if n_frames <= 0:
        return {
            "record": {"video_id": vid, "targetRatioWH": [int(target[0]), int(target[1])], "predictions": []},
            "raw": {
                "video_id": vid,
                "error": "unreadable video",
                "temporal_mode_requested": args.temporal_mode,
                "temporal_source_used": "input_error",
                "source": "input_error",
                "valid": False,
                "validation_errors": ["unreadable video"],
                "errors": ["unreadable video"],
                "vlm_attempts": 0,
                "attempts": 0,
                "fallback_used": None,
                "duration_sec": None,
                "coverage_ratio": 0.0,
                "vlm_raw_responses": [],
                "segments_before_normalization": [],
                "segments_after_normalization": [],
                "corrections": [],
                "temporal_segments_sec": [],
                "temporal_segments_frame": [],
                "subject_hint": None,
                "action_description": None,
                "temporal_raw": None,
                "num_predictions": 0,
            },
        }

    temporal = args.temporal_detector.detect(video_path)
    subject_hint = args.subject_hint or temporal.subject_hint
    action_description = getattr(temporal, "action_description", None)
    segments = merge_segments(sec_segments_to_frames(temporal.segments_sec, fps, n_frames), n_frames)
    duration_sec = getattr(temporal, "duration_sec", None)
    if duration_sec is None:
        duration_sec = n_frames / fps if fps > 0 else 0.0
    source = getattr(temporal, "source", "") or args.temporal_mode
    valid = getattr(temporal, "valid", None)
    if valid is None:
        valid = bool(temporal.segments_sec)
    errors = _as_list(getattr(temporal, "errors", []))
    attempts = int(getattr(temporal, "attempts", 1) or 0)
    fallback_used = getattr(temporal, "fallback_used", None)
    vlm_raw_responses = _as_list(getattr(temporal, "vlm_raw_responses", []))
    segments_before = getattr(temporal, "segments_before_normalization", None)
    if segments_before is None:
        segments_before = temporal.segments_sec
    segments_after = getattr(temporal, "segments_after_normalization", None)
    if segments_after is None:
        segments_after = temporal.segments_sec
    corrections = _as_list(getattr(temporal, "corrections", []))
    window_records = _json_ready(getattr(temporal, "window_records", []))
    score_curve = _json_ready(getattr(temporal, "score_curve", []))
    selected_peaks = _json_ready(getattr(temporal, "selected_peaks", []))
    score_diagnostics = _json_ready(getattr(temporal, "score_diagnostics", {}))
    coverage_ratio = getattr(temporal, "coverage_ratio", None)
    if coverage_ratio is None:
        coverage_ratio = _coverage_ratio(segments_after, float(duration_sec or 0.0))
    crop_w, crop_h = compute_crop_size(width, height, int(target[0]), int(target[1]))

    predictions = []
    spatial_logs = []
    spatial_summary = {}
    try:
        _call_spatial_lifecycle(
            args.spatial_detector, "begin_video", vid, video_path, fps, width, height
        )
        for segment_index, (start, end) in enumerate(segments):
            _call_spatial_lifecycle(
                args.spatial_detector,
                "begin_segment",
                segment_index,
                start,
                end,
                subject_hint,
            )
            try:
                kfs = keyframes_for_segment(start, end, args.crop_stride)
                frames = extract_frames(video_path, kfs)
                key_frames = []
                key_boxes = []
                for frame_id in kfs:
                    frame = frames.get(frame_id)
                    if frame is None:
                        continue
                    cx, cy, info = _detect_spatial_center(
                        args.spatial_detector, frame, subject_hint, frame_id
                    )
                    box = absolute_center_to_box(cx, cy, width, height, crop_w, crop_h)
                    key_frames.append(frame_id)
                    key_boxes.append(box)
                    if args.save_raw and len(spatial_logs) < max(0, int(args.max_raw_spatial)):
                        info.update(
                            {
                                "video_id": vid,
                                "segment_index": segment_index,
                                "frame": frame_id,
                                "subject_hint": subject_hint,
                            }
                        )
                        spatial_logs.append(info)
                if not key_frames:
                    center_box = absolute_center_to_box(width / 2.0, height / 2.0, width, height, crop_w, crop_h)
                    key_frames = [start, end]
                    key_boxes = [center_box, center_box]

                dense = densify_boxes(start, end, key_frames, key_boxes)
                dense = smooth_boxes(dense, args.smooth_window)
                for frame_id in range(start, end + 1):
                    box = dense.get(frame_id)
                    if not box:
                        continue
                    x, y, w, _h = box
                    predictions.append({"frame": int(frame_id), "bboxes": [int(x), int(y), int(w)]})
            finally:
                _call_spatial_lifecycle(args.spatial_detector, "end_segment")
    finally:
        detector_summary = _call_spatial_lifecycle(args.spatial_detector, "end_video")
        if isinstance(detector_summary, dict):
            spatial_summary = detector_summary

    predictions.sort(key=lambda item: item["frame"])
    dedup = {}
    for pred in predictions:
        dedup[pred["frame"]] = pred
    predictions = [dedup[f] for f in sorted(dedup)]

    record = {
        "video_id": vid,
        "targetRatioWH": [int(target[0]), int(target[1])],
        "predictions": predictions,
    }
    raw = {
        "video_id": vid,
        "fps": fps,
        "frames": n_frames,
        "duration_sec": duration_sec,
        "size": [width, height],
        "targetRatioWH": [int(target[0]), int(target[1])],
        "temporal_mode_requested": args.temporal_mode,
        "temporal_source_used": source,
        "source": source,
        "valid": bool(valid),
        "validation_errors": errors,
        "errors": errors,
        "vlm_attempts": attempts if args.temporal_mode in {"vlm", "window_score"} else 0,
        "attempts": attempts,
        "fallback_used": fallback_used,
        "coverage_ratio": coverage_ratio,
        "vlm_raw_responses": vlm_raw_responses,
        "segments_before_normalization": segments_before,
        "segments_after_normalization": segments_after,
        "corrections": corrections,
        "window_records": window_records,
        "score_curve": score_curve,
        "selected_peaks": selected_peaks,
        "score_diagnostics": score_diagnostics,
        "temporal_segments_sec": temporal.segments_sec,
        "temporal_segments_frame": segments,
        "subject_hint": subject_hint,
        "action_description": action_description,
        "temporal_raw": temporal.raw,
        "spatial": spatial_logs,
        "spatial_summary": spatial_summary,
        "num_predictions": len(predictions),
    }
    return {"record": record, "raw": raw}


def _aggregate_spatial_summaries(raw_records: List[dict]) -> dict:
    summaries = [raw.get("spatial_summary") for raw in raw_records]
    summaries = [summary for summary in summaries if isinstance(summary, dict)]
    numeric_totals = Counter()
    numeric_distributions = {}
    mapping_totals = {}
    categorical_counts = {}
    boolean_counts = {}

    for summary in summaries:
        for key, value in summary.items():
            if key in {"video_id", "video_path"} or value is None:
                continue
            if isinstance(value, bool):
                counts = boolean_counts.setdefault(key, Counter())
                counts[str(value).lower()] += 1
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                if key.endswith(("_mean", "_avg", "_ratio")):
                    numeric_distributions.setdefault(key, []).append(float(value))
                else:
                    numeric_totals[key] += value
            elif isinstance(value, str):
                categorical_counts.setdefault(key, Counter())[value] += 1
            elif isinstance(value, dict):
                merged = mapping_totals.setdefault(key, Counter())
                for item_key, item_value in value.items():
                    if (
                        isinstance(item_value, (int, float))
                        and not isinstance(item_value, bool)
                        and math.isfinite(float(item_value))
                    ):
                        merged[str(item_key)] += item_value

    distribution_summary = {
        key: {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
        for key, values in sorted(numeric_distributions.items())
        if values
    }
    result = {
        "videos_reported": len(summaries),
        "numeric_totals": dict(sorted(numeric_totals.items())),
        "numeric_distributions": distribution_summary,
        "mapping_totals": {
            key: dict(sorted(counter.items())) for key, counter in sorted(mapping_totals.items())
        },
        "categorical_counts": {
            key: dict(sorted(counter.items())) for key, counter in sorted(categorical_counts.items())
        },
        "boolean_counts": {
            key: dict(sorted(counter.items())) for key, counter in sorted(boolean_counts.items())
        },
    }
    selected_conf_sum = numeric_totals.get("selected_conf_sum", 0.0)
    selected_conf_count = numeric_totals.get("selected_conf_count", 0)
    result["selected_conf_mean"] = (
        float(selected_conf_sum) / float(selected_conf_count) if selected_conf_count else None
    )
    return result


def _build_summary(args, records: List[dict], raw_records: List[dict], total_predictions: int) -> dict:
    source_counts = Counter(str(raw.get("source") or "unknown") for raw in raw_records)
    fallback_counts = Counter(str(raw["fallback_used"]) for raw in raw_records if raw.get("fallback_used"))
    coverage_values = [
        float(raw["coverage_ratio"])
        for raw in raw_records
        if isinstance(raw.get("coverage_ratio"), (int, float)) and math.isfinite(float(raw["coverage_ratio"]))
    ]
    attempts_values = [int(raw.get("attempts", 0) or 0) for raw in raw_records]

    def has_retry(raw: dict) -> bool:
        window_records = [
            item
            for item in _as_list(raw.get("window_records"))
            if isinstance(item, dict)
        ]
        if window_records:
            return any(
                int(item.get("qwen_retry_count") or 0) > 0
                or int(item.get("attempts") or 0) > 1
                for item in window_records
            )
        return int(raw.get("attempts", 0) or 0) > 1

    empty_video_ids = [str(record["video_id"]) for record in records if not record.get("predictions")]
    all_error_messages = [
        str(error).lower()
        for raw in raw_records
        for error in _as_list(raw.get("validation_errors", raw.get("errors", [])))
    ]
    fixed_start_10 = 0
    fixed_end_20 = 0
    for raw in raw_records:
        for response in _as_list(raw.get("vlm_raw_responses")):
            parsed = parse_temporal_response(str(response))
            for start, end in parsed.segments_sec:
                fixed_start_10 += int(abs(float(start) - 10.0) <= 0.05)
                fixed_end_20 += int(abs(float(end) - 20.0) <= 0.05)
    return {
        "output": os.path.abspath(args.out),
        "config": os.path.abspath(args.config) if args.config else None,
        "temporal_mode_requested": args.temporal_mode,
        "vlm_settings": {
            "model_path": os.path.abspath(args.temporal_model_path)
            if getattr(args, "temporal_model_path", "")
            else None,
            "load_in_4bit": bool(getattr(args, "vlm_load_in_4bit", False)),
            "detect_fps": float(getattr(args, "detect_fps", 1.0)),
            "max_video_frames": int(getattr(args, "vlm_max_video_frames", 64)),
            "max_frame_pixels": int(getattr(args, "vlm_max_frame_pixels", 360 * 420)),
            "total_video_pixels": int(
                getattr(args, "vlm_total_video_pixels", 4096 * 28 * 28)
            ),
            "max_new_tokens": int(getattr(args, "max_new_tokens", 128)),
            "dtype": str(getattr(args, "dtype", "auto")),
            "device_map": str(getattr(args, "device_map", "auto")),
        },
        "window_score_settings": {
            "coarse": {
                "window_sec": float(getattr(args, "window_coarse_sec", 2.0)),
                "stride_sec": float(getattr(args, "window_coarse_stride_sec", 1.0)),
                "fps": float(getattr(args, "window_coarse_fps", 2.0)),
            },
            "fine": {
                "window_sec": float(getattr(args, "window_fine_sec", 2.0)),
                "stride_sec": float(getattr(args, "window_fine_stride_sec", 0.5)),
                "fps": float(getattr(args, "window_fine_fps", 4.0)),
                "max_regions": int(getattr(args, "window_max_fine_regions", 6)),
            },
            "absolute_threshold": float(
                getattr(args, "window_absolute_threshold", 0.45)
            ),
            "semantic_mode": str(
                getattr(args, "window_semantic_mode", "structured")
            ),
            "aggregation": str(getattr(args, "window_aggregation", "current")),
            "relative_top_fraction": float(
                getattr(args, "window_relative_top_fraction", 0.30)
            ),
            "cache_root": os.path.abspath(
                str(getattr(args, "window_cache_root", ""))
            ),
            "log_root": os.path.abspath(
                str(
                    getattr(
                        args,
                        "window_log_root",
                        r"E:\AI\logs\video_highlight",
                    )
                )
            ),
            "qwen_score_weights": {
                "action": 0.15,
                "completion": 0.15,
                "interaction": 0.15,
                "event": 0.15,
                "emotion": 0.10,
                "visual_impact": 0.10,
                "highlight": 0.20,
            },
            "qwen_retries": int(getattr(args, "window_score_retries", 1)),
            "max_new_tokens": int(
                getattr(
                    args,
                    "window_max_new_tokens",
                    max(192, int(getattr(args, "max_new_tokens", 128))),
                )
            ),
        },
        "videos_processed": len(records),
        "videos_with_predictions": len(records) - len(empty_video_ids),
        "empty_prediction_videos": len(empty_video_ids),
        "empty_video_ids": empty_video_ids,
        "total_frame_predictions": total_predictions,
        "temporal_source_counts": dict(sorted(source_counts.items())),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "short_video_full_count": source_counts.get("short_video_full", 0),
        "vlm_first_success_count": sum(
            bool(raw.get("valid")) and str(raw.get("source")) == "vlm" for raw in raw_records
        ),
        "vlm_retry_success_count": sum(
            bool(raw.get("valid")) and str(raw.get("source")) == "vlm_retry"
            for raw in raw_records
        ),
        "motion_fallback_count": fallback_counts.get("motion", 0),
        "full_fallback_count": fallback_counts.get("full", 0),
        "invalid_temporal_results": sum(not bool(raw.get("valid", False)) for raw in raw_records),
        "total_temporal_attempts": sum(attempts_values),
        "videos_with_retry": sum(has_retry(raw) for raw in raw_records),
        "total_vlm_raw_responses": sum(len(_as_list(raw.get("vlm_raw_responses"))) for raw in raw_records),
        "window_score": {
            "total_windows": sum(
                len(_as_list(raw.get("window_records"))) for raw in raw_records
            ),
            "coarse_windows": sum(
                sum(
                    1
                    for item in _as_list(raw.get("window_records"))
                    if isinstance(item, dict) and item.get("stage") == "coarse"
                )
                for raw in raw_records
            ),
            "fine_windows": sum(
                sum(
                    1
                    for item in _as_list(raw.get("window_records"))
                    if isinstance(item, dict) and item.get("stage") == "fine"
                )
                for raw in raw_records
            ),
            "valid_qwen_windows": sum(
                sum(
                    1
                    for item in _as_list(raw.get("window_records"))
                    if isinstance(item, dict) and bool(item.get("valid"))
                )
                for raw in raw_records
            ),
            "degenerate_qwen_videos": sum(
                bool((raw.get("score_diagnostics") or {}).get("qwen_degenerate"))
                for raw in raw_records
                if isinstance(raw.get("score_diagnostics"), dict)
            ),
        },
        "temporal_error_count": sum(len(_as_list(raw.get("errors"))) for raw in raw_records),
        "temporal_correction_count": sum(len(_as_list(raw.get("corrections"))) for raw in raw_records),
        "outside_or_illegal_time_error_count": sum(
            "outside" in error or "duration" in error for error in all_error_messages
        ),
        "zero_length_error_count": sum("zero length" in error for error in all_error_messages),
        "fixed_start_10_sec_count": fixed_start_10,
        "fixed_end_20_sec_count": fixed_end_20,
        "spatial": _aggregate_spatial_summaries(raw_records),
        "coverage_ratio": {
            "min": min(coverage_values) if coverage_values else None,
            "max": max(coverage_values) if coverage_values else None,
            "mean": sum(coverage_values) / len(coverage_values) if coverage_values else None,
            "median": statistics.median(coverage_values) if coverage_values else None,
        },
    }


def _video_error_result(args, vid: str, target, exc: Exception) -> dict:
    message = f"video processing failed: {type(exc).__name__}: {exc}"
    record = {
        "video_id": str(vid),
        "targetRatioWH": [int(target[0]), int(target[1])],
        "predictions": [],
    }
    raw = {
        "video_id": str(vid),
        "temporal_mode_requested": args.temporal_mode,
        "temporal_source_used": "video_error",
        "source": "video_error",
        "valid": False,
        "validation_errors": [message],
        "errors": [message],
        "vlm_attempts": 0,
        "attempts": 0,
        "fallback_used": None,
        "coverage_ratio": 0.0,
        "vlm_raw_responses": [],
        "segments_before_normalization": [],
        "segments_after_normalization": [],
        "corrections": [],
        "window_records": [],
        "score_curve": [],
        "selected_peaks": [],
        "score_diagnostics": {"video_error": message},
        "temporal_segments_sec": [],
        "temporal_segments_frame": [],
        "subject_hint": None,
        "temporal_raw": None,
        "spatial": [],
        "spatial_summary": {},
        "num_predictions": 0,
    }
    return {"record": record, "raw": raw}


def _atomic_write_jsonl(records: List[dict], path: str) -> None:
    partial = path + ".partial"
    write_jsonl(records, partial)
    os.replace(partial, path)


def _atomic_write_json(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(partial, path)


def _checkpoint_run(args, records: List[dict], raw_records: List[dict], total_predictions: int) -> None:
    """Atomically persist completed videos so an interruption does not erase them."""

    _atomic_write_jsonl(records, args.out)
    if args.save_raw:
        _atomic_write_jsonl(raw_records, args.out + ".raw.jsonl")
        _atomic_write_json(
            _build_summary(args, records, raw_records, total_predictions),
            args.out + ".summary.json",
        )


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    index = load_index(args.index)
    if args.num_videos > 0:
        index = index[: args.num_videos]

    print(f"videos: {len(index)}")
    _prepare_window_score_runtime(args)
    args.temporal_detector = build_temporal_detector(args)
    args.spatial_detector = build_spatial_detector(args)

    records: List[dict] = []
    raw_records: List[dict] = []
    total_predictions = 0
    try:
        for i, (vid, target) in enumerate(index, 1):
            try:
                result = process_video(args, vid, target)
            except Exception as exc:
                result = _video_error_result(args, vid, target, exc)
                print(
                    f"[{i}/{len(index)}] video={vid} isolated error: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            record = result["record"]
            raw = result["raw"]
            records.append(record)
            raw_records.append(raw)
            total_predictions += len(record["predictions"])
            _checkpoint_run(args, records, raw_records, total_predictions)
            print(
                f"[{i}/{len(index)}] video={vid} target={record['targetRatioWH']} "
                f"source={raw.get('source', 'unknown')} fallback={raw.get('fallback_used')} "
                f"attempts={raw.get('attempts', 0)} coverage={float(raw.get('coverage_ratio') or 0.0):.3f} "
                f"segments={raw.get('temporal_segments_frame', [])} preds={len(record['predictions'])}"
            )
    finally:
        close_temporal = getattr(args.temporal_detector, "close", None)
        if callable(close_temporal):
            close_temporal()

    _checkpoint_run(args, records, raw_records, total_predictions)
    print(f"wrote {len(records)} lines / {total_predictions} frame predictions -> {args.out}")

    summary = _build_summary(args, records, raw_records, total_predictions)
    print(
        "summary: "
        f"empty={summary['empty_prediction_videos']} "
        f"invalid={summary['invalid_temporal_results']} "
        f"fallbacks={sum(summary['fallback_counts'].values())} "
        f"sources={summary['temporal_source_counts']}"
    )

    if args.save_raw:
        raw_path = args.out + ".raw.jsonl"
        print(f"raw debug -> {raw_path}")
        summary_path = args.out + ".summary.json"
        print(f"run summary -> {summary_path}")

    if args.validate:
        ok, errors = validate_submission_for_items(args.out, index, args.video_dir)
        if ok:
            print("validation: OK")
        else:
            print("validation: FAILED")
            for error in errors[:50]:
                print("  -", error)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
