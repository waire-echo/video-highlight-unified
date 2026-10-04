#!/usr/bin/env python3
"""Run only the temporal half of the video-highlight pipeline.

This entry point deliberately never imports or constructs a spatial detector.
It emits a resumable ``temporal-handoff/v1`` JSONL file whose closed frame
ranges can be consumed by the teammate-owned YOLO/cropping stage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import ntpath
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from package_temporal_hints import package_handoff_records
from video_highlight.io_utils import load_index, video_meta, video_path_for
from video_highlight.temporal import build_temporal_detector
from video_highlight.prompts import bundle_hashes
from video_highlight.temporal_handoff_validation import (
    SCHEMA_VERSION,
    validate_temporal_handoff_records,
)


STATE_SCHEMA_VERSION = "temporal-run-state/v1"
RAW_SCHEMA_VERSION = "temporal-raw/v1"
IMPLEMENTATION_VERSION = "temporal-frontend/1"
DEFAULT_MODEL_PATH = r"E:\AI\models\Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_ROOT = r"E:\AI\tmp\video_highlight\window_score"
DEFAULT_LOG_ROOT = r"E:\AI\logs\video_highlight"
DEFAULT_PGL_CHECKPOINT = (
    r"E:\AI\models\mrhisum_pgl_sum\checkpoints\pgl_sum1_best_f1.pkl"
)
DEFAULT_PGL_FEATURE_MODEL_DIR = r"E:\AI\models\youtube8m"
DEFAULT_PGL_FEATURE_CACHE_DIR = r"E:\AI\cache\youtube8m_features"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporal-only high-light localization; no YOLO or crop stage is loaded."
    )
    parser.add_argument("--config", default="", help="Optional JSON defaults file")
    parser.add_argument("--index", default="", help="Index JSON/JSONL or per-video JSONL directory")
    parser.add_argument("--video-dir", default="", help="Directory containing <video_id>.mp4")
    parser.add_argument("--out", default="", help="Output temporal_handoff.jsonl")
    parser.add_argument(
        "--zip-out",
        default="",
        help="Legacy hints ZIP; defaults to temporal_hints.zip beside --out",
    )
    parser.add_argument(
        "--video-ids",
        default="",
        help="One comma-separated value, for example 76,100,107,121,154",
    )
    parser.add_argument("--num-videos", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a new artifact set even when checkpoint files already exist",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess invalid rows from a compatible checkpoint",
    )
    parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--temporal-mode",
        choices=["motion", "middle", "full", "vlm", "window_score", "pgl_sum"],
        default="pgl_sum",
    )
    parser.add_argument("--temporal-model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--detect-fps", type=float, default=1.0)
    parser.add_argument("--top-k-segments", type=int, default=1)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--min-segment-sec", type=float, default=4.0)
    parser.add_argument("--motion-min-segment-sec", type=float, default=1.0)
    parser.add_argument("--padding-sec", type=float, default=0.25)
    parser.add_argument("--segment-padding-sec", type=float, default=3.0)
    parser.add_argument("--vlm-retries", type=int, default=1)
    parser.add_argument(
        "--vlm-load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--vlm-max-video-frames", type=int, default=64)
    parser.add_argument("--vlm-max-frame-pixels", type=int, default=360 * 420)
    parser.add_argument("--vlm-total-video-pixels", type=int, default=4096 * 28 * 28)
    parser.add_argument(
        "--temporal-fallback",
        choices=["motion", "full", "middle", "none"],
        default="motion",
    )
    parser.add_argument("--short-video-full-sec", type=float, default=15.0)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.30)
    parser.add_argument("--max-coverage-sec", type=float, default=30.0)
    parser.add_argument("--max-temporal-segments", type=int, default=3)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--pgl-checkpoint-path", default=DEFAULT_PGL_CHECKPOINT)
    parser.add_argument(
        "--pgl-feature-model-dir", default=DEFAULT_PGL_FEATURE_MODEL_DIR
    )
    parser.add_argument(
        "--pgl-feature-cache-dir", default=DEFAULT_PGL_FEATURE_CACHE_DIR
    )
    parser.add_argument("--pgl-sample-fps", type=float, default=1.0)
    parser.add_argument("--pgl-boundary-grid-sec", type=float, default=0.25)
    parser.add_argument("--pgl-device", default="auto")
    parser.add_argument(
        "--pgl-hints-jsonl",
        default="",
        help="Optional cached Qwen handoff used only for subject/action text, never timing",
    )

    parser.add_argument(
        "--window-semantic-mode",
        choices=["structured", "global_event", "reze_yesno"],
        default="structured",
    )
    parser.add_argument("--window-aggregation", choices=["current", "reze_otsu"], default="current")
    parser.add_argument("--window-coarse-sec", type=float, default=2.0)
    parser.add_argument("--window-coarse-stride-sec", type=float, default=1.0)
    parser.add_argument("--window-coarse-fps", type=float, default=2.0)
    parser.add_argument("--window-fine-sec", type=float, default=2.0)
    parser.add_argument("--window-fine-stride-sec", type=float, default=0.5)
    parser.add_argument("--window-fine-fps", type=float, default=4.0)
    parser.add_argument("--window-max-fine-regions", type=int, default=6)
    parser.add_argument("--window-score-retries", type=int, default=1)
    parser.add_argument("--window-max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=1,
        help="Number of video windows scored in one GPU generation batch",
    )
    parser.add_argument(
        "--window-attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument(
        "--window-compact-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a fixed-order compact JSON array to reduce generation tokens",
    )
    parser.add_argument("--window-curve-step-sec", type=float, default=0.5)
    parser.add_argument("--window-smoothing-points", type=int, default=3)
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
    parser.add_argument("--window-cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--window-log-root", default=DEFAULT_LOG_ROOT)
    return parser


def _load_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> dict:
    if not config_path:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        parser.error(f"config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON config {config_path}: {exc}")
    if not isinstance(config, dict):
        parser.error(f"config must contain a JSON object: {config_path}")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    normalized = {}
    for key, value in config.items():
        if not isinstance(key, str) or not key.isidentifier():
            parser.error(f"invalid config key: {key!r}")
        action = actions.get(key)
        # The shared default config also contains teammate-owned spatial keys.
        # Keeping unknown values on the Namespace makes future temporal builder
        # options forward compatible, while this runner never reads those keys.
        if action is None:
            normalized[key] = value
            continue
        if value is None and action.default is None:
            normalized[key] = None
            continue
        if isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse.BooleanOptionalAction,
            ),
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
                f"choose from {', '.join(map(str, action.choices))}"
            )
        normalized[key] = converted
    return normalized


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = _build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_args, _unknown = config_parser.parse_known_args(argv)
    parser.set_defaults(**_load_config_defaults(parser, config_args.config))
    args = parser.parse_args(argv)
    for required_name in ("index", "video_dir", "out"):
        if not getattr(args, required_name):
            parser.error(f"--{required_name.replace('_', '-')} is required")
    if args.overwrite and args.retry_errors:
        parser.error("--overwrite and --retry-errors are mutually exclusive")
    if args.num_videos < 0:
        parser.error("--num-videos must be non-negative")
    if (
        args.window_semantic_mode in {"structured", "global_event"}
        and args.window_aggregation != "current"
    ):
        parser.error(
            "structured/global_event semantic modes require --window-aggregation current"
        )
    return args


def parse_video_ids(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    tokens = text.split(",")
    if any(not token.strip() for token in tokens):
        raise ValueError("--video-ids must be one comma-separated list without empty IDs")
    result = []
    seen = set()
    for token in tokens:
        video_id = token.strip()
        if video_id not in seen:
            result.append(video_id)
            seen.add(video_id)
    return result


def _require_e_absolute(path: str, label: str, *, must_exist: bool = False) -> str:
    raw = os.path.expanduser(str(path))
    if os.name == "nt":
        drive, _tail = ntpath.splitdrive(raw)
        if drive.upper() != "E:" or not ntpath.isabs(raw):
            raise ValueError(f"{label} must be an absolute E: path: {path}")
    absolute = os.path.abspath(raw)
    if must_exist and not os.path.exists(absolute):
        raise ValueError(f"{label} does not exist: {path}")
    return absolute


def _prepare_runtime(args) -> None:
    args.out = os.path.abspath(os.path.expanduser(str(args.out)))
    if not args.zip_out:
        args.zip_out = str(Path(args.out).with_name("temporal_hints.zip"))
    args.zip_out = os.path.abspath(os.path.expanduser(str(args.zip_out)))
    if os.environ.get('VIDEO_HIGHLIGHT_PORTABLE_RUNTIME') == '1':
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)
        return
    guarded_temporal_runtime = (
        args.temporal_mode in {"vlm", "window_score", "pgl_sum"} and os.name == "nt"
    )
    if not guarded_temporal_runtime:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)
        return
    if args.temporal_mode in {"vlm", "window_score"}:
        args.temporal_model_path = _require_e_absolute(
            args.temporal_model_path, "temporal model path", must_exist=True
        )
    if args.temporal_mode == "pgl_sum":
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
    args.window_cache_root = _require_e_absolute(
        args.window_cache_root, "window cache root"
    )
    args.window_log_root = _require_e_absolute(args.window_log_root, "window log root")
    args.out = _require_e_absolute(args.out, "handoff output")
    args.zip_out = _require_e_absolute(args.zip_out, "legacy ZIP output")
    # Validate every user-controlled path before creating anything.  A typo
    # pointing at C: must fail without leaving even an empty output directory.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)

    ai_root = r"E:\AI"
    hf_root = os.path.join(ai_root, "cache", "huggingface")
    temp_root = os.path.join(ai_root, "tmp", "video_highlight")
    runtime_dirs = {
        "HF_HOME": hf_root,
        "HF_HUB_CACHE": os.path.join(hf_root, "hub"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(hf_root, "hub"),
        "HF_ASSETS_CACHE": os.path.join(hf_root, "assets"),
        "HF_DATASETS_CACHE": os.path.join(hf_root, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(hf_root, "transformers"),
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
        "VIDEO_HIGHLIGHT_LOG_DIR": args.window_log_root,
    }
    for key, directory in runtime_dirs.items():
        os.makedirs(directory, exist_ok=True)
        os.environ[key] = directory
    # Environment variables are normally read at interpreter startup.  Set the
    # live process knobs as well so late model imports cannot fall back to a
    # user-profile cache on C:.
    sys.pycache_prefix = runtime_dirs["PYTHONPYCACHEPREFIX"]
    tempfile.tempdir = temp_root
    os.makedirs(args.window_cache_root, exist_ok=True)
    os.makedirs(args.window_log_root, exist_ok=True)
    if args.temporal_mode == "pgl_sum":
        os.makedirs(args.pgl_feature_cache_dir, exist_ok=True)
    os.environ["VIDEO_HIGHLIGHT_SET_TEMP"] = "1"


def _artifact_paths(args) -> dict[str, str]:
    return {
        "handoff": args.out,
        "raw": args.out + ".raw.jsonl",
        "summary": args.out + ".summary.json",
        "state": args.out + ".state.json",
        "zip": args.zip_out,
    }


def _json_ready(value):
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_ready(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def _atomic_write_jsonl(records: Sequence[dict], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        _json_ready(record),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()


def _atomic_write_json(payload: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_ready(payload),
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()


def _read_json(path: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint must contain a JSON object: {path}")
    return value


def _read_jsonl(path: str) -> list[dict]:
    records = []
    seen = set()
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read checkpoint {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"non-object record in {path} line {line_number}")
        video_id = str(record.get("video_id", "")).strip()
        if not video_id or video_id in seen:
            raise RuntimeError(f"missing or duplicate video_id in {path} line {line_number}")
        records.append(record)
        seen.add(video_id)
    return records


def _model_identity(path: str) -> dict:
    model_path = Path(path).expanduser()
    identity = {"path": str(model_path.resolve()) if model_path.exists() else str(model_path)}
    if not model_path.exists():
        identity["exists"] = False
        return identity
    identity["exists"] = True
    if model_path.is_file():
        stat = model_path.stat()
        identity["files"] = [[model_path.name, stat.st_size, stat.st_mtime_ns]]
        return identity
    files = []
    for item in sorted((entry for entry in model_path.rglob("*") if entry.is_file())):
        stat = item.stat()
        files.append([item.relative_to(model_path).as_posix(), stat.st_size, stat.st_mtime_ns])
    identity["files"] = files
    return identity


def _implementation_hash() -> str:
    hasher = hashlib.sha256()
    candidates = [
        Path(__file__),
        SRC_DIR / "video_highlight" / "temporal.py",
        SRC_DIR / "video_highlight" / "pgl_sum_model.py",
        SRC_DIR / "video_highlight" / "pgl_sum_frontend.py",
        SRC_DIR / "video_highlight" / "window_score_frontend.py",
        SRC_DIR / "video_highlight" / "window_scoring.py",
        SRC_DIR / "video_highlight" / "temporal_signals.py",
        SRC_DIR / "video_highlight" / "score_curve.py",
        SRC_DIR / "video_highlight" / "windowing.py",
        SRC_DIR / "video_highlight" / "reze_scoring.py",
        SRC_DIR / "video_highlight" / "reze_aggregation.py",
        SRC_DIR / "video_highlight" / "reze_frontend.py",
        SRC_DIR / "video_highlight" / "io_utils.py",
        SRC_DIR / "video_highlight" / "temporal_handoff_validation.py",
        SCRIPT_DIR / "package_temporal_hints.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _package_versions() -> dict[str, Optional[str]]:
    versions = {}
    for package in (
        "torch",
        "transformers",
        "qwen-vl-utils",
        "opencv-python",
        "numpy",
        "bitsandbytes",
        "torchcodec",
        "accelerate",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _temporal_parameters(args) -> dict:
    explicit = {
        "temporal_mode",
        "temporal_model_path",
        "detect_fps",
        "top_k_segments",
        "window_sec",
        "min_segment_sec",
        "motion_min_segment_sec",
        "padding_sec",
        "segment_padding_sec",
        "short_video_full_sec",
        "min_coverage_ratio",
        "max_coverage_sec",
        "max_temporal_segments",
        "device_map",
        "dtype",
        "max_new_tokens",
    }
    result = {}
    for key, value in vars(args).items():
        if key in explicit or key.startswith(("temporal_", "window_", "vlm_", "pgl_")):
            result[key] = _json_ready(value)
    return dict(sorted(result.items()))


def _build_fingerprint(args, index_items) -> tuple[str, dict]:
    model_identity = None
    if args.temporal_mode in {"vlm", "window_score"}:
        model_identity = _model_identity(args.temporal_model_path)
    elif args.temporal_mode == "pgl_sum":
        model_identity = {
            "checkpoint": _model_identity(args.pgl_checkpoint_path),
            "feature_model": _model_identity(args.pgl_feature_model_dir),
            "semantic_hints": (
                _model_identity(args.pgl_hints_jsonl)
                if args.pgl_hints_jsonl
                else None
            ),
        }
    payload = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_sha256": _implementation_hash(),
        "prompt_hashes": bundle_hashes(json.loads(os.environ.get('VIDEO_HIGHLIGHT_PROMPT_OVERRIDES', '{}'))),
        "parameters": _temporal_parameters(args),
        "model": model_identity,
        "package_versions": _package_versions(),
        "input": {
            "index": [[str(video_id), [int(target[0]), int(target[1])]] for video_id, target in index_items],
            "video_dir": os.path.abspath(args.video_dir),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), payload


def _load_existing(args, paths, fingerprint: str, index_items):
    artifacts_exist = {name: os.path.exists(path) for name, path in paths.items()}
    if args.overwrite:
        return {}, {}
    if not any(artifacts_exist.values()):
        return {}, {}
    if not args.resume:
        raise RuntimeError("output artifacts already exist; use --resume or --overwrite")
    for required in ("handoff", "raw", "state"):
        if not artifacts_exist[required]:
            raise RuntimeError(
                f"incomplete checkpoint ({required} is missing); use --overwrite to start clean"
            )
    state = _read_json(paths["state"])
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError("checkpoint state schema is incompatible; use --overwrite")
    if state.get("run_fingerprint") != fingerprint:
        raise RuntimeError(
            "checkpoint fingerprint conflicts with the current model, parameters, code, or index; "
            "use the original settings or --overwrite"
        )
    handoff_records = _read_jsonl(paths["handoff"])
    raw_records = _read_jsonl(paths["raw"])
    handoff_by_id = {str(record["video_id"]): record for record in handoff_records}
    raw_by_id = {str(record["video_id"]): record for record in raw_records}
    state_ids = {str(item) for item in state.get("completed_video_ids", [])}
    common_ids = set(handoff_by_id) & set(raw_by_id)
    if state_ids - common_ids:
        raise RuntimeError("state claims videos missing from handoff/raw; use --overwrite")
    if set(handoff_by_id) != set(raw_by_id):
        # Each file is atomically replaced, and state is committed last.  A
        # power loss between the handoff and raw replacements can therefore
        # leave one uncommitted extra row.  Roll both views back to the IDs in
        # the last committed state instead of forcing a full batch restart.
        handoff_by_id = {
            video_id: record
            for video_id, record in handoff_by_id.items()
            if video_id in state_ids
        }
        raw_by_id = {
            video_id: record
            for video_id, record in raw_by_id.items()
            if video_id in state_ids
        }
    index_ids = {str(video_id) for video_id, _target in index_items}
    unknown = sorted(set(handoff_by_id) - index_ids)
    if unknown:
        raise RuntimeError(f"checkpoint contains video IDs outside the current index: {unknown}")
    ok, errors = validate_temporal_handoff_records(list(handoff_by_id.values()))
    if not ok:
        raise RuntimeError("existing handoff is invalid: " + "; ".join(errors[:5]))
    return handoff_by_id, raw_by_id


def _normalise_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_segments(value, duration_sec: float, fps: float, frame_count: int):
    seconds = []
    for item in value or []:
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
        if end > start + 1e-9:
            seconds.append([start, end])
    seconds.sort()
    merged_seconds = []
    for start, end in seconds:
        if merged_seconds and start <= merged_seconds[-1][1] + 1e-9:
            merged_seconds[-1][1] = max(merged_seconds[-1][1], end)
        else:
            merged_seconds.append([start, end])

    paired = []
    for start, end in merged_seconds:
        start_frame = max(0, min(int(round(start * fps)), frame_count - 1))
        end_frame = max(0, min(int(round(end * fps)), frame_count - 1))
        if paired and start_frame <= paired[-1][1][1] + 1:
            paired[-1][0][1] = max(paired[-1][0][1], end)
            paired[-1][1][1] = max(paired[-1][1][1], end_frame)
        else:
            paired.append([[start, end], [start_frame, end_frame]])
    return [item[0] for item in paired], [item[1] for item in paired]


def _window_semantics(result, segments_sec):
    diagnostics = getattr(result, "score_diagnostics", None) or {}
    if isinstance(diagnostics, dict) and diagnostics.get("qwen_degenerate"):
        return None, None
    subject_hint = _normalise_text(getattr(result, "subject_hint", None))
    action_description = _normalise_text(getattr(result, "action_description", None))
    candidates = []
    for row in getattr(result, "window_records", None) or []:
        if not isinstance(row, dict) or not row.get("valid"):
            continue
        try:
            start, end = float(row["start_sec"]), float(row["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not any(min(end, right) > max(start, left) for left, right in segments_sec):
            continue
        score = -math.inf
        for key in ("final_score", "fused_score", "qwen_score"):
            try:
                candidate = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                score = candidate
                break
        candidates.append(
            (
                score,
                _normalise_text(row.get("main_subject")),
                _normalise_text(row.get("action_description")),
            )
        )
    if candidates:
        _score, candidate_subject, candidate_action = max(candidates, key=lambda item: item[0])
        subject_hint = subject_hint or candidate_subject
        action_description = action_description or candidate_action
    return subject_hint, action_description


def _error_records(args, video_id: str, target, message: str, *, meta=None):
    frame_count = fps = duration_sec = None
    if meta:
        candidate_frames, candidate_fps = int(meta[0]), float(meta[1])
        if candidate_frames > 0 and candidate_fps > 0:
            frame_count = candidate_frames
            fps = candidate_fps
            duration_sec = frame_count / fps
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "video_id": str(video_id),
        "targetRatioWH": [int(target[0]), int(target[1])],
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "segments_sec": [],
        "segments_frame": [],
        "subject_hint": None,
        "action_description": None,
        "source": "input_error" if message.startswith("missing video") else "video_error",
        "valid": False,
        "fallback_used": None,
        "errors": [message],
        "warnings": [],
        "raw_diagnostics_path": os.path.abspath(args.out + ".raw.jsonl"),
    }
    raw = {
        "schema_version": RAW_SCHEMA_VERSION,
        "video_id": str(video_id),
        "targetRatioWH": [int(target[0]), int(target[1])],
        "temporal_mode_requested": args.temporal_mode,
        "source": handoff["source"],
        "valid": False,
        "fallback_used": None,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "temporal_segments_sec": [],
        "temporal_segments_frame": [],
        "subject_hint": None,
        "action_description": None,
        "attempts": 0,
        "errors": [message],
        "warnings": [],
        "window_records": [],
        "score_curve": [],
        "selected_peaks": [],
        "score_diagnostics": {"video_error": message},
    }
    return handoff, raw


def process_temporal_video(args, detector, video_id: str, target):
    video_path = video_path_for(args.video_dir, video_id)
    if not os.path.isfile(video_path):
        return _error_records(args, video_id, target, f"missing video: {video_path}")
    meta = video_meta(video_path)
    frame_count, fps, width, height = meta
    if frame_count <= 0 or fps <= 0:
        return _error_records(args, video_id, target, "unreadable video metadata", meta=meta)
    duration_sec = frame_count / float(fps)
    result = detector.detect(video_path)
    detected_segments = getattr(result, "segments_sec", []) or []
    segments_sec, segments_frame = _normalise_segments(
        detected_segments, duration_sec, float(fps), int(frame_count)
    )
    errors = [str(item) for item in (getattr(result, "errors", None) or [])]
    result_valid = bool(getattr(result, "valid", True))
    valid = bool(result_valid and segments_sec)
    if not segments_sec:
        errors.append("detector returned no valid temporal segments")
    if not result_valid:
        errors.append("detector marked the temporal result invalid")
    errors = list(dict.fromkeys(errors))
    subject_hint, action_description = _window_semantics(result, segments_sec)
    if not valid:
        segments_sec = []
        segments_frame = []
        subject_hint = None
        action_description = None
    source = _normalise_text(getattr(result, "source", None)) or args.temporal_mode
    fallback_used = _normalise_text(getattr(result, "fallback_used", None))
    handoff_errors = [] if valid else errors
    recovered_warnings = errors if valid else []

    handoff = {
        "schema_version": SCHEMA_VERSION,
        "video_id": str(video_id),
        "targetRatioWH": [int(target[0]), int(target[1])],
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration_sec": float(duration_sec),
        "segments_sec": segments_sec,
        "segments_frame": segments_frame,
        "subject_hint": subject_hint,
        "action_description": action_description,
        "source": source,
        "valid": valid,
        "fallback_used": fallback_used,
        "errors": handoff_errors,
        "warnings": recovered_warnings,
        "raw_diagnostics_path": os.path.abspath(args.out + ".raw.jsonl"),
    }
    raw = {
        "schema_version": RAW_SCHEMA_VERSION,
        "video_id": str(video_id),
        "video_path": os.path.abspath(video_path),
        "targetRatioWH": [int(target[0]), int(target[1])],
        "temporal_mode_requested": args.temporal_mode,
        "source": source,
        "valid": valid,
        "fallback_used": fallback_used,
        "fps": float(fps),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "duration_sec": float(duration_sec),
        "detected_segments_sec": _json_ready(detected_segments),
        "temporal_segments_sec": segments_sec,
        "temporal_segments_frame": segments_frame,
        "subject_hint": subject_hint,
        "action_description": action_description,
        "attempts": int(getattr(result, "attempts", 0) or 0),
        "errors": errors,
        "warnings": recovered_warnings,
        "corrections": _json_ready(getattr(result, "corrections", None) or []),
        "vlm_raw_responses": _json_ready(
            getattr(result, "vlm_raw_responses", None) or []
        ),
        "temporal_raw": _json_ready(getattr(result, "raw", None)),
        "window_records": _json_ready(getattr(result, "window_records", None) or []),
        "score_curve": _json_ready(getattr(result, "score_curve", None) or []),
        "selected_peaks": _json_ready(getattr(result, "selected_peaks", None) or []),
        "score_diagnostics": _json_ready(
            getattr(result, "score_diagnostics", None) or {}
        ),
    }
    return handoff, raw


def _ordered_records(records_by_id: dict[str, dict], index_items) -> list[dict]:
    order = {str(video_id): index for index, (video_id, _target) in enumerate(index_items)}
    return [
        record
        for _video_id, record in sorted(
            records_by_id.items(), key=lambda item: order.get(item[0], len(order))
        )
    ]


def _build_summary(args, records, raw_records, fingerprint: str, requested_ids, zip_summary):
    valid_records = [record for record in records if record.get("valid")]
    invalid_records = [record for record in records if not record.get("valid")]
    coverage_seconds = sum(
        float(end) - float(start)
        for record in valid_records
        for start, end in record.get("segments_sec", [])
    )
    window_records = [
        item
        for raw in raw_records
        for item in raw.get("window_records", [])
        if isinstance(item, dict)
    ]
    return {
        "schema_version": "temporal-run-summary/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": fingerprint,
        "temporal_mode": args.temporal_mode,
        "semantic_mode": getattr(args, "window_semantic_mode", "structured"),
        "aggregation": getattr(args, "window_aggregation", "current"),
        "output": os.path.abspath(args.out),
        "raw_output": os.path.abspath(args.out + ".raw.jsonl"),
        "state_output": os.path.abspath(args.out + ".state.json"),
        "legacy_zip": os.path.abspath(args.zip_out),
        "requested_video_ids": list(requested_ids),
        "completed_video_ids": [str(record["video_id"]) for record in records],
        "videos_written": len(records),
        "valid_videos": len(valid_records),
        "invalid_videos": len(invalid_records),
        "invalid_video_ids": [str(record["video_id"]) for record in invalid_records],
        "segments_total": sum(len(record.get("segments_sec", [])) for record in valid_records),
        "selected_seconds_total": coverage_seconds,
        "source_counts": dict(sorted(Counter(str(record.get("source")) for record in records).items())),
        "fallback_counts": dict(
            sorted(
                Counter(
                    str(record.get("fallback_used"))
                    for record in records
                    if record.get("fallback_used")
                ).items()
            )
        ),
        "window_count": len(window_records),
        "window_retry_count": sum(
            max(0, int(item.get("attempts", 0) or 0) - 1) for item in window_records
        ),
        "legacy_zip_summary": zip_summary,
        "contains_spatial_output": False,
    }


def _checkpoint(
    args,
    paths,
    handoff_by_id,
    raw_by_id,
    index_items,
    fingerprint: str,
    fingerprint_payload: dict,
    requested_ids,
):
    handoff_records = _ordered_records(handoff_by_id, index_items)
    raw_records = _ordered_records(raw_by_id, index_items)
    ok, errors = validate_temporal_handoff_records(handoff_records)
    if not ok:
        raise RuntimeError("refusing to checkpoint invalid handoff: " + "; ".join(errors[:10]))
    _atomic_write_jsonl(handoff_records, paths["handoff"])
    _atomic_write_jsonl(raw_records, paths["raw"])
    zip_summary = package_handoff_records(handoff_records, Path(paths["zip"]))
    summary = _build_summary(
        args, handoff_records, raw_records, fingerprint, requested_ids, zip_summary
    )
    _atomic_write_json(summary, paths["summary"])
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "completed_video_ids": [str(record["video_id"]) for record in handoff_records],
        "valid_video_ids": [str(record["video_id"]) for record in handoff_records if record.get("valid")],
        "error_video_ids": [str(record["video_id"]) for record in handoff_records if not record.get("valid")],
        "artifacts": {key: os.path.abspath(value) for key, value in paths.items()},
    }
    # State is replaced last.  If interruption occurs earlier, the output/raw
    # superset remains safe and will be accepted on the next compatible resume.
    _atomic_write_json(state, paths["state"])
    return summary


def run(args) -> dict:
    _prepare_runtime(args)
    if not os.path.exists(args.index):
        raise ValueError(f"index does not exist: {args.index}")
    if not os.path.isdir(args.video_dir):
        raise ValueError(f"video directory does not exist: {args.video_dir}")
    index_items = load_index(args.index)
    index_ids = [str(video_id) for video_id, _target in index_items]
    if len(index_ids) != len(set(index_ids)):
        raise ValueError("input index contains duplicate video IDs")
    requested_ids = parse_video_ids(args.video_ids)
    if requested_ids:
        unknown = [video_id for video_id in requested_ids if video_id not in set(index_ids)]
        if unknown:
            raise ValueError(f"--video-ids not present in index: {', '.join(unknown)}")
    else:
        requested_ids = list(index_ids)
    if args.num_videos:
        requested_ids = requested_ids[: args.num_videos]
    index_by_id = {str(video_id): (str(video_id), target) for video_id, target in index_items}
    # Honour the explicit comma-list order.  This also makes sequential logs
    # and checkpoint progress deterministic for callers that choose a custom
    # processing order.
    selected_items = [index_by_id[video_id] for video_id in requested_ids]

    paths = _artifact_paths(args)
    fingerprint, fingerprint_payload = _build_fingerprint(args, index_items)
    handoff_by_id, raw_by_id = _load_existing(
        args, paths, fingerprint, index_items
    )
    pending_items = []
    for video_id, target in selected_items:
        video_id = str(video_id)
        existing = handoff_by_id.get(video_id)
        if existing is None or (args.retry_errors and not existing.get("valid")):
            pending_items.append((video_id, target))

    print(
        f"temporal-only videos requested={len(selected_items)} "
        f"pending={len(pending_items)} resumed={len(selected_items) - len(pending_items)}"
    )
    if args.overwrite and pending_items:
        # Commit the new fingerprint before loading a large model.  If model
        # construction itself is interrupted, the old incompatible artifact
        # set cannot be mistaken for this run and a normal resume can retry.
        _checkpoint(
            args,
            paths,
            handoff_by_id,
            raw_by_id,
            index_items,
            fingerprint,
            fingerprint_payload,
            requested_ids,
        )
    detector = None
    try:
        if pending_items:
            detector = build_temporal_detector(args)
            for index, (video_id, target) in enumerate(pending_items, 1):
                try:
                    handoff, raw = process_temporal_video(
                        args, detector, video_id, target
                    )
                except Exception as exc:
                    message = f"video processing failed: {type(exc).__name__}: {exc}"
                    handoff, raw = _error_records(args, video_id, target, message)
                    print(
                        f"[{index}/{len(pending_items)}] video={video_id} isolated error: {message}",
                        file=sys.stderr,
                    )
                handoff_by_id[video_id] = handoff
                raw_by_id[video_id] = raw
                summary = _checkpoint(
                    args,
                    paths,
                    handoff_by_id,
                    raw_by_id,
                    index_items,
                    fingerprint,
                    fingerprint_payload,
                    requested_ids,
                )
                print(
                    f"[{index}/{len(pending_items)}] video={video_id} "
                    f"valid={handoff['valid']} source={handoff['source']} "
                    f"segments={handoff['segments_frame']}"
                )
        else:
            # This path intentionally runs before detector construction.
            summary = _checkpoint(
                args,
                paths,
                handoff_by_id,
                raw_by_id,
                index_items,
                fingerprint,
                fingerprint_payload,
                requested_ids,
            )
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()

    missing_requested = [video_id for video_id in requested_ids if video_id not in handoff_by_id]
    if missing_requested:
        raise RuntimeError(f"requested videos missing after run: {', '.join(missing_requested)}")
    if args.validate:
        records = _ordered_records(handoff_by_id, index_items)
        ok, errors = validate_temporal_handoff_records(records)
        if not ok:
            raise RuntimeError("temporal handoff validation failed: " + "; ".join(errors[:10]))
        print("temporal handoff validation: OK")
    print(
        f"wrote temporal handoff videos={summary['videos_written']} "
        f"valid={summary['valid_videos']} invalid={summary['invalid_videos']} -> {args.out}"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "parse_args",
    "parse_video_ids",
    "process_temporal_video",
    "run",
]
