"""Qwen 3B per-window video scoring for the experimental temporal front end.

This module is intentionally independent from :mod:`temporal`.  The legacy
detectors return timestamp intervals directly; this front end first produces
structured, auditable window records.  A later signal/fusion stage can turn
those records into ``TemporalResult`` without changing the old code path.

The model is asked to score *only the supplied time range*.  The local-video
request therefore contains ``video_start``, ``video_end`` and ``fps`` instead
of asking the model to guess timestamps from a list of images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

from .windowing import WindowSpec, generate_windows, probe_video, validate_window


QWEN_SCORE_KEYS: Tuple[str, ...] = (
    "action",
    "completion",
    "interaction",
    "event",
    "emotion",
    "visual_impact",
    "highlight",
)

# The initial weights are part of the experiment contract documented in the
# project handoff.  Keep them in one public constant so fusion/evaluation code
# can report the exact formula instead of duplicating magic numbers.
QWEN_SCORE_WEIGHTS: Dict[str, float] = {
    "action": 0.15,
    "completion": 0.15,
    "interaction": 0.15,
    "event": 0.15,
    "emotion": 0.10,
    "visual_impact": 0.10,
    "highlight": 0.20,
}

WINDOW_RESPONSE_KEYS: Tuple[str, ...] = QWEN_SCORE_KEYS + (
    "main_subject",
    "action_description",
    "evidence",
)

DEFAULT_MODEL_PATH = r"E:\AI\models\Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR = r"E:\AI\cache\video_highlight"
DEFAULT_TEMP_DIR = r"E:\AI\tmp\video_highlight"


def _finite_float(value: Any, name: str, *, non_negative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (result < 0 if non_negative else result <= 0):
        qualifier = "non-negative" if non_negative else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return result


def _strict_json_object(text: str) -> Mapping[str, Any]:
    """Parse one complete JSON object and reject wrappers/duplicate keys."""

    if not isinstance(text, str):
        raise ValueError("response must be text")
    payload = text.strip()
    if not payload:
        raise ValueError("response is empty")
    # Qwen commonly wraps an otherwise complete JSON answer in a markdown
    # fence.  Accept exactly one complete fence, but continue rejecting a
    # prefix/suffix, prose around the object, or a truncated fence.
    if payload.startswith("```") or payload.endswith("```"):
        import re

        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", payload, re.I | re.S)
        if not match:
            raise ValueError("incomplete markdown JSON fence")
        payload = match.group(1).strip()

    duplicate_keys: List[str] = []

    def pairs_hook(pairs):
        obj: Dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                duplicate_keys.append(str(key))
            obj[key] = value
        return obj

    try:
        obj = json.loads(payload, object_pairs_hook=pairs_hook)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid complete JSON: {exc}") from exc
    if duplicate_keys:
        raise ValueError("duplicate JSON keys: " + ", ".join(sorted(set(duplicate_keys))))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    if set(obj) != set(WINDOW_RESPONSE_KEYS):
        missing = sorted(set(WINDOW_RESPONSE_KEYS) - set(obj))
        extra = sorted(set(obj) - set(WINDOW_RESPONSE_KEYS))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("JSON schema mismatch: " + "; ".join(details))
    return obj


def _strict_compact_array(text: str) -> Mapping[str, Any]:
    """Parse the token-efficient array form into the canonical object schema."""

    if not isinstance(text, str):
        raise ValueError("response must be text")
    payload = text.strip()
    if not payload:
        raise ValueError("response is empty")
    try:
        values = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid complete JSON: {exc}") from exc
    if not isinstance(values, list) or len(values) != len(WINDOW_RESPONSE_KEYS):
        raise ValueError(
            f"compact response must contain exactly {len(WINDOW_RESPONSE_KEYS)} values"
        )
    return dict(zip(WINDOW_RESPONSE_KEYS, values))


def qwen_score_from_components(components: Mapping[str, int]) -> float:
    """Convert seven integer 0--3 ratings into a 0--1 weighted score."""

    values: Dict[str, int] = {}
    for key in QWEN_SCORE_KEYS:
        value = components.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
            raise ValueError(f"{key} must be an integer in [0, 3]")
        values[key] = value
    score = sum(QWEN_SCORE_WEIGHTS[key] * values[key] / 3.0 for key in QWEN_SCORE_KEYS)
    # Guard against accidental floating point drift in downstream thresholds.
    return max(0.0, min(1.0, float(score)))


_FORBIDDEN_DESCRIPTION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b", re.I),
    re.compile(r"\b(?:timestamp|timecode)\b", re.I),
    re.compile(r"\b(?:window|frame)\s*(?:id|number|no\.?|#)?\s*[A-Z]?\d+\b", re.I),
    re.compile(r"\b(?:crop|bbox|bounding\s+box|[xy])\s*[:=]\s*[\[\(\d]", re.I),
    re.compile(r"\d+(?:\.\d+)?\s*秒"),
    re.compile(r"(?:第\s*)?\d+\s*(?:号|个)?\s*(?:窗口|帧)"),
    re.compile(r"(?:裁剪|边界框|坐标)\s*[:：=]"),
)


def _validate_description_scope(strings: Mapping[str, str]) -> None:
    """Reject explicit time/window/crop metadata inside otherwise valid JSON."""

    joined = " ".join(strings.values())
    for pattern in _FORBIDDEN_DESCRIPTION_PATTERNS:
        if pattern.search(joined):
            raise ValueError(
                "descriptive fields must not contain timestamps, window/frame IDs, or crop coordinates"
            )


def build_window_score_prompt(
    *,
    sample_fps: float = 2.0,
    include_rubric: bool = True,
    compact_output: bool = False,
) -> str:
    from .prompts import call_builder
    return call_builder("qwen/window-v1", "build_window_score_prompt", sample_fps=sample_fps, include_rubric=include_rubric, compact_output=compact_output)


@dataclass
class WindowScore:
    """Structured result for one model-scored temporal window."""

    window_id: int
    start_sec: float
    end_sec: float
    sample_fps: float
    qwen_score: Optional[float] = None
    qwen_components: Dict[str, int] = field(default_factory=dict)
    main_subject: Optional[str] = None
    action_description: Optional[str] = None
    evidence: Optional[str] = None
    raw: str = ""
    valid: bool = False
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    stage: str = "coarse"
    parent_id: Optional[int] = None
    raw_responses: List[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))

    @property
    def score(self) -> Optional[float]:
        """Alias used by generic score-curve code."""

        return self.qwen_score

    @property
    def components(self) -> Dict[str, int]:
        return dict(self.qwen_components)

    def to_dict(self) -> dict:
        """Return a raw JSONL-friendly record, preserving missing values."""

        result: Dict[str, Any] = {
            "window_id": self.window_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "sample_fps": self.sample_fps,
            "stage": self.stage,
            "parent_id": self.parent_id,
            "qwen_score": self.qwen_score,
            "qwen_components": dict(self.qwen_components),
            "main_subject": self.main_subject,
            "action_description": self.action_description,
            "evidence": self.evidence,
            "raw": self.raw,
            "valid": bool(self.valid),
            "errors": list(self.errors),
            "attempts": int(self.attempts),
            "raw_responses": list(self.raw_responses),
        }
        # Expose the seven ratings at the top level as well.  This makes the
        # record convenient for spreadsheet/JSONL inspection and is harmless
        # for consumers that use qwen_components.
        for key in QWEN_SCORE_KEYS:
            result[key] = self.qwen_components.get(key)
        return result

    as_dict = to_dict


def _metadata_for_window(
    window: Optional[WindowSpec],
    *,
    window_id: int = 0,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    sample_fps: float = 2.0,
    stage: str = "coarse",
    parent_id: Optional[int] = None,
) -> Tuple[int, float, float, float, str, Optional[int]]:
    if window is not None:
        window = validate_window(window)
        return (
            window.window_id,
            window.start_sec,
            window.end_sec,
            window.sample_fps,
            window.stage,
            window.parent_id,
        )
    start = _finite_float(start_sec, "start_sec", non_negative=True)
    end = _finite_float(end_sec, "end_sec", non_negative=True)
    if end <= start:
        raise ValueError("end_sec must be greater than start_sec")
    fps = _finite_float(sample_fps, "sample_fps")
    return int(window_id), start, end, fps, str(stage or "coarse"), parent_id


def parse_window_score_response(
    text: str,
    window: Optional[WindowSpec] = None,
    *,
    window_id: int = 0,
    start_sec: float = 0.0,
    end_sec: float = 1.0,
    sample_fps: float = 2.0,
    stage: str = "coarse",
    parent_id: Optional[int] = None,
    attempts: int = 1,
    compact_output: bool = False,
) -> WindowScore:
    """Strictly parse one Qwen response into :class:`WindowScore`.

    No attempt is made to recover a JSON object from prose or a markdown fence.
    A malformed response returns an invalid record rather than raising, which
    lets the caller retry just that window.
    """

    try:
        metadata = _metadata_for_window(
            window,
            window_id=window_id,
            start_sec=start_sec,
            end_sec=end_sec,
            sample_fps=sample_fps,
            stage=stage,
            parent_id=parent_id,
        )
    except Exception as exc:
        # Metadata errors are programmer/input errors, not model failures.  A
        # minimal invalid object is still useful to batch diagnostics.
        return WindowScore(
            int(window_id),
            float(start_sec or 0.0),
            float(end_sec or 0.0),
            float(sample_fps or 0.0),
            raw=str(text or ""),
            valid=False,
            errors=[f"invalid window metadata: {type(exc).__name__}: {exc}"],
            attempts=max(0, int(attempts)),
        )

    wid, start, end, fps, stage_name, parent = metadata
    base = dict(
        window_id=wid,
        start_sec=start,
        end_sec=end,
        sample_fps=fps,
        raw=str(text or ""),
        raw_responses=[str(text or "")],
        attempts=max(0, int(attempts)),
        stage=stage_name,
        parent_id=parent,
    )
    try:
        payload = (
            _strict_compact_array(text)
            if compact_output
            else _strict_json_object(text)
        )
        components: Dict[str, int] = {}
        for key in QWEN_SCORE_KEYS:
            value = payload[key]
            # bool is an int subclass but is not a valid model rating.
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
                raise ValueError(f"{key} must be an integer in [0, 3]")
            components[key] = value
        strings: Dict[str, str] = {}
        for key in ("main_subject", "action_description", "evidence"):
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
            strings[key] = " ".join(value.split())
        _validate_description_scope(strings)
        return WindowScore(
            **base,
            qwen_score=qwen_score_from_components(components),
            qwen_components=components,
            main_subject=strings["main_subject"],
            action_description=strings["action_description"],
            evidence=strings["evidence"],
            valid=True,
            errors=[],
        )
    except Exception as exc:
        return WindowScore(
            **base,
            valid=False,
            errors=[f"invalid window JSON: {type(exc).__name__}: {exc}"],
        )


def _invalid_score_for_spec(spec: WindowSpec, error: str) -> WindowScore:
    return WindowScore(
        spec.window_id,
        spec.start_sec,
        spec.end_sec,
        spec.sample_fps,
        valid=False,
        errors=[str(error)],
        attempts=0,
        stage=spec.stage,
        parent_id=spec.parent_id,
    )


@dataclass(frozen=True)
class RuntimePaths:
    """Configurable model/cache/temp locations (E-drive by default)."""

    model_path: str = DEFAULT_MODEL_PATH
    cache_dir: str = DEFAULT_CACHE_DIR
    temp_dir: str = DEFAULT_TEMP_DIR

    @classmethod
    def from_environment(
        cls,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> "RuntimePaths":
        return cls(
            str(model_path or os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH)),
            str(cache_dir or os.environ.get("VIDEO_HIGHLIGHT_CACHE_DIR", DEFAULT_CACHE_DIR)),
            str(temp_dir or os.environ.get("VIDEO_HIGHLIGHT_TEMP_DIR", DEFAULT_TEMP_DIR)),
        )

    def as_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "cache_dir": self.cache_dir,
            "temp_dir": self.temp_dir,
        }


def _windows_drive(path: Union[str, os.PathLike]) -> str:
    """Get a Windows drive letter even when tests run on another OS."""

    text = os.fspath(path)
    # ntpath is deliberately used instead of Path.drive: Path on Linux treats
    # ``E:\\...`` as a relative filename.
    import ntpath

    drive, _tail = ntpath.splitdrive(text)
    if drive:
        return drive.rstrip(":").upper()
    return ""


def assert_runtime_paths(
    paths: RuntimePaths,
    *,
    require_e_drive: bool = False,
) -> RuntimePaths:
    """Validate runtime locations before a model run.

    ``require_e_drive`` is opt-in so unit tests and Linux deployments can use
    ordinary temporary directories.  Production Windows scripts can enable it
    to fail fast instead of silently filling ``C:\\Users`` caches.
    """

    for name, value in paths.as_dict().items():
        if not str(value).strip():
            raise ValueError(f"{name} cannot be empty")
        if require_e_drive:
            drive = _windows_drive(value)
            # On Windows an unqualified path would resolve against the current
            # drive (usually C:), so it is unsafe for the guarded production
            # runner.  Linux/macOS unit tests may still use synthetic
            # ``E:\\...`` strings, for which ``_windows_drive`` returns E.
            if os.name == "nt" and drive != "E":
                raise ValueError(
                    f"{name} must be on E: and absolute when require_e_drive=True: {value}"
                )
            if drive and drive != "E":
                raise ValueError(
                    f"{name} must be on E: when require_e_drive=True: {value}"
                )
    return paths


def configure_runtime_environment(
    paths: RuntimePaths,
    *,
    create: bool = False,
    require_e_drive: bool = False,
) -> RuntimePaths:
    """Point common HuggingFace/Torch temporary caches at configured paths.

    The function is explicit and side-effect free by default.  ``create=True``
    is used only when a real model is about to load; mock/unit-test scorers do
    not create directories.
    """

    assert_runtime_paths(paths, require_e_drive=require_e_drive)
    if create:
        for directory in (paths.cache_dir, paths.temp_dir):
            Path(directory).expanduser().mkdir(parents=True, exist_ok=True)
    cache_environment = {
        "HF_HOME": paths.cache_dir,
        "HUGGINGFACE_HUB_CACHE": str(Path(paths.cache_dir) / "hub"),
        "TRANSFORMERS_CACHE": str(Path(paths.cache_dir) / "transformers"),
        "TORCH_HOME": str(Path(paths.cache_dir) / "torch"),
    }
    for key, value in cache_environment.items():
        if require_e_drive:
            # An old C-drive value is precisely what the guarded production
            # mode is intended to repair.  Explicit E-drive configuration wins.
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    # Do not override a caller's process-wide TMP/TEMP unless explicitly
    # requested through VIDEO_HIGHLIGHT_SET_TEMP.  This avoids surprising
    # unrelated tools while still supporting the E-drive runner script.
    if os.environ.get("VIDEO_HIGHLIGHT_SET_TEMP", "0") == "1":
        os.environ["TMP"] = paths.temp_dir
        os.environ["TEMP"] = paths.temp_dir
    return paths


class QwenWindowScorer:
    """Score overlapping video windows with one reusable local Qwen model.

    ``response_provider`` is the preferred test/integration seam.  Its
    canonical signature is ``(video_path, start_sec, end_sec, sample_fps,
    prompt)``; two-argument ``(video_path, prompt)`` providers from the old
    project are accepted as a compatibility convenience.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        response_provider: Optional[Callable[..., str]] = None,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        load_in_4bit: bool = True,
        batch_size: int = 1,
        attn_implementation: Optional[str] = None,
        compact_output: bool = False,
        max_frame_pixels: int = 151200,
        total_video_pixels: int = 3211264,
        max_retries: int = 1,
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        compat_dir: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg",
        require_e_drive: bool = False,
        model: Any = None,
        processor: Any = None,
    ):
        self.paths = RuntimePaths.from_environment(model_path, cache_dir, temp_dir)
        if compat_dir:
            self.compat_dir = str(compat_dir)
        else:
            self.compat_dir = str(Path(self.paths.temp_dir) / "qwen_video_compat")
        assert_runtime_paths(
            RuntimePaths(self.paths.model_path, self.paths.cache_dir, self.compat_dir),
            require_e_drive=require_e_drive,
        )
        self.require_e_drive = bool(require_e_drive)
        self.ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self.response_provider = response_provider
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.load_in_4bit = bool(load_in_4bit)
        self.batch_size = max(1, int(batch_size))
        self.attn_implementation = (
            str(attn_implementation).strip() if attn_implementation else None
        )
        self.compact_output = bool(compact_output)
        self.max_frame_pixels = max(28 * 28, int(max_frame_pixels))
        self.total_video_pixels = max(self.max_frame_pixels, int(total_video_pixels))
        self.max_retries = max(0, int(max_retries))
        self.model = model
        self.processor = processor
        self.model_load_error: Optional[str] = None
        self.fatal_disabled_reason: Optional[str] = None
        if self.response_provider is None and (self.model is None or self.processor is None):
            try:
                self._load_model()
            except Exception as exc:
                # Loading errors are recorded and surfaced per window.  This
                # preserves the batch invariant that one failed model must not
                # terminate the complete video run.
                self.model = None
                self.processor = None
                self.model_load_error = f"model load failed: {type(exc).__name__}: {exc}"
                self.fatal_disabled_reason = self.model_load_error
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    @staticmethod
    def _raise_if_resource_error(exc: Exception) -> None:
        message = str(exc).lower()
        if isinstance(exc, MemoryError) or "out of memory" in message or "cuda oom" in message:
            raise exc

    @staticmethod
    def _is_resource_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return bool(
            isinstance(exc, MemoryError)
            or "out of memory" in message
            or "cuda oom" in message
            or "model load failed" in message
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _retry_instruction(self) -> str:
        from .prompts import load_text
        name = 'retry_compact.txt' if self.compact_output else 'retry_object.txt'
        return load_text('qwen/window-v1', name).rstrip('\n')

    def _load_model(self) -> None:
        if not self.paths.model_path:
            raise ValueError("model_path is required when response_provider is not set")
        configure_runtime_environment(
            self.paths,
            create=True,
            require_e_drive=self.require_e_drive,
        )
        import torch
        from transformers import AutoProcessor

        resolved_dtype: Any = self.dtype
        if isinstance(self.dtype, str) and self.dtype != "auto":
            resolved_dtype = getattr(torch, self.dtype.replace("torch.", ""), self.dtype)
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        def load_from_pretrained(model_class, **kwargs):
            """Pass an explicit E-drive cache, with compatibility fallback."""
            try:
                return model_class.from_pretrained(
                    self.paths.model_path,
                    cache_dir=self.paths.cache_dir,
                    **kwargs,
                )
            except TypeError as exc:
                # A few older Transformers test doubles/model wrappers do not
                # expose ``cache_dir``.  Environment variables are already
                # configured, so retry only for that compatibility case.
                if "cache_dir" not in str(exc):
                    raise
                return model_class.from_pretrained(self.paths.model_path, **kwargs)

        model = None
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

            model = load_from_pretrained(ModelClass, **load_kwargs)
        except Exception as exc:
            self._raise_if_resource_error(exc)
            try:
                from transformers import AutoModelForImageTextToText as ModelClass

                model = load_from_pretrained(ModelClass, **load_kwargs)
            except Exception as exc:
                self._raise_if_resource_error(exc)
                from transformers import AutoModelForVision2Seq as ModelClass

                model = load_from_pretrained(ModelClass, **load_kwargs)
        try:
            processor = AutoProcessor.from_pretrained(
                self.paths.model_path, cache_dir=self.paths.cache_dir
            )
        except TypeError as exc:
            if "cache_dir" not in str(exc):
                raise
            processor = AutoProcessor.from_pretrained(self.paths.model_path)
        if hasattr(model, "eval"):
            model.eval()
        self.model = model
        self.processor = processor

    def _video_request(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float,
    ) -> dict:
        """Build qwen-vl-utils content for the *original* video and range."""

        start = _finite_float(start_sec, "start_sec", non_negative=True)
        end = _finite_float(end_sec, "end_sec", non_negative=True)
        fps = _finite_float(sample_fps, "sample_fps")
        if end <= start:
            raise ValueError("end_sec must be greater than start_sec")
        path = Path(video_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        requested_frames = max(2, int(round((end - start) * fps)))
        # Keep an even frame count where possible; some torchcodec versions
        # reject odd frame requests.  The temporal range itself is unchanged.
        if requested_frames > 2 and requested_frames % 2:
            requested_frames -= 1
        return {
            "type": "video",
            "video": str(path),
            "video_start": start,
            "video_end": end,
            "fps": fps,
            # qwen-vl-utils accepts either ``fps`` or ``nframes``, never both.
            # Keep fps so coarse/fine sampling remains explicit, and only use
            # the frame limits as resource guards for this short time range.
            "min_frames": 2,
            "max_frames": requested_frames,
            "max_pixels": self.max_frame_pixels,
            "total_pixels": self.total_video_pixels,
        }

    @staticmethod
    def _normalize_video_kwargs(video_kwargs: Optional[Mapping[str, Any]]) -> dict:
        normalized = dict(video_kwargs or {})
        fps = normalized.get("fps")
        if isinstance(fps, (list, tuple)) and len(fps) == 1:
            normalized["fps"] = fps[0]
        return normalized

    @staticmethod
    def _process_vision_inputs(process_vision_info, messages):
        try:
            return process_vision_info(messages, return_video_kwargs=True)
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            return image_inputs, video_inputs, {}

    def _transcode_video_for_rgb(self, video_path: str) -> str:
        """Create an E-drive BT.709 compatibility copy only on decoder error."""

        source = Path(video_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(str(source))
        destination_dir = Path(self.compat_dir).expanduser()
        destination_dir.mkdir(parents=True, exist_ok=True)
        stat = source.stat()
        digest = hashlib.sha256(
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:16]
        output = destination_dir / f"{source.stem}_{digest}_bt709.mp4"
        if output.exists() and output.stat().st_size > 0:
            return str(output)
        partial = output.with_name(output.stem + ".part.mp4")
        command = [
            self.ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(partial),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if not partial.exists() or partial.stat().st_size <= 0:
                raise RuntimeError("ffmpeg produced an empty compatibility video")
            os.replace(partial, output)
        except Exception:
            try:
                partial.unlink()
            except OSError:
                pass
            raise
        return str(output)

    def _generate_once(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float,
        prompt: str,
    ) -> str:
        if self.response_provider is not None:
            return self._invoke_provider(
                self.response_provider,
                video_path,
                start_sec,
                end_sec,
                sample_fps,
                prompt,
            )
        if self.model_load_error:
            raise RuntimeError(self.model_load_error)
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM model is unavailable")

        content = self._video_request(video_path, start_sec, end_sec, sample_fps)
        messages = [
            {
                "role": "user",
                "content": [content, {"type": "text", "text": prompt}],
            }
        ]
        try:
            from qwen_vl_utils import process_vision_info

            try:
                image_inputs, video_inputs, video_kwargs = self._process_vision_inputs(
                    process_vision_info, messages
                )
            except (OSError, RuntimeError, ValueError, EOFError, IndexError) as exc:
                # Decoder failures are the only reason to make a compatibility
                # copy.  Keep the original time boundaries and FPS in content.
                compatible = self._transcode_video_for_rgb(video_path)
                warnings.warn(
                    f"video decoder failed ({exc}); using cached compatibility copy: {compatible}",
                    RuntimeWarning,
                )
                messages[0]["content"][0] = dict(content, video=compatible)
                image_inputs, video_inputs, video_kwargs = self._process_vision_inputs(
                    process_vision_info, messages
                )
        except ImportError:
            raise
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video_kwargs = self._normalize_video_kwargs(video_kwargs)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        try:
            model_device = getattr(self.model, "device", None)
            if model_device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            import torch

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            input_ids = getattr(inputs, "input_ids", None)
            if input_ids is not None:
                generated = generated[:, input_ids.shape[1] :]
            decoded = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return str(decoded[0] if decoded else "")
        finally:
            # Release temporary tensor references between windows.  The model
            # itself remains loaded and is reused by the next call.
            try:
                del inputs
            except UnboundLocalError:
                pass

    def _generate_batch_once(
        self,
        video_path: str,
        specs: Sequence[WindowSpec],
        prompts: Sequence[str],
    ) -> List[str]:
        """Generate one response per window in a single GPU batch."""

        if len(specs) != len(prompts):
            raise ValueError("specs and prompts must have the same length")
        if not specs:
            return []
        if len(specs) == 1:
            spec = specs[0]
            return [
                self._generate_once(
                    video_path,
                    spec.start_sec,
                    spec.end_sec,
                    spec.sample_fps,
                    prompts[0],
                )
            ]
        if self.response_provider is not None:
            return [
                self._invoke_provider(
                    self.response_provider,
                    video_path,
                    spec.start_sec,
                    spec.end_sec,
                    spec.sample_fps,
                    prompt,
                )
                for spec, prompt in zip(specs, prompts)
            ]
        if self.model_load_error:
            raise RuntimeError(self.model_load_error)
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM model is unavailable")

        conversations = []
        for spec, prompt in zip(specs, prompts):
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            self._video_request(
                                video_path,
                                spec.start_sec,
                                spec.end_sec,
                                spec.sample_fps,
                            ),
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        from qwen_vl_utils import process_vision_info

        try:
            image_inputs, video_inputs, video_kwargs = self._process_vision_inputs(
                process_vision_info, conversations
            )
        except (OSError, RuntimeError, ValueError, EOFError, IndexError) as exc:
            compatible = self._transcode_video_for_rgb(video_path)
            warnings.warn(
                f"video decoder failed ({exc}); using cached compatibility copy: {compatible}",
                RuntimeWarning,
            )
            for conversation in conversations:
                conversation[0]["content"][0]["video"] = compatible
            image_inputs, video_inputs, video_kwargs = self._process_vision_inputs(
                process_vision_info, conversations
            )

        texts = [
            self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]
        video_kwargs = self._normalize_video_kwargs(video_kwargs)
        tokenizer = getattr(self.processor, "tokenizer", None)
        old_padding_side = getattr(tokenizer, "padding_side", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        try:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            model_device = getattr(self.model, "device", None)
            if model_device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            import torch

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            input_ids = getattr(inputs, "input_ids", None)
            if input_ids is not None:
                generated = generated[:, input_ids.shape[1] :]
            decoded = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            rows = [str(value) for value in decoded]
            if len(rows) != len(specs):
                raise RuntimeError(
                    f"batch generation returned {len(rows)} rows for {len(specs)} windows"
                )
            return rows
        finally:
            if tokenizer is not None and old_padding_side is not None:
                tokenizer.padding_side = old_padding_side
            try:
                del inputs
            except UnboundLocalError:
                pass

    def generate_range(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float,
        prompt: str,
    ) -> str:
        """Generate raw text for a caller-defined video-range prompt.

        The structured window scorer remains the stable default.  Experimental
        temporal frontends use this narrow public seam so they can reuse the
        same loaded Qwen model and the same original-MP4 decoding path without
        depending on the private generation implementation.
        """

        return self._generate_once(
            video_path,
            start_sec,
            end_sec,
            sample_fps,
            str(prompt),
        )

    def generate_ranges(
        self,
        video_path: str,
        specs: Sequence[WindowSpec],
        prompts: Sequence[str],
    ) -> List[str]:
        """Generate raw text for aligned range prompts in configured batches."""

        window_specs = [validate_window(spec) for spec in specs]
        prompt_rows = [str(prompt) for prompt in prompts]
        if len(window_specs) != len(prompt_rows):
            raise ValueError("specs and prompts must have the same length")
        output: List[str] = []
        for offset in range(0, len(window_specs), self.batch_size):
            chunk = window_specs[offset : offset + self.batch_size]
            chunk_prompts = prompt_rows[offset : offset + self.batch_size]
            output.extend(
                self._generate_batch_once(video_path, chunk, chunk_prompts)
            )
        return output

    @staticmethod
    def _invoke_provider(
        provider: Callable[..., str],
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float,
        prompt: str,
    ) -> str:
        """Call injected providers with several backwards-compatible shapes."""

        # Prefer parameter names when available; this avoids silently passing
        # the prompt as the second positional argument to a five-argument
        # provider that follows the documented window-first order.
        try:
            signature = inspect.signature(provider)
            params = list(signature.parameters.values())
            names = [p.name.lower() for p in params]
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        except (TypeError, ValueError):
            names, has_varargs, has_varkw = [], True, True

        if has_varargs or len(names) >= 5:
            if len(names) >= 2 and "prompt" in names[1]:
                return str(provider(video_path, prompt, start_sec, end_sec, sample_fps))
            return str(provider(video_path, start_sec, end_sec, sample_fps, prompt))
        if has_varkw or any(name in names for name in ("video_start", "start_sec", "fps")):
            try:
                return str(
                    provider(
                        video_path,
                        prompt,
                        video_start=start_sec,
                        video_end=end_sec,
                        fps=sample_fps,
                    )
                )
            except TypeError:
                return str(provider(video_path, prompt))
        if len(names) >= 2:
            return str(provider(video_path, prompt))
        if len(names) == 1:
            return str(provider(video_path))
        return str(provider())

    def score_window(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float,
        prompt: Optional[str] = None,
        *,
        window_id: int = 0,
        stage: str = "coarse",
        parent_id: Optional[int] = None,
    ) -> WindowScore:
        """Score one window, retrying malformed/generation responses once."""

        try:
            spec = validate_window(
                WindowSpec(
                    int(window_id),
                    float(start_sec),
                    float(end_sec),
                    float(sample_fps),
                    str(stage or "coarse"),
                    parent_id,
                )
            )
        except Exception as exc:
            return WindowScore(
                int(window_id),
                float(start_sec or 0.0),
                float(end_sec or 0.0),
                float(sample_fps or 0.0),
                stage=str(stage or "coarse"),
                parent_id=parent_id,
                valid=False,
                errors=[f"invalid window: {type(exc).__name__}: {exc}"],
                attempts=0,
            )

        base_prompt = prompt or build_window_score_prompt(
            sample_fps=spec.sample_fps,
            compact_output=self.compact_output,
        )
        if self.fatal_disabled_reason:
            return WindowScore(
                spec.window_id,
                spec.start_sec,
                spec.end_sec,
                spec.sample_fps,
                raw=f"<generation-error>{self.fatal_disabled_reason}</generation-error>",
                valid=False,
                errors=[self.fatal_disabled_reason],
                attempts=0,
                stage=spec.stage,
                parent_id=spec.parent_id,
            )
        current_prompt = base_prompt
        errors: List[str] = []
        raw_responses: List[str] = []
        raw = ""
        total_attempts = self.max_retries + 1
        attempts_used = 0
        for attempt in range(1, total_attempts + 1):
            attempts_used = attempt
            try:
                raw = self._generate_once(
                    video_path,
                    spec.start_sec,
                    spec.end_sec,
                    spec.sample_fps,
                    current_prompt,
                )
            except Exception as exc:
                raw = f"<generation-error>{type(exc).__name__}: {exc}</generation-error>"
                raw_responses.append(raw)
                errors.append(f"generation failed: {type(exc).__name__}: {exc}")
                parsed = None
                if self._is_resource_error(exc):
                    # Retrying an OOM/model-load failure only repeats the
                    # expensive failure and can further fragment VRAM.  The
                    # frontend will stop subsequent model windows and use
                    # traditional signals instead.
                    self._clear_cuda_cache()
                    self.fatal_disabled_reason = (
                        f"fatal Qwen resource failure: {type(exc).__name__}: {exc}"
                    )
                    break
            else:
                raw_responses.append(str(raw))
                parsed = parse_window_score_response(
                    raw,
                    spec,
                    attempts=attempt,
                    compact_output=self.compact_output,
                )
                if parsed.valid:
                    # Preserve earlier transient errors for diagnostics while
                    # marking the final record valid.
                    parsed.errors = list(errors)
                    parsed.attempts = attempt
                    parsed.raw_responses = list(raw_responses)
                    return parsed
                errors.extend(parsed.errors)

            if attempt < total_attempts:
                current_prompt = base_prompt + self._retry_instruction()

        return WindowScore(
            spec.window_id,
            spec.start_sec,
            spec.end_sec,
            spec.sample_fps,
            raw=raw,
            valid=False,
            errors=errors or ["window scoring failed"],
            attempts=attempts_used,
            stage=spec.stage,
            parent_id=spec.parent_id,
            raw_responses=raw_responses,
        )

    def score_windows(
        self,
        video_path: str,
        windows: Iterable[WindowSpec],
        prompt: Optional[str] = None,
    ) -> List[WindowScore]:
        """Score windows in GPU batches while isolating malformed responses."""

        window_list = list(windows)
        if self.batch_size <= 1 or len(window_list) <= 1:
            return self._score_windows_sequential(video_path, window_list, prompt)

        results: List[WindowScore] = []
        for offset in range(0, len(window_list), self.batch_size):
            chunk = window_list[offset : offset + self.batch_size]
            try:
                specs = [validate_window(window) for window in chunk]
            except Exception:
                # Invalid external metadata is rare and easier to isolate using
                # the existing per-window safety path.
                results.extend(self._score_windows_sequential(video_path, chunk, prompt))
                continue
            try:
                results.extend(self._score_window_batch(video_path, specs, prompt))
            except Exception as exc:
                if self._is_resource_error(exc):
                    self._clear_cuda_cache()
                    self.fatal_disabled_reason = (
                        f"fatal Qwen resource failure: {type(exc).__name__}: {exc}"
                    )
                    results.extend(
                        _invalid_score_for_spec(spec, self.fatal_disabled_reason)
                        for spec in specs
                    )
                    for remaining in window_list[offset + len(chunk) :]:
                        try:
                            remaining_spec = validate_window(remaining)
                        except Exception:
                            results.extend(
                                self._score_windows_sequential(
                                    video_path, [remaining], prompt
                                )
                            )
                        else:
                            results.append(
                                _invalid_score_for_spec(
                                    remaining_spec, self.fatal_disabled_reason
                                )
                            )
                    break
                # A decoder or processor edge case should cost only this chunk.
                # Splitting it into the proven sequential path preserves output.
                results.extend(self._score_windows_sequential(video_path, chunk, prompt))
        return results

    def _score_window_batch(
        self,
        video_path: str,
        specs: Sequence[WindowSpec],
        prompt: Optional[str],
    ) -> List[WindowScore]:
        base_prompts = [
            prompt
            or build_window_score_prompt(
                sample_fps=spec.sample_fps,
                compact_output=self.compact_output,
            )
            for spec in specs
        ]
        pending = list(range(len(specs)))
        responses: Dict[int, List[str]] = {index: [] for index in pending}
        errors: Dict[int, List[str]] = {index: [] for index in pending}
        completed: Dict[int, WindowScore] = {}

        for attempt in range(1, self.max_retries + 2):
            active_specs = [specs[index] for index in pending]
            active_prompts = []
            for index in pending:
                value = base_prompts[index]
                if attempt > 1:
                    value += self._retry_instruction()
                active_prompts.append(value)
            raw_rows = self._generate_batch_once(video_path, active_specs, active_prompts)
            retry: List[int] = []
            for index, raw in zip(pending, raw_rows):
                responses[index].append(raw)
                parsed = parse_window_score_response(
                    raw,
                    specs[index],
                    attempts=attempt,
                    compact_output=self.compact_output,
                )
                if parsed.valid:
                    parsed.errors = list(errors[index])
                    parsed.raw_responses = list(responses[index])
                    completed[index] = parsed
                else:
                    errors[index].extend(parsed.errors)
                    retry.append(index)
            pending = retry
            if not pending:
                break

        for index in pending:
            spec = specs[index]
            raw_rows = responses[index]
            completed[index] = WindowScore(
                spec.window_id,
                spec.start_sec,
                spec.end_sec,
                spec.sample_fps,
                raw=raw_rows[-1] if raw_rows else "",
                valid=False,
                errors=errors[index] or ["window scoring failed"],
                attempts=len(raw_rows),
                stage=spec.stage,
                parent_id=spec.parent_id,
                raw_responses=list(raw_rows),
            )
        return [completed[index] for index in range(len(specs))]

    def _score_windows_sequential(
        self,
        video_path: str,
        windows: Sequence[WindowSpec],
        prompt: Optional[str],
    ) -> List[WindowScore]:

        results: List[WindowScore] = []
        for window in windows:
            try:
                spec = validate_window(window)
                result = self.score_window(
                    video_path,
                    spec.start_sec,
                    spec.end_sec,
                    spec.sample_fps,
                    prompt,
                    window_id=spec.window_id,
                    stage=spec.stage,
                    parent_id=spec.parent_id,
                )
            except Exception as exc:
                # A bad external window should be isolated just like a model
                # failure; retain its ID when possible for diagnostics.
                wid = int(getattr(window, "window_id", len(results)))
                results.append(
                    WindowScore(
                        wid,
                        float(getattr(window, "start_sec", 0.0) or 0.0),
                        float(getattr(window, "end_sec", 0.0) or 0.0),
                        float(getattr(window, "sample_fps", 0.0) or 0.0),
                        valid=False,
                        errors=[f"window failed: {type(exc).__name__}: {exc}"],
                        attempts=0,
                    )
                )
            else:
                results.append(result)
        return results

    def score_video(
        self,
        video_path: str,
        *,
        duration_sec: Optional[float] = None,
        window_sec: float = 2.0,
        step_sec: float = 1.0,
        sample_fps: float = 2.0,
        stage: str = "coarse",
        prompt: Optional[str] = None,
        start_window_id: int = 0,
    ) -> "WindowScoreDetectionResult":
        """Generate windows and score a video sequentially."""

        if duration_sec is None:
            duration_sec, _fps, _n, _w, _h = probe_video(video_path)
        duration = _finite_float(duration_sec, "duration_sec", non_negative=True)
        windows = generate_windows(
            duration,
            window_sec,
            step_sec,
            sample_fps,
            stage=stage,
            start_window_id=start_window_id,
        )
        records = self.score_windows(video_path, windows, prompt=prompt)
        errors = [error for record in records for error in record.errors]
        valid_records = [record for record in records if record.valid]
        return WindowScoreDetectionResult(
            duration_sec=duration,
            windows=windows,
            records=records,
            valid=bool(valid_records),
            errors=errors,
        )

    def close(self) -> None:
        """Release model references and CUDA cache after a batch run."""

        self.processor = None
        self.model = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except (AttributeError, RuntimeError):
                    pass
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
        return False


@dataclass
class WindowScoreDetectionResult:
    """Plain result object consumed by the later signal/fusion stage."""

    duration_sec: float
    windows: List[WindowSpec] = field(default_factory=list)
    records: List[WindowScore] = field(default_factory=list)
    valid: bool = False
    errors: List[str] = field(default_factory=list)
    # These fields intentionally mirror the legacy TemporalResult names so a
    # runner can construct that type without a special adapter in temporal.py.
    segments_sec: List[Tuple[float, float]] = field(default_factory=list)
    subject_hint: Optional[str] = None
    source: str = "window_score"

    @property
    def window_records(self) -> List[WindowScore]:
        return self.records

    @property
    def score_curve(self) -> List[dict]:
        return [
            {
                "window_id": record.window_id,
                "start_sec": record.start_sec,
                "end_sec": record.end_sec,
                "qwen_score": record.qwen_score,
            }
            for record in self.records
        ]

    def best_record(self) -> Optional[WindowScore]:
        valid = [record for record in self.records if record.valid and record.qwen_score is not None]
        return max(valid, key=lambda record: record.qwen_score or 0.0) if valid else None

    def to_dict(self) -> dict:
        return {
            "duration_sec": self.duration_sec,
            "source": self.source,
            "valid": bool(self.valid),
            "segments_sec": [list(segment) for segment in self.segments_sec],
            "subject_hint": self.subject_hint,
            "windows": [window.as_dict() for window in self.windows],
            "window_records": [record.to_dict() for record in self.records],
            "errors": list(self.errors),
        }


class WindowScoreTemporalDetector:
    """Public ``window_score`` detector facade.

    ``detect(video_path)`` is the documented temporal entry point and returns
    the project's :class:`~video_highlight.temporal.TemporalResult`,
    including the full diagnostic curve.  ``detect_windows`` remains available
    for callers that only want the raw Qwen window records (and preserves the
    lightweight adapter that existed during early experiments).

    The import of :class:`WindowScoreFrontend` is deliberately lazy because
    that module imports :class:`QwenWindowScorer` from this module.
    """

    def __init__(
        self,
        scorer: Optional[QwenWindowScorer] = None,
        *,
        window_sec: float = 2.0,
        step_sec: float = 1.0,
        sample_fps: float = 2.0,
        prompt: Optional[str] = None,
        frontend_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.scorer = scorer or QwenWindowScorer()
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.sample_fps = float(sample_fps)
        self.prompt = prompt
        self.frontend_kwargs = dict(frontend_kwargs or {})
        self._frontend = None

    def detect_windows(self, video_path: str) -> WindowScoreDetectionResult:
        """Return raw coarse Qwen windows without signal fusion."""
        return self.scorer.score_video(
            video_path,
            window_sec=self.window_sec,
            step_sec=self.step_sec,
            sample_fps=self.sample_fps,
            prompt=self.prompt,
        )

    def detect(self, video_path: str) -> "TemporalResult":
        """Run the complete coarse-to-fine frontend and return ``TemporalResult``."""
        if self._frontend is None:
            from .window_score_frontend import WindowScoreFrontend

            options = {
                "coarse_window_sec": self.window_sec,
                "coarse_stride_sec": self.step_sec,
                "coarse_fps": self.sample_fps,
            }
            options.update(self.frontend_kwargs)
            self._frontend = WindowScoreFrontend(self.scorer, **options)
        return self._frontend.detect(video_path)

    def close(self) -> None:
        close = getattr(self.scorer, "close", None)
        if callable(close):
            close()


def build_window_score_detector(
    *,
    model_path: Optional[str] = None,
    response_provider: Optional[Callable[..., str]] = None,
    **kwargs,
) -> WindowScoreTemporalDetector:
    """Factory kept separate from the legacy ``build_temporal_detector``."""

    scorer = QwenWindowScorer(
        model_path=model_path,
        response_provider=response_provider,
        **{
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "device_map",
                "dtype",
                "max_new_tokens",
                "load_in_4bit",
                "max_frame_pixels",
                "total_video_pixels",
                "max_retries",
                "cache_dir",
                "temp_dir",
                "compat_dir",
                "ffmpeg_bin",
                "require_e_drive",
                "model",
                "processor",
            }
        },
    )
    frontend_keys = {
        "coarse_window_sec",
        "coarse_stride_sec",
        "coarse_fps",
        "fine_window_sec",
        "fine_stride_sec",
        "fine_fps",
        "max_fine_regions",
        "curve_step_sec",
        "smoothing_points",
        "relative_top_fraction",
        "absolute_threshold",
        "coarse_candidate_threshold",
        "merge_gap_sec",
        "boundary_expand_sec",
        "max_boundary_expand_sec",
        "max_segments",
        "max_coverage_ratio",
        "max_coverage_sec",
        "short_full_sec",
        "ffmpeg_bin",
        "audio_sample_rate",
        "signal_weights",
        "signal_provider",
        "yolo_interaction_provider",
    }
    frontend_kwargs = {
        key: kwargs[key]
        for key in frontend_keys
        if key in kwargs
    }
    return WindowScoreTemporalDetector(
        scorer,
        window_sec=kwargs.get("window_sec", 2.0),
        step_sec=kwargs.get("step_sec", 1.0),
        sample_fps=kwargs.get("sample_fps", 2.0),
        prompt=kwargs.get("prompt"),
        frontend_kwargs=frontend_kwargs,
    )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_TEMP_DIR",
    "QWEN_SCORE_KEYS",
    "QWEN_SCORE_WEIGHTS",
    "WINDOW_RESPONSE_KEYS",
    "QwenWindowScorer",
    "RuntimePaths",
    "WindowScore",
    "WindowScoreDetectionResult",
    "WindowScoreTemporalDetector",
    "assert_runtime_paths",
    "build_window_score_detector",
    "build_window_score_prompt",
    "configure_runtime_environment",
    "parse_window_score_response",
    "qwen_score_from_components",
]
