"""Window-level temporal signals used by the experimental highlight front end.

This module deliberately contains no model code and does not know anything about
the spatial/YOLO stage.  It provides two small pieces of functionality:

* inexpensive motion, audio and scene-change measurements for one video window;
* safe, missing-value-aware fusion of those measurements with a semantic score
  produced by a VLM.

The public helpers accept either mappings or dataclass/object records.  Fusion
always returns *ordinary dictionaries* so callers can serialise the result to a
diagnostic JSONL file without depending on a particular dataclass implementation.
All failures are converted to ``None`` for the affected signal; one bad window
must not abort a complete video run.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# These names are intentionally shared with the window scorer.  Keeping the
# spelling stable makes raw diagnostic records easy to inspect and merge.
SIGNAL_KEYS: Tuple[str, ...] = (
    "qwen_score",
    "motion_score",
    "audio_score",
    "scene_score",
    "yolo_interaction_score",
)
TRADITIONAL_SIGNAL_KEYS: Tuple[str, ...] = (
    "motion_score",
    "audio_score",
    "scene_score",
    "yolo_interaction_score",
)
DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {
    "qwen_score": 0.50,
    "motion_score": 0.20,
    "audio_score": 0.10,
    # Scene change is a boundary cue, not positive highlight evidence.  YOLO
    # remains a reserved callback owned by the spatial teammate, so neither
    # channel participates in the stable temporal fusion by default.
    "scene_score": 0.0,
    "yolo_interaction_score": 0.0,
}

_QWEN_COMPONENT_KEYS: Tuple[str, ...] = (
    "action",
    "completion",
    "interaction",
    "event",
    "emotion",
    "visual_impact",
    "highlight",
)
_QWEN_COMPONENT_WEIGHTS: Dict[str, float] = {
    "action": 0.15,
    "completion": 0.15,
    "interaction": 0.15,
    "event": 0.15,
    "emotion": 0.10,
    "visual_impact": 0.10,
    "highlight": 0.20,
}


def record_to_dict(record: Any) -> Dict[str, Any]:
    """Return a shallow ordinary-dict view of a mapping/dataclass/object.

    ``dataclasses.asdict`` is used for dataclasses (including slots dataclasses),
    while ``vars`` is used as a fallback for simple objects.  A malformed record
    is represented by an empty dictionary instead of raising from a long batch.
    """

    if record is None:
        return {}
    if isinstance(record, Mapping):
        return dict(record)
    if dataclasses.is_dataclass(record):
        try:
            return dict(dataclasses.asdict(record))
        except (TypeError, ValueError):
            pass
    if hasattr(record, "_asdict"):
        try:
            return dict(record._asdict())
        except Exception:  # pragma: no cover - unusual namedtuple-like object
            pass
    try:
        return dict(vars(record))
    except (TypeError, ValueError):
        return {}


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clamp01(value: Any) -> Optional[float]:
    """Convert a value to a finite number in ``[0, 1]`` or return ``None``."""

    number = _finite_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _interval_from_record(record: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    """Read the common start/end spellings used by window records."""

    start = None
    end = None
    for key in ("start_sec", "window_start_sec", "start", "t_start"):
        if key in record:
            start = _finite_number(record.get(key))
            break
    for key in ("end_sec", "window_end_sec", "end", "t_end"):
        if key in record:
            end = _finite_number(record.get(key))
            break
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    if end <= start:
        return None
    return (start, end)


def _normalise_qwen_components(record: Mapping[str, Any]) -> Optional[float]:
    """Derive a semantic score from the seven 0--3 Qwen fields.

    A supplied ``qwen_score``/``semantic_score`` takes precedence.  Component
    values outside ``[0, 1]`` are interpreted as the contract's ``0..3`` values;
    values already in ``[0, 1]`` are retained.  The weighted average is kept
    intentionally simple and deterministic because this is a ranking signal,
    not a learned classifier.
    """

    if record.get("qwen_valid") is False or record.get("valid") is False:
        return None

    for key in ("qwen_score", "semantic_score", "qwen_semantic_score"):
        if key in record:
            value = clamp01(record.get(key))
            if value is not None:
                return value

    nested = record.get("qwen_components")
    nested = nested if isinstance(nested, Mapping) else {}
    weighted_values: List[Tuple[float, float]] = []
    for key in _QWEN_COMPONENT_KEYS:
        from_contract = key in nested
        raw_value = nested.get(key) if from_contract else record.get(key)
        value = _finite_number(raw_value)
        if value is None:
            continue
        # Values inside ``qwen_components`` and top-level Python integers come
        # from the strict 0/1/2/3 response contract.  In particular, integer
        # ``1`` means one point (1/3), not a pre-normalised score of 1.0.
        # Float-only external callers may still supply values already in [0,1].
        if from_contract or (
            isinstance(raw_value, int) and not isinstance(raw_value, bool)
        ) or value > 1.0:
            value /= 3.0
        weighted_values.append(
            (max(0.0, min(1.0, value)), _QWEN_COMPONENT_WEIGHTS[key])
        )
    if not weighted_values:
        return None
    denominator = sum(weight for _value, weight in weighted_values)
    return float(
        sum(value * weight for value, weight in weighted_values) / denominator
    )


def _safe_minmax(
    values: Sequence[Any],
    *,
    min_dynamic_range: float = 1e-4,
) -> Tuple[List[Optional[float]], Dict[str, Any]]:
    """Min/max normalise finite values, safely handling missing/constant data."""

    finite = [_finite_number(value) for value in values]
    present = [value for value in finite if value is not None]
    diagnostics: Dict[str, Any] = {
        "present_count": len(present),
        "constant": False,
        "min": None,
        "max": None,
        "range": None,
    }
    if not present:
        return [None for _ in finite], diagnostics
    low = min(present)
    high = max(present)
    diagnostics["min"] = low
    diagnostics["max"] = high
    diagnostics["range"] = high - low
    if abs(high - low) <= max(0.0, float(min_dynamic_range)) + 1e-12:
        diagnostics["constant"] = True
        # A constant signal carries no ranking information.  Returning zero,
        # rather than NaN or 0.5, makes this explicit and lets fusion disable a
        # degenerate channel.
        return [0.0 if value is not None else None for value in finite], diagnostics
    scale = high - low
    return [((value - low) / scale) if value is not None else None for value in finite], diagnostics


def fit_signal_calibration(
    records: Iterable[Any],
    keys: Sequence[str] = TRADITIONAL_SIGNAL_KEYS,
    *,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
    min_dynamic_range: float = 1e-4,
) -> Dict[str, Dict[str, Any]]:
    """Fit one robust per-video scale for later coarse/fine normalisation.

    The stable window-score route fits this object from *coarse* windows only
    and applies it unchanged to both passes.  Consequently a two-window fine
    pass cannot turn a tiny numerical difference into artificial 0/1 evidence.
    Values outside the coarse P10--P90 range are clipped at application time.
    """

    rows = [record_to_dict(record) for record in records]
    low_q = max(0.0, min(100.0, float(lower_percentile)))
    high_q = max(low_q, min(100.0, float(upper_percentile)))
    calibration: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        values: List[float] = []
        for row in rows:
            value = (
                _normalise_qwen_components(row)
                if key == "qwen_score"
                else _finite_number(row.get(key))
            )
            if value is not None:
                values.append(float(value))
        detail: Dict[str, Any] = {
            "present_count": len(values),
            "constant": False,
            "low": None,
            "high": None,
            "range": None,
            "lower_percentile": low_q,
            "upper_percentile": high_q,
            "method": "percentile",
        }
        if values:
            array = np.asarray(values, dtype=np.float64)
            try:
                low = float(np.percentile(array, low_q))
                high = float(np.percentile(array, high_q))
            except (TypeError, ValueError, FloatingPointError):
                low = min(values)
                high = max(values)
            detail.update({"low": low, "high": high, "range": high - low})
            detail["constant"] = bool(
                high - low <= max(0.0, float(min_dynamic_range)) + 1e-12
            )
        calibration[str(key)] = detail
    return calibration


def apply_signal_calibration(
    records: Iterable[Any],
    calibration: Mapping[str, Mapping[str, Any]],
    keys: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply a fitted scale without recomputing statistics on fine windows."""

    rows = [record_to_dict(record) for record in records]
    selected_keys = tuple(keys or calibration.keys())
    diagnostics: Dict[str, Any] = {
        "signals": {},
        "row_count": len(rows),
        "calibration_scope": "external_reference",
    }
    for key in selected_keys:
        detail = dict(calibration.get(key) or {})
        diagnostics["signals"][key] = detail
        low = _finite_number(detail.get("low"))
        high = _finite_number(detail.get("high"))
        constant = bool(detail.get("constant")) or low is None or high is None
        scale = (high - low) if low is not None and high is not None else 0.0
        if scale <= 1e-12:
            constant = True
        for row in rows:
            raw = (
                _normalise_qwen_components(row)
                if key == "qwen_score"
                else _finite_number(row.get(key))
            )
            if raw is None:
                value = None
            elif key == "qwen_score":
                # Qwen already has a meaningful fixed 0..1 rubric.  Never
                # stretch it with a per-video or per-stage range.
                value = clamp01(raw)
            elif constant:
                value = 0.0
            else:
                value = max(0.0, min(1.0, (float(raw) - float(low)) / scale))
            row[f"normalized_{key}"] = value
    return rows, diagnostics


def normalize_signal_records(
    records: Iterable[Any],
    keys: Sequence[str] = SIGNAL_KEYS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalise signal columns across a video.

    The original values remain under their original names.  Normalised values
    are written as ``normalized_<key>``.  The returned diagnostics identify
    columns with no data or no variation.  This function never emits NaN.
    """

    rows = [record_to_dict(record) for record in records]
    diagnostics: Dict[str, Any] = {"signals": {}, "row_count": len(rows)}
    for key in keys:
        values = []
        for row in rows:
            if key == "qwen_score":
                value = _normalise_qwen_components(row)
                # Make the derived semantic score available to callers.
                if value is not None and row.get("qwen_score") is None:
                    row["qwen_score"] = value
            else:
                value = row.get(key)
            values.append(value)
        if key == "qwen_score":
            # The model rubric is already divided by three into a calibrated
            # 0..1 score.  Preserve that absolute meaning; per-video Min-Max
            # made harmless 0.51/0.52 differences look like 0/1 evidence.
            finite = [_finite_number(value) for value in values]
            present = [value for value in finite if value is not None]
            low = min(present) if present else None
            high = max(present) if present else None
            column_diag = {
                "present_count": len(present),
                "constant": bool(
                    present and high is not None and low is not None and high - low <= 1e-4 + 1e-12
                ),
                "min": low,
                "max": high,
                "range": (high - low) if high is not None and low is not None else None,
                "method": "fixed_0_1",
            }
            normalised = [clamp01(value) if value is not None else None for value in finite]
        else:
            normalised, column_diag = _safe_minmax(values)
        diagnostics["signals"][key] = column_diag
        for row, value in zip(rows, normalised):
            row[f"normalized_{key}"] = value
    return rows, diagnostics


def _channel_value(row: Mapping[str, Any], key: str) -> Optional[float]:
    normalized_key = f"normalized_{key}"
    value = row.get(normalized_key) if normalized_key in row else row.get(key)
    return clamp01(value)


def fuse_window_scores(
    records: Iterable[Any],
    weights: Optional[Mapping[str, float]] = None,
    *,
    normalize: bool = True,
    disable_degenerate_qwen: bool = True,
    qwen_degenerate_epsilon: float = 0.05,
    disabled_signals: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Fuse Qwen and traditional window signals with missing-value safety.

    Parameters are deliberately pure-data oriented so the function can be used
    by the runner without importing model classes.  Missing channels are removed
    from each row's denominator and the remaining weights are renormalised.  If
    all available Qwen scores are constant, the Qwen channel is disabled by
    default and a diagnostic warning is attached to every row.
    """

    aliases = {"qwen": "qwen_score", "semantic": "qwen_score", "yolo": "yolo_interaction_score"}
    selected_weights = dict(DEFAULT_SIGNAL_WEIGHTS)
    if weights:
        for key, value in weights.items():
            number = _finite_number(value)
            if number is not None and number >= 0:
                selected_weights[aliases.get(str(key), str(key))] = number

    if normalize:
        rows, column_diagnostics = normalize_signal_records(
            records, keys=tuple(selected_weights.keys())
        )
    else:
        rows = [record_to_dict(record) for record in records]
        column_diagnostics = {"signals": {}, "row_count": len(rows)}
        for row in rows:
            # Derive qwen from components even when callers request no min/max.
            if row.get("qwen_score") is None:
                derived = _normalise_qwen_components(row)
                if derived is not None:
                    row["qwen_score"] = derived

    disabled_channels = {
        aliases.get(str(key), str(key)) for key in (disabled_signals or [])
    }
    # A constant normalised traditional signal contains no ranking
    # information.  Remove it from every row's denominator just like a missing
    # channel; retaining its weight with an all-zero value would only suppress
    # otherwise valid evidence.
    for key, diagnostic in column_diagnostics.get("signals", {}).items():
        if isinstance(diagnostic, Mapping) and diagnostic.get("constant"):
            disabled_channels.add(aliases.get(str(key), str(key)))

    qwen_values = [
        value
        for value in (_normalise_qwen_components(row) for row in rows)
        if value is not None
    ]
    qwen_disabled = False
    qwen_range = None
    if disable_degenerate_qwen and qwen_values:
        qwen_range = max(qwen_values) - min(qwen_values)
        qwen_disabled = qwen_range <= max(0.0, qwen_degenerate_epsilon) + 1e-12
    elif qwen_values:
        qwen_range = max(qwen_values) - min(qwen_values)

    # A channel can be supplied under a raw value even when normalisation was
    # disabled.  ``normalized_*`` is preferred where present.
    for row in rows:
        available: List[Tuple[str, float, float]] = []
        for key, weight in selected_weights.items():
            canonical = aliases.get(key, key)
            if canonical in disabled_channels:
                continue
            if canonical == "qwen_score" and qwen_disabled:
                continue
            if canonical == "qwen_score" and _normalise_qwen_components(row) is None:
                continue
            value = _channel_value(row, canonical)
            if value is None or weight <= 0:
                continue
            available.append((canonical, value, float(weight)))

        denominator = sum(item[2] for item in available)
        fused = (
            sum(value * weight for _key, value, weight in available) / denominator
            if denominator > 0
            else None
        )

        traditional = [item for item in available if item[0] in TRADITIONAL_SIGNAL_KEYS]
        traditional_denominator = sum(item[2] for item in traditional)
        traditional_score = (
            sum(value * weight for _key, value, weight in traditional) / traditional_denominator
            if traditional_denominator > 0
            else None
        )
        row["traditional_score"] = traditional_score
        row["fused_score"] = fused
        # ``score`` is the short spelling consumed by score_curve helpers.
        row["score"] = fused
        row["fusion_available_signals"] = [item[0] for item in available]
        row["fusion_weight_sum"] = denominator
        row["qwen_disabled"] = qwen_disabled
        row["score_diagnostics"] = {
            "missing_signals": [
                key
                for key in selected_weights
                if aliases.get(key, key) not in row.get("fusion_available_signals", [])
            ],
            "qwen_degenerate": qwen_disabled,
            "qwen_range": qwen_range,
            "disabled_constant_signals": sorted(disabled_channels),
            "normalization": column_diagnostics.get("signals", {}),
        }

    # Keep diagnostics available without changing the return type.  A private
    # conventional key is useful to the runner's raw JSONL writer; callers that
    # only need rows can ignore it.
    if rows:
        column_diagnostics["qwen_degenerate"] = qwen_disabled
        column_diagnostics["qwen_range"] = qwen_range
        if qwen_disabled:
            column_diagnostics.setdefault("warnings", []).append(
                "Qwen window scores have no useful variation; channel disabled"
            )
        rows[0].setdefault("_fusion_diagnostics", column_diagnostics)
    return rows


def _read_window_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 2.0,
    max_frames: int = 64,
) -> List[np.ndarray]:
    """Read a small list of RGB/BGR frames without creating temporary files."""

    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    target_fps = max(0.1, float(sample_fps))
    cap = cv2.VideoCapture(str(video_path))
    if cap is None or not cap.isOpened():
        return []
    try:
        fps = _finite_number(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 0:
            fps = target_fps
        total = _finite_number(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
        first = max(0, int(math.floor(start * fps)))
        last = int(math.ceil(end * fps))
        if total > 0:
            last = min(last, int(total) - 1)
        if last < first:
            return []
        step = max(1, int(round(fps / target_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames: List[np.ndarray] = []
        frame_index = first
        while frame_index <= last and len(frames) < max(1, int(max_frames)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (frame_index - first) % step == 0:
                frames.append(frame)
            frame_index += 1
        return frames
    finally:
        cap.release()


def compute_motion_score(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 2.0,
    *,
    max_frames: int = 64,
    resize: Tuple[int, int] = (160, 90),
) -> Optional[float]:
    """Return a bounded motion/change score for one temporal window."""

    frames = _read_window_frames(video_path, start_sec, end_sec, sample_fps, max_frames)
    if not frames:
        return None
    if len(frames) < 2:
        return 0.0
    diffs: List[float] = []
    changed: List[float] = []
    width, height = int(resize[0]), int(resize[1])
    previous = None
    for frame in frames:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        except Exception:
            continue
        if previous is not None:
            delta = cv2.absdiff(previous, gray).astype(np.float32) / 255.0
            diffs.append(float(np.mean(delta)))
            changed.append(float(np.mean(delta > (12.0 / 255.0))))
        previous = gray
    if not diffs:
        return 0.0
    value = 0.70 * float(np.mean(diffs)) + 0.30 * float(np.mean(changed))
    return max(0.0, min(1.0, value))


def compute_scene_score(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 2.0,
    *,
    max_frames: int = 64,
) -> Optional[float]:
    """Return a bounded histogram-change score (a weak shot-boundary signal)."""

    frames = _read_window_frames(video_path, start_sec, end_sec, sample_fps, max_frames)
    if not frames:
        return None
    if len(frames) < 2:
        return 0.0
    previous = None
    changes: List[float] = []
    for frame in frames:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
        except Exception:
            continue
        if previous is not None:
            try:
                distance = float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA))
            except Exception:
                distance = 0.0
            changes.append(max(0.0, min(1.0, distance)))
        previous = hist
    return float(max(0.0, min(1.0, np.mean(changes)))) if changes else 0.0


def _audio_log_rms_levels(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_sec: float = 0.1,
) -> Optional[np.ndarray]:
    """Return a 0.1-second log-RMS envelope for normalised mono PCM."""

    try:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if samples.size == 0:
        return None
    finite_mask = np.isfinite(samples)
    if not bool(np.any(finite_mask)):
        return None
    if not bool(np.all(finite_mask)):
        samples = samples[finite_mask]
    chunk_size = max(1, int(round(max(1000, int(sample_rate)) * max(0.01, frame_sec))))
    levels: List[float] = []
    for start in range(0, samples.size, chunk_size):
        chunk = samples[start : start + chunk_size]
        if chunk.size:
            rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
            levels.append(float(np.log(rms + 1e-6)))
    return np.asarray(levels, dtype=np.float32) if levels else None


def _audio_change_score_from_levels(levels: np.ndarray) -> Optional[float]:
    """Score positive log-energy onsets and within-window energy change.

    Absolute loudness is intentionally absent: steady loud music should not
    outrank a quieter but abrupt cheer/impact merely because its RMS is larger.
    """

    try:
        values = np.asarray(levels, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if values.size < 2:
        return 0.0
    deltas = np.diff(values.astype(np.float64))
    positive_jump = float(np.max(np.maximum(deltas, 0.0))) if deltas.size else 0.0
    try:
        change = float(np.percentile(np.abs(deltas), 90.0)) if deltas.size else 0.0
    except (TypeError, ValueError, FloatingPointError):
        change = float(np.max(np.abs(deltas))) if deltas.size else 0.0
    # Natural-log changes around 0.5--1.0 already represent a meaningful
    # amplitude jump.  Exponential saturation bounds rare clipping spikes.
    jump_score = 1.0 - math.exp(-max(0.0, positive_jump) / 0.75)
    change_score = 1.0 - math.exp(-max(0.0, change) / 0.60)
    return max(0.0, min(1.0, 0.65 * jump_score + 0.35 * change_score))


def _audio_score_from_samples(
    samples: np.ndarray,
    sample_rate: int = 8000,
) -> Optional[float]:
    """Compute the bounded audio-change feature from mono PCM."""

    levels = _audio_log_rms_levels(samples, sample_rate)
    return _audio_change_score_from_levels(levels) if levels is not None else None


def _decode_audio_pcm(
    video_path: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 8000,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    timeout_sec: float = 120.0,
) -> Optional[np.ndarray]:
    """Decode mono PCM once, returning normalised float samples or ``None``."""

    command = [str(ffmpeg_bin), "-v", "error"]
    if start_sec is not None and float(start_sec) > 0:
        command.extend(["-ss", f"{float(start_sec):.6f}"])
    command.extend(["-i", str(video_path)])
    if duration_sec is not None and float(duration_sec) > 0:
        command.extend(["-t", f"{float(duration_sec):.6f}"])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(max(1000, int(sample_rate))),
            "-f",
            "s16le",
            "pipe:1",
        ]
    )
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(0.1, float(timeout_sec)),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    try:
        return np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except (TypeError, ValueError):
        return None


def compute_audio_score(
    video_path: str,
    start_sec: float,
    end_sec: float,
    *,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 8000,
    timeout_sec: float = 30.0,
) -> Optional[float]:
    """Read window PCM through FFmpeg and return a bounded energy/peak score.

    FFmpeg is intentionally used as a pipe, so no C-drive or repository cache is
    created.  Missing binaries, silent/no-audio streams, and decoder failures
    all return ``None`` (the fusion layer then removes this channel's weight).
    """

    start = max(0.0, float(start_sec))
    duration = max(0.0, float(end_sec) - start)
    if duration <= 0:
        return None
    samples = _decode_audio_pcm(
        video_path,
        ffmpeg_bin=ffmpeg_bin,
        sample_rate=sample_rate,
        start_sec=start,
        duration_sec=duration,
        timeout_sec=timeout_sec,
    )
    return (
        _audio_score_from_samples(samples, sample_rate)
        if samples is not None
        else None
    )


# Friendly aliases used by callers/tests that prefer noun-style names.
motion_score = compute_motion_score
audio_score = compute_audio_score
scene_score = compute_scene_score


def extract_window_signals(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 2.0,
    *,
    ffmpeg_bin: str = "ffmpeg",
    audio_sample_rate: int = 8000,
    include_audio: bool = True,
    motion_fn: Optional[Callable[..., Optional[float]]] = None,
    audio_fn: Optional[Callable[..., Optional[float]]] = None,
    scene_fn: Optional[Callable[..., Optional[float]]] = None,
) -> Dict[str, Any]:
    """Extract all traditional signals for one window.

    Optional function hooks make the routine deterministic in unit tests and
    allow a runner to reuse decoded frames.  The YOLO interaction signal is
    deliberately present as ``None`` until the spatial teammate supplies a
    callback.
    """

    interval = {
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "sample_fps": float(sample_fps),
    }
    try:
        interval["motion_score"] = (
            motion_fn(video_path, start_sec, end_sec, sample_fps)
            if motion_fn is not None
            else compute_motion_score(video_path, start_sec, end_sec, sample_fps)
        )
    except Exception as exc:  # one signal must not abort the window
        interval["motion_score"] = None
        interval["motion_error"] = f"{type(exc).__name__}: {exc}"
    try:
        if include_audio:
            interval["audio_score"] = (
                audio_fn(video_path, start_sec, end_sec)
                if audio_fn is not None
                else compute_audio_score(
                    video_path,
                    start_sec,
                    end_sec,
                    ffmpeg_bin=ffmpeg_bin,
                    sample_rate=audio_sample_rate,
                )
            )
        else:
            interval["audio_score"] = None
    except Exception as exc:
        interval["audio_score"] = None
        interval["audio_error"] = f"{type(exc).__name__}: {exc}"
    try:
        interval["scene_score"] = (
            scene_fn(video_path, start_sec, end_sec, sample_fps)
            if scene_fn is not None
            else compute_scene_score(video_path, start_sec, end_sec, sample_fps)
        )
    except Exception as exc:
        interval["scene_score"] = None
        interval["scene_error"] = f"{type(exc).__name__}: {exc}"
    # Explicitly reserve the hook; a missing YOLO value is removed from the
    # fusion denominator rather than interpreted as zero evidence.
    interval["yolo_interaction_score"] = None
    return interval


def _batch_visual_signals(
    video_path: str,
    windows: Sequence[Mapping[str, Any]],
    default_sample_fps: float,
    resize: Tuple[int, int] = (160, 90),
) -> List[Dict[str, Optional[float]]]:
    """Decode a video's visual stream once and aggregate pair features by window."""

    intervals = [_interval_from_record(window) for window in windows]
    output = [{"motion_score": None, "scene_score": None} for _ in windows]
    valid = [(index, interval) for index, interval in enumerate(intervals) if interval is not None]
    if not valid:
        return output
    first_sec = max(0.0, min(interval[0] for _index, interval in valid))
    last_sec = max(interval[1] for _index, interval in valid)
    # Qwen fine windows may request 4 FPS, but frame differences are only
    # comparable when both passes use the same temporal spacing.  The signal
    # extractor therefore owns one fixed sampling rate (2 FPS in production)
    # instead of inheriting each model window's sample_fps.
    target_fps = max(0.1, float(default_sample_fps))

    cap = cv2.VideoCapture(str(video_path))
    if cap is None or not cap.isOpened():
        return output
    motion_values: List[List[float]] = [[] for _ in windows]
    scene_values: List[List[float]] = [[] for _ in windows]
    try:
        fps = _finite_number(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 0:
            fps = target_fps
        total = _finite_number(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
        first_frame = max(0, int(math.floor(first_sec * fps)))
        last_frame = int(math.ceil(last_sec * fps))
        if total > 0:
            last_frame = min(last_frame, int(total) - 1)
        step = max(1, int(round(fps / target_fps)))
        seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        if first_frame > 0 and not seek_ok:
            return output
        previous_gray = None
        previous_hist = None
        previous_timestamp = None
        frame_index = first_frame
        width, height = int(resize[0]), int(resize[1])
        while frame_index <= last_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (frame_index - first_frame) % step != 0:
                frame_index += 1
                continue
            # Prefer the decoder's actual presentation timestamp for VFR
            # material.  OpenCV backends that do not expose POS_MSEC fall back
            # to the CFR metadata estimate.
            position_msec = _finite_number(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = (
                max(0.0, position_msec / 1000.0)
                if position_msec is not None and position_msec > 0
                else frame_index / fps
            )
            if timestamp + max(1.0 / fps, 1e-3) < first_sec:
                frame_index += 1
                continue
            if timestamp > last_sec + max(1.0 / fps, 1e-3):
                break
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hsv],
                    [0, 1, 2],
                    None,
                    [12, 8, 8],
                    [0, 180, 0, 256, 0, 256],
                )
                hist = cv2.normalize(hist, hist).flatten()
            except Exception:
                frame_index += 1
                continue
            if previous_gray is not None and previous_hist is not None:
                delta = cv2.absdiff(previous_gray, gray).astype(np.float32) / 255.0
                motion = 0.70 * float(np.mean(delta)) + 0.30 * float(
                    np.mean(delta > (12.0 / 255.0))
                )
                try:
                    scene = float(
                        cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    )
                except Exception:
                    scene = 0.0
                midpoint = (
                    (previous_timestamp + timestamp) / 2.0
                    if previous_timestamp is not None
                    else timestamp
                )
                for window_index, interval in valid:
                    if interval[0] - 1e-9 <= midpoint < interval[1] + 1e-9:
                        motion_values[window_index].append(max(0.0, min(1.0, motion)))
                        scene_values[window_index].append(max(0.0, min(1.0, scene)))
            previous_gray = gray
            previous_hist = hist
            previous_timestamp = timestamp
            frame_index += 1
    finally:
        cap.release()

    for index in range(len(windows)):
        if motion_values[index]:
            output[index]["motion_score"] = float(np.mean(motion_values[index]))
        if scene_values[index]:
            output[index]["scene_score"] = float(np.mean(scene_values[index]))
    return output


def _batch_audio_signals(
    video_path: str,
    windows: Sequence[Mapping[str, Any]],
    *,
    ffmpeg_bin: str,
    sample_rate: int,
) -> List[Optional[float]]:
    """Decode one PCM stream and slice it into every requested window."""

    intervals = [_interval_from_record(window) for window in windows]
    valid = [interval for interval in intervals if interval is not None]
    if not valid:
        return [None for _ in windows]
    first_sec = max(0.0, min(interval[0] for interval in valid))
    last_sec = max(interval[1] for interval in valid)
    samples = _decode_audio_pcm(
        video_path,
        ffmpeg_bin=ffmpeg_bin,
        sample_rate=sample_rate,
        start_sec=first_sec,
        duration_sec=max(0.0, last_sec - first_sec),
    )
    if samples is None:
        return [None for _ in windows]
    rate = max(1000, int(sample_rate))
    result: List[Optional[float]] = []
    for interval in intervals:
        if interval is None:
            result.append(None)
            continue
        start_index = max(0, int(math.floor((interval[0] - first_sec) * rate)))
        end_index = min(samples.size, int(math.ceil((interval[1] - first_sec) * rate)))
        result.append(
            _audio_score_from_samples(samples[start_index:end_index], rate)
            if end_index > start_index
            else None
        )
    return result


def _audio_scores_from_levels(
    levels: Optional[np.ndarray],
    windows: Sequence[Mapping[str, Any]],
    *,
    frame_sec: float = 0.1,
) -> List[Optional[float]]:
    """Slice one full-video log-RMS envelope into requested windows."""

    if levels is None:
        return [None for _ in windows]
    values = np.asarray(levels, dtype=np.float32).reshape(-1)
    result: List[Optional[float]] = []
    step = max(0.01, float(frame_sec))
    for window in windows:
        interval = _interval_from_record(window)
        if interval is None:
            result.append(None)
            continue
        start_index = max(0, int(math.floor(interval[0] / step)))
        end_index = min(values.size, int(math.ceil(interval[1] / step)))
        # Include one preceding envelope point so an onset exactly at the
        # window start is visible as a positive log-energy jump.
        context_start = max(0, start_index - 1)
        result.append(
            _audio_change_score_from_levels(values[context_start:end_index])
            if end_index > start_index
            else None
        )
    return result


class TemporalSignalExtractor:
    """Small state-free adapter suitable for ``WindowScoreTemporalDetector``."""

    def __init__(
        self,
        sample_fps: float = 2.0,
        *,
        ffmpeg_bin: str = "ffmpeg",
        audio_sample_rate: int = 8000,
        include_audio: bool = True,
        motion_fn: Optional[Callable[..., Optional[float]]] = None,
        audio_fn: Optional[Callable[..., Optional[float]]] = None,
        scene_fn: Optional[Callable[..., Optional[float]]] = None,
    ):
        self.sample_fps = float(sample_fps)
        self.ffmpeg_bin = ffmpeg_bin
        self.audio_sample_rate = max(1000, int(audio_sample_rate))
        self.include_audio = bool(include_audio)
        self.motion_fn = motion_fn
        self.audio_fn = audio_fn
        self.scene_fn = scene_fn
        self._audio_cache_path: Optional[str] = None
        self._audio_cache_levels: Optional[np.ndarray] = None
        self._audio_cache_attempted = False

    def _cached_audio_scores(
        self,
        video_path: str,
        windows: Sequence[Mapping[str, Any]],
    ) -> List[Optional[float]]:
        path = str(video_path)
        if self._audio_cache_path != path:
            self._audio_cache_path = path
            self._audio_cache_levels = None
            self._audio_cache_attempted = False
        if not self._audio_cache_attempted:
            self._audio_cache_attempted = True
            samples = _decode_audio_pcm(
                path,
                ffmpeg_bin=self.ffmpeg_bin,
                sample_rate=self.audio_sample_rate,
            )
            self._audio_cache_levels = (
                _audio_log_rms_levels(samples, self.audio_sample_rate)
                if samples is not None
                else None
            )
        return _audio_scores_from_levels(self._audio_cache_levels, windows)

    def score_window(self, video_path: str, start_sec: float, end_sec: float) -> Dict[str, Any]:
        return extract_window_signals(
            video_path,
            start_sec,
            end_sec,
            self.sample_fps,
            ffmpeg_bin=self.ffmpeg_bin,
            audio_sample_rate=self.audio_sample_rate,
            include_audio=self.include_audio,
            motion_fn=self.motion_fn,
            audio_fn=self.audio_fn,
            scene_fn=self.scene_fn,
        )

    # ``extract`` and ``detect`` are aliases used by different runner revisions.
    extract = score_window

    def score_windows(self, video_path: str, windows: Iterable[Any]) -> List[Dict[str, Any]]:
        """Score many windows with one visual decode and one FFmpeg audio decode.

        Custom test/caller hooks still run per window; the default production
        path sequentially reads motion+scene once and extracts full-window PCM
        once, which avoids repeatedly seeking the same MP4.
        """

        rows = [record_to_dict(window) for window in windows]
        if not rows:
            return []
        visual = (
            _batch_visual_signals(video_path, rows, self.sample_fps)
            if self.motion_fn is None or self.scene_fn is None
            else [{} for _ in rows]
        )
        audio = (
            self._cached_audio_scores(video_path, rows)
            if self.include_audio and self.audio_fn is None
            else [None for _ in rows]
        )

        output: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            copy = dict(row)
            interval = _interval_from_record(copy)
            if interval is None:
                copy.update(
                    {
                        "motion_score": None,
                        "audio_score": None,
                        "scene_score": None,
                        "yolo_interaction_score": copy.get("yolo_interaction_score"),
                        "signal_error": "window does not contain a valid start/end interval",
                    }
                )
                output.append(copy)
                continue
            start, end = interval
            signal_fps = self.sample_fps
            try:
                motion = (
                    self.motion_fn(video_path, start, end, signal_fps)
                    if self.motion_fn is not None
                    else visual[index].get("motion_score")
                )
            except Exception as exc:
                motion = None
                copy["motion_error"] = f"{type(exc).__name__}: {exc}"
            try:
                scene = (
                    self.scene_fn(video_path, start, end, signal_fps)
                    if self.scene_fn is not None
                    else visual[index].get("scene_score")
                )
            except Exception as exc:
                scene = None
                copy["scene_error"] = f"{type(exc).__name__}: {exc}"
            try:
                if not self.include_audio:
                    audio_value = None
                elif self.audio_fn is not None:
                    audio_value = self.audio_fn(video_path, start, end)
                else:
                    audio_value = audio[index]
            except Exception as exc:
                audio_value = None
                copy["audio_error"] = f"{type(exc).__name__}: {exc}"
            copy.update(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "signal_sample_fps": signal_fps,
                    "motion_score": clamp01(motion),
                    "audio_score": clamp01(audio_value),
                    "scene_score": clamp01(scene),
                    "yolo_interaction_score": copy.get("yolo_interaction_score"),
                }
            )
            output.append(copy)
        return output

    detect = score_windows

    def close(self) -> None:
        self._audio_cache_path = None
        self._audio_cache_levels = None
        self._audio_cache_attempted = False

    def __call__(self, video_path: str, start_sec: float, end_sec: float) -> Dict[str, Any]:
        return self.score_window(video_path, start_sec, end_sec)


__all__ = [
    "SIGNAL_KEYS",
    "TRADITIONAL_SIGNAL_KEYS",
    "DEFAULT_SIGNAL_WEIGHTS",
    "TemporalSignalExtractor",
    "apply_signal_calibration",
    "audio_score",
    "clamp01",
    "compute_audio_score",
    "compute_motion_score",
    "compute_scene_score",
    "extract_window_signals",
    "fit_signal_calibration",
    "fuse_window_scores",
    "motion_score",
    "normalize_signal_records",
    "record_to_dict",
    "scene_score",
]
