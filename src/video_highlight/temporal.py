import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io_utils import iter_sampled_frames, video_meta


SegmentSeconds = Tuple[float, float]


@dataclass
class TemporalResult:
    segments_sec: List[SegmentSeconds]
    subject_hint: Optional[str] = None
    raw: Optional[str] = None
    source: str = ""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    attempts: int = 1
    fallback_used: Optional[str] = None
    duration_sec: Optional[float] = None
    coverage_ratio: Optional[float] = None
    vlm_raw_responses: List[str] = field(default_factory=list)
    segments_before_normalization: List[SegmentSeconds] = field(default_factory=list)
    segments_after_normalization: List[SegmentSeconds] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    # Optional diagnostics produced by the window-score temporal frontend.
    # Keeping these fields on the existing result object lets the runner pass
    # only ``segments_sec`` to the spatial/YOLO stage while retaining a full
    # per-window audit trail in the raw JSONL file.  All fields are optional so
    # the legacy motion/full/vlm modes remain source-compatible.
    window_records: List[dict] = field(default_factory=list)
    score_curve: List[dict] = field(default_factory=list)
    selected_peaks: List[dict] = field(default_factory=list)
    score_diagnostics: dict = field(default_factory=dict)
    # Optional human-readable action attached by window-based frontends.  It
    # deliberately lives at the end of the dataclass so legacy positional
    # construction of TemporalResult remains compatible.
    action_description: Optional[str] = None


@dataclass
class TemporalParseResult:
    segments_sec: List[SegmentSeconds]
    subject_hint: Optional[str] = None
    raw: Optional[str] = None
    valid: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class TemporalNormalizationResult:
    segments_sec: List[SegmentSeconds]
    valid: bool
    errors: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)


def _json_dict_candidates(text: str):
    """Yield JSON dictionaries in strict-to-tolerant priority order."""
    stripped = (text or "").strip()
    if not stripped:
        return

    seen = set()

    def load(candidate: str):
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return None
        seen.add(candidate)
        try:
            obj = json.loads(candidate)
        except (TypeError, ValueError):
            return None
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
            # Qwen2.5-VL sometimes follows the official baseline's legacy
            # top-level list schema even when asked for a JSON object. Keep the
            # compatibility narrow: entries must expose either explicit
            # start/end keys or their own segments list. All flattened entries
            # are still validated by _segments_from_object below.
            hints = []
            for item in obj:
                hint = item.get("subject_hint") or item.get("subject") or item.get("main_subject")
                if hint:
                    hints.append(str(hint).strip().lower())
            if all(isinstance(item.get("segments"), list) for item in obj):
                segments = [segment for item in obj for segment in item["segments"]]
            elif all(
                ("start_sec" in item and "end_sec" in item)
                or ("start" in item and "end" in item)
                for item in obj
            ):
                segments = obj
            else:
                return None
            payload = {"segments": segments}
            if hints:
                payload["subject_hint"] = max(dict.fromkeys(hints), key=hints.count)
            return payload
        return None

    obj = load(stripped)
    if obj is not None:
        yield obj

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, re.I | re.S):
        obj = load(match.group(1))
        if obj is not None:
            yield obj

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stripped[index:])
        except ValueError:
            continue
        candidate = stripped[index : index + end]
        if candidate in seen:
            continue
        seen.add(candidate)
        if isinstance(obj, dict):
            yield obj


def _segments_from_object(obj: dict) -> Tuple[List[SegmentSeconds], List[str]]:
    if "segments" not in obj:
        return [], ["JSON object does not contain a segments key"]
    raw_segments = obj.get("segments")
    if not isinstance(raw_segments, list):
        return [], ["segments must be a list"]

    segments: List[SegmentSeconds] = []
    errors: List[str] = []
    for index, item in enumerate(raw_segments):
        pair = None
        allow_numeric_strings = False
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pair = (item[0], item[1])
        elif isinstance(item, dict):
            if "start_sec" in item and "end_sec" in item:
                pair = (item.get("start_sec"), item.get("end_sec"))
                allow_numeric_strings = True
            elif "start" in item and "end" in item:
                pair = (item.get("start"), item.get("end"))
                allow_numeric_strings = True
        if pair is None:
            errors.append(f"segment {index} is not a [start, end] pair")
            continue
        if any(isinstance(value, bool) for value in pair):
            errors.append(f"segment {index} contains boolean times")
            continue
        converted = []
        for value in pair:
            if isinstance(value, (int, float)):
                converted.append(float(value))
                continue
            if allow_numeric_strings and isinstance(value, str) and re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip()
            ):
                converted.append(float(value))
                continue
            errors.append(f"segment {index} times must be JSON numbers")
            converted = []
            break
        if not converted:
            continue
        try:
            start, end = converted
        except (TypeError, ValueError):
            errors.append(f"segment {index} contains non-numeric times")
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            errors.append(f"segment {index} contains a non-finite time")
            continue
        segments.append((start, end))

    if not segments:
        errors.append("segments is empty or contains no valid interval")
    return segments, errors


def parse_temporal_response(text: str, duration_sec: Optional[float] = None) -> TemporalParseResult:
    """Parse a VLM response without extracting arbitrary numeric pairs.

    When a duration is supplied, intervals that lie completely outside the video
    are rejected here as an early guard. Repairable clipping and expansion remain
    the responsibility of ``normalize_temporal_segments``.
    """
    if not isinstance(text, str) or not text.strip():
        return TemporalParseResult([], raw=text, valid=False, errors=["empty model response"])

    candidate_errors: List[str] = []
    found_json = False
    for obj in _json_dict_candidates(text):
        found_json = True
        if "segments" not in obj:
            continue
        segments, errors = _segments_from_object(obj)
        if duration_sec is not None and segments:
            try:
                duration = float(duration_sec)
            except (TypeError, ValueError):
                duration = 0.0
            in_range = []
            for index, (start, end) in enumerate(segments):
                left, right = sorted((start, end))
                if duration <= 0 or right < 0 or left > duration or (
                    left >= duration and right > duration
                ):
                    errors.append(
                        f"segment {index} is outside video duration 0..{duration:.3f}"
                    )
                    continue
                in_range.append((start, end))
            segments = in_range
            if not segments:
                errors.append("no interval overlaps the video duration")
        hint = obj.get("subject_hint") or obj.get("subject") or obj.get("main_subject")
        hint = str(hint).strip().lower() if hint else None
        if segments and not errors:
            return TemporalParseResult(
                segments_sec=segments,
                subject_hint=hint,
                raw=text,
                valid=True,
                errors=[],
            )
        candidate_errors.extend(errors)

    if found_json and not candidate_errors:
        candidate_errors.append("no parsed JSON object contained a valid segments array")
    else:
        if not found_json:
            candidate_errors.append("no valid JSON object found in model response")
    return TemporalParseResult([], raw=text, valid=False, errors=candidate_errors)


def parse_temporal_json(text: str) -> TemporalResult:
    """Backward-compatible wrapper around the stricter response parser."""
    parsed = parse_temporal_response(text)
    return TemporalResult(
        parsed.segments_sec,
        parsed.subject_hint,
        text,
        source="vlm",
        valid=parsed.valid,
        errors=list(parsed.errors),
        vlm_raw_responses=[text] if isinstance(text, str) else [],
        segments_before_normalization=list(parsed.segments_sec),
    )


def _merge_second_segments(segments: Iterable[SegmentSeconds]) -> List[SegmentSeconds]:
    ordered = sorted((float(start), float(end)) for start, end in segments)
    if not ordered:
        return []
    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def normalize_temporal_segments(
    segments: Iterable[Sequence[float]],
    duration_sec: float,
    min_segment_sec: float = 4.0,
    padding_sec: float = 3.0,
    max_segments: int = 3,
) -> TemporalNormalizationResult:
    """Validate and normalize model intervals before converting seconds to frames."""
    errors: List[str] = []
    corrections: List[str] = []
    clean: List[SegmentSeconds] = []

    try:
        duration = float(duration_sec)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        return TemporalNormalizationResult([], False, ["video duration must be positive"], corrections)

    minimum = max(0.0, min(float(min_segment_sec), duration))
    padding = max(0.0, float(padding_sec))
    limit = max(1, int(max_segments))

    for index, item in enumerate(segments or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            errors.append(f"segment {index} is malformed")
            continue
        try:
            start = float(item[0])
            end = float(item[1])
        except (TypeError, ValueError):
            errors.append(f"segment {index} contains non-numeric times")
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            errors.append(f"segment {index} contains a non-finite time")
            continue

        if end < start:
            start, end = end, start
            corrections.append(f"segment {index}: swapped reversed start/end")

        if end < 0 or start > duration or (start >= duration and end > duration):
            errors.append(
                f"segment {index} is outside video duration: [{start:.3f}, {end:.3f}] vs 0..{duration:.3f}"
            )
            continue

        clipped_start = max(0.0, start)
        clipped_end = min(duration, end)
        if clipped_start != start or clipped_end != end:
            corrections.append(f"segment {index}: clipped to video duration")
        start, end = clipped_start, clipped_end

        if end <= start + 1e-9:
            errors.append(f"segment {index} has zero length after validation")
            continue

        if minimum > 0 and end - start < minimum - 1e-9:
            center = (start + end) / 2.0
            start = center - minimum / 2.0
            end = center + minimum / 2.0
            if start < 0:
                end = min(duration, end - start)
                start = 0.0
            if end > duration:
                start = max(0.0, start - (end - duration))
                end = duration
            corrections.append(f"segment {index}: expanded to minimum duration")

        if padding > 0:
            padded_start = max(0.0, start - padding)
            padded_end = min(duration, end + padding)
            if padded_start != start or padded_end != end:
                corrections.append(f"segment {index}: applied temporal padding")
            start, end = padded_start, padded_end

        if end <= start + 1e-9:
            errors.append(f"segment {index} became zero length after normalization")
            continue
        clean.append((start, end))

    merged = _merge_second_segments(clean)
    if len(merged) < len(clean):
        corrections.append("merged overlapping or duplicate intervals")
    if len(merged) > limit:
        merged = sorted(merged, key=lambda pair: pair[1] - pair[0], reverse=True)[:limit]
        merged.sort()
        corrections.append(f"kept the {limit} longest intervals")

    if not merged:
        if not errors:
            errors.append("no temporal interval remained after normalization")
        return TemporalNormalizationResult([], False, errors, corrections)
    return TemporalNormalizationResult(merged, True, errors, corrections)


def temporal_coverage_ratio(segments: Iterable[SegmentSeconds], duration_sec: float) -> float:
    try:
        duration = float(duration_sec)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(duration) or duration <= 0:
        return 0.0
    clean = []
    for item in segments or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            clean.append((start, end))
    merged = _merge_second_segments(clean)
    covered = sum(max(0.0, end - start) for start, end in merged)
    return max(0.0, min(1.0, covered / duration))


def _minimum_coverage_seconds(duration_sec: float, ratio: float, max_coverage_sec: float) -> float:
    target = max(0.0, float(duration_sec) * max(0.0, min(1.0, float(ratio))))
    cap = max(0.0, float(max_coverage_sec))
    if cap > 0:
        target = min(target, cap)
    return min(float(duration_sec), target)


def _covered_seconds(segments: Iterable[SegmentSeconds]) -> float:
    return sum(max(0.0, end - start) for start, end in _merge_second_segments(segments))


def _expand_to_coverage(
    segments: List[SegmentSeconds], duration_sec: float, target_seconds: float, max_segments: int
) -> List[SegmentSeconds]:
    current = _merge_second_segments(segments)
    target_seconds = min(max(0.0, target_seconds), duration_sec)
    for _ in range(12):
        covered = _covered_seconds(current)
        missing = target_seconds - covered
        if missing <= 1e-6 or not current:
            break
        share = missing / float(len(current))
        expanded = []
        for start, end in current:
            left_available = max(0.0, start)
            right_available = max(0.0, duration_sec - end)
            left_add = min(share / 2.0, left_available)
            right_add = min(share - left_add, right_available)
            remaining = share - left_add - right_add
            if remaining > 0:
                extra_left = min(remaining, left_available - left_add)
                left_add += extra_left
                remaining -= extra_left
            if remaining > 0:
                right_add += min(remaining, right_available - right_add)
            expanded.append((start - left_add, end + right_add))
        normalized = normalize_temporal_segments(
            expanded,
            duration_sec=duration_sec,
            min_segment_sec=0.0,
            padding_sec=0.0,
            max_segments=max_segments,
        )
        if not normalized.valid or _covered_seconds(normalized.segments_sec) <= covered + 1e-6:
            break
        current = normalized.segments_sec
    return current


def sampled_frame_timestamps(
    n_frames: int, fps: float, detect_fps: float, max_items: int = 300
) -> List[float]:
    if n_frames <= 0 or fps <= 0:
        return []
    step_frames = max(1, int(round(float(fps) / max(0.1, float(detect_fps)))))
    frame_ids = list(range(0, int(n_frames), step_frames))
    if len(frame_ids) > max_items:
        indices = np.linspace(0, len(frame_ids) - 1, max_items, dtype=int)
        frame_ids = [frame_ids[int(index)] for index in indices]
    return [frame_id / float(fps) for frame_id in frame_ids]


def build_temporal_prompt(
    duration_sec: float,
    fps: float,
    detect_fps: float,
    max_segments: int,
    sample_timestamps: Sequence[float],
) -> str:
    from .prompts import call_builder
    return call_builder("qwen/direct-v1", "build_temporal_prompt", duration_sec, fps, detect_fps, max_segments, sample_timestamps)


def build_temporal_retry_prompt(
    base_prompt: str, duration_sec: float, previous_response: str, validation_errors: Sequence[str]
) -> str:
    from .prompts import call_builder
    return call_builder("qwen/direct-v1", "build_temporal_retry_prompt", base_prompt, duration_sec, previous_response, validation_errors)


class FullVideoTemporalDetector:
    def detect(self, video_path: str) -> TemporalResult:
        n_frames, fps, _w, _h = video_meta(video_path)
        if n_frames <= 0 or fps <= 0:
            return TemporalResult([], raw="full-video", source="full", valid=False, errors=["unreadable video"])
        duration = n_frames / fps
        return TemporalResult(
            [(0.0, duration)],
            None,
            "full-video",
            source="full",
            duration_sec=duration,
            coverage_ratio=1.0,
            segments_after_normalization=[(0.0, duration)],
        )


class MiddleTemporalDetector:
    def __init__(self, duration_sec: float = 4.0):
        self.duration_sec = float(duration_sec)

    def detect(self, video_path: str) -> TemporalResult:
        n_frames, fps, _w, _h = video_meta(video_path)
        if n_frames <= 0 or fps <= 0:
            return TemporalResult([], raw="middle-window", source="middle", valid=False, errors=["unreadable video"])
        total = n_frames / fps
        dur = min(total, self.duration_sec)
        start = max(0.0, (total - dur) / 2.0)
        segments = [(start, min(total, start + dur))]
        return TemporalResult(
            segments,
            None,
            "middle-window",
            source="middle",
            duration_sec=total,
            coverage_ratio=temporal_coverage_ratio(segments, total),
            segments_after_normalization=list(segments),
        )


class MotionTemporalDetector:
    """Training-free temporal detector: choose windows with strongest motion."""

    def __init__(
        self,
        detect_fps: float = 1.0,
        top_k: int = 1,
        window_sec: float = 4.0,
        min_segment_sec: float = 1.0,
        padding_sec: float = 0.25,
    ):
        self.detect_fps = max(0.1, float(detect_fps))
        self.top_k = max(1, int(top_k))
        self.window_sec = max(0.5, float(window_sec))
        self.min_segment_sec = max(0.1, float(min_segment_sec))
        self.padding_sec = max(0.0, float(padding_sec))

    def detect(self, video_path: str) -> TemporalResult:
        n_frames, fps, _w, _h = video_meta(video_path)
        if n_frames <= 0 or fps <= 0:
            return TemporalResult([], raw="motion-window", source="motion", valid=False, errors=["unreadable video"])
        total_sec = n_frames / fps
        step_frames = max(1, int(round(fps / self.detect_fps)))
        samples = []
        prev = None
        for frame_id, frame in iter_sampled_frames(video_path, step_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if prev is None:
                motion = 0.0
            else:
                motion = float(np.mean(cv2.absdiff(gray, prev)))
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            samples.append((frame_id / fps, motion, sharp))
            prev = gray

        if not samples:
            return TemporalResult([], raw="motion-window", source="motion", valid=False, errors=["no sampled frames"])
        motions = np.array([sample[1] for sample in samples], dtype=np.float32)
        sharpness = np.array([sample[2] for sample in samples], dtype=np.float32)
        if motions.max() > motions.min():
            motions = (motions - motions.min()) / (motions.max() - motions.min())
        if sharpness.max() > sharpness.min():
            sharpness = (sharpness - sharpness.min()) / (sharpness.max() - sharpness.min())
        scores = 0.8 * motions + 0.2 * sharpness

        windows = []
        step_sec = max(1.0 / self.detect_fps, 0.1)
        n_per_window = max(1, int(round(self.window_sec / step_sec)))
        for start_index in range(0, len(samples)):
            end_index = min(len(samples), start_index + n_per_window)
            if end_index <= start_index:
                continue
            score = float(np.mean(scores[start_index:end_index]))
            start = samples[start_index][0]
            end = min(total_sec, samples[end_index - 1][0] + step_sec)
            if end - start < self.min_segment_sec:
                end = min(total_sec, start + self.min_segment_sec)
            windows.append((score, start, end))

        windows.sort(reverse=True, key=lambda item: item[0])
        selected: List[SegmentSeconds] = []
        for _score, start, end in windows:
            start = max(0.0, start - self.padding_sec)
            end = min(total_sec, end + self.padding_sec)
            if any(not (end < left or start > right) for left, right in selected):
                continue
            selected.append((start, end))
            if len(selected) >= self.top_k:
                break
        selected.sort()
        return TemporalResult(
            selected,
            None,
            "motion-window",
            source="motion",
            valid=bool(selected),
            errors=[] if selected else ["motion detector returned no interval"],
            duration_sec=total_sec,
            coverage_ratio=temporal_coverage_ratio(selected, total_sec),
            segments_after_normalization=list(selected),
        )


class QwenTemporalDetector:
    """Reliable local Qwen/compatible-VLM temporal detector with bounded fallback."""

    def __init__(
        self,
        model_path: str = "",
        detect_fps: float = 1.0,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 128,
        load_in_4bit: bool = False,
        max_video_frames: int = 64,
        max_frame_pixels: int = 360 * 420,
        total_video_pixels: int = 4096 * 28 * 28,
        vlm_retries: int = 1,
        temporal_fallback: str = "motion",
        short_video_full_sec: float = 15.0,
        segment_padding_sec: float = 3.0,
        min_segment_sec: float = 4.0,
        min_coverage_ratio: float = 0.30,
        max_coverage_sec: float = 30.0,
        max_segments: int = 3,
        fallback_detector=None,
        full_detector=None,
        response_provider: Optional[Callable[[str, str], str]] = None,
    ):
        self.model_path = model_path
        self.detect_fps = max(0.1, float(detect_fps))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.load_in_4bit = bool(load_in_4bit)
        self.max_video_frames = max(2, int(max_video_frames))
        self.max_frame_pixels = max(28 * 28, int(max_frame_pixels))
        self.total_video_pixels = max(self.max_frame_pixels, int(total_video_pixels))
        self.vlm_retries = max(0, int(vlm_retries))
        self.temporal_fallback = str(temporal_fallback or "motion").lower()
        self.short_video_full_sec = max(0.0, float(short_video_full_sec))
        self.segment_padding_sec = max(0.0, float(segment_padding_sec))
        self.min_segment_sec = max(0.0, float(min_segment_sec))
        self.min_coverage_ratio = max(0.0, min(1.0, float(min_coverage_ratio)))
        self.max_coverage_sec = max(0.0, float(max_coverage_sec))
        self.max_segments = max(1, int(max_segments))
        self.fallback_detector = fallback_detector
        self.full_detector = full_detector or FullVideoTemporalDetector()
        self.response_provider = response_provider
        self.model = None
        self.processor = None
        self.model_load_error: Optional[str] = None

        if response_provider is None:
            try:
                self._load_model(device_map=device_map, dtype=dtype)
            except Exception as exc:  # A bad local environment must not bypass fallback.
                self.model = None
                self.processor = None
                self.model_load_error = f"model load failed: {type(exc).__name__}: {exc}"
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

    def _load_model(self, device_map: str, dtype: str) -> None:
        import torch
        from transformers import AutoProcessor

        resolved_dtype = dtype
        if isinstance(dtype, str) and dtype != "auto":
            resolved_dtype = getattr(torch, dtype.replace("torch.", ""), dtype)

        load_kwargs = {
            "torch_dtype": resolved_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = None
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

            model = ModelClass.from_pretrained(self.model_path, **load_kwargs)
        except Exception as exc:
            self._raise_if_resource_error(exc)
            try:
                from transformers import AutoModelForImageTextToText as ModelClass

                model = ModelClass.from_pretrained(self.model_path, **load_kwargs)
            except Exception as exc:
                self._raise_if_resource_error(exc)
                from transformers import AutoModelForVision2Seq as ModelClass

                model = ModelClass.from_pretrained(self.model_path, **load_kwargs)
        processor = AutoProcessor.from_pretrained(self.model_path)
        model.eval()
        self.model = model
        self.processor = processor

    def _video_request(self, video_path: str) -> dict:
        n_frames, fps, _width, _height = video_meta(video_path)
        if n_frames < 2 or fps <= 0:
            raise ValueError("unreadable video")
        duration_sec = n_frames / float(fps)
        requested = max(2, int(round(duration_sec * self.detect_fps)))
        requested = min(requested, self.max_video_frames, n_frames)
        if requested > 2 and requested % 2:
            requested -= 1
        content = {
            "type": "video",
            # qwen-vl-utils accepts local paths directly. A percent-encoded
            # file:// URI is not decoded by the Windows torchcodec backend.
            "video": str(Path(video_path).resolve()),
            "nframes": requested,
            "max_pixels": self.max_frame_pixels,
            "total_pixels": self.total_video_pixels,
        }
        return content

    @staticmethod
    def _normalize_video_kwargs(video_kwargs: Optional[dict]) -> dict:
        normalized = dict(video_kwargs or {})
        fps = normalized.get("fps")
        if isinstance(fps, (list, tuple)) and len(fps) == 1:
            normalized["fps"] = fps[0]
        return normalized

    @staticmethod
    def _transcode_video_for_rgb(video_path: str) -> str:
        source = Path(video_path).resolve()
        stat = source.stat()
        cache_root = Path(
            os.environ.get(
                "QWEN_VIDEO_COMPAT_DIR",
                str(Path(tempfile.gettempdir()) / "qwen_video_compat"),
            )
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        identity = f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:16]
        output = cache_root / f"{source.stem}_{digest}_bt709.mp4"
        if output.exists() and output.stat().st_size > 0:
            return str(output)

        partial = output.with_name(output.stem + ".part.mp4")
        command = [
            "ffmpeg",
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
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            str(partial),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if not partial.exists() or partial.stat().st_size <= 0:
                raise RuntimeError("ffmpeg produced an empty compatibility video")
            os.replace(partial, output)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return str(output)

    @staticmethod
    def _process_vision_inputs(process_vision_info, messages):
        try:
            return process_vision_info(messages, return_video_kwargs=True)
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            return image_inputs, video_inputs, {}

    def _generate_once(self, video_path: str, prompt: str) -> str:
        if self.response_provider is not None:
            return str(self.response_provider(video_path, prompt))
        if self.model_load_error:
            raise RuntimeError(self.model_load_error)
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM model is unavailable")

        messages = [
            {
                "role": "user",
                "content": [
                    self._video_request(video_path),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            from qwen_vl_utils import process_vision_info

            try:
                image_inputs, video_inputs, video_kwargs = self._process_vision_inputs(
                    process_vision_info, messages
                )
            except OSError as exc:
                compatible_video = self._transcode_video_for_rgb(video_path)
                warnings.warn(
                    f"video decoder failed ({exc}); using cached BT.709 copy: {compatible_video}",
                    RuntimeWarning,
                )
                messages[0]["content"][0] = self._video_request(compatible_video)
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
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)
        import torch

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _full_result(
        self,
        video_path: str,
        duration_sec: float,
        errors: List[str],
        attempts: int,
        raw_responses: List[str],
        fallback_used: Optional[str] = "full",
        source: str = "full",
        subject_hint: Optional[str] = None,
    ) -> TemporalResult:
        try:
            result = self.full_detector.detect(video_path)
            segments = list(result.segments_sec)
        except Exception as exc:
            errors.append(f"full fallback raised {type(exc).__name__}: {exc}")
            segments = []
        if not segments:
            segments = [(0.0, duration_sec)]
        return TemporalResult(
            segments,
            subject_hint,
            "full-video",
            source=source,
            valid=True,
            errors=errors,
            attempts=max(0, attempts),
            fallback_used=fallback_used,
            duration_sec=duration_sec,
            coverage_ratio=temporal_coverage_ratio(segments, duration_sec),
            vlm_raw_responses=raw_responses,
            segments_after_normalization=list(segments),
        )

    def _fallback_result(
        self,
        video_path: str,
        duration_sec: float,
        errors: List[str],
        attempts: int,
        raw_responses: List[str],
        best_vlm: Optional[TemporalResult],
    ) -> TemporalResult:
        fallback = self.temporal_fallback
        if fallback == "none":
            if best_vlm is not None:
                best_vlm.errors = errors
                best_vlm.attempts = attempts
                best_vlm.vlm_raw_responses = raw_responses
                return best_vlm
            return TemporalResult(
                [],
                raw=raw_responses[-1] if raw_responses else None,
                source="vlm",
                valid=False,
                errors=errors + ["fallback disabled"],
                attempts=attempts,
                duration_sec=duration_sec,
                coverage_ratio=0.0,
                vlm_raw_responses=raw_responses,
            )

        if fallback == "full":
            return self._full_result(
                video_path,
                duration_sec,
                errors,
                attempts,
                raw_responses,
                fallback_used="full",
                subject_hint=best_vlm.subject_hint if best_vlm else None,
            )

        if fallback == "middle":
            detector = MiddleTemporalDetector(duration_sec=self.min_segment_sec)
        else:
            detector = self.fallback_detector

        fallback_segments: List[SegmentSeconds] = []
        fallback_hint = None
        fallback_raw = None
        if detector is not None:
            try:
                detected = detector.detect(video_path)
                errors.extend(list(getattr(detected, "errors", []) or []))
                if bool(getattr(detected, "valid", bool(detected.segments_sec))) and detected.segments_sec:
                    fallback_segments = list(detected.segments_sec)
                    fallback_hint = detected.subject_hint
                    fallback_raw = detected.raw
                else:
                    errors.append(f"{fallback} fallback returned no valid interval")
            except Exception as exc:
                errors.append(f"{fallback} fallback raised {type(exc).__name__}: {exc}")
        else:
            errors.append(f"{fallback} fallback detector is unavailable")

        if not fallback_segments:
            return self._full_result(
                video_path,
                duration_sec,
                errors,
                attempts,
                raw_responses,
                fallback_used="full",
                subject_hint=best_vlm.subject_hint if best_vlm else None,
            )

        combined = list(best_vlm.segments_sec) if best_vlm is not None else []
        combined.extend(fallback_segments)
        before_for_log = (
            list(best_vlm.segments_before_normalization) if best_vlm is not None else []
        )
        before_for_log.extend(fallback_segments)
        normalized = normalize_temporal_segments(
            combined,
            duration_sec=duration_sec,
            min_segment_sec=self.min_segment_sec,
            padding_sec=0.0,
            max_segments=self.max_segments,
        )
        errors.extend(normalized.errors)
        if normalized.valid:
            target_seconds = _minimum_coverage_seconds(
                duration_sec, self.min_coverage_ratio, self.max_coverage_sec
            )
            expanded = _expand_to_coverage(
                normalized.segments_sec, duration_sec, target_seconds, self.max_segments
            )
            if _covered_seconds(expanded) + 1e-6 >= target_seconds:
                source = fallback if best_vlm is None else f"vlm_{fallback}"
                return TemporalResult(
                    expanded,
                    best_vlm.subject_hint if best_vlm and best_vlm.subject_hint else fallback_hint,
                    fallback_raw,
                    source=source,
                    valid=True,
                    errors=errors,
                    attempts=attempts,
                    fallback_used=fallback,
                    duration_sec=duration_sec,
                    coverage_ratio=temporal_coverage_ratio(expanded, duration_sec),
                    vlm_raw_responses=raw_responses,
                    segments_before_normalization=before_for_log,
                    segments_after_normalization=list(expanded),
                    corrections=list(normalized.corrections),
                )
            errors.append(f"{fallback} fallback coverage remained below configured minimum")
        else:
            errors.append(f"{fallback} fallback returned no valid interval")

        return self._full_result(
            video_path,
            duration_sec,
            errors,
            attempts,
            raw_responses,
            fallback_used="full",
            subject_hint=best_vlm.subject_hint if best_vlm else None,
        )

    def detect(self, video_path: str) -> TemporalResult:
        n_frames, fps, _w, _h = video_meta(video_path)
        if n_frames <= 0 or fps <= 0:
            return TemporalResult(
                [], raw=None, source="vlm", valid=False, errors=["unreadable video"], attempts=0
            )
        duration_sec = n_frames / fps
        if self.short_video_full_sec > 0 and duration_sec <= self.short_video_full_sec:
            result = self._full_result(
                video_path,
                duration_sec,
                errors=[],
                attempts=0,
                raw_responses=[],
                fallback_used=None,
                source="short_video_full",
            )
            result.corrections = ["short video protected by full coverage"]
            return result

        timestamps = sampled_frame_timestamps(
            n_frames, fps, self.detect_fps, max_items=self.max_video_frames
        )
        effective_detect_fps = min(self.detect_fps, len(timestamps) / max(duration_sec, 1e-6))
        base_prompt = build_temporal_prompt(
            duration_sec, fps, effective_detect_fps, self.max_segments, timestamps
        )
        prompt = base_prompt
        raw_responses: List[str] = []
        all_errors: List[str] = []
        best_vlm: Optional[TemporalResult] = None
        target_seconds = _minimum_coverage_seconds(
            duration_sec, self.min_coverage_ratio, self.max_coverage_sec
        )

        total_attempts = self.vlm_retries + 1
        for attempt_index in range(total_attempts):
            try:
                raw = self._generate_once(video_path, prompt)
            except Exception as exc:
                raw = f"<generation-error>{type(exc).__name__}: {exc}</generation-error>"
                attempt_errors = [f"VLM generation failed: {type(exc).__name__}: {exc}"]
                raw_responses.append(raw)
                all_errors.extend(attempt_errors)
                if attempt_index + 1 < total_attempts:
                    prompt = build_temporal_retry_prompt(
                        base_prompt, duration_sec, raw, attempt_errors
                    )
                continue

            raw_responses.append(raw)
            parsed = parse_temporal_response(raw, duration_sec=duration_sec)
            attempt_errors = list(parsed.errors)
            if not parsed.valid:
                all_errors.extend(attempt_errors)
                if attempt_index + 1 < total_attempts:
                    prompt = build_temporal_retry_prompt(
                        base_prompt, duration_sec, raw, attempt_errors
                    )
                continue

            normalized = normalize_temporal_segments(
                parsed.segments_sec,
                duration_sec=duration_sec,
                min_segment_sec=self.min_segment_sec,
                padding_sec=self.segment_padding_sec,
                max_segments=self.max_segments,
            )
            attempt_errors.extend(normalized.errors)
            all_errors.extend(attempt_errors)
            if not normalized.valid:
                if attempt_index + 1 < total_attempts:
                    prompt = build_temporal_retry_prompt(
                        base_prompt, duration_sec, raw, attempt_errors
                    )
                continue

            coverage = temporal_coverage_ratio(normalized.segments_sec, duration_sec)
            candidate = TemporalResult(
                normalized.segments_sec,
                parsed.subject_hint,
                raw,
                source="vlm" if attempt_index == 0 else "vlm_retry",
                valid=True,
                errors=list(all_errors),
                attempts=attempt_index + 1,
                duration_sec=duration_sec,
                coverage_ratio=coverage,
                vlm_raw_responses=list(raw_responses),
                segments_before_normalization=list(parsed.segments_sec),
                segments_after_normalization=list(normalized.segments_sec),
                corrections=list(normalized.corrections),
            )
            if best_vlm is None or _covered_seconds(candidate.segments_sec) > _covered_seconds(
                best_vlm.segments_sec
            ):
                best_vlm = candidate
            if _covered_seconds(normalized.segments_sec) + 1e-6 >= target_seconds:
                return candidate

            coverage_error = (
                f"coverage below minimum target: {_covered_seconds(normalized.segments_sec):.3f}s "
                f"< {target_seconds:.3f}s"
            )
            all_errors.append(coverage_error)
            if attempt_index + 1 < total_attempts:
                prompt = build_temporal_retry_prompt(
                    base_prompt, duration_sec, raw, [coverage_error]
                )

        return self._fallback_result(
            video_path,
            duration_sec,
            all_errors,
            total_attempts,
            raw_responses,
            best_vlm,
        )


def build_temporal_detector(args):
    mode = getattr(args, "temporal_mode", "motion")
    if mode == "full":
        return FullVideoTemporalDetector()
    if mode == "middle":
        return MiddleTemporalDetector(duration_sec=getattr(args, "window_sec", 4.0))
    if mode == "motion":
        return MotionTemporalDetector(
            detect_fps=getattr(args, "detect_fps", 1.0),
            top_k=getattr(args, "top_k_segments", 1),
            window_sec=getattr(args, "window_sec", 4.0),
            min_segment_sec=getattr(
                args, "motion_min_segment_sec", getattr(args, "min_segment_sec", 1.0)
            ),
            padding_sec=getattr(args, "padding_sec", 0.25),
        )
    if mode == "pgl_sum":
        from .pgl_sum_frontend import (
            MrHiSumPGLTemporalDetector,
            load_cached_semantic_hints,
        )

        checkpoint_path = str(getattr(args, "pgl_checkpoint_path", "") or "")
        feature_model_dir = str(getattr(args, "pgl_feature_model_dir", "") or "")
        feature_cache_dir = str(getattr(args, "pgl_feature_cache_dir", "") or "")
        if not checkpoint_path:
            raise ValueError("--pgl-checkpoint-path is required for --temporal-mode pgl_sum")
        if not feature_model_dir:
            raise ValueError("--pgl-feature-model-dir is required for --temporal-mode pgl_sum")
        if not feature_cache_dir:
            raise ValueError("--pgl-feature-cache-dir is required for --temporal-mode pgl_sum")
        hints_path = str(getattr(args, "pgl_hints_jsonl", "") or "")
        return MrHiSumPGLTemporalDetector(
            checkpoint_path=checkpoint_path,
            feature_model_dir=feature_model_dir,
            feature_cache_dir=feature_cache_dir,
            device=getattr(args, "pgl_device", "auto"),
            sample_fps=getattr(args, "pgl_sample_fps", 1.0),
            boundary_grid_sec=getattr(args, "pgl_boundary_grid_sec", 0.25),
            ffmpeg_bin=getattr(args, "window_ffmpeg_path", "ffmpeg"),
            audio_sample_rate=getattr(args, "window_audio_sample_rate", 8000),
            cached_hints=load_cached_semantic_hints(hints_path),
            fallback_detector=MotionTemporalDetector(
                detect_fps=2.0,
                top_k=max(1, int(getattr(args, "top_k_segments", 2))),
                window_sec=max(2.0, float(getattr(args, "window_sec", 4.0))),
                min_segment_sec=max(
                    1.0, float(getattr(args, "motion_min_segment_sec", 1.0))
                ),
                padding_sec=max(0.0, float(getattr(args, "padding_sec", 0.25))),
            ),
            require_e_drive=os.name == "nt",
        )
    if mode == "window_score":
        model_path = getattr(args, "temporal_model_path", "")
        if not model_path:
            raise ValueError(
                "--temporal-model-path is required for --temporal-mode window_score"
            )
        # Import lazily to avoid a circular import: the window frontend returns
        # the TemporalResult class defined in this module.
        from .window_score_frontend import WindowScoreFrontend
        from .window_scoring import QwenWindowScorer

        cache_root = str(
            getattr(
                args,
                "window_cache_root",
                r"E:\AI\tmp\video_highlight\window_score",
            )
        )
        hf_cache = os.environ.get("HF_HOME", r"E:\AI\cache\huggingface")
        scorer = QwenWindowScorer(
            model_path=model_path,
            device_map=getattr(args, "device_map", "auto"),
            dtype=getattr(args, "dtype", "auto"),
            max_new_tokens=getattr(
                args,
                "window_max_new_tokens",
                max(192, int(getattr(args, "max_new_tokens", 128))),
            ),
            load_in_4bit=getattr(args, "vlm_load_in_4bit", False),
            batch_size=getattr(args, "window_batch_size", 1),
            attn_implementation=getattr(args, "window_attn_implementation", None),
            compact_output=getattr(args, "window_compact_output", False),
            max_frame_pixels=getattr(args, "vlm_max_frame_pixels", 360 * 420),
            total_video_pixels=getattr(
                args, "vlm_total_video_pixels", 4096 * 28 * 28
            ),
            max_retries=getattr(args, "window_score_retries", 1),
            cache_dir=hf_cache,
            temp_dir=cache_root,
            compat_dir=os.path.join(cache_root, "qwen_video_compat"),
            ffmpeg_bin=getattr(args, "window_ffmpeg_path", "ffmpeg"),
            require_e_drive=os.name == "nt",
        )
        semantic_mode = str(
            getattr(args, "window_semantic_mode", "structured") or "structured"
        )
        aggregation = str(
            getattr(args, "window_aggregation", "current") or "current"
        )
        if semantic_mode == "reze_yesno":
            from .reze_frontend import RezeTemporalFrontend

            return RezeTemporalFrontend(
                scorer,
                aggregation=aggregation,
                fallback_detector=MotionTemporalDetector(
                    detect_fps=2.0,
                    top_k=max(1, int(getattr(args, "top_k_segments", 3))),
                    window_sec=max(2.0, float(getattr(args, "window_sec", 4.0))),
                    min_segment_sec=max(
                        1.0, float(getattr(args, "motion_min_segment_sec", 1.0))
                    ),
                    padding_sec=max(0.0, float(getattr(args, "padding_sec", 0.25))),
                ),
                short_full_sec=getattr(args, "window_short_full_sec", 4.0),
                current_seed_threshold=getattr(
                    args, "window_absolute_threshold", 0.45
                ),
                current_growth_threshold=getattr(
                    args, "window_growth_threshold", 0.40
                ),
                merge_gap_sec=getattr(args, "window_merge_gap_sec", 1.0),
                boundary_expand_sec=getattr(
                    args, "window_boundary_expand_sec", 0.75
                ),
                max_boundary_expand_sec=getattr(
                    args, "window_max_boundary_expand_sec", 1.5
                ),
                max_segments=getattr(args, "window_max_segments", 12),
                max_coverage_ratio=getattr(
                    args, "window_max_coverage_ratio", 0.60
                ),
                max_coverage_sec=getattr(args, "window_max_coverage_sec", 30.0),
            )
        if semantic_mode == "global_event":
            if aggregation != "current":
                scorer.close()
                raise ValueError(
                    "global_event semantic mode requires --window-aggregation current"
                )
            from .global_event_frontend import GlobalEventFrontend

            structured_fallback = WindowScoreFrontend(
                scorer,
                coarse_window_sec=getattr(args, "window_coarse_sec", 2.0),
                coarse_stride_sec=getattr(args, "window_coarse_stride_sec", 1.0),
                coarse_fps=getattr(args, "window_coarse_fps", 2.0),
                fine_window_sec=getattr(args, "window_fine_sec", 2.0),
                fine_stride_sec=getattr(args, "window_fine_stride_sec", 0.5),
                fine_fps=getattr(args, "window_fine_fps", 4.0),
                max_fine_regions=getattr(args, "window_max_fine_regions", 6),
                curve_step_sec=getattr(args, "window_curve_step_sec", 0.5),
                smoothing_points=getattr(args, "window_smoothing_points", 3),
                relative_top_fraction=getattr(args, "window_relative_top_fraction", 0.30),
                absolute_threshold=getattr(args, "window_absolute_threshold", 0.45),
                growth_threshold=getattr(args, "window_growth_threshold", 0.40),
                coarse_candidate_threshold=getattr(
                    args, "window_coarse_candidate_threshold", 0.45
                ),
                merge_gap_sec=getattr(args, "window_merge_gap_sec", 1.0),
                boundary_expand_sec=getattr(args, "window_boundary_expand_sec", 0.75),
                max_boundary_expand_sec=getattr(
                    args, "window_max_boundary_expand_sec", 1.5
                ),
                max_segments=getattr(args, "window_max_segments", 12),
                max_coverage_ratio=getattr(args, "window_max_coverage_ratio", 0.60),
                max_coverage_sec=getattr(args, "window_max_coverage_sec", 30.0),
                short_full_sec=getattr(args, "window_short_full_sec", 4.0),
                ffmpeg_bin=getattr(args, "window_ffmpeg_path", "ffmpeg"),
                audio_sample_rate=getattr(args, "window_audio_sample_rate", 8000),
            )
            return GlobalEventFrontend(
                scorer,
                fallback_detector=structured_fallback,
                window_sec=getattr(args, "window_coarse_sec", 2.0),
                stride_sec=getattr(args, "window_coarse_stride_sec", 1.0),
                sample_fps=getattr(args, "window_coarse_fps", 2.0),
                grid_sec=getattr(args, "window_curve_step_sec", 0.5),
                smoothing_points=getattr(args, "window_smoothing_points", 3),
                max_events=getattr(args, "global_event_max_events", 3),
                long_video_sec=getattr(args, "global_event_long_video_sec", 60.0),
                overview_chunk_sec=getattr(
                    args, "global_event_overview_chunk_sec", 30.0
                ),
                overview_max_frames=getattr(
                    args, "global_event_overview_max_frames", 64
                ),
                seed_threshold=getattr(args, "global_event_seed_threshold", 0.55),
                growth_threshold=getattr(
                    args, "global_event_growth_threshold", 0.40
                ),
                valley_threshold=getattr(args, "global_event_valley_threshold", 0.35),
                min_prominence=getattr(args, "global_event_min_prominence", 0.10),
                padding_sec=getattr(args, "global_event_padding_sec", 0.50),
                merge_gap_sec=getattr(args, "global_event_merge_gap_sec", 0.25),
                max_coverage_ratio=getattr(
                    args, "global_event_max_coverage_ratio", 0.75
                ),
                ffmpeg_bin=getattr(args, "window_ffmpeg_path", "ffmpeg"),
                audio_sample_rate=getattr(args, "window_audio_sample_rate", 8000),
            )
        if semantic_mode != "structured":
            scorer.close()
            raise ValueError(f"unsupported window semantic mode: {semantic_mode}")
        if aggregation != "current":
            scorer.close()
            raise ValueError(
                "structured window scoring currently requires --window-aggregation current"
            )
        return WindowScoreFrontend(
            scorer,
            coarse_window_sec=getattr(args, "window_coarse_sec", 2.0),
            coarse_stride_sec=getattr(args, "window_coarse_stride_sec", 1.0),
            coarse_fps=getattr(args, "window_coarse_fps", 2.0),
            fine_window_sec=getattr(args, "window_fine_sec", 2.0),
            fine_stride_sec=getattr(args, "window_fine_stride_sec", 0.5),
            fine_fps=getattr(args, "window_fine_fps", 4.0),
            max_fine_regions=getattr(args, "window_max_fine_regions", 6),
            curve_step_sec=getattr(args, "window_curve_step_sec", 0.5),
            smoothing_points=getattr(args, "window_smoothing_points", 3),
            relative_top_fraction=getattr(
                args, "window_relative_top_fraction", 0.30
            ),
            absolute_threshold=getattr(args, "window_absolute_threshold", 0.45),
            growth_threshold=getattr(args, "window_growth_threshold", 0.40),
            coarse_candidate_threshold=getattr(
                args, "window_coarse_candidate_threshold", 0.45
            ),
            merge_gap_sec=getattr(args, "window_merge_gap_sec", 1.0),
            boundary_expand_sec=getattr(
                args, "window_boundary_expand_sec", 0.75
            ),
            max_boundary_expand_sec=getattr(
                args, "window_max_boundary_expand_sec", 1.5
            ),
            max_segments=getattr(args, "window_max_segments", 12),
            max_coverage_ratio=getattr(args, "window_max_coverage_ratio", 0.60),
            max_coverage_sec=getattr(args, "window_max_coverage_sec", 30.0),
            short_full_sec=getattr(args, "window_short_full_sec", 4.0),
            ffmpeg_bin=getattr(args, "window_ffmpeg_path", "ffmpeg"),
            audio_sample_rate=getattr(args, "window_audio_sample_rate", 8000),
        )
    if mode == "vlm":
        model_path = getattr(args, "temporal_model_path", "")
        if not model_path:
            raise ValueError("--temporal-model-path is required for --temporal-mode vlm")
        fallback_detector = MotionTemporalDetector(
            detect_fps=getattr(args, "detect_fps", 1.0),
            top_k=getattr(args, "top_k_segments", 1),
            window_sec=getattr(args, "window_sec", 4.0),
            min_segment_sec=getattr(args, "motion_min_segment_sec", 1.0),
            padding_sec=getattr(args, "padding_sec", 0.25),
        )
        return QwenTemporalDetector(
            model_path=model_path,
            detect_fps=getattr(args, "detect_fps", 1.0),
            device_map=getattr(args, "device_map", "auto"),
            dtype=getattr(args, "dtype", "auto"),
            max_new_tokens=getattr(args, "max_new_tokens", 128),
            load_in_4bit=getattr(args, "vlm_load_in_4bit", False),
            max_video_frames=getattr(args, "vlm_max_video_frames", 64),
            max_frame_pixels=getattr(args, "vlm_max_frame_pixels", 360 * 420),
            total_video_pixels=getattr(args, "vlm_total_video_pixels", 4096 * 28 * 28),
            vlm_retries=getattr(args, "vlm_retries", 1),
            temporal_fallback=getattr(args, "temporal_fallback", "motion"),
            short_video_full_sec=getattr(args, "short_video_full_sec", 15.0),
            segment_padding_sec=getattr(args, "segment_padding_sec", 3.0),
            min_segment_sec=getattr(args, "min_segment_sec", 4.0),
            min_coverage_ratio=getattr(args, "min_coverage_ratio", 0.30),
            max_coverage_sec=getattr(args, "max_coverage_sec", 30.0),
            max_segments=getattr(args, "max_temporal_segments", 3),
            fallback_detector=fallback_detector,
        )
    raise ValueError(f"Unknown temporal mode: {mode}")
