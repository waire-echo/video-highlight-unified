"""Opt-in REZE-inspired temporal detector.

This module deliberately stays separate from the stable structured
``window_score`` frontend.  It reads a conditional Yes/No confidence for fixed
three-second clips, aggregates that curve, and falls back to the project's
traditional temporal detector when the semantic curve is missing or
degenerate.  It never imports the spatial/YOLO pipeline.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io_utils import video_meta
from .reze_aggregation import aggregate_reze_otsu, build_reze_curve
from .reze_scoring import RezeWindowScorer
from .score_curve import select_highlight_segments, smooth_score_curve
from .temporal import TemporalResult


Segment = Tuple[float, float]


def _coverage_ratio(segments: Sequence[Segment], duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return max(
        0.0,
        min(1.0, sum(max(0.0, end - start) for start, end in segments) / duration_sec),
    )


def _middle_or_full(duration_sec: float, short_full_sec: float = 4.0) -> List[Segment]:
    duration = max(0.0, float(duration_sec))
    if duration <= 0:
        return []
    if duration <= max(0.0, float(short_full_sec)):
        return [(0.0, duration)]
    width = min(4.0, duration)
    start = max(0.0, (duration - width) / 2.0)
    return [(start, min(duration, start + width))]


class RezeTemporalFrontend:
    """Return :class:`TemporalResult` from REZE Yes/No window confidence."""

    def __init__(
        self,
        qwen_scorer,
        *,
        aggregation: str = "reze_otsu",
        fallback_detector=None,
        short_full_sec: float = 4.0,
        current_seed_threshold: float = 0.45,
        current_growth_threshold: float = 0.40,
        merge_gap_sec: float = 1.0,
        boundary_expand_sec: float = 0.75,
        max_boundary_expand_sec: float = 1.5,
        max_segments: int = 12,
        max_coverage_ratio: float = 0.60,
        max_coverage_sec: float = 30.0,
    ):
        if aggregation not in {"current", "reze_otsu"}:
            raise ValueError(f"unsupported REZE aggregation: {aggregation}")
        self.qwen_scorer = qwen_scorer
        self.scorer = RezeWindowScorer(qwen_scorer)
        self.aggregation = aggregation
        self.fallback_detector = fallback_detector
        self.short_full_sec = float(short_full_sec)
        self.current_seed_threshold = float(current_seed_threshold)
        self.current_growth_threshold = float(current_growth_threshold)
        self.merge_gap_sec = float(merge_gap_sec)
        self.boundary_expand_sec = float(boundary_expand_sec)
        self.max_boundary_expand_sec = float(max_boundary_expand_sec)
        self.max_segments = int(max_segments)
        self.max_coverage_ratio = float(max_coverage_ratio)
        self.max_coverage_sec = float(max_coverage_sec)

    @staticmethod
    def _current_curve(records: Iterable[Any], duration_sec: float) -> List[Dict[str, Any]]:
        curve = build_reze_curve(records, duration_sec, grid_sec=1.0)
        compatible: List[Dict[str, Any]] = []
        for row in curve:
            value = row.get("p_yes")
            compatible.append(
                {
                    **dict(row),
                    "start_sec": row.get("grid_start_sec", row.get("start_sec")),
                    "end_sec": row.get("grid_end_sec", row.get("end_sec")),
                    "fused_score": value,
                    "score": value,
                }
            )
        return smooth_score_curve(compatible, window_points=3)

    def _aggregate(self, records: Sequence[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
        if self.aggregation == "reze_otsu":
            result = aggregate_reze_otsu(
                records,
                duration_sec,
                max_segments=self.max_segments,
                coverage_limit_sec=min(
                    self.max_coverage_sec,
                    max(0.0, float(duration_sec)) * self.max_coverage_ratio,
                ),
            )
            return {
                "segments_sec": list(result.segments_sec),
                "score_curve": list(result.score_curve),
                "selected_peaks": list(result.selected_candidates),
                "diagnostics": dict(result.diagnostics),
            }

        curve = self._current_curve(records, duration_sec)
        selected = select_highlight_segments(
            curve,
            duration_sec,
            score_threshold=self.current_seed_threshold,
            growth_threshold=self.current_growth_threshold,
            use_local_peak_seeds=True,
            min_peak_prominence=0.10,
            min_segment_sec=1.0,
            merge_gap_sec=self.merge_gap_sec,
            padding_sec=self.boundary_expand_sec,
            max_padding_sec=self.max_boundary_expand_sec,
            max_segments=self.max_segments,
            max_coverage_sec=self.max_coverage_sec,
            max_coverage_ratio=self.max_coverage_ratio,
        )
        return {
            "segments_sec": list(selected.get("segments_sec") or []),
            "score_curve": curve,
            "selected_peaks": list(selected.get("selected_peaks") or []),
            "diagnostics": dict(selected.get("score_diagnostics") or {}),
        }

    def _fallback(self, video_path: str, duration_sec: float) -> Tuple[List[Segment], str, List[str]]:
        errors: List[str] = []
        if self.fallback_detector is not None:
            try:
                result = self.fallback_detector.detect(video_path)
                segments = [
                    (float(start), float(end))
                    for start, end in (getattr(result, "segments_sec", None) or [])
                    if math.isfinite(float(start))
                    and math.isfinite(float(end))
                    and float(end) > float(start)
                ]
                if segments:
                    source = str(getattr(result, "source", "motion") or "motion")
                    return segments, source, list(getattr(result, "errors", []) or [])
                errors.extend(list(getattr(result, "errors", []) or []))
            except Exception as exc:  # one fallback must not abort the batch
                errors.append(f"traditional fallback failed: {type(exc).__name__}: {exc}")
        segments = _middle_or_full(duration_sec, self.short_full_sec)
        source = "short_video_full" if duration_sec <= self.short_full_sec else "middle"
        return segments, source, errors

    def detect(self, video_path: str) -> TemporalResult:
        frame_count, fps, _width, _height = video_meta(video_path)
        if frame_count <= 0 or fps <= 0:
            return TemporalResult(
                [],
                raw="reze_yesno",
                source="reze_yesno_input_error",
                valid=False,
                errors=["unreadable video"],
                attempts=0,
            )
        duration_sec = frame_count / float(fps)
        scored = self.scorer.score_video(video_path, duration_sec=duration_sec)
        records = [record.to_dict() for record in scored.records]
        aggregated = self._aggregate(records, duration_sec)
        segments = [tuple(map(float, item)) for item in aggregated["segments_sec"]]
        fallback_used: Optional[str] = None
        fallback_errors: List[str] = []
        if segments:
            source = f"reze_yesno_{self.aggregation}"
        else:
            segments, fallback_used, fallback_errors = self._fallback(video_path, duration_sec)
            source = f"reze_yesno_{fallback_used}_fallback"

        errors = [error for record in scored.records for error in record.errors]
        errors.extend(fallback_errors)
        diagnostics = {
            "semantic_mode": "reze_yesno",
            "aggregation": self.aggregation,
            "scoring": dict(scored.diagnostics),
            "aggregation_diagnostics": dict(aggregated["diagnostics"]),
            "fallback_used": fallback_used,
        }
        attempts = sum(int(record.attempts or 0) for record in scored.records)
        return TemporalResult(
            segments,
            subject_hint=None,
            raw=json.dumps(
                {
                    "mode": "reze_yesno",
                    "aggregation": self.aggregation,
                    "window_count": len(records),
                },
                ensure_ascii=False,
            ),
            source=source,
            valid=bool(segments),
            errors=errors,
            attempts=attempts,
            fallback_used=fallback_used,
            duration_sec=duration_sec,
            coverage_ratio=_coverage_ratio(segments, duration_sec),
            segments_before_normalization=list(segments),
            segments_after_normalization=list(segments),
            corrections=["REZE-inspired Yes/No confidence aggregated outside the VLM"],
            window_records=records,
            score_curve=list(aggregated["score_curve"]),
            selected_peaks=list(aggregated["selected_peaks"]),
            score_diagnostics=diagnostics,
            action_description=None,
        )

    def close(self) -> None:
        close = getattr(self.qwen_scorer, "close", None)
        if callable(close):
            close()


__all__ = ["RezeTemporalFrontend"]
