"""Deterministic Otsu aggregation for REZE-inspired temporal scores.

The input is a sequence of short-window ``P(Yes)`` records.  They are mapped
to a one-second curve, smoothed with a masked NumPy Gaussian (sigma=3 by
default), thresholded per video with a 256-bin Otsu rule and converted to
multiple temporal segments.  This module intentionally returns an empty
segment list when semantic evidence is missing or degenerate; the outer
temporal frontend owns the traditional-signal fallback policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Segment = Tuple[float, float]
DEFAULT_GRID_SEC = 1.0
DEFAULT_GAUSSIAN_SIGMA = 3.0
DEFAULT_OTSU_BINS = 256
DEFAULT_MIN_SEGMENT_SEC = 2.0
DEFAULT_MAX_SEGMENTS = 12
DEFAULT_DEGENERATE_RANGE = 0.05


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _record_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record):
        try:
            return dict(asdict(record))
        except (TypeError, ValueError):
            return {}
    try:
        return dict(vars(record))
    except (TypeError, ValueError):
        return {}


def _record_probability(row: Mapping[str, Any]) -> Optional[float]:
    if "valid" in row and not bool(row.get("valid")):
        return None
    for key in ("p_yes", "score", "semantic_score"):
        if key not in row:
            continue
        value = _finite(row.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value
        return None
    return None


def build_reze_curve(
    records: Iterable[Any],
    duration_sec: float,
    *,
    grid_sec: float = DEFAULT_GRID_SEC,
) -> List[Dict[str, Any]]:
    """Map window probabilities to overlap-weighted one-second bins."""

    duration = _finite(duration_sec) or 0.0
    step = _finite(grid_sec) or 0.0
    if duration <= 0 or step <= 0:
        return []
    usable: List[Tuple[float, float, float]] = []
    for record in records:
        row = _record_dict(record)
        start = _finite(row.get("start_sec"))
        end = _finite(row.get("end_sec"))
        probability = _record_probability(row)
        if start is None or end is None or probability is None:
            continue
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end > start + 1e-12:
            usable.append((start, end, probability))

    count = max(1, int(math.ceil(duration / step - 1e-12)))
    curve: List[Dict[str, Any]] = []
    for index in range(count):
        start = index * step
        end = min(duration, start + step)
        weighted_sum = 0.0
        overlap_sum = 0.0
        observations = 0
        for window_start, window_end, probability in usable:
            overlap = min(end, window_end) - max(start, window_start)
            if overlap <= 1e-12:
                continue
            weighted_sum += probability * overlap
            overlap_sum += overlap
            observations += 1
        value = weighted_sum / overlap_sum if overlap_sum > 0 else None
        curve.append(
            {
                "index": index,
                "start_sec": float(start),
                "end_sec": float(end),
                "time_sec": float((start + end) / 2.0),
                "p_yes": value,
                "smoothed_p_yes": None,
                "observations": observations,
                "valid": value is not None,
            }
        )
    return curve


def _same_convolution(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return a NumPy convolution with exactly ``len(values)`` outputs."""

    if values.size == 0:
        return values.copy()
    radius = kernel.size // 2
    full = np.convolve(values, kernel, mode="full")
    return full[radius : radius + values.size]


def gaussian_smooth_values(
    values: Sequence[Optional[float]],
    *,
    sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> List[Optional[float]]:
    """Gaussian-smooth finite values while preserving missing bins.

    A mask-normalised convolution avoids treating missing probabilities as
    zeros.  The original missing positions remain ``None`` after smoothing so
    a failed model window cannot be bridged into a selected highlight.
    """

    sigma_value = _finite(sigma)
    if sigma_value is None or sigma_value < 0:
        raise ValueError("sigma must be a finite non-negative number")
    parsed: List[Optional[float]] = []
    for value in values:
        number = _finite(value)
        parsed.append(number if number is not None and 0.0 <= number <= 1.0 else None)
    if not parsed or sigma_value == 0:
        return parsed

    radius = max(1, int(math.ceil(3.0 * sigma_value)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_value) ** 2)
    kernel /= float(kernel.sum())
    mask = np.asarray([value is not None for value in parsed], dtype=np.float64)
    data = np.asarray([0.0 if value is None else value for value in parsed], dtype=np.float64)
    numerator = _same_convolution(data, kernel)
    denominator = _same_convolution(mask, kernel)
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    return [
        float(smoothed[index]) if parsed[index] is not None and denominator[index] > 1e-12 else None
        for index in range(len(parsed))
    ]


def smooth_reze_curve(
    curve: Sequence[Mapping[str, Any]],
    *,
    sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> List[Dict[str, Any]]:
    """Copy a curve and attach ``smoothed_p_yes`` values."""

    result = [dict(row) for row in curve]
    smoothed = gaussian_smooth_values(
        [_finite(row.get("p_yes")) for row in result], sigma=sigma
    )
    for row, value in zip(result, smoothed):
        row["smoothed_p_yes"] = value
    return result


def otsu_threshold(
    values: Sequence[Optional[float]],
    *,
    bins: int = DEFAULT_OTSU_BINS,
) -> Optional[float]:
    """Compute a deterministic Otsu threshold on finite probabilities."""

    bin_count = int(bins)
    if bin_count < 2:
        raise ValueError("bins must be at least 2")
    finite_values = [
        number
        for value in values
        if (number := _finite(value)) is not None and 0.0 <= number <= 1.0
    ]
    if len(finite_values) < 2 or max(finite_values) - min(finite_values) <= 1e-12:
        return None
    histogram, edges = np.histogram(
        np.asarray(finite_values, dtype=np.float64),
        bins=bin_count,
        range=(0.0, 1.0),
    )
    histogram = histogram.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(histogram)
    weighted_low = np.cumsum(histogram * centres)
    total_weight = float(weight_low[-1])
    total_weighted = float(weighted_low[-1])
    weight_high = total_weight - weight_low
    valid_split = (weight_low > 0) & (weight_high > 0)
    if not np.any(valid_split):
        return None
    mean_low = np.divide(
        weighted_low,
        weight_low,
        out=np.zeros_like(weighted_low),
        where=weight_low > 0,
    )
    mean_high = np.divide(
        total_weighted - weighted_low,
        weight_high,
        out=np.zeros_like(weighted_low),
        where=weight_high > 0,
    )
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~valid_split] = -1.0
    split_index = int(np.argmax(between))
    # The split lies between this histogram bin and the next one.  Returning
    # the edge (and selecting strictly greater values) prevents the low class
    # itself from being included when several empty bins tie for the optimum.
    threshold = float(edges[split_index + 1])
    return threshold if math.isfinite(threshold) else None


def _curve_score(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("smoothed_p_yes", "p_yes", "score"):
        if key in row:
            value = _finite(row.get(key))
            if value is not None and 0.0 <= value <= 1.0:
                return value
    return None


def _candidate_from_rows(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    start = _finite(rows[0].get("start_sec"))
    end = _finite(rows[-1].get("end_sec"))
    if start is None or end is None or end <= start:
        return None
    area = 0.0
    peak = 0.0
    for row in rows:
        row_start = _finite(row.get("start_sec"))
        row_end = _finite(row.get("end_sec"))
        score = _curve_score(row)
        if row_start is None or row_end is None or score is None:
            continue
        duration = max(0.0, row_end - row_start)
        area += max(0.0, score - threshold) * duration
        peak = max(peak, score)
    return {
        "start_sec": float(start),
        "end_sec": float(end),
        "duration_sec": float(end - start),
        "excess_area": float(area),
        "peak_score": float(peak),
        "curve_rows": [dict(row) for row in rows],
    }


def extract_otsu_candidates(
    curve: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
) -> List[Dict[str, Any]]:
    """Extract contiguous strictly-above-threshold candidate intervals."""

    threshold_value = _finite(threshold)
    minimum = _finite(min_segment_sec)
    if threshold_value is None:
        return []
    if minimum is None or minimum < 0:
        raise ValueError("min_segment_sec must be non-negative")
    ordered = sorted(
        (dict(row) for row in curve),
        key=lambda row: (_finite(row.get("start_sec")) or 0.0),
    )
    candidates: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_end: Optional[float] = None

    def flush() -> None:
        nonlocal current, current_end
        candidate = _candidate_from_rows(current, threshold_value)
        if candidate is not None and candidate["duration_sec"] + 1e-9 >= minimum:
            candidates.append(candidate)
        current = []
        current_end = None

    for row in ordered:
        start = _finite(row.get("start_sec"))
        end = _finite(row.get("end_sec"))
        score = _curve_score(row)
        above = (
            start is not None
            and end is not None
            and end > start
            and score is not None
            and score > threshold_value
        )
        contiguous = current_end is None or (
            start is not None and start <= current_end + 1e-9
        )
        if not above or not contiguous:
            flush()
            if not above:
                continue
        current.append(row)
        current_end = end
    flush()
    return candidates


def _best_trimmed_candidate(
    candidate: Mapping[str, Any],
    threshold: float,
    max_duration_sec: float,
    min_duration_sec: float,
) -> Optional[Dict[str, Any]]:
    rows = [dict(row) for row in candidate.get("curve_rows", [])]
    best: Optional[Dict[str, Any]] = None
    for left in range(len(rows)):
        for right in range(left + 1, len(rows) + 1):
            subset = rows[left:right]
            proposal = _candidate_from_rows(subset, threshold)
            if proposal is None:
                continue
            duration = proposal["duration_sec"]
            if duration > max_duration_sec + 1e-9:
                break
            if duration + 1e-9 < min_duration_sec:
                continue
            if best is None:
                best = proposal
                continue
            key = (
                proposal["excess_area"],
                proposal["peak_score"],
                -proposal["start_sec"],
            )
            best_key = (
                best["excess_area"],
                best["peak_score"],
                -best["start_sec"],
            )
            if key > best_key:
                best = proposal
    return best


def select_otsu_segments(
    curve: Sequence[Mapping[str, Any]],
    duration_sec: float,
    threshold: float,
    *,
    min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    coverage_limit_sec: Optional[float] = None,
) -> Tuple[List[Segment], List[Dict[str, Any]], Dict[str, Any]]:
    """Rank candidates by excess area and enforce count/coverage guards."""

    duration = max(0.0, _finite(duration_sec) or 0.0)
    minimum = _finite(min_segment_sec)
    if minimum is None or minimum < 0:
        raise ValueError("min_segment_sec must be non-negative")
    hard_limit = min(30.0, duration * 0.60)
    if coverage_limit_sec is not None:
        requested_limit = _finite(coverage_limit_sec)
        if requested_limit is None or requested_limit < 0:
            raise ValueError("coverage_limit_sec must be non-negative")
        hard_limit = min(hard_limit, requested_limit)
    segment_limit = min(DEFAULT_MAX_SEGMENTS, max(0, int(max_segments)))
    candidates = extract_otsu_candidates(
        curve, threshold, min_segment_sec=minimum
    )
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["excess_area"],
            -candidate["peak_score"],
            candidate["start_sec"],
        ),
    )
    selected: List[Dict[str, Any]] = []
    coverage = 0.0
    for rank, candidate in enumerate(ranked, start=1):
        if len(selected) >= segment_limit:
            break
        remaining = hard_limit - coverage
        if remaining + 1e-9 < minimum:
            break
        proposal = candidate
        if candidate["duration_sec"] > remaining + 1e-9:
            proposal = _best_trimmed_candidate(
                candidate,
                float(threshold),
                remaining,
                minimum,
            )
            if proposal is None:
                continue
        proposal = dict(proposal)
        proposal["area_rank"] = rank
        proposal.pop("curve_rows", None)
        selected.append(proposal)
        coverage += proposal["duration_sec"]
    selected.sort(key=lambda candidate: candidate["start_sec"])
    segments = [
        (float(candidate["start_sec"]), float(candidate["end_sec"]))
        for candidate in selected
    ]
    diagnostics = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "coverage_limit_sec": float(hard_limit),
        "selected_coverage_sec": float(sum(end - start for start, end in segments)),
        "max_segments": segment_limit,
        "min_segment_sec": float(minimum),
    }
    return segments, selected, diagnostics


@dataclass
class RezeAggregationResult:
    segments_sec: List[Segment] = field(default_factory=list)
    score_curve: List[Dict[str, Any]] = field(default_factory=list)
    threshold: Optional[float] = None
    selected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    source: str = "reze_otsu"

    @property
    def fallback_required(self) -> bool:
        return not self.segments_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "segments_sec": [list(segment) for segment in self.segments_sec],
            "score_curve": [dict(row) for row in self.score_curve],
            "threshold": self.threshold,
            "selected_candidates": [dict(row) for row in self.selected_candidates],
            "score_diagnostics": dict(self.diagnostics),
            "fallback_required": self.fallback_required,
        }


def aggregate_reze_otsu(
    records: Iterable[Any],
    duration_sec: float,
    *,
    grid_sec: float = DEFAULT_GRID_SEC,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    otsu_bins: int = DEFAULT_OTSU_BINS,
    min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    coverage_limit_sec: Optional[float] = None,
    degenerate_range: float = DEFAULT_DEGENERATE_RANGE,
) -> RezeAggregationResult:
    """Run the complete REZE Otsu aggregation without applying a fallback."""

    rows = list(records)
    duration = max(0.0, _finite(duration_sec) or 0.0)
    curve = build_reze_curve(rows, duration, grid_sec=grid_sec)
    curve = smooth_reze_curve(curve, sigma=gaussian_sigma)
    raw_values = [
        value
        for row in curve
        if (value := _finite(row.get("p_yes"))) is not None
    ]
    raw_range = max(raw_values) - min(raw_values) if raw_values else None
    epsilon = max(0.0, _finite(degenerate_range) or 0.0)
    diagnostics: Dict[str, Any] = {
        "aggregation": "reze_otsu",
        "grid_sec": float(grid_sec),
        "gaussian_sigma": float(gaussian_sigma),
        "otsu_bins": int(otsu_bins),
        "valid_curve_bins": len(raw_values),
        "missing_curve_bins": len(curve) - len(raw_values),
        "p_yes_min": min(raw_values) if raw_values else None,
        "p_yes_max": max(raw_values) if raw_values else None,
        "p_yes_range": raw_range,
        "degenerate_range_threshold": epsilon,
    }
    if not raw_values:
        diagnostics.update(status="missing", reason="no_valid_probabilities")
        return RezeAggregationResult([], curve, None, [], diagnostics)
    if raw_range is None or raw_range <= epsilon:
        diagnostics.update(status="degenerate", reason="probability_range_too_small")
        return RezeAggregationResult([], curve, None, [], diagnostics)

    threshold = otsu_threshold(
        [row.get("smoothed_p_yes") for row in curve], bins=otsu_bins
    )
    if threshold is None:
        diagnostics.update(status="degenerate", reason="otsu_threshold_unavailable")
        return RezeAggregationResult([], curve, None, [], diagnostics)
    segments, selected, selection_diagnostics = select_otsu_segments(
        curve,
        duration,
        threshold,
        min_segment_sec=min_segment_sec,
        max_segments=max_segments,
        coverage_limit_sec=coverage_limit_sec,
    )
    diagnostics.update(selection_diagnostics)
    diagnostics["otsu_threshold"] = threshold
    diagnostics["status"] = "ok" if segments else "no_candidates"
    diagnostics["reason"] = None if segments else "no_segment_survived_guards"
    return RezeAggregationResult(segments, curve, threshold, selected, diagnostics)


__all__ = [
    "DEFAULT_DEGENERATE_RANGE",
    "DEFAULT_GAUSSIAN_SIGMA",
    "DEFAULT_GRID_SEC",
    "DEFAULT_MAX_SEGMENTS",
    "DEFAULT_MIN_SEGMENT_SEC",
    "DEFAULT_OTSU_BINS",
    "RezeAggregationResult",
    "aggregate_reze_otsu",
    "build_reze_curve",
    "extract_otsu_candidates",
    "gaussian_smooth_values",
    "otsu_threshold",
    "select_otsu_segments",
    "smooth_reze_curve",
]
