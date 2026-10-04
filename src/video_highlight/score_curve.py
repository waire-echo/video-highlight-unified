"""Pure helpers for turning window scores into temporal highlight segments.

The experimental ``window_score`` detector produces overlapping two-second
windows.  This module maps those windows to a regular 0.5-second curve, fills
small gaps, smooths the curve, and extracts bounded highlight intervals.  It has
no dependency on ``TemporalResult`` or the YOLO/spatial pipeline, making the
functions easy to test and reuse from a runner.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .temporal_signals import (
    TRADITIONAL_SIGNAL_KEYS,
    fuse_window_scores,
    record_to_dict,
)


Segment = Tuple[float, float]


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _interval(record: Mapping[str, Any]) -> Optional[Segment]:
    start = None
    end = None
    for key in ("start_sec", "window_start_sec", "grid_start_sec", "start", "t_start"):
        if key in record:
            start = _finite_number(record.get(key))
            break
    for key in ("end_sec", "window_end_sec", "grid_end_sec", "end", "t_end"):
        if key in record:
            end = _finite_number(record.get(key))
            break
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return (start, end)


def _score(record: Mapping[str, Any], score_key: str = "fused_score") -> Optional[float]:
    candidates = (score_key, "score", "smoothed_score", "fusion_score", "highlight_score")
    for key in candidates:
        if key in record:
            value = _finite_number(record.get(key))
            if value is not None:
                return max(0.0, min(1.0, value))
    return None


def _safe_duration(duration_sec: Any) -> float:
    duration = _finite_number(duration_sec)
    return max(0.0, duration or 0.0)


def _fill_curve_values(
    values: Sequence[Optional[float]],
    *,
    max_gap_bins: int = 2,
) -> Tuple[List[Optional[float]], List[bool]]:
    """Conservatively fill only short missing runs.

    Unlimited nearest-neighbour filling lets a single valid half-second bin
    spread across an otherwise missing ten-second video.  The intended use is
    repairing small decoder/model holes, so runs longer than two grid bins are
    left missing and handled by the normal fallback.
    """

    result = list(values)
    filled = [False] * len(result)
    limit = max(0, int(max_gap_bins))
    if limit <= 0:
        return result, filled
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        run_start = index
        while index < len(result) and result[index] is None:
            index += 1
        run_end = index
        if run_end - run_start > limit:
            continue
        left = run_start - 1 if run_start > 0 and result[run_start - 1] is not None else None
        right = run_end if run_end < len(result) and result[run_end] is not None else None
        if left is not None and right is not None:
            value = min(float(result[left]), float(result[right]))
        elif left is not None:
            value = float(result[left])
        elif right is not None:
            value = float(result[right])
        else:
            continue
        for fill_index in range(run_start, run_end):
            result[fill_index] = value
            filled[fill_index] = True
    return result, filled


def build_score_curve(
    records: Iterable[Any],
    duration_sec: float,
    *,
    grid_sec: float = 0.5,
    score_key: str = "fused_score",
    fill_missing: bool = True,
    max_fill_gap_bins: int = 2,
) -> List[Dict[str, Any]]:
    """Average overlapping window scores on a regular time grid.

    Each curve row is an ordinary dictionary with ``start_sec``, ``end_sec``,
    ``time_sec`` (bin centre), ``score`` and observation count.  A final partial
    grid bin is retained so the video tail cannot silently disappear.
    """

    duration = _safe_duration(duration_sec)
    step = _finite_number(grid_sec) or 0.0
    if duration <= 0 or step <= 0:
        return []
    rows = [record_to_dict(record) for record in records]
    usable: List[Tuple[float, float, float]] = []
    for row in rows:
        interval = _interval(row)
        value = _score(row, score_key)
        if interval is None or value is None:
            continue
        start, end = interval
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end <= start:
            continue
        usable.append((start, end, value))

    count = max(1, int(math.ceil(duration / step - 1e-12)))
    values: List[Optional[float]] = []
    observations: List[int] = []
    for index in range(count):
        start = index * step
        end = min(duration, start + step)
        # A positive overlap, rather than boundary equality, assigns a window to
        # a bin.  All overlapping windows contribute equally per the design.
        covering = [
            value
            for window_start, window_end, value in usable
            if min(end, window_end) > max(start, window_start) + 1e-12
        ]
        values.append(float(sum(covering) / len(covering)) if covering else None)
        observations.append(len(covering))

    filled_flags = [False] * len(values)
    if fill_missing:
        values, filled_flags = _fill_curve_values(
            values, max_gap_bins=max_fill_gap_bins
        )

    curve: List[Dict[str, Any]] = []
    for index, value in enumerate(values):
        start = index * step
        end = min(duration, start + step)
        curve.append(
            {
                "index": index,
                "start_sec": float(start),
                "end_sec": float(end),
                "time_sec": float((start + end) / 2.0),
                "raw_score": value,
                "score": value,
                "observation_count": observations[index],
                "filled": bool(filled_flags[index]),
            }
        )
    return curve


def smooth_score_curve(
    curve: Iterable[Any],
    *,
    window_points: int = 3,
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """Return a centred moving-average curve without ever emitting NaN."""

    rows = [record_to_dict(record) for record in curve]
    if not rows:
        return []
    width = max(1, int(window_points))
    if width % 2 == 0:
        width += 1
    radius = width // 2
    values = [_score(row, score_key) for row in rows]
    result: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        neighbours = [
            value
            for value in values[max(0, index - radius) : min(len(values), index + radius + 1)]
            if value is not None
        ]
        smoothed = float(sum(neighbours) / len(neighbours)) if neighbours else None
        copy = dict(row)
        copy["score_before_smoothing"] = values[index]
        copy["smoothed_score"] = smoothed
        copy["score"] = smoothed
        result.append(copy)
    return result


def _percentile_threshold(scores: Sequence[float], top_fraction: float) -> Optional[float]:
    if not scores:
        return None
    fraction = max(1e-9, min(1.0, float(top_fraction)))
    try:
        return float(np.quantile(np.asarray(scores, dtype=np.float64), 1.0 - fraction))
    except (TypeError, ValueError, FloatingPointError):
        return None


def _merge_segments(segments: Iterable[Segment], gap_sec: float = 0.0) -> List[Segment]:
    ordered = sorted(
        (float(start), float(end))
        for start, end in segments
        if math.isfinite(start) and math.isfinite(end) and end > start
    )
    if not ordered:
        return []
    merged: List[List[float]] = [[ordered[0][0], ordered[0][1]]]
    gap = max(0.0, float(gap_sec))
    for start, end in ordered[1:]:
        # The specification says "less than 1 second"; equality is kept split.
        if start <= merged[-1][1] + 1e-9 or start - merged[-1][1] < gap - 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _segment_strength(segment: Segment, curve: Sequence[Mapping[str, Any]]) -> Tuple[float, float]:
    start, end = segment
    scores = []
    for row in curve:
        interval = _interval(row)
        value = _score(row)
        if interval is not None and value is not None and min(end, interval[1]) > max(start, interval[0]):
            scores.append(value)
    return (max(scores) if scores else 0.0, sum(scores) / len(scores) if scores else 0.0)


def _limit_segment_count(
    segments: Sequence[Segment], curve: Sequence[Mapping[str, Any]], max_segments: int
) -> List[Segment]:
    limit = max(1, int(max_segments))
    if len(segments) <= limit:
        return list(segments)
    ranked = sorted(
        segments,
        key=lambda segment: (_segment_strength(segment, curve), segment[1] - segment[0]),
        reverse=True,
    )[:limit]
    return sorted(ranked)


def _limit_coverage(
    segments: Sequence[Segment],
    curve: Sequence[Mapping[str, Any]],
    coverage_limit: float,
    min_segment_sec: float = 0.0,
) -> List[Segment]:
    limit = max(0.0, float(coverage_limit))
    total = sum(end - start for start, end in segments)
    if limit <= 0 or total <= limit + 1e-9:
        return list(segments) if limit > 0 else []

    ranked = sorted(
        segments,
        key=lambda segment: (_segment_strength(segment, curve), segment[1] - segment[0]),
        reverse=True,
    )
    kept: List[Segment] = []
    remaining = limit
    minimum = max(0.0, float(min_segment_sec))
    for start, end in ranked:
        if remaining <= 1e-9:
            break
        length = end - start
        if length <= remaining + 1e-9:
            kept.append((start, end))
            remaining -= length
            continue
        if remaining + 1e-9 < minimum:
            continue
        # Keep the strongest part of the final interval.  If a local peak is
        # available, centre the trim there; otherwise use the interval centre.
        peak_rows = []
        for row in curve:
            interval = _interval(row)
            value = _score(row)
            if interval is not None and value is not None and min(end, interval[1]) > max(start, interval[0]):
                peak_rows.append((value, float(row.get("time_sec", sum(interval) / 2.0))))
        if peak_rows:
            best_score = max(value for value, _time in peak_rows)
            best_times = [
                time
                for value, time in peak_rows
                if abs(value - best_score) <= 1e-12
            ]
            centre = sum(best_times) / len(best_times)
        else:
            centre = (start + end) / 2.0
        clipped_start = max(start, min(end - remaining, centre - remaining / 2.0))
        kept.append((clipped_start, clipped_start + remaining))
        remaining = 0.0
    return sorted(kept)


def find_local_peaks(
    curve: Iterable[Any],
    *,
    min_score: float = 0.45,
    top_fraction: float = 0.30,
    min_prominence: float = 0.0,
    max_peaks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Find de-duplicated local maxima for selecting fine-pass regions."""

    rows = [record_to_dict(record) for record in curve]
    values = [_score(row) for row in rows]
    finite = [value for value in values if value is not None]
    percentile = _percentile_threshold(finite, top_fraction)
    threshold = max(float(min_score), percentile if percentile is not None else float(min_score))
    candidates: List[Tuple[int, float]] = []
    index = 0
    while index < len(rows):
        value = values[index]
        if value is None or value + 1e-12 < threshold:
            index += 1
            continue
        plateau_end = index
        while plateau_end + 1 < len(rows) and values[plateau_end + 1] == value:
            plateau_end += 1
        previous = values[index - 1] if index > 0 else None
        following = values[plateau_end + 1] if plateau_end + 1 < len(rows) else None
        if (previous is None or value >= previous) and (following is None or value >= following):
            left_values: List[float] = []
            for left_index in range(index - 1, -1, -1):
                left_value = values[left_index]
                if left_value is None:
                    continue
                if left_value > value + 1e-12:
                    break
                left_values.append(left_value)
            right_values: List[float] = []
            for right_index in range(plateau_end + 1, len(values)):
                right_value = values[right_index]
                if right_value is None:
                    continue
                if right_value > value + 1e-12:
                    break
                right_values.append(right_value)
            bases = []
            if left_values:
                bases.append(min(left_values))
            if right_values:
                bases.append(min(right_values))
            base = max(bases) if bases else value
            prominence = max(0.0, value - base)
            if prominence + 1e-12 >= max(0.0, float(min_prominence)):
                candidates.append(((index + plateau_end) // 2, prominence))
        index = plateau_end + 1

    peaks = []
    for candidate, prominence in candidates:
        row = dict(rows[candidate])
        row["peak_score"] = values[candidate]
        row["prominence"] = prominence
        row["selection_threshold"] = threshold
        peaks.append(row)
    if max_peaks is not None and len(peaks) > max(0, int(max_peaks)):
        peaks = sorted(peaks, key=lambda item: item.get("peak_score", 0.0), reverse=True)[
            : max(0, int(max_peaks))
        ]
        peaks.sort(key=lambda item: item.get("time_sec", 0.0))
    return peaks


def select_refinement_regions(
    curve: Iterable[Any],
    duration_sec: float,
    *,
    min_score: float = 0.45,
    top_fraction: float = 0.30,
    min_prominence: float = 0.0,
    padding_sec: float = 1.0,
    merge_gap_sec: float = 1.0,
    max_regions: int = 6,
) -> List[Segment]:
    """Convert coarse local peaks into at most six merged fine-pass regions."""

    rows = [record_to_dict(record) for record in curve]
    duration = _safe_duration(duration_sec)
    if duration <= 0 or int(max_regions) <= 0:
        return []
    peaks = find_local_peaks(
        rows,
        min_score=min_score,
        top_fraction=top_fraction,
        min_prominence=min_prominence,
    )
    padding = max(0.0, float(padding_sec))
    regions = []
    for peak in peaks:
        interval = _interval(peak)
        if interval is None:
            centre = _finite_number(peak.get("time_sec"))
            if centre is None:
                continue
            interval = (centre, centre)
        regions.append((max(0.0, interval[0] - padding), min(duration, interval[1] + padding)))
    regions = _merge_segments(regions, gap_sec=merge_gap_sec)
    return _limit_segment_count(regions, rows, max_regions)


def select_multisignal_refinement_regions(
    signal_curves: Mapping[str, Iterable[Any]],
    duration_sec: float,
    *,
    min_score: float = 0.45,
    top_fraction: float = 0.30,
    min_prominence: float = 0.10,
    padding_sec: float = 1.0,
    merge_gap_sec: float = 1.0,
    max_regions: int = 6,
) -> List[Segment]:
    """Return the union of fused/Qwen/motion/audio coarse peak regions."""

    duration = _safe_duration(duration_sec)
    limit = max(0, int(max_regions))
    if duration <= 0 or limit <= 0:
        return []
    candidates: List[Tuple[Segment, float]] = []
    padding = max(0.0, float(padding_sec))
    for _signal_name, curve in signal_curves.items():
        rows = [record_to_dict(record) for record in curve]
        peaks = find_local_peaks(
            rows,
            min_score=min_score,
            top_fraction=top_fraction,
            min_prominence=min_prominence,
        )
        for peak in peaks:
            interval = _interval(peak)
            if interval is None:
                centre = _finite_number(peak.get("time_sec"))
                if centre is None:
                    continue
                interval = (centre, centre)
            segment = (
                max(0.0, interval[0] - padding),
                min(duration, interval[1] + padding),
            )
            if segment[1] > segment[0]:
                candidates.append((segment, float(peak.get("peak_score") or 0.0)))
    if not candidates:
        return []
    merged = _merge_segments((segment for segment, _score_value in candidates), merge_gap_sec)
    if len(merged) <= limit:
        return merged
    ranked = sorted(
        merged,
        key=lambda region: max(
            (
                strength
                for candidate, strength in candidates
                if min(region[1], candidate[1]) > max(region[0], candidate[0])
            ),
            default=0.0,
        ),
        reverse=True,
    )[:limit]
    return sorted(ranked)


def detect_tail_barrier(
    records: Iterable[Any],
    duration_sec: float,
    *,
    grid_sec: float = 0.5,
    tail_fraction: float = 0.25,
    strong_scene_threshold: float = 0.75,
    low_motion_threshold: float = 0.25,
    low_qwen_threshold: float = 0.30,
    min_low_sec: float = 1.0,
) -> Dict[str, Any]:
    """Detect a likely end-card boundary without treating cuts as highlights."""

    rows = [record_to_dict(record) for record in records]
    duration = _safe_duration(duration_sec)
    step = max(0.1, float(grid_sec))
    diagnostics: Dict[str, Any] = {
        "barrier_sec": None,
        "tail_start_sec": duration * (1.0 - max(0.0, min(1.0, tail_fraction))),
        "reason": "no_tail_barrier",
    }
    if duration <= 0 or not rows:
        return diagnostics

    def value_for(row: Mapping[str, Any], key: str) -> Optional[float]:
        if key == "scene":
            return _finite_number(
                row.get("normalized_scene_score")
                if row.get("normalized_scene_score") is not None
                else row.get("scene_score")
            )
        if key == "motion":
            return _finite_number(
                row.get("normalized_motion_score")
                if row.get("normalized_motion_score") is not None
                else row.get("motion_score")
            )
        if row.get("valid") is False:
            return None
        return _finite_number(row.get("qwen_score"))

    bin_count = max(1, int(math.ceil(duration / step - 1e-12)))
    channels: Dict[str, List[Optional[float]]] = {
        "scene": [],
        "motion": [],
        "qwen": [],
    }
    for bin_index in range(bin_count):
        bin_start = bin_index * step
        bin_end = min(duration, bin_start + step)
        for key in channels:
            covering: List[float] = []
            for row in rows:
                interval = _interval(row)
                value = value_for(row, key)
                if (
                    interval is not None
                    and value is not None
                    and min(bin_end, interval[1]) > max(bin_start, interval[0]) + 1e-12
                ):
                    covering.append(max(0.0, min(1.0, value)))
            channels[key].append(
                (
                    float(max(covering))
                    if covering and key == "scene"
                    else float(sum(covering) / len(covering))
                    if covering
                    else None
                )
            )

    first_tail_bin = max(0, int(math.floor(diagnostics["tail_start_sec"] / step)))
    required = max(1, int(math.ceil(max(0.1, float(min_low_sec)) / step)))
    for index in range(first_tail_bin, max(first_tail_bin, bin_count - required)):
        scene = channels["scene"][index]
        if scene is None or scene + 1e-12 < max(0.0, float(strong_scene_threshold)):
            continue
        following_motion = channels["motion"][index + 1 : index + 1 + required]
        following_qwen = channels["qwen"][index + 1 : index + 1 + required]
        if len(following_motion) < required or any(value is None for value in following_motion):
            continue
        if any(float(value) > low_motion_threshold + 1e-12 for value in following_motion):
            continue
        # A missing semantic channel is not proof of a low-semantic end card.
        if len(following_qwen) < required or any(value is None for value in following_qwen):
            continue
        if any(float(value) > low_qwen_threshold + 1e-12 for value in following_qwen):
            continue
        barrier = min(duration, (index + 1) * step)
        diagnostics.update(
            {
                "barrier_sec": barrier,
                "reason": "strong_tail_cut_followed_by_low_motion_and_qwen",
                "scene_score": scene,
                "low_evidence_sec": required * step,
            }
        )
        return diagnostics
    return diagnostics


def overlay_score_curves(
    coarse_curve: Iterable[Any],
    fine_curve: Iterable[Any],
    *,
    max_refine_weight: float = 0.35,
    full_evidence_windows: int = 4,
    conflict_threshold: float = 0.25,
    conflict_weight_cap: float = 0.15,
) -> List[Dict[str, Any]]:
    """Conservatively blend fine evidence into the coarse curve.

    Fine windows refine timing; they do not erase a coarse event.  Evidence
    below four overlapping fine observations receives proportionally less than
    the normal 0.35 weight, and a large under-supported disagreement is capped
    at 0.15.  Every row exposes the complete blend diagnostic for raw JSONL.
    """

    coarse = [record_to_dict(record) for record in coarse_curve]
    fine = [record_to_dict(record) for record in fine_curve]
    result = []
    for row in coarse:
        interval = _interval(row)
        covering: List[Tuple[float, int]] = []
        if interval is not None:
            for fine_row in fine:
                fine_interval = _interval(fine_row)
                fine_value = _score(fine_row)
                if (
                    fine_interval is not None
                    and fine_value is not None
                    and min(interval[1], fine_interval[1]) > max(interval[0], fine_interval[0]) + 1e-12
                ):
                    evidence = _finite_number(fine_row.get("observation_count"))
                    covering.append((fine_value, max(1, int(evidence or 1))))
        copy = dict(row)
        coarse_score = _score(row)
        copy["coarse_score"] = coarse_score
        copy["fine_score"] = None
        copy["final_score"] = coarse_score
        copy["refine_weight"] = 0.0
        copy["conflict_reason"] = None
        if covering:
            fine_score = float(sum(value for value, _count in covering) / len(covering))
            evidence_count = max(count for _value, count in covering)
            required = max(1, int(full_evidence_windows))
            weight = max(0.0, min(1.0, float(max_refine_weight))) * min(
                1.0, evidence_count / float(required)
            )
            reasons: List[str] = []
            if evidence_count < required:
                reasons.append("fine_evidence_lt_4")
            if (
                coarse_score is not None
                and abs(fine_score - coarse_score) > max(0.0, float(conflict_threshold))
                and evidence_count < required
            ):
                weight = min(weight, max(0.0, float(conflict_weight_cap)))
                reasons.append("coarse_fine_conflict_limited")
            final_score = (
                (1.0 - weight) * coarse_score + weight * fine_score
                if coarse_score is not None
                else fine_score
            )
            copy["fine_score"] = fine_score
            copy["final_score"] = final_score
            copy["refine_weight"] = weight
            copy["fine_evidence_count"] = evidence_count
            copy["conflict_reason"] = ";".join(reasons) if reasons else None
            copy["score"] = final_score
            copy["smoothed_score"] = final_score
            copy["refined"] = True
        else:
            copy["score"] = coarse_score
            copy["refined"] = False
        result.append(copy)
    return result


def select_highlight_segments(
    curve: Iterable[Any],
    duration_sec: float,
    *,
    score_threshold: float = 0.55,
    top_fraction: float = 0.30,
    min_segment_sec: float = 1.0,
    merge_gap_sec: float = 1.0,
    padding_sec: float = 0.75,
    max_padding_sec: float = 1.5,
    max_segments: int = 12,
    max_coverage_sec: float = 30.0,
    max_coverage_ratio: float = 0.60,
    growth_threshold: Optional[float] = None,
    use_local_peak_seeds: bool = False,
    min_peak_prominence: float = 0.0,
    end_barrier_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Select, merge, pad and cap temporal ranges from a score curve."""

    rows = [record_to_dict(record) for record in curve]
    duration = _safe_duration(duration_sec)
    values = [_score(row) for row in rows]
    finite = [value for value in values if value is not None]
    percentile = _percentile_threshold(finite, top_fraction)
    threshold = max(
        max(0.0, min(1.0, float(score_threshold))),
        percentile if percentile is not None else 0.0,
    )

    selection_end = duration
    barrier = _finite_number(end_barrier_sec)
    if barrier is not None:
        selection_end = max(0.0, min(duration, barrier))

    selected_bins: List[Segment] = []
    if use_local_peak_seeds:
        seeded_rows = []
        for position, row in enumerate(rows):
            copy = dict(row)
            copy["_selection_position"] = position
            seeded_rows.append(copy)
        seeds = find_local_peaks(
            seeded_rows,
            min_score=max(0.0, min(1.0, float(score_threshold))),
            # The 0.45 absolute seed is intentional; a true second event must
            # not disappear merely because another event occupies the top 30%.
            top_fraction=1.0,
            min_prominence=min_peak_prominence,
        )
        grow = max(
            0.0,
            min(
                1.0,
                float(growth_threshold)
                if growth_threshold is not None
                else max(0.0, min(1.0, float(score_threshold))),
            ),
        )
        for seed in seeds:
            try:
                seed_index = int(seed["_selection_position"])
            except (KeyError, TypeError, ValueError):
                continue
            left = seed_index
            right = seed_index
            while left > 0 and values[left - 1] is not None and values[left - 1] + 1e-12 >= grow:
                left -= 1
            while (
                right + 1 < len(rows)
                and values[right + 1] is not None
                and values[right + 1] + 1e-12 >= grow
            ):
                right += 1
            left_interval = _interval(rows[left])
            right_interval = _interval(rows[right])
            if left_interval is None or right_interval is None:
                continue
            start = max(0.0, min(selection_end, left_interval[0]))
            end = max(0.0, min(selection_end, right_interval[1]))
            if end > start:
                selected_bins.append((start, end))
        selected_bins = _merge_segments(selected_bins, gap_sec=0.0)
    else:
        for row, value in zip(rows, values):
            interval = _interval(row)
            if interval is None or value is None or value + 1e-12 < threshold:
                continue
            start = max(0.0, min(selection_end, interval[0]))
            end = max(0.0, min(selection_end, interval[1]))
            if end > start:
                selected_bins.append((start, end))

    contiguous = _merge_segments(selected_bins, gap_sec=0.0)
    minimum = max(0.0, float(min_segment_sec))
    long_enough = [segment for segment in contiguous if segment[1] - segment[0] + 1e-9 >= minimum]
    merged = _merge_segments(long_enough, gap_sec=max(0.0, float(merge_gap_sec)))

    padding = min(max(0.0, float(padding_sec)), max(0.0, float(max_padding_sec)))
    padded = [
        (max(0.0, start - padding), min(selection_end, end + padding))
        for start, end in merged
        if end > start
    ]
    padded = _merge_segments(padded, gap_sec=0.0)
    limited = _limit_segment_count(padded, rows, max_segments)
    coverage_limit = min(
        max(0.0, float(max_coverage_sec)),
        duration * max(0.0, min(1.0, float(max_coverage_ratio))),
    )
    limited = _limit_coverage(
        limited,
        rows,
        coverage_limit,
        min_segment_sec=minimum,
    )

    peaks = find_local_peaks(
        rows,
        min_score=(float(score_threshold) if use_local_peak_seeds else threshold),
        top_fraction=(1.0 if use_local_peak_seeds else top_fraction),
        min_prominence=(min_peak_prominence if use_local_peak_seeds else 0.0),
        max_peaks=max_segments,
    )
    diagnostics = {
        "score_threshold": float(score_threshold),
        "percentile_threshold": percentile,
        "threshold_used": threshold,
        "seed_threshold": (
            max(0.0, min(1.0, float(score_threshold)))
            if use_local_peak_seeds
            else threshold
        ),
        "growth_threshold": growth_threshold,
        "local_peak_seed_mode": bool(use_local_peak_seeds),
        "end_barrier_sec": barrier,
        "finite_score_count": len(finite),
        "candidate_bin_count": len(selected_bins),
        "removed_short_segment_count": len(contiguous) - len(long_enough),
        "coverage_limit_sec": coverage_limit,
        "coverage_sec": sum(end - start for start, end in limited),
        "fallback_required": not bool(limited),
    }
    return {
        "segments_sec": [(float(start), float(end)) for start, end in limited],
        "segments": [[float(start), float(end)] for start, end in limited],
        "selected_peaks": peaks,
        "score_diagnostics": diagnostics,
    }


def fallback_segments(
    records: Iterable[Any],
    duration_sec: float,
    *,
    grid_sec: float = 0.5,
    short_video_sec: float = 4.0,
    middle_sec: float = 4.0,
    min_segment_sec: float = 1.0,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Produce a non-empty conservative interval when normal selection fails.

    The order is: traditional fused signals, motion-only evidence, complete short
    video, then the middle four seconds.  The return value includes the source so
    the caller can report the fallback in ``.raw.jsonl``.
    """

    rows = [record_to_dict(record) for record in records]
    duration = _safe_duration(duration_sec)
    if duration <= 0:
        return {
            "segments_sec": [],
            "segments": [],
            "source": "unreadable_video",
            "score_diagnostics": {"fallback_required": True, "reason": "non-positive duration"},
        }

    has_traditional = any(
        _finite_number(row.get(key)) is not None for row in rows for key in TRADITIONAL_SIGNAL_KEYS
    )
    if has_traditional:
        traditional_rows = []
        for row in rows:
            copy = dict(row)
            copy["qwen_score"] = None
            traditional_rows.append(copy)
        traditional_weights = dict(weights or {})
        traditional_weights["qwen_score"] = 0.0
        has_precalibrated = any(
            row.get(f"normalized_{key}") is not None
            for row in traditional_rows
            for key in TRADITIONAL_SIGNAL_KEYS
        )
        disabled_precalibrated = set()
        if has_precalibrated:
            for key in TRADITIONAL_SIGNAL_KEYS:
                values = [
                    _finite_number(row.get(f"normalized_{key}"))
                    for row in traditional_rows
                ]
                present = [value for value in values if value is not None]
                if present and max(present) - min(present) <= 1e-6 + 1e-12:
                    disabled_precalibrated.add(key)
        fused = fuse_window_scores(
            traditional_rows,
            weights=traditional_weights,
            # Stable frontend rows already carry the coarse P10/P90 scale.
            # Re-fitting on the handful of fine rows would recreate the exact
            # 0/1 amplification this fallback is meant to avoid.  Standalone
            # callers without calibrated fields retain the legacy safe scale.
            normalize=not has_precalibrated,
            disable_degenerate_qwen=True,
            disabled_signals=disabled_precalibrated,
        )
        curve = smooth_score_curve(
            build_score_curve(fused, duration, grid_sec=grid_sec), window_points=3
        )
        selected = select_highlight_segments(
            curve,
            duration,
            min_segment_sec=min_segment_sec,
        )
        if selected["segments_sec"]:
            selected["source"] = "traditional_signals"
            selected["score_curve"] = curve
            return selected

    motion_rows = []
    motion_values: List[float] = []
    for row in rows:
        motion = _finite_number(
            row.get("normalized_motion_score")
            if row.get("normalized_motion_score") is not None
            else row.get("motion_score")
        )
        interval = _interval(row)
        if motion is not None and interval is not None:
            motion_values.append(motion)
            motion_rows.append(
                {
                    "start_sec": interval[0],
                    "end_sec": interval[1],
                    "fused_score": max(0.0, min(1.0, motion)),
                }
            )
    motion_has_information = bool(motion_values) and max(motion_values) > 1e-6
    if len(motion_values) > 1:
        motion_has_information = (
            motion_has_information
            and max(motion_values) - min(motion_values) > 1e-6
        )
    if motion_rows and motion_has_information:
        curve = smooth_score_curve(
            build_score_curve(motion_rows, duration, grid_sec=grid_sec), window_points=3
        )
        selected = select_highlight_segments(
            curve,
            duration,
            score_threshold=0.0,
            top_fraction=0.30,
            min_segment_sec=min_segment_sec,
        )
        if selected["segments_sec"]:
            selected["source"] = "motion"
            selected["score_curve"] = curve
            return selected

    if duration <= max(0.0, float(short_video_sec)):
        segments = [(0.0, duration)]
        source = "short_video_full"
    else:
        length = min(duration, max(float(min_segment_sec), float(middle_sec)))
        start = max(0.0, (duration - length) / 2.0)
        segments = [(start, min(duration, start + length))]
        source = "middle"
    return {
        "segments_sec": segments,
        "segments": [[start, end] for start, end in segments],
        "source": source,
        "selected_peaks": [],
        "score_diagnostics": {
            "fallback_required": True,
            "reason": "no usable fused or motion curve",
        },
    }


def score_windows_to_segments(
    records: Iterable[Any],
    duration_sec: float,
    *,
    grid_sec: float = 0.5,
    smooth_points: int = 3,
    **selection_kwargs: Any,
) -> Dict[str, Any]:
    """Convenience high-level pure pipeline used by the detector."""

    rows = [record_to_dict(record) for record in records]
    curve = build_score_curve(rows, duration_sec, grid_sec=grid_sec)
    curve = smooth_score_curve(curve, window_points=smooth_points)
    selected = select_highlight_segments(curve, duration_sec, **selection_kwargs)
    selected["score_curve"] = curve
    if not selected["segments_sec"]:
        fallback = fallback_segments(rows, duration_sec, grid_sec=grid_sec)
        fallback.setdefault("score_curve", curve)
        fallback.setdefault("selected_peaks", selected.get("selected_peaks", []))
        return fallback
    selected["source"] = "window_score_curve"
    return selected


# Backward/alternate spelling aliases for integration code.
segments_from_score_curve = select_highlight_segments
build_fallback_segments = fallback_segments


__all__ = [
    "Segment",
    "build_fallback_segments",
    "build_score_curve",
    "detect_tail_barrier",
    "fallback_segments",
    "find_local_peaks",
    "overlay_score_curves",
    "score_windows_to_segments",
    "segments_from_score_curve",
    "select_highlight_segments",
    "select_multisignal_refinement_regions",
    "select_refinement_regions",
    "smooth_score_curve",
]
