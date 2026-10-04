"""Conservative fusion of TimeLens ranges with traditional temporal signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .temporal import TemporalResult, normalize_temporal_segments


@dataclass(frozen=True)
class TemporalFusionPolicy:
    """Traditional signals may adjust boundaries, but may not invent highlights."""

    max_boundary_adjust_sec: float = 1.5
    max_segments: int = 3


def fuse_temporal_results(
    grounded: TemporalResult,
    signals: TemporalResult,
    duration_sec: float,
    policy: TemporalFusionPolicy = TemporalFusionPolicy(),
) -> TemporalResult:
    if not grounded.valid or not grounded.segments_sec:
        if signals.valid and signals.segments_sec:
            signals.source = f"{signals.source or 'traditional_signal'}_fallback"
            signals.fallback_used = signals.source
        return signals
    if not signals.valid or not signals.segments_sec:
        grounded.source = "qwen_action_timelens_unchecked"
        grounded.corrections.append("traditional signals unavailable; kept TimeLens boundaries")
        return grounded

    adjusted = []
    corrections: List[str] = list(grounded.corrections)
    limit = max(0.0, float(policy.max_boundary_adjust_sec))
    for start, end in grounded.segments_sec:
        overlapping = [
            (signal_start, signal_end)
            for signal_start, signal_end in signals.segments_sec
            if min(end, signal_end) > max(start, signal_start)
        ]
        if not overlapping:
            adjusted.append((start, end))
            corrections.append("traditional signals did not confirm one TimeLens range")
            continue
        signal_start = min(item[0] for item in overlapping)
        signal_end = max(item[1] for item in overlapping)
        new_start = max(0.0, min(start, max(start - limit, signal_start)))
        new_end = min(duration_sec, max(end, min(end + limit, signal_end)))
        adjusted.append((new_start, new_end))
        if new_start != start or new_end != end:
            corrections.append("traditional signals adjusted a TimeLens boundary")
        else:
            corrections.append("traditional signals confirmed a TimeLens range")

    normalized = normalize_temporal_segments(
        adjusted,
        duration_sec=duration_sec,
        min_segment_sec=0.25,
        padding_sec=0.0,
        max_segments=max(1, int(policy.max_segments)),
    )
    return TemporalResult(
        list(normalized.segments_sec),
        raw=grounded.raw,
        source="qwen_action_timelens_signals_fused",
        valid=normalized.valid,
        errors=list(grounded.errors) + list(signals.errors) + list(normalized.errors),
        attempts=grounded.attempts,
        duration_sec=duration_sec,
        segments_before_normalization=list(adjusted),
        segments_after_normalization=list(normalized.segments_sec),
        corrections=corrections + list(normalized.corrections),
    )
