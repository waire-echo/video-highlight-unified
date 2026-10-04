"""Offline re-aggregation of saved window-score temporal diagnostics.

The expensive Qwen pass writes every window score to ``.raw.jsonl``.  This
module deliberately consumes only those saved records: it can change fusion
and segment selection without decoding videos, loading a model, or touching the
spatial/YOLO stage.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .score_curve import (
    build_score_curve,
    overlay_score_curves,
    select_highlight_segments,
    smooth_score_curve,
)
from .temporal_handoff_validation import validate_temporal_handoff_records


Segment = Tuple[float, float]


@dataclass(frozen=True)
class ReaggregationProfile:
    """Configuration for one reproducible offline temporal variant."""

    name: str
    qwen_weight: float = 0.80
    motion_weight: float = 0.15
    audio_weight: float = 0.05
    absolute_threshold: float = 0.45
    top_fraction: float = 0.40
    min_segment_sec: float = 1.0
    merge_gap_sec: float = 0.25
    padding_sec: float = 0.50
    max_segments: int = 12
    max_coverage_ratio: float = 0.60
    max_coverage_sec: float = 30.0
    fallback_sec: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PROFILES: Dict[str, ReaggregationProfile] = {
    # The three profiles share the same calibrated evidence.  Only the relative
    # amount retained changes, which makes a later same-YOLO A/B interpretable.
    "balanced": ReaggregationProfile(
        name="balanced",
        top_fraction=0.40,
        padding_sec=0.50,
    ),
    "conservative": ReaggregationProfile(
        name="conservative",
        top_fraction=0.30,
        padding_sec=0.50,
    ),
    "precision": ReaggregationProfile(
        name="precision",
        top_fraction=0.30,
        padding_sec=0.25,
    ),
}


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _window_fused_score(
    row: Mapping[str, Any], profile: ReaggregationProfile
) -> Optional[float]:
    """Fuse already-calibrated channels with Qwen as ranking evidence.

    Motion and audio remain weak helpers.  The original frontend normalised the
    positive weights 0.50/0.20/0.10 to an effective 62.5/25/12.5 percent.  The
    offline profiles intentionally use 80/15/5 so routine motion or sound cannot
    manufacture a highlight peak by itself.
    """

    values: List[Tuple[float, float]] = []
    qwen = None if row.get("qwen_disabled") else _finite(row.get("qwen_score"))
    motion = _finite(row.get("normalized_motion_score"))
    audio = _finite(row.get("normalized_audio_score"))
    if row.get("valid") is True and qwen is not None and profile.qwen_weight > 0:
        values.append((max(0.0, min(1.0, qwen)), profile.qwen_weight))
    if motion is not None and profile.motion_weight > 0:
        values.append((max(0.0, min(1.0, motion)), profile.motion_weight))
    if audio is not None and profile.audio_weight > 0:
        values.append((max(0.0, min(1.0, audio)), profile.audio_weight))
    denominator = sum(weight for _value, weight in values)
    if denominator <= 0:
        return None
    return float(sum(value * weight for value, weight in values) / denominator)


def rebuild_score_curve(
    raw_record: Mapping[str, Any], profile: ReaggregationProfile
) -> List[Dict[str, Any]]:
    """Rebuild the final 0.5-second curve from saved coarse/fine windows."""

    duration = max(0.0, _finite(raw_record.get("duration_sec")) or 0.0)
    rows: List[Dict[str, Any]] = []
    for value in raw_record.get("window_records") or []:
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        score = _window_fused_score(row, profile)
        row["original_fused_score"] = row.get("fused_score")
        row["reaggregated_fused_score"] = score
        row["fused_score"] = score
        row["score"] = score
        rows.append(row)

    coarse = build_score_curve(
        [row for row in rows if row.get("stage") == "coarse"],
        duration,
        grid_sec=0.5,
    )
    fine = build_score_curve(
        [row for row in rows if row.get("stage") == "fine"],
        duration,
        grid_sec=0.5,
        fill_missing=False,
    )
    fine = [
        row for row in fine if int(row.get("observation_count", 0) or 0) > 0
    ]
    combined = (
        overlay_score_curves(
            coarse,
            fine,
            max_refine_weight=0.35,
            full_evidence_windows=4,
            conflict_threshold=0.25,
            conflict_weight_cap=0.15,
        )
        if fine
        else coarse
    )
    return smooth_score_curve(combined, window_points=3)


def _top_peak_fallback(
    curve: Sequence[Mapping[str, Any]], duration_sec: float, fallback_sec: float
) -> List[Segment]:
    candidates: List[Tuple[float, float]] = []
    for row in curve:
        score = _finite(row.get("score"))
        centre = _finite(row.get("time_sec"))
        if score is None or centre is None:
            continue
        candidates.append((score, centre))
    duration = max(0.0, float(duration_sec))
    if duration <= 0:
        return []
    centre = max(candidates)[1] if candidates else duration / 2.0
    length = min(duration, max(1.0, float(fallback_sec)))
    start = max(0.0, min(duration - length, centre - length / 2.0))
    return [(start, start + length)]


def reaggregate_curve(
    raw_record: Mapping[str, Any], profile: ReaggregationProfile
) -> Dict[str, Any]:
    """Select relative high-score islands and retain separated events."""

    duration = max(0.0, _finite(raw_record.get("duration_sec")) or 0.0)
    curve = rebuild_score_curve(raw_record, profile)
    selected = select_highlight_segments(
        curve,
        duration,
        score_threshold=profile.absolute_threshold,
        top_fraction=profile.top_fraction,
        min_segment_sec=profile.min_segment_sec,
        merge_gap_sec=profile.merge_gap_sec,
        padding_sec=profile.padding_sec,
        max_padding_sec=profile.padding_sec,
        max_segments=profile.max_segments,
        max_coverage_sec=profile.max_coverage_sec,
        max_coverage_ratio=profile.max_coverage_ratio,
        # Important: unlike the cloud run, the percentile threshold remains in
        # force.  We select separated score islands instead of growing every
        # weak local peak across a globally high 0.40 baseline.
        use_local_peak_seeds=False,
    )
    fallback_used = None
    segments = [tuple(map(float, item)) for item in selected.get("segments_sec") or []]
    if not segments:
        segments = _top_peak_fallback(curve, duration, profile.fallback_sec)
        fallback_used = "top_peak_window"
    diagnostics = dict(selected.get("score_diagnostics") or {})
    diagnostics.update(
        {
            "profile": profile.to_dict(),
            "fallback_used": fallback_used,
            "original_segments_sec": raw_record.get("temporal_segments_sec")
            or raw_record.get("detected_segments_sec")
            or [],
        }
    )
    return {
        "segments_sec": segments,
        "score_curve": curve,
        "selected_peaks": list(selected.get("selected_peaks") or []),
        "score_diagnostics": diagnostics,
        "fallback_used": fallback_used,
    }


def segments_to_frames(
    segments: Iterable[Segment], fps: float, frame_count: int
) -> Tuple[List[List[float]], List[List[int]]]:
    """Convert second intervals to inclusive original-frame intervals."""

    safe_fps = max(1e-9, float(fps))
    count = max(1, int(frame_count))
    paired: List[Tuple[List[float], List[int]]] = []
    for start, end in sorted((float(a), float(b)) for a, b in segments if b > a):
        start_frame = max(0, min(int(round(start * safe_fps)), count - 1))
        end_frame = max(0, min(int(round(end * safe_fps)), count - 1))
        if paired and start_frame <= paired[-1][1][1] + 1:
            paired[-1][0][1] = max(paired[-1][0][1], end)
            paired[-1][1][1] = max(paired[-1][1][1], end_frame)
        else:
            paired.append(([start, end], [start_frame, end_frame]))
    return [item[0] for item in paired], [item[1] for item in paired]


def canonical_subject_hint(value: Any) -> Optional[str]:
    """Return only the stable class aliases accepted by the spatial handoff."""

    text = str(value or "").strip().lower()
    tokens = set(re.findall(r"[a-z]+", text))
    if tokens & {"dog", "dogs", "puppy", "puppies", "canine"}:
        return "dog"
    if tokens & {"cat", "cats", "kitten", "kittens", "feline"}:
        return "cat"
    if tokens & {
        "person",
        "people",
        "human",
        "man",
        "woman",
        "women",
        "boy",
        "girl",
        "child",
        "children",
        "athlete",
        "player",
        "runner",
        "runners",
        "rider",
        "operator",
    }:
        return "person"
    if "sports ball" in text or tokens & {
        "ball",
        "football",
        "basketball",
        "soccer",
    }:
        return "ball"
    if tokens & {"car", "truck", "bus", "vehicle", "bicycle", "bike", "motorcycle"}:
        return "vehicle"
    if tokens & {"animal", "horse", "bird", "bear", "pig"}:
        return "animal"
    return None


def reaggregate_records(
    handoff_records: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
    profile: ReaggregationProfile,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return aligned handoff/raw rows for one profile."""

    handoff_by_id = {str(row.get("video_id")): dict(row) for row in handoff_records}
    raw_by_id = {str(row.get("video_id")): dict(row) for row in raw_records}
    if set(handoff_by_id) != set(raw_by_id):
        raise ValueError("handoff/raw video IDs differ")
    output: List[Dict[str, Any]] = []
    raw_output: List[Dict[str, Any]] = []
    for video_id in sorted(
        handoff_by_id,
        key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
    ):
        handoff = dict(handoff_by_id[video_id])
        raw = dict(raw_by_id[video_id])
        result = reaggregate_curve(raw, profile)
        seconds, frames = segments_to_frames(
            result["segments_sec"],
            float(handoff.get("fps") or raw.get("fps") or 0.0),
            int(handoff.get("frame_count") or raw.get("frame_count") or 0),
        )
        original_seconds = handoff.get("segments_sec") or []
        handoff.update(
            {
                "segments_sec": seconds,
                "segments_frame": frames,
                "subject_hint": canonical_subject_hint(handoff.get("subject_hint")),
                "source": f"window_score_reaggregated_{profile.name}",
                "valid": bool(seconds),
                "fallback_used": result["fallback_used"],
                "errors": [] if seconds else ["offline reaggregation produced no segment"],
            }
        )
        warnings = list(handoff.get("warnings") or [])
        warnings.append(
            f"offline reaggregation profile={profile.name}; original_segments={original_seconds}"
        )
        handoff["warnings"] = warnings
        raw.update(
            {
                "original_temporal_segments_sec": original_seconds,
                "temporal_segments_sec": seconds,
                "temporal_segments_frame": frames,
                "subject_hint": handoff.get("subject_hint"),
                "source": handoff["source"],
                "fallback_used": result["fallback_used"],
                "score_curve": result["score_curve"],
                "selected_peaks": result["selected_peaks"],
                "score_diagnostics": result["score_diagnostics"],
                "valid": bool(seconds),
            }
        )
        output.append(handoff)
        raw_output.append(raw)
    ok, errors = validate_temporal_handoff_records(output)
    if not ok:
        raise ValueError("reaggregated temporal handoff is invalid: " + "; ".join(errors[:20]))
    return output, raw_output


def summarize_reaggregation(
    records: Sequence[Mapping[str, Any]], profile: ReaggregationProfile
) -> Dict[str, Any]:
    coverages: List[float] = []
    cap_hits = 0
    segments_total = 0
    multi_video_count = 0
    fallback_count = 0
    for row in records:
        duration = max(0.0, _finite(row.get("duration_sec")) or 0.0)
        segments = row.get("segments_sec") or []
        covered = sum(max(0.0, float(end) - float(start)) for start, end in segments)
        coverage = covered / duration if duration > 0 else 0.0
        coverages.append(coverage)
        limit = min(profile.max_coverage_sec, duration * profile.max_coverage_ratio)
        if duration > 0 and abs(covered - limit) <= 1e-6:
            cap_hits += 1
        segments_total += len(segments)
        multi_video_count += len(segments) > 1
        fallback_count += bool(row.get("fallback_used"))
    ordered = sorted(coverages)
    median = (
        (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2.0
        if ordered
        else 0.0
    )
    return {
        "schema_version": "temporal-reaggregation-summary/v1",
        "profile": profile.to_dict(),
        "videos": len(records),
        "valid_videos": sum(bool(row.get("valid")) for row in records),
        "segments_total": segments_total,
        "multi_segment_videos": multi_video_count,
        "coverage_cap_hits": cap_hits,
        "coverage_median": median,
        "coverage_mean": sum(coverages) / len(coverages) if coverages else 0.0,
        "fallback_videos": fallback_count,
    }


__all__ = [
    "PROFILES",
    "ReaggregationProfile",
    "canonical_subject_hint",
    "reaggregate_curve",
    "reaggregate_records",
    "rebuild_score_curve",
    "segments_to_frames",
    "summarize_reaggregation",
]
