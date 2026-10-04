import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from video_highlight.reze_aggregation import (
    aggregate_reze_otsu,
    build_reze_curve,
    extract_otsu_candidates,
    gaussian_smooth_values,
    otsu_threshold,
    select_otsu_segments,
    smooth_reze_curve,
)
from video_highlight.reze_scoring import RezeWindowScore


def make_record(window_id, start, end, probability, *, valid=True):
    return RezeWindowScore(
        window_id,
        start,
        end,
        2.0,
        p_yes=probability,
        valid=valid,
    )


def make_curve(scores):
    return [
        {
            "index": index,
            "start_sec": float(index),
            "end_sec": float(index + 1),
            "time_sec": index + 0.5,
            "p_yes": score,
            "smoothed_p_yes": score,
            "valid": score is not None,
        }
        for index, score in enumerate(scores)
    ]


def test_build_one_second_curve_keeps_partial_tail_and_missing_values():
    records = [
        make_record(0, 0.0, 3.0, 0.2),
        make_record(1, 3.0, 6.0, 0.8),
        make_record(2, 6.0, 7.2, 0.5),
    ]
    curve = build_reze_curve(records, 7.2)
    assert len(curve) == 8
    assert curve[-1]["start_sec"] == 7.0
    assert curve[-1]["end_sec"] == pytest.approx(7.2)
    assert [row["p_yes"] for row in curve] == pytest.approx(
        [0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.5, 0.5]
    )

    missing = build_reze_curve(
        [
            make_record(0, 0.0, 3.0, float("nan")),
            make_record(1, 3.0, 6.0, 0.9, valid=False),
        ],
        6.0,
    )
    assert all(row["p_yes"] is None for row in missing)


def test_gaussian_smoothing_uses_sigma_three_and_does_not_fill_missing_bins():
    values = [0.1, 0.1, None, 0.9, 0.9]
    smoothed = gaussian_smooth_values(values, sigma=3.0)
    assert len(smoothed) == len(values)
    assert smoothed[2] is None
    assert all(
        value is None or (math.isfinite(value) and 0.0 <= value <= 1.0)
        for value in smoothed
    )
    assert gaussian_smooth_values(values, sigma=0.0) == values

    curve = smooth_reze_curve(make_curve(values), sigma=3.0)
    assert curve[2]["smoothed_p_yes"] is None
    assert curve[0]["p_yes"] == 0.1


def test_otsu_uses_two_classes_and_rejects_constant_or_nonfinite_input():
    threshold = otsu_threshold([0.1] * 10 + [0.9] * 10, bins=256)
    assert threshold is not None
    assert 0.09 <= threshold < 0.9
    assert otsu_threshold([0.5, 0.5, 0.5]) is None
    assert otsu_threshold([None, float("nan"), float("inf")]) is None
    with pytest.raises(ValueError):
        otsu_threshold([0.1, 0.9], bins=1)


def test_short_isolated_candidate_is_deleted():
    curve = make_curve([0.1, 0.9, 0.1, 0.1])
    assert extract_otsu_candidates(curve, 0.5, min_segment_sec=2.0) == []

    curve = make_curve([0.1, 0.9, 0.9, 0.1])
    candidates = extract_otsu_candidates(curve, 0.5, min_segment_sec=2.0)
    assert [(row["start_sec"], row["end_sec"]) for row in candidates] == [
        (1.0, 3.0)
    ]


def test_segments_rank_by_excess_area_before_coverage_cap():
    scores = [0.1] * 20
    scores[0:5] = [0.7] * 5
    scores[10:15] = [0.9] * 5
    segments, selected, diagnostics = select_otsu_segments(
        make_curve(scores),
        20.0,
        0.5,
        coverage_limit_sec=6.0,
    )
    assert segments == [(10.0, 15.0)]
    assert selected[0]["area_rank"] == 1
    assert diagnostics["selected_coverage_sec"] == 5.0
    assert diagnostics["coverage_limit_sec"] == 6.0


def test_segment_count_and_total_coverage_guards_are_hard_limits():
    # Fifteen separate two-second candidates.  The 12-segment guard is more
    # restrictive than the 30-second coverage guard in this construction.
    scores = []
    for _ in range(15):
        scores.extend([0.9, 0.9, 0.1])
    duration = float(len(scores))
    segments, _selected, diagnostics = select_otsu_segments(
        make_curve(scores), duration, 0.5, max_segments=99
    )
    assert len(segments) == 12
    assert diagnostics["max_segments"] == 12
    assert diagnostics["selected_coverage_sec"] == 24.0
    assert diagnostics["selected_coverage_sec"] <= min(30.0, duration * 0.60)

    # One long candidate is trimmed on the one-second grid to the project cap.
    long_segments, _selected, long_diagnostics = select_otsu_segments(
        make_curve([0.9] * 60), 60.0, 0.5
    )
    assert sum(end - start for start, end in long_segments) <= 30.0
    assert long_diagnostics["coverage_limit_sec"] == 30.0


def test_full_aggregation_finds_high_region_and_exports_diagnostics():
    records = []
    probabilities = [0.1, 0.1, 0.95, 0.95, 0.95, 0.15, 0.15, 0.15]
    for index, probability in enumerate(probabilities):
        records.append(
            make_record(index, index * 3.0, (index + 1) * 3.0, probability)
        )
    result = aggregate_reze_otsu(records, 24.0)
    assert result.segments_sec
    assert any(start < 10.5 < end for start, end in result.segments_sec)
    assert result.threshold is not None
    assert result.diagnostics["aggregation"] == "reze_otsu"
    assert result.diagnostics["grid_sec"] == 1.0
    assert result.diagnostics["gaussian_sigma"] == 3.0
    assert result.diagnostics["otsu_bins"] == 256
    assert result.diagnostics["status"] == "ok"
    assert result.fallback_required is False
    payload = result.to_dict()
    assert payload["source"] == "reze_otsu"
    assert payload["fallback_required"] is False


def test_full_aggregation_honours_a_stricter_caller_coverage_limit():
    records = [
        make_record(index, index * 3.0, (index + 1) * 3.0, probability)
        for index, probability in enumerate([0.1, 0.9, 0.9, 0.9, 0.1])
    ]
    result = aggregate_reze_otsu(records, 15.0, coverage_limit_sec=4.0)
    assert sum(end - start for start, end in result.segments_sec) <= 4.0
    assert result.diagnostics["coverage_limit_sec"] == 4.0


@pytest.mark.parametrize(
    "records,reason",
    [
        (
            [
                make_record(0, 0.0, 3.0, 0.60),
                make_record(1, 3.0, 6.0, 0.60),
            ],
            "probability_range_too_small",
        ),
        (
            [
                make_record(0, 0.0, 3.0, 0.51),
                make_record(1, 3.0, 6.0, 0.55),
            ],
            "probability_range_too_small",
        ),
        (
            [
                make_record(0, 0.0, 3.0, float("nan")),
                make_record(1, 3.0, 6.0, None, valid=False),
            ],
            "no_valid_probabilities",
        ),
    ],
)
def test_degenerate_or_missing_probabilities_return_empty_for_outer_fallback(
    records, reason
):
    result = aggregate_reze_otsu(records, 6.0)
    assert result.segments_sec == []
    assert result.fallback_required is True
    assert result.threshold is None
    assert result.diagnostics["reason"] == reason


def test_coverage_cap_cannot_reintroduce_a_sub_two_second_segment():
    segments, _selected, diagnostics = select_otsu_segments(
        make_curve([0.9, 0.9, 0.9]),
        3.0,
        0.5,
        min_segment_sec=2.0,
    )
    # 60% of three seconds is 1.8 seconds, so the outer fallback must decide
    # how to handle this short video rather than violating either guard.
    assert segments == []
    assert diagnostics["coverage_limit_sec"] == pytest.approx(1.8)
