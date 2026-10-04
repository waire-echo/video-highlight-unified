from video_highlight.temporal_reaggregation import (
    ReaggregationProfile,
    canonical_subject_hint,
    reaggregate_curve,
    reaggregate_records,
    rebuild_score_curve,
    summarize_reaggregation,
)


def _window(index, start, end, qwen, *, stage="coarse", motion=0.0, audio=0.0):
    return {
        "window_id": f"{stage}-{index}",
        "start_sec": start,
        "end_sec": end,
        "stage": stage,
        "valid": True,
        "qwen_score": qwen,
        "qwen_disabled": False,
        "normalized_motion_score": motion,
        "normalized_audio_score": audio,
        "fused_score": qwen,
    }


def _raw(video_id="0"):
    scores = [0.10, 0.90, 0.10, 0.85, 0.10]
    return {
        "video_id": video_id,
        "duration_sec": 10.0,
        "fps": 10.0,
        "frame_count": 100,
        "window_records": [
            _window(index, index * 2.0, (index + 1) * 2.0, score)
            for index, score in enumerate(scores)
        ],
        "temporal_segments_sec": [[0.0, 6.0]],
    }


def _handoff(video_id="0"):
    return {
        "schema_version": "temporal-handoff/v1",
        "video_id": video_id,
        "targetRatioWH": [16, 9],
        "fps": 10.0,
        "frame_count": 100,
        "duration_sec": 10.0,
        "segments_sec": [[0.0, 6.0]],
        "segments_frame": [[0, 60]],
        "subject_hint": "Two women",
        "action_description": "running",
        "source": "window_score_curve",
        "valid": True,
        "fallback_used": None,
        "errors": [],
        "warnings": [],
    }


def test_reaggregation_keeps_separated_relative_high_score_islands():
    profile = ReaggregationProfile(
        name="test",
        top_fraction=0.40,
        absolute_threshold=0.45,
        merge_gap_sec=0.0,
        padding_sec=0.0,
        max_coverage_ratio=0.80,
    )

    result = reaggregate_curve(_raw(), profile)

    assert result["fallback_used"] is None
    assert len(result["segments_sec"]) == 2
    assert result["segments_sec"][0][1] < result["segments_sec"][1][0]
    assert result["score_diagnostics"]["local_peak_seed_mode"] is False


def test_rebuild_curve_uses_traditional_signal_when_qwen_is_disabled():
    raw = _raw()
    raw["window_records"] = [
        {
            **_window(0, 0.0, 10.0, 1.0, motion=0.75, audio=0.25),
            "qwen_disabled": True,
        }
    ]
    profile = ReaggregationProfile(name="test")

    curve = rebuild_score_curve(raw, profile)

    assert curve
    # With Qwen excluded, 0.75 motion and 0.25 audio are renormalised over
    # their 0.15/0.05 weights: (0.75*3 + 0.25)/4 = 0.625.
    assert abs(curve[0]["score"] - 0.625) < 1e-9


def test_empty_threshold_selection_falls_back_to_top_peak_window():
    profile = ReaggregationProfile(
        name="test",
        absolute_threshold=0.99,
        top_fraction=0.10,
        fallback_sec=4.0,
    )

    result = reaggregate_curve(_raw(), profile)

    assert result["fallback_used"] == "top_peak_window"
    assert len(result["segments_sec"]) == 1
    assert abs(result["segments_sec"][0][1] - result["segments_sec"][0][0] - 4.0) < 1e-9


def test_reaggregate_records_updates_frames_and_normalises_subject_hint():
    profile = ReaggregationProfile(
        name="test",
        top_fraction=0.40,
        absolute_threshold=0.45,
        merge_gap_sec=0.0,
        padding_sec=0.0,
        max_coverage_ratio=0.80,
    )

    handoff, raw = reaggregate_records([_handoff()], [_raw()], profile)

    assert handoff[0]["valid"] is True
    assert handoff[0]["subject_hint"] == "person"
    assert handoff[0]["source"] == "window_score_reaggregated_test"
    assert len(handoff[0]["segments_sec"]) == len(handoff[0]["segments_frame"]) == 2
    assert raw[0]["original_temporal_segments_sec"] == [[0.0, 6.0]]
    assert raw[0]["score_diagnostics"]["profile"]["name"] == "test"


def test_subject_hint_whitelist_and_summary():
    assert canonical_subject_hint("Runners") == "person"
    assert canonical_subject_hint("Teddy bear") == "animal"
    assert canonical_subject_hint("door") is None
    profile = ReaggregationProfile(name="test", max_coverage_ratio=0.8)
    records = [
        {
            **_handoff(),
            "segments_sec": [[1.0, 3.0], [6.0, 8.0]],
            "valid": True,
            "fallback_used": None,
        }
    ]

    summary = summarize_reaggregation(records, profile)

    assert summary["videos"] == 1
    assert summary["segments_total"] == 2
    assert summary["multi_segment_videos"] == 1
    assert summary["coverage_median"] == 0.4
