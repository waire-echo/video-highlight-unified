import math
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from video_highlight import temporal_signals
from video_highlight.score_curve import (
    build_score_curve,
    detect_tail_barrier,
    fallback_segments,
    find_local_peaks,
    overlay_score_curves,
    score_windows_to_segments,
    select_highlight_segments,
    select_multisignal_refinement_regions,
    select_refinement_regions,
    smooth_score_curve,
)
from video_highlight.temporal_signals import (
    TemporalSignalExtractor,
    apply_signal_calibration,
    compute_audio_score,
    compute_motion_score,
    compute_scene_score,
    extract_window_signals,
    fit_signal_calibration,
    fuse_window_scores,
    normalize_signal_records,
)


@dataclass
class ExampleWindow:
    start_sec: float
    end_sec: float
    qwen_score: float
    motion_score: float
    audio_score: float = None
    scene_score: float = None
    yolo_interaction_score: float = None


def test_signal_normalization_accepts_dataclasses_and_never_emits_nan():
    rows, diagnostics = normalize_signal_records(
        [
            ExampleWindow(0.0, 2.0, 0.5, 4.0),
            ExampleWindow(1.0, 3.0, 0.5, 8.0),
        ]
    )
    assert all(isinstance(row, dict) for row in rows)
    assert [row["normalized_motion_score"] for row in rows] == [0.0, 1.0]
    assert [row["normalized_qwen_score"] for row in rows] == [0.5, 0.5]
    assert diagnostics["signals"]["qwen_score"]["constant"] is True
    for row in rows:
        for key, value in row.items():
            if key.startswith("normalized_") and value is not None:
                assert math.isfinite(value)


def test_fusion_reweights_missing_channels_and_disables_constant_qwen():
    missing = fuse_window_scores(
        [
            {
                "start_sec": 0,
                "end_sec": 2,
                "qwen_score": 0.9,
                "motion_score": 0.2,
                "audio_score": None,
            }
        ],
        normalize=False,
        disable_degenerate_qwen=False,
    )[0]
    assert missing["fused_score"] == pytest_approx((0.9 * 0.5 + 0.2 * 0.2) / 0.7)
    assert missing["fusion_weight_sum"] == pytest_approx(0.7)

    degenerate = fuse_window_scores(
        [
            {"start_sec": 0, "end_sec": 2, "qwen_score": 0.75, "motion_score": 1},
            {"start_sec": 1, "end_sec": 3, "qwen_score": 0.75, "motion_score": 3},
        ]
    )
    assert all(row["qwen_disabled"] for row in degenerate)
    assert degenerate[0]["fused_score"] == 0.0
    assert degenerate[1]["fused_score"] == 1.0
    assert all("qwen_score" not in row["fusion_available_signals"] for row in degenerate)

    nearly_constant = fuse_window_scores(
        [
            {"start_sec": 0, "end_sec": 2, "qwen_score": 0.75, "motion_score": 0.1},
            {"start_sec": 1, "end_sec": 3, "qwen_score": 0.80, "motion_score": 0.9},
        ]
    )
    assert all(row["qwen_disabled"] for row in nearly_constant)

    invalid = fuse_window_scores(
        [
            {
                "start_sec": 0,
                "end_sec": 2,
                "valid": False,
                "qwen_score": 1.0,
                "motion_score": 0.4,
            }
        ],
        normalize=False,
    )[0]
    assert invalid["fusion_available_signals"] == ["motion_score"]
    assert invalid["fused_score"] == pytest_approx(0.4)


def test_integer_qwen_components_keep_zero_to_three_scale_and_weights():
    ones = fuse_window_scores(
        [{"start_sec": 0, "end_sec": 2, "qwen_components": {
            "action": 1, "completion": 1, "interaction": 1, "event": 1,
            "emotion": 1, "visual_impact": 1, "highlight": 1,
        }}],
        normalize=False,
        disable_degenerate_qwen=False,
    )[0]
    twos = fuse_window_scores(
        [{"start_sec": 0, "end_sec": 2, "qwen_components": {
            "action": 2, "completion": 2, "interaction": 2, "event": 2,
            "emotion": 2, "visual_impact": 2, "highlight": 2,
        }}],
        normalize=False,
        disable_degenerate_qwen=False,
    )[0]
    assert ones["qwen_score"] == pytest_approx(1.0 / 3.0)
    assert twos["qwen_score"] == pytest_approx(2.0 / 3.0)
    assert ones["qwen_score"] < twos["qwen_score"]


def test_constant_traditional_channel_is_removed_from_fusion_denominator():
    rows = fuse_window_scores(
        [
            {"start_sec": 0, "end_sec": 2, "qwen_score": 0.2, "scene_score": 0.4},
            {"start_sec": 1, "end_sec": 3, "qwen_score": 0.8, "scene_score": 0.4},
        ]
    )
    assert all("scene_score" not in row["fusion_available_signals"] for row in rows)


def pytest_approx(value, abs=1e-9):
    # Avoid adding a module-level pytest dependency just for two float assertions.
    import pytest

    return pytest.approx(value, abs=abs)


def test_signal_extractors_bound_scores_and_handle_missing_audio(monkeypatch):
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(
        temporal_signals,
        "_read_window_frames",
        lambda *_args, **_kwargs: [black, white],
    )
    motion = compute_motion_score("video.mp4", 0.0, 2.0)
    scene = compute_scene_score("video.mp4", 0.0, 2.0)
    assert motion is not None and 0.0 < motion <= 1.0
    assert scene is not None and 0.0 < scene <= 1.0

    samples = (np.sin(np.linspace(0, 20, 4000)) * 12000).astype(np.int16).tobytes()
    monkeypatch.setattr(
        temporal_signals.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, samples, b""),
    )
    audio = compute_audio_score("video.mp4", 0.0, 2.0)
    assert audio is not None and 0.0 <= audio <= 1.0

    monkeypatch.setattr(
        temporal_signals.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b"no stream"),
    )
    assert compute_audio_score("video.mp4", 0.0, 2.0) is None


def test_extract_window_signals_isolates_one_signal_failure():
    def broken(*_args):
        raise RuntimeError("decoder failed")

    record = extract_window_signals(
        "video.mp4",
        1.0,
        3.0,
        motion_fn=broken,
        audio_fn=lambda *_args: 0.4,
        scene_fn=lambda *_args: 0.2,
    )
    assert record["motion_score"] is None
    assert "decoder failed" in record["motion_error"]
    assert record["audio_score"] == 0.4
    assert record["scene_score"] == 0.2
    assert record["yolo_interaction_score"] is None


def test_batch_extractor_decodes_visual_and_audio_once(monkeypatch):
    calls = {"visual": 0, "audio": 0, "sample_rate": None}

    def visual(_path, windows, _fps):
        calls["visual"] += 1
        return [
            {"motion_score": index / 10.0, "scene_score": index / 20.0}
            for index, _window in enumerate(windows)
        ]

    def audio(_path, *, ffmpeg_bin, sample_rate, **_kwargs):
        calls["audio"] += 1
        calls["sample_rate"] = sample_rate
        return np.concatenate(
            [np.zeros(sample_rate, dtype=np.float32), np.ones(sample_rate, dtype=np.float32) * 0.5]
        )

    monkeypatch.setattr(temporal_signals, "_batch_visual_signals", visual)
    monkeypatch.setattr(temporal_signals, "_decode_audio_pcm", audio)
    extractor = TemporalSignalExtractor(audio_sample_rate=6000)
    rows = extractor.score_windows(
        "video.mp4",
        [
            {"start_sec": 0.0, "end_sec": 2.0},
            {"start_sec": 1.0, "end_sec": 3.0},
        ],
    )
    fine_rows = extractor.score_windows(
        "video.mp4",
        [{"start_sec": 0.5, "end_sec": 1.5, "sample_fps": 4.0}],
    )
    assert calls == {"visual": 2, "audio": 1, "sample_rate": 6000}
    assert rows[0]["audio_score"] is not None
    assert rows[1]["motion_score"] == 0.1
    assert fine_rows[0]["signal_sample_fps"] == 2.0


def test_score_curve_averages_overlap_and_conservatively_fills_gaps():
    curve = build_score_curve(
        [
            {"start_sec": 0.0, "end_sec": 2.0, "fused_score": 0.2},
            {"start_sec": 1.0, "end_sec": 3.0, "fused_score": 0.8},
        ],
        3.0,
    )
    assert len(curve) == 6
    assert curve[0]["score"] == pytest_approx(0.2)
    assert curve[2]["score"] == pytest_approx(0.5)
    assert curve[5]["score"] == pytest_approx(0.8)

    gap = build_score_curve(
        [
            {"start_sec": 0.0, "end_sec": 0.5, "score": 0.8},
            {"start_sec": 1.0, "end_sec": 1.5, "score": 0.2},
        ],
        1.5,
    )
    assert gap[1]["filled"] is True
    assert gap[1]["score"] == pytest_approx(0.2)


def test_three_point_smoothing_and_refinement_overlay():
    curve = [
        {"start_sec": 0.0, "end_sec": 0.5, "score": 0.0},
        {"start_sec": 0.5, "end_sec": 1.0, "score": 1.0},
        {"start_sec": 1.0, "end_sec": 1.5, "score": 0.0},
    ]
    smoothed = smooth_score_curve(curve)
    assert smoothed[0]["score"] == pytest_approx(0.5)
    assert smoothed[1]["score"] == pytest_approx(1.0 / 3.0)

    overlaid = overlay_score_curves(
        curve,
        [{"start_sec": 0.5, "end_sec": 1.0, "score": 0.9}],
    )
    assert overlaid[0]["refined"] is False
    assert overlaid[1]["refined"] is True
    assert overlaid[1]["score"] == pytest_approx(0.99125)
    assert overlaid[1]["coarse_score"] == 1.0
    assert overlaid[1]["fine_score"] == 0.9
    assert overlaid[1]["refine_weight"] == pytest_approx(0.0875)


def test_selection_removes_isolated_bin_merges_small_gap_and_pads():
    curve = []
    for index in range(20):
        start = index * 0.5
        score = 0.9 if index in {0, 4, 5, 7, 8} else 0.1
        curve.append({"start_sec": start, "end_sec": start + 0.5, "score": score})
    selected = select_highlight_segments(
        curve,
        10.0,
        top_fraction=1.0,
    )
    # 0--0.5 is discarded (<1s); 2--3 and 3.5--4.5 merge across a 0.5s gap,
    # then the default 0.75s padding is applied to each side.
    assert selected["segments_sec"] == [(1.25, 5.25)]
    assert selected["score_diagnostics"]["removed_short_segment_count"] == 1


def test_selection_enforces_segment_count_and_total_coverage_cap():
    curve = [
        {"start_sec": index * 0.5, "end_sec": (index + 1) * 0.5, "score": 0.9}
        for index in range(40)
    ]
    selected = select_highlight_segments(
        curve,
        20.0,
        score_threshold=0.0,
        top_fraction=1.0,
        padding_sec=0.0,
    )
    coverage = sum(end - start for start, end in selected["segments_sec"])
    assert coverage == pytest_approx(12.0)
    assert selected["score_diagnostics"]["coverage_limit_sec"] == pytest_approx(12.0)


def test_local_peaks_and_fine_regions_use_configured_cap():
    curve = []
    for index in range(30):
        score = 0.9 if index % 4 == 1 else 0.1
        curve.append(
            {
                "start_sec": index * 0.5,
                "end_sec": (index + 1) * 0.5,
                "time_sec": (index + 0.5) * 0.5,
                "score": score,
            }
        )
    peaks = find_local_peaks(curve, max_peaks=3)
    regions = select_refinement_regions(curve, 15.0, padding_sec=0.0, max_regions=6)
    assert len(peaks) == 3
    assert len(regions) == 6
    assert select_refinement_regions(curve, 15.0, max_regions=0) == []


def test_fallback_uses_full_short_video_then_middle_window():
    short = fallback_segments([], 3.5)
    long = fallback_segments([], 10.0)
    assert short["source"] == "short_video_full"
    assert short["segments_sec"] == [(0.0, 3.5)]
    assert long["source"] == "middle"
    assert long["segments_sec"] == [(3.0, 7.0)]

    static_short = fallback_segments(
        [{"start_sec": 0.0, "end_sec": 3.0, "motion_score": 0.0}], 3.0
    )
    static_long = fallback_segments(
        [
            {"start_sec": 0.0, "end_sec": 2.0, "motion_score": 0.0},
            {"start_sec": 8.0, "end_sec": 10.0, "motion_score": 0.0},
        ],
        10.0,
    )
    assert static_short["source"] == "short_video_full"
    assert static_short["segments_sec"] == [(0.0, 3.0)]
    assert static_long["source"] == "middle"
    assert static_long["segments_sec"] == [(3.0, 7.0)]


def test_coverage_cap_never_reintroduces_subsecond_segment():
    curve = [
        {"start_sec": 0.0, "end_sec": 0.5, "score": 0.9},
        {"start_sec": 0.5, "end_sec": 1.0, "score": 0.9},
        {"start_sec": 1.0, "end_sec": 1.2, "score": 0.9},
    ]
    selected = select_highlight_segments(
        curve,
        1.2,
        score_threshold=0.0,
        top_fraction=1.0,
        padding_sec=0.0,
    )
    assert selected["segments_sec"] == []


def test_high_level_curve_pipeline_uses_motion_when_qwen_is_missing():
    records = [
        {"start_sec": 0.0, "end_sec": 2.0, "motion_score": 0.1},
        {"start_sec": 4.0, "end_sec": 6.0, "motion_score": 0.9},
    ]
    result = score_windows_to_segments(records, 8.0)
    assert result["segments_sec"]
    assert result["source"] in {"traditional_signals", "motion"}
    assert result["score_curve"]


def test_qwen_keeps_fixed_scale_and_fine_reuses_coarse_percentiles():
    qwen_rows, _diagnostics = normalize_signal_records(
        [{"qwen_score": 0.51}, {"qwen_score": 0.52}], keys=("qwen_score",)
    )
    assert [row["normalized_qwen_score"] for row in qwen_rows] == [0.51, 0.52]

    coarse = [{"motion_score": float(value)} for value in range(10)]
    calibration = fit_signal_calibration(coarse, keys=("motion_score",))
    fine, diagnostics = apply_signal_calibration(
        [{"motion_score": 4.5}, {"motion_score": 4.6}, {"motion_score": 100.0}],
        calibration,
        keys=("motion_score",),
    )
    assert fine[0]["normalized_motion_score"] == pytest_approx(0.5)
    assert 0.5 < fine[1]["normalized_motion_score"] < 0.52
    assert fine[2]["normalized_motion_score"] == 1.0
    assert diagnostics["signals"]["motion_score"]["low"] == pytest_approx(0.9)
    assert diagnostics["signals"]["motion_score"]["high"] == pytest_approx(8.1)


def test_scene_is_boundary_only_and_audio_scores_changes_not_loudness():
    fused = fuse_window_scores(
        [{"motion_score": 0.2, "scene_score": 1.0}],
        normalize=False,
        disable_degenerate_qwen=False,
    )[0]
    assert fused["fusion_available_signals"] == ["motion_score"]
    assert fused["fused_score"] == pytest_approx(0.2)

    constant = np.full(8000 * 2, 0.25, dtype=np.float32)
    onset = np.concatenate(
        [np.zeros(8000, dtype=np.float32), np.full(8000, 0.25, dtype=np.float32)]
    )
    constant_score = temporal_signals._audio_score_from_samples(constant, 8000)
    onset_score = temporal_signals._audio_score_from_samples(onset, 8000)
    assert constant_score == pytest_approx(0.0)
    assert onset_score is not None and onset_score > constant_score


def test_conservative_overlay_caps_under_supported_conflict():
    overlaid = overlay_score_curves(
        [{"start_sec": 0.0, "end_sec": 0.5, "score": 0.9}],
        [
            {
                "start_sec": 0.0,
                "end_sec": 0.5,
                "score": 0.1,
                "observation_count": 2,
            }
        ],
    )[0]
    assert overlaid["refine_weight"] == pytest_approx(0.15)
    assert overlaid["final_score"] == pytest_approx(0.78)
    assert "coarse_fine_conflict_limited" in overlaid["conflict_reason"]


def test_multisignal_refinement_keeps_independent_events():
    def curve_with_peak(peak_index):
        return [
            {
                "start_sec": index * 0.5,
                "end_sec": (index + 1) * 0.5,
                "score": 0.9 if index == peak_index else 0.1,
            }
            for index in range(20)
        ]

    regions = select_multisignal_refinement_regions(
        {"fused": curve_with_peak(3), "audio_change": curve_with_peak(15)},
        10.0,
        padding_sec=0.0,
        merge_gap_sec=0.0,
    )
    assert regions == [(1.5, 2.0), (7.5, 8.0)]


def test_local_peak_seeds_grow_to_lower_boundary_and_keep_second_event():
    scores = [0.1, 0.41, 0.46, 0.41, 0.1, 0.1, 0.41, 0.47, 0.41, 0.1]
    curve = [
        {"start_sec": i * 0.5, "end_sec": (i + 1) * 0.5, "score": score}
        for i, score in enumerate(scores)
    ]
    selected = select_highlight_segments(
        curve,
        5.0,
        score_threshold=0.45,
        growth_threshold=0.40,
        use_local_peak_seeds=True,
        padding_sec=0.0,
    )
    assert selected["segments_sec"] == [(0.5, 2.0), (3.0, 4.5)]


def test_tail_barrier_requires_cut_then_low_motion_and_semantics():
    records = []
    for index in range(20):
        records.append(
            {
                "start_sec": index * 0.5,
                "end_sec": (index + 1) * 0.5,
                "valid": True,
                "qwen_score": 0.1 if index >= 17 else 0.6,
                "normalized_motion_score": 0.1 if index >= 17 else 0.6,
                "normalized_scene_score": 0.9 if index == 16 else 0.1,
            }
        )
    barrier = detect_tail_barrier(records, 10.0)
    assert barrier["barrier_sec"] == 8.5
    assert "strong_tail_cut" in barrier["reason"]
