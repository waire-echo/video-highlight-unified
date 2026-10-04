from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from video_highlight.pgl_sum_frontend import (
    FeatureSequence,
    MrHiSumPGLTemporalDetector,
    YouTube8MFeatureExtractor,
    build_boundary_records,
    load_cached_semantic_hints,
    refine_event_boundaries,
    select_pgl_event_candidates,
)
from video_highlight.pgl_sum_model import MrHiSumPGLScorer, PGL_SUM
from video_highlight.temporal import TemporalResult, build_temporal_detector


def test_pgl_network_matches_checkpoint_shape_and_supports_short_sequences(tmp_path):
    model = PGL_SUM().eval()
    checkpoint = tmp_path / "pgl.pkl"
    torch.save(model.state_dict(), checkpoint)
    scorer = MrHiSumPGLScorer(str(checkpoint), device="cpu")

    for length in (1, 2, 5, 9, 10, 12):
        scores = scorer.score(np.zeros((length, 1024), dtype=np.float32))
        assert scores.shape == (length,)
        assert np.all(np.isfinite(scores))
        assert np.all((0.0 <= scores) & (scores <= 1.0))


def test_pgl_checkpoint_mismatch_is_rejected(tmp_path):
    checkpoint = tmp_path / "bad.pkl"
    torch.save({"linear_2.weight": torch.zeros((1, 1024))}, checkpoint)
    with pytest.raises(ValueError, match="architecture mismatch"):
        MrHiSumPGLScorer(str(checkpoint), device="cpu")


def test_youtube8m_quantize_dequantize_matches_training_record_roundtrip():
    values = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    result = YouTube8MFeatureExtractor._quantize_dequantize(values)
    expected_bytes = np.rint((np.clip(values, -2, 2) + 2) * 255 / 4).astype(np.uint8)
    expected = expected_bytes.astype(np.float32) * 4 / 255 - 2
    np.testing.assert_allclose(result, expected, atol=1e-7)
    assert float(result.min()) == -2.0
    assert float(result.max()) == 2.0


def test_youtube8m_feature_cache_is_reused(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    cache_dir = tmp_path / "cache"
    model_dir.mkdir()
    for name in YouTube8MFeatureExtractor.REQUIRED_FILES:
        (model_dir / name).write_bytes(name.encode("ascii"))
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")

    monkeypatch.setattr(
        "video_highlight.pgl_sum_frontend.video_meta",
        lambda _path: (50, 10.0, 64, 64),
    )
    frames = {index: np.zeros((8, 8, 3), dtype=np.uint8) + index for index in range(0, 50, 10)}
    monkeypatch.setattr(
        "video_highlight.pgl_sum_frontend.extract_frames",
        lambda _path, frame_ids: {frame_id: frames[frame_id] for frame_id in frame_ids},
    )
    extractor = YouTube8MFeatureExtractor(
        str(model_dir), str(cache_dir), require_e_drive=False
    )
    calls = []

    def fake_feature(frame):
        calls.append(int(frame[0, 0, 0]))
        return np.full((1024,), len(calls), dtype=np.float32)

    monkeypatch.setattr(extractor, "_frame_feature", fake_feature)
    first = extractor.extract(str(video))
    second = extractor.extract(str(video))
    assert first.features.shape == (5, 1024)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == [0, 10, 20, 30, 40]
    np.testing.assert_array_equal(first.features, second.features)


def test_pgl_candidate_selection_keeps_separated_peaks_without_coverage_target():
    scores = [0.05, 0.10, 0.20, 0.95, 0.20, 0.08, 0.06, 0.12, 0.85, 0.12, 0.05, 0.04]
    timestamps = list(range(len(scores)))
    candidates, curve, diagnostics = select_pgl_event_candidates(
        scores, timestamps, duration_sec=12.0
    )
    assert len(candidates) == 2
    assert [candidate["index"] for candidate in candidates] == [3, 8]
    assert candidates[0]["initial_end_sec"] <= candidates[1]["initial_start_sec"]
    assert diagnostics["ranking_signal"] == "pgl_only"
    assert len(curve) == len(scores)
    assert all(row["fused_score"] == row["smoothed_score"] for row in curve)


def test_pgl_candidate_selection_uses_global_max_for_flat_curve():
    candidates, _curve, diagnostics = select_pgl_event_candidates(
        [0.5] * 5, [0, 1, 2, 3, 4], duration_sec=5.0
    )
    assert len(candidates) == 1
    assert diagnostics["normalization"]["degenerate"] is True
    assert diagnostics["peak_selection"]["used_global_max_fallback"] is True


def test_pgl_endpoint_requires_real_one_sided_prominence_not_artificial_padding():
    # The weak first point used to become a peak only because the curve was
    # padded with an artificial value below it.  The strong final event should
    # win without recreating the historical fixed-intro bias.
    candidates, _curve, diagnostics = select_pgl_event_candidates(
        [0.31, 0.20, 0.10, 0.30, 0.95, 0.70],
        [0, 1, 2, 3, 4, 5],
        duration_sec=6.0,
    )
    assert all(candidate["index"] != 0 for candidate in candidates)
    assert candidates[0]["index"] in {4, 5}
    assert diagnostics["peak_selection"]["endpoint_policy"]["artificial_padding"] is False


def test_pgl_strong_endpoint_is_allowed_and_competes_with_interior_peaks():
    scores = [0.98, 0.82, 0.25, 0.10, 0.20, 0.90, 0.15, 0.10, 0.05]
    candidates, _curve, _diagnostics = select_pgl_event_candidates(
        scores, list(range(len(scores))), duration_sec=9.0
    )
    assert [candidate["index"] for candidate in candidates] == [0, 5]


def test_boundary_refinement_uses_signals_only_inside_selected_ownership():
    candidates = [
        {
            "index": 3,
            "score": 0.9,
            "prominence": 0.6,
            "rank": 1,
            "time_sec": 3.0,
            "ownership_start_sec": 0.0,
            "ownership_end_sec": 5.0,
            "initial_start_sec": 2.0,
            "initial_end_sec": 4.0,
        },
        {
            "index": 8,
            "score": 0.8,
            "prominence": 0.5,
            "rank": 2,
            "time_sec": 8.0,
            "ownership_start_sec": 5.0,
            "ownership_end_sec": 10.0,
            "initial_start_sec": 7.0,
            "initial_end_sec": 9.0,
        },
    ]
    records = []
    for index in range(40):
        start = index * 0.25
        records.append(
            {
                "start_sec": start,
                "end_sec": start + 0.25,
                "boundary_activity": 0.8 if 1.75 <= start < 4.25 or 7 <= start < 9 else 0.0,
                "strong_scene_boundary": start + 0.25 in {5.0},
            }
        )
    refined, diagnostics = refine_event_boundaries(candidates, records, 10.0)
    assert len(refined) == 2
    assert refined[0]["refined_end_sec"] <= 5.0
    assert refined[1]["refined_start_sec"] >= 5.0
    assert diagnostics["ranking_signal"] == "pgl_only"
    assert diagnostics["motion_audio_role"] == "boundary_only"


def test_boundary_records_handle_static_video_and_missing_audio_without_nan():
    class StaticSignals:
        def score_windows(self, _video_path, windows):
            output = []
            for index, window in enumerate(windows):
                output.append(
                    {
                        **window,
                        "motion_score": 1.0,
                        "audio_score": None,
                        "scene_score": 0.9 if index == 2 else 0.0,
                        "yolo_interaction_score": None,
                    }
                )
            return output

    records = build_boundary_records(
        "static-no-audio.mp4", 1.0, StaticSignals(), grid_sec=0.25
    )
    assert len(records) == 4
    assert all(record["normalized_motion_score"] == 0.0 for record in records)
    assert all(record["normalized_audio_score"] is None for record in records)
    assert all(
        value is None or np.isfinite(value)
        for record in records
        for value in (
            record["normalized_motion_score"],
            record["normalized_audio_score"],
            record["normalized_scene_score"],
        )
    )
    assert [index for index, record in enumerate(records) if record["strong_scene_boundary"]] == [2]


class FakeFeatureExtractor:
    sample_fps = 1.0

    def extract(self, _video_path):
        return FeatureSequence(
            features=np.zeros((12, 1024), dtype=np.float32),
            timestamps_sec=np.arange(12, dtype=np.float64),
            frame_ids=np.arange(12, dtype=np.int64) * 25,
            cache_path="E:/cache/demo.npz",
            cache_hit=True,
            extractor_fingerprint={"extractor": "fake"},
        )

    def close(self):
        pass


class FakeScorer:
    fingerprint = {"model": "fake-pgl"}

    def score(self, _features):
        return np.asarray(
            [0.05, 0.10, 0.20, 0.95, 0.20, 0.08, 0.06, 0.12, 0.85, 0.12, 0.05, 0.04]
        )

    def close(self):
        pass


class FakeSignals:
    def score_windows(self, _video_path, windows):
        output = []
        for window in windows:
            start = float(window["start_sec"])
            active = 1.0 if 2 <= start < 4 or 7 <= start < 9 else 0.0
            output.append(
                {
                    **window,
                    "motion_score": active,
                    "audio_score": None,
                    "scene_score": 0.0,
                    "yolo_interaction_score": None,
                }
            )
        return output

    def close(self):
        pass


def test_detector_emits_temporal_only_diagnostics_and_whitelisted_cached_hint(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.pgl_sum_frontend.video_meta",
        lambda _path: (300, 25.0, 1920, 1080),
    )
    detector = MrHiSumPGLTemporalDetector(
        scorer=FakeScorer(),
        feature_extractor=FakeFeatureExtractor(),
        signal_extractor=FakeSignals(),
        cached_hints={"7": {"subject_hint": "dog", "action_description": "dog jumps"}},
        require_e_drive=False,
    )
    result = detector.detect("7.mp4")
    assert result.valid is True
    assert result.source == "mrhisum_pgl_sum"
    assert len(result.segments_sec) == 2
    assert result.subject_hint == "dog"
    assert result.action_description == "dog jumps"
    assert result.score_diagnostics["qwen_used_for_timing"] is False
    assert result.score_diagnostics["motion_audio_scene_used_for_ranking"] is False
    assert all("bbox" not in str(row).lower() for row in result.score_curve)


def test_detector_isolates_pgl_failure_with_motion_fallback(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.pgl_sum_frontend.video_meta",
        lambda _path: (100, 25.0, 100, 100),
    )

    class BrokenExtractor:
        def extract(self, _path):
            raise RuntimeError("decode failed")

        def close(self):
            pass

    class Fallback:
        def detect(self, _path):
            return TemporalResult([(1.0, 3.0)], source="motion", valid=True)

    detector = MrHiSumPGLTemporalDetector(
        scorer=FakeScorer(),
        feature_extractor=BrokenExtractor(),
        signal_extractor=FakeSignals(),
        fallback_detector=Fallback(),
        require_e_drive=False,
    )
    result = detector.detect("3.mp4")
    assert result.segments_sec == [(1.0, 3.0)]
    assert result.source == "pgl_sum_fallback:motion"
    assert result.fallback_used == "motion"
    assert "decode failed" in result.errors[-1]


def test_cached_hints_are_whitelisted(tmp_path):
    path = tmp_path / "hints.jsonl"
    path.write_text(
        '{"video_id":"1","subject_hint":"dog","action_description":"jump"}\n'
        '{"video_id":"2","subject_hint":"skier","action_description":"finish"}\n',
        encoding="utf-8",
    )
    hints = load_cached_semantic_hints(str(path))
    assert hints["1"] == {"subject_hint": "dog", "action_description": "jump"}
    assert hints["2"] == {"subject_hint": None, "action_description": "finish"}


def test_builder_constructs_pgl_mode(monkeypatch, tmp_path):
    checkpoint = tmp_path / "pgl.pkl"
    checkpoint.write_bytes(b"checkpoint")
    model_dir = tmp_path / "yt8m"
    model_dir.mkdir()
    cache_dir = tmp_path / "cache"
    captured = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "video_highlight.pgl_sum_frontend.MrHiSumPGLTemporalDetector",
        FakeDetector,
    )
    args = SimpleNamespace(
        temporal_mode="pgl_sum",
        pgl_checkpoint_path=str(checkpoint),
        pgl_feature_model_dir=str(model_dir),
        pgl_feature_cache_dir=str(cache_dir),
        pgl_sample_fps=1.0,
        pgl_boundary_grid_sec=0.25,
        pgl_device="cpu",
        pgl_hints_jsonl="",
        window_ffmpeg_path="ffmpeg",
        window_audio_sample_rate=8000,
    )
    detector = build_temporal_detector(args)
    assert isinstance(detector, FakeDetector)
    assert captured["checkpoint_path"] == str(checkpoint)
    assert captured["sample_fps"] == 1.0
