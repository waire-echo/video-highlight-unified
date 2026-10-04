import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from video_highlight.window_score_frontend import WindowScoreFrontend
from video_highlight.window_scoring import WindowScore, WindowScoreTemporalDetector
from video_highlight.windowing import WindowSpec


class FakeScorer:
    def __init__(self, score_fn=None, fail=False):
        self.score_fn = score_fn or (lambda _start, _end: 0.2)
        self.fail = fail
        self.calls = []
        self.closed = False

    def score_window(
        self,
        _video_path,
        start_sec,
        end_sec,
        sample_fps,
        *,
        window_id=0,
        stage="coarse",
        parent_id=None,
    ):
        self.calls.append((start_sec, end_sec, sample_fps, stage))
        if self.fail:
            return WindowScore(
                window_id,
                start_sec,
                end_sec,
                sample_fps,
                valid=False,
                errors=["invalid JSON"],
                attempts=2,
                stage=stage,
                parent_id=parent_id,
            )
        score = float(self.score_fn(start_sec, end_sec))
        return WindowScore(
            window_id,
            start_sec,
            end_sec,
            sample_fps,
            qwen_score=score,
            main_subject="dog",
            action_description="dog jumps toward a person",
            evidence="visible jump and interaction",
            raw="{}",
            valid=True,
            attempts=1,
            stage=stage,
            parent_id=parent_id,
        )

    def close(self):
        self.closed = True


class ExplodingScorer(FakeScorer):
    def score_window(self, *_args, **_kwargs):
        self.calls.append((0.0, 0.0, 0.0, "coarse"))
        raise RuntimeError("one decoder window failed")


class BatchScorer(FakeScorer):
    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def score_windows(self, video_path, windows):
        self.batch_calls.append((video_path, len(windows)))
        return [
            self.score_window(
                video_path,
                window.start_sec,
                window.end_sec,
                window.sample_fps,
                window_id=window.window_id,
                stage=window.stage,
                parent_id=window.parent_id,
            )
            for window in windows
        ]


class ResourceFailScorer(FakeScorer):
    def score_window(
        self,
        _video_path,
        start_sec,
        end_sec,
        sample_fps,
        *,
        window_id=0,
        stage="coarse",
        parent_id=None,
    ):
        self.calls.append((start_sec, end_sec, sample_fps, stage))
        return WindowScore(
            window_id,
            start_sec,
            end_sec,
            sample_fps,
            valid=False,
            errors=["model load failed: CUDA out of memory"],
            attempts=2,
            stage=stage,
            parent_id=parent_id,
        )


def _signals(_video_path, window):
    centre = (window.start_sec + window.end_sec) / 2.0
    return {
        "motion_score": 0.9 if 2.0 <= centre <= 6.0 else 0.1,
        "audio_score": None,
        "scene_score": 0.2,
        "yolo_interaction_score": None,
    }


def test_frontend_uses_available_batch_scorer():
    scorer = BatchScorer()
    frontend = WindowScoreFrontend(scorer, signal_provider=_signals)
    windows = [
        WindowSpec(0, 0.0, 2.0, 2.0),
        WindowSpec(1, 1.0, 3.0, 2.0),
    ]

    records, stopped = frontend._score_windows("mock.mp4", windows)

    assert stopped is None
    assert len(records) == 2
    assert scorer.batch_calls == [("mock.mp4", 2)]


def test_frontend_runs_coarse_and_fine_video_windows(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (240, 30.0, 1280, 720),
    )
    scorer = FakeScorer(
        lambda start, end: 0.9 if min(end, 6.0) > max(start, 2.0) else 0.2
    )
    detector = WindowScoreFrontend(
        scorer, signal_provider=_signals, growth_threshold=0.37
    )

    result = detector.detect("video.mp4")

    assert result.valid
    assert result.source == "window_score_curve"
    assert result.segments_sec
    assert any(call[2] == 4.0 and call[3] == "fine" for call in scorer.calls)
    assert result.score_diagnostics["coarse_window_count"] == 7
    assert result.score_diagnostics["fine_window_count"] > 0
    assert any(record["stage"] == "fine" for record in result.window_records)
    assert all("selection_reasons" in record for record in result.window_records)
    assert any(record["selected_for_output"] for record in result.window_records)
    assert result.subject_hint and "dog" in result.subject_hint
    assert result.score_curve
    assert result.score_diagnostics["growth_threshold"] == 0.37


def test_frontend_qwen_failure_and_no_signals_keeps_short_video(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (90, 30.0, 640, 360),
    )
    scorer = FakeScorer(fail=True)
    detector = WindowScoreFrontend(
        scorer,
        signal_provider=lambda _video, _window: {
            "motion_score": None,
            "audio_score": None,
            "scene_score": None,
            "yolo_interaction_score": None,
        },
    )

    result = detector.detect("short.mp4")

    assert result.valid
    assert result.segments_sec == [(0.0, 3.0)]
    assert result.fallback_used == "short_video_full"
    assert result.source == "window_score_short_video_full_fallback"


def test_frontend_all_signal_failure_uses_middle_four_seconds(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (300, 30.0, 640, 360),
    )
    scorer = FakeScorer(fail=True)
    detector = WindowScoreFrontend(
        scorer,
        signal_provider=lambda _video, _window: {
            "motion_score": None,
            "audio_score": None,
            "scene_score": None,
            "yolo_interaction_score": None,
        },
    )

    result = detector.detect("normal.mp4")

    assert result.valid
    assert result.segments_sec == [(3.0, 7.0)]
    assert result.fallback_used == "middle"


def test_frontend_disables_constant_qwen_channel(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (240, 30.0, 640, 360),
    )
    detector = WindowScoreFrontend(FakeScorer(lambda _start, _end: 0.75), signal_provider=_signals)

    result = detector.detect("video.mp4")

    assert result.valid
    assert result.score_diagnostics["qwen_degenerate"] is True
    assert all(record["qwen_disabled"] for record in result.window_records)
    assert result.subject_hint is None
    assert result.action_description is None


def test_frontend_isolates_unexpected_single_window_exception(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (90, 30.0, 640, 360),
    )
    # The exploding scorer is intentionally used only to exercise the
    # isolation contract; traditional signals still provide a fallback.
    detector = WindowScoreFrontend(
        ExplodingScorer(),
        signal_provider=lambda _video, _window: {
            "motion_score": 0.2,
            "audio_score": None,
            "scene_score": None,
            "yolo_interaction_score": None,
        },
    )
    result = detector.detect("normal.mp4")
    assert result.valid
    assert result.segments_sec
    assert all(record["valid"] is False for record in result.window_records)


def test_public_window_score_temporal_detector_returns_temporal_result(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (90, 30.0, 640, 360),
    )
    scorer = FakeScorer(lambda _start, _end: 0.8)
    detector = WindowScoreTemporalDetector(
        scorer,
        frontend_kwargs={
            "signal_provider": lambda _video, _window: {
                "motion_score": 0.5,
                "audio_score": None,
                "scene_score": None,
                "yolo_interaction_score": None,
            }
        },
    )
    result = detector.detect("normal.mp4")
    assert result.__class__.__name__ == "TemporalResult"
    assert result.window_records
    detector.close()
    assert scorer.closed


def test_optional_yolo_interaction_callback_runs_once_per_window(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (60, 30.0, 640, 360),
    )
    calls = []

    def yolo_callback(_video, window):
        calls.append(window.window_id)
        return 0.4

    detector = WindowScoreFrontend(
        FakeScorer(lambda _start, _end: 0.8),
        signal_provider=lambda _video, _window: {
            "motion_score": 0.2,
            "audio_score": None,
            "scene_score": None,
        },
        yolo_interaction_provider=yolo_callback,
        max_fine_regions=0,
    )
    result = detector.detect("video.mp4")
    assert len(calls) == len(result.window_records)


def test_yolo_callback_is_not_lost_with_default_signal_extractor(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (60, 30.0, 640, 360),
    )
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.TemporalSignalExtractor.score_windows",
        lambda _self, _video, windows: [
            {
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "motion_score": 0.2,
                "audio_score": None,
                "scene_score": 0.1,
                "yolo_interaction_score": None,
            }
            for window in windows
        ],
    )
    calls = []
    detector = WindowScoreFrontend(
        FakeScorer(),
        yolo_interaction_provider=lambda _video, window: calls.append(window.window_id)
        or 0.4,
        max_fine_regions=0,
    )
    result = detector.detect("video.mp4")
    assert len(calls) == len(result.window_records)


def test_subject_hint_is_normalized_to_existing_yolo_alias(monkeypatch):
    assert (
        WindowScoreFrontend._canonical_subject_hint(
            "woman", "woman walking beside a bicycle"
        )
        == "person"
    )
    assert WindowScoreFrontend._canonical_subject_hint("woman,", "walking") == "person"
    assert (
        WindowScoreFrontend._canonical_subject_hint("basketball player", "shoots")
        == "person"
    )
    assert WindowScoreFrontend._canonical_subject_hint("truck", "driving") == "vehicle"
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (120, 30.0, 640, 360),
    )
    detector = WindowScoreFrontend(
        FakeScorer(lambda start, _end: 0.9 if start >= 1.0 else 0.2),
        signal_provider=lambda _video, _window: {
            "motion_score": 0.5,
            "audio_score": None,
            "scene_score": None,
        },
        max_fine_regions=0,
    )
    result = detector.detect("video.mp4")
    assert result.subject_hint == "dog"
    assert " " not in result.subject_hint


def test_resource_failure_is_not_retried_again_in_fine_stage(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (240, 30.0, 640, 360),
    )
    scorer = ResourceFailScorer()
    detector = WindowScoreFrontend(scorer, signal_provider=_signals)
    result = detector.detect("video.mp4")
    assert len(scorer.calls) == 1
    assert all(call[3] == "coarse" for call in scorer.calls)
    assert result.score_diagnostics["qwen_stopped_reason"]


def test_semantic_handoff_uses_best_valid_window_inside_final_segments():
    records = [
        {
            "start_sec": 0.0,
            "end_sec": 2.0,
            "valid": True,
            "fused_score": 0.99,
            "qwen_score": 0.99,
            "main_subject": "dog",
            "action_description": "dog waits before the event",
        },
        {
            "start_sec": 5.0,
            "end_sec": 7.0,
            "valid": True,
            "fused_score": 0.80,
            "qwen_score": 0.75,
            "main_subject": "cat",
            "action_description": "cat completes a jump",
        },
    ]
    subject, action = WindowScoreFrontend._selected_semantics(records, [(5.5, 6.5)])
    assert subject == "cat"
    assert action == "cat completes a jump"


def test_action_rating_stays_numeric_and_description_is_returned(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (120, 30.0, 640, 360),
    )

    class ComponentScorer(FakeScorer):
        def score_window(
            self,
            _video_path,
            start_sec,
            end_sec,
            sample_fps,
            *,
            window_id=0,
            stage="coarse",
            parent_id=None,
        ):
            return WindowScore(
                window_id,
                start_sec,
                end_sec,
                sample_fps,
                qwen_score=min(0.9, 0.6 + 0.1 * float(start_sec)),
                qwen_components={
                    "action": 3,
                    "completion": 2,
                    "interaction": 2,
                    "event": 2,
                    "emotion": 1,
                    "visual_impact": 2,
                    "highlight": 3,
                },
                main_subject="dog",
                action_description="dog catches a ball",
                evidence="the catch is visible",
                valid=True,
                attempts=1,
                stage=stage,
                parent_id=parent_id,
            )

    detector = WindowScoreFrontend(
        ComponentScorer(),
        signal_provider=lambda _video, window: {
            "motion_score": 0.9 if window.start_sec >= 1.0 else 0.1,
            "audio_score": None,
            "scene_score": None,
        },
        max_fine_regions=0,
    )
    result = detector.detect("video.mp4")
    assert result.action_description == "dog catches a ball"
    assert result.subject_hint == "dog"
    assert all(record["action"] == 3 for record in result.window_records)
    assert all(record["action_description"] == "dog catches a ball" for record in result.window_records)


def test_frontend_reuses_one_audio_decode_across_coarse_and_fine(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.window_score_frontend.video_meta",
        lambda _path: (240, 30.0, 640, 360),
    )
    monkeypatch.setattr(
        "video_highlight.temporal_signals._batch_visual_signals",
        lambda _path, windows, _fps: [
            {
                "motion_score": 0.9 if float(window["start_sec"]) >= 2.0 else 0.1,
                "scene_score": 0.1,
            }
            for window in windows
        ],
    )
    calls = {"audio": 0}

    def decode(*_args, **_kwargs):
        calls["audio"] += 1
        return np.concatenate(
            [np.zeros(32000, dtype=np.float32), np.ones(32000, dtype=np.float32) * 0.25]
        )

    monkeypatch.setattr(
        "video_highlight.temporal_signals._decode_audio_pcm", decode
    )
    detector = WindowScoreFrontend(
        FakeScorer(
            lambda start, end: 0.9 if min(end, 6.0) > max(start, 2.0) else 0.1
        )
    )
    result = detector.detect("video.mp4")
    assert result.score_diagnostics["fine_window_count"] > 0
    assert calls["audio"] == 1
    assert {row["signal_sample_fps"] for row in result.window_records} == {2.0}
