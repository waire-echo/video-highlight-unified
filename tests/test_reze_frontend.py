import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from video_highlight.reze_frontend import RezeTemporalFrontend
from video_highlight.temporal import TemporalResult


class _Record:
    def __init__(self, index, start, end, value):
        self.window_id = index
        self.start_sec = start
        self.end_sec = end
        self.p_yes = value
        self.valid = value is not None
        self.errors = [] if self.valid else ["missing"]
        self.attempts = 1

    def to_dict(self):
        return {
            "window_id": self.window_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "p_yes": self.p_yes,
            "valid": self.valid,
            "errors": list(self.errors),
            "attempts": self.attempts,
        }


class _Scoring:
    def __init__(self, values):
        self.records = [
            _Record(index, index * 3.0, min(12.0, (index + 1) * 3.0), value)
            for index, value in enumerate(values)
        ]
        self.diagnostics = {"status": "ok"}
        self.duration_sec = 12.0


def test_reze_frontend_aggregates_without_spatial_pipeline(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.reze_frontend.video_meta",
        lambda _path: (360, 30.0, 1280, 720),
    )
    frontend = RezeTemporalFrontend(object(), aggregation="reze_otsu")
    frontend.scorer = SimpleNamespace(
        score_video=lambda _path, duration_sec: _Scoring([0.05, 0.95, 0.90, 0.05])
    )
    result = frontend.detect("video.mp4")
    assert result.valid
    assert result.source == "reze_yesno_reze_otsu"
    assert result.segments_sec
    assert result.subject_hint is None
    assert result.action_description is None
    assert result.window_records[1]["p_yes"] == 0.95


def test_reze_degenerate_curve_uses_explicit_temporal_fallback(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.reze_frontend.video_meta",
        lambda _path: (300, 30.0, 1280, 720),
    )

    class Fallback:
        def detect(self, _path):
            return TemporalResult([(2.0, 5.0)], source="motion", valid=True)

    frontend = RezeTemporalFrontend(
        object(), aggregation="reze_otsu", fallback_detector=Fallback()
    )
    frontend.scorer = SimpleNamespace(
        score_video=lambda _path, duration_sec: _Scoring([0.5, 0.5, 0.5, 0.5])
    )
    result = frontend.detect("video.mp4")
    assert result.segments_sec == [(2.0, 5.0)]
    assert result.fallback_used == "motion"
    assert result.source == "reze_yesno_motion_fallback"
    assert result.score_diagnostics["aggregation_diagnostics"]["status"] == "degenerate"


def test_reze_current_aggregation_is_supported(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.reze_frontend.video_meta",
        lambda _path: (360, 30.0, 1280, 720),
    )
    frontend = RezeTemporalFrontend(object(), aggregation="current")
    frontend.scorer = SimpleNamespace(
        score_video=lambda _path, duration_sec: _Scoring([0.05, 0.95, 0.90, 0.05])
    )
    result = frontend.detect("video.mp4")
    assert result.valid
    assert result.source == "reze_yesno_current"
    assert result.score_curve
