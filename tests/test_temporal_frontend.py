import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from video_highlight.action_description import (
    QwenActionDescriber,
    parse_action_description_response,
)
from video_highlight.temporal import TemporalResult
from video_highlight.temporal_frontend import TemporalFrontEnd
from video_highlight.temporal_fusion import (
    TemporalFusionPolicy,
    fuse_temporal_results,
)
from video_highlight.timelens import TimeLensAdapter, parse_timelens_response


def test_action_description_parser_accepts_only_complete_contract():
    raw = json.dumps(
        {
            "highlight_actions": [
                {"description": "dog runs toward the girl", "main_subject": "animal"}
            ]
        }
    )
    result = parse_action_description_response(raw)
    assert result.valid
    assert result.actions[0].description == "dog runs toward the girl"

    truncated = parse_action_description_response(raw[:-1])
    extra_key = parse_action_description_response(
        '{"highlight_actions": [], "segments": [[0, 1]]}'
    )
    assert not truncated.valid
    assert not extra_key.valid


def test_action_description_rejects_timestamps_and_unknown_subject_schema():
    result = parse_action_description_response(
        '{"highlight_actions":[{"description":"cat jumps","main_subject":"cat","time":2}]}'
    )
    assert not result.valid


def test_timelens_parser_supports_official_sentence_and_json():
    official = parse_timelens_response("The event happens in 3.2 - 8.5 seconds.", 10.0)
    structured = parse_timelens_response('{"segments":[[1.0, 2.5]]}', 10.0)
    assert official.valid and official.segments_sec == [(3.2, 8.5)]
    assert structured.valid and structured.segments_sec == [(1.0, 2.5)]


def test_timelens_adapter_runs_each_action_and_merges_ranges():
    responses = {
        "dog approaches": "The event happens in 2.0 - 5.0 seconds.",
        "girl meets dog": '{"segments":[[4.5, 8.0]]}',
    }
    describer = QwenActionDescriber(
        lambda _video, _prompt: json.dumps(
            {
                "highlight_actions": [
                    {"description": "dog approaches", "main_subject": "animal"},
                    {"description": "girl meets dog", "main_subject": "animal"},
                ]
            }
        )
    )
    actions = describer.describe("video.mp4")
    result = TimeLensAdapter(lambda _video, query: responses[query]).ground(
        "video.mp4", actions.actions, 10.0
    )
    assert result.valid
    assert result.segments_sec == [(2.0, 8.0)]


def test_frontend_falls_back_without_touching_spatial_pipeline(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.temporal_frontend.video_meta",
        lambda _path: (300, 30.0, 1280, 720),
    )

    class Fallback:
        def detect(self, _video_path):
            return TemporalResult([(1.0, 4.0)], source="motion", valid=True)

    describer = QwenActionDescriber(
        lambda _video, _prompt: '{"highlight_actions":[{"description":"cat jumps","main_subject":"animal"}]}'
    )
    result = TemporalFrontEnd(describer, TimeLensAdapter(), Fallback()).detect("video.mp4")
    assert result.valid
    assert result.segments_sec == [(1.0, 4.0)]
    assert result.source == "motion_fallback"
    assert result.fallback_used == "motion_fallback"


def test_traditional_signals_check_and_adjust_timelens_boundaries():
    grounded = TemporalResult([(3.0, 7.0)], source="timelens", valid=True)
    signals = TemporalResult([(2.0, 8.0)], source="motion_audio", valid=True)
    fused = fuse_temporal_results(
        grounded,
        signals,
        duration_sec=10.0,
        policy=TemporalFusionPolicy(max_boundary_adjust_sec=0.5),
    )
    assert fused.valid
    assert fused.segments_sec == [(2.5, 7.5)]
    assert fused.source == "qwen_action_timelens_signals_fused"
    assert any("adjusted" in item for item in fused.corrections)
