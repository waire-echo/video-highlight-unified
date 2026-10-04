import json
from types import SimpleNamespace

import pytest

from video_highlight.global_event_frontend import (
    GlobalEventFrontend,
    build_event_window_prompt,
    parse_event_ranking_response,
    parse_event_window_response,
    parse_global_event_response,
)
from video_highlight.temporal import build_temporal_detector


def _overview(events):
    return json.dumps({"events": events})


EVENTS = [
    {
        "event_id": "E1",
        "subject": "dog",
        "action": "dog catches a ball",
        "completion_cue": "ball reaches the dog",
    },
    {
        "event_id": "E2",
        "subject": "dog",
        "action": "dog jumps into owner arms",
        "completion_cue": "owner catches the dog",
    },
]


class FakeScorer:
    def __init__(self, rows, overview=None, subject="dog"):
        self.rows = list(rows)
        self.overview = overview or _overview(EVENTS)
        self.subject = subject
        self.closed = False

    def generate_range(self, _video, _start, _end, _fps, prompt):
        if "selected_candidate_ids" in prompt:
            return json.dumps({"selected_candidate_ids": ["C1", "C2"]})
        if "Watch the supplied video range as one story" in prompt:
            return self.overview
        return self.rows[0]

    def generate_ranges(self, _video, specs, _prompts):
        return [self.rows[spec.window_id] for spec in specs]

    def close(self):
        self.closed = True


def _window(matches, completion=None, subject="dog"):
    return json.dumps(
        {
            "matches": matches,
            "completion": completion if completion is not None else matches,
            "visible_subject": subject,
            "evidence": "visible action evidence",
        }
    )


def _signals(_video, window):
    # Deliberately make the middle of the video the strongest motion.  It must
    # not create an event unless a semantic event curve has a seed there.
    high = 1.0 if 4.0 <= window.start_sec <= 6.0 else 0.1
    return {"motion_score": high, "audio_score": high, "scene_score": 0.0}


def test_global_event_parsers_are_strict():
    parsed = parse_global_event_response(_overview(EVENTS), max_events=3)
    assert [item["event_id"] for item in parsed] == ["E1", "E2"]
    assert parse_event_window_response(_window([3, 0]), 2)["matches"] == [3, 0]
    assert parse_event_ranking_response(
        '{"selected_candidate_ids":["C2"]}', ["C1", "C2"]
    ) == ["C2"]
    compact_events = parse_global_event_response(
        '[["cat","cat leaves carrier","cat fully exits carrier"]]'
    )
    assert compact_events[0]["event_id"] == "E1"
    temporal_grounding_events = parse_global_event_response(
        '[{"start_time":"6","end_time":"8",'
        '"description":"person pulls suitcase with cat inside"}]'
    )
    assert temporal_grounding_events[0]["subject"] == "cat"
    assert temporal_grounding_events[0]["overview_time_hint_sec"] == [6.0, 8.0]
    compact_window = parse_event_window_response(
        '[[3],[2],"cat","cat exits carrier"]', 1
    )
    assert compact_window["completion"] == [2]
    observed_small_model_shape = parse_event_window_response(
        '{"matches":3,"completion_cue":3,"visible_subject":"cat",'
        '"evidence":[{"type":"object","name":"cat exits carrier"}]}',
        1,
    )
    assert observed_small_model_shape["matches"] == [3]
    assert "cat exits carrier" in observed_small_model_shape["evidence"]
    observed_event_rows = parse_event_window_response(
        '[["E1",[2,"suitcase is open"],"cat in suitcase"],'
        '["E2",[3,"person pulls suitcase"],"person with cat"]]',
        2,
    )
    assert observed_event_rows["matches"] == [2, 3]
    assert observed_event_rows["completion"] == [2, 3]
    assert observed_event_rows["compatibility_schema"] == "event_rows_without_completion"
    omitted_outer_array = parse_event_window_response(
        '[2,3,"cat","cat inside suitcase"],'
        '[0,0,"cat","no person pulls suitcase"]',
        2,
    )
    assert omitted_outer_array["matches"] == [2, 0]
    assert omitted_outer_array["completion"] == [3, 0]
    assert omitted_outer_array["compatibility_schema"] == "per_event_compact_rows"

    with pytest.raises(ValueError, match="timestamps"):
        parse_global_event_response(
            _overview(
                [
                    {
                        "event_id": "E1",
                        "subject": "dog",
                        "action": "dog jumps at 3 seconds",
                        "completion_cue": "dog lands",
                    }
                ]
            )
        )
    with pytest.raises(ValueError, match="exactly 2"):
        parse_event_window_response(_window([3]), 2)
    with pytest.raises(ValueError, match="unknown candidate"):
        parse_event_ranking_response(
            '{"selected_candidate_ids":["C9"]}', ["C1", "C2"]
        )


def test_prompt_contains_all_video_level_events():
    prompt = build_event_window_prompt(EVENTS)
    assert "dog catches a ball" in prompt
    assert "dog jumps into owner arms" in prompt
    assert "exactly 2 integers" in prompt
    assert "four-value compact JSON array" in prompt


def test_whole_video_time_hint_survives_candidate_normalization():
    overview = json.dumps(
        [
            {
                "start_time": 4,
                "end_time": 8,
                "description": "dog catches a ball",
            }
        ]
    )
    detector = GlobalEventFrontend(FakeScorer([], overview=overview))
    parsed = detector._global_events("ignored.mp4", 10.0)

    assert parsed["events"][0]["overview_time_hint_sec"] == [4.0, 8.0]


def test_only_short_parser_failure_gaps_are_bridgeable():
    detector = GlobalEventFrontend(FakeScorer([]))
    missing_rows = [
        {"start_sec": 8.0, "end_sec": 10.0, "valid": False},
        {"start_sec": 9.0, "end_sec": 11.0, "valid": False},
    ]
    assert detector._gap_is_missing_only(missing_rows, 9.5, 10.5)

    explicit_low = missing_rows + [
        {"start_sec": 9.5, "end_sec": 10.5, "valid": True, "event_scores": {"E1": 0.0}}
    ]
    assert not detector._gap_is_missing_only(explicit_low, 9.5, 10.5)
    assert not detector._gap_is_missing_only(missing_rows, 8.0, 10.0)


def test_global_event_frontend_keeps_separated_semantic_events(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.global_event_frontend.video_meta",
        lambda _path: (300, 30.0, 720, 1280),
    )
    rows = []
    for index in range(9):
        if index in {1, 2}:
            rows.append(_window([3, 0]))
        elif index in {7, 8}:
            rows.append(_window([0, 3]))
        else:
            rows.append(_window([0, 0]))
    detector = GlobalEventFrontend(
        FakeScorer(rows),
        signal_provider=_signals,
        min_prominence=0.05,
    )
    result = detector.detect("ignored.mp4")

    assert result.valid
    assert len(result.segments_sec) == 2
    assert result.segments_sec[0][1] < result.segments_sec[1][0]
    assert result.subject_hint == "dog"
    assert result.source == "global_event_curve"
    assert result.score_diagnostics["boundary_policy"].startswith("semantic_seed")


def test_motion_cannot_create_a_highlight_without_event_match(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.global_event_frontend.video_meta",
        lambda _path: (300, 30.0, 720, 1280),
    )
    rows = [_window([3, 0])] + [_window([0, 0]) for _ in range(8)]
    detector = GlobalEventFrontend(
        FakeScorer(rows),
        signal_provider=_signals,
        min_prominence=0.0,
    )
    result = detector.detect("ignored.mp4")

    assert result.valid
    assert len(result.segments_sec) == 1
    # Motion is strongest around 4--7 seconds, but semantic evidence exists
    # only at the beginning.
    assert result.segments_sec[0][0] < 3.0
    assert result.segments_sec[0][1] < 5.0


def test_subject_conflict_is_null_instead_of_misleading_yolo(monkeypatch):
    monkeypatch.setattr(
        "video_highlight.global_event_frontend.video_meta",
        lambda _path: (180, 30.0, 720, 1280),
    )
    rows = [_window([3, 0], subject="person") for _ in range(5)]
    detector = GlobalEventFrontend(
        FakeScorer(rows), signal_provider=_signals, min_prominence=0.0
    )
    result = detector.detect("ignored.mp4")

    assert result.valid
    assert result.subject_hint is None
    assert result.score_diagnostics["subject_consensus"]["conflict"] is True


def test_long_video_uses_chunks_and_ranks_candidates():
    class LongScorer(FakeScorer):
        def __init__(self):
            super().__init__([_window([3, 0])])
            self.calls = []

        def generate_range(self, _video, start, end, _fps, prompt):
            self.calls.append((start, end, prompt))
            if "selected_candidate_ids" in prompt:
                return '{"selected_candidate_ids":["C2","C1"]}'
            actions = {
                (0, 90): "dog swims through water",
                (0, 30): "dog runs toward owner",
                (30, 60): "dog jumps over fence",
                (60, 90): "dog catches a ball",
            }
            event = {
                "event_id": "E1",
                "subject": "dog",
                "action": actions[(int(start), int(end))],
                "completion_cue": "dog completes action",
            }
            return _overview([event])

    scorer = LongScorer()
    detector = GlobalEventFrontend(scorer, long_video_sec=60.0, overview_chunk_sec=30.0)
    result = detector._global_events("ignored.mp4", 90.0)

    assert len(result["overview_ranges"]) == 4  # whole video + three chunks
    assert result["selected_candidate_ids"] == ["C2", "C1"]
    assert [event["event_id"] for event in result["events"]] == ["E1", "E2"]


def test_build_temporal_detector_accepts_global_event(monkeypatch):
    fake_scorer = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        "video_highlight.window_scoring.QwenWindowScorer",
        lambda **_kwargs: fake_scorer,
    )
    args = SimpleNamespace(
        temporal_mode="window_score",
        temporal_model_path=r"E:\AI\models\Qwen2.5-VL-3B-Instruct",
        window_semantic_mode="global_event",
        window_aggregation="current",
    )
    detector = build_temporal_detector(args)
    assert isinstance(detector, GlobalEventFrontend)
