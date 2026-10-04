import json
import os

import pytest

from video_highlight.window_scoring import (
    QwenWindowScorer,
    RuntimePaths,
    assert_runtime_paths,
    build_window_score_prompt,
    configure_runtime_environment,
    parse_window_score_response,
    qwen_score_from_components,
)
from video_highlight.windowing import WindowSpec, generate_windows


def valid_response(**overrides):
    payload = {
        "action": 3,
        "completion": 2,
        "interaction": 1,
        "event": 0,
        "emotion": 3,
        "visual_impact": 2,
        "highlight": 3,
        "main_subject": "dog",
        "action_description": "dog jumps toward a person",
        "evidence": "the dog leaves the ground and approaches the person",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_generate_windows_covers_tail_without_tiny_duplicate():
    windows = generate_windows(5.4, 2.0, 1.0, 2.0)
    assert [(window.start_sec, window.end_sec) for window in windows] == [
        (0.0, 2.0),
        (1.0, 3.0),
        (2.0, 4.0),
        (3.0, 5.0),
        (3.4, 5.4),
    ]
    assert [window.window_id for window in windows] == list(range(5))
    assert all(window.sample_fps == 2.0 for window in windows)


def test_generate_windows_short_and_empty_video():
    assert generate_windows(0.0) == []
    short = generate_windows(1.25, window_sec=2.0, step_sec=1.0, sample_fps=4.0)
    assert short == [WindowSpec(0, 0.0, 1.25, 4.0, "coarse", None)]


def test_strict_parser_accepts_all_seven_integer_scores_and_formula():
    spec = WindowSpec(7, 4.0, 6.0, 2.0, "coarse")
    parsed = parse_window_score_response(valid_response(), spec)
    expected = qwen_score_from_components(
        {
            "action": 3,
            "completion": 2,
            "interaction": 1,
            "event": 0,
            "emotion": 3,
            "visual_impact": 2,
            "highlight": 3,
        }
    )
    assert parsed.valid
    assert parsed.qwen_score == pytest.approx(expected)
    assert parsed.qwen_components["completion"] == 2
    assert parsed.window_id == 7
    assert parsed.to_dict()["visual_impact"] == 2


@pytest.mark.parametrize(
    "raw",
    [
        lambda: valid_response()[:-1],
        lambda: "```json\n" + valid_response()[:-1] + "\n```",
        lambda: "answer: " + valid_response(),
        lambda: valid_response(completion=4),
        lambda: valid_response(completion=1.5),
        lambda: valid_response(completion=True),
        lambda: valid_response(evidence="the action occurs at 1.5 seconds"),
        lambda: valid_response(action_description="window W03 contains a jump"),
        lambda: valid_response(evidence="bbox=[10,20,30,40]"),
        lambda: json.dumps(
            {
                key: value
                for key, value in json.loads(valid_response()).items()
                if key != "completion"
            }
        ),
        lambda: valid_response(extra="not allowed"),
    ],
)
def test_strict_parser_rejects_wrappers_truncation_and_schema_errors(raw):
    parsed = parse_window_score_response(raw(), start_sec=0.0, end_sec=2.0)
    assert not parsed.valid
    assert parsed.qwen_score is None
    assert parsed.errors


def test_strict_parser_accepts_one_complete_markdown_fence():
    parsed = parse_window_score_response(
        "```json\n" + valid_response() + "\n```", start_sec=0.0, end_sec=2.0
    )
    assert parsed.valid
    assert parsed.qwen_score is not None


def test_compact_parser_expands_fixed_order_array():
    raw = json.dumps([3, 2, 1, 0, 3, 2, 3, "dog", "dog jumps", "dog leaves ground"])
    parsed = parse_window_score_response(
        raw,
        start_sec=0.0,
        end_sec=2.0,
        compact_output=True,
    )
    assert parsed.valid
    assert parsed.qwen_components["action"] == 3
    assert parsed.qwen_components["completion"] == 2
    assert parsed.main_subject == "dog"
    assert parsed.action_description == "dog jumps"


def test_compact_parser_rejects_wrong_length_and_string_score():
    short = parse_window_score_response(
        json.dumps([3, 2, 1]),
        start_sec=0.0,
        end_sec=2.0,
        compact_output=True,
    )
    wrong_type = parse_window_score_response(
        json.dumps(["jumping", 2, 1, 0, 3, 2, 3, "dog", "dog jumps", "jump"]),
        start_sec=0.0,
        end_sec=2.0,
        compact_output=True,
    )
    assert not short.valid
    assert not wrong_type.valid


def test_prompt_has_completion_and_forbids_timestamps_and_markdown():
    prompt = build_window_score_prompt(sample_fps=4.0)
    assert "completion" in prompt
    assert "independently chosen integer" in prompt
    assert "timestamps" in prompt
    assert "no markdown" in prompt
    assert "4 frames per second" in prompt
    assert "action is the numeric action-intensity SCORE" in prompt
    assert "put those words only in action_description" in prompt


def test_compact_prompt_uses_fixed_order_without_key_names_in_output():
    prompt = build_window_score_prompt(sample_fps=2.0, compact_output=True)
    assert "compact JSON array" in prompt
    assert "Never output key names" in prompt
    assert "numeric action-intensity score" in prompt


def test_score_window_passes_time_range_and_fps_and_retries_once():
    calls = []

    def provider(video_path, start_sec, end_sec, sample_fps, prompt):
        calls.append((video_path, start_sec, end_sec, sample_fps, prompt))
        if len(calls) == 1:
            return '{"action": 3}'
        return valid_response()

    scorer = QwenWindowScorer(response_provider=provider, max_retries=1)
    result = scorer.score_window(
        "not-opened-by-a-mock.mp4",
        1.5,
        3.5,
        4.0,
        window_id=8,
        stage="fine",
    )
    assert result.valid
    assert result.attempts == 2
    assert result.window_id == 8
    assert result.stage == "fine"
    assert len(result.raw_responses) == 2
    assert result.raw_responses[0] == '{"action": 3}'
    assert [(call[1], call[2], call[3]) for call in calls] == [
        (1.5, 3.5, 4.0),
        (1.5, 3.5, 4.0),
    ]
    assert "previous response was invalid" in calls[1][4]


def test_failed_window_does_not_stop_following_windows():
    attempts = {0.0: 0, 2.0: 0}

    def provider(_video, start, _end, _fps, _prompt):
        attempts[start] += 1
        return "bad" if start == 0.0 else valid_response()

    scorer = QwenWindowScorer(response_provider=provider, max_retries=1)
    windows = [
        WindowSpec(0, 0.0, 2.0, 2.0),
        WindowSpec(1, 2.0, 4.0, 2.0),
    ]
    records = scorer.score_windows("mock.mp4", windows)
    assert len(records) == 2
    assert not records[0].valid
    assert records[0].attempts == 2
    assert records[1].valid
    assert attempts == {0.0: 2, 2.0: 1}


def test_batch_scoring_retries_only_invalid_rows(monkeypatch):
    scorer = QwenWindowScorer(
        response_provider=lambda *_args: valid_response(),
        max_retries=1,
        batch_size=3,
    )
    calls = []

    def generate_batch(_video_path, specs, prompts):
        calls.append(([spec.window_id for spec in specs], list(prompts)))
        if len(calls) == 1:
            return [valid_response(), "bad", valid_response(highlight=1)]
        return [valid_response(highlight=2)]

    monkeypatch.setattr(scorer, "_generate_batch_once", generate_batch)
    windows = [
        WindowSpec(0, 0.0, 2.0, 2.0),
        WindowSpec(1, 1.0, 3.0, 2.0),
        WindowSpec(2, 2.0, 4.0, 2.0),
    ]

    records = scorer.score_windows("mock.mp4", windows)

    assert [record.window_id for record in records] == [0, 1, 2]
    assert all(record.valid for record in records)
    assert [record.attempts for record in records] == [1, 2, 1]
    assert calls[0][0] == [0, 1, 2]
    assert calls[1][0] == [1]
    assert "previous response was invalid" in calls[1][1][0]


def test_batch_failure_splits_to_sequential_provider(monkeypatch):
    provider_calls = []

    def provider(_video, start, _end, _fps, _prompt):
        provider_calls.append(start)
        return valid_response()

    scorer = QwenWindowScorer(
        response_provider=provider,
        max_retries=0,
        batch_size=2,
    )
    monkeypatch.setattr(
        scorer,
        "_generate_batch_once",
        lambda *_args: (_ for _ in ()).throw(ValueError("batch processor failed")),
    )
    windows = [
        WindowSpec(0, 0.0, 2.0, 2.0),
        WindowSpec(1, 1.0, 3.0, 2.0),
    ]

    records = scorer.score_windows("mock.mp4", windows)

    assert all(record.valid for record in records)
    assert provider_calls == [0.0, 1.0]


def test_resource_failure_is_not_retried_as_json_error():
    calls = []

    def provider(*_args):
        calls.append(1)
        raise RuntimeError("CUDA out of memory")

    scorer = QwenWindowScorer(response_provider=provider, max_retries=1)
    result = scorer.score_window("mock.mp4", 0.0, 2.0, 2.0)
    assert not result.valid
    assert result.attempts == 1
    second = scorer.score_window("mock.mp4", 2.0, 4.0, 2.0)
    assert not second.valid
    assert second.attempts == 0
    assert len(calls) == 1


def test_real_video_request_uses_original_path_and_range(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    scorer = QwenWindowScorer(response_provider=lambda *_args: valid_response())
    request = scorer._video_request(str(video), 2.0, 4.0, 2.0)
    assert request["video"] == str(video.resolve())
    assert request["video_start"] == 2.0
    assert request["video_end"] == 4.0
    assert request["fps"] == 2.0
    assert request["min_frames"] == 2
    assert request["max_frames"] == 4


def test_e_drive_guard_is_configurable():
    assert_runtime_paths(RuntimePaths(r"E:\models\qwen", r"E:\cache", r"E:\tmp"), require_e_drive=True)
    with pytest.raises(ValueError, match="must be on E"):
        assert_runtime_paths(
            RuntimePaths(r"C:\models\qwen", r"E:\cache", r"E:\tmp"),
            require_e_drive=True,
        )


def test_guarded_runtime_overrides_old_c_drive_cache(monkeypatch):
    monkeypatch.setenv("HF_HOME", r"C:\Users\example\.cache\huggingface")
    paths = RuntimePaths(r"E:\models\qwen", r"E:\cache\window_score", r"E:\tmp")
    configure_runtime_environment(paths, require_e_drive=True)
    assert os.environ["HF_HOME"] == r"E:\cache\window_score"
