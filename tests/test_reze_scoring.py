import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import numpy as np
import pytest

from video_highlight.reze_scoring import (
    GENERIC_HIGHLIGHT_PROMPT,
    RezeWindowScore,
    RezeWindowScorer,
    build_generic_highlight_prompt,
    diagnose_reze_scores,
    generate_reze_windows,
    resolve_yes_no_tokens,
    stable_binary_probability,
)
from video_highlight.window_scoring import QwenWindowScorer


class FakeTokenizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {"Yes": [7], "No": [11]}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(self.mapping[text])


class FakeProcessor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or FakeTokenizer()


class FakeModel:
    def __init__(self, *, yes_logit=3.0, no_logit=1.0, fail_first=False):
        self.yes_logit = yes_logit
        self.no_logit = no_logit
        self.fail_first = fail_first
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("synthetic forward failure")
        logits = np.full((1, 1, 32), -5.0, dtype=np.float64)
        logits[0, 0, 7] = self.yes_logit
        logits[0, 0, 11] = self.no_logit
        return SimpleNamespace(logits=logits)


def make_scorer(model=None, tokenizer=None, input_builder=None):
    model = model or FakeModel()
    processor = FakeProcessor(tokenizer)
    base = QwenWindowScorer(model=model, processor=processor, load_in_4bit=False)
    return (
        RezeWindowScorer(
            base,
            input_builder=input_builder or (lambda *_args: {"input_ids": np.asarray([[1]])}),
        ),
        model,
    )


def test_reze_windows_are_non_overlapping_and_keep_only_meaningful_tail():
    windows = generate_reze_windows(7.2)
    assert [(row.start_sec, row.end_sec) for row in windows] == [
        (0.0, 3.0),
        (3.0, 6.0),
        (6.0, 7.2),
    ]
    assert all(row.sample_fps == 2.0 and row.stage == "reze" for row in windows)

    discarded = generate_reze_windows(6.2)
    assert [(row.start_sec, row.end_sec) for row in discarded] == [
        (0.0, 3.0),
        (3.0, 6.0),
    ]
    assert generate_reze_windows(0.29) == []
    assert [(row.start_sec, row.end_sec) for row in generate_reze_windows(0.3)] == [
        (0.0, 0.3)
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_sec": -1.0},
        {"duration_sec": 3.0, "window_sec": 0.0},
        {"duration_sec": 3.0, "sample_fps": 0.0},
        {"duration_sec": 3.0, "tail_min_sec": 3.1},
    ],
)
def test_reze_window_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        generate_reze_windows(**kwargs)


def test_prompt_is_generic_strict_and_has_no_external_action_query():
    prompt = build_generic_highlight_prompt()
    assert prompt == GENERIC_HIGHLIGHT_PROMPT
    assert "supplied video clip" in prompt
    assert "before or after" in prompt
    assert "Static setup" in prompt
    assert prompt.endswith("Yes or No.")
    assert "timestamp" not in prompt.lower()


def test_runtime_token_resolution_requires_distinct_single_tokens():
    valid = resolve_yes_no_tokens(FakeTokenizer())
    assert valid.valid
    assert valid.yes_token_id == 7
    assert valid.no_token_id == 11
    assert valid.to_dict()["yes_token_ids"] == [7]

    multi = resolve_yes_no_tokens(FakeTokenizer({"Yes": [7, 8], "No": [11]}))
    assert not multi.valid
    assert any("Yes must be one token" in error for error in multi.errors)

    same = resolve_yes_no_tokens(FakeTokenizer({"Yes": [7], "No": [7]}))
    assert not same.valid
    assert any("same token" in error for error in same.errors)


def test_stable_binary_probability_handles_large_logits_and_rejects_nonfinite():
    assert stable_binary_probability(10_000.0, 9_999.0) == pytest.approx(
        0.7310585786
    )
    assert stable_binary_probability(-10_000.0, -9_999.0) == pytest.approx(
        0.2689414214
    )
    assert stable_binary_probability(float("nan"), 0.0) is None
    assert stable_binary_probability(0.0, float("inf")) is None


def test_fake_model_forward_uses_last_logit_only_and_records_raw_values():
    scorer, model = make_scorer()
    record = scorer.score_window("unused.mp4", 3.0, 6.0, 2.0, window_id=4)
    assert record.valid
    assert record.p_yes == pytest.approx(0.8807970779)
    assert record.yes_token_id == 7
    assert record.no_token_id == 11
    assert record.yes_logit == 3.0
    assert record.no_logit == 1.0
    assert record.attempts == 1
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["logits_to_keep"] == 1
    assert call["use_cache"] is False
    assert call["return_dict"] is True
    raw = record.to_dict()
    assert raw["semantic_mode"] == "reze_yesno"
    assert raw["p_yes"] == pytest.approx(record.p_yes)
    assert raw["yes_token_ids"] == [7]
    assert raw["no_token_ids"] == [11]


def test_multitoken_answer_is_missing_and_never_calls_model():
    scorer, model = make_scorer(
        tokenizer=FakeTokenizer({"Yes": [7, 8], "No": [11]})
    )
    record = scorer.score_window("unused.mp4", 0.0, 3.0)
    assert not record.valid
    assert record.p_yes is None
    assert record.attempts == 0
    assert model.calls == []


def test_individual_forward_failure_does_not_stop_following_window():
    model = FakeModel(fail_first=True)
    scorer, _ = make_scorer(model=model)
    result = scorer.score_windows("unused.mp4", generate_reze_windows(6.0))
    assert len(result.records) == 2
    assert not result.records[0].valid
    assert result.records[0].p_yes is None
    assert result.records[1].valid
    assert len(model.calls) == 2


def test_score_diagnostics_disable_constant_or_narrow_semantic_curve():
    equal = [
        RezeWindowScore(0, 0.0, 3.0, 2.0, p_yes=0.6, valid=True),
        RezeWindowScore(1, 3.0, 6.0, 2.0, p_yes=0.6, valid=True),
    ]
    equal_diagnostics = diagnose_reze_scores(equal)
    assert equal_diagnostics["degenerate"] is True
    assert equal_diagnostics["all_equal"] is True
    assert equal_diagnostics["reason"] == "all_probabilities_equal"

    narrow = [
        RezeWindowScore(0, 0.0, 3.0, 2.0, p_yes=0.51, valid=True),
        RezeWindowScore(1, 3.0, 6.0, 2.0, p_yes=0.55, valid=True),
    ]
    narrow_diagnostics = diagnose_reze_scores(narrow)
    assert narrow_diagnostics["degenerate"] is True
    assert narrow_diagnostics["reason"] == "probability_range_too_small"

    varied = [
        RezeWindowScore(0, 0.0, 3.0, 2.0, p_yes=0.2, valid=True),
        RezeWindowScore(1, 3.0, 6.0, 2.0, p_yes=0.8, valid=True),
    ]
    assert diagnose_reze_scores(varied)["semantic_usable"] is True


def test_default_input_path_reuses_qwen_video_range_request(monkeypatch):
    captured = {}

    class Processor(FakeProcessor):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            captured["messages"] = messages
            assert tokenize is False
            assert add_generation_prompt is True
            return "templated"

        def __call__(self, **kwargs):
            captured["processor_kwargs"] = kwargs
            return {"input_ids": np.asarray([[1]])}

    class Base:
        model = FakeModel()
        processor = Processor()
        model_load_error = None

        def _video_request(self, video_path, start_sec, end_sec, sample_fps):
            captured["request_args"] = (video_path, start_sec, end_sec, sample_fps)
            return {
                "type": "video",
                "video": video_path,
                "video_start": start_sec,
                "video_end": end_sec,
                "fps": sample_fps,
            }

        @staticmethod
        def _process_vision_inputs(processor, messages):
            return processor(messages, return_video_kwargs=True)

        @staticmethod
        def _normalize_video_kwargs(kwargs):
            return dict(kwargs)

    def process_vision_info(messages, return_video_kwargs):
        captured["vision_messages"] = messages
        assert return_video_kwargs is True
        return [], ["decoded-video"], {"fps": 2.0}

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=process_vision_info),
    )
    scorer = RezeWindowScorer(Base())
    inputs = scorer._prepare_model_inputs(
        "original.mp4", 6.0, 9.0, 2.0, GENERIC_HIGHLIGHT_PROMPT
    )
    assert "input_ids" in inputs
    assert captured["request_args"] == ("original.mp4", 6.0, 9.0, 2.0)
    request = captured["messages"][0]["content"][0]
    assert request["video"] == "original.mp4"
    assert request["video_start"] == 6.0
    assert request["video_end"] == 9.0
    assert request["fps"] == 2.0
