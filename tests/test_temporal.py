import inspect
import os
import sys
from pathlib import Path

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from video_highlight import temporal
from video_highlight.temporal import (
    QwenTemporalDetector,
    TemporalNormalizationResult,
    TemporalParseResult,
    TemporalResult,
    normalize_temporal_segments,
    parse_temporal_json,
    parse_temporal_response,
    temporal_coverage_ratio,
)


def _segments(result):
    value = getattr(result, "segments_sec", None)
    if value is None:
        value = getattr(result, "segments", None)
    assert value is not None, "temporal result must expose segments_sec or segments"
    return [(float(start), float(end)) for start, end in value]


def _errors(result):
    return list(getattr(result, "errors", []) or [])


def _is_valid(result):
    if hasattr(result, "valid"):
        return bool(result.valid)
    return bool(_segments(result)) and not _errors(result)


def _source(result):
    return str(getattr(result, "source", "") or "").lower()


def _fallback_used(result):
    return str(getattr(result, "fallback_used", "") or "").lower()


def _recursive_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_recursive_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_recursive_strings(item))
        return out
    return []


class ScriptedResponseProvider:
    """Model-free response provider which also records generated prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self.responses:
            raise AssertionError("response provider was called more often than expected")
        return self.responses.pop(0)

    @property
    def prompts(self):
        prompts = []
        for args, kwargs in self.calls:
            strings = _recursive_strings(args) + _recursive_strings(kwargs)
            likely = [
                text
                for text in strings
                if "segment" in text.lower()
                or "interval" in text.lower()
                or "previous response" in text.lower()
            ]
            prompts.append(max(likely or strings or [""], key=len))
        return prompts


class StubTemporalDetector:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def detect(self, video_path):
        self.calls.append(video_path)
        return self.result


def _make_qwen_detector(provider, **overrides):
    """Construct the detector without importing transformers or loading weights.

    Only response_provider is a fixed public requirement. The remaining safety
    option aliases keep these behavior-focused tests usable while the CLI and
    detector constructor are being integrated in parallel.
    """

    signature = inspect.signature(QwenTemporalDetector)
    params = signature.parameters
    assert "response_provider" in params, "QwenTemporalDetector must support model-free response_provider"

    values = {
        "model_path": "",
        "detect_fps": 1.0,
        "response_provider": provider,
        "vlm_retries": overrides.get("vlm_retries", 1),
        "retries": overrides.get("vlm_retries", 1),
        "temporal_fallback": overrides.get("temporal_fallback", "motion"),
        "fallback": overrides.get("temporal_fallback", "motion"),
        "short_video_full_sec": overrides.get("short_video_full_sec", 0.0),
        "segment_padding_sec": overrides.get("padding_sec", 0.0),
        "padding_sec": overrides.get("padding_sec", 0.0),
        "min_segment_sec": overrides.get("min_segment_sec", 1.0),
        "min_coverage_ratio": overrides.get("min_coverage_ratio", 0.0),
        "max_temporal_segments": overrides.get("max_segments", 3),
        "max_segments": overrides.get("max_segments", 3),
        "fallback_detector": overrides.get("fallback_detector"),
        "full_detector": overrides.get("full_detector"),
    }
    kwargs = {name: value for name, value in values.items() if name in params and value is not None}
    detector = QwenTemporalDetector(**kwargs)

    # Keep behavior settings explicit even if the implementation exposes them
    # as attributes instead of constructor arguments.
    for name in (
        "vlm_retries",
        "temporal_fallback",
        "short_video_full_sec",
        "min_segment_sec",
        "min_coverage_ratio",
    ):
        setattr(detector, name, overrides.get(name, values.get(name)))
    detector.segment_padding_sec = overrides.get("padding_sec", 0.0)
    detector.padding_sec = overrides.get("padding_sec", 0.0)
    detector.max_segments = overrides.get("max_segments", 3)
    detector.max_temporal_segments = overrides.get("max_segments", 3)
    return detector


def _patch_video_meta(monkeypatch, duration_sec=10.0, fps=30.0):
    n_frames = int(round(duration_sec * fps))
    monkeypatch.setattr(temporal, "video_meta", lambda _path: (n_frames, fps, 1280, 720))


def test_parse_standard_json_and_compatibility_entrypoint():
    text = '{"segments": [[1.2, 4.8]], "subject_hint": "Person"}'
    parsed = parse_temporal_response(text, duration_sec=8.0)
    assert isinstance(parsed, TemporalParseResult)
    assert _is_valid(parsed)
    assert _segments(parsed) == [(1.2, 4.8)]
    assert parsed.subject_hint == "person"

    compat = parse_temporal_json(text)
    assert isinstance(compat, TemporalResult)
    assert _segments(compat) == [(1.2, 4.8)]
    assert compat.subject_hint == "person"


def test_temporal_result_keeps_legacy_three_positional_arguments():
    result = TemporalResult([(1.0, 2.0)], "person", "legacy-raw")
    assert _segments(result) == [(1.0, 2.0)]
    assert result.subject_hint == "person"
    assert result.raw == "legacy-raw"


def test_parse_markdown_json_object_segments_and_surrounding_text():
    markdown = """```json
    {"segments": [[1.0, 2.5]], "subject_hint": "sports ball"}
    ```"""
    parsed_markdown = parse_temporal_response(markdown, duration_sec=5.0)
    assert _is_valid(parsed_markdown)
    assert _segments(parsed_markdown) == [(1.0, 2.5)]

    object_form = (
        'The result is: {"segments": [{"start_sec": 2.0, "end_sec": 4.0}], '
        '"subject_hint": "car"} End.'
    )
    parsed_object = parse_temporal_response(object_form, duration_sec=6.0)
    assert _is_valid(parsed_object)
    assert _segments(parsed_object) == [(2.0, 4.0)]
    assert parsed_object.subject_hint == "car"


def test_parse_qwen_legacy_top_level_list_with_named_numeric_strings():
    response = """```json
    [
        {"start_sec": "0.0", "end_sec": "1.0", "subject_hint": "person"},
        {"start_sec": "1.0", "end_sec": "2.0", "subject_hint": "door"},
        {"start_sec": "2.0", "end_sec": "5.920", "subject_hint": "person"}
    ]
    ```"""

    parsed = parse_temporal_response(response, duration_sec=5.92)

    assert _is_valid(parsed)
    assert _segments(parsed) == [(0.0, 1.0), (1.0, 2.0), (2.0, 5.92)]
    assert parsed.subject_hint == "person"


def test_parse_legacy_list_still_rejects_unstructured_or_invalid_times():
    unstructured = parse_temporal_response('[[1.0, 2.0]]', duration_sec=5.0)
    invalid = parse_temporal_response(
        '[{"start_sec": "zero", "end_sec": "2.0", "subject_hint": "person"}]',
        duration_sec=5.0,
    )

    assert not _is_valid(unstructured)
    assert not _is_valid(invalid)
    assert _segments(unstructured) == []
    assert _segments(invalid) == []


def test_parse_qwen_grouped_top_level_list_flattens_all_segment_arrays():
    response = """```json
    [
        {"segments": [[0.0, 2.0], [4.0, 6.0]], "subject_hint": "object"},
        {"segments": [[7.0, 10.0]], "subject_hint": "object"},
        {"segments": [[11.0, 17.0]], "subject_hint": "object"}
    ]
    ```"""

    parsed = parse_temporal_response(response, duration_sec=17.1)

    assert _is_valid(parsed)
    assert _segments(parsed) == [(0.0, 2.0), (4.0, 6.0), (7.0, 10.0), (11.0, 17.0)]
    assert parsed.subject_hint == "object"


@pytest.mark.parametrize(
    "text",
    [
        "The highlight is between ten and twenty seconds.",
        "Duration 8.0 seconds, FPS 30.0, likely interval [1.0, 4.0].",
        '{"duration": 8.0, "fps": 30.0}',
        '{"segments": []}',
    ],
)
def test_parse_rejects_arbitrary_numbers_or_empty_segments(text):
    parsed = parse_temporal_response(text, duration_sec=8.0)
    assert not _is_valid(parsed)
    assert _segments(parsed) == []
    assert _errors(parsed)


def test_parse_rejects_fully_out_of_range_template_answer():
    parsed = parse_temporal_response('{"segments": [[10, 20]]}', duration_sec=8.0)
    assert not _is_valid(parsed)
    assert _segments(parsed) == []
    assert any("outside" in error.lower() or "duration" in error.lower() for error in _errors(parsed))


def test_parse_rejects_mixed_valid_and_invalid_segments_strictly():
    parsed = parse_temporal_response(
        '{"segments": [[1, 3], [10, 20]], "subject_hint": "person"}',
        duration_sec=8.0,
    )
    assert not _is_valid(parsed)
    assert _segments(parsed) == []
    assert _errors(parsed)


def test_parse_uses_later_valid_json_object_and_rejects_boolean_times():
    text = 'draft {"segments": []} final {"segments": [[1, 3]], "subject_hint": "car"}'
    parsed = parse_temporal_response(text, duration_sec=8.0)
    assert _is_valid(parsed)
    assert _segments(parsed) == [(1.0, 3.0)]
    assert parsed.subject_hint == "car"

    boolean_result = parse_temporal_response('{"segments": [[true, false]]}', duration_sec=8.0)
    assert not _is_valid(boolean_result)
    assert _segments(boolean_result) == []


def test_normalize_reverses_clips_pads_and_merges_segments():
    result = normalize_temporal_segments(
        [(8.0, 3.0), (-2.0, 4.0), (5.5, 9.0), (5.5, 9.0)],
        duration_sec=10.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        max_segments=3,
    )
    assert isinstance(result, TemporalNormalizationResult)
    assert _is_valid(result)
    assert _segments(result) == [(0.0, 9.0)]
    corrections = list(getattr(result, "corrections", []) or [])
    assert corrections, "normalization must record repairs such as reversed or clipped bounds"


def test_normalize_does_not_collapse_out_of_range_to_zero_length():
    result = normalize_temporal_segments(
        [(10.0, 20.0)],
        duration_sec=8.0,
        min_segment_sec=4.0,
        padding_sec=3.0,
        max_segments=3,
    )
    assert not _is_valid(result)
    assert _segments(result) == []
    assert (8.0, 8.0) not in _segments(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_normalize_rejects_non_finite_times(value):
    result = normalize_temporal_segments(
        [(value, 4.0)],
        duration_sec=8.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        max_segments=3,
    )
    assert not _is_valid(result)
    assert _segments(result) == []
    assert _errors(result)


def test_normalize_zero_length_is_never_preserved():
    result = normalize_temporal_segments(
        [(5.0, 5.0)],
        duration_sec=10.0,
        min_segment_sec=4.0,
        padding_sec=0.0,
        max_segments=3,
    )
    segments = _segments(result)
    assert segments != [(5.0, 5.0)]
    assert not segments or all(end > start for start, end in segments)
    assert not _is_valid(result) or all(end - start >= 4.0 - 1e-6 for start, end in segments)


def test_normalize_expands_short_segment_within_video_bounds():
    result = normalize_temporal_segments(
        [(5.0, 5.1)],
        duration_sec=10.0,
        min_segment_sec=4.0,
        padding_sec=0.0,
        max_segments=3,
    )
    assert _is_valid(result)
    [(start, end)] = _segments(result)
    assert 0.0 <= start < end <= 10.0
    assert end - start == pytest.approx(4.0)


def test_normalize_padding_overlap_and_max_segments():
    result = normalize_temporal_segments(
        [(2.0, 3.0), (3.5, 4.0), (8.0, 9.0), (12.0, 13.0)],
        duration_sec=20.0,
        min_segment_sec=1.0,
        padding_sec=0.5,
        max_segments=2,
    )
    assert _is_valid(result)
    assert len(_segments(result)) == 2
    assert _segments(result)[0] == pytest.approx((1.5, 4.75))


def test_temporal_coverage_ratio_uses_union_and_handles_empty_duration():
    assert temporal_coverage_ratio([(0.0, 2.0), (1.0, 4.0)], 8.0) == pytest.approx(0.5)
    assert temporal_coverage_ratio([], 8.0) == 0.0
    assert temporal_coverage_ratio([(0.0, 1.0)], 0.0) == 0.0


def test_qwen_prompt_contains_real_metadata_and_no_copyable_numeric_schema(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=10.0, fps=30.0)
    provider = ScriptedResponseProvider(['{"segments": [[1.0, 4.0]], "subject_hint": "person"}'])
    detector = _make_qwen_detector(
        provider,
        short_video_full_sec=0.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        min_coverage_ratio=0.0,
        max_segments=3,
    )

    result = detector.detect("unused.mp4")
    assert _segments(result) == [(1.0, 4.0)]
    assert len(provider.calls) == 1
    prompt = provider.prompts[0]
    compact = "".join(prompt.split())
    assert "10.000" in prompt or "10.0" in prompt or "10 seconds" in prompt
    assert "30.000" in prompt or "30.0" in prompt or "30 FPS" in prompt
    assert "1.000" in prompt or "1.0" in prompt
    assert "start_sec" in prompt and "end_sec" in prompt
    assert "JSON" in prompt
    assert "Never enumerate every sampled frame or every second" in prompt
    assert "Merge touching or adjacent intervals" in prompt
    assert "do not quote the numbers" in prompt
    assert "title card" in prompt and "camera shake" in prompt
    assert "Every input video has at least one highlight-worthy moment" in prompt
    assert "appearance, class, and action" in prompt
    assert "Avoid the generic word" in prompt
    assert "never put subject" in prompt
    assert "[[10,20]]" not in compact


def test_qwen_video_request_uses_local_uri_and_bounded_visual_budget(monkeypatch, tmp_path):
    _patch_video_meta(monkeypatch, duration_sec=200.0, fps=30.0)
    detector = _make_qwen_detector(ScriptedResponseProvider([]))
    detector.detect_fps = 1.0
    detector.max_video_frames = 64
    detector.max_frame_pixels = 360 * 420
    detector.total_video_pixels = 4096 * 28 * 28

    request = detector._video_request(str(tmp_path / "video.mp4"))

    assert os.path.isabs(request["video"])
    assert request["nframes"] == 64
    assert request["nframes"] % 2 == 0
    assert request["max_pixels"] == 360 * 420
    assert request["total_pixels"] == 4096 * 28 * 28


def test_qwen_single_video_fps_list_is_normalized_for_new_transformers():
    normalized = QwenTemporalDetector._normalize_video_kwargs(
        {"fps": [1.0135], "do_sample_frames": False}
    )
    assert normalized == {"fps": 1.0135, "do_sample_frames": False}


def test_qwen_compat_transcode_is_cached_on_e_drive_style_temp(monkeypatch, tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    cache = tmp_path / "compat"
    monkeypatch.setenv("QWEN_VIDEO_COMPAT_DIR", str(cache))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"compatible")

    monkeypatch.setattr(temporal.subprocess, "run", fake_run)

    first = QwenTemporalDetector._transcode_video_for_rgb(str(source))
    second = QwenTemporalDetector._transcode_video_for_rgb(str(source))

    assert first == second
    assert Path(first).parent == cache
    assert Path(first).read_bytes() == b"compatible"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == "ffmpeg"
    assert "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709" in command
    assert kwargs == {"check": True, "capture_output": True, "text": True}


def test_qwen_retries_once_with_previous_output_error_and_duration(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=8.0, fps=25.0)
    invalid = '{"segments": [[10, 20]], "subject_hint": "person"}'
    provider = ScriptedResponseProvider(
        [invalid, '{"segments": [[2.0, 6.0]], "subject_hint": "person"}']
    )
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        short_video_full_sec=0.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        min_coverage_ratio=0.0,
    )

    result = detector.detect("unused.mp4")
    assert _segments(result) == [(2.0, 6.0)]
    assert len(provider.calls) == 2
    retry_prompt = provider.prompts[1]
    assert invalid in retry_prompt
    assert "8.000" in retry_prompt or "8.0" in retry_prompt or "8 seconds" in retry_prompt
    assert "invalid" in retry_prompt.lower() or "error" in retry_prompt.lower()
    assert "per-second or per-frame" in retry_prompt
    assert "Merge adjacent ranges" in retry_prompt
    assert "Remove every non-time entry" in retry_prompt
    assert getattr(result, "attempts", 2) == 2
    assert _source(result) in {"vlm", "vlm_retry"}


def test_two_vlm_failures_fall_back_to_motion(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=10.0, fps=30.0)
    provider = ScriptedResponseProvider(["not json", '{"segments": []}'])
    motion = StubTemporalDetector(TemporalResult([(2.0, 7.0)], raw="motion-window", source="motion"))
    full = StubTemporalDetector(TemporalResult([(0.0, 10.0)], raw="full-video", source="full"))
    monkeypatch.setattr(temporal, "MotionTemporalDetector", lambda *args, **kwargs: motion)
    monkeypatch.setattr(temporal, "FullVideoTemporalDetector", lambda *args, **kwargs: full)
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        temporal_fallback="motion",
        short_video_full_sec=0.0,
        min_coverage_ratio=0.0,
        fallback_detector=motion,
        full_detector=full,
    )

    result = detector.detect("unused.mp4")
    assert len(provider.calls) == 2
    assert _segments(result) == [(2.0, 7.0)]
    assert motion.calls == ["unused.mp4"]
    assert not full.calls
    assert "motion" in _source(result) or _fallback_used(result) == "motion"


def test_empty_motion_fallback_escalates_to_full(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=10.0, fps=30.0)
    provider = ScriptedResponseProvider(["not json", "still not json"])
    motion = StubTemporalDetector(TemporalResult([], raw="motion-empty", source="motion", valid=False))
    full = StubTemporalDetector(TemporalResult([(0.0, 10.0)], raw="full-video", source="full"))
    monkeypatch.setattr(temporal, "MotionTemporalDetector", lambda *args, **kwargs: motion)
    monkeypatch.setattr(temporal, "FullVideoTemporalDetector", lambda *args, **kwargs: full)
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        temporal_fallback="motion",
        short_video_full_sec=0.0,
        min_coverage_ratio=0.0,
        fallback_detector=motion,
        full_detector=full,
    )

    result = detector.detect("unused.mp4")
    assert _segments(result) == [(0.0, 10.0)]
    assert motion.calls == ["unused.mp4"]
    assert full.calls == ["unused.mp4"]
    assert "full" in _source(result) or _fallback_used(result) == "full"


def test_vlm_provider_exception_still_uses_fallback(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=10.0, fps=30.0)

    class RaisingProvider:
        def __init__(self):
            self.calls = 0

        def __call__(self, _video_path, _prompt):
            self.calls += 1
            raise RuntimeError("synthetic inference failure")

    provider = RaisingProvider()
    motion = StubTemporalDetector(TemporalResult([(1.0, 8.0)], raw="motion-window", source="motion"))
    full = StubTemporalDetector(TemporalResult([(0.0, 10.0)], raw="full-video", source="full"))
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        temporal_fallback="motion",
        short_video_full_sec=0.0,
        min_coverage_ratio=0.0,
        fallback_detector=motion,
        full_detector=full,
    )

    result = detector.detect("unused.mp4")
    assert provider.calls == 2
    assert _segments(result) == [(1.0, 8.0)]
    assert motion.calls == ["unused.mp4"]
    assert any("failure" in error.lower() or "runtime" in error.lower() for error in _errors(result))


def test_short_video_uses_full_video_without_calling_vlm(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=3.3, fps=30.0)
    provider = ScriptedResponseProvider(['{"segments": [[1.0, 2.0]]}'])
    detector = _make_qwen_detector(
        provider,
        short_video_full_sec=15.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        min_coverage_ratio=0.0,
    )

    result = detector.detect("short.mp4")
    assert provider.calls == []
    [(start, end)] = _segments(result)
    assert start == 0.0
    assert 3.2 <= end <= 3.3 + 1e-6
    assert temporal_coverage_ratio(_segments(result), 3.3) >= 0.98
    assert "short" in _source(result) or "full" in _source(result)


def test_low_coverage_vlm_result_cannot_be_final_empty_or_tiny(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=20.0, fps=30.0)
    provider = ScriptedResponseProvider(
        [
            '{"segments": [[9.0, 10.0]], "subject_hint": "person"}',
            '{"segments": [[9.0, 10.0]], "subject_hint": "person"}',
        ]
    )
    motion = StubTemporalDetector(TemporalResult([(4.0, 12.0)], raw="motion-window", source="motion"))
    full = StubTemporalDetector(TemporalResult([(0.0, 20.0)], raw="full-video", source="full"))
    monkeypatch.setattr(temporal, "MotionTemporalDetector", lambda *args, **kwargs: motion)
    monkeypatch.setattr(temporal, "FullVideoTemporalDetector", lambda *args, **kwargs: full)
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        temporal_fallback="motion",
        short_video_full_sec=0.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        min_coverage_ratio=0.30,
        fallback_detector=motion,
        full_detector=full,
    )

    result = detector.detect("unused.mp4")
    assert _segments(result)
    assert temporal_coverage_ratio(_segments(result), 20.0) >= 0.30
    assert _segments(result) != [(9.0, 10.0)]


def test_motion_fallback_at_video_boundary_expands_without_unnecessary_full(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=20.0, fps=30.0)
    provider = ScriptedResponseProvider(["not json", "still not json"])
    motion = StubTemporalDetector(TemporalResult([(16.0, 20.0)], raw="motion-window", source="motion"))
    full = StubTemporalDetector(TemporalResult([(0.0, 20.0)], raw="full-video", source="full"))
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        temporal_fallback="motion",
        short_video_full_sec=0.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        min_coverage_ratio=0.30,
        fallback_detector=motion,
        full_detector=full,
    )

    result = detector.detect("unused.mp4")
    assert temporal_coverage_ratio(_segments(result), 20.0) >= 0.30
    assert motion.calls == ["unused.mp4"]
    assert full.calls == []
    assert _fallback_used(result) == "motion"


def test_empty_motion_after_low_coverage_vlm_escalates_to_full(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=20.0, fps=30.0)
    provider = ScriptedResponseProvider(
        [
            '{"segments": [[9.0, 10.0]], "subject_hint": "person"}',
            '{"segments": [[9.0, 10.0]], "subject_hint": "person"}',
        ]
    )
    motion = StubTemporalDetector(
        TemporalResult([], raw="motion-empty", source="motion", valid=False, errors=["no samples"])
    )
    full = StubTemporalDetector(TemporalResult([(0.0, 20.0)], raw="full-video", source="full"))
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        temporal_fallback="motion",
        short_video_full_sec=0.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        min_coverage_ratio=0.30,
        fallback_detector=motion,
        full_detector=full,
    )

    result = detector.detect("unused.mp4")
    assert _segments(result) == [(0.0, 20.0)]
    assert full.calls == ["unused.mp4"]
    assert _fallback_used(result) == "full"
    assert result.subject_hint == "person"
    assert any("no samples" in error.lower() for error in _errors(result))


def test_best_low_coverage_vlm_candidate_is_not_replaced_by_worse_retry(monkeypatch):
    _patch_video_meta(monkeypatch, duration_sec=20.0, fps=30.0)
    provider = ScriptedResponseProvider(
        [
            '{"segments": [[2.0, 7.0]], "subject_hint": "person"}',
            '{"segments": [[9.0, 10.0]], "subject_hint": "object"}',
        ]
    )
    detector = _make_qwen_detector(
        provider,
        vlm_retries=1,
        temporal_fallback="none",
        short_video_full_sec=0.0,
        min_segment_sec=1.0,
        padding_sec=0.0,
        min_coverage_ratio=0.50,
    )

    result = detector.detect("unused.mp4")
    assert _segments(result) == [(2.0, 7.0)]
    assert result.subject_hint == "person"
