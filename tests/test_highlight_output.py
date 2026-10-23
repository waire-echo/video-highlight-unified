import json

import pytest

from video_highlight.highlight_output import (
    incomplete_prefix_candidate,
    lenient_normalized_answer,
    lenient_parse_answer,
    strict_parse_answer,
)


@pytest.mark.parametrize(
    ("raw", "action"),
    [
        (
            '{"highlights":[{"start":0.0, "end":12.0"}]}',
            "remove_extra_field_value_quote",
        ),
        (
            '{"highlights":[{"start":1.0, "end":5.0"}, '
            '"highlights":[{"start":2.0, "end":6.0}]}',
            "remove_repeated_highlights_key",
        ),
        (
            '<highlights>[{"start":10.0, "end":28.0"}, '
            '{"start":29.0, "end":37.0"}]}',
            "remove_output_prefix",
        ),
        (
            '{"highlights":[{"start":0.0, "end":8.4]}}',
            "extract_complete_start_end_pairs",
        ),
        (
            '{"highlights":[{"start":0.0, "end":5.0], '
            '"description":"test"}]}',
            "remove_unsupported_output_field",
        ),
    ],
)
def test_lenient_repairs_known_output_shapes(raw, action):
    obj, repair = lenient_parse_answer(raw)
    assert obj["highlights"]
    assert action in repair["actions"]
    assert repair["changed"] is True
    assert repair["original_sha256"]


def test_strict_mode_remains_strict():
    with pytest.raises(json.JSONDecodeError):
        strict_parse_answer('{"highlights":[{"start":0,"end":4"}]}')


def test_strict_mode_does_not_discard_zero_length_rows():
    with pytest.raises(ValueError, match="positive duration"):
        lenient_normalized_answer(
            '{"highlights":[{"start":0,"end":5},'
            '{"start":5,"end":5}]}',
            10,
            lenient=False,
        )


def test_small_video_end_overrun_is_clipped_and_audited():
    obj, spans, normalization = lenient_normalized_answer(
        '{"highlights":[{"start":0,"end":6.2}]}', 5.92, lenient=True
    )
    assert obj == {"highlights": [{"start": 0.0, "end": 5.92}]}
    assert spans == [[0.0, 5.92]]
    assert normalization["time_changes"][0]["reason"] == "clipped_to_video_end"


def test_large_video_end_overrun_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        lenient_normalized_answer(
            '{"highlights":[{"start":0,"end":6.5}]}', 5.92, lenient=True
        )


def test_start_outside_video_is_not_silently_clipped():
    with pytest.raises(ValueError, match="outside"):
        lenient_normalized_answer(
            '{"highlights":[{"start":-0.2,"end":4}]}', 10, lenient=True
        )


def test_lenient_mode_does_not_invent_missing_pairs():
    obj, spans, normalization = lenient_normalized_answer(
        '{"highlights":[{"text":"no timing fields"}]}',
        10,
        lenient=True,
    )
    assert obj == {"highlights": []}
    assert spans == []
    assert normalization["repair"]["actions"]


def test_lenient_mode_preserves_an_intentional_empty_result():
    obj, spans, normalization = lenient_normalized_answer(
        '{"highlights":[]}', 10, lenient=True
    )
    assert obj == {"highlights": []}
    assert spans == []
    assert normalization["repair"]["changed"] is False


def test_lenient_mode_discards_zero_length_rows_with_audit():
    obj, spans, normalization = lenient_normalized_answer(
        '{"highlights":[{"start":0,"end":5},'
        '{"start":5,"end":5}]}',
        10,
        lenient=True,
    )
    assert obj == {"highlights": [{"start": 0.0, "end": 5.0}]}
    assert spans == [[0.0, 5.0]]
    assert "discard_nonpositive_highlights" in (
        normalization["repair"]["actions"]
    )


def test_exclusive_last_frame_timestamp_is_not_a_real_overrun():
    obj, spans, normalization = lenient_normalized_answer(
        '{"highlights":[{"start":0,"end":14.933333}]}',
        14.916667,
        lenient=True,
        final_frame_pts_sec=14.933333,
    )
    assert obj == {"highlights": [{"start": 0.0, "end": 14.916667}]}
    assert spans == [[0.0, 14.916667]]
    assert normalization["time_changes"][0]["reason"] == (
        "exclusive_last_frame_pts"
    )


def test_strict_small_end_roundoff_keeps_legacy_reason():
    _, spans, normalization = lenient_normalized_answer(
        '{"highlights":[{"start":0,"end":5.0000000001}]}',
        5,
        lenient=False,
    )
    assert spans == [[0.0, 5.0]]
    assert normalization["time_changes"][0]["reason"] == "floating_roundoff"


def test_closing_loop_is_cut_before_repetition_and_audited():
    raw = (
        '{"highlights":[{"start":1,"end":2}]}}]}}]}}]}}'
    )
    obj, repair = lenient_parse_answer(raw)
    assert obj == {"highlights": [{"start": 1, "end": 2}]}
    assert "truncate_repeated_closing_loop" in repair["actions"]
    assert repair["closing_loop"]["repetitions"] >= 3
    fragment = repair["discarded_fragments"][-1]
    assert fragment["reason"] == "repeated_closing_loop"
    assert fragment["removed_chars"] == len(fragment["removed_text"])


def test_only_one_nested_highlights_wrapper_is_allowed():
    one = '{"highlights":[{"highlights":[{"start":1,"end":2}]}]}'
    obj, repair = lenient_parse_answer(one)
    assert obj == {"highlights": [{"start": 1, "end": 2}]}
    assert repair["nested_highlights_containers"] == 1
    with pytest.raises(ValueError, match="More than one"):
        lenient_parse_answer(
            '{"highlights":[{"highlights":[{"highlights":'
            '[{"start":1,"end":2}]}]}]}'
        )


def test_duplicate_overlap_and_adjacent_intervals_are_fully_audited():
    obj, spans, normalization = lenient_normalized_answer(
        '{"highlights":['
        '{"start":1,"end":3},'
        '{"start":1,"end":3},'
        '{"start":2,"end":4},'
        '{"start":4,"end":5},'
        '{"start":8,"end":9}'
        ']}',
        10,
        lenient=True,
    )
    assert spans == [[1.0, 5.0], [8.0, 9.0]]
    assert obj["highlights"] == [
        {"start": 1.0, "end": 5.0},
        {"start": 8.0, "end": 9.0},
    ]
    repair = normalization["repair"]
    assert len(repair["removed_duplicate_intervals"]) == 1
    assert len(repair["merged_intervals"]) == 2
    assert [row["kind"] for row in repair["merged_intervals"]] == [
        "overlap",
        "adjacent",
    ]


def test_incomplete_last_pair_is_dropped_and_audited():
    obj, repair = lenient_parse_answer(
        '{"highlights":[{"start":1,"end":2},'
        '{"start":3,"end":'
    )
    assert obj == {"highlights": [{"start": 1, "end": 2}]}
    assert repair["dropped_incomplete_pairs"][0]["field"] == "start"
    assert repair["dropped_incomplete_pairs"][0]["value"] == 3


def test_contiguous_rolling_windows_merge_to_one_envelope():
    windows = [
        {"start": float(i), "end": float(i + 1)}
        for i in range(10)
    ]
    _, spans, normalization = lenient_normalized_answer(
        json.dumps({"highlights": windows}), 10, lenient=True
    )
    assert spans == [[0.0, 10.0]]
    detection = normalization["repair"]["rolling_window_detections"][0]
    assert detection["window_count"] == 10
    assert detection["overlap"] is True
    assert "rolling_window_loop_merged_to_envelope" in (
        normalization["quality_flags"]
    )


def test_contiguous_rolling_windows_preserve_envelope_past_duration():
    windows = [
        {"start": round(0.3 * index, 1), "end": round(2 + 0.3 * index, 1)}
        for index in range(10)
    ]
    with pytest.raises(ValueError, match="outside original time range"):
        lenient_normalized_answer(
            json.dumps({"highlights": windows}), 1.2, lenient=True
        )


def test_fixed_step_disjoint_window_enumeration_is_preserved():
    windows = [
        {"start": float(4 + 10 * index), "end": float(10 + 10 * index)}
        for index in range(9)
    ]
    _, spans, normalization = lenient_normalized_answer(
        json.dumps({"highlights": windows}), 120, lenient=True
    )
    assert len(spans) == 9
    detection = normalization["repair"]["rolling_window_detections"][0]
    assert detection["window_count"] == 9
    assert detection["overlap"] is False
    assert detection["decision"] == "quarantine"


def test_rolling_detection_does_not_join_runs_across_a_large_gap():
    windows = [
        {"start": float(4 + 10 * index), "end": float(10 + 10 * index)}
        for index in range(9)
    ] + [
        {"start": 136.0, "end": 142.0},
        {"start": 146.0, "end": 152.0},
    ]
    _, spans, normalization = lenient_normalized_answer(
        json.dumps({"highlights": windows}), 180, lenient=True
    )
    assert spans == [
        [4.0, 10.0],
        [14.0, 20.0],
        [24.0, 30.0],
        [34.0, 40.0],
        [44.0, 50.0],
        [54.0, 60.0],
        [64.0, 70.0],
        [74.0, 80.0],
        [84.0, 90.0],
        [136.0, 142.0],
        [146.0, 152.0],
    ]
    detection = normalization["repair"]["rolling_window_detections"][0]
    assert detection["window_count"] == 9


def test_single_interval_quality_flags_do_not_imply_semantic_correctness():
    _, _, normalization = lenient_normalized_answer(
        '{"highlights":[{"start":0,"end":5}]}', 10, lenient=True
    )
    assert "single_highlight_starts_at_zero" in normalization["quality_flags"]
    _, _, normalization = lenient_normalized_answer(
        '{"highlights":[{"start":2,"end":5}]}', 10, lenient=True
    )
    assert "single_highlight" in normalization["quality_flags"]
    _, _, normalization = lenient_normalized_answer(
        '{"highlights":[{"start":1,"end":2},{"start":5,"end":6}]}',
        10,
        lenient=True,
    )
    assert "multiple_highlights" in normalization["quality_flags"]


def test_eight_fixed_step_windows_do_not_cross_loop_threshold():
    windows = [
        {"start": float(4 + 10 * index), "end": float(10 + 10 * index)}
        for index in range(8)
    ]
    _, spans, _ = lenient_normalized_answer(
        json.dumps({"highlights": windows}), 120, lenient=True
    )
    assert len(spans) == 8


def test_client_rejects_more_than_the_declared_schema_limit():
    raw = json.dumps({
        "highlights": [
            {"start": float(index), "end": float(index + 0.5)}
            for index in range(3)
        ]
    })
    with pytest.raises(ValueError, match="at most 2"):
        lenient_normalized_answer(raw, 10, max_items=2)
    with pytest.raises(ValueError, match="at most 2"):
        lenient_normalized_answer(raw, 10, lenient=True, max_items=2)


def test_removed_character_total_matches_fragment_audit():
    raw = (
        '{"highlights":[{"start":1,"end":2"}]}'
        '}}]}}]}}'
    )
    _, repair = lenient_parse_answer(raw)
    assert sum(
        item["removed_chars"]
        for item in repair["discarded_fragments"]
    ) == repair["removed_characters"]
    assert repair["original_sha256"][:12]


def test_truncated_prefix_candidate_is_incomplete_and_never_auto_accepted():
    raw = (
        '{"highlights":[{"start":1,"end":2},'
        '{"start":3,"end":4'
    )
    candidate = incomplete_prefix_candidate(raw, 10)
    assert candidate["status"] == "incomplete"
    assert candidate["segments"] == [[1.0, 2.0]]
    assert candidate["closed_pair_count"] == 1
    assert candidate["diagnostic_only"] is True
    assert "generation_truncated" in candidate["quality_flags"]


def test_unclosed_truncated_prefix_is_still_diagnostic_only():
    candidate = incomplete_prefix_candidate(
        '{"highlights":[{"start":0.0', 10
    )
    assert candidate["status"] == "incomplete"
    assert candidate["segments"] == []
    assert candidate["diagnostic_only"] is True
