import math
import os
import sys

import cv2
import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from video_highlight.spatial_legacy import (
    CandidateScore,
    CenterSpatialDetector,
    DetectionCandidate,
    YOLOSpatialDetector,
)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoords:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeBox:
    def __init__(self, xyxy, class_id, confidence):
        self.xyxy = [FakeCoords(xyxy)]
        self.cls = [FakeScalar(class_id)]
        self.conf = [FakeScalar(confidence)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, results, names):
        self.results = list(results)
        self.names = names
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append({"shape": frame.shape, **kwargs})
        if not self.results:
            raise AssertionError("fake model was called more often than expected")
        return [FakeResult(self.results.pop(0))]


def _frame(width=1000, height=1000):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _detector(results, names=None, **kwargs):
    if names is None:
        names = {0: "person", 2: "car", 7: "truck", 32: "sports ball", 58: "potted plant"}
    model = FakeModel(results, names)
    detector = YOLOSpatialDetector(model=model, **kwargs)
    return detector, model


def _candidate(index, xyxy, class_id, class_name, confidence, frame_size=(1000, 1000)):
    width, height = frame_size
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    area = (x2 - x1) * (y2 - y1)
    group = {
        "person": "person",
        "sports ball": "ball",
        "car": "vehicle",
        "truck": "vehicle",
    }.get(class_name, "other")
    return DetectionCandidate(
        index=index,
        xyxy=(x1, y1, x2, y2),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        width=x2 - x1,
        height=y2 - y1,
        area=area,
        area_ratio=area / float(width * height),
        class_id=class_id,
        class_name=class_name,
        class_group=group,
        confidence=confidence,
    )


def test_enhanced_person_hint_beats_large_sports_irrelevant_box():
    boxes = [
        FakeBox([410, 220, 610, 620], 0, 0.72),
        FakeBox([40, 40, 960, 800], 58, 0.96),
    ]
    detector, model = _detector([boxes], stateful=False, person_ball_weight=0.0)

    cx, cy, info = detector.detect_center(_frame(), subject_hint="person", frame_id=10)

    assert (cx, cy) == pytest.approx((510.0, 420.0))
    assert info["class"] == "person"
    assert info["hint_supported"] is True
    assert info["hint_match"] is True
    assert "hint" in info["selection_reason"]
    assert model.calls == [{"shape": (1000, 1000, 3), "conf": 0.15, "verbose": False}]


def test_ball_hint_selects_small_low_confidence_ball_and_uses_ball_threshold():
    boxes = [
        FakeBox([330, 240, 670, 600], 0, 0.90),
        FakeBox([620, 430, 680, 497], 32, 0.18),
    ]
    detector, model = _detector(
        [boxes], conf=0.25, ball_conf=0.15, stateful=False, person_ball_weight=0.0
    )

    cx, cy, info = detector.detect_center(_frame(), subject_hint="ball", frame_id=20)

    assert model.calls[0]["conf"] == pytest.approx(0.15)
    assert info["detections"] == 2
    assert info["class"] == "sports ball"
    assert info["conf"] == pytest.approx(0.18)
    assert info["hint_match"] is True
    assert (cx, cy) == pytest.approx((650.0, 463.5))


def test_non_ball_below_main_confidence_is_filtered_but_low_conf_ball_remains():
    boxes = [
        FakeBox([100, 100, 300, 500], 0, 0.24),
        FakeBox([480, 470, 520, 510], 32, 0.16),
    ]
    detector, _model = _detector([boxes], conf=0.25, ball_conf=0.15, stateful=False)

    candidates = detector.detect_candidates(_frame())

    assert [candidate.class_name for candidate in candidates] == ["sports ball"]
    assert candidates[0].confidence == pytest.approx(0.16)


def test_car_hint_is_specific_while_vehicle_hint_accepts_the_stronger_truck():
    boxes = [
        FakeBox([180, 330, 430, 650], 2, 0.72),
        FakeBox([540, 310, 820, 670], 7, 0.78),
    ]
    car_detector, _ = _detector([boxes], stateful=False, person_ball_weight=0.0)
    vehicle_detector, _ = _detector([boxes], stateful=False, person_ball_weight=0.0)

    _cx, _cy, car_info = car_detector.detect_center(_frame(), subject_hint="car")
    _cx, _cy, vehicle_info = vehicle_detector.detect_center(_frame(), subject_hint="vehicle")

    assert car_info["class"] == "car"
    assert car_info["hint_match"] is True
    assert vehicle_info["class"] == "truck"
    assert vehicle_info["hint_match"] is True


@pytest.mark.parametrize("hint", [None, "", "object", "stage", "unicorn"])
def test_neutral_or_unsupported_hint_does_not_change_selection(hint):
    boxes = [
        FakeBox([120, 200, 360, 650], 0, 0.80),
        FakeBox([600, 300, 850, 650], 2, 0.75),
    ]
    detector, _model = _detector([boxes], stateful=False, person_ball_weight=0.0)

    _cx, _cy, info = detector.detect_center(_frame(), subject_hint=hint)

    assert info["class"] == "person"
    assert info["hint_supported"] is False
    assert info["hint_match"] is False


def test_continuity_keeps_same_person_when_other_person_becomes_base_best():
    first = [
        FakeBox([100, 200, 300, 600], 0, 0.92),
        FakeBox([650, 200, 850, 600], 0, 0.68),
    ]
    second = [
        FakeBox([120, 200, 320, 600], 0, 0.64),
        FakeBox([650, 200, 850, 600], 0, 0.98),
    ]
    detector, _model = _detector(
        [first, second], stateful=True, continuity_weight=0.70, person_ball_weight=0.0
    )
    detector.begin_video("v", "unused.mp4", 30.0, 1000, 1000)
    detector.begin_segment(0, 0, 30, subject_hint="person")

    first_center = detector.detect_center(_frame(), subject_hint="person", frame_id=0)
    second_center = detector.detect_center(_frame(), subject_hint="person", frame_id=15)

    assert first_center[:2] == pytest.approx((200.0, 400.0))
    assert second_center[:2] == pytest.approx((220.0, 400.0))
    second_info = second_center[2]
    assert second_info["continuity_changed_selection"] is True
    assert "continuity" in second_info["selection_reason"]
    assert second_info["track_id"] == 1
    assert second_info["track_age"] == 2
    assert second_info["track_switched"] is False


def test_begin_segment_resets_track_and_allows_new_base_best_person():
    first = [
        FakeBox([100, 200, 300, 600], 0, 0.92),
        FakeBox([650, 200, 850, 600], 0, 0.68),
    ]
    second = [
        FakeBox([120, 200, 320, 600], 0, 0.64),
        FakeBox([650, 200, 850, 600], 0, 0.98),
    ]
    detector, _model = _detector(
        [first, second], stateful=True, continuity_weight=0.70, person_ball_weight=0.0
    )
    detector.begin_video("v", "unused.mp4", 30.0, 1000, 1000)
    detector.begin_segment(0, 0, 10, subject_hint="person")
    detector.detect_center(_frame(), subject_hint="person", frame_id=0)
    detector.end_segment()
    detector.begin_segment(1, 20, 30, subject_hint="person")

    cx, cy, info = detector.detect_center(_frame(), subject_hint="person", frame_id=20)

    assert (cx, cy) == pytest.approx((750.0, 400.0))
    assert info["previous_bbox"] is None
    assert info["continuity_changed_selection"] is False
    assert info["track_id"] == 1
    assert info["track_age"] == 1


@pytest.mark.parametrize(
    ("fallback_frame", "expected_fallback"),
    [(_frame(200, 100), "center"), (None, "motion-region")],
)
def test_one_missing_keyframe_holds_track_then_second_uses_motion_or_center(
    fallback_frame, expected_fallback
):
    detected = [FakeBox([20, 20, 60, 80], 0, 0.90)]
    detector, _model = _detector(
        [detected, [], []],
        names={0: "person"},
        stateful=True,
        max_missed_keyframes=1,
        person_ball_weight=0.0,
    )
    detector.begin_video("v", "unused.mp4", 30.0, 200, 100)
    detector.begin_segment(0, 0, 30, subject_hint="person")
    base_frame = _frame(200, 100)
    if fallback_frame is None:
        fallback_frame = _frame(200, 100)
        cv2.rectangle(fallback_frame, (35, 15), (165, 85), (255, 255, 255), -1)

    detector.detect_center(base_frame, subject_hint="person", frame_id=0)
    held_cx, held_cy, held_info = detector.detect_center(
        fallback_frame, subject_hint="person", frame_id=15
    )
    final_cx, final_cy, final_info = detector.detect_center(
        fallback_frame, subject_hint="person", frame_id=30
    )

    assert (held_cx, held_cy) == pytest.approx((40.0, 50.0))
    assert held_info["fallback"] == "track-hold"
    assert held_info["track_misses"] == 1
    assert final_info["fallback"] == expected_fallback
    if expected_fallback == "center":
        assert (final_cx, final_cy) == pytest.approx((100.0, 50.0))
    else:
        assert 80.0 <= final_cx <= 120.0
        assert 40.0 <= final_cy <= 60.0


def test_invalid_nan_degenerate_and_low_confidence_boxes_are_filtered():
    boxes = [
        FakeBox([math.nan, 0, 10, 10], 0, 0.9),
        FakeBox([10, 10, 10, 50], 0, 0.9),
        FakeBox([20, 20, 80, 80], 0, math.nan),
        FakeBox([20, 20, 80, 80], 0, 0.20),
        FakeBox([-20, -10, 40, 50], 32, 0.18),
    ]
    detector, _model = _detector([boxes], conf=0.25, ball_conf=0.15, stateful=False)

    candidates = detector.detect_candidates(_frame(100, 100))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.class_name == "sports ball"
    assert candidate.xyxy == pytest.approx((0.0, 0.0, 40.0, 50.0))
    assert candidate.touches_border is True


@pytest.mark.parametrize(
    ("names", "class_id", "expected"),
    [({0: "PERSON"}, 0, "person"), (["PERSON", "CAR"], 1, "car")],
)
def test_names_mapping_supports_dict_and_list(names, class_id, expected):
    detector, _model = _detector(
        [[FakeBox([10, 10, 80, 90], class_id, 0.9)]], names=names, stateful=False
    )

    candidates = detector.detect_candidates(_frame(100, 100))

    assert len(candidates) == 1
    assert candidates[0].class_name == expected


def test_legacy_scoring_matches_area_confidence_and_old_detect_signature():
    boxes = [
        FakeBox([0, 0, 100, 100], 0, 0.40),
        FakeBox([120, 0, 200, 80], 2, 0.80),
    ]
    detector, _model = _detector(
        [boxes], scoring="legacy", stateful=False, person_ball_weight=0.0
    )

    cx, cy, info = detector.detect_center(_frame(300, 200), "person")

    # person: 2 * 10000 * .4 = 8000; car: 6400 * .8 = 5120
    assert (cx, cy) == pytest.approx((50.0, 50.0))
    assert info["class"] == "person"
    assert info["scoring"] == "legacy"
    assert info["selection_reason"] == "hint"


def test_candidate_score_api_and_raw_candidate_log_are_truncated():
    boxes = [
        FakeBox([100, 100, 300, 500], 0, 0.95),
        FakeBox([350, 100, 550, 500], 0, 0.85),
        FakeBox([600, 100, 800, 500], 0, 0.75),
        FakeBox([400, 520, 450, 570], 32, 0.30),
    ]
    detector, _model = _detector(
        [boxes], stateful=False, max_raw_candidates=2, person_ball_weight=0.0
    )
    candidates = detector.detect_candidates(_frame())
    scores = detector.score_candidates(candidates, _frame().shape, subject_hint="person")

    assert all(isinstance(item, CandidateScore) for item in scores)
    assert all(isinstance(item.candidate, DetectionCandidate) for item in scores)

    # Feed the same boxes through a fresh detector because detect_candidates consumes one fake result.
    detector, _model = _detector(
        [boxes], stateful=False, max_raw_candidates=2, person_ball_weight=0.0
    )
    _cx, _cy, info = detector.detect_center(_frame(), subject_hint="person")

    assert info["candidates_total"] == 4
    assert info["candidates_logged"] == 2
    assert len(info["candidates"]) == 2
    assert info["candidates"][0]["selected"] is True
    assert sum(bool(item["selected"]) for item in info["candidates"]) == 1


def test_video_summary_counts_lifecycle_selection_and_fallbacks():
    detected = [
        FakeBox([100, 100, 300, 500], 0, 0.80),
        FakeBox([600, 100, 800, 500], 2, 0.70),
    ]
    detector, _model = _detector(
        [detected, [], []],
        stateful=True,
        max_missed_keyframes=1,
        max_raw_candidates=1,
        person_ball_weight=0.0,
    )
    detector.begin_video("v", "unused.mp4", 30.0, 1000, 1000)
    detector.begin_segment(0, 0, 30, subject_hint="person")
    detector.detect_center(_frame(), subject_hint="person", frame_id=0)
    detector.detect_center(_frame(), subject_hint="person", frame_id=15)
    detector.detect_center(_frame(), subject_hint="person", frame_id=30)
    detector.end_segment()

    summary = detector.end_video()

    assert summary["backend"] == "yolo"
    assert summary["scoring"] == "enhanced"
    assert summary["stateful"] is True
    assert summary["segments_started"] == 1
    assert summary["state_resets"] == 1
    assert summary["keyframes_seen"] == 3
    assert summary["detection_frames"] == 1
    assert summary["multi_candidate_frames"] == 1
    assert summary["candidates_total"] == 2
    assert summary["hint_selected_frames"] == 1
    assert summary["track_hold_frames"] == 1
    assert summary["center_fallback_frames"] == 1
    assert summary["selected_class_counts"] == {"person": 2}
    # A held track may either repeat the last confidence or count confidence only
    # on fresh detections; in both cases the aggregate must remain meaningful.
    assert summary["selected_conf_count"] in {1, 2}
    assert summary["selected_conf_mean"] == pytest.approx(0.8)
    assert summary["scoring_counts"] == {"enhanced": 3}


def test_center_detector_lifecycle_summary_remains_compatible():
    detector = CenterSpatialDetector()
    detector.begin_video("v", "unused.mp4", 30.0, 200, 100)
    detector.begin_segment(0, 0, 1)

    cx, cy, info = detector.detect_center(_frame(200, 100))
    summary = detector.end_video()

    assert (cx, cy) == pytest.approx((100.0, 50.0))
    assert info == {"backend": "center", "fallback": "center", "detections": 0}
    assert summary["segments_started"] == 1
    assert summary["keyframes_seen"] == 1
    assert summary["center_fallback_frames"] == 1


def test_exact_car_hint_rejects_much_stronger_truck_as_full_hint_match():
    boxes = [
        FakeBox([430, 420, 570, 560], 2, 0.25),
        FakeBox([50, 80, 950, 850], 7, 0.99),
    ]
    detector, _model = _detector([boxes], stateful=False, person_ball_weight=0.0)

    _cx, _cy, info = detector.detect_center(_frame(), subject_hint="car")

    assert info["class"] == "car"
    assert info["hint_match"] is True


def test_non_finite_class_id_and_reversed_box_are_rejected():
    boxes = [
        FakeBox([100, 100, 300, 500], float("nan"), 0.90),
        FakeBox([900, 100, 100, 500], 0, 0.95),
        FakeBox([400, 200, 600, 700], 0, 0.80),
    ]
    detector, _model = _detector([boxes], stateful=False, person_ball_weight=0.0)

    candidates = detector.detect_candidates(_frame())

    assert len(candidates) == 1
    assert candidates[0].xyxy == pytest.approx((400.0, 200.0, 600.0, 700.0))


def test_track_hold_continues_actual_person_ball_focus_center():
    detected = [
        FakeBox([300, 250, 500, 650], 0, 0.90),
        FakeBox([620, 420, 680, 480], 32, 0.70),
    ]
    detector, _model = _detector(
        [detected, []],
        stateful=True,
        max_missed_keyframes=1,
        person_ball_weight=0.20,
    )

    first = detector.detect_center(_frame(), subject_hint="ball", frame_id=0)
    held = detector.detect_center(_frame(), subject_hint="ball", frame_id=15)

    assert first[:2] == pytest.approx((600.0, 450.0))
    assert held[:2] == pytest.approx(first[:2])
    assert held[2]["fallback"] == "track-hold"


def test_stateless_mode_does_not_maintain_or_report_track_switches():
    frames = [
        [FakeBox([100, 100, 300, 600], 0, 0.90)],
        [FakeBox([700, 100, 900, 600], 2, 0.95)],
    ]
    detector, _model = _detector(frames, stateful=False, person_ball_weight=0.0)
    detector.begin_video("v", "unused.mp4", 30.0, 1000, 1000)
    detector.begin_segment(0, 0, 15)

    _first = detector.detect_center(_frame(), frame_id=0)
    _cx, _cy, second_info = detector.detect_center(_frame(), frame_id=15)
    summary = detector.end_video()

    assert second_info["previous_bbox"] is None
    assert second_info["track_switched"] is False
    assert second_info["track_id"] == 0
    assert summary["track_switches"] == 0


def test_legacy_mode_disables_state_and_uses_immediate_center_fallback():
    detector, _model = _detector([[]], scoring="legacy", stateful=True)

    cx, cy, info = detector.detect_center(_frame(), frame_id=10)

    assert (cx, cy) == pytest.approx((500.0, 500.0))
    assert detector.stateful is False
    assert info["fallback"] == "center"


def test_far_ball_reacquisition_starts_new_track_and_clears_velocity():
    frames = [
        [FakeBox([40, 440, 100, 500], 32, 0.80)],
        [FakeBox([900, 440, 960, 500], 32, 0.82)],
    ]
    detector, _model = _detector(
        frames,
        stateful=True,
        continuity_weight=0.30,
        person_ball_weight=0.0,
    )

    _first = detector.detect_center(_frame(), subject_hint="ball", frame_id=0)
    _cx, _cy, second_info = detector.detect_center(
        _frame(), subject_hint="ball", frame_id=15
    )

    assert second_info["track_switched"] is True
    assert second_info["track_id"] == 2
    assert detector.track.velocity_xy == pytest.approx((0.0, 0.0))
    assert detector.track.focus_velocity_xy == pytest.approx((0.0, 0.0))
