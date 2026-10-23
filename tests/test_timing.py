import math
import pytest

from video_highlight.timing import normalize_interval, frame_bounds


def test_binary_float_boundary_noise_is_accepted_and_audited():
    span, changes = normalize_interval(-1e-16, 13.920000000000002, 13.92)
    assert span == [0, 13.92]
    assert len(changes) == 2


def test_lenient_end_rule_clips_only_a_small_overrun():
    span, changes = normalize_interval(0, 25.5, 25.2, lenient=True)
    assert span == [0, 25.2]
    assert changes[0]["reason"] == "clipped_to_video_end"
    assert changes[0]["difference_sec"] == pytest.approx(0.3)


def test_lenient_end_rule_has_a_hard_limit():
    with pytest.raises(ValueError, match="outside"):
        normalize_interval(0, 25.71, 25.2, lenient=True)


@pytest.mark.parametrize("end", [25.5, 25.20000001])
def test_real_overrun_is_rejected(end):
    with pytest.raises(ValueError, match="outside"):
        normalize_interval(0, end, 25.2)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "0"])
def test_invalid_number_is_rejected(bad):
    with pytest.raises(ValueError):
        normalize_interval(bad, 2, 3)


def test_roundoff_cannot_create_an_empty_highlight():
    with pytest.raises(ValueError, match="positive"):
        normalize_interval(4, 4.000000000000001, 4)


def test_half_open_pts_mapping_does_not_add_or_drop_a_frame_for_float_noise():
    bounds, changes = frame_bounds([0, .04, .1, .14, .23],
                                   math.nextafter(.04, math.inf), math.nextafter(.14, math.inf))
    assert bounds == [1, 3]
    assert len(changes) == 2


def test_actual_time_shift_is_preserved():
    bounds, changes = frame_bounds([0, .04, .1, .14, .23], .04001, .14001)
    assert bounds == [2, 4]
    assert changes == []
