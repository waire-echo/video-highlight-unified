import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from video_highlight.geometry import compute_crop_size


def test_vertical_crop_rounding_never_exceeds_source_height():
    crop_w, crop_h = compute_crop_size(1920, 1080, 9, 16)

    evaluator_height = int(round(crop_w * 16 / 9.0))
    assert (crop_w, crop_h) == (607, 1079)
    assert evaluator_height == crop_h
    assert evaluator_height <= 1080


def test_crop_height_matches_evaluator_formula_for_common_ratios():
    cases = [
        (720, 1280, 16, 9),
        (1280, 720, 9, 16),
        (1920, 1080, 16, 9),
        (1080, 1920, 9, 16),
    ]
    for width, height, target_w, target_h in cases:
        crop_w, crop_h = compute_crop_size(width, height, target_w, target_h)
        assert crop_w <= width
        assert crop_h <= height
        assert crop_h == int(round(crop_w * target_h / float(target_w)))


def test_crop_uses_largest_width_allowed_by_evaluator_rounding():
    crop_w, crop_h = compute_crop_size(4, 2, 16, 9)

    assert (crop_w, crop_h) == (4, 2)


def test_vertical_non_integral_crop_stays_inside_frame():
    crop_w, crop_h = compute_crop_size(640, 360, 9, 16)

    assert (crop_w, crop_h) == (202, 359)
    assert round((crop_w + 1) * 16 / 9.0) > 360
