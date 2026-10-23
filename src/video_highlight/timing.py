"""One time contract: finite seconds, 1 ns roundoff tolerance, original PTS."""
from bisect import bisect_left
from decimal import Decimal

TIME_EPSILON_SEC = 1e-9
_EPSILON = Decimal("0.000000001")


def _seconds(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return number


_LENIENT_END_TOLERANCE_SEC = Decimal("0.5")


def normalize_interval(start, end, duration, *, lenient=False,
                       end_tolerance_sec=None):
    """Normalize boundaries, optionally clipping a small video-end overrun."""
    limit = _seconds(duration, "duration")
    if limit <= 0:
        raise ValueError("duration must be positive")
    if end_tolerance_sec is None:
        end_tolerance = _LENIENT_END_TOLERANCE_SEC if lenient else _EPSILON
    else:
        end_tolerance = _seconds(end_tolerance_sec, "end_tolerance_sec")
        if not Decimal(0) <= end_tolerance <= Decimal("1"):
            raise ValueError("end_tolerance_sec must be between 0 and 1")
    values = []
    changes = []
    for field, raw in (("start", start), ("end", end)):
        value = _seconds(raw, field)
        original = value
        if value < 0 and -value <= _EPSILON:
            value = Decimal(0)
        elif value > limit:
            tolerance = end_tolerance if field == "end" else _EPSILON
            if value - limit <= tolerance:
                value = limit
        if value < 0 or value > limit:
            raise ValueError(f"Highlight outside original time range: {start}, {end}; duration={duration}")
        values.append(float(value))
        if value != original:
            overrun = original > limit
            changes.append({"field": field, "raw": raw, "normalized": float(value),
                            "difference_sec": float(abs(original - value)),
                            "reason": (
                                "clipped_to_video_end"
                                if overrun and end_tolerance > _EPSILON
                                else "floating_roundoff"
                            )})
    if values[0] >= values[1]:
        raise ValueError("Highlight must have positive duration after roundoff normalization")
    return values, changes


def frame_bounds(times, start, end):
    """Half-open interval on original PTS; avoid float tails adding/dropping a frame."""
    changes = []
    indices = []
    for field, raw in (("start", start), ("end", end)):
        _seconds(raw, field)
        index = bisect_left(times, raw)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
        nearest = min(candidates, key=lambda i: abs(float(times[i]) - raw)) if candidates else None
        value = raw
        if nearest is not None:
            point = float(times[nearest])
            delta = abs(Decimal(str(point)) - Decimal(str(raw)))
            if Decimal(0) < delta <= _EPSILON:
                value = point
                changes.append({"field": field, "raw": raw, "normalized": point,
                                "difference_sec": float(delta), "reason": "original_pts_roundoff"})
        indices.append(bisect_left(times, value))
    return indices, changes
