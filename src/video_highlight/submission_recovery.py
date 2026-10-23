"""Auditable local recovery for submission inference; never consult labels."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import re
import signal
import sys


POLICY = "submission_intersection/v1"
TIME_RECOVERY_POLICY = "intersection"
TIME_FAILURE_POLICY = "drop_windows"
SPATIAL_FAILURE_POLICY = "center"
RUNTIME_POLICIES = {
    "recovery_policy": TIME_RECOVERY_POLICY,
    "unresolved_policy": TIME_FAILURE_POLICY,
    "spatial_failure_policy": SPATIAL_FAILURE_POLICY,
}


def runtime_policy(name: str, requested=None) -> str:
    """Legacy options cannot disable recovery in a submission run."""
    active = RUNTIME_POLICIES[name]
    if requested is not None and requested != active:
        print(json.dumps({"warning": "strict_runtime_disabled", "option": name,
                          "ignored_value": requested, "active_value": active}),
              file=sys.stderr, flush=True)
    return active


def recover_intervals(raw: str, duration: float):
    """Keep independently decoded entries, intersect local seconds, never guess.

    Unlike the historical diagnostic parser, this does not extract numbers
    from arbitrary prose. A truncated answer needs a recognizable highlights
    array and fully JSON-decoded entries. An invalid nonempty answer is not a
    model judgment of no highlights.
    """
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid window duration")
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("highlights"), list):
            candidates.append(value["highlights"])
    if len(candidates) > 1:
        raise ValueError("Ambiguous multiple highlights answers")
    partial = False
    if candidates:
        rows = candidates[0]
    else:
        # Do not cross arbitrary text or invent a missing endpoint/brace.
        starts = list(re.finditer(r'\{\s*"highlights"\s*:\s*\[', raw))
        if len(starts) != 1:
            raise ValueError("No unambiguous highlights array")
        position = starts[0].end()
        rows = []
        while position < len(raw):
            position += len(raw[position:]) - len(raw[position:].lstrip())
            try:
                value, position = decoder.raw_decode(raw, position)
            except json.JSONDecodeError:
                break
            rows.append(value)
            position += len(raw[position:]) - len(raw[position:].lstrip())
            if position >= len(raw) or raw[position] != ",":
                break
            position += 1
        partial = True
        if not rows:
            raise ValueError("Truncated answer has no complete entries")
    accepted, changes, discarded = [], [], []
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict) or not {"start", "end"}.issubset(row):
                raise ValueError("missing start/end")
            if any(isinstance(row[key], bool) or not isinstance(row[key], (int, float, str))
                   for key in ("start", "end")):
                raise ValueError("nonnumeric endpoint")
            start, end = float(row["start"]), float(row["end"])
            if not all(math.isfinite(x) for x in (start, end)) or start >= end:
                raise ValueError("nonfinite or nonpositive interval")
            left, right = max(0.0, start), min(duration, end)
            if left >= right:
                raise ValueError("interval entirely outside the window")
            accepted.append([left, right])
            if (left, right) != (start, end):
                changes.append({"index": index, "raw": [start, end], "result": [left, right],
                                "clipped_seconds": (left - start) + (end - right)})
        except (ValueError, TypeError, OverflowError) as error:
            discarded.append({"index": index, "row": row, "reason": str(error)})
    if rows and not accepted:
        raise ValueError("All entries invalid; not a no-highlights judgment")
    merged = []
    for start, end in sorted(accepted):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    audit = {"policy": POLICY, "status": "partial" if partial or discarded else "repaired",
             "time_changes": changes, "discarded_entries": discarded,
             "truncated": partial, "explicit_no_highlights": not rows and not partial}
    return {"highlights": [{"start": a, "end": b} for a, b in merged]}, merged, audit


def center_box(size, ratio):
    """Largest centered integer crop with the requested aspect ratio."""
    width, height = size
    rw, rh = ratio
    if any(type(x) is not int or x <= 0 for x in (width, height, rw, rh)):
        raise ValueError("Invalid source dimensions or target ratio")
    low, high, crop_w = 1, width, 0
    while low <= high:
        candidate = (low + high) // 2
        crop_h = round(candidate * rh / rw)
        if crop_h <= height:
            crop_w, low = candidate, candidate + 1
        else:
            high = candidate - 1
    crop_h = round(crop_w * rh / rw)
    if crop_w < 1 or crop_h < 1:
        raise ValueError("Target ratio cannot fit an integer crop in the source")
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    return [left, top, crop_w]


def selected_frames(first):
    """Temporal selection is frozen here; spatial recovery cannot add frames."""
    total = first["frame_count"]
    previous = -1
    for span in first["segments_frame"]:
        start, end = span
        if any(type(x) is not int for x in span) or not 0 <= start <= end < total or start <= previous:
            raise ValueError("Invalid or overlapping temporal frame spans")
        yield from range(start, end + 1)
        previous = end


def repair_spatial_row(second, first, size):
    """Retain valid boxes verbatim; fill only missing/bad selected frames."""
    baseline = center_box(size, first["targetRatioWH"])
    width, height = size
    ratio = first["targetRatioWH"][0] / first["targetRatioWH"][1]
    by_frame, duplicates = {}, set()
    items = second.get("predictions", []) if isinstance(second, dict) else []
    for prediction in items if isinstance(items, list) else []:
        if not isinstance(prediction, dict) or type(prediction.get("frame")) is not int:
            continue
        frame = prediction["frame"]
        if frame in by_frame:
            duplicates.add(frame)
        by_frame[frame] = prediction.get("bboxes")
    predictions, repaired = [], []
    for frame in selected_frames(first):
        box = by_frame.get(frame)
        valid = (frame not in duplicates and isinstance(box, list) and len(box) == 3
                 and all(type(x) is int for x in box))
        if valid:
            x, y, crop_w = box
            crop_h = round(crop_w / ratio)
            valid = (0 <= x and 0 <= y and crop_w > 0 and crop_h > 0
                     and x + crop_w <= width and y + crop_h <= height)
        if not valid:
            box = baseline.copy()
            repaired.append(frame)
        predictions.append({"frame": frame, "bboxes": box})
    return {"video_id": str(first["video_id"]), "targetRatioWH": first["targetRatioWH"],
            "predictions": predictions}, repaired


@contextmanager
def inference_deadline(seconds):
    """POSIX request deadline; a parent wall-clock watchdog remains mandatory."""
    if not seconds or not hasattr(signal, "setitimer"):
        yield
        return
    def expired(signum, frame):
        raise TimeoutError(f"Window inference exceeded {seconds:g} seconds")
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
