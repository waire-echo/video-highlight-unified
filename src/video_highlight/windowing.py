"""Time-window utilities for the experimental window-score front end.

The rest of the project works with seconds, while the competition output is
frame based.  This module intentionally stays in the seconds domain.  It
provides deterministic, overlap-aware windows and a small amount of video
metadata probing without importing a VLM or changing the legacy temporal
detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class WindowSpec:
    """A single temporal query sent to a video model.

    ``start_sec`` is inclusive and ``end_sec`` is exclusive by convention.
    The convention is only used for bookkeeping; the underlying decoder may
    sample the last frame at the end boundary.  ``stage`` is ``coarse`` or
    ``fine`` but is deliberately not interpreted by this module.
    """

    window_id: int
    start_sec: float
    end_sec: float
    sample_fps: float
    stage: str = "coarse"
    parent_id: Optional[int] = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable representation used in raw logs."""

        result = {
            "window_id": self.window_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "sample_fps": self.sample_fps,
            "stage": self.stage,
        }
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        return result


def _positive_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


def iter_windows(
    duration_sec: float,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    sample_fps: float = 2.0,
    *,
    stage: str = "coarse",
    start_window_id: int = 0,
    parent_id: Optional[int] = None,
) -> Iterator[WindowSpec]:
    """Yield deterministic overlapping windows for a video duration.

    Full-size windows are generated at ``step_sec``.  If the final full-size
    window would not end at the video boundary, one *tail* window ending at
    exactly ``duration_sec`` is appended.  This avoids a stream of tiny,
    duplicate windows at the tail while still covering every frame.  Videos
    shorter than ``window_sec`` produce one full-video window.

    Floating point values are rounded to six decimal places before being
    emitted.  That keeps JSONL diagnostics stable and prevents values such as
    ``0.9999999997`` from creating accidental gaps during later fusion.
    """

    try:
        duration = float(duration_sec)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_sec must be a finite non-negative number") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration_sec must be a finite non-negative number")
    width = _positive_float(window_sec, "window_sec")
    step = _positive_float(step_sec, "step_sec")
    fps = _positive_float(sample_fps, "sample_fps")
    stage_name = str(stage or "coarse")
    next_id = int(start_window_id)

    if duration <= 1e-9:
        # There is no decodable interval, so do not manufacture a zero-length
        # model request.  Callers can use the empty list as an unreadable
        # video signal and apply their normal fallback.
        return

    if duration <= width + 1e-9:
        yield WindowSpec(
            next_id,
            0.0,
            round(duration, 6),
            fps,
            stage_name,
            parent_id,
        )
        return

    last_full_start = duration - width
    start = 0.0
    emitted_last = False
    while start <= last_full_start + 1e-9:
        start_value = round(max(0.0, start), 6)
        end_value = round(min(duration, start + width), 6)
        if end_value > start_value + 1e-9:
            yield WindowSpec(next_id, start_value, end_value, fps, stage_name, parent_id)
            next_id += 1
            emitted_last = abs(start - last_full_start) <= 1e-9
        start += step

    # A step which does not divide the duration leaves a final uncovered tail.
    # Reuse the last full-width interval when it already ends at the boundary.
    if not emitted_last:
        tail_start = round(max(0.0, last_full_start), 6)
        tail_end = round(duration, 6)
        if tail_end > tail_start + 1e-9:
            yield WindowSpec(next_id, tail_start, tail_end, fps, stage_name, parent_id)


def generate_windows(
    duration_sec: float,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    sample_fps: float = 2.0,
    *,
    stage: str = "coarse",
    start_window_id: int = 0,
    parent_id: Optional[int] = None,
) -> List[WindowSpec]:
    """Materialise :func:`iter_windows` as a list."""

    return list(
        iter_windows(
            duration_sec,
            window_sec,
            step_sec,
            sample_fps,
            stage=stage,
            start_window_id=start_window_id,
            parent_id=parent_id,
        )
    )


def probe_video(video_path: str) -> Tuple[float, float, int, int, int]:
    """Return ``(duration, fps, frame_count, width, height)`` using OpenCV.

    Importing OpenCV lazily keeps window-generation tests usable in a minimal
    Python environment.  A zero duration/fps tuple means the file could not
    be opened or had invalid metadata; callers should use their own fallback.
    """

    try:
        import cv2
    except ImportError:
        return 0.0, 0.0, 0, 0, 0

    path = str(Path(video_path).expanduser())
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0.0, 0.0, 0, 0, 0
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        return 0.0, max(0.0, fps), frame_count, width, height
    return frame_count / fps, fps, frame_count, width, height


def windows_from_video(
    video_path: str,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    sample_fps: float = 2.0,
    *,
    stage: str = "coarse",
    start_window_id: int = 0,
) -> Tuple[List[WindowSpec], float]:
    """Probe a video and return ``(windows, duration_sec)``."""

    duration, _fps, _n, _w, _h = probe_video(video_path)
    if duration <= 0:
        return [], duration
    return (
        generate_windows(
            duration,
            window_sec,
            step_sec,
            sample_fps,
            stage=stage,
            start_window_id=start_window_id,
        ),
        duration,
    )


def validate_window(window: WindowSpec, duration_sec: Optional[float] = None) -> WindowSpec:
    """Validate and normalise a window supplied by an external caller."""

    if not isinstance(window, WindowSpec):
        raise TypeError("window must be a WindowSpec")
    start = float(window.start_sec)
    end = float(window.end_sec)
    fps = _positive_float(window.sample_fps, "sample_fps")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("window must have finite 0 <= start_sec < end_sec")
    if duration_sec is not None:
        duration = float(duration_sec)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_sec must be finite and non-negative")
        if end > duration + 1e-6:
            raise ValueError("window end_sec exceeds video duration")
    return WindowSpec(
        int(window.window_id),
        round(start, 6),
        round(end, 6),
        fps,
        str(window.stage or "coarse"),
        None if window.parent_id is None else int(window.parent_id),
    )


__all__ = [
    "WindowSpec",
    "generate_windows",
    "iter_windows",
    "probe_video",
    "validate_window",
    "windows_from_video",
]
