"""Orchestration for action description -> TimeLens -> signal fallback."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from .action_description import QwenActionDescriber
from .io_utils import video_meta
from .temporal import TemporalResult
from .temporal_fusion import TemporalFusionPolicy, fuse_temporal_results
from .timelens import TimeLensAdapter


class TemporalFrontEnd:
    """Front-half pipeline that returns time segments only.

    Spatial detection, tracking, interpolation, and crop generation are
    intentionally outside this class and remain owned by the YOLO stage.
    """

    def __init__(
        self,
        action_describer: QwenActionDescriber,
        time_grounder: TimeLensAdapter,
        signal_detector=None,
        fusion_policy: TemporalFusionPolicy = TemporalFusionPolicy(),
    ):
        self.action_describer = action_describer
        self.time_grounder = time_grounder
        self.signal_detector = signal_detector
        self.fusion_policy = fusion_policy

    def detect(self, video_path: str) -> TemporalResult:
        n_frames, fps, _width, _height = video_meta(video_path)
        duration_sec = n_frames / fps if n_frames > 0 and fps > 0 else 0.0
        if duration_sec <= 0:
            return TemporalResult([], source="temporal_frontend", valid=False, errors=["unreadable video"])

        action_result = self.action_describer.describe(video_path)
        if action_result.valid:
            grounded = self.time_grounder.ground(
                video_path, action_result.actions, duration_sec
            )
            grounded.raw = json.dumps(
                {
                    "action_description": {
                        "actions": [asdict(action) for action in action_result.actions],
                        "raw": action_result.raw,
                    },
                    "timelens_raw": grounded.raw,
                },
                ensure_ascii=False,
            )
            signals = (
                self.signal_detector.detect(video_path)
                if self.signal_detector is not None
                else TemporalResult([], source="signals_unavailable", valid=False)
            )
            if grounded.valid and grounded.segments_sec:
                return fuse_temporal_results(
                    grounded, signals, duration_sec, self.fusion_policy
                )
        else:
            signals = (
                self.signal_detector.detect(video_path)
                if self.signal_detector is not None
                else TemporalResult([], source="signals_unavailable", valid=False)
            )

        errors = list(action_result.errors)
        if action_result.valid:
            errors.append("TimeLens did not return a valid temporal result")
        if signals.valid and signals.segments_sec:
            signals.source = f"{signals.source or 'signal'}_fallback"
            signals.fallback_used = signals.source
            signals.errors = errors + list(signals.errors)
            return signals
        return TemporalResult(
            [],
            raw=action_result.raw,
            source="temporal_frontend_failed",
            valid=False,
            errors=errors or ["temporal front end returned no result"],
            duration_sec=duration_sec,
        )
