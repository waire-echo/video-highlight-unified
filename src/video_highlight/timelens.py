"""TimeLens temporal-grounding adapter.

The repository does not bundle TimeLens weights.  This adapter defines the
provider contract and strict output handling so the model can be connected in
an isolated environment later without changing YOLO or crop code.
"""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Sequence

from .action_description import HighlightAction
from .temporal import TemporalResult, normalize_temporal_segments


GroundingProvider = Callable[[str, str], str]


def parse_timelens_response(text: str, duration_sec: float):
    raw = str(text or "").strip()
    segments = []
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        for item in payload["segments"]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                segments.append((item[0], item[1]))
    if not segments:
        pattern = re.compile(
            r"(?:event\s+happens\s+in\s+)?"
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:-|to)\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*seconds?",
            re.I,
        )
        segments = [(match.group(1), match.group(2)) for match in pattern.finditer(raw)]
    normalized = normalize_temporal_segments(
        segments,
        duration_sec=duration_sec,
        min_segment_sec=0.25,
        padding_sec=0.0,
        max_segments=3,
    )
    return normalized


class TimeLensAdapter:
    def __init__(self, provider: Optional[GroundingProvider] = None):
        self.provider = provider

    def ground(
        self,
        video_path: str,
        actions: Sequence[HighlightAction],
        duration_sec: float,
    ) -> TemporalResult:
        if self.provider is None:
            return TemporalResult(
                [],
                source="timelens_unavailable",
                valid=False,
                errors=["TimeLens provider is not configured"],
                duration_sec=duration_sec,
            )
        segments = []
        raw_responses: List[str] = []
        errors: List[str] = []
        for action in actions:
            try:
                raw = str(self.provider(video_path, action.description))
            except Exception as exc:
                errors.append(
                    f"TimeLens query failed for {action.description!r}: {type(exc).__name__}: {exc}"
                )
                continue
            raw_responses.append(raw)
            parsed = parse_timelens_response(raw, duration_sec)
            if parsed.valid:
                segments.extend(parsed.segments_sec)
            else:
                errors.extend(parsed.errors)
        normalized = normalize_temporal_segments(
            segments,
            duration_sec=duration_sec,
            min_segment_sec=0.25,
            padding_sec=0.0,
            max_segments=3,
        )
        return TemporalResult(
            list(normalized.segments_sec),
            raw=json.dumps(raw_responses, ensure_ascii=False),
            source="timelens",
            valid=normalized.valid,
            errors=errors + list(normalized.errors),
            attempts=len(raw_responses),
            duration_sec=duration_sec,
            segments_before_normalization=list(segments),
            segments_after_normalization=list(normalized.segments_sec),
            corrections=list(normalized.corrections),
        )
