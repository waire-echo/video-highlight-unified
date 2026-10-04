"""Global-event-conditioned Qwen temporal frontend.

The original structured frontend rates every short window in isolation.  This
experimental route first asks the same local Qwen model what the complete video
is about, then verifies those *video-specific* events in every short window.
Traditional signals may adjust boundaries but can never create an event seed.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .io_utils import video_meta
from .score_curve import build_score_curve, find_local_peaks, smooth_score_curve
from .temporal import TemporalResult, temporal_coverage_ratio
from .temporal_signals import (
    TRADITIONAL_SIGNAL_KEYS,
    TemporalSignalExtractor,
    apply_signal_calibration,
    fit_signal_calibration,
)
from .windowing import WindowSpec, generate_windows


Segment = Tuple[float, float]
SignalProvider = Callable[[str, WindowSpec], Mapping[str, Any]]

_TIME_PATTERN = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b|\d+(?:\.\d+)?\s*秒|timestamp|timecode)",
    re.I,
)


def _json_value(text: str) -> Any:
    payload = str(text or "").strip()
    if not payload:
        raise ValueError("response is empty")
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", payload, re.I | re.S)
    if payload.startswith("```") or payload.endswith("```"):
        if not fence:
            raise ValueError("incomplete markdown JSON fence")
        payload = fence.group(1).strip()
    duplicates: List[str] = []

    def hook(pairs):
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(str(key))
            result[str(key)] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=hook)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        # The 3B model occasionally emits several individually valid JSON
        # arrays separated by commas but forgets the outer array.  Wrapping
        # those complete JSON values is deterministic and does not attempt to
        # recover prose or truncated output.
        try:
            value = json.loads(f"[{payload}]", object_pairs_hook=hook)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError(f"invalid complete JSON: {exc}") from exc
    if duplicates:
        raise ValueError("duplicate JSON keys: " + ", ".join(sorted(set(duplicates))))
    return value


def _json_object(text: str) -> Dict[str, Any]:
    value = _json_value(text)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _clean_text(value: Any, key: str, *, forbid_time: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    text = " ".join(value.split())
    if forbid_time and _TIME_PATTERN.search(text):
        raise ValueError(f"{key} must not contain timestamps")
    return text


def parse_global_event_response(text: str, *, max_events: int = 3) -> List[Dict[str, Any]]:
    """Parse the complete-video event contract."""

    payload = _json_value(text)
    if isinstance(payload, dict):
        if set(payload) != {"events"}:
            raise ValueError("global event JSON must contain only events")
        events = payload["events"]
    else:
        # Qwen2.5-VL-3B reliably follows the event item contract but often
        # omits a redundant {"events": ...} wrapper.  Accept only the complete
        # top-level list; prose and partial JSON remain invalid.
        events = payload
    if not isinstance(events, list) or not 1 <= len(events) <= max(1, int(max_events)):
        raise ValueError(f"events must contain 1..{max(1, int(max_events))} items")
    output: List[Dict[str, Any]] = []
    seen = set()
    required = {"event_id", "subject", "action", "completion_cue"}
    for index, item in enumerate(events):
        overview_time_hint: Optional[List[float]] = None
        if isinstance(item, list) and len(item) == 3:
            item = {
                "event_id": f"E{index + 1}",
                "subject": item[0],
                "action": item[1],
                "completion_cue": item[2],
            }
        if not isinstance(item, dict):
            raise ValueError(f"event {index} schema mismatch")
        # Qwen2.5-VL-3B sometimes falls back to its familiar temporal-grounding
        # contract even when asked for compact event descriptions.  Preserve the
        # model-proposed range as a weak temporal prior, but still verify every
        # event against all short windows; the hint never directly selects output.
        if set(item) == {"start_time", "end_time", "description"}:
            description = _clean_text(
                item["description"], "description", forbid_time=True
            )
            try:
                start_hint = float(item["start_time"])
                end_hint = float(item["end_time"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"event {index} has invalid time hints") from exc
            if not math.isfinite(start_hint) or not math.isfinite(end_hint) or end_hint <= start_hint:
                raise ValueError(f"event {index} has invalid time hints")
            overview_time_hint = [max(0.0, start_hint), max(0.0, end_hint)]
            item = {
                "event_id": f"E{index + 1}",
                "subject": _canonical_subject(description) or "unknown",
                "action": description,
                "completion_cue": "the described action visibly reaches its result",
            }
        if set(item) == required - {"event_id"}:
            item = dict(item, event_id=f"E{index + 1}")
        if set(item) != required:
            raise ValueError(f"event {index} schema mismatch")
        event_id = _clean_text(item["event_id"], "event_id")
        if event_id in seen:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen.add(event_id)
        parsed_event: Dict[str, Any] = {
            "event_id": event_id,
            "subject": _clean_text(item["subject"], "subject", forbid_time=True),
            "action": _clean_text(item["action"], "action", forbid_time=True),
            "completion_cue": _clean_text(
                item["completion_cue"], "completion_cue", forbid_time=True
            ),
        }
        if overview_time_hint is not None:
            parsed_event["overview_time_hint_sec"] = overview_time_hint
        output.append(parsed_event)
    return output


def parse_event_ranking_response(
    text: str, candidate_ids: Sequence[str], *, max_events: int = 3
) -> List[str]:
    payload = _json_object(text)
    if set(payload) != {"selected_candidate_ids"}:
        raise ValueError("ranking JSON must contain only selected_candidate_ids")
    values = payload["selected_candidate_ids"]
    if not isinstance(values, list) or not 1 <= len(values) <= max(1, int(max_events)):
        raise ValueError("selected_candidate_ids has an invalid length")
    allowed = set(candidate_ids)
    output = []
    for value in values:
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"unknown candidate id: {value!r}")
        if value in output:
            raise ValueError(f"duplicate candidate id: {value}")
        output.append(value)
    return output


def parse_event_window_response(text: str, event_count: int) -> Dict[str, Any]:
    """Parse one event-conditioned short-window response."""

    payload = _json_value(text)
    count = max(1, int(event_count))
    # Observed Qwen2.5-VL-3B fallback shape:
    # [["E1", [2, "visible evidence"], "observed action"], ...]
    # It still provides a bounded match score and event-specific evidence.  A
    # separate completion score is unavailable, so conservatively reuse match
    # strength and record the compatibility schema for diagnostics.
    if (
        isinstance(payload, list)
        and len(payload) == count
        and all(isinstance(item, list) and len(item) == 3 for item in payload)
    ):
        matches: List[int] = []
        evidence_parts: List[str] = []
        subjects: List[str] = []
        event_ids: List[str] = []
        for index, item in enumerate(payload):
            event_id, scored_evidence, observed_action = item
            if not isinstance(event_id, str) or not event_id.strip():
                raise ValueError(f"event-window row {index} has invalid event id")
            if (
                not isinstance(scored_evidence, list)
                or len(scored_evidence) != 2
                or isinstance(scored_evidence[0], bool)
                or not isinstance(scored_evidence[0], int)
                or not 0 <= scored_evidence[0] <= 3
            ):
                raise ValueError(f"event-window row {index} has invalid score")
            evidence = _clean_text(scored_evidence[1], "evidence", forbid_time=True)
            action = _clean_text(observed_action, "observed_action", forbid_time=True)
            matches.append(scored_evidence[0])
            event_ids.append(event_id.strip())
            evidence_parts.append(f"{event_id.strip()}: {evidence}; {action}")
            subject = _canonical_subject(action) or _canonical_subject(evidence)
            if subject:
                subjects.append(subject)
        majority = Counter(subjects).most_common(1)[0][0] if subjects else "unknown"
        return {
            "matches": matches,
            "completion": list(matches),
            "visible_subject": majority,
            "evidence": " | ".join(evidence_parts),
            "event_ids": event_ids,
            "compatibility_schema": "event_rows_without_completion",
        }
    # Another common shape is one compact score record per event, with the
    # outer list omitted by generation: [match, completion, subject, evidence].
    if (
        isinstance(payload, list)
        and len(payload) == count
        and all(isinstance(item, list) and len(item) == 4 for item in payload)
    ):
        matches: List[int] = []
        completions: List[int] = []
        subjects: List[str] = []
        evidence_parts: List[str] = []
        for index, item in enumerate(payload):
            match, completion, subject_value, evidence_value = item
            if (
                isinstance(match, bool)
                or not isinstance(match, int)
                or not 0 <= match <= 3
                or isinstance(completion, bool)
                or not isinstance(completion, int)
                or not 0 <= completion <= 3
            ):
                raise ValueError(f"event-window row {index} has invalid scores")
            subject_text = _clean_text(
                subject_value, "visible_subject", forbid_time=True
            )
            evidence_text = _clean_text(evidence_value, "evidence", forbid_time=True)
            matches.append(match)
            completions.append(completion)
            subjects.append(_canonical_subject(subject_text) or subject_text.lower())
            evidence_parts.append(f"E{index + 1}: {evidence_text}")
        majority = Counter(subjects).most_common(1)[0][0] if subjects else "unknown"
        return {
            "matches": matches,
            "completion": completions,
            "visible_subject": majority,
            "evidence": " | ".join(evidence_parts),
            "compatibility_schema": "per_event_compact_rows",
        }
    if isinstance(payload, list) and len(payload) == 4:
        payload = {
            "matches": payload[0],
            "completion": payload[1],
            "visible_subject": payload[2],
            "evidence": payload[3],
        }
    if not isinstance(payload, dict):
        raise ValueError("event-window JSON schema mismatch")
    allowed = {"matches", "completion", "completion_cue", "visible_subject", "evidence"}
    if not set(payload) <= allowed or not {"matches", "visible_subject", "evidence"} <= set(payload):
        raise ValueError("event-window JSON schema mismatch")
    if "completion" not in payload and "completion_cue" not in payload:
        raise ValueError("event-window JSON is missing completion")
    parsed: Dict[str, Any] = {}
    for key in ("matches", "completion"):
        source_key = key if key in payload else "completion_cue"
        values = payload[source_key]
        if count == 1 and isinstance(values, int) and not isinstance(values, bool):
            values = [values]
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"{key} must contain exactly {count} scores")
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3 for value in values):
            raise ValueError(f"{key} values must be integer scores in [0, 3]")
        parsed[key] = list(values)
    parsed["visible_subject"] = _clean_text(
        payload["visible_subject"], "visible_subject", forbid_time=True
    )
    evidence = payload["evidence"]
    if isinstance(evidence, list):
        parts = []
        for item in evidence:
            if isinstance(item, dict):
                parts.extend(str(value) for value in item.values() if isinstance(value, str))
            elif isinstance(item, str):
                parts.append(item)
        evidence = " ".join(parts)
    parsed["evidence"] = _clean_text(evidence, "evidence", forbid_time=True)
    return parsed


def build_global_event_prompt(*, max_events: int = 3) -> str:
    from .prompts import call_builder
    return call_builder("qwen/global-event-v1", "build_global_event_prompt", max_events=max_events)


def build_event_ranking_prompt(candidates: Sequence[Mapping[str, str]], *, max_events: int = 3) -> str:
    from .prompts import call_builder
    return call_builder("qwen/global-event-v1", "build_event_ranking_prompt", candidates, max_events=max_events)


def build_event_window_prompt(events: Sequence[Mapping[str, str]]) -> str:
    from .prompts import call_builder
    return call_builder("qwen/global-event-v1", "build_event_window_prompt", events)


def _canonical_subject(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("猫", "小猫", "猫咪")):
        return "cat"
    if any(token in text for token in ("狗", "小狗", "狗狗")):
        return "dog"
    if any(token in text for token in ("人", "男人", "女人", "男孩", "女孩", "运动员")):
        return "person"
    tokens = set(re.findall(r"[a-z]+", text))
    if tokens & {"dog", "dogs", "puppy", "puppies", "canine"}:
        return "dog"
    if tokens & {"cat", "cats", "kitten", "kittens", "feline"}:
        return "cat"
    if tokens & {
        "person", "people", "human", "man", "woman", "boy", "girl", "child",
        "children", "athlete", "player", "rider", "cyclist",
    }:
        return "person"
    if "sports ball" in text or tokens & {"ball", "football", "basketball", "soccer"}:
        return "ball"
    if tokens & {"car", "truck", "bus", "vehicle", "bicycle", "bike", "motorcycle"}:
        return "vehicle"
    if tokens & {"animal", "horse", "bird", "bear", "pig"}:
        return "animal"
    return None


def _token_key(event: Mapping[str, str]) -> set:
    return set(re.findall(r"[a-z]+", f"{event.get('subject', '')} {event.get('action', '')}".lower()))


def _deduplicate_candidates(candidates: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for candidate in candidates:
        tokens = _token_key(candidate)
        duplicate = False
        for existing in output:
            other = _token_key(existing)
            union = tokens | other
            if union and len(tokens & other) / len(union) >= 0.65:
                duplicate = True
                break
        if not duplicate:
            output.append(candidate)
    return output


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _interval_iou(left: Segment, right: Segment) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(0.0, left[1] - left[0]) + max(0.0, right[1] - right[0]) - overlap
    return overlap / union if union > 0 else 0.0


class GlobalEventFrontend:
    """Find video-level events first, then localise them in short windows."""

    def __init__(
        self,
        scorer: Any,
        *,
        fallback_detector: Any = None,
        window_sec: float = 2.0,
        stride_sec: float = 1.0,
        sample_fps: float = 2.0,
        grid_sec: float = 0.5,
        smoothing_points: int = 3,
        max_events: int = 3,
        long_video_sec: float = 60.0,
        overview_chunk_sec: float = 30.0,
        overview_max_frames: int = 64,
        seed_threshold: float = 0.55,
        growth_threshold: float = 0.40,
        valley_threshold: float = 0.35,
        min_prominence: float = 0.10,
        padding_sec: float = 0.50,
        merge_gap_sec: float = 0.25,
        max_coverage_ratio: float = 0.75,
        ffmpeg_bin: str = "ffmpeg",
        audio_sample_rate: int = 8000,
        signal_provider: Optional[SignalProvider] = None,
    ):
        self.scorer = scorer
        self.fallback_detector = fallback_detector
        self.window_sec = max(0.5, float(window_sec))
        self.stride_sec = max(0.25, float(stride_sec))
        self.sample_fps = max(0.1, float(sample_fps))
        self.grid_sec = max(0.1, float(grid_sec))
        self.smoothing_points = max(1, int(smoothing_points))
        self.max_events = max(1, int(max_events))
        self.long_video_sec = max(1.0, float(long_video_sec))
        self.overview_chunk_sec = max(5.0, float(overview_chunk_sec))
        self.overview_max_frames = max(8, int(overview_max_frames))
        self.seed_threshold = max(0.0, min(1.0, float(seed_threshold)))
        self.growth_threshold = max(0.0, min(1.0, float(growth_threshold)))
        self.valley_threshold = max(0.0, min(1.0, float(valley_threshold)))
        self.min_prominence = max(0.0, float(min_prominence))
        self.padding_sec = max(0.0, float(padding_sec))
        self.merge_gap_sec = max(0.0, float(merge_gap_sec))
        self.max_coverage_ratio = max(0.05, min(1.0, float(max_coverage_ratio)))
        self.ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self.audio_sample_rate = max(1000, int(audio_sample_rate))
        self.signal_provider = signal_provider

    def _overview_fps(self, length: float) -> float:
        return max(0.1, min(2.0, self.overview_max_frames / max(0.1, float(length))))

    def _generate_parsed(
        self,
        video_path: str,
        start: float,
        end: float,
        fps: float,
        prompt: str,
        parser: Callable[[str], Any],
    ) -> Tuple[Any, List[str], List[str]]:
        raw_rows: List[str] = []
        errors: List[str] = []
        for attempt in range(2):
            current = prompt
            if attempt:
                current += " Your previous response was invalid. Return only the complete required JSON."
            try:
                raw = self.scorer.generate_range(video_path, start, end, fps, current)
            except Exception as exc:
                raw = f"<generation-error>{type(exc).__name__}: {exc}</generation-error>"
                raw_rows.append(raw)
                errors.append(f"generation failed: {type(exc).__name__}: {exc}")
                break
            raw_rows.append(str(raw))
            try:
                return parser(str(raw)), raw_rows, errors
            except Exception as exc:
                errors.append(f"parse failed: {type(exc).__name__}: {exc}")
        return None, raw_rows, errors

    def _global_events(self, video_path: str, duration: float) -> Dict[str, Any]:
        overview_prompt = build_global_event_prompt(max_events=self.max_events)
        ranges: List[Tuple[str, float, float]] = [("global", 0.0, duration)]
        if duration > self.long_video_sec:
            cursor = 0.0
            chunk_index = 0
            while cursor < duration - 1e-9:
                end = min(duration, cursor + self.overview_chunk_sec)
                ranges.append((f"chunk_{chunk_index}", cursor, end))
                cursor = end
                chunk_index += 1

        candidates: List[Dict[str, str]] = []
        diagnostics: List[Dict[str, Any]] = []
        raw_responses: List[str] = []
        errors: List[str] = []
        for source, start, end in ranges:
            parsed, raw, row_errors = self._generate_parsed(
                video_path,
                start,
                end,
                self._overview_fps(end - start),
                overview_prompt,
                lambda value: parse_global_event_response(value, max_events=self.max_events),
            )
            raw_responses.extend(raw)
            errors.extend(f"{source}: {error}" for error in row_errors)
            diagnostics.append(
                {
                    "source": source,
                    "start_sec": start,
                    "end_sec": end,
                    "valid": parsed is not None,
                    "events": parsed or [],
                    "raw_responses": raw,
                    "errors": row_errors,
                }
            )
            for event in parsed or []:
                copy = dict(event)
                copy["candidate_id"] = f"C{len(candidates) + 1}"
                copy["overview_source"] = source
                candidates.append(copy)

        candidates = _deduplicate_candidates(candidates)
        # Deduplication may remove a middle candidate.  Renumber afterwards so
        # the IDs exposed to the ranking prompt are contiguous and cannot
        # reference a removed event.
        for index, candidate in enumerate(candidates):
            candidate["candidate_id"] = f"C{index + 1}"
        selected_ids: List[str] = []
        ranking_raw: List[str] = []
        ranking_errors: List[str] = []
        if len(candidates) > self.max_events:
            prompt = build_event_ranking_prompt(candidates, max_events=self.max_events)
            ranked, ranking_raw, ranking_errors = self._generate_parsed(
                video_path,
                0.0,
                duration,
                self._overview_fps(duration),
                prompt,
                lambda value: parse_event_ranking_response(
                    value,
                    [item["candidate_id"] for item in candidates],
                    max_events=self.max_events,
                ),
            )
            selected_ids = list(ranked or [])
            raw_responses.extend(ranking_raw)
            errors.extend(f"ranking: {error}" for error in ranking_errors)
        if not selected_ids:
            selected_ids = [item["candidate_id"] for item in candidates[: self.max_events]]
        by_id = {item["candidate_id"]: item for item in candidates}
        events: List[Dict[str, Any]] = []
        for index, candidate_id in enumerate(selected_ids):
            if candidate_id not in by_id:
                continue
            candidate = by_id[candidate_id]
            event: Dict[str, Any] = {
                "event_id": f"E{index + 1}",
                "subject": candidate["subject"],
                "action": candidate["action"],
                "completion_cue": candidate["completion_cue"],
            }
            hint = candidate.get("overview_time_hint_sec")
            # Only a whole-video overview has an unambiguous absolute timeline.
            # Chunk-local ranges remain diagnostics until their coordinate system
            # can be established reliably.
            if candidate.get("overview_source") == "global" and isinstance(hint, list):
                event["overview_time_hint_sec"] = list(hint)
            events.append(event)
        return {
            "events": events,
            "overview_ranges": diagnostics,
            "candidate_events": candidates,
            "selected_candidate_ids": selected_ids,
            "ranking_raw_responses": ranking_raw,
            "ranking_errors": ranking_errors,
            "raw_responses": raw_responses,
            "errors": errors,
        }

    def _window_rows(
        self, video_path: str, windows: Sequence[WindowSpec], events: Sequence[Dict[str, str]]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        prompt = build_event_window_prompt(events)
        prompts = [prompt for _window in windows]
        errors: List[str] = []
        all_raw: List[str] = []
        try:
            raw_rows = list(self.scorer.generate_ranges(video_path, windows, prompts))
            if len(raw_rows) != len(windows):
                raise RuntimeError("event batch returned an unexpected row count")
        except Exception as exc:
            errors.append(f"event batch failed: {type(exc).__name__}: {exc}")
            raw_rows = []
            for window in windows:
                try:
                    raw_rows.append(
                        self.scorer.generate_range(
                            video_path,
                            window.start_sec,
                            window.end_sec,
                            window.sample_fps,
                            prompt,
                        )
                    )
                except Exception as row_exc:
                    raw_rows.append(
                        f"<generation-error>{type(row_exc).__name__}: {row_exc}</generation-error>"
                    )

        output: List[Dict[str, Any]] = []
        for window, first_raw in zip(windows, raw_rows):
            attempts = 1
            raw_attempts = [str(first_raw)]
            parsed = None
            row_errors: List[str] = []
            try:
                parsed = parse_event_window_response(str(first_raw), len(events))
            except Exception as exc:
                row_errors.append(f"parse failed: {type(exc).__name__}: {exc}")
                try:
                    retry_raw = self.scorer.generate_range(
                        video_path,
                        window.start_sec,
                        window.end_sec,
                        window.sample_fps,
                        prompt
                        + " Your previous response was invalid. Return only the complete required JSON.",
                    )
                    attempts = 2
                    raw_attempts.append(str(retry_raw))
                    parsed = parse_event_window_response(str(retry_raw), len(events))
                except Exception as retry_exc:
                    row_errors.append(
                        f"retry failed: {type(retry_exc).__name__}: {retry_exc}"
                    )
            all_raw.extend(raw_attempts)
            row: Dict[str, Any] = {
                "window_id": window.window_id,
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "sample_fps": window.sample_fps,
                "stage": "global_event_verify",
                "valid": parsed is not None,
                "attempts": attempts,
                "raw": raw_attempts[-1],
                "raw_responses": raw_attempts,
                "errors": row_errors,
            }
            event_scores: Dict[str, Optional[float]] = {}
            if parsed is not None:
                row.update(parsed)
                for index, event in enumerate(events):
                    match = parsed["matches"][index]
                    completion = parsed["completion"][index]
                    base_score = (
                        0.70 * match / 3.0 + 0.30 * completion / 3.0
                    )
                    temporal_prior = 1.0
                    hint = event.get("overview_time_hint_sec")
                    if isinstance(hint, list) and len(hint) == 2:
                        start_hint = _finite(hint[0])
                        end_hint = _finite(hint[1])
                        centre = (window.start_sec + window.end_sec) / 2.0
                        if (
                            start_hint is not None
                            and end_hint is not None
                            and not start_hint - 2.0 <= centre <= end_hint + 2.0
                        ):
                            temporal_prior = 0.25
                    event_scores[event["event_id"]] = base_score * temporal_prior
                    row.setdefault("event_time_priors", {})[event["event_id"]] = temporal_prior
            else:
                row.update(
                    {
                        "matches": None,
                        "completion": None,
                        "visible_subject": None,
                        "evidence": None,
                    }
                )
                for event in events:
                    event_scores[event["event_id"]] = None
            row["event_scores"] = event_scores
            finite = [value for value in event_scores.values() if value is not None]
            row["qwen_score"] = max(finite) if finite else None
            output.append(row)
        return output, all_raw, errors

    def _attach_signals(
        self, video_path: str, windows: Sequence[WindowSpec], rows: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.signal_provider is not None:
            signals = []
            for window in windows:
                try:
                    signals.append(dict(self.signal_provider(video_path, window) or {}))
                except Exception as exc:
                    signals.append({"signal_error": f"{type(exc).__name__}: {exc}"})
        else:
            extractor = TemporalSignalExtractor(
                sample_fps=self.sample_fps,
                ffmpeg_bin=self.ffmpeg_bin,
                audio_sample_rate=self.audio_sample_rate,
            )
            try:
                signals = extractor.score_windows(video_path, windows)
            except Exception as exc:
                signals = [
                    {
                        "motion_score": None,
                        "audio_score": None,
                        "scene_score": None,
                        "signal_error": f"{type(exc).__name__}: {exc}",
                    }
                    for _window in windows
                ]
            finally:
                extractor.close()
        combined = []
        for row, signal in zip(rows, signals):
            copy = dict(row)
            copy.update(signal)
            combined.append(copy)
        calibration = fit_signal_calibration(combined, keys=TRADITIONAL_SIGNAL_KEYS)
        normalized, diagnostics = apply_signal_calibration(
            combined, calibration, keys=TRADITIONAL_SIGNAL_KEYS
        )
        diagnostics["calibration"] = calibration
        return normalized, diagnostics

    def _event_curves(
        self, rows: Sequence[Dict[str, Any]], events: Sequence[Dict[str, str]], duration: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        curves: Dict[str, List[Dict[str, Any]]] = {}
        for event in events:
            event_id = event["event_id"]
            payload = []
            for row in rows:
                score = (row.get("event_scores") or {}).get(event_id)
                if score is None:
                    continue
                payload.append(
                    {
                        "start_sec": row["start_sec"],
                        "end_sec": row["end_sec"],
                        "fused_score": score,
                    }
                )
            curves[event_id] = smooth_score_curve(
                build_score_curve(payload, duration, grid_sec=self.grid_sec),
                window_points=self.smoothing_points,
            )
        return curves

    @staticmethod
    def _row_interval(row: Mapping[str, Any]) -> Optional[Segment]:
        start = _finite(row.get("start_sec"))
        end = _finite(row.get("end_sec"))
        if start is None or end is None or end <= start:
            return None
        return start, end

    def _grow_peak(self, curve: Sequence[Dict[str, Any]], peak: Mapping[str, Any]) -> Segment:
        target = _finite(peak.get("time_sec"))
        if target is None:
            interval = self._row_interval(peak)
            return interval or (0.0, 0.0)
        index = min(
            range(len(curve)),
            key=lambda value: abs((_finite(curve[value].get("time_sec")) or 0.0) - target),
        )
        left = right = index
        while left > 0 and (_finite(curve[left - 1].get("score")) or -1.0) >= self.growth_threshold:
            left -= 1
        while right + 1 < len(curve) and (_finite(curve[right + 1].get("score")) or -1.0) >= self.growth_threshold:
            right += 1
        left_interval = self._row_interval(curve[left]) or (target, target)
        right_interval = self._row_interval(curve[right]) or (target, target)
        return left_interval[0], right_interval[1]

    def _boundary_activity(self, rows: Sequence[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
        payload = []
        for row in rows:
            values = [
                _finite(row.get("normalized_motion_score")),
                _finite(row.get("normalized_audio_score")),
            ]
            available = [value for value in values if value is not None]
            if not available:
                continue
            payload.append(
                {
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "fused_score": sum(available) / len(available),
                }
            )
        return build_score_curve(payload, duration, grid_sec=self.grid_sec, fill_missing=False)

    def _repair_boundary(
        self,
        segment: Segment,
        activity: Sequence[Dict[str, Any]],
        rows: Sequence[Dict[str, Any]],
        duration: float,
    ) -> Segment:
        start, end = segment

        def valley(boundary: float, is_start: bool) -> float:
            candidates = []
            for row in activity:
                interval = self._row_interval(row)
                score = _finite(row.get("score"))
                if interval is None or score is None:
                    continue
                centre = (interval[0] + interval[1]) / 2.0
                if boundary - 1.0 <= centre <= boundary + 1.0:
                    candidates.append((score, abs(centre - boundary), interval))
            if not candidates:
                return boundary
            _score, _distance, interval = min(candidates)
            return interval[1] if is_start else interval[0]

        start = valley(start, True)
        end = valley(end, False)
        # A strong cut close to the semantic edge is a harder boundary than an
        # activity valley.  Scene never creates a candidate; it only stops one.
        for row in rows:
            scene = _finite(row.get("normalized_scene_score"))
            interval = self._row_interval(row)
            if scene is None or scene < 0.75 or interval is None:
                continue
            cut = interval[1]
            if abs(cut - start) <= 1.0:
                start = cut
            if abs(cut - end) <= 1.0:
                end = interval[0]
        if end <= start:
            start, end = segment
        return max(0.0, start - self.padding_sec), min(duration, end + self.padding_sec)

    @staticmethod
    def _segment_budget(duration: float) -> int:
        if duration <= 30.0:
            return 2
        if duration <= 120.0:
            return 4
        return 6

    def _gap_is_missing_only(
        self,
        rows: Sequence[Mapping[str, Any]],
        start: float,
        end: float,
    ) -> bool:
        """Return true when a short gap contains no valid semantic judgement."""

        if end <= start or end - start > 1.5:
            return False
        overlapping = []
        for row in rows:
            interval = self._row_interval(row)
            if interval is None or min(interval[1], end) <= max(interval[0], start):
                continue
            overlapping.append(row)
        return bool(overlapping) and all(not bool(row.get("valid")) for row in overlapping)

    def _select_segments(
        self,
        rows: Sequence[Dict[str, Any]],
        events: Sequence[Dict[str, str]],
        curves: Mapping[str, Sequence[Dict[str, Any]]],
        duration: float,
    ) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        for event in events:
            event_id = event["event_id"]
            curve = list(curves.get(event_id) or [])
            peaks = find_local_peaks(
                curve,
                min_score=self.seed_threshold,
                top_fraction=1.0,
                min_prominence=self.min_prominence,
            )
            for peak in peaks:
                segment = self._grow_peak(curve, peak)
                if segment[1] > segment[0]:
                    candidates.append(
                        {
                            "event_id": event_id,
                            "score": float(peak.get("peak_score") or 0.0),
                            "prominence": float(peak.get("prominence") or 0.0),
                            "time_sec": peak.get("time_sec"),
                            "semantic_segment": list(segment),
                        }
                    )
        fallback_used = None
        if not candidates:
            best = None
            for event in events:
                event_id = event["event_id"]
                for row in curves.get(event_id) or []:
                    score = _finite(row.get("score"))
                    if score is None:
                        continue
                    candidate = (score, event_id, row)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is not None:
                score, event_id, row = best
                centre = _finite(row.get("time_sec")) or duration / 2.0
                length = min(duration, 4.0)
                start = max(0.0, min(duration - length, centre - length / 2.0))
                candidates.append(
                    {
                        "event_id": event_id,
                        "score": score,
                        "prominence": 0.0,
                        "time_sec": centre,
                        "semantic_segment": [start, start + length],
                    }
                )
                fallback_used = "top_event_peak"

        activity = self._boundary_activity(rows, duration)
        for candidate in candidates:
            candidate["segment"] = list(
                self._repair_boundary(
                    tuple(candidate["semantic_segment"]), activity, rows, duration
                )
            )
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        accepted: List[Dict[str, Any]] = []
        limit = self._segment_budget(duration)
        coverage_limit = duration * self.max_coverage_ratio
        for candidate in ranked:
            segment = tuple(candidate["segment"])
            if segment[1] - segment[0] < 1.0:
                centre = (segment[0] + segment[1]) / 2.0
                segment = (max(0.0, centre - 0.5), min(duration, centre + 0.5))
                candidate["segment"] = list(segment)
            if any(_interval_iou(segment, tuple(item["segment"])) > 0.30 for item in accepted):
                continue
            if len(accepted) >= limit:
                break
            current = sum(item["segment"][1] - item["segment"][0] for item in accepted)
            room = coverage_limit - current
            if room < 1.0:
                continue
            if segment[1] - segment[0] > room:
                centre = _finite(candidate.get("time_sec")) or sum(segment) / 2.0
                start = max(0.0, min(duration - room, centre - room / 2.0))
                segment = (start, start + room)
                candidate["segment"] = list(segment)
            accepted.append(candidate)

        ordered = sorted((tuple(item["segment"]), item) for item in accepted)
        merged: List[Segment] = []
        bridged_missing_gaps: List[List[float]] = []
        for segment, _item in ordered:
            gap_start = merged[-1][1] if merged else 0.0
            gap_end = segment[0]
            missing_only = bool(
                merged and self._gap_is_missing_only(rows, gap_start, gap_end)
            )
            if merged and (
                segment[0] - merged[-1][1] <= self.merge_gap_sec or missing_only
            ):
                if missing_only and segment[0] - merged[-1][1] > self.merge_gap_sec:
                    bridged_missing_gaps.append([gap_start, gap_end])
                merged[-1] = (merged[-1][0], max(merged[-1][1], segment[1]))
            else:
                merged.append(segment)
        return {
            "segments_sec": merged,
            "selected_event_anchors": accepted,
            "candidate_event_anchors": candidates,
            "fallback_used": fallback_used,
            "segment_budget": limit,
            "coverage_limit_sec": coverage_limit,
            "bridged_missing_gaps": bridged_missing_gaps,
        }

    def _selected_semantics(
        self,
        events: Sequence[Dict[str, str]],
        rows: Sequence[Dict[str, Any]],
        selection: Mapping[str, Any],
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        anchors = list(selection.get("selected_event_anchors") or [])
        if not anchors:
            return None, None, {"reason": "no_selected_anchor"}
        best = max(anchors, key=lambda item: float(item.get("score") or 0.0))
        event = next(
            (item for item in events if item["event_id"] == best.get("event_id")), None
        )
        if event is None:
            return None, None, {"reason": "selected_event_missing"}
        expected = _canonical_subject(event.get("subject"))
        segment = tuple(best["segment"])
        subjects = []
        for row in rows:
            interval = self._row_interval(row)
            if interval is None or min(interval[1], segment[1]) <= max(interval[0], segment[0]):
                continue
            value = _canonical_subject(row.get("visible_subject"))
            if value:
                subjects.append(value)
        majority = Counter(subjects).most_common(1)[0][0] if subjects else None
        subject = expected if expected is not None and expected == majority else None
        diagnostics = {
            "event_subject": expected,
            "window_subject_majority": majority,
            "window_subject_counts": dict(Counter(subjects)),
            "conflict": expected != majority,
        }
        return subject, event.get("action") or None, diagnostics

    def _fallback(
        self,
        video_path: str,
        reason: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        raw_responses: Optional[Sequence[str]] = None,
    ) -> TemporalResult:
        context = dict(context or {})
        preserved_raw = list(raw_responses or [])
        if self.fallback_detector is not None:
            result = self.fallback_detector.detect(video_path)
            result.errors = [reason] + list(result.errors or [])
            result.source = f"global_event_fallback_{result.source}"
            result.fallback_used = result.fallback_used or "structured_window_score"
            diagnostics = dict(result.score_diagnostics or {})
            diagnostics["global_event_fallback_reason"] = reason
            diagnostics["global_event_failed_attempt"] = context
            result.score_diagnostics = diagnostics
            result.vlm_raw_responses = preserved_raw + list(result.vlm_raw_responses or [])
            return result
        n_frames, fps, _width, _height = video_meta(video_path)
        duration = n_frames / fps if n_frames > 0 and fps > 0 else 0.0
        length = min(duration, 4.0)
        start = max(0.0, (duration - length) / 2.0)
        segments = [(start, start + length)] if length > 0 else []
        return TemporalResult(
            segments,
            raw="global-event-fallback",
            source="global_event_middle_fallback",
            valid=bool(segments),
            errors=[reason],
            fallback_used="middle",
            duration_sec=duration,
            coverage_ratio=temporal_coverage_ratio(segments, duration),
        )

    def detect(self, video_path: str) -> TemporalResult:
        n_frames, fps, _width, _height = video_meta(video_path)
        if n_frames <= 0 or fps <= 0:
            return TemporalResult(
                [], raw="global-event", source="global_event", valid=False,
                errors=["unreadable video"], duration_sec=0.0,
            )
        duration = n_frames / float(fps)
        overview = self._global_events(video_path, duration)
        events = list(overview.get("events") or [])
        if not events:
            return self._fallback(
                video_path,
                "global event overview produced no valid events",
                context={
                    "stage": "overview",
                    "overview_ranges": overview.get("overview_ranges") or [],
                    "candidate_events": overview.get("candidate_events") or [],
                    "errors": overview.get("errors") or [],
                },
                raw_responses=overview.get("raw_responses") or [],
            )

        windows = generate_windows(
            duration,
            self.window_sec,
            self.stride_sec,
            self.sample_fps,
            stage="global_event_verify",
        )
        rows, window_raw, window_errors = self._window_rows(video_path, windows, events)
        valid_rows = [row for row in rows if row.get("valid")]
        if not valid_rows:
            return self._fallback(
                video_path,
                "global event verification produced no valid windows",
                context={
                    "stage": "window_verification",
                    "global_events": events,
                    "overview_ranges": overview.get("overview_ranges") or [],
                    "window_records": rows,
                    "errors": list(overview.get("errors") or []) + list(window_errors),
                },
                raw_responses=list(overview.get("raw_responses") or []) + list(window_raw),
            )
        rows, signal_diagnostics = self._attach_signals(video_path, windows, rows)
        curves = self._event_curves(rows, events, duration)
        selection = self._select_segments(rows, events, curves, duration)
        segments = list(selection.get("segments_sec") or [])
        if not segments:
            return self._fallback(
                video_path,
                "global event selection produced no segment",
                context={
                    "stage": "segment_selection",
                    "global_events": events,
                    "window_records": rows,
                    "event_curves": curves,
                    "selection": selection,
                    "signal_diagnostics": signal_diagnostics,
                },
                raw_responses=list(overview.get("raw_responses") or []) + list(window_raw),
            )
        subject_hint, action_description, subject_diagnostics = self._selected_semantics(
            events, rows, selection
        )

        combined_curve: List[Dict[str, Any]] = []
        curve_lengths = [len(value) for value in curves.values()]
        for index in range(max(curve_lengths, default=0)):
            values = []
            for event_id, curve in curves.items():
                if index >= len(curve):
                    continue
                score = _finite(curve[index].get("score"))
                if score is not None:
                    values.append((score, event_id, curve[index]))
            if values:
                score, event_id, row = max(values)
                copy = dict(row)
                copy["score"] = score
                copy["best_event_id"] = event_id
                combined_curve.append(copy)

        for row in rows:
            interval = self._row_interval(row)
            row["selected_for_output"] = bool(
                interval
                and any(min(interval[1], end) > max(interval[0], start) for start, end in segments)
            )
        all_errors = list(overview.get("errors") or []) + list(window_errors)
        all_errors.extend(error for row in rows for error in row.get("errors") or [])
        raw_responses = list(overview.get("raw_responses") or []) + list(window_raw)
        diagnostics = {
            "mode": "global_event",
            "global_events": events,
            "overview_ranges": overview.get("overview_ranges") or [],
            "candidate_events": overview.get("candidate_events") or [],
            "selected_candidate_ids": overview.get("selected_candidate_ids") or [],
            "event_curves": curves,
            "selected_event_anchors": selection.get("selected_event_anchors") or [],
            "candidate_event_anchors": selection.get("candidate_event_anchors") or [],
            "bridged_missing_gaps": selection.get("bridged_missing_gaps") or [],
            "boundary_policy": "semantic_seed_then_motion_audio_valley_scene_stop",
            "subject_consensus": subject_diagnostics,
            "signal_diagnostics": signal_diagnostics,
            "window_count": len(windows),
            "valid_window_count": len(valid_rows),
            "segment_budget": selection.get("segment_budget"),
            "coverage_limit_sec": selection.get("coverage_limit_sec"),
        }
        return TemporalResult(
            [tuple(map(float, item)) for item in segments],
            subject_hint=subject_hint,
            action_description=action_description,
            raw=json.dumps(
                {"mode": "global_event", "events": events, "window_count": len(windows)},
                ensure_ascii=False,
            ),
            source="global_event_curve",
            valid=True,
            errors=all_errors,
            attempts=len(raw_responses),
            fallback_used=selection.get("fallback_used"),
            duration_sec=duration,
            coverage_ratio=temporal_coverage_ratio(segments, duration),
            vlm_raw_responses=raw_responses,
            segments_before_normalization=list(segments),
            segments_after_normalization=list(segments),
            corrections=["global event queries condition all short-window judgements"],
            window_records=rows,
            score_curve=combined_curve,
            selected_peaks=list(selection.get("selected_event_anchors") or []),
            score_diagnostics=diagnostics,
        )

    def close(self) -> None:
        close = getattr(self.scorer, "close", None)
        if callable(close):
            close()


__all__ = [
    "GlobalEventFrontend",
    "build_event_ranking_prompt",
    "build_event_window_prompt",
    "build_global_event_prompt",
    "parse_event_ranking_response",
    "parse_event_window_response",
    "parse_global_event_response",
]
