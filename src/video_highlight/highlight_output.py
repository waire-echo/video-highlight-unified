"""Strict and audited lenient parsing for temporal highlight responses."""
from __future__ import annotations

from hashlib import sha256
import json
import re
from statistics import median

from video_highlight.timing import normalize_interval


POLICY = "diagnostic_lenient/v2"

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_FIELD_NUMBER = re.compile(
    rf'("(?:start|end)"\s*:\s*{_NUMBER})(\s*)(")(\s*[,}}\]])'
)
_NESTED_HIGHLIGHTS = re.compile(
    r',\s*"highlights"\s*:\s*\[',
)
_START_END_PAIR = re.compile(
    rf'"start"\s*:\s*({_NUMBER})\s*,\s*"end"\s*:\s*({_NUMBER})'
)
_START_OR_END_FIELD = re.compile(
    rf'"(start|end)"\s*:\s*({_NUMBER})'
)
_UNSUPPORTED_FIELD = re.compile(
    rf'(,?\s*"(?!start"|end")[^"\\]+"\s*:\s*'
    rf'(?:"(?:[^"\\]|\\.)*"|true|false|null|{_NUMBER}))'
)
_ROLLING_MIN_WINDOWS = 9


def _digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def strict_parse_answer(raw):
    """Parse the exact previously accepted answer form."""
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _audit_removal(repair, *, stage, candidate, start, end, reason,
                   **details):
    removed = candidate[start:end]
    repair.setdefault("discarded_fragments", []).append({
        "stage": stage,
        "candidate_start": start,
        "candidate_end": end,
        "removed_chars": len(removed),
        "removed_text": removed,
        "reason": reason,
        **details,
    })
    repair["removed_characters"] = (
        repair.get("removed_characters", 0) + len(removed)
    )


def _regex_sub_with_audit(repair, candidate, pattern, replacement, reason,
                          stage):
    pieces = []
    position = 0
    count = 0
    for match in pattern.finditer(candidate):
        pieces.append(candidate[position:match.start()])
        replacement_text = match.expand(replacement)
        _audit_removal(
            repair,
            stage=stage,
            candidate=candidate,
            start=match.start(),
            end=match.end(),
            reason=reason,
            replacement=replacement_text,
        )
        pieces.append(replacement_text)
        position = match.end()
        count += 1
    pieces.append(candidate[position:])
    return "".join(pieces), count


def _find_repeated_closing_loop(text):
    """Find the earliest run of at least three identical delimiter tokens."""
    closing = set("]} `\t\r\n")
    for start in range(len(text)):
        for token_length in range(2, 9):
            stop = start + token_length
            if stop > len(text):
                continue
            token = text[start:stop]
            if (
                "]" not in token
                or "}" not in token
                or any(character not in closing for character in token)
            ):
                continue
            position = start
            repeats = 0
            while text[position:position + token_length] == token:
                repeats += 1
                position += token_length
            if repeats >= 3:
                return {
                    "start": start,
                    "end": position,
                    "token": token,
                    "repeats": repeats,
                }
    return None


def _truncate_closing_loop(candidate, repair):
    loop = _find_repeated_closing_loop(candidate)
    if loop is None:
        return candidate
    truncated = candidate[:loop["start"]]
    _audit_removal(
        repair,
        stage="closing_loop",
        candidate=candidate,
        start=loop["start"],
        end=loop["end"],
        reason="repeated_closing_loop",
        repeated_token=loop["token"],
        repetitions=loop["repeats"],
    )
    repair["closing_loop"] = {
        "candidate_start": loop["start"],
        "candidate_end": loop["end"],
        "token": loop["token"],
        "repetitions": loop["repeats"],
    }
    return truncated


def _patch_trailing_closers(candidate, repair):
    """Remove only extra closing delimiters after an otherwise complete value."""
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass
    stripped = candidate.rstrip()
    start = len(stripped)
    while start and stripped[start - 1] in "]}":
        try:
            parsed = json.loads(stripped[:start])
        except json.JSONDecodeError:
            start -= 1
            continue
        if isinstance(parsed, dict) and "highlights" in parsed:
            _audit_removal(
                repair,
                stage="trailing_closers",
                candidate=candidate,
                start=start,
                end=len(candidate),
                reason="extra_trailing_closers",
            )
            return candidate[:start]
        break
    return candidate


def _extract_complete_pairs(candidate, repair):
    rows = []
    covered = []
    for match in _START_END_PAIR.finditer(candidate):
        start, end = float(match.group(1)), float(match.group(2))
        covered.append((match.start(), match.end()))
        if start < end:
            rows.append({"start": start, "end": end})
        else:
            repair.setdefault("discarded_highlights", []).append({
                "start": start,
                "end": end,
                "reason": "nonpositive_pair",
            })
    dropped = []
    for match in _START_OR_END_FIELD.finditer(candidate):
        if any(
            left <= match.start() and match.end() <= right
            for left, right in covered
        ):
            continue
        dropped.append({
            "field": match.group(1),
            "value": float(match.group(2)),
            "candidate_start": match.start(),
            "candidate_end": match.end(),
            "reason": "incomplete_start_or_end",
        })
    if dropped:
        repair.setdefault("dropped_incomplete_pairs", []).extend(dropped)
    if rows:
        repair.setdefault("quality_flags", []).append(
            "complete_pair_extraction"
        )
    return rows


def _flatten_highlight_rows(value, repair, *, nested_count=0):
    """Flatten at most one repeated highlights wrapper into timing rows."""
    rows = []
    if isinstance(value, dict) and {"start", "end"}.issubset(value):
        if set(value) != {"start", "end"}:
            repair.setdefault("quality_flags", []).append(
                "extra_row_fields_discarded"
            )
        rows.append({"start": value["start"], "end": value["end"]})
        return rows, nested_count
    if isinstance(value, dict):
        if "highlights" in value:
            nested_count += 1
            if nested_count > 1:
                raise ValueError(
                    "More than one repeated highlights container"
                )
            nested_rows, nested_count = _flatten_highlight_rows(
                value["highlights"], repair, nested_count=nested_count
            )
            rows.extend(nested_rows)
        elif "start" in value or "end" in value:
            repair.setdefault("dropped_incomplete_pairs", []).append({
                "reason": "incomplete_start_or_end",
                "row": value,
            })
        else:
            repair.setdefault("discarded_fragments", []).append({
                "stage": "non_timing_row",
                "reason": "discard_non_timing_row",
                "removed_text": json.dumps(
                    value, ensure_ascii=False, sort_keys=True
                ),
                "removed_chars": 0,
            })
            repair["actions"].append("discard_non_timing_row")
            for item in value.values():
                item_rows, nested_count = _flatten_highlight_rows(
                    item, repair, nested_count=nested_count
                )
                rows.extend(item_rows)
        return rows, nested_count
    if isinstance(value, list):
        for item in value:
            item_rows, nested_count = _flatten_highlight_rows(
                item, repair, nested_count=nested_count
            )
            rows.extend(item_rows)
        return rows, nested_count
    return rows, nested_count


def _normalize_object(obj, repair):
    if not isinstance(obj, dict) or "highlights" not in obj:
        raise ValueError("Expected highlights field")
    if not isinstance(obj["highlights"], list):
        raise ValueError("highlights must be an array")
    rows, nested_count = _flatten_highlight_rows(
        obj["highlights"], repair
    )
    if nested_count:
        repair.setdefault("quality_flags", []).append(
            "nested_highlights_container_flattened"
        )
        repair["nested_highlights_containers"] = nested_count
    if set(obj) != {"highlights"}:
        repair.setdefault("quality_flags", []).append("extra_top_fields")
    clean_rows = []
    discarded_nonpositive = 0
    for row in rows:
        if not {"start", "end"}.issubset(row):
            repair.setdefault("dropped_incomplete_pairs", []).append(row)
            continue
        if (
            type(row.get("start")) in (int, float)
            and type(row.get("end")) in (int, float)
            and row["start"] >= row["end"]
        ):
            discarded_nonpositive += 1
            repair.setdefault("discarded_highlights", []).append({
                **row,
                "reason": "nonpositive_pair",
            })
            continue
        clean_rows.append({"start": row["start"], "end": row["end"]})
    if discarded_nonpositive:
        repair["actions"].append("discard_nonpositive_highlights")
        repair.setdefault("quality_flags", []).append(
            "nonpositive_intervals_discarded"
        )
    if rows and not clean_rows and not repair.get("discarded_highlights"):
        repair.setdefault("quality_flags", []).append(
            "discard_non_timing_rows_as_empty"
        )
        repair["actions"].append("discard_non_timing_rows_as_empty")
    if len(clean_rows) == 1 and clean_rows[0]["start"] == 0:
        repair.setdefault("quality_flags", []).append(
            "single_highlight_starts_at_zero"
        )
    elif len(clean_rows) == 1:
        repair.setdefault("quality_flags", []).append(
            "single_highlight"
        )
    elif len(clean_rows) > 1:
        repair.setdefault("quality_flags", []).append(
            "multiple_highlights"
        )
    return {"highlights": clean_rows}


def lenient_parse_answer(raw):
    """Apply explicit, audited repairs and return (object, repair record)."""
    original_sha256 = _digest(raw)
    repair = {
        "enabled": True,
        "policy": POLICY,
        "actions": [],
        "quality_flags": [],
        "discarded_fragments": [],
        "removed_duplicate_intervals": [],
        "merged_intervals": [],
        "dropped_incomplete_pairs": [],
        "rolling_window_detections": [],
        "original_sha256": original_sha256,
    }
    try:
        obj = strict_parse_answer(raw)
    except (json.JSONDecodeError, IndexError) as strict_error:
        repair["strict_error"] = (
            f"{type(strict_error).__name__}: {strict_error}"
        )
        candidate = raw.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            inner = candidate.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            start = candidate.find(inner)
            _audit_removal(
                repair,
                stage="markdown_fence",
                candidate=candidate,
                start=0,
                end=start,
                reason="remove_markdown_fence_prefix",
            )
            suffix_start = start + len(inner)
            _audit_removal(
                repair,
                stage="markdown_fence",
                candidate=candidate,
                start=suffix_start,
                end=len(candidate),
                reason="remove_markdown_fence_suffix",
            )
            candidate = inner
            repair["actions"].append("remove_markdown_fence")
        for prefix, suffix in (
            ("<highlights>", "</highlights>"),
            ("<output>", "</output>"),
        ):
            if candidate.startswith(prefix):
                _audit_removal(
                    repair,
                    stage="wrapper",
                    candidate=candidate,
                    start=0,
                    end=len(prefix),
                    reason="remove_output_prefix",
                )
                candidate = candidate[len(prefix):].strip()
                repair["actions"].append("remove_output_prefix")
                if candidate.endswith(suffix):
                    _audit_removal(
                        repair,
                        stage="wrapper",
                        candidate=candidate,
                        start=len(candidate) - len(suffix),
                        end=len(candidate),
                        reason="remove_output_suffix",
                    )
                    candidate = candidate[:-len(suffix)].strip()
                    repair["actions"].append("remove_output_suffix")
        if candidate.endswith("</output>"):
            _audit_removal(
                repair,
                stage="wrapper",
                candidate=candidate,
                start=len(candidate) - len("</output>"),
                end=len(candidate),
                reason="remove_output_suffix",
            )
            candidate = candidate[:-len("</output>")].strip()
            repair["actions"].append("remove_output_suffix")

        candidate, count = _regex_sub_with_audit(
            repair,
            candidate,
            _FIELD_NUMBER,
            r"\1\2\4",
            "remove_extra_field_value_quote",
            "field_value_quote",
        )
        if count:
            repair["actions"].append("remove_extra_field_value_quote")
        candidate, count = _regex_sub_with_audit(
            repair,
            candidate,
            _NESTED_HIGHLIGHTS,
            ", ",
            "remove_repeated_highlights_key",
            "repeated_highlights",
        )
        if count:
            repair["actions"].append("remove_repeated_highlights_key")
            repair.setdefault("quality_flags", []).append(
                "repeated_highlights_key"
            )
        candidate, count = _regex_sub_with_audit(
            repair,
            candidate,
            _UNSUPPORTED_FIELD,
            "",
            "remove_unsupported_output_field",
            "unsupported_field",
        )
        if count:
            repair["actions"].append("remove_unsupported_output_field")
            repair.setdefault("quality_flags", []).append(
                "unsupported_fields_discarded"
            )

        before_loop = candidate
        candidate = _truncate_closing_loop(candidate, repair)
        if candidate != before_loop:
            repair["actions"].append("truncate_repeated_closing_loop")
            repair["quality_flags"].append("repeated_closing_loop")
        before_closers = candidate
        candidate = _patch_trailing_closers(candidate, repair)
        if candidate != before_closers:
            repair["actions"].append("remove_extra_trailing_closers")
            repair["quality_flags"].append("extra_trailing_closers")
        if candidate != raw.strip():
            repair["repaired_text_sha256"] = _digest(candidate)
        repair["repaired_text"] = candidate
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as repaired_error:
            rows = _extract_complete_pairs(candidate, repair)
            if not rows:
                raise ValueError(
                    "Lenient output repair failed: " + str(repaired_error)
                ) from repaired_error
            obj = {"highlights": rows}
            repair["actions"].append("extract_complete_start_end_pairs")
            repair["pair_extraction_error"] = str(repaired_error)
        obj = _normalize_object(obj, repair)
    else:
        obj = _normalize_object(obj, repair)
    repair["changed"] = bool(repair["actions"])
    repair["normalization"] = POLICY
    return obj, repair


def _detect_rolling_window_runs(spans):
    """Detect long fixed-step runs, overlapping or disjoint.

    A run of many uniform windows is a model output pattern rather than a
    credible list of independent highlights. Disjoint runs are included so
    an every-ten-seconds enumeration cannot evade the loop guard.
    """
    detections = []
    index = 0
    while index < len(spans):
        end = index + 1
        steps = []
        while end < len(spans):
            previous = spans[end - 1]
            current = spans[end]
            step = current[0] - previous[0]
            if step <= 0:
                break
            if steps:
                reference = median(steps)
                tolerance = max(0.03, 0.10 * max(abs(reference), 1.0))
                if abs(step - reference) > tolerance:
                    break
            steps.append(step)
            end += 1
        if end - index >= _ROLLING_MIN_WINDOWS:
            run = spans[index:end]
            step = median(steps)
            tolerance = max(0.03, 0.10 * max(abs(step), 1.0))
            overlaps = any(
                current[0] <= previous[1]
                for previous, current in zip(run, run[1:])
            )
            detections.append({
                "start_index": index,
                "end_index_exclusive": end,
                "window_count": len(run),
                "step_sec": step,
                "step_tolerance_sec": tolerance,
                "overlap": overlaps,
                "intervals": [list(span) for span in run],
                "envelope": [
                    min(span[0] for span in run),
                    max(span[1] for span in run),
                ],
                "decision": "quarantine",
            })
            index = end
        else:
            index += 1
    return detections


def lenient_normalized_answer(raw, duration, lenient=False,
                              end_tolerance_sec=0.5,
                              final_frame_pts_sec=None,
                              max_items=None):
    """Parse and normalize response fields, retaining every repair decision."""
    if lenient:
        obj, repair = lenient_parse_answer(raw)
        repair["end_tolerance_sec"] = end_tolerance_sec
    else:
        obj = strict_parse_answer(raw)
        if not isinstance(obj, dict) or set(obj) != {"highlights"}:
            raise ValueError("Expected only the highlights field")
        if not isinstance(obj["highlights"], list):
            raise ValueError("highlights must be an array")
        for row in obj["highlights"]:
            if not isinstance(row, dict) or set(row) != {"start", "end"}:
                raise ValueError(
                    "Each highlight requires only start and end"
                )
        repair = {
            "enabled": False,
            "changed": False,
            "actions": [],
            "quality_flags": [],
            "discarded_fragments": [],
            "removed_duplicate_intervals": [],
            "merged_intervals": [],
            "dropped_incomplete_pairs": [],
            "rolling_window_detections": [],
        }
    if max_items is not None and (
        type(max_items) is not int or max_items < 1
    ):
        raise ValueError("max_items must be a positive integer")
    exclusive_normalizations = []
    if lenient and final_frame_pts_sec is not None:
        final_pts = float(final_frame_pts_sec)
        if abs(final_pts - float(duration)) > 1e-6:
            for row in obj["highlights"]:
                original = row["end"]
                if float(original) == final_pts:
                    row["end"] = duration
                    exclusive_normalizations.append({
                        "field": "end",
                        "raw": original,
                        "normalized": float(duration),
                        "difference_sec": float(
                            abs(float(original) - float(duration))
                        ),
                        "reason": "exclusive_last_frame_pts",
                    })
            if exclusive_normalizations:
                repair["actions"].append(
                    "normalize_exclusive_last_frame_end"
                )
                repair.setdefault(
                    "exclusive_end_normalizations", []
                ).extend(exclusive_normalizations)
                repair["changed"] = True
    changes = list(exclusive_normalizations)
    raw_rolling = []
    raw_rolling_spans = set()
    if lenient:
        raw_indexed_spans = []
        for index, row in enumerate(obj["highlights"]):
            start = row.get("start")
            end = row.get("end")
            if (
                type(start) in (int, float)
                and type(end) in (int, float)
                and start < end
            ):
                raw_indexed_spans.append(
                    (float(start), float(end), index)
                )
        detected_runs = _detect_rolling_window_runs([
            [start, end]
            for start, end, _ in sorted(raw_indexed_spans)
        ])
        raw_rolling = [
            detection
            for detection in detected_runs
            if detection["overlap"]
            and all(
                current[0] == previous[1]
                for previous, current in zip(
                    detection["intervals"],
                    detection["intervals"][1:],
                )
            )
            and all(
                abs((current[1] - current[0]) - 1.0) <= 1e-9
                for current in detection["intervals"]
            )
        ]
        if raw_rolling:
            repair["rolling_window_detections"] = raw_rolling
            repair["actions"].append(
                "merge_rolling_window_loop_to_envelope"
            )
            repair["quality_flags"].append(
                "rolling_window_loop_merged_to_envelope"
            )
            raw_rolling_spans = {
                tuple(span)
                for detection in raw_rolling
                for span in detection["intervals"]
            }
        elif detected_runs:
            repair["rolling_window_detections"] = detected_runs
    indexed_spans = []
    canonical_rows = []
    for index, row in enumerate(obj["highlights"]):
        if (
            type(row.get("start")) in (int, float)
            and type(row.get("end")) in (int, float)
            and (float(row["start"]), float(row["end"]))
            in raw_rolling_spans
        ):
            continue
        span, row_changes = normalize_interval(
            row["start"],
            row["end"],
            duration,
            lenient=lenient,
            end_tolerance_sec=end_tolerance_sec if lenient else None,
        )
        changes.extend({
            "highlight_index": index,
            **change,
        } for change in row_changes)
        indexed_spans.append((span[0], span[1], index))
        canonical_rows.append({
            "start": span[0],
            "end": span[1],
            "source_index": index,
        })

    for detection in raw_rolling:
        envelope = detection["envelope"]
        span, row_changes = normalize_interval(
            envelope[0],
            envelope[1],
            duration,
            lenient=lenient,
            end_tolerance_sec=end_tolerance_sec if lenient else None,
        )
        changes.extend({
            "highlight_index": None,
            "source": "rolling_window_envelope",
            **change,
        } for change in row_changes)
        indexed_spans.append((span[0], span[1], len(obj["highlights"])))
        canonical_rows.append({
            "start": span[0],
            "end": span[1],
            "source_index": None,
        })

    unique = []
    seen = set()
    for span in sorted(indexed_spans):
        key = (span[0], span[1])
        if key in seen:
            repair["removed_duplicate_intervals"].append({
                "start": span[0],
                "end": span[1],
                "source_indices": [
                    item["source_index"]
                    for item in canonical_rows
                    if item["start"] == span[0]
                    and item["end"] == span[1]
                ],
            })
            continue
        seen.add(key)
        unique.append([span[0], span[1]])
    if repair["removed_duplicate_intervals"]:
        repair["actions"].append("remove_exact_duplicate_intervals")
        repair["quality_flags"].append("exact_duplicate_intervals_removed")

    merged = []
    for start, end in unique:
        if merged and start <= merged[-1][1]:
            previous = merged[-1]
            repair["merged_intervals"].append({
                "inputs": [
                    [previous[0], previous[1]],
                    [start, end],
                ],
                "result": [previous[0], max(previous[1], end)],
                "kind": (
                    "adjacent" if start == previous[1] else "overlap"
                ),
            })
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    if repair["merged_intervals"]:
        repair["actions"].append(
            "merge_overlapping_or_adjacent_intervals"
        )
        repair["quality_flags"].append(
            "overlapping_or_adjacent_intervals_merged"
        )

    if max_items is not None and len(merged) > max_items:
        raise ValueError(
            f"highlights contains {len(merged)} merged items; "
            f"the output contract allows at most {max_items}"
        )

    repair["time_normalizations"] = changes
    repair["quality_flags"] = list(dict.fromkeys(
        repair.get("quality_flags", [])
    ))
    return {"highlights": [
        {"start": start, "end": end} for start, end in merged
    ]}, merged, {
        "enabled": lenient,
        "policy": POLICY if lenient else "strict",
        "repair": repair,
        "time_changes": changes,
        "quality_flags": repair["quality_flags"],
    }


def incomplete_prefix_candidate(raw, duration, *,
                                end_tolerance_sec=0.5):
    """Return the leading closed candidate for a truncated generation.

    This is diagnostic evidence only. It is never marked complete and callers
    must not generate the final frame handoff from it automatically.
    """
    text = raw.lstrip()
    candidate = None
    pair_error = None
    if text:
        opening = text.find("{")
        if opening >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[opening:])
            except json.JSONDecodeError as error:
                pair_error = str(error)
            else:
                try:
                    _, spans, normalization = lenient_normalized_answer(
                        json.dumps(obj, ensure_ascii=False),
                        duration,
                        lenient=True,
                        end_tolerance_sec=end_tolerance_sec,
                    )
                except ValueError as error:
                    pair_error = str(error)
                else:
                    if spans:
                        candidate = {
                            "kind": "closed_json",
                            "segments": spans,
                            "closed_pair_count": len(spans),
                            "quality_flags": normalization["quality_flags"],
                            "repair": normalization["repair"],
                        }
    if candidate is None:
        loop = _find_repeated_closing_loop(text)
        scan_text = text[:loop["start"]] if loop else text
        match = _START_END_PAIR.search(scan_text)
        if match:
            try:
                span, row_changes = normalize_interval(
                    float(match.group(1)),
                    float(match.group(2)),
                    duration,
                    lenient=True,
                    end_tolerance_sec=end_tolerance_sec,
                )
            except ValueError as error:
                pair_error = str(error)
            else:
                candidate = {
                    "kind": "complete_start_end_pair",
                    "segments": [span],
                    "closed_pair_count": 1,
                    "quality_flags": (
                        ["repeated_closing_loop"] if loop else []
                    ),
                    "time_changes": row_changes,
                    "candidate_start": match.start(),
                    "candidate_end": match.end(),
                }
    if candidate is None:
        return {
            "status": "incomplete",
            "segments": [],
            "closed_pair_count": 0,
            "quality_flags": ["generation_truncated"],
            "diagnostic_only": True,
            "error": pair_error,
        }
    candidate["status"] = "incomplete"
    candidate["quality_flags"] = list(dict.fromkeys(
        ["generation_truncated", *candidate["quality_flags"]]
    ))
    candidate["diagnostic_only"] = True
    candidate["candidate_sha256"] = _digest(raw)
    return candidate
