"""Versioned prompt builders, extracted without changing emitted text."""
from __future__ import annotations
import json
from typing import Sequence, Mapping


def build_global_event_prompt(*, max_events: int = 3) -> str:
    return (
        "Watch the supplied video range as one story, not as isolated frames. "
        f"Identify 1 to {max(1, int(max_events))} distinct visible events most worth "
        "preserving in a highlight edit. Prefer a decisive action, interaction, "
        "surprise, completed result, or meaningful state change. Routine walking, "
        "standing, setup, camera motion, and image quality are not highlights unless "
        "they are the central event. Do not output timestamps or crop information. "
        "Answer in English. Return only one complete compact JSON array. Each event is "
        "exactly [subject, action, completion_cue], in ranked order. Do not output key "
        "names, event IDs, markdown, or explanation. Keep every string under 12 words."
    )


def build_event_ranking_prompt(candidates: Sequence[Mapping[str, str]], *, max_events: int = 3) -> str:
    compact = [
        {
            "candidate_id": item["candidate_id"],
            "subject": item["subject"],
            "action": item["action"],
            "completion_cue": item["completion_cue"],
        }
        for item in candidates
    ]
    return (
        "Watch the complete supplied video and compare the candidate event descriptions "
        "below. Select the candidates that are visibly present and most worth preserving. "
        "Reject routine setup and duplicates. Do not invent a new candidate. Return only "
        "one JSON object with exactly selected_candidate_ids, containing 1 to "
        f"{max(1, int(max_events))} IDs in ranked order. Candidates: "
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    )


def build_event_window_prompt(events: Sequence[Mapping[str, str]]) -> str:
    compact = [
        {
            "event_id": item["event_id"],
            "subject": item["subject"],
            "action": item["action"],
            "completion_cue": item["completion_cue"],
        }
        for item in events
    ]
    return (
        "Evaluate only the supplied short video range against the complete-video event "
        "queries below. For each event in the listed order, give match strength and "
        "completion evidence as integer 0,1,2,3. Use 0 when the event is absent; routine "
        "motion that does not match the event must be 0. Do not output timestamps, window "
        "IDs, comparisons, or crop information. Answer in English. Return only this "
        "four-value compact JSON array: [matches, completion, visible_subject, evidence]. "
        f"matches and completion must each be an array of exactly {len(events)} integers; "
        "visible_subject and evidence must be short strings. Do not output key names, "
        "markdown, or explanation. Events: "
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    )
