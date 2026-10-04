"""Qwen 3B action-description contract for the temporal front end.

This module deliberately does not predict timestamps.  It turns a complete
video into one to three visible action descriptions that a temporal-grounding
backend such as TimeLens can use as queries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


ALLOWED_SUBJECTS = {
    "person",
    "animal",
    "sports_ball",
    "vehicle",
    "sports_equipment",
    "other",
}


@dataclass(frozen=True)
class HighlightAction:
    description: str
    main_subject: str = "other"


@dataclass
class ActionDescriptionResult:
    actions: List[HighlightAction]
    valid: bool
    raw: str = ""
    errors: List[str] = field(default_factory=list)


def build_action_description_prompt(max_actions: int = 3) -> str:
    return (
        "Watch the complete video and describe the most highlight-worthy visible actions.\n"
        "Do not output timestamps, frame numbers, window IDs, scores, or crop locations.\n"
        "Prefer a complete action, interaction, reveal, decisive change, or visible result.\n"
        "Reject intros, outros, title cards, static setup, ordinary walking, subtitles, and "
        "camera shake unless they are part of the key event.\n"
        f"Return between 1 and {max(1, int(max_actions))} distinct actions.\n"
        "Return exactly one complete JSON object whose only key is highlight_actions.\n"
        "highlight_actions must be a list of objects. Each object has exactly description "
        "and main_subject. main_subject is one of person, animal, sports_ball, vehicle, "
        "sports_equipment, other.\n"
        "No markdown and no explanation."
    )


def _load_complete_json(text: str):
    payload = str(text or "").strip()
    if payload.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", payload, re.I | re.S)
        if not match:
            raise ValueError("incomplete markdown JSON fence")
        payload = match.group(1).strip()
    return json.loads(payload)


def parse_action_description_response(
    text: str, max_actions: int = 3
) -> ActionDescriptionResult:
    try:
        payload = _load_complete_json(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return ActionDescriptionResult([], False, str(text or ""), [f"invalid complete JSON: {exc}"])
    if not isinstance(payload, dict) or set(payload) != {"highlight_actions"}:
        return ActionDescriptionResult(
            [], False, str(text or ""), ["top-level object must contain only highlight_actions"]
        )
    items = payload.get("highlight_actions")
    limit = max(1, int(max_actions))
    if not isinstance(items, list) or not 1 <= len(items) <= limit:
        return ActionDescriptionResult(
            [], False, str(text or ""), [f"highlight_actions must contain 1 to {limit} items"]
        )

    actions: List[HighlightAction] = []
    errors: List[str] = []
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"description", "main_subject"}:
            errors.append(f"action {index} must contain exactly description and main_subject")
            continue
        description = item.get("description")
        subject = item.get("main_subject")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"action {index} description must be non-empty")
            continue
        if subject not in ALLOWED_SUBJECTS:
            errors.append(f"action {index} main_subject is unsupported")
            continue
        normalized = " ".join(description.split())
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        actions.append(HighlightAction(normalized, subject))
    if errors or not actions:
        return ActionDescriptionResult(actions, False, str(text or ""), errors or ["no valid actions"])
    return ActionDescriptionResult(actions, True, str(text or ""), [])


class QwenActionDescriber:
    """Adapter around an already-loaded local Qwen video generator."""

    def __init__(
        self,
        generator: Optional[Callable[[str, str], str]],
        max_actions: int = 3,
    ):
        self.generator = generator
        self.max_actions = max(1, int(max_actions))

    def describe(self, video_path: str) -> ActionDescriptionResult:
        if self.generator is None:
            return ActionDescriptionResult([], False, errors=["Qwen action generator is not configured"])
        prompt = build_action_description_prompt(self.max_actions)
        try:
            raw = str(self.generator(video_path, prompt))
        except Exception as exc:
            return ActionDescriptionResult(
                [], False, errors=[f"Qwen action generation failed: {type(exc).__name__}: {exc}"]
            )
        return parse_action_description_response(raw, self.max_actions)
