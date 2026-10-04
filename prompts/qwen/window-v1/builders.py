"""Versioned prompt builders, extracted without changing emitted text."""
from __future__ import annotations
import json
from typing import Sequence, Mapping
from video_highlight.window_scoring import _finite_float


def build_window_score_prompt(
    *,
    sample_fps: float = 2.0,
    include_rubric: bool = True,
    compact_output: bool = False,
) -> str:
    """Build the strict prompt used for each video time window.

    The prompt never asks Qwen to infer a timestamp.  The decoder supplies the
    range, and the model only describes evidence visible inside that range.
    """

    fps = _finite_float(sample_fps, "sample_fps")
    rubric = ""
    if include_rubric:
        rubric = (
            "Use this rubric for every integer score: 0=not present, 1=weak, "
            "2=moderate, 3=clear/strong. "
            "action=movement strength; completion=whether a key action/result "
            "is completed; interaction=interaction between visible subjects; "
            "event=an abrupt or decisive change; emotion=visible emotion or "
            "state change; visual_impact=visual salience; highlight=overall "
            "worth keeping as a sports/video highlight.\n"
        )
    # Do not put a concrete numeric JSON example in the prompt.  Small VLMs
    # frequently copy the example's all-zero (or all-75) values for every
    # window, which destroys the ranking signal.  A key/type list is enough to
    # constrain the output while forcing an independent judgement.
    if compact_output:
        key_contract = (
            "Return exactly one compact JSON array in this fixed order: "
            "[action, completion, interaction, event, emotion, visual_impact, "
            "highlight, main_subject, action_description, evidence]. The first "
            "seven values must be integer digits 0/1/2/3 and the final three "
            "values must be strings. Never output key names. CRITICAL: the first "
            "action value is the numeric action-intensity score; put verbs such "
            "as walking, running, or jumping only in action_description.\n"
        )
    else:
        key_contract = (
        "Required keys exactly: action, completion, interaction, event, emotion, "
        "visual_impact, highlight (each an independently chosen integer 0/1/2/3), "
        "main_subject (string), action_description (string), evidence (string).\n"
        "CRITICAL: action is the numeric action-intensity SCORE, never an action "
        "name or verb. For example, do not write action as walking, running, or "
        "jumping; put those words only in action_description. Before answering, "
        "verify that the values of all first seven keys are digits 0, 1, 2, or 3.\n"
        )
    return (
        "Evaluate only the currently supplied video time window. Do not infer or "
        "mention anything outside it. Do not output timestamps, frame numbers, "
        "window IDs, comparisons with other windows, or crop locations. Do not "
        "reward image quality or a static setup by itself. Base every score on "
        "visible action/event evidence.\n"
        f"The video is sampled at approximately {fps:g} frames per second for "
        "this evaluation.\n"
        + rubric
        + "Return exactly one complete JSON value with no markdown or explanation. "
        "Do not copy a template score: judge this "
        "window independently, and do not make every field zero unless the "
        "visual evidence truly supports that. All seven score values must be "
        "integers from 0 to 3; main_subject, action_description, and evidence "
        "must be non-empty strings.\n"
        "Use compact one-line JSON. Keep main_subject to 1-3 words and keep "
        "action_description and evidence to at most 10 words each.\n"
        + key_contract
    )
