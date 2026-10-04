"""Versioned prompt builders, extracted without changing emitted text."""
from __future__ import annotations
import json
from typing import Sequence, Mapping


def build_temporal_prompt(
    duration_sec: float,
    fps: float,
    detect_fps: float,
    max_segments: int,
    sample_timestamps: Sequence[float],
) -> str:
    timestamp_text = ", ".join(f"{value:.3f}s" for value in sample_timestamps)
    if not timestamp_text:
        timestamp_text = "unavailable"
    return (
        "You are performing temporal highlight localization for a sports-video editing task.\n\n"
        f"The video duration is {duration_sec:.3f} seconds.\n"
        f"The source video frame rate is {fps:.3f} FPS.\n"
        f"The video is provided at approximately {detect_fps:.3f} sampled frames per second.\n"
        "The sampled frame timestamps below are computed from original frame IDs divided by source FPS. "
        "For long videos the list may be uniformly shortened while retaining true timestamps.\n"
        f"Sampled timestamps: {timestamp_text}\n\n"
        f"Find one to {max(1, int(max_segments))} continuous highlight-worthy intervals.\n"
        "A highlight contains the most important, exciting, decisive, or visually meaningful action, "
        "such as a score, successful shot, finish, overtaking, save, collision, celebration, "
        "decisive movement, or peak action.\n\n"
        "Do not treat an intro, title card, subtitle, ordinary walking, passive audience shot, "
        "or camera shake by itself as a highlight.\n"
        "Every input video has at least one highlight-worthy moment for this task. If there is no "
        "score or obvious sports event, choose the strongest visible action, reveal, interaction, "
        "turn, jump, run, finish, reaction, or result in the video.\n\n"
        "Every start and end time must be measured in seconds from the beginning of the video.\n"
        f"Every interval must satisfy 0 <= start_sec < end_sec <= {duration_sec:.3f}.\n"
        "Do not invent times outside the video duration. Do not return zero-length intervals.\n"
        "Keep the complete action rather than a single instant, including roughly one second of "
        "useful context before and after when the video boundaries allow it.\n"
        "For a short video, stay within its actual duration.\n"
        "Do not copy generic template time ranges.\n"
        f"STRICT: return no more than {max(1, int(max_segments))} interval entries in total.\n"
        "Never enumerate every sampled frame or every second as separate intervals.\n"
        "Merge touching or adjacent intervals before answering. If the whole action is continuous, "
        "return one interval instead of many short intervals.\n"
        "Keep the complete response concise enough to fit within 100 output tokens.\n"
        "Return JSON only, with no Markdown and no explanation.\n"
        "Return one top-level JSON object. It must contain a segments array of numeric "
        "[start_sec, end_sec] pairs; do not quote the numbers. It must also contain "
        "a subject_hint string. The segments array may contain time pairs only; never put subject "
        "words, class labels, or action words inside segments. Put the entire visible-subject "
        "description only in the single subject_hint text string. Describe what downstream YOLO "
        "should keep using a concise phrase with appearance, class, and action. Include a "
        "recognizable class word such as "
        "person, dog, cat, animal, sports ball, car, vehicle, or bicycle. Avoid the generic word "
        "object when a more specific visible class is available."
    )


def build_temporal_retry_prompt(
    base_prompt: str, duration_sec: float, previous_response: str, validation_errors: Sequence[str]
) -> str:
    error_text = "; ".join(validation_errors) if validation_errors else "invalid temporal response"
    return (
        f"{base_prompt}\n\nYour previous response was invalid.\n"
        f"Video duration: {duration_sec:.3f} seconds.\n"
        f"Previous response:\n{previous_response}\n"
        f"Validation error: {error_text}\n"
        "Do not repeat a truncated per-second or per-frame list. Merge adjacent ranges and keep the "
        "complete answer concise. Remove every non-time entry from the segments array; all subject "
        "description words belong only in subject_hint.\n"
        f"Return corrected JSON only. Every interval must satisfy "
        f"0 <= start_sec < end_sec <= {duration_sec:.3f}."
    )
