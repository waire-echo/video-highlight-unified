"""Versioned prompt builders, extracted without changing emitted text."""
from __future__ import annotations
import json
from typing import Sequence, Mapping


def build_generic_highlight_prompt() -> str:
    """Return the strict, query-free REZE prompt used by this competition.

    The original REZE task verifies a supplied textual action query.  This
    competition does not provide such a query, so this MVP asks only whether
    the supplied clip itself contains a clear, active and highlight-worthy
    event.  Keeping the answer contract at the end makes the next-token
    distribution directly interpretable.
    """

    return (
        "Judge only the supplied video clip. Does it contain a clearly visible, "
        "active and meaningful action or interaction, including a decisive "
        "moment or a clearly completed action, that is worth retaining as a "
        "video highlight? Do not infer anything before or after this clip. "
        "Static setup, ordinary waiting, empty scenery, title cards and end "
        "cards are not highlights. Answer exactly one word: Yes or No."
    )
