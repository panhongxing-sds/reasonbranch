"""Detect DeepScaler (and similar) problems that require missing visual input."""

from __future__ import annotations

import re

# Case-insensitive patterns for text-only runs without figure assets.
_VISUAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bfigure\b",
        r"\bdiagram\b",
        r"\bgraph\b(?!\s+of\s+the\s+function)",  # avoid "graph of the function" algebra
        r"shown\s+below",
        r"shown\s+in\s+the\s+figure",
        r"in\s+the\s+figure",
        r"adjoining\s+figure",
        r"solid\s+lines?",
        r"dashed\s+lines?",
        r"positions?\s+indicated",
        r"not\s+drawn\s+to\s+scale",
        r"\[asy\]",
        r"illustrated\s+below",
        r"see\s+the\s+(?:figure|diagram)",
    )
)


def assess_visual_input(question: str, *, has_image: bool = False) -> dict:
    """Return metadata for whether a text-only prompt is complete."""
    requires = bool(has_image) or any(p.search(question) for p in _VISUAL_PATTERNS)
    input_complete = not requires or has_image
    return {
        "requires_visual_input": requires,
        "input_complete": input_complete,
        "exclusion_reason": None if input_complete else "missing_figure",
    }
