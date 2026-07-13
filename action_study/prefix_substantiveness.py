"""Detect whether a reasoning prefix has made substantive commitments."""

from __future__ import annotations

import re

NO_COMMITMENT_PATTERNS = [
    re.compile(r"^\s*(let'?s|we will|i will|now|next)\s+(solve|start|begin|work|proceed)", re.I),
    re.compile(r"^\s*(the\s+)?(goal|plan|approach|strategy)\s+(is|will)", re.I),
    re.compile(r"^\s*###?\s*step\s*\d*\s*[:.]?\s*$", re.I),
    re.compile(r"^\s*(problem|question)\s*[:.]?\s*$", re.I),
]

MATH_SIGNAL = re.compile(
    r"(\\\[|\\\(|\\frac|\\sum|\\int|\\boxed|=|\d+\s*[+\-*/^=]|"
    r"assume|let\s+\$|substitut|equation|theorem|lemma|therefore|hence|"
    r"we have|it follows|implies|contradiction|mod\s+\d|gcd|lcm)",
    re.I,
)


def reasoning_body(prefix_text: str) -> str:
    if "</think>" in prefix_text:
        return prefix_text.split("</think>", 1)[1].strip()
    if "Problem:" in prefix_text:
        tail = prefix_text.split("Problem:", 1)[1]
        return tail[tail.find("\n") + 1 :].strip() if "\n" in tail else tail.strip()
    return prefix_text.strip()


def prefix_substantiveness(prefix_text: str, *, api_label: str | None = None) -> str:
    """Return NO_COMMITMENT | SUBSTANTIVE."""
    if api_label in {"NO_COMMITMENT", "SUBSTANTIVE"}:
        return api_label

    body = reasoning_body(prefix_text)
    if not body or len(body) < 20:
        return "NO_COMMITMENT"

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return "NO_COMMITMENT"

    substantive_lines = 0
    for ln in lines:
        if any(p.search(ln) for p in NO_COMMITMENT_PATTERNS) and not MATH_SIGNAL.search(ln):
            continue
        if MATH_SIGNAL.search(ln) or len(ln) > 80:
            substantive_lines += 1

    if substantive_lines == 0:
        return "NO_COMMITMENT"
    if substantive_lines == 1 and not MATH_SIGNAL.search(body):
        # single short intro sentence without math
        first = lines[0].lower()
        if any(
            k in first
            for k in ("solve", "find", "compute", "determine", "approach", "strategy", "step")
        ) and not MATH_SIGNAL.search(body):
            return "NO_COMMITMENT"
    return "SUBSTANTIVE"
