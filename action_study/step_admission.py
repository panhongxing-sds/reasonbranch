"""Classify whether a extracted candidate step is eligible for utility oracle."""

from __future__ import annotations

import re
from typing import Any

THINK_TAG_RE = re.compile(r"</?redacted_thinking>", re.IGNORECASE)
HEADING_RE = re.compile(
    r"^(\*\*)?\s*Step\s*\d+\s*[:.)-]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
MATHISH_RE = re.compile(r"[$\\=+\-*/^_{}\[\]0-9]")

MARKER_ONLY_RE = re.compile(
    r"^(wait[,.!]?\s*)?(<think>\s*)?(let\s+me\s+think|let's\s+think|hmm|okay)[.!]?\s*$",
    re.IGNORECASE,
)

STEP_QUALITY_LABELS = (
    "COMPLETE_SUBSTANTIVE_STEP",
    "HEADING_ONLY",
    "THINK_TAG_ONLY",
    "MARKER_ONLY",
    "MALFORMED_INPUT",
    "TRUNCATED_STEP",
)


def _strip_think(text: str) -> str:
    return THINK_TAG_RE.sub("", text or "").strip()


def _question_malformed(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    # Repeated phrase blocks (e.g. "For x real ... For x real ...")
    words = q.split()
    if len(words) < 12:
        return False
    for win in (6, 8, 10):
        if len(words) < win * 3:
            continue
        for i in range(len(words) - win * 2):
            chunk = " ".join(words[i : i + win])
            if q.count(chunk) >= 3:
                return True
    return False


def classify_step_quality(
    step_text: str,
    *,
    question: str = "",
    min_substantive_chars: int = 40,
) -> dict[str, Any]:
    raw = (step_text or "").strip()
    if not raw:
        return {
            "step_quality": "THINK_TAG_ONLY",
            "eligible_for_oracle": False,
            "cleaned_step": "",
        }

    if re.fullmatch(r"<think>\s*</think>", raw, flags=re.IGNORECASE):
        return {
            "step_quality": "THINK_TAG_ONLY",
            "eligible_for_oracle": False,
            "cleaned_step": "",
        }

    cleaned = _strip_think(raw)

    if not cleaned or cleaned.lower() == "redacted_thinking":
        return {
            "step_quality": "THINK_TAG_ONLY",
            "eligible_for_oracle": False,
            "cleaned_step": cleaned,
        }

    if _question_malformed(question):
        return {
            "step_quality": "MALFORMED_INPUT",
            "eligible_for_oracle": False,
            "cleaned_step": cleaned,
        }

    if MARKER_ONLY_RE.match(cleaned) or (
        len(cleaned) < 30 and cleaned.lower().startswith("wait")
    ):
        return {
            "step_quality": "MARKER_ONLY",
            "eligible_for_oracle": False,
            "cleaned_step": cleaned,
        }

    body = cleaned
    m = HEADING_RE.match(cleaned)
    if m:
        body = (m.group(2) or "").strip()

    if len(body) < min_substantive_chars // 2 and not MATHISH_RE.search(body):
        if m and not body:
            quality = "HEADING_ONLY"
        elif len(cleaned) < 25:
            quality = "TRUNCATED_STEP"
        else:
            quality = "HEADING_ONLY"
        return {
            "step_quality": quality,
            "eligible_for_oracle": False,
            "cleaned_step": cleaned,
        }

    if len(cleaned) < min_substantive_chars and not MATHISH_RE.search(cleaned):
        return {
            "step_quality": "TRUNCATED_STEP",
            "eligible_for_oracle": False,
            "cleaned_step": cleaned,
        }

    return {
        "step_quality": "COMPLETE_SUBSTANTIVE_STEP",
        "eligible_for_oracle": True,
        "cleaned_step": cleaned,
    }


def annotate_candidate_details(
    details: list[dict[str, Any]],
    *,
    question: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in details:
        step = d.get("candidate_step") or ""
        qual = classify_step_quality(step, question=question)
        out.append({**d, **qual, "step_chars": len(step)})
    return out


def prefix_oracle_eligibility(annotated: list[dict[str, Any]]) -> dict[str, Any]:
    qualities = [a.get("step_quality", "UNKNOWN") for a in annotated]
    eligible = [a.get("eligible_for_oracle", False) for a in annotated]
    n_complete = sum(1 for e in eligible if e)
    return {
        "step_qualities": qualities,
        "n_complete_substantive": n_complete,
        "all_complete": n_complete == len(annotated) and len(annotated) > 0,
        "any_incomplete": n_complete < len(annotated),
        "oracle_eligible": n_complete == len(annotated) and len(annotated) >= 5,
    }
