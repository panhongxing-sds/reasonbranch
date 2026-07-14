"""Extract the next complete substantive reasoning step from a continuation."""

from __future__ import annotations

import re
from typing import Any

from reasoning_branch_dataset.action_study.step_admission import classify_step_quality

STEP_END_RE = re.compile(r"<\s*STEP_END\s*>", re.IGNORECASE)
_THINK_OPEN = "\x3cthink\x3e"
_THINK_CLOSE = "\x3c/think\x3e"
THINK_BLOCK_RE = re.compile(
    re.escape(_THINK_OPEN) + r"[\s\S]*?" + re.escape(_THINK_CLOSE),
    re.IGNORECASE,
)
THINK_OPEN_RE = re.compile(re.escape(_THINK_OPEN), re.IGNORECASE)


def strip_model_thinking(text: str) -> str:
    """Drop R1 / redacted thinking wrappers; prefer visible text after closing think."""
    text = (text or "").strip()
    if not text:
        return ""

    after = re.search(re.escape(_THINK_CLOSE) + r"\s*", text, flags=re.IGNORECASE)
    if after:
        tail = text[after.end() :].strip()
        if tail:
            return tail

    inner = re.search(
        re.escape(_THINK_OPEN) + r"([\s\S]*?)" + re.escape(_THINK_CLOSE),
        text,
        flags=re.IGNORECASE,
    )
    if inner and inner.group(1).strip():
        return inner.group(1).strip()

    if THINK_OPEN_RE.search(text):
        opened = re.sub("^" + re.escape(_THINK_OPEN) + r"\s*", "", text, flags=re.IGNORECASE).strip()
        return opened

    text = THINK_BLOCK_RE.sub("", text)
    text = re.sub(r"</?redacted_thinking>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _strip_step_end(text: str) -> str:
    m = STEP_END_RE.search(text or "")
    if m:
        return text[: m.start()].rstrip()
    return (text or "").strip()


def split_step_blocks(text: str) -> list[str]:
    """Split continuation into paragraph blocks (future: also honor <STEP_END>)."""
    text = _strip_step_end(text)
    if not text:
        return []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


def extract_next_substantive_step(
    continuation: str,
    *,
    question: str = "",
    min_substantive_chars: int = 40,
    max_blocks: int = 4,
) -> dict[str, Any]:
    """
  From a continuation after prefix, take the next complete substantive step.

  Operational rule (v1): accumulate ``\\n\\n`` blocks until admission passes.
  Heading-only first blocks are merged with following blocks when possible.
  """
    blocks = split_step_blocks(continuation)
    if not blocks:
        qual = classify_step_quality("", question=question, min_substantive_chars=min_substantive_chars)
        return {
            "candidate_step": "",
            "step_blocks_used": 0,
            "step_boundary": "empty",
            **qual,
        }

    accumulated = ""
    used = 0
    for i, block in enumerate(blocks[:max_blocks]):
        used = i + 1
        accumulated = block if not accumulated else accumulated + "\n\n" + block
        qual = classify_step_quality(
            accumulated,
            question=question,
            min_substantive_chars=min_substantive_chars,
        )
        if qual["eligible_for_oracle"]:
            return {
                "candidate_step": qual["cleaned_step"] or accumulated.strip(),
                "step_blocks_used": used,
                "step_boundary": "substantive" if used == 1 else f"merged_{used}_blocks",
                **qual,
            }

    qual = classify_step_quality(
        accumulated,
        question=question,
        min_substantive_chars=min_substantive_chars,
    )
    return {
        "candidate_step": qual["cleaned_step"] or accumulated.strip(),
        "step_blocks_used": used,
        "step_boundary": "best_effort",
        **qual,
    }


def extract_candidate_bundle(
    *,
    question: str,
    continue_continuation: str,
    branch_continuations: list[str],
) -> dict[str, Any]:
    """Extract greedy + 4 branch next steps with admission metadata."""
    keys = ["candidate_g", "candidate_b1", "candidate_b2", "candidate_b3", "candidate_b4"]
    conts = [continue_continuation] + list(branch_continuations[:4])
    details: list[dict[str, Any]] = []
    for key, cont in zip(keys, conts):
        ext = extract_next_substantive_step(cont, question=question)
        details.append(
            {
                "candidate": key,
                "continuation": cont,
                "candidate_step": ext["candidate_step"],
                "step_quality": ext["step_quality"],
                "eligible_for_oracle": ext["eligible_for_oracle"],
                "step_blocks_used": ext["step_blocks_used"],
                "step_boundary": ext["step_boundary"],
                "step_chars": len(ext["candidate_step"] or ""),
            }
        )
    eligible = [d["eligible_for_oracle"] for d in details]
    return {
        "candidate_details": details,
        "n_complete_substantive": sum(1 for e in eligible if e),
        "all_complete": all(eligible) and len(eligible) == 5,
        "oracle_eligible": all(eligible) and len(eligible) == 5,
    }


def extract_handoff_step(
    continuation: str,
    *,
    question: str = "",
    min_substantive_chars: int = 15,
) -> str:
    """Lenient next-step extraction for target handoff (must be non-empty when possible)."""
    visible = strip_model_thinking(continuation)
    if not visible:
        return ""

    ext = extract_next_substantive_step(
        visible,
        question=question,
        min_substantive_chars=min_substantive_chars,
        max_blocks=6,
    )
    step = (ext.get("candidate_step") or "").strip()
    if step:
        return step

    blocks = split_step_blocks(visible)
    if blocks:
        acc = blocks[0]
        for block in blocks[1:4]:
            if len(acc) >= min_substantive_chars:
                break
            acc = acc + "\n\n" + block
        acc = acc.strip()
        if acc:
            return acc

    cleaned = visible.strip()
    if cleaned:
        return cleaned[:2000]
    return ""
