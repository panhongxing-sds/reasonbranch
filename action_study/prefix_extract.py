"""Simplified prefix extraction for action-matching study."""

from __future__ import annotations

import re
from dataclasses import dataclass

MARKER_RE = re.compile(r"(^|[\n\.!?]\s+|\s)(Wait|But)\b", re.MULTILINE)
# Qwen3.5 reasoning tags
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
THINKING_BLOCK_RE = re.compile(
    re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE),
    re.DOTALL | re.IGNORECASE,
)
MIN_MEANINGFUL_CHARS = 20
MARKER_EVENT_MERGE_CHARS = 200


@dataclass
class StudyPrefix:
    prefix_id: str
    prefix_type: str
    token_index: int
    step_index: int
    prefix_text: str
    reasoning_prefix: str
    previous_checkpoint: str
    local_window_before: str
    local_window_after: str
    reasoning_progress: float | None
    char_pos: int


def _strip_thinking(text: str) -> str:
    return THINKING_BLOCK_RE.sub("", text)


def _meaningful_tail(reasoning: str) -> str:
    tail = _strip_thinking(reasoning).strip()
    return re.sub(r"\s+", " ", tail)


def _paragraph_positions(text: str) -> list[int]:
    out: list[int] = []
    start = 0
    while True:
        idx = text.find("\n\n", start)
        if idx < 0:
            break
        out.append(idx + 2)
        start = idx + 2
    return out


def _uniform_pick(positions: list[int], k: int) -> list[int]:
    if not positions or k <= 0:
        return []
    if len(positions) <= k:
        return positions
    step = len(positions) / k
    return [positions[int(i * step)] for i in range(k)]


def _previous_checkpoint(reasoning: str, char_pos: int) -> str:
    before = reasoning[:char_pos]
    if len(before) < 2:
        return ""
    search_end = max(0, len(before) - 2)
    idx = before.rfind("\n\n", 0, search_end)
    if idx < 0:
        return ""
    checkpoint = before[: idx + 2]
    if len(_meaningful_tail(checkpoint)) < MIN_MEANINGFUL_CHARS:
        return ""
    return checkpoint


def _make_prefix(
    reasoning: str,
    char_pos: int,
    prefix_type: str,
    problem_id: str,
    idx: int,
    *,
    trace_complete: bool,
) -> StudyPrefix | None:
    prefix_text = reasoning[:char_pos]
    if len(_meaningful_tail(prefix_text)) < MIN_MEANINGFUL_CHARS:
        return None

    before = reasoning[max(0, char_pos - 120) : char_pos]
    after = reasoning[char_pos : char_pos + 120]
    return StudyPrefix(
        prefix_id=f"{problem_id}_p{idx:02d}_{prefix_type.lower()}",
        prefix_type=prefix_type,
        token_index=-1,
        step_index=prefix_text.count("\n\n"),
        prefix_text=prefix_text,
        reasoning_prefix=prefix_text,
        previous_checkpoint=_previous_checkpoint(reasoning, char_pos),
        local_window_before=before,
        local_window_after=after,
        reasoning_progress=(char_pos / max(len(reasoning), 1)) if trace_complete else None,
        char_pos=char_pos,
    )


def _cluster_marker_events(hits: list[tuple[str, int, int]]) -> list[list[tuple[str, int, int]]]:
    if not hits:
        return []
    clusters: list[list[tuple[str, int, int]]] = [[hits[0]]]
    for marker, start, end in hits[1:]:
        prev_end = clusters[-1][-1][2]
        if start - prev_end <= MARKER_EVENT_MERGE_CHARS:
            clusters[-1].append((marker, start, end))
        else:
            clusters.append([(marker, start, end)])
    return clusters


def extract_study_prefixes(
    reasoning: str,
    problem_id: str,
    *,
    max_markers: int = 3,
    max_paragraphs: int = 5,
    trace_complete: bool = True,
) -> list[StudyPrefix]:
    seen: set[tuple[str, int]] = set()
    out: list[StudyPrefix] = []
    counter = 0

    def add(char_pos: int, ptype: str) -> None:
        nonlocal counter
        key = (ptype, char_pos)
        if key in seen:
            return
        seen.add(key)
        pfx = _make_prefix(reasoning, char_pos, ptype, problem_id, counter, trace_complete=trace_complete)
        if pfx is None:
            return
        out.append(pfx)
        counter += 1

    for pos in _uniform_pick(_paragraph_positions(reasoning), max_paragraphs):
        add(pos, "PARAGRAPH_END")

    marker_hits: list[tuple[str, int, int]] = []
    for m in MARKER_RE.finditer(reasoning):
        marker_hits.append((m.group(2), m.start(2), m.end(2)))

    for event_idx, cluster in enumerate(_cluster_marker_events(marker_hits)):
        if event_idx >= max_markers:
            break
        before_pos = min(start for _, start, _ in cluster)
        after_pos = max(end for _, _, end in cluster)
        markers = {m for m, _, _ in cluster}
        if len(markers) == 1:
            tag = next(iter(markers)).upper()
        else:
            tag = "MIXED"
        add(before_pos, f"{tag}_BEFORE")
        add(after_pos, f"{tag}_AFTER")

    return out
