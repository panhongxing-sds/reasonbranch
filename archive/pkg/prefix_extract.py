"""Extract key prefixes from reasoning traces."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

MARKER_RE = re.compile(r"(^|[\n\.]\s+)(Wait|But)\b", re.MULTILINE)


@dataclass
class PrefixCandidate:
    prefix_type: str
    token_index: int
    step_index: int
    prefix_text: str
    local_window_before: str
    local_window_after: str
    reasoning_progress: float


def _paragraph_boundaries(text: str) -> list[int]:
    positions = []
    start = 0
    while True:
        idx = text.find("\n\n", start)
        if idx < 0:
            break
        positions.append(idx + 2)
        start = idx + 2
    return positions


def _char_to_token_index(text: str, char_pos: int, token_char_spans: list[tuple[int, int]]) -> int:
    for i, (s, e) in enumerate(token_char_spans):
        if s <= char_pos < e:
            return i
        if char_pos < s:
            return max(0, i - 1)
    return max(0, len(token_char_spans) - 1)


def build_token_char_spans(token_texts: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for tok in token_texts:
        start = pos
        pos += len(tok)
        spans.append((start, pos))
    return spans


def _slice_windows(text: str, char_pos: int, window: int = 120) -> tuple[str, str]:
    before = text[max(0, char_pos - window) : char_pos]
    after = text[char_pos : char_pos + window]
    return before, after


def _prefix_at_char(
    text: str,
    char_pos: int,
    prefix_type: str,
    token_char_spans: list[tuple[int, int]],
    token_texts: list[str],
) -> PrefixCandidate:
    token_index = _char_to_token_index(text, char_pos, token_char_spans)
    prefix_text = text[:char_pos]
    before, after = _slice_windows(text, char_pos)
    progress = char_pos / max(len(text), 1)
    step_index = prefix_text.count("\n\n")
    return PrefixCandidate(
        prefix_type=prefix_type,
        token_index=token_index,
        step_index=step_index,
        prefix_text=prefix_text,
        local_window_before=before,
        local_window_after=after,
        reasoning_progress=progress,
    )


def prefix_at_char(
    text: str,
    char_pos: int,
    prefix_type: str,
    token_char_spans: list[tuple[int, int]],
    token_texts: list[str],
) -> PrefixCandidate:
    return _prefix_at_char(text, char_pos, prefix_type, token_char_spans, token_texts)


def extract_entropy_spike_prefixes(
    reasoning_text: str,
    token_texts: list[str],
    token_trace: list[dict],
    *,
    max_spikes: int = 3,
) -> list[PrefixCandidate]:
    if not token_trace:
        return []
    entropies = [t.get("entropy", 0.0) for t in token_trace]
    if not entropies:
        return []
    mean_e = sum(entropies) / len(entropies)
    std_e = (sum((e - mean_e) ** 2 for e in entropies) / max(len(entropies), 1)) ** 0.5
    threshold = mean_e + std_e
    token_char_spans = build_token_char_spans(token_texts)
    spike_positions = sorted(
        [i for i, e in enumerate(entropies) if e >= threshold],
        key=lambda i: entropies[i],
        reverse=True,
    )[:max_spikes]
    out: list[PrefixCandidate] = []
    for pos in spike_positions:
        if pos >= len(token_char_spans):
            continue
        char_pos = token_char_spans[pos][1]
        out.append(prefix_at_char(reasoning_text, char_pos, "ENTROPY_SPIKE", token_char_spans, token_texts))
    return out


def extract_prefix_pool(
    reasoning_text: str,
    token_texts: list[str],
    *,
    max_paragraph: int = 5,
    max_marker: int = 3,
    max_random: int = 2,
    seed: int = 0,
) -> list[PrefixCandidate]:
    token_char_spans = build_token_char_spans(token_texts)
    out: list[PrefixCandidate] = []
    seen: set[tuple[str, int]] = set()

    def add(candidate: PrefixCandidate) -> None:
        key = (candidate.prefix_type, candidate.token_index)
        if key in seen:
            return
        seen.add(key)
        out.append(candidate)

    # Paragraph boundaries
    para_positions = _paragraph_boundaries(reasoning_text)
    for char_pos in para_positions[:max_paragraph]:
        add(
            _prefix_at_char(
                reasoning_text,
                char_pos,
                "PARAGRAPH_END",
                token_char_spans,
                token_texts,
            )
        )

    # Wait / But markers
    marker_hits: list[tuple[str, int, int]] = []
    for m in MARKER_RE.finditer(reasoning_text):
        marker = m.group(2)
        marker_start = m.start(2)
        marker_end = m.end(2)
        marker_hits.append((marker, marker_start, marker_end))

    marker_hits = marker_hits[:max_marker]
    for marker, marker_start, marker_end in marker_hits:
        if marker == "Wait":
            add(
                _prefix_at_char(
                    reasoning_text,
                    marker_start,
                    "WAIT_BEFORE",
                    token_char_spans,
                    token_texts,
                )
            )
            add(
                _prefix_at_char(
                    reasoning_text,
                    marker_end,
                    "WAIT_AFTER",
                    token_char_spans,
                    token_texts,
                )
            )
        elif marker == "But":
            add(
                _prefix_at_char(
                    reasoning_text,
                    marker_start,
                    "BUT_BEFORE",
                    token_char_spans,
                    token_texts,
                )
            )
            add(
                _prefix_at_char(
                    reasoning_text,
                    marker_end,
                    "BUT_AFTER",
                    token_char_spans,
                    token_texts,
                )
            )

    # Random paragraph controls (exclude marker-adjacent)
    rng = random.Random(seed)
    if para_positions and max_random > 0:
        candidates = [p for p in para_positions if all(abs(p - h[1]) > 20 for h in marker_hits)]
        rng.shuffle(candidates)
        for char_pos in candidates[:max_random]:
            add(
                _prefix_at_char(
                    reasoning_text,
                    char_pos,
                    "RANDOM",
                    token_char_spans,
                    token_texts,
                )
            )

    return out


extract_prefixes = extract_prefix_pool
