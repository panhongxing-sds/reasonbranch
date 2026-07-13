"""Mixed prefix pool construction and API-guided selection."""

from __future__ import annotations

import random

from reasoning_branch_dataset.prefix_extract import (
    PrefixCandidate,
    extract_entropy_spike_prefixes,
    extract_prefix_pool,
)


def extract_candidate_pool(
    reasoning_text: str,
    token_texts: list[str],
    token_trace: list[dict],
    *,
    seed: int = 0,
) -> list[PrefixCandidate]:
    base = extract_prefix_pool(
        reasoning_text,
        token_texts,
        max_paragraph=50,
        max_marker=20,
        max_random=5,
        seed=seed,
    )
    spikes = extract_entropy_spike_prefixes(reasoning_text, token_texts, token_trace, max_spikes=3)
    seen = {(c.prefix_type, c.token_index) for c in base}
    out = list(base)
    for cand in spikes:
        key = (cand.prefix_type, cand.token_index)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


from dataclasses import dataclass


@dataclass
class ScoredPrefix:
    candidate: PrefixCandidate
    prefix_id: str
    entropy: float = 0.0
    margin: float = 0.0
    branch_worthiness: float = 0.0
    rollback_risk: float = 0.0
    decision_point_score: float = 0.0
    selected: bool = False
    selection_reason: str = ""


def select_prefixes_for_rollout(
    scored: list[ScoredPrefix],
    *,
    top_branch: int = 2,
    top_rollback: int = 1,
    max_wait_but: int = 4,
    max_paragraph: int = 3,
    n_random: int = 1,
    n_low_control: int = 1,
    rng: random.Random | None = None,
) -> list[ScoredPrefix]:
    rng = rng or random.Random(0)
    if not scored:
        return []

    selected_ids: set[str] = set()

    def pick(items: list[ScoredPrefix], reason: str, limit: int) -> None:
        for sp in items[:limit]:
            if sp.prefix_id not in selected_ids:
                sp.selected = True
                sp.selection_reason = reason
                selected_ids.add(sp.prefix_id)

    markers = [s for s in scored if s.candidate.prefix_type in {
        "WAIT_BEFORE", "WAIT_AFTER", "BUT_BEFORE", "BUT_AFTER"
    }]
    pick(markers, "wait_but_marker", max_wait_but)

    by_branch = sorted(scored, key=lambda s: s.branch_worthiness, reverse=True)
    pick(by_branch, "api_top_branch", top_branch)

    by_rollback = sorted(scored, key=lambda s: s.rollback_risk, reverse=True)
    pick(by_rollback, "api_top_rollback", top_rollback)

    structural = [s for s in scored if s.candidate.prefix_type in {"PARAGRAPH_END", "ENTROPY_SPIKE"}]
    structural = sorted(structural, key=lambda s: (s.branch_worthiness + s.entropy), reverse=True)
    pick(structural, "paragraph_or_entropy", max_paragraph)

    randoms = [s for s in scored if s.candidate.prefix_type == "RANDOM" and s.prefix_id not in selected_ids]
    rng.shuffle(randoms)
    pick(randoms, "random_control", n_random)

    low = sorted(scored, key=lambda s: s.branch_worthiness)
    lows = [s for s in low if s.prefix_id not in selected_ids]
    pick(lows, "low_score_control", n_low_control)

    return [s for s in scored if s.selected]
