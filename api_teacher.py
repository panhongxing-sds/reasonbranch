"""Offline strong-model API teacher for prefix curation and annotation."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMPT_VERSION = "v1"

PREFIX_ONLY_SYSTEM = """You are an offline reasoning dynamics annotator for math CoT traces.
You only see the question and the current prefix (no future tokens).
Judge whether this prefix is a decision point worth branching, or shows rollback risk.
Respond with JSON only."""

PREFIX_ONLY_USER = """Question:
{question}

Current reasoning prefix (ends at cut point):
{prefix_tail}

Prefix type: {prefix_type}
Reasoning progress: {progress:.0%}

Return JSON:
{{
  "branch_worthiness": <float 0-1>,
  "rollback_risk": <float 0-1>,
  "decision_point_score": <float 0-1>,
  "ambiguity_score": <float 0-1>,
  "reasoning_state": "<e.g. decision_point|routine_calc|self_correction|dead_end>",
  "ambiguity_type": "<e.g. strategy_choice|calculation|none>",
  "suggested_action": "<continue|branch|rollback|none>",
  "explanation": "<one sentence>"
}}"""

TRACE_AWARE_SYSTEM = """You are an offline trace analyst for math reasoning.
You may use the full trace and gold answer for analysis only (not for online inference).
Respond with JSON only."""

TRACE_AWARE_USER = """Question:
{question}

Gold answer:
{gold_answer}

Full reasoning trace:
{full_reasoning}

Candidate prefix ends at:
{prefix_tail}

Return JSON:
{{
  "prefix_role": "<e.g. before_self_correction|routine_step|wrong_path_start>",
  "future_correction_happens": <bool>,
  "current_path_error_type": "<string or none>",
  "branch_may_help": <bool>,
  "self_correction_likelihood": <float 0-1>,
  "rollback_checkpoint": "<string or none>",
  "explanation": "<one sentence>"
}}"""

BRANCH_RANK_USER = """Question:
{question}

Gold answer:
{gold_answer}

Reasoning prefix:
{prefix_tail}

Candidate branches (numbered):
{branches}

Rank branches by likelihood of leading to a correct solution.
Return JSON:
{{
  "ranking": [<branch_ids in best-first order>],
  "best_branch": "<branch_id>",
  "qualities": {{"<branch_id>": <float 0-1>}},
  "reasons": {{"<branch_id>": "<short reason>"}}
}}"""


@dataclass
class TeacherConfig:
    base_url: str = "https://endpoint.greatrouter.com"
    model: str = "gpt-5.5"
    api_key: str = ""
    temperature: float = 0.2
    max_retries: int = 3
    timeout_sec: float = 60.0
    cache_path: Path | None = None
    enabled: bool = True

    @classmethod
    def from_env(cls, cache_path: Path | None = None) -> "TeacherConfig":
        key = (
            os.environ.get("TEACHER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GREATROUTER_API_KEY")
            or ""
        )
        base = os.environ.get("TEACHER_BASE_URL", "https://endpoint.greatrouter.com")
        model = os.environ.get("TEACHER_MODEL", "gpt-5.5")
        enabled = os.environ.get("TEACHER_API_DISABLED", "").lower() not in {"1", "true", "yes"}
        return cls(base_url=base, model=model, api_key=key, cache_path=cache_path, enabled=enabled and bool(key))


def _parse_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
        raise


@dataclass
class TeacherClient:
    cfg: TeacherConfig
    _cache: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cfg.cache_path and self.cfg.cache_path.exists():
            for line in self.cfg.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["cache_key"]] = row

    def _save_cache(self, cache_key: str, payload: dict) -> None:
        self._cache[cache_key] = payload
        if self.cfg.cache_path:
            self.cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.cache_path.open("a") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _chat(self, system: str, user: str, cache_key: str) -> dict[str, Any]:
        if cache_key in self._cache:
            return self._cache[cache_key]["response"]

        if not self.cfg.enabled:
            return self._heuristic_fallback(user)

        from openai import OpenAI

        client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    timeout=self.cfg.timeout_sec,
                )
                content = resp.choices[0].message.content or "{}"
                parsed = _parse_json_blob(content)
                record = {
                    "cache_key": cache_key,
                    "response": parsed,
                    "raw": content,
                    "api_model": self.cfg.model,
                    "temperature": self.cfg.temperature,
                    "prompt_version": PROMPT_VERSION,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._save_cache(cache_key, record)
                return parsed
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Teacher API failed after retries: {last_err}")

    @staticmethod
    def _heuristic_fallback(user: str) -> dict[str, Any]:
        low = user.lower()
        branch = 0.35
        rollback = 0.2
        decision = 0.3
        if any(k in low for k in ("wait", "but", "alternatively", "let me check")):
            branch, decision = 0.72, 0.68
        if "prefix type: entropy_spike" in low:
            branch, decision = 0.55, 0.5
        if "prefix type: random" in low:
            branch, decision = 0.25, 0.2
        return {
            "branch_worthiness": branch,
            "rollback_risk": rollback,
            "decision_point_score": decision,
            "ambiguity_score": branch * 0.8,
            "reasoning_state": "heuristic_fallback",
            "ambiguity_type": "unknown",
            "suggested_action": "branch" if branch > 0.5 else "continue",
            "explanation": "heuristic fallback (no API key)",
        }

    def annotate_prefix_only(
        self,
        *,
        prefix_id: str,
        question: str,
        prefix_tail: str,
        prefix_type: str,
        progress: float,
    ) -> dict[str, Any]:
        user = PREFIX_ONLY_USER.format(
            question=question,
            prefix_tail=prefix_tail[-1500:],
            prefix_type=prefix_type,
            progress=progress,
        )
        cache_key = f"prefix_only:{prefix_id}:{PROMPT_VERSION}"
        result = self._chat(PREFIX_ONLY_SYSTEM, user, cache_key)
        return {
            "prefix_id": prefix_id,
            "api_model": self.cfg.model,
            "api_prompt_version": PROMPT_VERSION,
            "annotation_mode": "prefix_only",
            "branch_worthiness": float(result.get("branch_worthiness", 0.0)),
            "rollback_risk": float(result.get("rollback_risk", 0.0)),
            "decision_point_score": float(result.get("decision_point_score", 0.0)),
            "ambiguity_score": float(result.get("ambiguity_score", 0.0)),
            "self_correction_likelihood": float(result.get("self_correction_likelihood", 0.0)),
            "reasoning_state": str(result.get("reasoning_state", "")),
            "ambiguity_type": str(result.get("ambiguity_type", "")),
            "suggested_action": str(result.get("suggested_action", "")),
            "api_explanation": str(result.get("explanation", "")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": self.cfg.temperature,
        }

    def annotate_trace_aware(
        self,
        *,
        prefix_id: str,
        question: str,
        gold_answer: str,
        full_reasoning: str,
        prefix_tail: str,
    ) -> dict[str, Any]:
        user = TRACE_AWARE_USER.format(
            question=question,
            gold_answer=gold_answer,
            full_reasoning=full_reasoning[-3000:],
            prefix_tail=prefix_tail[-1500:],
        )
        cache_key = f"trace_aware:{prefix_id}:{PROMPT_VERSION}"
        result = self._chat(TRACE_AWARE_SYSTEM, user, cache_key)
        return {
            "prefix_id": prefix_id,
            "api_model": self.cfg.model,
            "api_prompt_version": PROMPT_VERSION,
            "annotation_mode": "trace_aware",
            "prefix_role": str(result.get("prefix_role", "")),
            "future_correction_happens": int(bool(result.get("future_correction_happens", False))),
            "current_path_error_type": str(result.get("current_path_error_type", "")),
            "branch_may_help": int(bool(result.get("branch_may_help", False))),
            "self_correction_likelihood": float(result.get("self_correction_likelihood", 0.0)),
            "rollback_checkpoint": str(result.get("rollback_checkpoint", "")),
            "api_explanation": str(result.get("explanation", "")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": self.cfg.temperature,
        }

    def rank_branches(
        self,
        *,
        prefix_id: str,
        question: str,
        gold_answer: str,
        prefix_tail: str,
        branches: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        if not branches:
            return []
        branch_text = "\n".join(f"{b['branch_id']}: {b['branch_text'][:400]}" for b in branches)
        user = BRANCH_RANK_USER.format(
            question=question,
            gold_answer=gold_answer,
            prefix_tail=prefix_tail[-1200:],
            branches=branch_text,
        )
        cache_key = f"branch_rank:{prefix_id}:{PROMPT_VERSION}"
        result = self._chat(PREFIX_ONLY_SYSTEM, user, cache_key)
        qualities = result.get("qualities", {}) or {}
        reasons = result.get("reasons", {}) or {}
        ranking = result.get("ranking", [])
        rows = []
        for rank_idx, bid in enumerate(ranking):
            rows.append(
                {
                    "prefix_id": prefix_id,
                    "branch_id": str(bid),
                    "api_model": self.cfg.model,
                    "api_prompt_version": PROMPT_VERSION,
                    "api_branch_quality": float(qualities.get(bid, max(0.0, 1.0 - 0.1 * rank_idx))),
                    "api_rank": rank_idx + 1,
                    "api_error_type": "",
                    "api_correctness_judgment": reasons.get(bid, ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "temperature": self.cfg.temperature,
                }
            )
        return rows
