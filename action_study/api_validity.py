"""Strong-model API for offline labeling: validity, clustering, review."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMPT_VERSION = "validity_v3"
CLUSTER_PROMPT_VERSION = "cluster_v2"
REVIEW_PROMPT_VERSION = "review_v1"

SYSTEM_VALIDITY = """You are an offline math reasoning verifier.
Judge whether the CURRENT reasoning prefix is still logically valid.
Also judge whether the prefix has made any substantive mathematical commitment.
Do NOT judge future steps. Respond with JSON only."""

USER_VALIDITY = """Question:
{question}

Gold answer:
{gold_answer}

Current reasoning prefix (ends at cut point; paragraphs separated by blank lines):
{prefix}

Tasks:
1. Has the first obvious logical/mathematical error already occurred in this prefix?
2. Has the model made a substantive reasoning commitment (formula, assumption, substitution, derivation)?
   Prefixes that only restate the problem or say "let's solve" are NO_COMMITMENT.

Return JSON:
{{
  "prefix_status": "VALID" | "INVALID" | "UNCLEAR" | "NO_COMMITMENT",
  "prefix_substantiveness": "SUBSTANTIVE" | "NO_COMMITMENT",
  "first_error_step": <null or integer step number starting at 1>,
  "error_type": "algebra" | "assumption" | "interpretation" | "arithmetic" | "none",
  "confidence": <float 0-1>,
  "explanation": "<one sentence>"
}}"""

SYSTEM_CLUSTER = """You cluster math reasoning next-steps by MATHEMATICAL STRATEGY.
Group steps that use the same core operation, key assumption, or solution approach.
Do NOT split by wording, titles, or phrasing alone.
Example: "change of variables", "index substitution", "let n=j+k" are ONE strategy.
Only assign different cluster ids when strategies genuinely differ (e.g. reindexing vs fixing outer index vs generating functions).
Respond with JSON only."""

USER_CLUSTER = """Question:
{question}

Reasoning prefix:
{prefix}

Sampled next steps (one per line):
{steps}

Assign each step an integer cluster id (0-based). Steps with the same mathematical strategy share an id.
Return JSON:
{{
  "clusters": [<int>, ...],
  "num_semantic_clusters": <int>,
  "multiple_genuine_strategies": <bool>,
  "strategy_descriptions": ["<short label per unique cluster>"]
}}"""

SYSTEM_REVIEW = """You review edge cases in math reasoning action-matching studies.
Respond with JSON only."""

USER_REVIEW = """Question:
{question}

Gold answer:
{gold_answer}

Prefix validity label: {prefix_status}
Diversity: {diversity_label}

Branch pass@4: {branch_pass}
Rollback pass@4: {rollback_pass}
Continue correct: {continue_correct}

Reason for review: {reason}

Re-assess prefix_status and briefly explain the apparent conflict.
Return JSON:
{{
  "prefix_status": "VALID" | "INVALID" | "UNCLEAR",
  "review_verdict": "<one sentence>",
  "confidence": <float 0-1>
}}"""


@dataclass
class ValidityClient:
    base_url: str = "https://endpoint.greatrouter.com"
    model: str = "gpt-5.5"
    api_key: str = ""
    temperature: float | None = None  # omit for gpt-5.5 / GreatRouter (custom temp → 400)
    cache_path: Path | None = None
    enabled: bool = True
    _cache: dict[str, dict] = field(default_factory=dict)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls, cache_path: Path | None = None) -> "ValidityClient":
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

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["cache_key"]] = row

    def _save_cache(self, cache_key: str, record: dict) -> None:
        with self._cache_lock:
            self._cache[cache_key] = record
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group())
            raise

    def _chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "timeout": 120,
        }
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or "{}"

    def _api_call(self, cache_key: str, system: str, user: str, *, temperature: float | None = None) -> dict:
        if not self._cache and self.cache_path:
            with self._cache_lock:
                if not self._cache:
                    self._load_cache()
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]["response"]

        if not self.enabled:
            return {"api_error": "API disabled or key missing"}

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                raw = self._chat(system, user, temperature=temperature)
                parsed = self._parse_json(raw)
                self._save_cache(cache_key, {"cache_key": cache_key, "response": parsed, "raw": raw})
                return parsed
            except Exception as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        return {"api_error": str(last_err)}

    def label_prefix(
        self,
        *,
        prefix_id: str,
        question: str,
        gold_answer: str,
        reasoning_prefix: str,
    ) -> dict[str, Any]:
        cache_key = f"validity:{prefix_id}:{PROMPT_VERSION}"
        if not self._cache and self.cache_path:
            self._load_cache()
        if cache_key in self._cache:
            return self._normalize_validity(self._cache[cache_key]["response"], prefix_id)

        if not self.enabled:
            return self._normalize_validity({}, prefix_id, fallback_unclear=True)

        parsed = self._api_call(
            cache_key,
            SYSTEM_VALIDITY,
            USER_VALIDITY.format(
                question=question,
                gold_answer=gold_answer,
                prefix=reasoning_prefix[-3000:],
            ),
        )
        return self._normalize_validity(parsed, prefix_id)

    def cluster_next_steps(
        self,
        *,
        prefix_id: str,
        question: str,
        reasoning_prefix: str,
        next_steps: list[str],
    ) -> dict[str, Any]:
        cache_key = f"cluster:{prefix_id}:{CLUSTER_PROMPT_VERSION}"
        steps_text = "\n".join(f"[{i}] {s[:400]}" for i, s in enumerate(next_steps))
        if not self.enabled:
            return {"prefix_id": prefix_id, "api_error": "disabled", "clusters": []}

        parsed = self._api_call(
            cache_key,
            SYSTEM_CLUSTER,
            USER_CLUSTER.format(question=question, prefix=reasoning_prefix[-2000:], steps=steps_text),
        )
        clusters = parsed.get("clusters", [])
        if isinstance(clusters, list) and len(clusters) == len(next_steps):
            n_sem = int(parsed.get("num_semantic_clusters", len(set(clusters))))
            multi = parsed.get("multiple_genuine_strategies")
            if multi is None:
                multi = n_sem >= 2
            return {
                "prefix_id": prefix_id,
                "clusters": [int(c) for c in clusters],
                "num_semantic_clusters": n_sem,
                "multiple_genuine_strategies": bool(multi),
                "strategy_descriptions": parsed.get("strategy_descriptions", []),
                "api_model": self.model,
                "api_prompt_version": CLUSTER_PROMPT_VERSION,
            }
        return {"prefix_id": prefix_id, "api_error": parsed.get("api_error", "bad cluster response"), "clusters": []}

    def review_case(
        self,
        *,
        prefix_id: str,
        question: str,
        gold_answer: str,
        prefix_status: str,
        diversity_label: str,
        branch_pass: int,
        rollback_pass: int,
        continue_correct: int,
        reason: str,
    ) -> dict[str, Any]:
        cache_key = f"review:{prefix_id}:{REVIEW_PROMPT_VERSION}:{hash(reason) & 0xFFFF}"
        if not self.enabled:
            return {"prefix_id": prefix_id, "review_verdict": "api disabled"}

        parsed = self._api_call(
            cache_key,
            SYSTEM_REVIEW,
            USER_REVIEW.format(
                question=question,
                gold_answer=gold_answer,
                prefix_status=prefix_status,
                diversity_label=diversity_label,
                branch_pass=branch_pass,
                rollback_pass=rollback_pass,
                continue_correct=continue_correct,
                reason=reason,
            ),
        )
        return {
            "prefix_id": prefix_id,
            "prefix_status": str(parsed.get("prefix_status", "UNCLEAR")).upper(),
            "review_verdict": parsed.get("review_verdict", ""),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reason": reason,
            "api_model": self.model,
        }

    @staticmethod
    def _normalize_validity(parsed: dict, prefix_id: str, *, fallback_unclear: bool = False) -> dict[str, Any]:
        if parsed.get("api_error"):
            status = "API_ERROR"
            conf = 0.0
            err_type = "api_error"
            first_err = None
            expl = parsed.get("api_error", "API call failed")
            substantiveness = "SUBSTANTIVE"
        elif fallback_unclear:
            status = "API_ERROR"
            conf = 0.0
            err_type = "api_disabled"
            first_err = None
            expl = "API disabled or key missing"
            substantiveness = "SUBSTANTIVE"
        else:
            status = str(parsed.get("prefix_status") or parsed.get("prefix_validity", "UNCLEAR")).upper()
            if status not in {"VALID", "INVALID", "UNCLEAR", "NO_COMMITMENT"}:
                status = "UNCLEAR"
            conf = float(parsed.get("confidence", 0.0))
            err_type = str(parsed.get("error_type", "none"))
            first_err = parsed.get("first_error_step")
            expl = str(parsed.get("explanation", ""))
            substantiveness = str(parsed.get("prefix_substantiveness", "SUBSTANTIVE")).upper()
            if status == "NO_COMMITMENT":
                substantiveness = "NO_COMMITMENT"
            elif substantiveness not in {"SUBSTANTIVE", "NO_COMMITMENT"}:
                substantiveness = "SUBSTANTIVE"
        return {
            "prefix_id": prefix_id,
            "prefix_validity": status,
            "prefix_status": status,
            "prefix_substantiveness": substantiveness,
            "first_error_step": first_err,
            "error_type": err_type,
            "confidence": conf,
            "explanation": expl,
            "api_model": parsed.get("api_model", ""),
            "api_prompt_version": PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
