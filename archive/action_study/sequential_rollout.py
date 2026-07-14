"""V3.4 sequential oracle policy rollout engine."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from reasoning_branch_dataset.action_study.gpt_step_oracle import (
    BRANCH_KEYS,
    CANDIDATE_KEYS,
    GREEDY_KEY,
    GPTStepOracleClient,
    action_stable,
)
from reasoning_branch_dataset.action_study.step_extraction import (
    extract_handoff_step,
    extract_next_substantive_step,
)
from reasoning_branch_dataset.action_study.step_admission import classify_step_quality
from reasoning_branch_dataset.action_study.target_verifier import build_target_verifier, greedy_generate_vllm
from reasoning_branch_dataset.action_study.vllm_backend import HFEngine, VLLMEngine, build_engine
from reasoning_branch_dataset.action_study.technical_errors import TECHNICAL_FAILURES, valid_for_comparison
from reasoning_branch_dataset.grading import classify_generation_outcome, extract_math_answer
from reasoning_branch_dataset.model_utils import build_prompt

BOXED_RE = re.compile(r"\\boxed\s*\{[^}]+\}")
MAX_REPEAT_CHARS = 200


class Policy(str, Enum):
    DRAFT_ONLY = "DRAFT_ONLY"
    TARGET_ONLY = "TARGET_ONLY"
    SPECREASON = "SPECREASON"
    CONDITIONAL_BRANCH = "CONDITIONAL_BRANCH"
    ALWAYS_BRANCH = "ALWAYS_BRANCH"


@dataclass
class RolloutConfig:
    draft_model: str = "/mnt/afs/L202500372/specreason/models/DeepSeek-R1-Distill-Qwen-1.5B"
    target_model: str = "/mnt/afs/L202500372/specreason/models/DeepSeek-R1-Distill-Qwen-14B"
    target_quantization: str | None = None
    max_steps: int = 20
    step_max_tokens: int = 256
    branch_k: int = 4
    branch_temperature: float = 0.7
    branch_top_p: float = 0.95
    target_step_max_tokens: int = 384
    target_handoff_max_attempts: int = 3
    oracle_max_retries: int = 3
    gpt_cache_path: Path | None = None
    dual_resident: bool = False
    draft_gpu_util: float = 0.90
    target_gpu_util: float = 0.92
    target_max_model_len: int = 4096
    # Dual-resident on one 80GB card: 14B bf16 (~28GB) loads first, 1.5B (~4GB)
    # second. This vLLM enforces desired=total*util <= *free* memory, so the
    # target takes ~0.42 (~33GB) and the 1.5B uses a fraction of what's free.
    dual_target_gpu_util: float = 0.42
    dual_draft_gpu_util: float = 0.45


@dataclass
class StepRecord:
    problem_id: str
    rollout_id: str
    policy: str
    seed: int
    step_index: int
    prefix_text: str
    prefix_hash: str
    greedy_step: str | None = None
    greedy_acceptable: bool | None = None
    branch_steps: list[str] = field(default_factory=list)
    branch_acceptable: list[bool] = field(default_factory=list)
    n_acceptable_branches: int = 0
    selected_branch_index: int | None = None
    target_step: str | None = None
    action: str = ""
    selected_step: str = ""
    next_prefix_hash: str = ""
    oracle_stable: bool | None = None
    oracle_note: str = ""
    step_generation_tokens: dict[str, Any] = field(default_factory=dict)
    termination: bool = False
    termination_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def prefix_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def append_step(prefix: str, step: str) -> str:
    step = step.strip()
    if not step:
        return prefix
    if prefix.endswith("\n\n"):
        return prefix + step + "\n\n"
    if prefix.endswith("\n"):
        return prefix + "\n" + step + "\n\n"
    return prefix.rstrip() + "\n\n" + step + "\n\n"


def reasoning_tail(full_prefix: str, question: str) -> str:
    prompt = build_prompt(question)
    if full_prefix.startswith(prompt):
        return full_prefix[len(prompt) :]
    return full_prefix


def extract_step(text: str, *, question: str) -> dict[str, Any]:
    ext = extract_next_substantive_step(text, question=question)
    return ext


def has_final_answer(reasoning_text: str) -> bool:
    """True if reasoning section contains a non-empty \\boxed{...} answer."""
    ans = extract_math_answer(reasoning_text, require_marker=True)
    return bool(ans and BOXED_RE.search(reasoning_text))


def detect_loop(prefix: str, step: str, *, window: int = 3) -> bool:
    """Detect repeated step text in recent history."""
    blocks = [b.strip() for b in prefix.split("\n\n") if b.strip()]
    step_norm = step.strip()[:MAX_REPEAT_CHARS]
    if not step_norm:
        return False
    recent = blocks[-window:] if len(blocks) >= window else blocks
    return sum(1 for b in recent if b[:MAX_REPEAT_CHARS] == step_norm) >= 2


class ModelSession:
    """vLLM session: single-model swap or dual-resident (draft+target, no per-step reload)."""

    def __init__(self, cfg: RolloutConfig) -> None:
        self.cfg = cfg
        self._draft: VLLMEngine | HFEngine | None = None
        self._target = None
        self._target_tok = None
        self._dual_ready = False

    def _cuda_gc(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _free(self) -> None:
        if self._draft is not None and isinstance(self._draft, HFEngine):
            try:
                del self._draft.model
            except Exception:
                pass
        self._draft = None
        self._target = None
        self._target_tok = None
        self._dual_ready = False
        self._cuda_gc()

    def _load_draft(self) -> VLLMEngine | HFEngine:
        if self.cfg.dual_resident:
            # Second vLLM engine (1.5B) loads after the 32B-AWQ target. This
            # vLLM enforces desired=total*util <= *free* mem, so util is the
            # fraction of the whole card still free (~0.50 after ~18GB AWQ).
            self._draft = VLLMEngine(
                self.cfg.draft_model,
                gpu_memory_utilization=self.cfg.dual_draft_gpu_util,
                max_model_len=4096,
            )
            return self._draft
        self._draft = VLLMEngine(
            self.cfg.draft_model,
            gpu_memory_utilization=self.cfg.draft_gpu_util,
            max_model_len=8192,
        )
        return self._draft

    def _load_target(self):
        util = self.cfg.dual_target_gpu_util if self.cfg.dual_resident else self.cfg.target_gpu_util
        mlen = self.cfg.target_max_model_len
        self._target = build_target_verifier(
            self.cfg.target_model,
            engine="vllm",
            debug=not self.cfg.dual_resident,
            dual_resident=self.cfg.dual_resident,
            gpu_memory_utilization=util,
            max_model_len=mlen,
            quantization=self.cfg.target_quantization,
        )
        self._target_tok = self._target.tokenizer
        return self._target

    def draft(self) -> VLLMEngine | HFEngine:
        if self._draft is not None:
            return self._draft
        if not self.cfg.dual_resident:
            self._free()
        return self._load_draft()

    def target(self):
        if self._target is not None:
            return self._target
        if not self.cfg.dual_resident:
            self._free()
        return self._load_target()

    def preload_dual(self) -> None:
        """Both vLLM engines resident on one GPU — zero swap on handoff.

        Order matters: QwQ first (fixed share), then 4B with a util that
        accounts for the memory QwQ already holds.
        """
        if self._dual_ready:
            return
        self._free()
        self.cfg.dual_resident = True
        try:
            self._load_target()
            self._load_draft()
            self._dual_ready = True
            print("INFO: dual-resident ready (vLLM QwQ + vLLM 4B, no handoff swap)")
        except Exception as exc:
            print(f"WARN: dual-resident load failed ({exc}); falling back to single-model swap")
            self._free()
            self.cfg.dual_resident = False

    def unload_all(self) -> None:
        self._free()


class StepOracle:
    """GPT dual-pass with optional 3rd tie-break; API errors are not Handoff."""

    def __init__(self, client: GPTStepOracleClient, *, max_retries: int = 3) -> None:
        self.client = client
        self.max_retries = max_retries

    def judge_five(
        self,
        *,
        judge_id: str,
        question: str,
        prefix_tail: str,
        steps: dict[str, str],
    ) -> dict[str, Any]:
        last: dict[str, Any] = {}
        for attempt in range(self.max_retries):
            last = self._judge_five_once(
                judge_id=judge_id,
                question=question,
                prefix_tail=prefix_tail,
                steps=steps,
            )
            if not last.get("oracle_api_error"):
                return last
            time.sleep(2.0 * (attempt + 1))
        return last

    def _judge_five_once(
        self,
        *,
        judge_id: str,
        question: str,
        prefix_tail: str,
        steps: dict[str, str],
    ) -> dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        def _pass(pid: int, seed_suffix: str = "") -> dict[str, Any]:
            return self.client.judge_shuffled_pass(
                prefix_id=judge_id,
                question=question,
                prefix_tail=prefix_tail,
                steps=steps,
                shuffle_seed=abs(hash(judge_id + seed_suffix)) % 10_000_000,
                pass_id=pid,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_pass, 1)
            f2 = pool.submit(_pass, 2, ":2")
            p1 = f1.result()
            p2 = f2.result()

        if "api_error" in p1 or "api_error" in p2:
            return {
                "oracle_stable": False,
                "oracle_note": "ORACLE_API_ERROR",
                "oracle_api_error": True,
                "prefix_status": None,
                "g_acceptable": False,
                "branch_acceptable": [False] * 4,
                "any_branch_acceptable": False,
                "n_acceptable_branches": 0,
                "force_handoff": False,
            }

        stable = action_stable(p1, p2)
        use = p1
        note = ""
        if not stable:
            p3 = self.client.judge_shuffled_pass(
                prefix_id=judge_id,
                question=question,
                prefix_tail=prefix_tail,
                steps=steps,
                shuffle_seed=abs(hash(judge_id + ":3")) % 10_000_000,
                pass_id=3,
            )
            if "api_error" in p3:
                return {
                    "oracle_stable": False,
                    "oracle_note": "ORACLE_API_ERROR",
                    "oracle_api_error": True,
                    "prefix_status": None,
                    "g_acceptable": False,
                    "branch_acceptable": [False] * 4,
                    "any_branch_acceptable": False,
                    "n_acceptable_branches": 0,
                    "force_handoff": False,
                }
            g3 = p3.get("g_acceptable")
            if g3 == p1.get("g_acceptable") == p2.get("g_acceptable"):
                use = p1
                stable = p1.get("any_branch_acceptable") == p2.get("any_branch_acceptable")
            elif g3 == p1.get("g_acceptable"):
                use = p1
                stable = p1.get("any_branch_acceptable") == p3.get("any_branch_acceptable")
            elif g3 == p2.get("g_acceptable"):
                use = p2
                stable = p2.get("any_branch_acceptable") == p3.get("any_branch_acceptable")
            else:
                note = "ORACLE_UNSTABLE_HANDOFF"
                stable = False

        if not stable and not note:
            note = "ORACLE_UNSTABLE_HANDOFF"

        return {
            "oracle_stable": stable,
            "oracle_note": note,
            "oracle_api_error": False,
            "prefix_status": use.get("prefix_status"),
            "g_acceptable": use.get("g_acceptable") if stable else False,
            "branch_acceptable": use.get("branch_acceptables", [False] * 4) if stable else [False] * 4,
            "any_branch_acceptable": use.get("any_branch_acceptable") if stable else False,
            "n_acceptable_branches": use.get("n_acceptable_branches") if stable else 0,
            "force_handoff": not stable,
        }

    def judge_greedy_only(
        self,
        *,
        judge_id: str,
        question: str,
        prefix_tail: str,
        greedy_step: str,
    ) -> dict[str, Any]:
        empty = {k: "" for k in BRANCH_KEYS}
        steps = {GREEDY_KEY: greedy_step, **empty}
        return self.judge_five(judge_id=judge_id, question=question, prefix_tail=prefix_tail, steps=steps)


def draft_normalized_logprob(engine: VLLMEngine | HFEngine, prefix: str, step: str) -> float:
    """Length-normalized log P(step|prefix)."""
    if isinstance(engine, HFEngine):
        try:
            import torch

            tok = engine.tokenizer
            prefix_ids = tok.encode(prefix, add_special_tokens=False)
            step_ids = tok.encode(step, add_special_tokens=False)
            if not step_ids:
                return float("-inf")
            full_ids = prefix_ids + step_ids
            inp = torch.tensor([full_ids], device="cuda")
            with torch.no_grad():
                logits = engine.model(inp).logits[0]
            total = 0.0
            for j, tid in enumerate(step_ids):
                lp = torch.log_softmax(logits[len(prefix_ids) + j - 1], dim=-1)[tid]
                total += float(lp)
            return total / len(step_ids)
        except Exception:
            return -len(step) / 1000.0
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        tok = AutoTokenizer.from_pretrained(engine.model_path, trust_remote_code=True)
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        step_ids = tok.encode(step, add_special_tokens=False)
        if not step_ids:
            return float("-inf")
        full_ids = prefix_ids + step_ids
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5, detokenize=False)
        out = engine.llm.generate([{"prompt_token_ids": full_ids}], params)[0]
        logprob_steps = out.prompt_logprobs or []
        total = 0.0
        base = len(prefix_ids)
        for j, tid in enumerate(step_ids):
            idx = base + j
            if idx >= len(logprob_steps) or logprob_steps[idx] is None:
                return float("-inf")
            lp_map = logprob_steps[idx]
            tok_lp = None
            for k, v in lp_map.items():
                if int(k) == tid or getattr(v, "decoded_token", None):
                    tok_lp = float(v.logprob)
                    break
            if tok_lp is None:
                vals = [float(v.logprob) for v in lp_map.values()]
                tok_lp = max(vals) if vals else float("-inf")
            total += tok_lp
        return total / len(step_ids)
    except Exception:
        return -len(step) / 1000.0


def select_branch(
    engine: VLLMEngine | HFEngine,
    prefix: str,
    branch_steps: list[str],
    branch_acceptable: list[bool],
) -> int:
    candidates = [(i, s) for i, (s, ok) in enumerate(zip(branch_steps, branch_acceptable)) if ok]
    if not candidates:
        return -1
    if len(candidates) == 1:
        return candidates[0][0]
    scored = [(i, draft_normalized_logprob(engine, prefix, s)) for i, s in candidates]
    return max(scored, key=lambda x: x[1])[0]


def generate_draft_step(
    engine: VLLMEngine | HFEngine, prefix: str, *, temperature: float, k: int = 1
) -> list[dict[str, Any]]:
    return engine.generate_next_steps(
        prefix, k=k, max_tokens=256, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0
    )


def generate_target_step(
    session: ModelSession,
    prefix: str,
    cfg: RolloutConfig,
    *,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    verifier = session.target()
    ids = verifier.tokenize(prefix)
    gen = greedy_generate_vllm(verifier.llm, ids, max_tokens=max_tokens or cfg.target_step_max_tokens)
    text = verifier.tokenizer.decode(gen["token_ids"], skip_special_tokens=False)
    return {
        "text": text,
        "num_tokens": len(gen["token_ids"]),
        "token_ids": gen["token_ids"],
        "finish_reason": gen.get("finish_reason"),
        "stop_reason": gen.get("stop_reason"),
        "prompt_length": len(ids),
    }


def _target_failure_reason(row: dict[str, Any], extracted: str) -> str:
    if not row.get("text", "").strip() or row.get("num_tokens", 0) == 0:
        return "TARGET_GENERATION_ERROR"
    if not extracted.strip():
        return "STEP_EXTRACTION_ERROR"
    return ""


def select_target_handoff_step(
    session: ModelSession,
    prefix: str,
    cfg: RolloutConfig,
    *,
    question: str,
) -> tuple[str, dict[str, Any], str]:
    """Generate target step with retries; return (step, row, failure_reason)."""
    last_row: dict[str, Any] = {"text": "", "num_tokens": 0}
    for attempt in range(cfg.target_handoff_max_attempts):
        max_tokens = cfg.target_step_max_tokens + attempt * 128
        last_row = generate_target_step(session, prefix, cfg, max_tokens=max_tokens)
        step = extract_handoff_step(last_row["text"], question=question)
        fail = _target_failure_reason(last_row, step)
        if not fail:
            return step, last_row, ""
    return "", last_row, _target_failure_reason(last_row, extract_handoff_step(last_row["text"], question=question))


def _fail_target_step(
    rec: StepRecord,
    steps_out: list[StepRecord],
    *,
    action: str,
    fail_reason: str,
    row: dict[str, Any],
    selected: str = "",
) -> str:
    rec.action = action
    rec.selected_step = selected
    rec.target_step = selected
    rec.step_generation_tokens["target"] = row.get("num_tokens", 0)
    rec.termination = True
    rec.termination_reason = fail_reason
    steps_out.append(rec)
    return fail_reason


def _oracle_api_failure(
    rec: StepRecord,
    steps_out: list[StepRecord],
    oracle_result: dict[str, Any],
) -> str:
    rec.action = "ORACLE_API_ERROR"
    rec.oracle_note = oracle_result.get("oracle_note", "ORACLE_API_ERROR")
    rec.oracle_stable = False
    rec.termination = True
    rec.termination_reason = "ORACLE_API_ERROR"
    steps_out.append(rec)
    return "ORACLE_API_ERROR"


def run_rollout(
    problem: dict[str, Any],
    *,
    policy: Policy,
    seed: int,
    cfg: RolloutConfig,
    oracle: StepOracle | None,
    session: ModelSession,
) -> tuple[list[StepRecord], dict[str, Any]]:
    rng = random.Random(seed)
    question = problem["question"]
    gold = problem.get("gold_answer", "")
    prompt = build_prompt(question)
    prefix = prompt
    reasoning_offset = len(prompt)
    rollout_id = f"{problem['problem_id']}::{policy.value}::s{seed}"
    steps_out: list[StepRecord] = []
    n_continue = n_branch = n_handoff = n_oracle_api_error = 0
    termination_reason = ""

    for step_idx in range(cfg.max_steps):
        reasoning = prefix[reasoning_offset:]
        if has_final_answer(reasoning):
            termination_reason = "FINAL_ANSWER"
            break

        rec = StepRecord(
            problem_id=problem["problem_id"],
            rollout_id=rollout_id,
            policy=policy.value,
            seed=seed,
            step_index=step_idx,
            prefix_text=prefix,
            prefix_hash=prefix_hash(prefix),
        )
        tail = reasoning_tail(prefix, question)
        action = ""
        selected = ""
        oracle_result: dict[str, Any] = {}
        branch_steps: list[str] = []
        branch_acceptable: list[bool] = [False] * 4

        if policy == Policy.DRAFT_ONLY:
            rows = generate_draft_step(session.draft(), prefix, temperature=0.0, k=1)
            ext = extract_step(rows[0]["text"], question=question)
            selected = ext["candidate_step"]
            action = "CONTINUE"
            rec.greedy_step = selected
            rec.greedy_acceptable = True
            rec.step_generation_tokens = {"greedy": rows[0].get("num_tokens", 0)}
            n_continue += 1

        elif policy == Policy.TARGET_ONLY:
            selected, row, fail_reason = select_target_handoff_step(session, prefix, cfg, question=question)
            action = "HANDOFF"
            rec.target_step = selected
            rec.step_generation_tokens = {"target": row.get("num_tokens", 0)}
            if fail_reason:
                termination_reason = _fail_target_step(
                    rec, steps_out, action=action, fail_reason=fail_reason, row=row, selected=selected
                )
                break
            n_handoff += 1

        else:
            # Oracle-guided policies
            assert oracle is not None
            greedy_rows = generate_draft_step(session.draft(), prefix, temperature=0.0, k=1)
            greedy_ext = extract_step(greedy_rows[0]["text"], question=question)
            greedy_step = greedy_ext["candidate_step"]
            rec.greedy_step = greedy_step
            rec.step_generation_tokens["greedy"] = greedy_rows[0].get("num_tokens", 0)

            if policy == Policy.ALWAYS_BRANCH:
                branch_rows = generate_draft_step(session.draft(), prefix, temperature=cfg.branch_temperature, k=cfg.branch_k)
                branch_steps = [
                    extract_step(r["text"], question=question)["candidate_step"] for r in branch_rows
                ]
                rec.step_generation_tokens["branches"] = [r.get("num_tokens", 0) for r in branch_rows]
                steps_dict = {GREEDY_KEY: greedy_step}
                for i, k in enumerate(BRANCH_KEYS):
                    steps_dict[k] = branch_steps[i] if i < len(branch_steps) else ""
                oracle_result = oracle.judge_five(
                    judge_id=f"{rollout_id}:t{step_idx}",
                    question=question,
                    prefix_tail=tail,
                    steps=steps_dict,
                )
                if oracle_result.get("oracle_api_error"):
                    termination_reason = _oracle_api_failure(rec, steps_out, oracle_result)
                    n_oracle_api_error += 1
                    break
                branch_acceptable = oracle_result.get("branch_acceptable", [False] * 4)
                rec.greedy_acceptable = oracle_result.get("g_acceptable")
                g_ok = rec.greedy_acceptable and not oracle_result.get("force_handoff")
                if g_ok:
                    action, selected = "CONTINUE", greedy_step
                    n_continue += 1
                elif oracle_result.get("any_branch_acceptable"):
                    bi = select_branch(session.draft(), prefix, branch_steps, branch_acceptable)
                    action, selected = "BRANCH", branch_steps[bi]
                    rec.selected_branch_index = bi
                    n_branch += 1
                elif oracle_result.get("force_handoff"):
                    selected, row, fail_reason = select_target_handoff_step(session, prefix, cfg, question=question)
                    action = "HANDOFF"
                    rec.target_step = selected
                    rec.oracle_note = oracle_result.get("oracle_note", "ORACLE_UNSTABLE_HANDOFF")
                    rec.step_generation_tokens["target"] = row.get("num_tokens", 0)
                    if fail_reason:
                        termination_reason = _fail_target_step(
                            rec, steps_out, action=action, fail_reason=fail_reason, row=row, selected=selected
                        )
                        break
                    n_handoff += 1
                else:
                    selected, row, fail_reason = select_target_handoff_step(session, prefix, cfg, question=question)
                    action = "HANDOFF"
                    rec.target_step = selected
                    rec.step_generation_tokens["target"] = row.get("num_tokens", 0)
                    if fail_reason:
                        termination_reason = _fail_target_step(
                            rec, steps_out, action=action, fail_reason=fail_reason, row=row, selected=selected
                        )
                        break
                    n_handoff += 1

            elif policy in (Policy.SPECREASON, Policy.CONDITIONAL_BRANCH):
                oracle_result = oracle.judge_greedy_only(
                    judge_id=f"{rollout_id}:t{step_idx}:g",
                    question=question,
                    prefix_tail=tail,
                    greedy_step=greedy_step,
                )
                if oracle_result.get("oracle_api_error"):
                    termination_reason = _oracle_api_failure(rec, steps_out, oracle_result)
                    n_oracle_api_error += 1
                    break
                rec.greedy_acceptable = oracle_result.get("g_acceptable")
                g_ok = rec.greedy_acceptable and not oracle_result.get("force_handoff")

                if g_ok:
                    action, selected = "CONTINUE", greedy_step
                    n_continue += 1
                else:
                    need_branch = policy == Policy.CONDITIONAL_BRANCH
                    if need_branch:
                        branch_rows = generate_draft_step(
                            session.draft(), prefix, temperature=cfg.branch_temperature, k=cfg.branch_k
                        )
                        branch_steps = [
                            extract_step(r["text"], question=question)["candidate_step"] for r in branch_rows
                        ]
                        rec.step_generation_tokens["branches"] = [r.get("num_tokens", 0) for r in branch_rows]
                        steps_dict = {GREEDY_KEY: greedy_step}
                        for i, k in enumerate(BRANCH_KEYS):
                            steps_dict[k] = branch_steps[i] if i < len(branch_steps) else ""
                        oracle_result = oracle.judge_five(
                            judge_id=f"{rollout_id}:t{step_idx}",
                            question=question,
                            prefix_tail=tail,
                            steps=steps_dict,
                        )
                        if oracle_result.get("oracle_api_error"):
                            termination_reason = _oracle_api_failure(rec, steps_out, oracle_result)
                            n_oracle_api_error += 1
                            break
                        branch_acceptable = oracle_result.get("branch_acceptable", [False] * 4)
                        if oracle_result.get("any_branch_acceptable") and not oracle_result.get("force_handoff"):
                            bi = select_branch(session.draft(), prefix, branch_steps, branch_acceptable)
                            action, selected = "BRANCH", branch_steps[bi]
                            rec.selected_branch_index = bi
                            n_branch += 1
                        else:
                            selected, row, fail_reason = select_target_handoff_step(session, prefix, cfg, question=question)
                            action = "HANDOFF"
                            rec.target_step = selected
                            if oracle_result.get("force_handoff"):
                                rec.oracle_note = oracle_result.get("oracle_note", "")
                            rec.step_generation_tokens["target"] = row.get("num_tokens", 0)
                            if fail_reason:
                                termination_reason = _fail_target_step(
                                    rec, steps_out, action=action, fail_reason=fail_reason, row=row, selected=selected
                                )
                                break
                            n_handoff += 1
                    else:
                        # SPECREASON: direct handoff
                        selected, row, fail_reason = select_target_handoff_step(session, prefix, cfg, question=question)
                        action = "HANDOFF"
                        rec.target_step = selected
                        if oracle_result.get("force_handoff"):
                            rec.oracle_note = oracle_result.get("oracle_note", "")
                        rec.step_generation_tokens["target"] = row.get("num_tokens", 0)
                        if fail_reason:
                            termination_reason = _fail_target_step(
                                rec, steps_out, action=action, fail_reason=fail_reason, row=row, selected=selected
                            )
                            break
                        n_handoff += 1

        rec.branch_steps = branch_steps if policy != Policy.DRAFT_ONLY else []
        rec.branch_acceptable = branch_acceptable if policy not in (Policy.DRAFT_ONLY, Policy.TARGET_ONLY) else []
        rec.n_acceptable_branches = sum(rec.branch_acceptable)
        rec.oracle_stable = oracle_result.get("oracle_stable")
        if oracle_result.get("oracle_note"):
            rec.oracle_note = oracle_result["oracle_note"]
        rec.action = action
        rec.selected_step = selected

        qual = classify_step_quality(selected, question=question)
        if not qual["eligible_for_oracle"] and action != "HANDOFF":
            termination_reason = "MALFORMED_STEP"
            rec.termination = True
            rec.termination_reason = termination_reason
            steps_out.append(rec)
            break

        if detect_loop(prefix, selected):
            termination_reason = "LOOP_DETECTED"
            rec.termination = True
            rec.termination_reason = termination_reason
            steps_out.append(rec)
            break

        new_prefix = append_step(prefix, selected)
        rec.next_prefix_hash = prefix_hash(new_prefix)
        # Sanity: prefix must change
        if rec.next_prefix_hash == rec.prefix_hash:
            termination_reason = "PREFIX_UNCHANGED"
            rec.termination = True
            rec.termination_reason = termination_reason
            steps_out.append(rec)
            break

        prefix = new_prefix
        steps_out.append(rec)

        if has_final_answer(prefix[reasoning_offset:]):
            termination_reason = "FINAL_ANSWER"
            break
    else:
        termination_reason = "MAX_STEP_TRUNCATED"

    outcome = classify_generation_outcome(prefix[reasoning_offset:], gold)
    extracted = (
        outcome.get("extracted_answer")
        or outcome.get("final_answer")
        or outcome.get("predicted_answer")
        or ""
    )
    summary = {
        "rollout_id": rollout_id,
        "problem_id": problem["problem_id"],
        "policy": policy.value,
        "seed": seed,
        "n_steps": len(steps_out),
        "n_continue": n_continue,
        "n_branch": n_branch,
        "n_handoff": n_handoff,
        "n_oracle_api_error": n_oracle_api_error,
        "n_target_steps": n_handoff if policy != Policy.TARGET_ONLY else len(steps_out),
        "termination_reason": termination_reason,
        "is_correct": outcome.get("is_correct"),
        "extracted_answer": extracted,
        "evaluation_status": outcome.get("evaluation_status"),
        "valid_for_comparison": valid_for_comparison(termination_reason),
        "final_prefix_hash": prefix_hash(prefix),
        "actions": [s.action for s in steps_out],
    }
    return steps_out, summary
