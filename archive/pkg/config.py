"""Configuration for Reasoning Branch/Rollback Analysis Dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

DEFAULT_DRAFT_MODEL = str(WORKSPACE / "models" / "Qwen3.5-4B")
DEFAULT_TARGET_MODEL = str(WORKSPACE / "models" / "Qwen3.5-9B")

DATA_PATHS = {
    "math500": WORKSPACE / "data" / "math500" / "test.jsonl",
    "gsm8k": WORKSPACE / "reasoning_branch_dataset" / "data" / "gsm8k_test.jsonl",
    "aime": WORKSPACE / "reasoning_branch_dataset" / "data" / "aime_train.jsonl",
    "deepscaler": WORKSPACE / "reasoning_branch_dataset" / "data" / "deepscaler_preview.jsonl",
}

PREFIX_TYPES = (
    "WAIT_BEFORE",
    "WAIT_AFTER",
    "BUT_BEFORE",
    "BUT_AFTER",
    "PARAGRAPH_END",
    "ENTROPY_SPIKE",
    "RANDOM",
)

HIDDEN_POOLS = ("last", "step_mean", "local4")
HIDDEN_SOURCES = ("draft", "target")


@dataclass
class DatasetConfig:
    output_dir: Path = ROOT / "outputs" / "pilot_v1"
    draft_model: str = DEFAULT_DRAFT_MODEL
    target_model: str = DEFAULT_TARGET_MODEL
    datasets: list[str] = field(default_factory=lambda: ["math500", "aime"])
    max_problems_per_dataset: dict[str, int] = field(
        default_factory=lambda: {"math500": 50, "aime": 30}
    )
    max_new_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    topk_save: int = 5
    token_branch_k: int = 5
    step_branch_k: int = 4
    next_step_max_tokens: int = 128
    full_answer_branch_frac: float = 0.3
    max_paragraph_prefixes: int = 5
    max_marker_prefixes: int = 3
    max_random_prefixes: int = 2
    # API-guided mixed selection quotas (per problem)
    api_top_branch: int = 2
    api_top_rollback: int = 1
    max_wait_but_selected: int = 4
    max_paragraph_selected: int = 3
    n_random_control: int = 1
    n_low_score_control: int = 1
    use_api_teacher: bool = True
    use_trace_aware_api: bool = True
    teacher_base_url: str = "https://endpoint.greatrouter.com"
    teacher_model: str = "gpt-5.5"
    resume: bool = True
    hidden_layers: tuple[int, ...] = (16, 28, 32)
    spec_gamma: int = 4
    rollback_accept_threshold: float = 0.25
    rollback_reject_pos_threshold: int = 2
    rollback_kl_threshold: float = 0.5
    dtype: str = "bfloat16"
    device: str = "cuda"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
