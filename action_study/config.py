"""Config for Oracle Action-Matching Study (phase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent

DEFAULT_MODEL = str(WORKSPACE / "models" / "Qwen3.5-4B")


@dataclass
class ActionStudyConfig:
    output_dir: Path = ROOT / "outputs" / "action_study_v1"
    model_path: str = DEFAULT_MODEL
    engine: str = "vllm"  # vllm for generation; hidden via export_hidden_pass.sh
    vllm_gpu_util: float = 0.82
    datasets: list[str] = field(default_factory=lambda: ["math500", "deepscaler", "gsm8k"])
    max_problems: dict[str, int] = field(
        default_factory=lambda: {"math500": 200, "deepscaler": 50, "aime": 30, "gsm8k": 100}
    )
    dataset_offsets: dict[str, int] = field(
        default_factory=lambda: {"math500": 0, "deepscaler": 0}
    )
    max_new_tokens: int = 4096
    continuation_max_tokens: int = 2048
    continuation_retry_tokens: int = 2048
    next_step_max_tokens: int = 128
    branch_k: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    topk_logits: int = 10
    max_marker_prefixes: int = 3
    max_paragraph_prefixes: int = 5
    hidden_layers: tuple[int, ...] = (-4, -2, -1)  # middle / late / final (resolved at runtime)
    save_draft_hidden: bool = True
    save_prefix_logits: bool = True
    use_api_validity: bool = True
    use_api_clustering: bool = True
    use_api_review: bool = True
    teacher_base_url: str = "https://endpoint.greatrouter.com"
    teacher_model: str = "gpt-5.5"
    diversity_high_entropy_threshold: float = 0.6
    resume: bool = True
    # Diagnostic only — not a primary action; writes counterfactual_regeneration.jsonl
    run_counterfactual_regeneration: bool = False

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
