# Reasoning Branch Dataset

Step-level reasoning action study: **Continue / Branch / Handoff** under SpecReason-style speculative decoding, with GPT oracle labels, local probes, and sequential rollout evaluation.

## Project layout

```
reasoning_branch_dataset/
├── action_study/          # Core experiment code (V2–V3.4)
│   ├── sequential_rollout.py    # V3.4 sequential policy engine
│   ├── gpt_step_oracle.py       # V3.3 GPT per-step oracle
│   ├── build_probe_dataset.py   # Two-stage probe data export
│   ├── train_local_probe.py     # Logit-feature probe (no API)
│   ├── build_verifier_dataset.py
│   ├── local_step_verifier.py   # 14B zero-shot ACCEPT/REJECT
│   └── target_step_diagnostic.py
├── scripts/               # Runnable shell entrypoints
├── tests/                 # Unit tests
├── docs/                  # Design notes & samples
├── data/                  # Small bundled data (+ download guide)
└── outputs/               # Reports (committed) + artifacts (gitignored)
```

## Setup

```bash
export AFS=/path/to/parent          # directory containing reasoning_branch_dataset/
export PYTHONPATH="${AFS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# API (optional — not needed for local probe / verifier dataset / grading)
cp reasoning_branch_dataset/scripts/env.example .env
# edit TEACHER_API_KEY / TEACHER_KEYFILE
source reasoning_branch_dataset/scripts/load_api_env.sh
```

Models (cluster paths, not in repo):

- Draft: `specreason/models/DeepSeek-R1-Distill-Qwen-1.5B`
- Target: `specreason/models/DeepSeek-R1-Distill-Qwen-14B`

## Experiment pipeline

| Phase | Script | API |
|-------|--------|-----|
| V2 data collection | `scripts/run_batch.sh` | optional |
| V3.3 GPT step oracle | `scripts/run_v3_3_gpt_step_oracle.sh` | yes |
| V3.4 sequential rollout | `scripts/run_v3_4b.sh` | yes |
| **Local pipeline (no API)** | `scripts/run_local_pipeline.sh` | no |

### No-API local work (recommended when API unavailable)

```bash
bash reasoning_branch_dataset/scripts/run_local_pipeline.sh
```

Runs: probe dataset → probe CV → verifier dataset → grading regression → target step diagnostic (GPU).

## Reports

See [`outputs/INDEX.md`](outputs/INDEX.md) for all pilot reports (V2 → V3.4).

## Tests

```bash
cd "${AFS}"
python -m pytest reasoning_branch_dataset/tests/ -q
```

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md): fix pipeline → train V3.3 probe → distill local verifier → V3.4b local rollout.

## Git

Large artifacts (`outputs/**/*.jsonl`, logs, models, `data/deepscaler_preview.jsonl`) are gitignored. Regenerate via scripts or download per `data/README.md`.
