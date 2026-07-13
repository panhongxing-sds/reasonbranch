# Roadmap

Priority order (no API required until local verifier is ready):

```
修管线 → 训练 V3.3 local probe → GPT 标签蒸馏本地 verifier → V3.4b local rollout
```

## P0 — Engineering (no API)

- [x] Separate technical failures (`ORACLE_API_ERROR`, `TARGET_GENERATION_ERROR`, `STEP_EXTRACTION_ERROR`)
- [x] Target handoff extraction (`extract_handoff_step`, R1 think-block handling)
- [x] Grading regression harness
- [x] Target non-empty rate >99% — diagnostic 100/100 OK (`outputs/target_step_diagnostic/`)

## P1 — Local probe (V3.3 labels)

- [x] Stage 1: Continue vs Intervention dataset (1395 samples)
- [x] Stage 2: Branch vs Handoff dataset (166 samples)
- [x] Logit-feature logistic probe with problem-level GroupKFold
- [ ] Hidden-state probe (export draft hidden layers)
- [ ] MLP / hidden+logit combined probe

## P2 — Local verifier (7320 GPT candidate labels)

- [x] Build `candidate_labels.jsonl` from stable V3.3 passes
- [ ] Zero-shot 14B ACCEPT/REJECT eval
- [ ] LoRA/SFT if zero-shot below gate (action agreement ≥85%)

## P3 — V3.4b local rollout

- Local SpecReason vs Local Conditional Branch
- No GPT API; 1.5B draft + 14B verifier/target dual-resident

## Do not do yet

- Expand polluted V3.4 results as mechanism evidence
- Train sequential probe on V3.3 static labels directly
- QwQ 0–9 utility labels for new experiments
