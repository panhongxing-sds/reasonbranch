# Action Study 报告索引

**只看报告，打开下面文件即可。** JSON/原始数据在子目录。

| 版本 | 报告 | 数据目录 |
|------|------|----------|
| **v2** 数据采集 | [`pilot_v2_report.md`](pilot_v2_report.md) | `action_study_pilot_v2/` |
| **v3** utility oracle | [`pilot_v3_report.md`](pilot_v3_report.md) | `action_study_pilot_v3/` |
| **v3 audit phase 1** | [`pilot_v3_audit_report.md`](pilot_v3_audit_report.md) | step quality, weak/strong tiers |
| **v3 audit phase 2** (QwQ) | [`pilot_v3_audit_phase2_report.md`](pilot_v3_audit_phase2_report.md) | shuffle + pairwise |
| **v3.2 GPT oracle** | [`pilot_v3_2_report.md`](pilot_v3_2_report.md) | GPT-5.5 pairwise judge (greedy vs best branch) |
| **v3.3 GPT step oracle** | [`pilot_v3_3_report.md`](pilot_v3_3_report.md) | GPT-5.5 per-candidate next-step (1G+4B) |
| **v3.4 sequential rollout** | [`pilot_v3_4_report.md`](pilot_v3_4_report.md) | `action_study_pilot_v34/` |
| **reachable-state** | [`reachable_state_report.md`](reachable_state_report.md) | `reachable_state_pilot/` |

重新生成全部报告：

```bash
export AFS=/mnt/afs/L202500372 PYTHONPATH=$AFS
/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.write_output_reports
```

仅 V3 audit：

```bash
/tmp/vllm-cu124/bin/python -m reasoning_branch_dataset.action_study.run_v3_oracle_audit
```

V3.3 GPT next-step oracle（audit 312 或 `FULL=1` 全量 1548）：

```bash
source /mnt/afs/L202500372/reasoning_branch_dataset/scripts/load_api_env.sh
bash /mnt/afs/L202500372/reasoning_branch_dataset/scripts/run_v3_3_gpt_step_oracle.sh
# 全量: FULL=1 bash .../run_v3_3_gpt_step_oracle.sh
```

V3.4b sequential rollout (P0 fixes, core policies only):

```bash
bash reasoning_branch_dataset/scripts/run_v3_4b.sh
```

No-API local pipeline (probe + verifier dataset + grading + target diagnostic):

```bash
bash reasoning_branch_dataset/scripts/run_local_pipeline.sh
```
