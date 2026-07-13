"""Post-pass: export hidden states + logits from saved prefixes (HF only)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors.torch import load_file
from tqdm import tqdm

from reasoning_branch_dataset.action_study.config import ActionStudyConfig
from reasoning_branch_dataset.action_study.hidden import DraftHiddenExporter
from reasoning_branch_dataset.hidden_export import flush_hidden_store
from reasoning_branch_dataset.io_utils import save_table


def _exported_prefix_ids(hidden_path: Path) -> set[str]:
    if not hidden_path.exists():
        return set()
    keys = load_file(str(hidden_path)).keys()
    return {k.split("/")[0] for k in keys}


def export_hidden(data_dir: Path, cfg: ActionStudyConfig, *, flush_every: int = 50) -> None:
    prefixes_path = data_dir / "prefixes.jsonl"
    if not prefixes_path.exists():
        raise FileNotFoundError(prefixes_path)

    rows = [json.loads(l) for l in prefixes_path.read_text().splitlines() if l.strip()]
    hidden_path = data_dir / "hidden.safetensors"
    done_ids = _exported_prefix_ids(hidden_path)
    pending = [r for r in rows if r["prefix_id"] not in done_ids]
    if not pending:
        print(f"hidden already exported for {len(rows)} prefixes")
        return

    exporter = DraftHiddenExporter(cfg.model_path, cfg.hidden_layers, topk_logits=cfg.topk_logits)
    logits_by_id: dict[str, dict] = {r["prefix_id"]: r for r in rows}

    for i, row in enumerate(tqdm(pending, desc="hidden_export"), 1):
        logits = exporter.export(
            row["prefix_id"],
            row["prefix_text"],
            int(row.get("step_index", row.get("reasoning_progress", 0) * 100)),
        )
        logits_by_id[row["prefix_id"]] = {**logits_by_id[row["prefix_id"]], **logits}
        if i % flush_every == 0:
            flush_hidden_store(hidden_path, exporter.store)
            exporter.store.clear()

    flush_hidden_store(hidden_path, exporter.store)
    exporter.store.clear()
    exporter.unload()

    updated = [logits_by_id[r["prefix_id"]] for r in rows]
    out_jsonl = data_dir / "prefixes.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in updated) + "\n")
    save_table(updated, data_dir / "prefixes.parquet")
    print(f"Exported hidden for {len(pending)} new prefixes ({len(rows)} total) -> {hidden_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--flush-every", type=int, default=int(os.environ.get("HIDDEN_FLUSH_EVERY", "50")))
    args = parser.parse_args()
    cfg = ActionStudyConfig()
    if args.model_path:
        cfg.model_path = args.model_path
    export_hidden(args.data_dir, cfg, flush_every=args.flush_every)


if __name__ == "__main__":
    main()
