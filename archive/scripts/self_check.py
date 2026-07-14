"""Dataset integrity self-checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file


def _load_rows(path: Path) -> list[dict]:
    jsonl = path.with_suffix(".jsonl")
    parquet = path
    if jsonl.exists():
        rows = []
        for line in jsonl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if parquet.exists():
        import pandas as pd

        return pd.read_parquet(parquet).to_dict("records")
    return []


def run_self_check(data_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    stats: dict[str, Any] = {}

    tables = {
        "traces": _load_rows(data_dir / "traces.parquet"),
        "prefixes": _load_rows(data_dir / "prefixes.parquet"),
        "token_features": _load_rows(data_dir / "token_features.parquet"),
        "step_branches": _load_rows(data_dir / "step_branches.parquet"),
        "api_annotations": _load_rows(data_dir / "api_annotations.parquet"),
    }
    stats["table_rows"] = {k: len(v) for k, v in tables.items()}

    hidden_path = data_dir / "hidden.safetensors"
    if hidden_path.exists():
        tensors = load_file(str(hidden_path))
        bad_shapes = []
        for key, vec in tensors.items():
            if vec.dim() != 1:
                bad_shapes.append((key, tuple(vec.shape)))
        stats["hidden_keys"] = len(tensors)
        if bad_shapes:
            issues.append(f"hidden not 1D: {bad_shapes[:5]}")
    else:
        issues.append("missing hidden.safetensors")

    prefix_ids = {r["prefix_id"] for r in tables["prefixes"]}
    for name in ("token_features", "step_branches"):
        ids = {r["prefix_id"] for r in tables[name]}
        orphan = ids - prefix_ids
        if orphan:
            issues.append(f"{name} has orphan prefix_ids: {list(orphan)[:3]}")

    selected = [r for r in tables["prefixes"] if r.get("selected_for_rollout")]
    stats["selected_prefixes"] = len(selected)
    if tables["prefixes"] and not selected:
        issues.append("no prefix marked selected_for_rollout")

    rollout_ids = {r["prefix_id"] for r in tables["step_branches"]}
    unrolled_selected = [r["prefix_id"] for r in selected if r["prefix_id"] not in rollout_ids]
    if unrolled_selected:
        issues.append(f"selected prefixes missing rollout: {unrolled_selected[:3]}")

    api_rows = tables["api_annotations"]
    if api_rows:
        stats["api_annotation_rows"] = len(api_rows)
        modes = {}
        for r in api_rows:
            modes[r.get("annotation_mode", "?")] = modes.get(r.get("annotation_mode", "?"), 0) + 1
        stats["api_modes"] = modes
    elif tables["prefixes"]:
        issues.append("missing api_annotations (teacher not run?)")

    labels = _load_rows(data_dir / "labels.parquet")
    if labels:
        stats["labels_rows"] = len(labels)
        stats["branch_rate"] = sum(r.get("branch_label", 0) for r in labels) / len(labels)
    else:
        issues.append("missing labels.parquet — run postprocess_labels.py")

    ok = len(issues) == 0
    return {"ok": ok, "issues": issues, "stats": stats}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_self_check(args.data_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
