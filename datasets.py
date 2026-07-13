"""Load benchmark problems for trace collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from reasoning_branch_dataset.config import DATA_PATHS, WORKSPACE
from reasoning_branch_dataset.action_study.visual_input import assess_visual_input


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_math500(limit: int | None = None, offset: int = 0) -> list[dict]:
    path = DATA_PATHS["math500"]
    rows = []
    for i, row in enumerate(_iter_jsonl(path)):
        if i < offset:
            continue
        if limit is not None and len(rows) >= limit:
            break
        rows.append(
            {
                "problem_id": f"math500_{i:04d}",
                "dataset": "math500",
                "question": row["problem"],
                "gold_answer": row.get("answer") or row.get("solution", ""),
                "unique_id": row.get("unique_id", str(i)),
            }
        )
    return rows


def load_aime(limit: int | None = None) -> list[dict]:
    path = DATA_PATHS["aime"]
    rows: list[dict] = []

    if path.exists() and path.suffix == ".jsonl":
        for i, row in enumerate(_iter_jsonl(path)):
            if limit is not None and i >= limit:
                break
            problem = row.get("problem") or row.get("question") or row.get("Problem")
            answer = row.get("answer") or row.get("Answer") or ""
            rows.append(
                {
                    "problem_id": f"aime_{i:04d}",
                    "dataset": "aime",
                    "question": problem,
                    "gold_answer": str(answer),
                    "unique_id": str(i),
                }
            )
        return rows

    legacy = (
        WORKSPACE
        / "specreason"
        / "hf_cache"
        / "datasets"
        / "HuggingFaceH4___aime_2024"
        / "default"
        / "0.0.0"
        / "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
    )
    if legacy.exists():
        arrow_files = sorted(legacy.glob("*.arrow"))
        if arrow_files:
            try:
                import pyarrow.ipc as ipc

                with arrow_files[0].open("rb") as f:
                    table = ipc.open_stream(f).read_all()
                for i, row in enumerate(table.to_pylist()):
                    if limit is not None and i >= limit:
                        break
                    problem = row.get("problem") or row.get("question") or row.get("Problem")
                    answer = row.get("answer") or row.get("Answer") or ""
                    rows.append(
                        {
                            "problem_id": f"aime_{i:04d}",
                            "dataset": "aime",
                            "question": problem,
                            "gold_answer": str(answer),
                            "unique_id": str(i),
                        }
                    )
                return rows
            except Exception:
                pass

    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            rows.append(
                {
                    "problem_id": f"aime_{i:04d}",
                    "dataset": "aime",
                    "question": row["problem"],
                    "gold_answer": str(row.get("answer", "")),
                    "unique_id": str(i),
                }
            )
        return rows
    except Exception as exc:
        raise RuntimeError(f"AIME dataset unavailable at {path}") from exc


def load_gsm8k(limit: int | None = None) -> list[dict]:
    path = DATA_PATHS["gsm8k"]
    rows: list[dict] = []

    if path.exists():
        for i, row in enumerate(_iter_jsonl(path)):
            if limit is not None and i >= limit:
                break
            question = row.get("question") or row.get("problem", "")
            answer = row.get("answer") or row.get("gold_answer", "")
            if "####" in str(answer):
                answer = str(answer).split("####")[-1].strip()
            rows.append(
                {
                    "problem_id": f"gsm8k_{i:04d}",
                    "dataset": "gsm8k",
                    "question": question,
                    "gold_answer": str(answer).strip(),
                    "unique_id": str(i),
                }
            )
        return rows

    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            ans = row["answer"]
            if "####" in ans:
                ans = ans.split("####")[-1].strip()
            rows.append(
                {
                    "problem_id": f"gsm8k_{i:04d}",
                    "dataset": "gsm8k",
                    "question": row["question"],
                    "gold_answer": ans,
                    "unique_id": str(i),
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps({"question": r["question"], "answer": r["gold_answer"]}) + "\n")
        return rows
    except Exception as exc:
        raise RuntimeError(f"GSM8K unavailable at {path}") from exc


def load_deepscaler(limit: int | None = None, offset: int = 0) -> list[dict]:
    path = DATA_PATHS["deepscaler"]
    rows: list[dict] = []

    if path.exists():
        for i, row in enumerate(_iter_jsonl(path)):
            if i < offset:
                continue
            if limit is not None and len(rows) >= limit:
                break
            answer = str(row.get("answer") or "").strip()
            question = row["problem"]
            visual = assess_visual_input(question)
            rows.append(
                {
                    "problem_id": f"deepscaler_{i:05d}",
                    "dataset": "deepscaler",
                    "question": question,
                    "gold_answer": answer,
                    "unique_id": str(i),
                    **visual,
                }
            )
        return rows

    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from datasets import load_dataset

        ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for i, row in enumerate(ds):
                rec = {
                    "problem": row["problem"],
                    "answer": row.get("answer", ""),
                    "solution": row.get("solution", ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i < offset:
                    continue
                if limit is not None and len(rows) >= limit:
                    continue
                question = row["problem"]
                visual = assess_visual_input(question)
                rows.append(
                    {
                        "problem_id": f"deepscaler_{i:05d}",
                        "dataset": "deepscaler",
                        "question": question,
                        "gold_answer": str(row.get("answer", "")).strip(),
                        "unique_id": str(i),
                        **visual,
                    }
                )
        if offset:
            rows = []
            for i, row in enumerate(_iter_jsonl(path)):
                if i < offset:
                    continue
                if limit is not None and len(rows) >= limit:
                    break
                rows.append(
                    {
                        "problem_id": f"deepscaler_{i:05d}",
                        "dataset": "deepscaler",
                        "question": row["problem"],
                        "gold_answer": str(row.get("answer", "")).strip(),
                        "unique_id": str(i),
                        **assess_visual_input(row["problem"]),
                    }
                )
        return rows
    except Exception as exc:
        raise RuntimeError(f"DeepScaler unavailable at {path}") from exc


LOADERS = {
    "math500": load_math500,
    "gsm8k": load_gsm8k,
    "aime": load_aime,
    "deepscaler": load_deepscaler,
}


def load_problems(dataset: str, limit: int | None = None, offset: int = 0) -> list[dict]:
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}")
    loader = LOADERS[dataset]
    if dataset == "math500":
        return loader(limit=limit, offset=offset)
    if dataset == "deepscaler":
        return loader(limit=limit, offset=offset)
    return loader(limit=limit)
