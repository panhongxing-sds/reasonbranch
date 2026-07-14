"""Parallel API labeling for uncertainty study prefixes."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from reasoning_branch_dataset.action_study.api_validity import ValidityClient
from reasoning_branch_dataset.action_study.diversity import compute_diversity, state_bucket
from reasoning_branch_dataset.action_study.prefix_substantiveness import prefix_substantiveness


def api_workers() -> int:
    return int(os.environ.get("DS_API_CONCURRENCY_LIMIT", os.environ.get("MAX_CONCURRENT", "96")))


def reasoning_only(prefix_text: str) -> str:
    if "</think>" in prefix_text:
        return prefix_text.split("</think>", 1)[1].strip()
    if "Problem:" in prefix_text:
        tail = prefix_text.split("Problem:", 1)[1]
        return tail[tail.find("\n") + 1 :] if "\n" in tail else tail
    return prefix_text


def label_one_prefix(
    client: ValidityClient,
    *,
    prefix: dict,
    question: str,
    gold_answer: str,
    steps: list[str],
) -> tuple[dict, dict, dict]:
    rp = reasoning_only(prefix["prefix_text"])
    val = client.label_prefix(
        prefix_id=prefix["prefix_id"],
        question=question,
        gold_answer=gold_answer,
        reasoning_prefix=rp,
    )
    cl: dict[str, Any] = {"prefix_id": prefix["prefix_id"], "clusters": [], "cluster_source": "none"}
    api_clusters = None
    api_num = None
    multi = None
    if steps and client.enabled:
        cl = client.cluster_next_steps(
            prefix_id=prefix["prefix_id"],
            question=question,
            reasoning_prefix=rp,
            next_steps=steps,
        )
        if cl.get("clusters"):
            api_clusters = cl["clusters"]
            api_num = cl.get("num_semantic_clusters")
            multi = cl.get("multiple_genuine_strategies")

    div = compute_diversity(
        steps,
        api_clusters=api_clusters,
        api_num_clusters=api_num,
        multiple_genuine=multi,
    )
    subst = prefix_substantiveness(prefix["prefix_text"], api_label=val.get("prefix_substantiveness"))
    state = state_bucket(val["prefix_validity"], div["diversity_label"])

    new_p = {
        **prefix,
        "prefix_validity": val["prefix_validity"],
        "prefix_status": val.get("prefix_status", val["prefix_validity"]),
        "prefix_substantiveness": subst,
        "include_in_main_experiment": subst == "SUBSTANTIVE",
        "validity_confidence": val.get("confidence", 0.0),
        "strategy_diversity": div["strategy_diversity"],
        "diversity_label": div["diversity_label"],
        "diversity_entropy": div["diversity_entropy"],
        "num_clusters": div["num_clusters"],
        "multiple_genuine_strategies": div.get("multiple_genuine_strategies", False),
        "cluster_source": div.get("cluster_source", "heuristic_conservative"),
        "state_bucket": state,
    }
    cl_out = {
        "prefix_id": prefix["prefix_id"],
        "clusters": cl.get("clusters") or div["cluster_labels"],
        "num_semantic_clusters": div["num_clusters"],
        "multiple_genuine_strategies": div.get("multiple_genuine_strategies", False),
        "cluster_source": div.get("cluster_source", "heuristic_conservative"),
    }
    return val, cl_out, new_p


def label_prefixes_parallel(
    client: ValidityClient,
    prefixes: list[dict],
    traces: dict[str, dict],
    next_by: dict[str, list],
    *,
    workers: int | None = None,
    progress_every: int = 50,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Label all prefixes with validity + strategy cluster API calls in parallel."""
    n_workers = workers or api_workers()
    validity_rows: list[dict] = []
    cluster_rows: list[dict] = []
    new_prefixes: list[dict] = []
    total = len(prefixes)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for p in prefixes:
            prob = traces[p["problem_id"]]
            steps = [x["text"] for x in sorted(next_by.get(p["prefix_id"], []), key=lambda x: x["sample_id"])]
            futures.append(
                pool.submit(
                    label_one_prefix,
                    client,
                    prefix=p,
                    question=prob["question"],
                    gold_answer=prob["gold_answer"],
                    steps=steps,
                )
            )
        for i, fut in enumerate(as_completed(futures), 1):
            val, cl, new_p = fut.result()
            validity_rows.append(val)
            cluster_rows.append(cl)
            new_prefixes.append(new_p)
            if progress_every and (i % progress_every == 0 or i == total):
                print(f"API labeled {i}/{total} (workers={n_workers})")

    new_prefixes.sort(key=lambda x: x["prefix_id"])
    validity_rows.sort(key=lambda x: x["prefix_id"])
    cluster_rows.sort(key=lambda x: x["prefix_id"])
    return validity_rows, cluster_rows, new_prefixes
