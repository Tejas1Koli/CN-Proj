"""Run the benchmark and aggregate the results.

Every generation is appended to results/raw_results.jsonl as it completes, so
a long run can be interrupted and resumed without losing work.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import (
    CLOUD_MODEL_ID,
    FIGURES_DIR,
    MODE,
    MODEL_LABELS,
    RAW_PATH,
    RESULTS_DIR,
    SUMMARY_PATH,
    SYSTEM_RAW_PATH,
)
from .data import Task
from .hardware import MemorySampler, current_rss_mb, directory_size_mb, profile_host
from .models import build_systems, result_to_dict
from .stats import bootstrap_ci, mcnemar_exact, summarize, wilson_interval


# ------------------------------------------------------------------ raw I/O
def _load_raw(path: Path = SYSTEM_RAW_PATH) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _append_raw(record: dict, path: Path = SYSTEM_RAW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def clear_raw(path: Path = SYSTEM_RAW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


# ------------------------------------------------------------------- runner
def run_task(
    system: Any,
    task: Task,
    resume: bool = True,
    raw_path: Path = SYSTEM_RAW_PATH,
    progress_every: int = 10,
) -> list[dict]:
    """Run one system over one task. Returns the records produced."""
    existing = _load_raw(raw_path) if resume else []
    done = {
        (r["system"], r["task"], r["item_id"], r.get("prompt_hash"))
        for r in existing
        if r.get("task") == task.name and r.get("system") == system.name
    }

    produced: list[dict] = []
    started = time.perf_counter()
    for index, item in enumerate(task.items, start=1):
        key = (system.name, task.name, item.item_id, item.prompt_hash)
        if key in done:
            continue

        try:
            result = system.generate(item.prompt, task.max_output_tokens)
            error = None
        except Exception as exc:  # keep the run alive, record the failure
            result = None
            error = f"{type(exc).__name__}: {exc}"[:300]

        if result is None:
            record = {
                "system": system.name,
                "task": task.name,
                "item_id": item.item_id,
                "prompt_hash": item.prompt_hash,
                "correct": False,
                "error": error,
                "text": "",
                "route": "error",
            }
        else:
            try:
                correct = bool(task.scorer(result.text, item))
                score_error = None
            except Exception as exc:
                correct, score_error = False, f"{type(exc).__name__}: {exc}"[:200]
            record = {
                "system": system.name,
                "task": task.name,
                "item_id": item.item_id,
                "prompt_hash": item.prompt_hash,
                "correct": correct,
                "error": error or score_error,
                **result_to_dict(result),
                "meta": item.meta,
            }

        _append_raw(record, raw_path)
        produced.append(record)

        if progress_every and index % progress_every == 0:
            rate = index / max(1e-9, time.perf_counter() - started)
            print(
                f"    {system.name:7} {task.name:15} {index:4d}/{len(task.items):<4d} "
                f"({rate:.2f} items/s)",
                flush=True,
            )
    return produced


def run_all(
    systems: dict[str, Any],
    tasks: list[Task],
    resume: bool = True,
    raw_path: Path = SYSTEM_RAW_PATH,
) -> None:
    for name, system in systems.items():
        print(f"\n=== {name} ===", flush=True)
        if name in ("local", "hybrid"):
            system.local.warm() if hasattr(system, "local") else system.warm()
        for task in tasks:
            run_task(system, task, resume=resume, raw_path=raw_path)


# -------------------------------------------------------------- aggregation
def _dedupe(records: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """Keep the newest record per key.

    A prompt change re-runs an item, leaving two records with the same item id
    but different prompt hashes. The later record is the current one.
    """
    latest: dict[tuple, dict] = {}
    for record in records:
        latest[tuple(record.get(f) for f in key_fields)] = record
    return list(latest.values())


def aggregate(raw_path: Path = SYSTEM_RAW_PATH) -> dict:
    records = _dedupe(_load_raw(raw_path), ("system", "task", "item_id"))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["system"], record["task"])].append(record)

    per_cell: dict[str, dict] = {}
    for (system, task), rows in sorted(grouped.items()):
        valid = [r for r in rows if not r.get("error")]
        n = len(valid)
        correct = sum(1 for r in valid if r.get("correct"))
        lo, hi = wilson_interval(correct, n)

        latencies = [r["total_ms"] for r in valid if r.get("total_ms")]
        ttfts = [r["ttft_ms"] for r in valid if r.get("ttft_ms")]
        throughput = [
            r["output_tok_per_s"] for r in valid if r.get("output_tok_per_s")
        ]
        cost = sum(r.get("cost_usd", 0.0) or 0.0 for r in valid)

        routes: dict[str, int] = defaultdict(int)
        for record in valid:
            routes[record.get("route", "?")] += 1

        per_cell[f"{system}|{task}"] = {
            "system": system,
            "task": task,
            "n": n,
            "n_errors": len(rows) - n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else None,
            "accuracy_ci95": [round(lo, 4), round(hi, 4)],
            "latency_ms": summarize(latencies),
            "ttft_ms": summarize(ttfts),
            "output_tok_per_s": summarize(throughput),
            "total_cost_usd": round(cost, 6),
            "routes": dict(routes),
        }

    # Paired comparisons on the shared item ids.
    def paired(system_a: str, system_b: str, task: str) -> dict | None:
        a = {
            r["item_id"]: r
            for r in records
            if r["system"] == system_a and r["task"] == task
        }
        b = {
            r["item_id"]: r
            for r in records
            if r["system"] == system_b and r["task"] == task
        }
        shared = sorted(set(a) & set(b))
        if not shared:
            return None
        a_only = sum(1 for i in shared if a[i]["correct"] and not b[i]["correct"])
        b_only = sum(1 for i in shared if b[i]["correct"] and not a[i]["correct"])
        both = sum(1 for i in shared if a[i]["correct"] and b[i]["correct"])
        neither = len(shared) - a_only - b_only - both
        return {
            "task": task,
            "n_shared": len(shared),
            f"{system_a}_only_correct": a_only,
            f"{system_b}_only_correct": b_only,
            "both_correct": both,
            "neither_correct": neither,
            "mcnemar_p": round(mcnemar_exact(a_only, b_only), 6),
        }

    comparisons = []
    for task in sorted({t for _, t in grouped}):
        for pair in [("cloud", "local"), ("hybrid", "local"), ("hybrid", "cloud")]:
            result = paired(*pair, task)
            if result:
                comparisons.append({"a": pair[0], "b": pair[1], **result})

    # Oracle / random-routing baselines for the hybrid system.
    oracle: dict[str, dict] = {}
    for task in sorted({t for _, t in grouped}):
        local_rows = {
            r["item_id"]: r["correct"]
            for r in records
            if r["system"] == "local" and r["task"] == task
        }
        cloud_rows = {
            r["item_id"]: r["correct"]
            for r in records
            if r["system"] == "cloud" and r["task"] == task
        }
        hybrid_rows = {
            r["item_id"]: r
            for r in records
            if r["system"] == "hybrid" and r["task"] == task
        }
        shared = sorted(set(local_rows) & set(cloud_rows) & set(hybrid_rows))
        if not shared:
            continue
        best = sum(1 for i in shared if local_rows[i] or cloud_rows[i])
        oracle[task] = {
            "n": len(shared),
            "oracle_accuracy": round(best / len(shared), 4),
            "local_accuracy": round(
                sum(local_rows[i] for i in shared) / len(shared), 4
            ),
            "cloud_accuracy": round(
                sum(cloud_rows[i] for i in shared) / len(shared), 4
            ),
            "hybrid_accuracy": round(
                sum(hybrid_rows[i]["correct"] for i in shared) / len(shared), 4
            ),
        }

    return {
        "mode": MODE,
        "cloud_model": CLOUD_MODEL_ID,
        "model_labels": MODEL_LABELS,
        "host": vars(profile_host()),
        "local_model_dir_mb": directory_size_mb(
            str(Path.home() / ".litert-lm" / "models" / "gemma4-e2b")
        ),
        "rss_mb_at_end": round(current_rss_mb(), 1),
        "cells": per_cell,
        "comparisons": comparisons,
        "oracle": oracle,
    }


def save_summary(summary: dict, path: Path = SUMMARY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))


# ------------------------------------------------------- paired aggregation
def aggregate_paired(raw_path: Path = RAW_PATH) -> dict:
    """Aggregate the paired protocol into the tables the notebook renders."""
    records = _dedupe(
        [r for r in _load_raw(raw_path) if r.get("protocol") == "paired"],
        ("task", "item_id"),
    )

    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)

    per_task = {}
    for task, rows in sorted(by_task.items()):
        # All four arms are scored over the SAME items - the ones where both
        # backends produced an answer. Comparing over different denominators
        # would not be a paired comparison.
        complete = [
            r
            for r in rows
            if r.get("paired_complete")
            or (r.get("local_attempted", True) and r.get("cloud_attempted"))
        ]
        n = len(complete)

        local_c = sum(1 for r in complete if r["local_correct"])
        cloud_c = sum(1 for r in complete if r["cloud_correct"])
        hybrid_c = sum(1 for r in complete if r["hybrid_correct"])
        oracle_c = sum(1 for r in complete if r["oracle_correct"])

        def block(correct: int, denom: int) -> dict:
            lo, hi = wilson_interval(correct, denom)
            return {
                "n": denom,
                "correct": correct,
                "accuracy": round(correct / denom, 4) if denom else None,
                "accuracy_ci95": [round(lo, 4), round(hi, 4)],
            }

        routes: dict[str, int] = defaultdict(int)
        for row in rows:
            routes[row.get("route", "?")] += 1

        local_lat = [r["local_latency_ms"] for r in rows if r.get("local_latency_ms")]
        cloud_lat = [r["cloud_latency_ms"] for r in rows if r.get("cloud_latency_ms")]
        hybrid_lat = [
            r["hybrid_latency_ms"] for r in complete if r.get("hybrid_latency_ms")
        ]
        decisions = [r["decision_ms"] for r in rows if r.get("decision_ms") is not None]

        per_task[task] = {
            "n": n,
            "n_incomplete": len(rows) - n,
            "local": block(local_c, n),
            "cloud": block(cloud_c, n),
            "hybrid": block(hybrid_c, n),
            "oracle": block(oracle_c, n),
            "routes": dict(routes),
            "route_fraction": {
                k: round(v / n, 4) for k, v in routes.items()
            }
            if n
            else {},
            "latency_ms": {
                "local": summarize(local_lat),
                "cloud": summarize(cloud_lat),
                "hybrid": summarize(hybrid_lat),
                "decision": summarize(decisions),
            },
            "tok_per_s": {
                "local": summarize(
                    [r["local_tok_per_s"] for r in rows if r.get("local_tok_per_s")]
                ),
                "cloud": summarize(
                    [r["cloud_tok_per_s"] for r in rows if r.get("cloud_tok_per_s")]
                ),
            },
            "cloud_output_tokens_total": sum(
                r.get("cloud_output_tokens", 0) or 0 for r in complete
            ),
        }

    # Paired significance tests over the shared items.
    comparisons = []
    for task, rows in sorted(by_task.items()):
        complete = [
            r
            for r in rows
            if r.get("paired_complete")
            or (r.get("local_attempted", True) and r.get("cloud_attempted"))
        ]
        hybrid_only = sum(
            1 for r in complete if r["hybrid_correct"] and not r["local_correct"]
        )
        local_only = sum(
            1 for r in complete if r["local_correct"] and not r["hybrid_correct"]
        )
        cloud_only = sum(
            1 for r in complete if r["cloud_correct"] and not r["local_correct"]
        )
        local_wins = sum(
            1 for r in complete if r["local_correct"] and not r["cloud_correct"]
        )
        comparisons.append(
            {
                "task": task,
                "n": len(complete),
                "hybrid_vs_local": {
                    "hybrid_only_correct": hybrid_only,
                    "local_only_correct": local_only,
                    "p_value": round(mcnemar_exact(hybrid_only, local_only), 6),
                },
                "cloud_vs_local": {
                    "cloud_only_correct": cloud_only,
                    "local_only_correct": local_wins,
                    "p_value": round(mcnemar_exact(cloud_only, local_wins), 6),
                },
            }
        )

    # How good is the router? Compare it against oracle routing.
    router = {}
    for task, rows in sorted(by_task.items()):
        complete = [
            r
            for r in rows
            if r.get("paired_complete")
            or (r.get("local_attempted", True) and r.get("cloud_attempted"))
        ]
        n = len(complete)
        if not n:
            continue
        hybrid_c = sum(1 for r in rows if r["hybrid_correct"])
        oracle_c = sum(1 for r in rows if r["oracle_correct"])
        cloud_c = sum(1 for r in rows if r["cloud_correct"])
        local_c = sum(1 for r in rows if r["local_correct"])

        # Items where the two backends disagree are the only ones routing can fix.
        disagreements = [
            r for r in complete if r["local_correct"] != r["cloud_correct"]
        ]
        router[task] = {
            "n": n,
            "n_disagreements": len(disagreements),
            "disagreement_rate": round(len(disagreements) / n, 4),
            "hybrid_accuracy": round(hybrid_c / n, 4),
            "oracle_accuracy": round(oracle_c / n, 4),
            "local_accuracy": round(local_c / n, 4),
            "cloud_accuracy": round(cloud_c / n, 4),
            # Fraction of the oracle headroom the router actually captures.
            "oracle_gap_captured": round(
                (hybrid_c / n - local_c / n) / (oracle_c / n - local_c / n), 4
            )
            if oracle_c != local_c
            else None,
            # Would routing everything to the cloud have been better?
            "router_vs_cloud": round(hybrid_c / n - cloud_c / n, 4),
        }

    return {
        "mode": MODE,
        "cloud_model": CLOUD_MODEL_ID,
        "model_labels": MODEL_LABELS,
        "host": vars(profile_host()),
        "local_model_dir_mb": directory_size_mb(
            str(Path.home() / ".litert-lm" / "models" / "gemma4-e2b")
        ),
        "rss_mb_at_end": round(current_rss_mb(), 1),
        "total_items": len(records),
        "per_task": per_task,
        "comparisons": comparisons,
        "router": router,
    }


def load_summary(path: Path = SUMMARY_PATH) -> dict:
    return json.loads(path.read_text())


def figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def latency_bootstrap(records: list[dict], group_a: str, group_b: str) -> tuple:
    a = [
        r["total_ms"]
        for r in records
        if r["system"] == group_a and r.get("total_ms")
    ]
    b = [
        r["total_ms"]
        for r in records
        if r["system"] == group_b and r.get("total_ms")
    ]
    return bootstrap_ci(a), bootstrap_ci(b)
