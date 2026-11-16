"""Figures for the hybrid benchmark. Each function saves a PNG and returns the path."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR, MODEL_LABELS

TASK_LABELS = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "boolq": "BoolQ",
    "arc_challenge": "ARC-C",
}

ARM_COLORS = {
    "local": "#4C78A8",
    "cloud": "#F58518",
    "hybrid": "#54A24B",
    "oracle": "#B279A2",
}

ARM_LABELS = {
    "local": "Local only\n(Gemma 4 E2B)",
    "cloud": "Cloud only\n(Nemotron-3-Ultra)",
    "hybrid": "Hybrid\n(Laya router)",
    "oracle": "Oracle routing\n(upper bound)",
}


def _out(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


def _order(tasks: list[str]) -> list[str]:
    preferred = ["mmlu", "gsm8k", "arc_challenge", "boolq", "humaneval", "mbpp"]
    ordered = [t for t in preferred if t in tasks]
    ordered += [t for t in tasks if t not in ordered]
    return ordered


def accuracy_by_task(summary: dict) -> Path:
    """Grouped accuracy bars with 95% Wilson intervals."""
    per_task = summary["per_task"]
    tasks = _order(list(per_task))
    arms = ["local", "cloud", "hybrid", "oracle"]
    labels = [TASK_LABELS.get(t, t) for t in tasks]

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.2
    x = np.arange(len(tasks))

    for i, arm in enumerate(arms):
        values, errors = [], []
        for task in tasks:
            cell = per_task[task][arm]
            values.append((cell["accuracy"] or 0) * 100)
            lo, hi = cell["accuracy_ci95"]
            value = (cell["accuracy"] or 0) * 100
            errors.append([value - lo * 100, hi * 100 - value])
        errors = np.array(errors).T
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=ARM_LABELS[arm],
            color=ARM_COLORS[arm],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by task - error bars are 95% Wilson intervals")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    path = _out("accuracy_by_task.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def latency_by_task(summary: dict) -> Path:
    """Median end-to-end latency per task and per arm."""
    per_task = summary["per_task"]
    tasks = _order(list(per_task))

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.26
    x = np.arange(len(tasks))

    for i, arm in enumerate(["local", "cloud", "hybrid"]):
        values = []
        for task in tasks:
            median = per_task[task]["latency_ms"][arm]["median"]
            values.append((median or 0) / 1000)
        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=ARM_LABELS[arm],
            color=ARM_COLORS[arm],
            edgecolor="white",
        )
        for xi, value in zip(x + (i - 1) * width, values):
            ax.text(xi, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks])
    ax.set_ylabel("Median end-to-end latency (s)")
    ax.set_title("End-to-end latency (hybrid includes Laya routing overhead)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    path = _out("latency_by_task.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def route_breakdown(summary: dict) -> Path:
    """How often the router chose each tier, per task."""
    per_task = summary["per_task"]
    tasks = _order(list(per_task))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    local_share, cloud_share = [], []
    for task in tasks:
        fractions = per_task[task]["route_fraction"]
        local_share.append(fractions.get("local", 0) * 100)
        cloud_share.append(fractions.get("cloud", 0) * 100)

    x = np.arange(len(tasks))
    ax.bar(x, local_share, 0.55, label="routed to local", color=ARM_COLORS["local"])
    ax.bar(x, cloud_share, 0.55, bottom=local_share, label="routed to cloud",
           color=ARM_COLORS["cloud"])
    for xi, (lo, cl) in enumerate(zip(local_share, cloud_share)):
        ax.text(xi, lo / 2, f"{lo:.0f}%", ha="center", va="center", color="white", fontsize=9)
        ax.text(xi, lo + cl / 2, f"{cl:.0f}%", ha="center", va="center", color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks])
    ax.set_ylabel("Share of items (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Laya routing decisions per task")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    path = _out("route_breakdown.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def router_scatter(records: list[dict]) -> Path:
    """Does Laya's difficulty score separate items the local model gets wrong?"""
    rows = [
        r
        for r in records
        if r.get("protocol") == "paired" and r.get("reason")
    ]
    if not rows:
        raise ValueError("no paired records to plot")

    difficulty = [float(r["reason"].get("difficulty", 0.0)) for r in rows]
    local_ok = [1.0 if r["local_correct"] else 0.0 for r in rows]
    routed_cloud = [1.0 if r.get("route") == "cloud" else 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    jitter = (np.random.default_rng(0).random(len(rows)) - 0.5) * 0.08
    ax.scatter(
        np.array(difficulty) + jitter,
        local_ok + np.array(routed_cloud) * 2.2 + jitter,
        s=28,
        alpha=0.75,
        color=[ARM_COLORS["cloud"] if c else ARM_COLORS["local"] for c in routed_cloud],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(
        ["local wrong\n(local)", "local right\n(local)",
         "local wrong\n(cloud)", "local right\n(cloud)"],
        fontsize=8,
    )
    ax.set_xlabel("Laya difficulty score")
    ax.set_title("Router behaviour vs. local-model correctness")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = _out("router_scatter.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def latency_distribution(records: list[dict]) -> Path:
    """Per-arm end-to-end latency distribution (log scale).

    On this machine the cloud API is often *faster* than local generation for
    long answers, which is the opposite of the usual assumption - so this is
    the most surprising hardware finding and worth plotting raw.
    """
    rows = [r for r in records if r.get("protocol") == "paired"]
    arms = {
        "local": [r["local_latency_ms"] / 1000 for r in rows if r.get("local_latency_ms")],
        "cloud": [r["cloud_latency_ms"] / 1000 for r in rows if r.get("cloud_latency_ms")],
        "hybrid": [r["hybrid_latency_ms"] / 1000 for r in rows if r.get("hybrid_latency_ms")],
    }
    arms = {k: v for k, v in arms.items() if v}
    if not arms:
        raise ValueError("no paired latency records")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    positions = range(len(arms))
    parts = ax.boxplot(
        arms.values(),
        positions=list(positions),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for patch, (_, values) in zip(parts["boxes"], arms.items()):
        patch.set_facecolor(ARM_COLORS[_])
        patch.set_alpha(0.7)
    for i, (name, values) in enumerate(arms.items()):
        jitter = (np.random.default_rng(0).random(len(values)) - 0.5) * 0.18
        ax.scatter(np.full(len(values), i) + jitter, values, s=16, alpha=0.5,
                   color="black", zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([ARM_LABELS[k] for k in arms], fontsize=8)
    ax.set_ylabel("End-to-end latency (s, log scale)")
    ax.set_title("Latency distribution by arm")
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    path = _out("latency_distribution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def accuracy_vs_cost(summary: dict) -> Path:
    """Accuracy against cloud spend - the hybrid's cost/quality trade-off."""
    per_task = summary["per_task"]
    tasks = _order(list(per_task))

    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, marker in [("local", "o"), ("cloud", "s"), ("hybrid", "D")]:
        xs, ys = [], []
        for task in tasks:
            cell = per_task[task]
            cost = cell["total_cost_usd"] if arm == "cloud" else 0.0
            if arm == "local":
                cost = 0.0
            xs.append(cost)
            ys.append((cell[arm]["accuracy"] or 0) * 100)
        ax.scatter(xs, ys, marker=marker, s=60, label=ARM_LABELS[arm],
                   color=ARM_COLORS[arm], edgecolor="white")
        for x, y, task in zip(xs, ys, tasks):
            ax.annotate(TASK_LABELS.get(task, task), (x, y), fontsize=7,
                        textcoords="offset points", xytext=(5, 4))
    ax.set_xlabel("Cloud spend for the task (USD)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Quality vs. cloud spend")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = _out("accuracy_vs_cost.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
