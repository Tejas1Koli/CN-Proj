"""Paired per-item protocol.

Instead of running three independent passes over the dataset (one per system),
we evaluate every item ONCE for every backend that matters:

    local  ->  Gemma 4 E2B on this machine          (no API calls)
    cloud  ->  Nemotron-3-Ultra via OpenRouter       (1 API call per item)
    router ->  Laya's decision                       (no API calls)

The hybrid system is then *derived* from those same per-item results:

    hybrid picks the cloud answer when Laya routes to the cloud,
    and the local answer otherwise.

Why this matters:

  * It halves the API cost (one request per item instead of two).
  * Every system is scored on byte-identical items, so the comparison is
    exactly paired -- McNemar's test applies directly and item difficulty
    cancels out.
  * It yields the oracle-routing accuracy (best-of-both per item) for free,
    which is the quantity that says how much headroom a perfect router has.
  * The hybrid's latency is derived exactly as
    `decision_ms + backend_latency`, matching what the real service does.

`validate_hybrid` runs the hybrid system end to end on a small subset so the
derived numbers can be checked against the real service.
"""

from __future__ import annotations

import time
from typing import Any

from .data import Task
from .models import CloudModel, HybridModel, LocalModel, result_to_dict
from .quota import Quota, QuotaExhausted, RateLimiter


def _score(task: Task, text: str, item) -> tuple[bool, str | None]:
    try:
        return bool(task.scorer(text, item)), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


def run_paired(
    tasks: list[Task],
    local: LocalModel,
    cloud: CloudModel,
    hybrid: HybridModel,
    max_api_calls: int | None = None,
    limiter: RateLimiter | None = None,
    resume: bool = True,
    raw_path=None,
    progress_every: int = 10,
) -> dict:
    """Evaluate the paired protocol over `tasks`, respecting the daily quota.

    `max_api_calls` caps how many cloud requests this run may make. Items are
    consumed round-robin across tasks so that a partially-completed day still
    gives balanced coverage of every task.
    """
    from .runner import _append_raw, _load_raw

    limiter = limiter or RateLimiter()
    raw_path = raw_path or _default_raw_path()

    existing = _load_raw(raw_path) if resume else []
    # The prompt fingerprint is part of the key, so editing a prompt
    # invalidates that item's old record instead of mixing two prompts.
    done = {
        (r["item_id"], r.get("prompt_hash"))
        for r in existing
        if r.get("protocol") == "paired"
    }

    # `max_api_calls` is the number of cloud requests THIS run may make.
    # The authoritative signal for "no quota left" is OpenRouter's own 429,
    # which surfaces as QuotaExhausted below; the budget only stops us from
    # starting a run we already know cannot finish.
    budget = max_api_calls
    if budget is None:
        budget = float("inf")
    if budget <= 0:
        raise QuotaExhausted("no cloud-request budget for this run")

    # Round-robin the item queues so partial days stay balanced across tasks.
    queues = {task.name: list(task.items) for task in tasks}
    order: list[tuple[Task, Any]] = []
    while any(queues.values()):
        for task in tasks:
            if queues[task.name]:
                order.append((task, queues[task.name].pop(0)))

    calls_made = 0
    completed = 0
    started = time.perf_counter()
    peak_rss_mb = 0.0

    from .hardware import MemorySampler

    sampler = MemorySampler()
    sampler.__enter__()

    for task, item in order:
        if (item.item_id, item.prompt_hash) in done:
            continue
        if calls_made >= budget:
            break

        # 1. Local answer (no API).
        try:
            local_result = local.generate(item.prompt, task.max_output_tokens)
            local_correct, local_err = _score(task, local_result.text, item)
        except Exception as exc:
            local_result = None
            local_correct, local_err = False, f"{type(exc).__name__}: {exc}"[:200]

        # 2. Cloud answer (1 API call, rate limited).
        cloud_result = None
        cloud_correct = False
        cloud_err = None
        try:
            limiter.wait()
            cloud_result = cloud.generate(item.prompt, task.max_output_tokens)
            cloud_correct, cloud_err = _score(task, cloud_result.text, item)
            calls_made += 1
        except QuotaExhausted:
            raise
        except Exception as exc:
            cloud_err = f"{type(exc).__name__}: {exc}"[:300]
            message = str(exc)
            if "free-models-per-day" in message:
                raise QuotaExhausted(message) from exc

        # 3. Router decision (no API). The decision time is measured here
        #    because it is pure overhead the hybrid adds on top of whichever
        #    backend it selects.
        decision_start = time.perf_counter()
        try:
            route, reason = hybrid.decide(item.prompt)
        except Exception as exc:
            reason = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            route = "local"
        decision_ms = (time.perf_counter() - decision_start) * 1000

        # 4. Derive the hybrid outcome from the same two answers.
        if route == "cloud":
            hybrid_correct = cloud_correct
            backend = cloud_result
            hybrid_text = cloud_result.text if cloud_result else ""
        else:
            hybrid_correct = local_correct
            backend = local_result
            hybrid_text = local_result.text if local_result else ""

        hybrid_latency = decision_ms + (backend.total_ms if backend else 0.0)

        record = {
            "protocol": "paired",
            "task": task.name,
            "item_id": item.item_id,
            "prompt_hash": item.prompt_hash,
            "local_correct": local_correct,
            "cloud_correct": cloud_correct,
            "hybrid_correct": hybrid_correct,
            "oracle_correct": bool(local_correct or cloud_correct),
            "route": route,
            "reason": reason,
            "decision_ms": round(decision_ms, 2),
            "hybrid_latency_ms": round(hybrid_latency, 2),
            "local_latency_ms": local_result.total_ms if local_result else None,
            "cloud_latency_ms": cloud_result.total_ms if cloud_result else None,
            "local_ttft_ms": local_result.ttft_ms if local_result else None,
            "cloud_ttft_ms": cloud_result.ttft_ms if cloud_result else None,
            "local_output_tokens": local_result.output_tokens if local_result else 0,
            "cloud_output_tokens": cloud_result.output_tokens if cloud_result else 0,
            "local_tok_per_s": local_result.output_tok_per_s if local_result else 0,
            "cloud_tok_per_s": cloud_result.output_tok_per_s if cloud_result else 0,
            "cloud_cost_usd": cloud_result.cost_usd if cloud_result else 0.0,
            "cloud_attempted": cloud_result is not None,
            "local_attempted": local_result is not None,
            "paired_complete": bool(local_result is not None and cloud_result is not None),
            "local_error": local_err,
            "cloud_error": cloud_err,
            "local_text": local_result.text if local_result else "",
            "cloud_text": cloud_result.text if cloud_result else "",
        }
        _append_raw(record, raw_path)
        completed += 1

        if progress_every and completed % progress_every == 0:
            rate = completed / max(1e-9, time.perf_counter() - started)
            print(
                f"    paired {completed:4d} items "
                f"({calls_made} API calls, {rate:.2f} items/s)",
                flush=True,
            )

    sampler.__exit__(None, None, None)
    peak_rss_mb = round(sampler.peak_rss_mb, 1)

    return {
        "items_completed": completed,
        "api_calls_made": calls_made,
        "items_skipped_existing": len(done),
        "budget": budget if budget != float("inf") else None,
        "peak_rss_mb": peak_rss_mb,
    }


def validate_hybrid(
    tasks: list[Task],
    hybrid: HybridModel,
    n_items: int = 10,
    raw_path=None,
) -> list[dict]:
    """Run the hybrid system end to end on a few items, for validation.

    The derived hybrid accuracy should match this measured accuracy; if it does
    not, the derivation in `run_paired` is wrong.
    """
    from .runner import _append_raw, _load_raw

    raw_path = raw_path or _default_raw_path()
    existing = _load_raw(raw_path)
    done = {r["item_id"] for r in existing if r.get("protocol") == "hybrid_e2e"}

    items: list[tuple[Task, Any]] = []
    for task in tasks:
        for item in task.items:
            items.append((task, item))
    items = items[:n_items]

    out = []
    for task, item in items:
        if item.item_id in done:
            continue
        result = hybrid.generate(item.prompt, task.max_output_tokens)
        try:
            correct = bool(task.scorer(result.text, item))
        except Exception:
            correct = False
        record = {
            "protocol": "hybrid_e2e",
            "task": task.name,
            "item_id": item.item_id,
            "prompt_hash": item.prompt_hash,
            "correct": correct,
            "route": result.route,
            "total_ms": result.total_ms,
            "decision_ms": result.extra.get("decision_ms"),
            "text": result.text[:400],
            **result_to_dict(result),
        }
        _append_raw(record, raw_path)
        out.append(record)
    return out


def _default_raw_path():
    from .config import RAW_PATH

    return RAW_PATH
