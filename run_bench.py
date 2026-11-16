#!/usr/bin/env python
"""Command-line runner for the paired benchmark.

Useful for accumulating samples across the OpenRouter free-tier daily reset,
because it can be driven from a script or cron job:

    # use whatever quota is left today
    .venv/bin/python run_bench.py

    # cap the spend explicitly
    .venv/bin/python run_bench.py --max-calls 40

    # use the paid variant of the same model (no daily cap)
    CLOUD_MODEL=nvidia/nemotron-3-ultra-550b-a55b .venv/bin/python run_bench.py

The run is resumable: items already present in results/raw_paired.jsonl are
skipped, so running it repeatedly accumulates the sample.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from bench import data, models, protocol, quota, runner  # noqa: E402
from bench.config import CLOUD_MODEL_ID, MODE, RAW_PATH, RESULTS_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="cap on cloud API calls this run (default: all remaining daily quota)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="comma-separated task names (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore previously recorded items and start over",
    )
    parser.add_argument(
        "--validate-e2e",
        type=int,
        default=0,
        help="also run N items through the real end-to-end hybrid path",
    )
    args = parser.parse_args()

    task_names = args.tasks.split(",") if args.tasks else None

    status = quota.check_quota()
    print(f"mode           : {MODE}")
    print(f"cloud model    : {CLOUD_MODEL_ID}")
    print(f"results        : {RESULTS_DIR}")
    print(f"quota          : {status.summary()}")

    if args.max_calls is not None:
        budget = args.max_calls
    else:
        budget = status.remaining if status.available else 0
    print(f"api call budget: {budget}")

    tasks = data.build_all(task_names)
    print(
        f"eval set       : {sum(len(t.items) for t in tasks)} items "
        f"across {len(tasks)} tasks"
    )
    if budget <= 0:
        print("\nNothing to do: no cloud quota available today.")
        print("Re-run after the UTC reset, or use the paid model id.")
        return 1

    local = models.LocalModel()
    cloud = models.CloudModel(CLOUD_MODEL_ID)
    hybrid = models.HybridModel(local, cloud)
    local.warm()
    print(f"local backend  : {local.backend_desc}")

    started = time.perf_counter()
    try:
        info = protocol.run_paired(
            tasks=tasks,
            local=local,
            cloud=cloud,
            hybrid=hybrid,
            max_api_calls=budget,
            limiter=quota.RateLimiter(),
            resume=not args.no_resume,
        )
        print(f"\n{info}")
        if args.validate_e2e:
            e2e = protocol.validate_hybrid(tasks, hybrid, n_items=args.validate_e2e)
            print(f"end-to-end hybrid validation items: {len(e2e)}")
    except quota.QuotaExhausted as exc:
        print(f"\n!! quota exhausted, stopping cleanly: {exc}")

    summary = runner.aggregate_paired()
    summary["last_run"] = info if "info" in dir() else {}
    runner.save_summary(summary)

    print(f"\nfinished in {time.perf_counter() - started:.0f}s")
    print(f"summary -> {runner.SUMMARY_PATH}")
    for task, cell in summary["per_task"].items():
        print(
            f"  {task:16} n={cell['n']:<4} "
            f"local={cell['local']['accuracy']} "
            f"cloud={cell['cloud']['accuracy']} "
            f"hybrid={cell['hybrid']['accuracy']} "
            f"oracle={cell['oracle']['accuracy']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
