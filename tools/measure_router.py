"""Measure Laya routing latency in two conditions and save both.

The router is a ModernBERT encoder that never generates a token, so it is easy
to assume it is free. This script checks that assumption.

Phase A (idle engine): predictions back-to-back after the local model is loaded
but has not generated anything. This is the "router alone on the machine" case.

Phase B (deployed path): every decision follows a short local generation, which
is what app.py does -- the router and the local LLM share one accelerator.
Phase B runs second because once the local engine has been exercised the
machine does not return to the phase-A state.

Writes results/router_latency_clean.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import data  # noqa: E402
from bench.models import HybridModel, LocalModel  # noqa: E402


def stats(values):
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values)),
        "mean_ms": round(statistics.mean(values)),
        "min_ms": round(min(values)),
        "max_ms": round(max(values)),
        "total_s": round(sum(values) / 1000, 2),
    }


def main() -> int:
    items = []
    for task in data.build_all():
        for item in task.items:
            items.append((task.name, item.item_id, item.prompt))

    local = LocalModel()
    local.warm()
    hybrid = HybridModel(local, None)

    import laya

    questions = laya.router_questions()
    probe = {"request": "What is the capital of France?"}

    hybrid.router.predict(probe, questions)  # load Laya weights, then discard

    idle = []
    for _ in range(10):
        started = time.perf_counter()
        hybrid.router.predict(probe, questions)
        idle.append((time.perf_counter() - started) * 1000)

    interleaved = []
    per_item = {}
    for task, item_id, prompt in items:
        local.generate(prompt, 24)  # keep the accelerator in the state app.py leaves it
        started = time.perf_counter()
        hybrid.router.predict({"request": prompt}, questions)
        elapsed = (time.perf_counter() - started) * 1000
        interleaved.append(elapsed)
        per_item[f"{task}:{item_id}"] = round(elapsed, 2)

    payload = {
        "condition": "single process, warm router",
        "idle": stats(idle),
        "interleaved": stats(interleaved),
        "per_item_ms": per_item,
    }
    out = ROOT / "results" / "router_latency_clean.json"
    out.write_text(json.dumps(payload, indent=2))

    for name in ("idle", "interleaved"):
        s = payload[name]
        print(
            f"{name:12} n={s['n']:>3} median={s['median_ms']:>5} ms "
            f"mean={s['mean_ms']:>5} ms  min={s['min_ms']} max={s['max_ms']}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
