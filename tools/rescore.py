"""Re-score stored answers with the corrected code scorer (no API calls).

Background: `score_code` concatenated the extracted program directly onto the
test harness. HumanEval's suffix begins with newlines, but MBPP's begins with
`assert`, so any MBPP completion that did not end in a newline became
`return xassert ...` and raised SyntaxError. Correct programs were recorded as
wrong. This script re-scores every stored answer with the fixed join and
rewrites `results/raw_paired.jsonl` plus `results/summary.json`.

The generated text is untouched -- only the correctness fields change. The
pre-fix file is preserved as `results/raw_paired.original.jsonl`.

Run:  PYTHONPATH="$PWD" .venv/bin/python tools/rescore.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import config, data, runner  # noqa: E402

RAW = ROOT / "results" / "raw_paired.jsonl"
BACKUP = ROOT / "results" / "raw_paired.original.jsonl"
SUMMARY = ROOT / "results" / "summary.json"


def main() -> int:
    if not BACKUP.exists():
        shutil.copy2(RAW, BACKUP)
        print(f"backup -> {BACKUP.name}")

    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    current = runner._dedupe(
        [r for r in rows if r.get("protocol") == "paired"], ("task", "item_id")
    )

    counts: dict[str, int] = {}
    for record in current:
        counts[record["task"]] = counts.get(record["task"], 0) + 1

    items: dict[tuple[str, str], data.Item] = {}
    scorers = {}
    for name, n in counts.items():
        task = data.TASK_BUILDERS[name](n)
        scorers[name] = task.scorer
        for item in task.items:
            items[(name, item.item_id)] = item

    flips = 0
    live_keys = {(r["task"], r["item_id"]) for r in current}
    for record in rows:
        if record.get("protocol") != "paired":
            continue
        key = (record["task"], record["item_id"])
        # Stale records (older prompt hash) are not part of the reported sample.
        if key not in live_keys or key not in items:
            record["rescored"] = False
            continue
        item = items[key]
        scorer = scorers[record["task"]]
        local_ok = bool(scorer(record.get("local_text") or "", item))
        cloud_ok = bool(scorer(record.get("cloud_text") or "", item))
        if local_ok != record["local_correct"] or cloud_ok != record["cloud_correct"]:
            flips += 1
        record["local_correct"] = local_ok
        record["cloud_correct"] = cloud_ok
        record["hybrid_correct"] = (
            cloud_ok if record.get("route") == "cloud" else local_ok
        )
        record["oracle_correct"] = bool(local_ok or cloud_ok)
        record["rescored"] = "v2-newline-join"

    RAW.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"records changed by re-scoring: {flips}")

    before = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    summary = runner.aggregate_paired(RAW)
    for key in ("mode", "cloud_model", "model_labels", "host", "last_run",
                "local_model_dir_mb", "rss_mb_at_end"):
        if key in before:
            summary[key] = before[key]
    summary["scorer_version"] = "v2-newline-join"
    summary["rescored_from_text"] = True
    runner.save_summary(summary, SUMMARY)
    print(f"summary -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
