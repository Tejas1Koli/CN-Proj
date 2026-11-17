"""Recompute output-token volume from the stored answer text.

The LiteRT-LM runtime reports `conv.token_count`, which includes the chat
template, so a one-letter answer is counted as 11 tokens. Subtracting the
prompt length does not remove that constant, which inflates the local arm's
output volume and its apparent throughput -- most visibly on multiple-choice
tasks where both arms answer with a single letter.

This script tokenizes the stored `local_text` and `cloud_text` with the local
model's tokenizer, so both arms are counted the same way and template tokens
are excluded. Time measurements are untouched.

Run:  PYTHONPATH="$PWD" .venv/bin/python tools/recount_tokens.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import config, runner  # noqa: E402
from bench.models import LocalModel  # noqa: E402

RAW = ROOT / "results" / "raw_paired.jsonl"
SUMMARY = ROOT / "results" / "summary.json"


def main() -> int:
    local = LocalModel()
    local.warm()
    tok = local.engine.tokenize

    rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
    changed = 0
    for record in rows:
        if record.get("protocol") != "paired":
            continue
        local_tokens = len(tok(record.get("local_text") or ""))
        cloud_tokens = len(tok(record.get("cloud_text") or ""))
        if (
            record.get("local_text_tokens") != local_tokens
            or record.get("cloud_text_tokens") != cloud_tokens
        ):
            changed += 1
        record["local_text_tokens"] = local_tokens
        record["cloud_text_tokens"] = cloud_tokens
    RAW.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"records updated: {changed}")

    before = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    summary = runner.aggregate_paired(RAW)
    for key in ("mode", "cloud_model", "model_labels", "host", "last_run",
                "local_model_dir_mb", "rss_mb_at_end", "scorer_version",
                "rescored_from_text"):
        if key in before:
            summary[key] = before[key]

    # Totals the write-up reads. Counted over the deduped paired sample.
    current = runner._dedupe(
        [r for r in rows if r.get("protocol") == "paired"], ("task", "item_id")
    )
    summary["output_tokens_text"] = {
        "local": sum(r.get("local_text_tokens") or 0 for r in current),
        "cloud": sum(r.get("cloud_text_tokens") or 0 for r in current),
        "hybrid": sum(
            (r.get("local_text_tokens") or 0)
            if r.get("route") == "local"
            else (r.get("cloud_text_tokens") or 0)
            for r in current
        ),
    }
    summary["output_tokens_reported"] = {
        "local": sum(r.get("local_output_tokens") or 0 for r in current),
        "cloud": sum(r.get("cloud_output_tokens") or 0 for r in current),
    }
    for task, cell in summary["per_task"].items():
        rs = [r for r in current if r["task"] == task]
        cell["local_text_tokens_total"] = sum(r.get("local_text_tokens") or 0 for r in rs)
        cell["cloud_text_tokens_total"] = sum(r.get("cloud_text_tokens") or 0 for r in rs)

    runner.save_summary(summary, SUMMARY)
    print(
        "text-token volume: local {local:,}  cloud {cloud:,}  ratio {ratio:.2f}x".format(
            local=summary["output_tokens_text"]["local"],
            cloud=summary["output_tokens_text"]["cloud"],
            ratio=summary["output_tokens_text"]["local"]
            / max(1, summary["output_tokens_text"]["cloud"]),
        )
    )
    print(f"summary -> {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
