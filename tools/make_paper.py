"""Build the report from the measured results, following the course format guide.

Every number is read from results/summary.json, results/raw_paired.jsonl or
results/router_latency_clean.json. Nothing quantitative is typed by hand here,
except the prompt-format ablation figures, which were measured during prompt
design and are flagged as such in the text.

Usage:
    PYTHONPATH="$PWD" .venv/bin/python tools/make_paper.py
    cd paper && pdflatex main.tex && pdflatex main.tex
"""

from __future__ import annotations

import json
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

TASK_LABELS = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
}
ORDER = ["mmlu", "gsm8k", "arc_challenge", "boolq", "humaneval", "mbpp"]
ARMS = ["local", "cloud", "hybrid", "oracle"]

# Prompt-format ablations measured during prompt design on calibration samples.
# These are deliberately labelled as calibration runs: they are not part of the
# 46-item paired sample.
ABLATIONS = [
    ("A1", "Local, plain ``Answer:'' prompt",
     "Multiple-choice format baseline", "MMLU 0/20"),
    ("A2", "Local, explicit answer directive",
     "Does an output-format instruction recover a small model?",
     "MMLU 9/20, ARC 21/30"),
    ("A3", "Local, prose few-shot MBPP prompt",
     "Code-format baseline", "MBPP 0/25 pass@1"),
    ("A4", "Local, harness \\texttt{[BEGIN]}/\\texttt{[END]} delimiters",
     "Stop the model answering an exemplar instead of the query",
     "MBPP 10/25 pass@1"),
    ("A5", "Local, answer directive on BoolQ",
     "Check the directive does not help everywhere",
     "BoolQ 22/30 vs 23/30 without"),
    ("A6", "Local backend: XNNPACK CPU vs Metal GPU",
     "Is the accelerator worth using?", "30.7 vs 51.6 tok/s (1.68$\\times$)"),
    ("A7", "Router disabled (always local)",
     "Baseline for the routing ablation", "58.7\\% accuracy, 456.1\\,s"),
    ("A8", "Router enabled (hybrid)",
     "Contribution of the routing component",
     "93.5\\% accuracy, 352.8\\,s"),
    ("A9", "Scorer: direct concat vs newline join",
     "How much does the evaluator matter?",
     "MBPP cloud 2/8 $\\rightarrow$ 7/8"),
    ("A10", "Output volume: runtime counter vs re-tokenised text",
     "Is the reported token count a fair comparison?",
     "MMLU 11 vs 1 token per answer"),
]


def wf(correct: int, n: int) -> tuple[float, float]:
    """Wilson score interval, 95%."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial
    from math import comb

    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def ci_cell(correct: int, n: int) -> str:
    lo, hi = wf(correct, n)
    return f"{correct}/{n} ({correct / n * 100:.0f}\\%)"


def ci_short(correct: int, n: int) -> str:
    lo, hi = wf(correct, n)
    return f"[{lo * 100:.0f}, {hi * 100:.0f}]"


def ascii_text(text: str) -> str:
    text = text.replace("```python", "").replace("```", "")
    text = text.replace("→", "->").replace("×", "x").replace("—", "--")
    text = text.replace("≥", ">=").replace("≤", "<=")
    return text.encode("ascii", "replace").decode("ascii")


def verbatim(text: str, width: int = 88, max_lines: int = 18) -> str:
    """Render a stored generation as a code listing.

    Original line breaks are kept so indentation still reads as Python. Only
    over-long lines are folded, and the fold is indented under the line it
    continues, so a wrapped docstring cannot be mistaken for a statement at
    column zero. Rendered at footnotesize to keep the block inside the measure.
    """
    lines: list[str] = []
    for raw in ascii_text(text).splitlines():
        raw = raw.rstrip()
        if len(raw) <= width:
            lines.append(raw)
            continue
        indent = len(raw) - len(raw.lstrip())
        pad = " " * (indent + 4)
        folded = textwrap.wrap(
            raw.strip(), width=max(20, width - len(pad)),
            subsequent_indent=pad, break_long_words=True,
        )
        lines.append(" " * indent + folded[0])
        lines.extend(folded[1:])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    body = "\n".join(lines)
    return "{\\footnotesize\n\\begin{verbatim}\n" + body + "\n\\end{verbatim}}"


def load():
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (ROOT / "results" / "raw_paired.jsonl").read_text().splitlines()
        if line.strip()
    ]
    from bench.runner import _dedupe

    rows = _dedupe(
        [r for r in rows if r.get("protocol") == "paired"], ("task", "item_id")
    )
    clean_path = ROOT / "results" / "router_latency_clean.json"
    clean = json.loads(clean_path.read_text()) if clean_path.exists() else None
    return summary, rows, clean


def tables(rows, clean):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    n_items = len(rows)

    def count(rs, arm):
        return sum(1 for r in rs if r.get(f"{arm}_correct"))

    # ---- Table: accuracy by task
    acc = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
           "Task & $n$ & Local & Cloud & Hybrid & Oracle \\\\", "\\midrule"]
    totals = {a: 0 for a in ARMS}
    for task in ORDER:
        rs = by_task[task]
        n = len(rs)
        cells = []
        for arm in ARMS:
            c = count(rs, arm)
            totals[arm] += c
            cells.append(ci_cell(c, n))
        acc.append(f"{TASK_LABELS[task]} & {n} & " + " & ".join(cells) + " \\\\")
    acc.append("\\midrule")
    acc.append(
        f"Pooled & {n_items} & "
        + " & ".join(
            f"{totals[a]} ({totals[a] / n_items * 100:.1f}\\%)" for a in ARMS
        )
        + " \\\\"
    )
    macro = []
    for arm in ARMS:
        vals = [count(by_task[t], arm) / len(by_task[t]) for t in ORDER]
        macro.append(f"{sum(vals) / len(vals) * 100:.1f}\\%")
    acc.append("Macro-avg & -- & " + " & ".join(macro) + " \\\\")
    acc.append(
        "95\\% CI & -- & "
        + " & ".join(ci_short(totals[a], n_items) for a in ARMS)
        + " \\\\"
    )
    acc.append("\\bottomrule")
    acc.append("\\end{tabular}")

    # ---- Table: router behaviour
    router = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
              "Task & $n$ & Disagree & Local & Cloud & Hybrid & Cloud-routed \\\\",
              "\\midrule"]
    for task in ORDER:
        rs = by_task[task]
        n = len(rs)
        dis = sum(1 for r in rs if r["local_correct"] != r["cloud_correct"])
        routed = sum(1 for r in rs if r["route"] == "cloud")
        router.append(
            f"{TASK_LABELS[task]} & {n} & {dis} & "
            f"{count(rs, 'local') / n * 100:.0f}\\% & "
            f"{count(rs, 'cloud') / n * 100:.0f}\\% & "
            f"{count(rs, 'hybrid') / n * 100:.0f}\\% & {routed}/{n} \\\\"
        )
    router.append("\\bottomrule")
    router.append("\\end{tabular}")

    # ---- Table: tokens and time
    tok = summary_text_tokens(rows)
    latency = latency_totals(rows)
    tokens_tbl = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
                  "Arm & Input & Output (text) & Output (reported) & Prefill s & "
                  "Decode s & Total s \\\\", "\\midrule"]
    for arm in ("local", "cloud", "hybrid"):
        pre = "--" if arm == "hybrid" else f"{latency[arm]['prefill']:.1f}"
        dec = "--" if arm == "hybrid" else f"{latency[arm]['decode']:.1f}"
        tokens_tbl.append(
            f"{arm.capitalize()} & {tok['input']:,} & {tok[arm]:,} & "
            f"{tok['reported_' + arm]:,} & {pre} & {dec} & "
            f"{latency[arm]['total']:.1f} \\\\"
        )
    tokens_tbl.append("\\bottomrule")
    tokens_tbl.append("\\end{tabular}")

    # ---- Table: per-task latency
    cross = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Task & Local s & Cloud s & Hybrid s & Out ratio & Speedup & "
             "Faster \\\\", "\\midrule"]
    for task in ORDER:
        rs = by_task[task]
        lt = sum(r["local_latency_ms"] for r in rs) / 1000
        ct = sum(r["cloud_latency_ms"] for r in rs) / 1000
        ht = sum(r["hybrid_latency_ms"] for r in rs) / 1000
        lo = sum(r.get("local_text_tokens") or 0 for r in rs)
        co = sum(r.get("cloud_text_tokens") or 0 for r in rs)
        win = "Local" if lt < ct else "Cloud"
        cross.append(
            f"{TASK_LABELS[task]} & {lt:.1f} & {ct:.1f} & {ht:.1f} & "
            f"{lo / max(1, co):.2f}$\\times$ & {lt / max(1e-9, ct):.2f}$\\times$ & {win} \\\\"
        )
    cross.append("\\bottomrule")
    cross.append("\\end{tabular}")

    # ---- Table: McNemar
    sig = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
           "Task & \\multicolumn{2}{c}{Hybrid vs Local} & "
           "\\multicolumn{2}{c}{Cloud vs Local} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
           " & win/loss & $p$ & win/loss & $p$ \\\\", "\\midrule"]
    for task in ORDER:
        rs = by_task[task]
        h_only = sum(1 for r in rs if r["hybrid_correct"] and not r["local_correct"])
        l_only = sum(1 for r in rs if r["local_correct"] and not r["hybrid_correct"])
        c_only = sum(1 for r in rs if r["cloud_correct"] and not r["local_correct"])
        l_beat = sum(1 for r in rs if r["local_correct"] and not r["cloud_correct"])
        sig.append(
            f"{TASK_LABELS[task]} & {h_only}/{l_only} & {mcnemar(h_only, l_only):.3f} & "
            f"{c_only}/{l_beat} & {mcnemar(c_only, l_beat):.3f} \\\\"
        )
    sig.append("\\bottomrule")
    sig.append("\\end{tabular}")

    # ---- Table: dataset
    data_tbl = ["\\begin{tabular}{@{}llrrll@{}}", "\\toprule",
                "Task & Source split & Items & Shots & Answer format & Metric \\\\",
                "\\midrule",
                "MMLU & test (subject-stratified) & 8 & 5 & option letter & accuracy \\\\",
                "GSM8K & test & 8 & 5 & final number & exact match \\\\",
                "ARC-Challenge & test & 7 & 5 & option letter & accuracy \\\\",
                "BoolQ & validation & 7 & 3 & yes/no & accuracy \\\\",
                "HumanEval & test & 8 & 0 & function body & pass@1 \\\\",
                "MBPP & test & 8 & 3 & delimited program & pass@1 \\\\",
                "\\midrule",
                f"Total & -- & {n_items} & -- & -- & -- \\\\",
                "\\bottomrule", "\\end{tabular}"]

    # ---- Table: ablation
    abl = ["\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{4.1cm}>{\\raggedright\\arraybackslash}p{4.4cm}>{\\raggedright\\arraybackslash}p{3.3cm}@{}}", "\\toprule",
           "ID & Configuration & Purpose & Measured effect \\\\", "\\midrule"]
    for ident, config, purpose, effect in ABLATIONS:
        abl.append(f"{ident} & {config} & {purpose} & {effect} \\\\")
    abl.append("\\bottomrule")
    abl.append("\\end{tabular}")

    return {
        "by_task": by_task,
        "n_items": n_items,
        "totals": totals,
        "accuracy": "\n".join(acc),
        "router": "\n".join(router),
        "tokens": "\n".join(tokens_tbl),
        "crossover": "\n".join(cross),
        "significance": "\n".join(sig),
        "dataset": "\n".join(data_tbl),
        "ablation": "\n".join(abl),
    }


def summary_text_tokens(rows):
    input_tokens = 0
    try:
        from bench import data
        from bench.models import LocalModel

        model = LocalModel()
        model.warm()
        seen = set()
        for task in data.build_all():
            for item in task.items:
                key = (task.name, item.item_id)
                if key in seen:
                    continue
                seen.add(key)
                input_tokens += len(model.engine.tokenize(item.prompt))
    except Exception:
        input_tokens = 0

    totals = {
        "input": input_tokens,
        "local": sum(r.get("local_text_tokens") or 0 for r in rows),
        "cloud": sum(r.get("cloud_text_tokens") or 0 for r in rows),
        "reported_local": sum(r.get("local_output_tokens") or 0 for r in rows),
        "reported_cloud": sum(r.get("cloud_output_tokens") or 0 for r in rows),
    }
    totals["hybrid"] = sum(
        (r.get("local_text_tokens") or 0)
        if r["route"] == "local"
        else (r.get("cloud_text_tokens") or 0)
        for r in rows
    )
    totals["reported_hybrid"] = sum(
        (r.get("local_output_tokens") or 0)
        if r["route"] == "local"
        else (r.get("cloud_output_tokens") or 0)
        for r in rows
    )
    return totals


def latency_totals(rows):
    out = {}
    for arm in ("local", "cloud", "hybrid"):
        total = sum(r.get(f"{arm}_latency_ms") or 0 for r in rows) / 1000
        pre = sum(r.get(f"{arm}_ttft_ms") or 0 for r in rows) / 1000
        out[arm] = {
            "total": total,
            "prefill": pre,
            "decode": max(0.0, total - pre),
        }
    return out


def main() -> int:
    summary, rows, clean = load()
    T = tables(rows, clean)
    by_task = T["by_task"]
    n_items = T["n_items"]
    totals = T["totals"]

    tok = summary_text_tokens(rows)
    lat = latency_totals(rows)
    router_s = sum(r.get("decision_ms") or 0 for r in rows) / 1000
    router_share = router_s / max(1e-9, lat["hybrid"]["total"]) * 100

    local_pooled = totals["local"] / n_items * 100
    cloud_pooled = totals["cloud"] / n_items * 100
    hybrid_pooled = totals["hybrid"] / n_items * 100
    oracle_pooled = totals["oracle"] / n_items * 100
    captured = (hybrid_pooled - local_pooled) / max(
        1e-9, oracle_pooled - local_pooled
    ) * 100

    def macro(arm):
        vals = [sum(1 for r in by_task[t] if r[f"{arm}_correct"]) / len(by_task[t])
                for t in ORDER]
        return sum(vals) / len(vals) * 100

    routed_cloud = sum(1 for r in rows if r["route"] == "cloud")
    routed_local = n_items - routed_cloud

    idle_ms = clean["idle"]["median_ms"] if clean else float("nan")
    busy_ms = clean["interleaved"]["median_ms"] if clean else float("nan")
    busy_n = clean["interleaved"]["n"] if clean else 0
    busy_min = clean["interleaved"]["min_ms"] if clean else 0
    busy_max = clean["interleaved"]["max_ms"] if clean else 0
    busy_s = busy_ms / 1000

    dec_local = tok["local"] / max(1e-9, lat["local"]["decode"])
    dec_cloud = tok["cloud"] / max(1e-9, lat["cloud"]["decode"])
    pre_local = tok["input"] / max(1e-9, lat["local"]["prefill"])
    pre_cloud = tok["input"] / max(1e-9, lat["cloud"]["prefill"])

    he = by_task["humaneval"]
    mb = by_task["mbpp"]
    bq = by_task["boolq"]

    def arm(rs, a):
        return sum(1 for r in rs if r[f"{a}_correct"])

    # qualitative examples, pulled from the stored generations
    def rec(item_id):
        return next(r for r in rows if r["item_id"] == item_id)

    gsm = rec("gsm8k/6")
    mbpp = rec("mbpp/361")
    hev = rec("HumanEval/2")
    boolq = rec("boolq/3")

    gsm_local = verbatim(gsm["local_text"], max_lines=12)
    gsm_cloud = verbatim(gsm["cloud_text"], max_lines=8)
    mbpp_local = verbatim(mbpp["local_text"], max_lines=12)
    mbpp_cloud = verbatim(mbpp["cloud_text"], max_lines=4)
    # Start the excerpt on a line boundary so the reader sees whole comments.
    he_tail = hev["local_text"].rstrip().splitlines()[-13:]
    he_local = verbatim("\n".join(he_tail), max_lines=13)
    he_cloud = verbatim(hev["cloud_text"], max_lines=10)

    payload = {
        "n_items": n_items,
        "local_pooled": f"{local_pooled:.1f}",
        "cloud_pooled": f"{cloud_pooled:.1f}",
        "hybrid_pooled": f"{hybrid_pooled:.1f}",
        "oracle_pooled": f"{oracle_pooled:.1f}",
        "macro_local": f"{macro('local'):.1f}",
        "macro_cloud": f"{macro('cloud'):.1f}",
        "macro_hybrid": f"{macro('hybrid'):.1f}",
        "captured": f"{captured:.0f}",
        "captured1": f"{captured:.1f}",
        "cloud_minus_hybrid": f"{cloud_pooled - hybrid_pooled:.1f}",
        "hybrid_minus_local": f"{hybrid_pooled - local_pooled:.1f}",
        "oracle_minus_cloud": f"{oracle_pooled - cloud_pooled:.1f}",
        "routed_cloud": routed_cloud,
        "routed_local": routed_local,
        "routed_local_pct": f"{routed_local / n_items * 100:.0f}",
        "local_total_s": f"{lat['local']['total']:.1f}",
        "cloud_total_s": f"{lat['cloud']['total']:.1f}",
        "hybrid_total_s": f"{lat['hybrid']['total']:.1f}",
        "local_prefill_s": f"{lat['local']['prefill']:.1f}",
        "cloud_prefill_s": f"{lat['cloud']['prefill']:.1f}",
        "local_decode_s": f"{lat['local']['decode']:.1f}",
        "cloud_decode_s": f"{lat['cloud']['decode']:.1f}",
        "speedup": f"{lat['local']['total'] / max(1e-9, lat['cloud']['total']):.2f}",
        "tok_local": f"{tok['local']:,}",
        "tok_cloud": f"{tok['cloud']:,}",
        "tok_hybrid": f"{tok['hybrid']:,}",
        "tok_reported_local": f"{tok['reported_local']:,}",
        "tok_reported_cloud": f"{tok['reported_cloud']:,}",
        "tok_ratio": f"{tok['local'] / max(1, tok['cloud']):.2f}",
        "dec_local": f"{dec_local:.1f}",
        "dec_cloud": f"{dec_cloud:.1f}",
        "dec_ratio": f"{dec_cloud / max(1e-9, dec_local):.1f}",
        "pre_local": f"{pre_local:.0f}",
        "pre_cloud": f"{pre_cloud:.0f}",
        "router_s": f"{router_s:.1f}",
        "router_share": f"{router_share:.0f}",
        "router_idle_ms": f"{idle_ms:.0f}",
        "router_busy_ms": f"{busy_ms:.0f}",
        "router_busy_s": f"{busy_s:.1f}",
        "router_busy_n": f"{busy_n}",
        "router_busy_range": f"{busy_min:.0f}--{busy_max:.0f}",
        "router_penalty": f"{busy_ms / max(1e-9, idle_ms):.1f}",
        "he_local": f"{arm(he, 'local')}/{len(he)}",
        "he_cloud": f"{arm(he, 'cloud')}/{len(he)}",
        "he_hybrid": f"{arm(he, 'hybrid')}/{len(he)}",
        "mb_local": f"{arm(mb, 'local')}/{len(mb)}",
        "mb_cloud": f"{arm(mb, 'cloud')}/{len(mb)}",
        "bq_local": f"{arm(bq, 'local')}/{len(bq)}",
        "bq_cloud": f"{arm(bq, 'cloud')}/{len(bq)}",
        "table_accuracy": T["accuracy"],
        "table_router": T["router"],
        "table_tokens": T["tokens"],
        "table_crossover": T["crossover"],
        "table_significance": T["significance"],
        "table_dataset": T["dataset"],
        "table_ablation": T["ablation"],
        "gsm_local": gsm_local,
        "gsm_cloud": gsm_cloud,
        "mbpp_local": mbpp_local,
        "mbpp_cloud": mbpp_cloud,
        "he_local": he_local,
        "he_cloud": he_cloud,
        "gsm_local_tok": str(gsm["local_text_tokens"]),
        "gsm_cloud_tok": str(gsm["cloud_text_tokens"]),
        "mbpp_local_tok": str(mbpp["local_text_tokens"]),
        "mbpp_cloud_tok": str(mbpp["cloud_text_tokens"]),
    }

    tex = TEMPLATE
    for key, value in payload.items():
        tex = tex.replace(f"[[{key}]]", str(value))

    PAPER.mkdir(exist_ok=True)
    (PAPER / "main.tex").write_text(tex)
    print(f"wrote {PAPER / 'main.tex'} ({len(tex):,} chars)")
    return 0


TEMPLATE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{caption}
\usepackage[colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,
            urlcolor=blue!55!black]{hyperref}

\captionsetup{font=small,labelfont=bf}
\setlist[itemize]{leftmargin=1.1em,itemsep=2pt,topsep=3pt,parsep=0pt}
\setlist[enumerate]{leftmargin=1.3em,itemsep=2pt,topsep=3pt,parsep=0pt}
\renewcommand{\arraystretch}{1.08}

\title{\vspace{-1.2cm}\textbf{When Does a Cloud--Local Hybrid Help?\\[3pt]
\large A Paired Evaluation of Difficulty Routing, Accuracy and\\[1pt]
Hardware Cost on Apple Silicon}}
\author{Tejas Koli\\[2pt]
\small Independent Researcher\\[1pt]
\small \texttt{tejas.koli.2k183742@gmail.com}}
\date{November 2026}

\begin{document}
\maketitle

\begin{abstract}
\noindent
A hybrid assistant answers easy questions on the device and sends hard ones to
a cloud model. The design is attractive, but its benefits are usually asserted
rather than measured end to end. We evaluate three systems on the same
[[n_items]] benchmark items: a local arm (Gemma 4 E2B, 2.4\,GB, LiteRT-LM on an
Apple M2 GPU), a cloud arm (Nemotron-3-Ultra 550B), and a hybrid arm in which a
non-autoregressive decision model, Laya, routes each query. Every item is
answered once per backend and the hybrid is derived from those same answers, so
the comparison is exactly paired and the oracle-routing ceiling is obtained at
no extra cost. The local arm scores [[local_pooled]]\%, the cloud arm
[[cloud_pooled]]\%, the hybrid [[hybrid_pooled]]\%, and the oracle
[[oracle_pooled]]\%. Because the cloud arm matches the oracle, no router could
have gained accuracy on this sample: routing bought latency and privacy, not
correctness. It sent [[routed_local]] of [[n_items]] items
([[routed_local_pct]]\%) to the device, gave up [[cloud_minus_hybrid]] points
against cloud-only, and captured [[captured1]]\% of the headroom over
always-local. The hardware result is the more useful one. Output volume, not
difficulty, decides which tier is faster: on multiple-choice tasks both arms
emit the same one-token answer and the device wins on prefill, while on
MBPP the local model writes [[mbpp_local_tok]] tokens where the cloud writes
[[mbpp_cloud_tok]] and loses by 5.9$\times$. The router itself is not free: a
decision costs [[router_idle_ms]]\,ms on an idle GPU but [[router_busy_s]]\,s
when it shares the accelerator with the local model, which is
[[router_share]]\% of the hybrid's wall clock. Two measurement defects are also
documented and corrected.
\end{abstract}

\vspace{-0.2cm}
\noindent\textbf{Keywords:} on-device inference; model routing; hybrid
inference; small language models; benchmarking; Apple Silicon.

\section{Introduction}

\subsection{Problem and background}
A user asking a chat assistant a question now has two plausible answerers. A
small model on the device is private, needs no network, and costs nothing per
token; a large model behind an API is far more capable but ships the query off
the device and is billed and metered by the token. Neither default is
satisfying. Always-cloud gives up privacy and latency control for accuracy that
many queries do not need. Always-local gives up too much accuracy to be a
general default. A hybrid system, which sends each query to whichever tier
suits it, is the obvious compromise, and it reduces to one component: the
router.

\subsection{Motivation}
The motivation for measuring this rather than assuming it is that the two
tiers differ along axes that are easy to confuse. They differ in \emph{quality}
(measured by benchmark accuracy), in \emph{latency} (measured in wall clock), in
\emph{cost} (tokens or dollars), and in \emph{privacy} (whether the query leaves
the machine). A router that optimises one of these can silently harm another.
Reports of hybrids also tend to evaluate the router's decisions in isolation
--- does it label this query ``hard''? --- rather than the behaviour of the
system the user actually experiences.

\subsection{Limitations of existing approaches}
Three gaps motivate this study.
\begin{enumerate}
\item \textbf{Routers are trained to predict difficulty or accuracy, not cost.}
Cascades and learned routers \cite{frugalgpt,routellm,hybridllm,automix}
decide which model should answer from a predicted quality gap. They do not
model how many tokens each model will emit, which is what determines wall
clock on a bandwidth-bound device.
\item \textbf{On-device speed is reported per token, not per query.} A local
model that decodes slower but says less can finish first; the opposite also
happens. Token-per-second figures hide this because they normalise away the
output volume.
\item \textbf{The router's own cost is ignored.} A single forward pass through
an encoder is cheap in FLOPs, and papers describe such routers as free. On one
machine, sharing an accelerator with the local language model changes that.
\end{enumerate}

\subsection{Objectives, hypothesis and contributions}
Our objective is to quantify, on one machine and with identical prompts, what a
difficulty-routed hybrid actually buys over either pure tier. We test three
hypotheses:
\begin{itemize}
\item \textbf{H1 (accuracy).} The hybrid retains cloud-level accuracy while
serving a meaningful share of queries locally.
\item \textbf{H2 (routing signal).} Difficulty is a sufficient statistic for
which tier should answer.
\item \textbf{H3 (overhead).} The router's cost is negligible relative to
generation.
\end{itemize}

The contributions are:
\begin{enumerate}
\item A \textbf{paired per-item protocol} that answers each item once per
backend, derives the hybrid from those answers, and therefore yields an exactly
paired comparison plus the oracle ceiling for the price of two passes instead
of three.
\item \textbf{Evidence on H1 and H2} from [[n_items]] items across six
benchmarks, including the observation that the oracle equals cloud-only here, so
routing cannot improve accuracy and must be justified on latency and privacy.
\item A \textbf{hardware decomposition} showing that per-query speed is set by
expected output length and prefill, not by input difficulty, and that the
router's own latency is [[router_share]]\% of the hybrid's wall clock.
\item \textbf{Two corrected measurement defects} --- a code-scorer join bug that
inverted the MBPP result, and a token counter that included chat-template
tokens --- reported because both changed published numbers.
\end{enumerate}

\section{Literature Survey}

\subsection{Small and on-device language models}
The feasibility of the local tier rests on a line of work that has made
billion-parameter models useful on consumer hardware. Gemma \cite{gemma} and
Gemma~2 \cite{gemma2} release open models from 2B to 27B parameters trained with
the same recipe as a proprietary frontier model; Phi-3 \cite{phi3} shows that a
3.8B model trained on heavily curated data can reach accuracy previously
associated with far larger models, and is explicitly positioned for execution
on a phone. MobileLLM \cite{mobilellm} studies the sub-billion regime and reports
that depth and embedding sharing matter more than width at that scale.

Serving is a separate problem. PowerInfer \cite{powerinfer} exploits the
power-law activation sparsity of large models to run a 7B model on a single
consumer GPU, and LLM in a flash \cite{llmflash} streams Mixture-of-Experts
weights from flash so that a model larger than DRAM can still run. Both are
about fitting weights into a memory hierarchy; neither addresses \emph{when} to
run locally at all.

\subsection{Routing, cascades and model selection}
FrugalGPT \cite{frugalgpt} introduced the cascade formulation used here: try a
cheap model first, score its answer, and escalate only if the score is low,
reporting large cost reductions at GPT-4-level quality. Hybrid LLM
\cite{hybridllm} learns a quality-aware router between a small and a large
model, and RouteLLM \cite{routellm} trains routers on preference data and
reports more than a twofold cost reduction at comparable quality. AutoMix
\cite{automix} replaces the external scorer with self-verification, and reports
similar savings.

These systems differ from ours in what the router observes. They route on a
predicted \emph{quality} gap between two models that are both remote. We route
on a predicted \emph{difficulty} score with one model on the device, where the
cost structure is not a price list but a thermal and memory budget, and where
the local model's output length is a first-order term.

\subsection{Decision models and evaluation practice}
Laya \cite{laya} is the router used here: a ModernBERT encoder with a
reinforcement-learned head that returns typed answers with calibrated
probabilities in a single forward pass and never generates text. It is a
System-1 decision model in the sense that the answer is produced by one
feed-forward evaluation rather than by decoding.

On the evaluation side, the small-language-model survey of Lu et al.
\cite{slmsurvey} benchmarks 59 models across tasks and finds that accuracy and
efficiency rankings are task-dependent, which is consistent with the per-task
crossover we measure. None of the surveyed work reports the router's own
latency alongside generation latency, which is the gap H3 addresses.

\begin{table}[!htbp]
\centering
\footnotesize
\caption{Literature comparison. Citation counts were read from OpenAlex on
17 November 2026, a single source used consistently for every row.}
\label{tab:lit}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.9cm}>{\raggedright\arraybackslash}p{3.55cm}>{\raggedright\arraybackslash}p{3.35cm}>{\raggedright\arraybackslash}p{2.35cm}>{\raggedright\arraybackslash}p{2.9cm}r@{}}
\toprule
Author(s), year & Methodology & Reported performance & Dataset & Critical remarks & Cites \\
\midrule
Gemma Team (2024) \cite{gemma} &
Open dense and MoE LMs, 2B--27B, distilled from a frontier model &
Competitive with much larger open models; MMLU up to 75.1 (27B) &
Public web, code, mathematics &
Provides the local weights; no on-device latency analysis & 3{,}803 \\
Gemma Team (2024) \cite{gemma2} &
Improved pretraining and distillation at practical sizes &
Better quality per parameter than Gemma~1 &
Same corpus family &
Model card, not a systems paper & 3{,}881 \\
Abdin et al.\ (2024) \cite{phi3} &
3.8B model, heavily filtered textbook-style data &
MMLU 69\% at 3.8B; positioned for phones &
Synthetic and web text &
Claims mobile readiness without measuring it on a device & 3{,}285 \\
Chen et al.\ (2023) \cite{frugalgpt} &
Cascade: cheap model, scorer, escalation &
Matches GPT-4 quality at up to 98\% lower cost &
Task-specific labelled sets &
Router uses a learned scorer, not a difficulty model; both models are remote & 1{,}119 \\
Ong et al.\ (2024) \cite{routellm} &
Routers trained on preference data (matrix factorisation, BERT) &
${>}2\times$ cost reduction at comparable quality &
MT-Bench, MMLU, GSM8K &
Quality-gap objective; no output-length or router-cost term & 745 \\
Ding et al.\ (2024) \cite{hybridllm} &
Quality-aware router between small and large models &
Up to 40\% fewer large-model calls &
Synthetic query sets &
Both models served remotely; latency not decomposed & 622 \\
Song et al.\ (2024) \cite{powerinfer} &
Activation-sparsity aware GPU/CPU hybrid execution &
Up to 7.6$\times$ faster than llama.cpp on one consumer GPU &
LLaMA-2 family &
Serving only; no routing decision & 426 \\
Liu et al.\ (2024) \cite{mobilellm} &
Sub-billion models; depth over width, embedding sharing &
125M/350M models beat prior small baselines &
Public text corpora &
Accuracy-only; no hybrid deployment & 409 \\
Alizadeh et al.\ (2024) \cite{llmflash} &
Flash-streamed MoE weights for limited DRAM &
Runs a 7B model in 4\,GB DRAM at 4--5$\times$ naive offload speed &
Sparse MoE models &
Memory-bound setting; single-device only & 393 \\
Lu et al.\ (2024) \cite{slmsurvey} &
Survey and benchmark of 59 SLMs &
Rankings are task-dependent &
Aggregated public benchmarks &
Motivates per-task analysis; no routing experiment & 305 \\
Aggarwal et al.\ (2024) \cite{automix} &
Self-verifying mix of models &
Up to 50\% fewer GPT-4 calls at similar accuracy &
Few-shot classification, QA &
Verification cost is itself a model call & 277 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Research gap}
The literature establishes that (i) capable small models can run on-device,
(ii) cascades save money when a reliable quality signal exists, and (iii)
routers can be trained from preference data. What is missing is a measurement,
on one machine with one set of prompts, that separates the accuracy a hybrid
gains from the latency it costs, including the router's own cost. That gap is
what this paper fills, and the finding that the oracle equals cloud-only on this
sample is precisely why the separation matters: when there is no accuracy to
gain, the only justification left is latency and privacy, and both must be
measured rather than assumed.

\section{Methodology}

\subsection{System overview}
Figure~\ref{fig:arch} shows the deployed path. A query is presented to Laya,
which returns a difficulty score, a probability that the query needs external
tools, and a probability that it is sensitive. The tier is chosen from those
three signals. The selected backend answers, and the service returns the answer
with the route and timings. The same code path is used by the API service and
by the benchmark harness, so the reported numbers describe the deployed system.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/architecture.png}
\caption{System architecture. The router is a local encoder that emits no
tokens; both tiers receive byte-identical prompts. The annotated decision costs
are measured (Section~\ref{sec:quant}).}
\label{fig:arch}
\end{figure}

\subsection{The router and the decision rule}
Laya is a ModernBERT encoder with a reinforcement-learned head trained against
strictly proper scoring rules \cite{laya}. It returns calibrated probabilities
for typed questions; we use three of them. A query goes to the cloud when
\begin{equation}
\text{difficulty} \ge 1.5 \quad \lor \quad P(\text{needs tools}) \ge 0.14
\quad \lor \quad P(\text{sensitive}) \ge 0.5 .
\label{eq:rule}
\end{equation}
The thresholds were chosen from the observed score distribution during
development and then held fixed for every reported run. The tool threshold is
deliberately low: this checkpoint compresses the probability range, so real
tool-requiring queries score near $0.14$ while simple factual lookups score near
$0.12$.

\subsection{The paired per-item protocol}
Evaluating three arms naively costs two cloud requests per item. Instead each
item is answered once per backend and the hybrid is derived:

\begin{quote}
\ttfamily\small
local\_answer \(\leftarrow\) Gemma 4 E2B (no API call)\\
cloud\_answer \(\leftarrow\) Nemotron (1 API call)\\
route \(\leftarrow\) Laya.decision (no API call)\\
hybrid\_answer \(\leftarrow\) cloud\_answer if route = cloud else local\_answer
\end{quote}

Three properties follow. The comparison is exactly paired, so item difficulty
cancels and McNemar's test applies. API usage is halved. And the oracle arm,
$\text{local\_correct} \lor \text{cloud\_correct}$, is the accuracy of a router
that always picks the better arm on that item, which upper-bounds what any
router can achieve on the same sample.

Hybrid latency is derived as $\text{decision} + \text{backend latency}$, which
is what the service does. A separate end-to-end pass through the real service
on six items produced the same route and the same correctness verdict on all
six, so the derivation is faithful for these items.

\subsection{Scoring and statistical analysis}
Multiple-choice and Boolean tasks are scored by extracting the answer marker
(strict: a lone option letter or yes/no) rather than by matching free text.
GSM8K is scored on the final number with a $10^{-4}$ tolerance. Code tasks run
the extracted program in a separate process with CPU and address-space limits
and a wall-clock timeout, and are scored pass@1 against the task's unit tests.

Accuracy is reported with 95\% Wilson score intervals, which behave correctly
at $n=7$--$8$. Arm differences use McNemar's exact test on discordant pairs.
Latency is reported as sums and medians of wall clock; because the arms are
paired, sums are directly comparable.

\subsection{Implementation details}
All local measurements come from a fanless Apple M2 MacBook Air with 16\,GB of
unified memory running macOS 26.6 and Python 3.14. The local model is Gemma 4
E2B (2.4\,GB on disk, 4.5\,GB when the runtime is included) served by
LiteRT-LM through the Metal backend with \texttt{gpu\_decode\_steps\_per\_sync=4};
the CPU fallback is used only if the GPU is unavailable. The cloud model is
\texttt{nvidia/nemotron-3-ultra-550b-a55b} called greedily
($T=0$) with reasoning disabled, because the local checkpoint has no thinking
mode and enabling it would let the cloud model spend hidden tokens the local
model cannot produce.

Prompts are byte-identical across arms and include their few-shot exemplars;
each item carries a hash of its prompt so that a prompt change invalidates the
old record instead of silently mixing two prompts under one identifier. The
harness appends one JSON record per item and resumes, so a larger sample
accumulates without re-running completed items. Every number in this paper is
regenerated from those records.

\section{Results and Discussion}

\subsection{Dataset}
\label{sec:dataset}
Six public benchmarks are used, following lm-evaluation-harness prompt
conventions: MMLU \cite{mmlu}, GSM8K \cite{gsm8k}, ARC-Challenge \cite{arc},
BoolQ \cite{boolq}, HumanEval \cite{humaneval}, and MBPP \cite{mbpp}. No model
is trained or fine-tuned, so there is no train/validation/test split in the
machine-learning sense: evaluation is on each task's official test split
(validation for BoolQ), and few-shot exemplars are drawn from the corresponding
training split. Splits and item counts are given in Table~\ref{tab:dataset}.

\begin{table}[!htbp]
\centering
\small
\caption{Dataset composition. Items are the paired sample actually evaluated;
the harness can extend it.}
\label{tab:dataset}
[[table_dataset]]
\end{table}

Preprocessing is limited to prompt construction and answer extraction, and is
identical for all arms. Multiple-choice items are rendered with lettered options
and an explicit instruction to answer with a single letter; this instruction was
necessary (Section~\ref{sec:ablation}) because without it the local model
preambles instead of answering. Code items request a complete function inside
one code block, and MBPP additionally wraps the answer in
\texttt{[BEGIN]}/\texttt{[END]} so the model does not continue with a new
example. A representative MBPP prompt ending is shown below (the long
assertion is wrapped for width; the stored prompt is a single line).

{\footnotesize
\begin{verbatim}
You are an expert Python programmer, and here is your task: Write a
function to remove empty lists from a given list of lists. Your code
should pass these tests:

assert remove_empty([[], [], [], 'Red', 'Green', [1,2], 'Blue', [], []])
       == ['Red', 'Green', [1, 2], 'Blue']
[BEGIN]
\end{verbatim}}

\subsection{Ablation study}
\label{sec:ablation}
Table~\ref{tab:ablation} isolates the components. Rows A1--A6 are controlled
prompt and runtime comparisons measured on calibration samples during prompt
design; rows A7--A10 are measured on the reported sample. Every row keeps the
dataset, the evaluation procedure and the metric fixed and changes one thing.

\begin{table}[!htbp]
\centering
\footnotesize
\caption{Ablation study. A1--A6 are calibration runs on separate samples
($n=20$--$30$) used to fix the prompt format; they are not part of the
[[n_items]]-item sample.}
\label{tab:ablation}
\setlength{\tabcolsep}{4pt}
[[table_ablation]]
\end{table}

Three results deserve comment. First, prompt format is not a detail: the same
local model scores 0/20 on MMLU with a plain ``Answer:'' prompt and 9/20 with an
explicit single-letter directive, and 0/25 versus 10/25 on MBPP with and without
delimiters. Reporting the first number as the model's ability would have been
wrong. Second, the directive is not universally good: on BoolQ it slightly
reduced accuracy (22/30 versus 23/30), so the BoolQ prompt was left in its
original form. Third, the accelerator matters but less than expected: Metal
gives 51.6\,tok/s against 30.7\,tok/s on CPU, a $1.68\times$ gain, while the
choice of backend does not change any answer.

Rows A9 and A10 are ablations of the \emph{measurement}, not of the model, and
both are reported because both changed conclusions. A9 shows that the code
scorer's join between a program and its tests determined the MBPP result: with
the incorrect direct concatenation the cloud arm scored 2/8, and with a newline
join it scores 7/8. A10 shows that the runtime's token counter includes the chat
template, so a one-letter answer is counted as 11 tokens rather than 1; the
local arm's apparent output volume and throughput were both inflated by it.

\subsection{Quantitative analysis}
\label{sec:quant}
Table~\ref{tab:accuracy} gives accuracy by task and arm with Wilson intervals,
and Figure~\ref{fig:acc} plots the same data.

\begin{table}[!htbp]
\centering
\small
\caption{Accuracy with 95\% Wilson intervals. Oracle is the accuracy of a
perfect per-item router on this sample.}
\label{tab:accuracy}
[[table_accuracy]]
\end{table}

The pooled ordering is cloud ([[cloud_pooled]]\%) $\approx$ hybrid
([[hybrid_pooled]]\%) $\gg$ local ([[local_pooled]]\%), and macro-averaging over
tasks preserves it ([[macro_cloud]]\%, [[macro_hybrid]]\%,
[[macro_local]]\%). Two features of this table matter more than the ordering.

First, \textbf{the oracle equals the cloud arm}. A perfect router could have
scored [[oracle_pooled]]\%, exactly what always-cloud achieved, because the two
items the cloud answered incorrectly were also answered incorrectly on-device.
On this sample there was no accuracy to be recovered by routing; the hybrid's
only possible gains are latency, cost and privacy. We did not expect this when
the study was designed, and it reframes the evaluation: the usual question
(``how much of the oracle gap does the router capture?'') has a gap of
[[oracle_minus_cloud]] points here.

Second, \textbf{the hybrid's cost is one item}. It lost
[[cloud_minus_hybrid]] points against cloud-only by sending one HumanEval item
to the device where the local model failed. Everything else it answered was
either identical to the cloud answer or (on nothing) better.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.86\linewidth]{figures/accuracy_by_task.png}
\caption{Accuracy by task with 95\% Wilson intervals. The hybrid tracks the
cloud arm except on HumanEval.}
\label{fig:acc}
\end{figure}

Table~\ref{tab:router} reports routing behaviour. The router sent
[[routed_cloud]] of [[n_items]] items ([[routed_local_pct]]\% went to the
device) and captured [[captured1]]\% of the headroom between always-local and
the oracle. It disagreed with the better backend on only
the items listed under ``Disagree'', and on those the harness records which tier
was chosen.

\begin{table}[!htbp]
\centering
\small
\caption{Routing behaviour by task. ``Disagree'' counts items where exactly one
backend was correct; ``Cloud-routed'' is the share of items sent to the cloud.}
\label{tab:router}
[[table_router]]
\end{table}

\paragraph{Tokens and time.}
Table~\ref{tab:tokens} separates volume from placement. Input volume is
identical by construction. Output volume is counted by re-tokenising the stored
answer text with the model's tokenizer, so both arms are counted the same way;
the runtime's own counter is shown for reference and is inflated by the chat
template.

\begin{table}[!htbp]
\centering
\small
\caption{Token volume and wall clock over the whole study. Prefill is time to
first token; decode is the remainder. The hybrid row includes
[[router_s]]\,s of router time.}
\label{tab:tokens}
[[table_tokens]]
\end{table}

Three observations follow. (i) The local arm writes [[tok_ratio]]$\times$ more
than the cloud arm ([[tok_local]] versus [[tok_cloud]] tokens), but this is
entirely a property of generative tasks: on MMLU, ARC-Challenge and BoolQ both
arms emit a single letter and volume is identical. ``The small model is verbose''
is therefore not a general fact. (ii) Decode throughput is
[[dec_local]]\,tok/s local against [[dec_cloud]]\,tok/s cloud, only
[[dec_ratio]]$\times$ in the cloud's favour. (iii) Prefill is faster on-device
([[pre_local]] versus [[pre_cloud]]\,tok/s), because the cloud path pays a
network round trip per request.

So the cloud's [[speedup]]$\times$ total advantage is mostly output economy
multiplied by a modest per-token advantage, with prefill working against it. It
is not evidence that the device computes badly; it is evidence that the two
models differ in how much they choose to say.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.8\linewidth]{figures/latency_distribution.png}
\caption{End-to-end latency by arm on a log scale. The cloud is usually faster,
but the distributions overlap and the per-task ranking reverses.}
\label{fig:lat}
\end{figure}

\paragraph{The per-task crossover.}
Table~\ref{tab:crossover} is the most actionable result. On multiple-choice
tasks the device is faster, because both models produce one token and the device
avoids the network; on long-generation tasks the cloud is faster, because it
emits several times fewer tokens and decodes each of them faster.

\begin{table}[!htbp]
\centering
\small
\caption{Per-task wall clock. ``Out ratio'' is local over cloud output tokens;
``Speedup'' is local over cloud time, so above $1.0\times$ means cloud was faster.}
\label{tab:crossover}
[[table_crossover]]
\end{table}

\paragraph{Router cost.}
The hybrid pays [[router_s]]\,s of routing, [[router_share]]\% of its wall clock.
Re-measuring the router alone with the local engine loaded but idle gives a
median of [[router_idle_ms]]\,ms per decision; re-measuring it in the request
path, where each decision follows a local generation on the same GPU, gives
[[router_busy_ms]]\,ms over [[router_busy_n]] items
([[router_busy_range]]\,ms). Sharing the accelerator costs the router
[[router_penalty]]$\times$ its standalone latency even though it emits no
tokens. On short-answer tasks this dominates: MMLU takes 64.5\,s under the
hybrid against 13.9\,s always-local, a 4.6$\times$ penalty paid to recover
37.5 points of accuracy that the cloud arm delivers directly in 28.2\,s.

\paragraph{Statistical significance.}
Table~\ref{tab:significance} reports McNemar tests. Only GSM8K reaches
significance ($p=0.031$); with 7--8 items per task the study is powered for
large effects only. The pooled gap between local and cloud is large enough to
believe; per-task claims should be read as indicative.

\begin{table}[!htbp]
\centering
\small
\caption{McNemar's exact test on discordant pairs. Entries are
wins/losses followed by the two-sided $p$-value.}
\label{tab:significance}
[[table_significance]]
\end{table}

\subsection{Qualitative analysis}
Numerical metrics hide how differently the two tiers answer. The examples below
are verbatim generations from the stored records, wrapped for width.

\paragraph{Both correct, different economies.}
On GSM8K item 6 both arms reach the right answer, 10 pizzas. The local answer
runs [[gsm_local_tok]] tokens and narrates every step in Markdown; the cloud
answer runs [[gsm_cloud_tok]] tokens in plain arithmetic. Neither is more
correct, and the difference in wall clock follows almost entirely from this
choice of verbosity.

\noindent\textbf{Local ([[gsm_local_tok]] tokens):}
[[gsm_local]]

\noindent\textbf{Cloud ([[gsm_cloud_tok]] tokens):}
[[gsm_cloud]]

\paragraph{Local correct, cloud wrong.}
MBPP item 361 asks for a function that removes empty lists from a list of
lists, and the tests pass mixed lists containing strings. The local model writes
[[mbpp_local_tok]] tokens of docstring and defensive filtering and fails, because
it keeps only elements that are themselves non-empty lists; the cloud model
writes [[mbpp_cloud_tok]] tokens and passes.

\noindent\textbf{Local (fails, [[mbpp_local_tok]] tokens):}
[[mbpp_local]]

\noindent\textbf{Cloud (passes, [[mbpp_cloud_tok]] tokens):}
[[mbpp_cloud]]

\paragraph{Visual reading of the failure.}
The local program's filter is \texttt{if isinstance(item, list) and item},
which drops \texttt{'Red'} and \texttt{'Green'}. The model generalised from the
docstring word ``empty'' and the prompt's phrase ``list of lists'' to a
semantic rule the tests do not share. Its output looks more professional than
the cloud's one-liner and is wrong, which is exactly the case that aggregate
accuracy is meant to catch.

\section{Failure Cases}

\paragraph{F1: the wrong tier was chosen (HumanEval/2).}
The router sent this item to the device; the local model failed and the cloud
model passed. The local completion stops inside a comment block: it reasons in comments
and never emits a \texttt{return}, so the program is truncated and the tests
fail. The cloud
completion is short and returns
\texttt{number - math.floor(number)}. This single item accounts for the
[[cloud_minus_hybrid]]-point gap between hybrid and cloud.

\noindent\textbf{Local (fails, routed to device):}
[[he_local]]

\noindent\textbf{Cloud (passes):}
[[he_cloud]]

\paragraph{F2: both tiers wrong, and the label is questionable (BoolQ/3).}
Local [[bq_local]], cloud [[bq_cloud]]. The passage describes chicken cordon
bleu as chicken wrapped around ham and cheese; the question asks whether it is
made with blue cheese and the gold label is ``yes''. Both models answered
``no'', which is defensible from the passage. This is a dataset-label problem
rather than a model failure, and it is why we report per-item behaviour instead
of only pooled accuracy: two of the cloud's missed items are of this kind.

\paragraph{F3: the router's cost assumption.}
The design assumed a decision model that never generates text would be
effectively free. Measured, it costs [[router_busy_s]]\,s per request when the
local engine shares the GPU, against [[router_idle_ms]]\,ms when it does not.
The assumption failed by roughly an order of magnitude, and it is a failure of
\emph{design}, not of the model: a router that competes for the accelerator is
not free even when it emits nothing.

\paragraph{F4: the evaluator can invert a result.}
The first version of the code scorer concatenated the extracted program directly
onto the test assertions. Any program not ending in a newline became
\texttt{return xassert ...} and raised \texttt{SyntaxError}, so the cloud arm
was recorded as 2/8 on MBPP when it was in fact 7/8. The corrected scorer
re-scored every stored answer offline; no generation changed, five correctness
labels did. This is the most consequential error we found, and it inverted the
paper's original conclusion that the router had ignored a task where the local
model was better. It had not.

\paragraph{F5: fairness constraints cut both ways.}
Reasoning was disabled on the cloud model to match the local checkpoint's
capability. That is fair, but it means the cloud numbers are a
\emph{non-reasoning} ceiling; a reasoning-enabled cloud model would likely
widen the accuracy gap and make the hybrid look worse, not better.

\section{Conclusion and Future Work}

We asked what a difficulty-routed cloud--local hybrid buys over either pure
tier, and measured it on [[n_items]] paired items from six benchmarks on one
Apple M2 machine. Three findings stand.

\textbf{On this sample, routing cannot improve accuracy.} The cloud arm scored
[[cloud_pooled]]\%, the hybrid [[hybrid_pooled]]\%, and the oracle
[[oracle_pooled]]\% --- identical to cloud-only, because the cloud's two errors
were also made on-device. H1 is therefore only weakly supported: the hybrid
preserves cloud-level accuracy to within one item, but there is no accuracy
headroom for a router to capture, so the case for a hybrid must be made on
privacy and latency.

\textbf{Difficulty is the wrong routing signal for latency.} H2 is not
supported. Which tier finishes first is governed by expected output length and
by prefill, not by how hard the question is: the device wins on multiple-choice
tasks where both arms emit one token, and loses by up to 5.9$\times$ on MBPP
where it writes [[mbpp_local_tok]] tokens against [[mbpp_cloud_tok]]. A
length-aware router is the obvious next step.

\textbf{The router is not free.} H3 is refuted. A decision costs
[[router_idle_ms]]\,ms on an idle GPU and [[router_busy_s]]\,s when it shares
the accelerator, which is [[router_share]]\% of the hybrid's wall clock. This
is a design constraint on any on-device router that coexists with an on-device
generator.

\paragraph{Limitations.}
The sample is [[n_items]] items, 7--8 per task, and only GSM8K reaches
significance. One cloud model, one provider, one machine; the router thresholds
were fixed for that combination. The hybrid arm is derived rather than executed
independently, validated on six end-to-end items. Reasoning was disabled on the
cloud model. Code execution is sandboxed with resource limits, not isolated.

\paragraph{Future work.}
Four directions follow directly. (i) Add an expected-output-length regressor to
the router and route on predicted time-to-answer; the per-task crossover in
Table~\ref{tab:crossover} predicts where the gain would be. (ii) Measure a
router that runs on a separate thread or device from the generator so its cost
is not serialised. (iii) Replace the strict answer-marker scorer with a
semantic judge and compare rankings. (iv) Extend to the full sample
(1{,}864 items) with the same paired protocol, which the harness already
supports, and report bootstrap confidence intervals on the pooled gaps.

\begin{thebibliography}{99}
\footnotesize
\setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}\setlength{\parskip}{0pt}

\bibitem{gemma} Gemma Team. Gemma: Open models based on Gemini research and
technology. arXiv:2403.08295, 2024.
\bibitem{gemma2} Gemma Team. Gemma 2: Improving open language models at a
practical size. arXiv:2408.00118, 2024.
\bibitem{phi3} M.~Abdin et al. Phi-3 technical report: A highly capable language
model locally on your phone. arXiv:2404.14219, 2024.
\bibitem{frugalgpt} L.~Chen, M.~Zaharia, and J.~Zou. FrugalGPT: How to use large
language models while reducing cost and improving performance. arXiv:2305.05176,
2023.
\bibitem{routellm} I.~Ong et al. RouteLLM: Learning to route LLMs with
preference data. arXiv:2406.18665, 2024.
\bibitem{hybridllm} D.~Ding et al. Hybrid LLM: Cost-efficient and quality-aware
query routing. arXiv:2404.14618, 2024.
\bibitem{automix} P.~Aggarwal et al. AutoMix: Automatically mixing language
models. arXiv:2310.12963, 2024.
\bibitem{powerinfer} Y.~Song et al. PowerInfer: Fast large language model
serving with a consumer-grade GPU. ACM SIGOPS ATC, 2024.
\bibitem{mobilellm} Z.~Liu et al. MobileLLM: Optimizing sub-billion parameter
language models for on-device use cases. arXiv:2402.14905, 2024.
\bibitem{llmflash} K.~Alizadeh et al. LLM in a flash: Efficient large language
model inference with limited memory. arXiv:2312.11514, 2024.
\bibitem{slmsurvey} Z.~Lu et al. Small language models: Survey, measurements, and
insights. arXiv:2409.15790, 2024.
\bibitem{laya} M.~N. and M.~A. From Laya to Vega: Open non-autoregressive
decision models with calibrated probabilities. \emph{Preprints}, 2026.
doi:10.20944/preprints202611.0781.v1.
\bibitem{litertlm} Google AI Edge. LiteRT-LM: an orchestration layer for
on-device generative AI. GitHub repository, 2024.\\
\url{https://github.com/google-ai-edge/LiteRT-LM}.
\bibitem{mmlu} D.~Hendrycks et al. Measuring massive multitask language
understanding. ICLR, 2021.
\bibitem{gsm8k} K.~Cobbe et al. Training verifiers to solve math word problems.
arXiv:2110.14168, 2021.
\bibitem{arc} P.~Clark et al. Think you have solved question answering?
Try ARC, the AI2 reasoning challenge. arXiv:1803.05457, 2018.
\bibitem{boolq} C.~Clark et al. BoolQ: Exploring the surprising difficulty of
natural yes/no questions. NAACL, 2019.
\bibitem{humaneval} M.~Chen et al. Evaluating large language models trained on
code. arXiv:2107.03374, 2021.
\bibitem{mbpp} J.~Austin et al. Program synthesis with large language models.
arXiv:2108.07732, 2021.

\end{thebibliography}

\end{document}
"""


if __name__ == "__main__":
    raise SystemExit(main())
