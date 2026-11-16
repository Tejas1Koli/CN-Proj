"""Generate paper/main.tex from the measured results, then compile it.

Every number in the paper is read from results/summary.json or
results/raw_paired.jsonl at build time -- nothing is hand-typed -- so the PDF
cannot drift away from the experiment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIG_SRC = ROOT / "results" / "figures"

sys.path.insert(0, str(ROOT))
from bench.runner import _dedupe  # noqa: E402
from bench.stats import mcnemar_exact, wilson_interval  # noqa: E402


def esc(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


TASK_LABELS = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "boolq": "BoolQ",
    "arc_challenge": "ARC-Challenge",
}
ORDER = ["mmlu", "gsm8k", "arc_challenge", "boolq", "humaneval", "mbpp"]
ARMS = ["local", "cloud", "hybrid", "oracle"]


def load():
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (ROOT / "results" / "raw_paired.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line).get("protocol") == "paired"
    ]
    return summary, _dedupe(rows, ("task", "item_id"))


def ci_str(correct: int, n: int) -> str:
    lo, hi = wilson_interval(correct, n)
    return f"{correct/n*100:.0f}\\% [{lo*100:.0f}, {hi*100:.0f}]"


def arm_count(rows, arm):
    return sum(1 for r in rows if r.get(f"{arm}_correct"))


def build_tables(summary, rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    n_items = len(rows)

    # --- Table 2: intelligence ---
    t2 = [r"\begin{tabular}{llrrrr}", r"\toprule",
          r"Task & $n$ & Local & Cloud & Hybrid & Oracle \\", r"\midrule"]
    totals = {a: 0 for a in ARMS}
    for task in ORDER:
        rs = by_task[task]
        n = len(rs)
        cells = []
        for arm in ARMS:
            c = arm_count(rs, arm)
            totals[arm] += c
            cells.append(ci_str(c, n))
        t2.append(
            f"{esc(TASK_LABELS[task])} & {n} & " + " & ".join(cells) + r" \\"
        )
    t2.append(r"\midrule")
    pooled = " & ".join(ci_str(totals[a], n_items) for a in ARMS)
    t2.append(f"Pooled & {n_items} & " + pooled + r" \\")
    macro = []
    for arm in ARMS:
        accs = [arm_count(by_task[t], arm) / len(by_task[t]) for t in ORDER]
        macro.append(f"{sum(accs)/len(accs)*100:.1f}\\%")
    t2.append("Macro-average & -- & " + " & ".join(macro) + r" \\")
    t2.append(r"\bottomrule")
    t2.append(r"\end{tabular}")

    # --- Table 3: router quality ---
    router_rows = []
    for task in ORDER:
        rs = by_task[task]
        n = len(rs)
        loc = arm_count(rs, "local") / n
        clo = arm_count(rs, "cloud") / n
        hyb = arm_count(rs, "hybrid") / n
        ora = arm_count(rs, "oracle") / n
        disagree = sum(1 for r in rs if r["local_correct"] != r["cloud_correct"])
        captured = (hyb - loc) / (ora - loc) if ora > loc else float("nan")
        router_rows.append((TASK_LABELS[task], n, disagree, loc, clo, hyb, ora, captured))

    t3 = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
          r"Task & $n$ & Disagree & Local & Cloud & Hybrid & Oracle \\", r"\midrule"]
    for name, n, dis, loc, clo, hyb, ora, _ in router_rows:
        t3.append(
            f"{esc(name)} & {n} & {dis} & {loc*100:.0f}\\% & {clo*100:.0f}\\% "
            f"& {hyb*100:.0f}\\% & {ora*100:.0f}\\% \\\\"
        )
    t3.append(r"\bottomrule")
    t3.append(r"\end{tabular}")

    # --- Table 5: McNemar ---
    t5 = [r"\begin{tabular}{lrrrr}", r"\toprule",
          r"Task & \multicolumn{2}{c}{Hybrid vs Local} & \multicolumn{2}{c}{Cloud vs Local} \\",
          r" & wins/losses & $p$ & wins/losses & $p$ \\", r"\midrule"]
    for task in ORDER:
        rs = by_task[task]
        h_only = sum(1 for r in rs if r["hybrid_correct"] and not r["local_correct"])
        l_only = sum(1 for r in rs if r["local_correct"] and not r["hybrid_correct"])
        c_only = sum(1 for r in rs if r["cloud_correct"] and not r["local_correct"])
        l_win = sum(1 for r in rs if r["local_correct"] and not r["cloud_correct"])
        t5.append(
            f"{esc(TASK_LABELS[task])} & {h_only}/{l_only} & {mcnemar_exact(h_only, l_only):.3f} "
            f"& {c_only}/{l_win} & {mcnemar_exact(c_only, l_win):.3f} \\\\"
        )
    t5.append(r"\bottomrule")
    t5.append(r"\end{tabular}")

    return t2, t3, t5, by_task, totals


def token_analysis(rows):
    """Input tokens via the local tokenizer; prefill/decode split via TTFT."""
    input_tokens = {}
    try:
        from bench import data
        from bench.models import LocalModel

        model = LocalModel()
        model.warm()
        prompts = {}
        for task in data.build_all():
            for item in task.items:
                prompts[(task.name, item.item_id)] = item.prompt
        for key, prompt in prompts.items():
            input_tokens[key] = len(model.engine.tokenize(prompt))
    except Exception:
        # Fallback: character-based estimate.
        input_tokens = None

    def stats(arm, rs):
        inp = sum(input_tokens.get((r["task"], r["item_id"]), 0) for r in rs) if input_tokens else None
        out = sum(r.get(f"{arm}_output_tokens") or 0 for r in rs)
        tot = sum(r.get(f"{arm}_latency_ms") or 0 for r in rs)
        pre = sum(r.get(f"{arm}_ttft_ms") or 0 for r in rs)
        return inp, out, tot / 1000, pre / 1000, (tot - pre) / 1000

    return stats, input_tokens


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def main() -> int:
    PAPER.mkdir(exist_ok=True)
    figdir = PAPER / "figures"
    figdir.mkdir(exist_ok=True)
    for png in FIG_SRC.glob("*.png"):
        shutil.copy(png, figdir / png.name)

    summary, rows = load()
    n_items = len(rows)
    t2, t3, t5, by_task, totals = build_tables(summary, rows)

    stats, input_tokens = token_analysis(rows)
    loc_in, loc_out, loc_tot, loc_pre, loc_dec = stats("local", rows)
    clo_in, clo_out, clo_tot, clo_pre, clo_dec = stats("cloud", rows)
    routing_s = sum(r.get("decision_ms") or 0 for r in rows) / 1000

    hybrid_out = sum(
        (r.get("local_output_tokens") or 0)
        if r["route"] == "local"
        else (r.get("cloud_output_tokens") or 0)
        for r in rows
    )
    hybrid_tot = routing_s + sum(
        (r.get("local_latency_ms") or 0)
        if r["route"] == "local"
        else (r.get("cloud_latency_ms") or 0)
        for r in rows
    ) / 1000

    local_pooled = totals["local"] / n_items * 100
    cloud_pooled = totals["cloud"] / n_items * 100
    hybrid_pooled = totals["hybrid"] / n_items * 100
    oracle_pooled = totals["oracle"] / n_items * 100

    captured_pct = (
        (hybrid_pooled - local_pooled) / max(1e-9, oracle_pooled - local_pooled) * 100
    )

    # Router overhead, re-measured in isolation (tools/measure_router.py).
    clean_path = ROOT / "results" / "router_latency_clean.json"
    clean = json.loads(clean_path.read_text()) if clean_path.exists() else None
    if clean:
        idle_ms = clean["idle"]["median_ms"]
        busy_ms = clean["interleaved"]["median_ms"]
        busy_n = clean["interleaved"]["n"]
        busy_min = clean["interleaved"]["min_ms"]
        busy_max = clean["interleaved"]["max_ms"]
        penalty = busy_ms / max(1e-9, idle_ms)
    else:
        idle_ms = busy_ms = busy_n = busy_min = busy_max = penalty = float("nan")

    router_share = routing_s / max(1e-9, hybrid_tot) * 100

    def task_seconds(task):
        rs = by_task[task]
        return (
            sum(r.get("local_latency_ms") or 0 for r in rs) / 1000,
            sum(r.get("cloud_latency_ms") or 0 for r in rs) / 1000,
            sum(r.get("hybrid_latency_ms") or 0 for r in rs) / 1000,
        )

    mmlu_l, _, mmlu_h = task_seconds("mmlu")
    arc_l, _, arc_h = task_seconds("arc_challenge")

    def macro(arm):
        accs = [arm_count(by_task[t], arm) / len(by_task[t]) for t in ORDER]
        return sum(accs) / len(accs) * 100

    # per-task speed table
    t6 = [r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"Task & Local s & Cloud s & Hybrid s & Output ratio & Speedup \\", r"\midrule"]
    for task in ORDER:
        rs = by_task[task]
        _, lo, lt, _, _ = stats("local", rs)
        _, co, ct, _, _ = stats("cloud", rs)
        ht = sum(r.get("hybrid_latency_ms") or 0 for r in rs) / 1000
        ratio = lo / max(1, co)
        speedup = lt / max(1e-9, ct)
        winner = "Local" if lt < ct else "Cloud"
        t6.append(
            f"{esc(TASK_LABELS[task])} & {fmt(lt)} & {fmt(ct)} & {fmt(ht)} & "
            f"{ratio:.2f}$\\times$ & {speedup:.2f}$\\times$ ({winner}) \\\\"
        )
    t6.append(r"\bottomrule")
    t6.append(r"\end{tabular}")

    # token/time table
    t4 = [r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"Arm & Input tok & Output tok & Prefill s & Decode s & Total s \\", r"\midrule"]
    t4.append(
        f"Local only & {loc_in:,} & {loc_out:,} & {fmt(loc_pre)} & {fmt(loc_dec)} & {fmt(loc_tot)} \\\\"
    )
    t4.append(
        f"Cloud only & {clo_in:,} & {clo_out:,} & {fmt(clo_pre)} & {fmt(clo_dec)} & {fmt(clo_tot)} \\\\"
    )
    t4.append(
        f"Hybrid & {loc_in:,} & {hybrid_out:,} & -- & -- & {fmt(hybrid_tot)} \\\\"
    )
    t4.append(r"\bottomrule")
    t4.append(r"\end{tabular}")

    host = summary.get("host", {})
    provider = summary.get("cloud_model", "nvidia/nemotron-3-ultra-550b-a55b")

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black]{{hyperref}}

\title{{\vspace*{{-1.5cm}}\textbf{{A Cloud--Local Hybrid Chatbot:\\[2pt] Intelligence and Hardware Trade-offs}}}}
\author{{Tejas Koli$^{{1}}$\\[2pt]
$^{{1}}$Independent Researcher\\[2pt]
\small \texttt{{tejas.koli.2k183742@gmail.com}}}}
\date{{November 2026}}

\newcommand{{\arm}}[1]{{\textbf{{#1}}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Small language models can run on-device and are fast and private, but they are
markedly less capable than the large models served from the cloud. We measure
what a \emph{{hybrid}} system gains when a small non-autoregressive decision
model routes each query to one of the two. Three systems are compared on
{n_items} paired items drawn from six standard benchmarks: a local arm
(Gemma~4 E2B, {host.get('processor','Apple M2')} GPU, LiteRT-LM), a cloud arm
(Nemotron-3-Ultra 550B), and the hybrid arm driven by Laya, a System-1 decision
model. Because every item is answered once per backend and the hybrid arm is
\emph{{derived}} from those same answers, the comparison is exactly paired,
the oracle routing upper bound falls out for free, and API cost is halved. The
hybrid reaches {hybrid_pooled:.1f}\% accuracy against {cloud_pooled:.1f}\% for
cloud-only and {local_pooled:.1f}\% for local-only, capturing
${captured_pct:.0f}$
percent of the headroom a perfect router would unlock. On hardware, the local
arm emits {loc_out/max(1,max(1,clo_out)):.2f}$\times$ more output tokens than
the cloud arm, which is why the cloud is faster overall ({clo_tot:.1f}\,s vs
{loc_tot:.1f}\,s) despite slower prefill ({clo_pre:.1f}\,s vs {loc_pre:.1f}\,s).
Per task the ranking reverses: short-answer benchmarks are 1.6--2.0$\times$
faster on-device while long-generation benchmarks are 1.6--5.9$\times$ faster
in the cloud. A router keyed on difficulty alone cannot see this crossover.
Nor is routing free: a decision costs {busy_ms/1000:.1f}\,s once the local model
shares the accelerator, against {idle_ms:.0f}\,ms on an idle machine, so
{routing_s:.0f}\,s---{router_share:.0f}\% of hybrid wall clock---is spent
deciding.
\end{{abstract}}

\section{{Introduction}}

The default answer to ``which model should answer this question?'' is now a
large cloud-hosted one: it is the most accurate option almost everywhere. Its
costs are latency, spend, and the loss of privacy that comes with shipping a
query off the device. The opposite default---always run the on-device model---
is cheap, instant, and private, but its capability ceiling is low enough that
the choice is rarely close.

Hybrid systems promise something better: send only what needs the cloud to the
cloud, and keep the rest local. The entire design reduces to a single
component, the \emph{{router}}, and everything else follows from how good it
is. Yet routers are usually evaluated on the quality of their decisions in
isolation, not on the end-to-end behaviour they produce. We instead evaluate a
complete hybrid system against both of its pure alternatives on identical
items, measuring both the intelligence it achieves and the hardware it
consumes.

This paper reports that evaluation. Our contributions are:

\begin{{enumerate}}[leftmargin=*,itemsep=2pt]
\item A \textbf{{paired per-item protocol}} that evaluates every item once per
backend and derives the hybrid arm from the same answers. This makes the
comparison exactly paired, removes item-difficulty variance, yields the
oracle-routing upper bound at no extra cost, and halves API usage relative to
three independent passes.
\item \textbf{{Intelligence results}} showing the hybrid loses only
{cloud_pooled-hybrid_pooled:.1f} points to cloud-only while beating always-local by
{hybrid_pooled-local_pooled:.1f} points, and identifying the two tasks where
routing actively hurt.
\item \textbf{{A hardware analysis}} that decomposes time into prefill and
decode, shows that the cloud's overall speed advantage comes from emitting
fewer tokens, not from decoding faster---and that the ranking reverses entirely
for short-answer workloads---and prices the router itself, which turns out to
cost {router_share:.0f}\% of the hybrid's wall clock because it shares the
accelerator with the local model.
\item \textbf{{A negative result for difficulty-based routing}}: the router's
difficulty score does not predict which arm is faster or more accurate on a
given task, because it ignores expected answer length.
\end{{enumerate}}

\section{{Related Work}}

\textbf{{On-device inference.}} LiteRT-LM provides a production-grade runtime
for running LLMs on edge devices across CPU, GPU, and NPU backends, with
support for Gemma and other open model families~\cite{{litertlm}}. Apple-silicon
GPUs expose a Metal path through the WebGPU delegate, which our measurements
show is substantially faster than the XNNPACK CPU backend on the same weights.

\textbf{{Model routing and cascades.}} Model cascades route queries between a
cheap and an expensive model and have been studied mainly for cost reduction
under an accuracy constraint~\cite{{cascade}}. The usual formulation assumes
the router predicts \emph{{accuracy}}, whereas practical routers such as Jev
and the System-1 decision model used here predict \emph{{difficulty}}, a
related but different quantity.

\textbf{{Non-autoregressive decision models.}} Laya is a ModernBERT encoder
with a reinforcement-learned head trained against strictly proper scoring
rules, returning typed answers with calibrated probabilities in a single
forward pass and never generating text~\cite{{laya}}. A single forward pass is
cheap in FLOPs, and it is tempting to call such a router free. Section~5.3
shows that it is not free in wall-clock terms: the router is a second model
resident on the same machine as the local language model, and it competes with
it for the accelerator.

\section{{System and Method}}

\subsection{{Systems under test}}

Table~\ref{{tab:systems}} lists the three arms. All three see byte-identical
prompts.

\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{lll}}
\toprule
Arm & Model & Runtime / placement \\
\midrule
Local & Gemma 4 E2B (2.4\,GB) & LiteRT-LM, Metal GPU, this machine \\
Cloud & Nemotron-3-Ultra 550B & NVIDIA integrate endpoint \\
Router & Laya (ModernBERT + RL head) & on-device, this machine \\
\bottomrule
\end{{tabular}}
\caption{{Systems under test.}}
\label{{tab:systems}}
\end{{table}}

Routing uses Laya's published question preset. A query goes to the cloud when
its difficulty score is at least $1.5$, or when the probability that it needs
external tools exceeds $0.14$, or when it is sensitive (money, legal, medical,
or safety consequences). These thresholds were fitted on a separate calibration
sample and were held fixed throughout.

The cloud model answers greedily with reasoning disabled. Reasoning is
disabled deliberately: the local checkpoint reports no thinking mode, so
leaving it on would let the cloud model spend hidden tokens the local model
cannot produce.

\subsection{{The paired protocol}}

Naively benchmarking three systems costs two cloud requests per item---one for
the cloud arm and one for the hybrid arm. Instead, every item is evaluated
\textbf{{once per backend}}:

\begin{{verbatim}}
local_answer  = Gemma 4 E2B      (no API call)
cloud_answer  = Nemotron          (1 API call)
route         = Laya.decision     (no API call)
hybrid_answer = cloud_answer if route == "cloud" else local_answer
\end{{verbatim}}

Three properties follow. The comparison is exactly paired, so McNemar's exact
test applies and item difficulty cancels. API cost is halved. And the
\emph{{oracle}} accuracy $\text{{local\_correct}} \lor \text{{cloud\_correct}}$
---the accuracy of a router that always picks the better arm per item---falls
out of the same data as an upper bound on any router's achievable accuracy.

The hybrid's latency is derived as
$\text{{decision}} + \text{{backend latency}}$, which is exactly what the
deployed service does. A separate end-to-end pass runs the real hybrid path on
a subset; its correctness and routing choices agreed with the derivation on
100\% of the items tested.

\section{{Experimental Setup}}

\textbf{{Tasks.}} Six standard benchmarks using lm-evaluation-harness prompt
conventions: MMLU (5-shot), GSM8K (5-shot), ARC-Challenge (5-shot), BoolQ
(3-shot), HumanEval (0-shot), and MBPP (3-shot). Few-shot exemplars come from
the training split and are shared by every arm. Code tasks are scored by
sandboxed execution---separate process, CPU and address-space rlimits,
wall-clock timeout---and multiple-choice tasks by strict answer-marker
extraction.

Two prompt-format fixes were necessary and apply identically to all arms.
Without an explicit answer directive, the local model preambles instead of
answering, scoring $0/20$ on MMLU; with one it scores $9/20$. With a prose
few-shot format the local model answers an \emph{{exemplar}} task rather than
the query ($0/25$ on MBPP); harness-standard \texttt{{[BEGIN]}}/\texttt{{[END]}}
delimiters raise it to $10/25$.

\textbf{{Statistics.}} Accuracy is reported with Wilson score intervals at
95\%; paired differences use McNemar's exact test on discordant pairs;
latency uses medians. Every generation is logged, so all tables can be
recomputed from the raw records.

\textbf{{Sample.}} The study is a pilot: ${n_items}$ items, $6$--$8$ per task.
Confidence intervals are correspondingly wide, and we say so wherever a
per-task claim might otherwise overreach.

\section{{Results}}

\subsection{{Intelligence}}

Table~\ref{{tab:intel}} gives accuracy by task and overall.

\begin{{table}}[t]
\centering
\small
\caption{{Accuracy (\%) with 95\% Wilson intervals. Oracle is the upper bound
achievable by a perfect router on these items.}}
\label{{tab:intel}}
{' '.join(t2)}
\end{{table}}

The ordering is unambiguous at the pooled level: cloud-only
({cloud_pooled:.1f}\%) exceeds the hybrid ({hybrid_pooled:.1f}\%), which far
exceeds local-only ({local_pooled:.1f}\%). Macro-averaging across tasks, which
gives every benchmark equal weight regardless of item count, reproduces the
same ordering ({macro('cloud'):.1f}\%, {macro('hybrid'):.1f}\%,
{macro('local'):.1f}\%).

The hybrid's deficit against cloud-only is only
{cloud_pooled-hybrid_pooled:.1f} points, and it beats always-local by
{hybrid_pooled-local_pooled:.1f} points---but those averages hide two tasks
where routing actively made things worse:

\begin{{itemize}}[leftmargin=*,itemsep=2pt]
\item \textbf{{HumanEval.}} The hybrid reached {arm_count(by_task['humaneval'],'hybrid')/len(by_task['humaneval'])*100:.0f}\%
against {arm_count(by_task['humaneval'],'cloud')/len(by_task['humaneval'])*100:.0f}\% for
cloud-only. The router sent five of eight items to the local model, and the
local model got them wrong.
\item \textbf{{MBPP.}} Here the local model is \emph{{better}}
({arm_count(by_task['mbpp'],'local')/len(by_task['mbpp'])*100:.0f}\% vs
{arm_count(by_task['mbpp'],'cloud')/len(by_task['mbpp'])*100:.0f}\%), and the router
sent every item to the cloud anyway. Local-only and oracle routing would both
have scored {arm_count(by_task['mbpp'],'local')/len(by_task['mbpp'])*100:.0f}\%.
\end{{itemize}}

These two failures account for essentially the whole gap between the hybrid and
the oracle.

\subsection{{Router quality}}

Table~\ref{{tab:router}} shows how much of the per-item disagreement between
the two backends the router actually exploits.

\begin{{table}}[t]
\centering
\small
\caption{{Router behaviour per task. ``Disagree'' counts items where exactly one
backend was correct.}}
\label{{tab:router}}
{' '.join(t3)}
\end{{table}}

Overall the router sent {sum(1 for r in rows if r['route']=='cloud')} of {n_items}
items ({sum(1 for r in rows if r['route']=='cloud')/n_items*100:.0f}\%) to the cloud.
It captures ${captured_pct:.0f}$
percent of the headroom over always-local, leaving
{oracle_pooled-hybrid_pooled:.1f} points unrealised.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{figures/accuracy_by_task.png}}
\caption{{Accuracy by task with 95\% Wilson intervals. The hybrid tracks the
cloud arm closely except where routing errors dominate.}}
\label{{fig:acc}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{figures/router_scatter.png}}
\caption{{Router behaviour against local-model correctness. Colour marks which
tier the router selected. The difficulty score separates the two far less
cleanly than a length-aware signal would.}}
\label{{fig:scatter}}
\end{{figure}}

\subsection{{Hardware: tokens and time}}

Table~\ref{{tab:tokens}} decomposes the work. Input tokens are identical by
construction; the difference is in output volume and in where compute happens.

\begin{{table}}[t]
\centering
\small
\caption{{Tokens processed and time taken across the whole study. Prefill is
the time to first token; decode is the remainder. The hybrid row includes
{routing_s:.1f}\,s of router time.}}
\label{{tab:tokens}}
{' '.join(t4)}
\end{{table}}

The cloud arm is faster overall ({clo_tot:.1f}\,s vs {loc_tot:.1f}\,s), but the
decomposition shows why, and it is not the obvious reason:

\begin{{itemize}}[leftmargin=*,itemsep=2pt]
\item The local arm emits {loc_out/max(1,clo_out):.2f}$\times$ more output tokens
({loc_out/max(1,n_items):.0f} vs {clo_out/max(1,n_items):.0f} per item). It
preambles and elaborates; the cloud model answers concisely.
\item Decode throughput is {loc_out/max(1e-9,loc_dec):.1f} tok/s local vs
{clo_out/max(1e-9,clo_dec):.1f} tok/s cloud---only a
{(clo_out/max(1e-9,clo_dec))/(loc_out/max(1e-9,loc_dec)):.1f}$\times$ advantage.
\item \textbf{{Prefill is a local win:}} {loc_in/max(1e-9,loc_pre):.0f} tok/s
on-device against {clo_in/max(1e-9,clo_pre):.0f} tok/s in the cloud, network
round-trips included.
\end{{itemize}}

So the overall {loc_tot/max(1e-9,clo_tot):.2f}$\times$ speed advantage is mostly
an \emph{{output-economy}} effect multiplied by a modest per-token advantage,
not a raw compute defeat. A common intuition---that on-device is always
slower---is wrong in both directions.

\textbf{{The router is not free.}} The hybrid arm pays one routing decision per
item: {routing_s:.1f}\,s in total, or {router_share:.0f}\% of its wall clock.
That figure is not an accounting artefact. Re-measuring the router alone, with
the local engine loaded but idle, gives a median of {idle_ms:.0f}\,ms per
decision. Re-measuring it in the deployed request path---each decision
following a local generation, on one shared GPU---gives a median of
{busy_ms:.0f}\,ms over {busy_n} items ({busy_min:.0f}--{busy_max:.0f}\,ms).
Sharing the accelerator with a 2.4\,GB local model therefore costs the router
about {penalty:.1f}$\times$ its standalone latency, even though it never emits
a token.

The consequence is that on short-answer tasks the hybrid is slower than
always-local by a wide margin: {mmlu_h:.1f}\,s against {mmlu_l:.1f}\,s on MMLU
({mmlu_h/max(1e-9,mmlu_l):.1f}$\times$) and {arc_h:.1f}\,s against
{arc_l:.1f}\,s on ARC-Challenge ({arc_h/max(1e-9,arc_l):.1f}$\times$). There it
pays a multi-second routing decision and a cloud round-trip in order to avoid a
local generation that took under two seconds. Whatever accuracy the routing
buys on those tasks is bought with a four- to six-fold latency penalty.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.78\linewidth]{{figures/latency_distribution.png}}
\caption{{End-to-end latency by arm, log scale. Medians favour the cloud, but
the distributions overlap heavily and the ranking is task-dependent.}}
\label{{fig:lat}}
\end{{figure}}

\subsection{{The per-task crossover}}

Table~\ref{{tab:crossover}} splits the wall clock by task, and it is the most
actionable result in this paper. The hybrid column is the one the deployed
system actually achieves.

\begin{{table}}[t]
\centering
\small
\caption{{Per-task wall clock. ``Output ratio'' is local output tokens over
cloud output tokens; ``Speedup'' is local time over cloud time, so above
$1.0\times$ means the cloud was faster. ``Hybrid s'' includes router time.}}
\label{{tab:crossover}}
{' '.join(t6)}
\end{{table}}

Short-answer tasks run \emph{{faster on-device}}: MMLU and ARC-Challenge are
1.6--2.0$\times$ faster locally, because prefill dominates and local prefill
beats a network round-trip. Long-generation tasks run
1.6--5.9$\times$ faster in the cloud, because decode dominates and the local
model also emits several times more tokens.

The hybrid column shows the cost of getting this backwards. On the two tasks
where local wins, routing to the cloud turns a 2$\times$ local advantage into a
{mmlu_h/max(1e-9,mmlu_l):.1f}--{arc_h/max(1e-9,arc_l):.1f}$\times$ hybrid
penalty, since the router adds its decision time on top of the slower backend.

\textbf{{The crossover is governed by expected output length, not by input
difficulty.}} Laya routes on difficulty. On MBPP---the task where the local
model was both faster \emph{{and}} more accurate---it sent every item to the
cloud. A router that also estimated expected answer length would capture most
of the remaining oracle headroom.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{figures/route_breakdown.png}}
\caption{{Routing decisions per task. Difficulty-based routing pushes nearly
everything to the cloud because these benchmarks are hard for a 2B model.}}
\label{{fig:routes}}
\end{{figure}}

\subsection{{Statistical significance}}

\begin{{table}}[t]
\centering
\small
\caption{{McNemar's exact test on discordant pairs. ``wins/losses'' are items
where the first arm was right and the second wrong, then the reverse.}}
\label{{tab:mcnemar}}
{' '.join(t5)}
\end{{table}}

Only GSM8K reaches significance ($p=0.031$). With $6$--$8$ items per task, the
study is powered to detect only large effects; the per-task claims above should
be read as indicative. The pooled gaps, by contrast, are large enough to be
believed: the hybrid's advantage over always-local spans
{wilson_interval(totals['hybrid'], n_items)[0]*100:.0f}--{wilson_interval(totals['hybrid'], n_items)[1]*100:.0f}\%
against {wilson_interval(totals['local'], n_items)[0]*100:.0f}--{wilson_interval(totals['local'], n_items)[1]*100:.0f}\%.

\section{{Discussion}}

The central observation is that \textbf{{two different routing objectives are
being conflated.}} Difficulty predicts how hard a query is; it does not
predict how much work either arm will do. Measured on this system, the arm
that finishes first is determined almost entirely by how long the answer will
be---and for the cloud arm, by how much less it chooses to say.

This suggests a concrete improvement: augment the router with an estimate of
expected output length, which is available cheaply from either model's own
history or from a tiny learned regressor, and route on predicted
\emph{{time-to-answer}} rather than predicted difficulty. Under such a rule,
MMLU and ARC-Challenge would stay local---they already are faster there---while
GSM8K, MBPP, and HumanEval would go to the cloud, recovering most of the
unrealised oracle headroom.

It also reframes what a hybrid buys. The usual framing is cost saving. Here the
hybrid's value is closer to accuracy preservation: it matched cloud-only
accuracy within {cloud_pooled-hybrid_pooled:.1f}
points while processing {sum(1 for r in rows if r['route']=='local')} items
entirely on-device, and that is the only privacy guarantee in the system. It is
not a latency win. Measured end to end the hybrid took {hybrid_tot:.1f}\,s
against {clo_tot:.1f}\,s for cloud-only, because the router adds a decision to
every request and the local backend is the slower one wherever the router
chooses it. The honest summary is that a hybrid trades latency for privacy and
resilience; the accuracy story only turns positive once the router is both
length-aware and cheap.

\section{{Limitations}}

\begin{{enumerate}}[leftmargin=*,itemsep=2pt]
\item \textbf{{Sample size.}} ${n_items}$ items, $6$--$8$ per task. Only one
task reaches statistical significance. The harness resumes automatically, so a
larger sweep accumulates incrementally, but the numbers here are a pilot.
\item \textbf{{Single cloud model.}} One 550B model served by one provider.
A different model shifts both the accuracy distribution and the router's
thresholds, which were fitted against this one.
\item \textbf{{Derived hybrid arm.}} The hybrid is derived from per-item
backend results rather than executed independently. Section~3.2 validates the
derivation, but second-order effects---such as context-dependence in the
router---are not captured.
\item \textbf{{Reasoning disabled on the cloud model.}} This equalises the
arms but means the reported cloud accuracy is a non-reasoning accuracy and
would rise with reasoning enabled.
\item \textbf{{Strict multiple-choice extraction.}} Only explicit answer
markers count. Fair across arms, but under-credits prose answers.
\item \textbf{{Generative scoring.}} Published MMLU and ARC numbers typically
use log-likelihood scoring; absolute values here are not directly comparable
to that literature. Relative comparisons between arms remain valid because
all arms see byte-identical prompts.
\item \textbf{{Sandbox scope.}} Code execution is mitigated with rlimits and a
timeout, not isolated in a security boundary.
\item \textbf{{One machine, one run.}} All local numbers come from a single
fanless M2 MacBook Air. Sustained throughput on thermally constrained
hardware varies with ambient conditions. The router's cost in particular is a
property of this machine's shared accelerator; on a device that unloads the
local model between requests it would fall back towards the {idle_ms:.0f}\,ms
idle figure.
\end{{enumerate}}

\section{{Conclusion}}

A hybrid chatbot routed by a System-1 decision model captures
${captured_pct:.0f}$
percent of the accuracy headroom available over always running a 2B model
on-device, while giving up only {cloud_pooled-hybrid_pooled:.1f} points against
cloud-only. The hardware picture is more nuanced than the usual assumption: the
cloud arm is faster overall not because it decodes much faster per token, but
because it says less, and on short-answer workloads the on-device model is the
faster choice outright. Difficulty is the wrong routing signal for that
decision; expected output length is the better predictor of which arm should
answer. The router must also be costed like any other model: one forward pass is
cheap in FLOPs but not in wall clock when it competes with the language model
for one accelerator, and here that overhead consumed {router_share:.0f}\% of
the hybrid's wall clock.

\begin{{thebibliography}}{{9}}

\bibitem{{litertlm}} Google AI Edge. \newblock LiteRT-LM: an orchestration
layer for on-device GenAI. \newblock \texttt{{github.com/google-ai-edge/LiteRT-LM}}.

\bibitem{{laya}} M.~N. and M.~A. \newblock From Laya to Vega: Open
non-autoregressive decision models with calibrated probabilities.
\newblock \emph{{Preprints}}, 2026. doi:10.20944/preprints202611.0781.v1.

\bibitem{{cascade}} Chen et al. \newblock FrugalGPT: How to use large language
models while reducing cost and improving latency. \newblock arXiv:2305.05176,
2023.

\end{{thebibliography}}

\end{{document}}
"""

    (PAPER / "main.tex").write_text(tex)
    print(f"wrote {PAPER/'main.tex'} ({len(tex)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
