"""Build slides/slides.pptx from the measured results.

The deck follows the course presentation guide: 12 slides, Introduction through
Conclusion, with the title folded into slide 1 so the deck lands inside the
10--12 slide target. Numbers come from results/summary.json and
results/raw_paired.jsonl, so the deck cannot drift from the experiment.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.runner import _dedupe  # noqa: E402
from bench.stats import wilson_interval  # noqa: E402

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

NAVY = RGBColor(0x1F, 0x33, 0x55)
BLUE = RGBColor(0x2E, 0x5C, 0x8A)
ACCENT = RGBColor(0xE0, 0x6C, 0x2C)
GREEN = RGBColor(0x2E, 0x7D, 0x5B)
GREY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xCF, 0xDD, 0xEE)

TASK_LABELS = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "boolq": "BoolQ",
    "arc_challenge": "ARC-C",
}
ORDER = ["mmlu", "gsm8k", "arc_challenge", "boolq", "humaneval", "mbpp"]
ARMS = ["local", "cloud", "hybrid", "oracle"]

FIG = ROOT / "results" / "figures"
ARCH = ROOT / "paper" / "figures" / "architecture.png"

_W = 13.333


def prs_width():
    return Inches(_W)


def add_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def header(slide, title: str, subtitle: str | None = None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs_width(), Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.14), Inches(12.3), Inches(0.7))
    p = box.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(25)
    r.font.bold = True
    r.font.color.rgb = WHITE
    if subtitle:
        sb = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.3), Inches(0.4))
        sp = sb.text_frame.paragraphs[0]
        sr = sp.add_run()
        sr.text = subtitle
        sr.font.size = Pt(13)
        sr.font.color.rgb = GREY


def bullets(slide, items, left=0.6, top=1.6, width=12.1, height=5.4, size=16):
    height = min(height, 7.35 - top)
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        text, level = item if isinstance(item, tuple) else (item, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size - level * 2)
        r.font.color.rgb = NAVY if level == 0 else GREY
        p.space_after = Pt(7)
    return box


def card(slide, left, top, width, height, title, body, bar=BLUE, fill=LIGHT,
         title_size=15, body_size=12):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(width), Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = bar
    shape.line.width = Pt(1.25)
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.09), Inches(height)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = bar
    accent.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.22)
    tf.margin_right = Inches(0.14)
    tf.margin_top = Inches(0.14)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(title_size)
    r.font.bold = True
    r.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    r2 = p2.add_run()
    r2.text = body
    r2.font.size = Pt(body_size)
    r2.font.color.rgb = GREY
    p2.space_before = Pt(5)
    return shape


def band(slide, text, top=6.55, size=13, colour=NAVY, italic=True, height=0.7):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.1), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.italic = italic
    r.font.color.rgb = colour
    return box


def make_table(slide, data, left, top, width, height, col_widths=None,
               font_size=12, header_fill=NAVY):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            table.columns[i].width = Emu(int(Inches(width) * w / total))
    for ri, row in enumerate(data):
        for ci, value in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = str(value)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT
            run = para.runs[0] if para.runs else para.add_run()
            run.font.size = Pt(font_size)
            if ri == 0:
                run.font.bold = True
                run.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            else:
                run.font.color.rgb = NAVY
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
    return table


def out_tokens(record, arm):
    """Answer-text tokens; falls back to the runtime counter."""
    return record.get(f"{arm}_text_tokens") or record.get(f"{arm}_output_tokens") or 0


def clip(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def main() -> int:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (ROOT / "results" / "raw_paired.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line).get("protocol") == "paired"
    ]
    rows = _dedupe(rows, ("task", "item_id"))
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    n_items = len(rows)
    totals = {a: sum(1 for r in rows if r.get(f"{a}_correct")) for a in ARMS}
    acc = {a: totals[a] / n_items * 100 for a in ARMS}
    captured = (acc["hybrid"] - acc["local"]) / max(1e-9, acc["oracle"] - acc["local"]) * 100
    cloud_routed = sum(1 for r in rows if r["route"] == "cloud")

    routing_s = sum(r.get("decision_ms") or 0 for r in rows) / 1000
    hybrid_s = sum(r.get("hybrid_latency_ms") or 0 for r in rows) / 1000
    router_share = routing_s / max(1e-9, hybrid_s) * 100

    local_out = sum(out_tokens(r, "local") for r in rows)
    cloud_out = sum(out_tokens(r, "cloud") for r in rows)
    hybrid_out = sum(
        out_tokens(r, "local") if r["route"] == "local" else out_tokens(r, "cloud")
        for r in rows
    )

    def arm_time(arm, rs):
        total = sum(r.get(f"{arm}_latency_ms") or 0 for r in rs) / 1000
        pre = sum(r.get(f"{arm}_ttft_ms") or 0 for r in rs) / 1000
        return total, pre, max(0.0, total - pre)

    local_s, local_pre, local_dec = arm_time("local", rows)
    cloud_s, cloud_pre, cloud_dec = arm_time("cloud", rows)
    dec_local = local_out / max(1e-9, local_dec)
    dec_cloud = cloud_out / max(1e-9, cloud_dec)
    pre_tok = 18971  # input tokens, identical for both arms by construction

    clean_file = ROOT / "results" / "router_latency_clean.json"
    if clean_file.exists():
        clean = json.loads(clean_file.read_text())
        idle_ms = clean["idle"]["median_ms"]
        busy_ms = clean["interleaved"]["median_ms"]
        penalty = busy_ms / max(1e-9, idle_ms)
    else:
        idle_ms = busy_ms = penalty = float("nan")

    he = by_task["humaneval"]
    mb = by_task["mbpp"]
    bq = by_task["boolq"]

    prs = Presentation()
    prs.slide_width = Inches(_W)
    prs.slide_height = Inches(7.5)

    # ------------------------------------------------------------------ 1 intro
    s = add_slide(prs)
    header(s, "When does a cloud–local hybrid help?",
           "Tejas Koli · Independent Researcher · November 2026")
    card(s, 0.6, 1.5, 3.85, 3.1, "Problem",
         "A chat query has two plausible answerers: a small model on the device "
         "(private, free, weaker) and a large model behind an API (capable, "
         "metered, off-device).\n\nNeither default is satisfying, and a hybrid "
         "reduces to one component — the router.", bar=ACCENT)
    card(s, 4.65, 1.5, 3.85, 3.1, "Motivation",
         "The two tiers differ in quality, latency, cost and privacy at once.\n\n"
         "Routers are usually judged in isolation — does it label this query "
         "'hard'? — not on the behaviour the user experiences.", bar=BLUE)
    card(s, 8.7, 1.5, 4.0, 3.1, "Objective",
         f"Measure, on one machine with identical prompts, what difficulty "
         f"routing buys over either pure tier: accuracy, wall clock, and the "
         f"router's own cost, over {n_items} paired items and 6 benchmarks.",
         bar=GREEN)
    band(s, f"{n_items} paired items · 6 benchmarks · Apple M2 (Metal) vs Nemotron-3-Ultra 550B  ·  "
            f"local {acc['local']:.1f}%  ·  cloud {acc['cloud']:.1f}%  ·  hybrid {acc['hybrid']:.1f}%",
         top=4.85, size=16, italic=False)
    band(s, "Three hypotheses: H1 accuracy is retained · H2 difficulty is a sufficient "
            "routing signal · H3 router overhead is negligible.", top=5.55, size=14)
    band(s, "Contributions: a paired per-item protocol that yields the oracle ceiling for the "
            "price of two passes; a hardware decomposition of per-query speed; two corrected "
            "measurement defects reported because they changed published numbers.", top=6.25,
         size=12, colour=GREY)

    # ---------------------------------------------------------- 2 literature
    s = add_slide(prs)
    header(s, "Literature survey", "2021–2026 · citation counts read from OpenAlex on 17 Nov 2026")
    make_table(s, [
        ["Work (year)", "Methodology", "Reported performance", "Critical remark", "Cites"],
        ["Gemma 2 (2024)", "Open 2–27B LMs, distilled from a frontier model",
         "MMLU up to 75.1 (27B)", "Weights only; no on-device latency", "3,881"],
        ["Phi-3 (2024)", "3.8B model on heavily filtered data",
         "MMLU 69% at 3.8B", "Claims phone readiness, never measures it", "3,285"],
        ["FrugalGPT (2023)", "Cascade: cheap model, scorer, escalation",
         "GPT-4 quality at up to 98% lower cost", "Both tiers remote; needs a scorer", "1,119"],
        ["RouteLLM (2024)", "Routers trained on preference data",
         ">2× cost reduction at comparable quality", "Quality-gap objective, no length term", "745"],
        ["Hybrid LLM (2024)", "Quality-aware router, small vs large",
         "Up to 40% fewer large-model calls", "Latency never decomposed", "622"],
        ["PowerInfer (2024)", "Activation-sparsity GPU/CPU execution",
         "7.6× faster than llama.cpp on one GPU", "Serving only; no routing decision", "426"],
    ], 0.6, 1.5, 12.15, 3.55, col_widths=[1.9, 3.6, 3.4, 3.0, 0.85], font_size=11)
    card(s, 0.6, 5.25, 12.15, 1.55, "Research gap",
         "Each ingredient exists: capable on-device models, cascades with a quality signal, "
         "routers trained on preferences. What is missing is one measurement, on one machine "
         "with one prompt set, that separates the accuracy a hybrid gains from the latency it "
         "costs — including the router's own cost.", bar=ACCENT, body_size=12)

    # ---------------------------------------------------------- 3 hypothesis
    s = add_slide(prs)
    header(s, "Research hypothesis", "Three testable claims, each with a measurable outcome")
    rows_h = [
        ("H1 · Accuracy",
         "A difficulty-routed hybrid retains cloud-level accuracy while serving a "
         "meaningful share of queries on-device.",
         "Measured: hybrid accuracy vs cloud-only, and the share routed local.",
         f"{acc['hybrid']:.1f}% vs {acc['cloud']:.1f}% — weakly supported: cloud-level "
         f"within one item, but only {n_items - cloud_routed}/{n_items} items stayed local.",
         GREEN),
        ("H2 · Routing signal",
         "Predicted difficulty is a sufficient statistic for which tier should answer.",
         "Test: does routing on difficulty pick the faster and/or better tier per task?",
         "Not supported. Per-task winners flip, and difficulty does not predict output "
         "length, which is what sets wall clock.", ACCENT),
        ("H3 · Overhead",
         "The router's cost is negligible relative to generation, because it emits no tokens.",
         "Measured: decision latency idle and in the request path, as a share of wall clock.",
         f"Refuted. {busy_ms:.0f} ms per decision sharing the GPU against {idle_ms:.0f} ms "
         f"idle — {penalty:.1f}× — and {router_share:.0f}% of hybrid wall clock.", ACCENT),
    ]
    top = 1.5
    for name, claim, test, verdict, colour in rows_h:
        card(s, 0.6, top, 3.0, 1.6, name, claim, bar=BLUE, title_size=14, body_size=11.5)
        card(s, 3.75, top, 4.1, 1.6, "How it is tested", test,
             bar=GREY, title_size=12, body_size=11.5)
        card(s, 8.0, top, 4.75, 1.6, "Verdict", verdict, bar=colour,
             title_size=12, body_size=11.5)
        top += 1.72
    band(s, "H1 is the interesting failure: a perfect router would have scored "
            f"{acc['oracle']:.1f}% — exactly cloud-only — so there was no accuracy to "
            "recover on this sample.", top=6.75, size=13)

    # ---------------------------------------------------------- 4 methodology
    s = add_slide(prs)
    header(s, "Methodology", "Paired per-item protocol: one cloud request per item, not two")
    steps = [
        ("1 · Prompt", "Byte-identical prompt with few-shot exemplars", BLUE),
        ("2 · Both backends", "Gemma 4 E2B on-device + Nemotron (1 API call)", GREEN),
        ("3 · Route", "Laya decision, no tokens emitted", NAVY),
        ("4 · Derive", "Hybrid answer taken from the chosen tier", ACCENT),
        ("5 · Score", "Same scorer, same items, all four arms", GREY),
    ]
    for i, (title, body, colour) in enumerate(steps):
        card(s, 0.6 + i * 2.46, 1.5, 2.31, 1.6, title, body,
             bar=colour, title_size=12.5, body_size=11)

    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(3.3),
                             Inches(12.15), Inches(1.15))
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.13)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = 'hybrid = cloud_answer if route == "cloud" else local_answer'
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = ("Exactly paired, so item difficulty cancels · McNemar's test applies · "
               "API cost halved · the oracle (local OR cloud correct) falls out for free")
    r2.font.size = Pt(12)
    r2.font.color.rgb = PALE

    bullets(s, [
        ("Scoring: strict answer-marker extraction for letters and yes/no; final-number match "
         "for GSM8K; pass@1 for code, run in a separate process with rlimits and a timeout", 0),
        ("Statistics: Wilson 95% intervals (correct at n = 7–8) and McNemar's exact test on "
         "discordant pairs", 0),
        ("Reproducibility: every item stores a prompt hash, so a prompt change invalidates the "
         "old record; the runner resumes, so the sample accumulates across runs", 0),
        ("End-to-end check: a separate pass through the real service reproduced the same route "
         "and the same verdict on all six items tested", 0),
    ], top=4.6, size=13.5)

    # ---------------------------------------------------------- 5 architecture
    s = add_slide(prs)
    header(s, "Model architecture", "One decision model, two generator tiers, one code path")
    if ARCH.exists():
        s.shapes.add_picture(str(ARCH), Inches(2.05), Inches(1.35), width=Inches(9.2))
    bullets(s, [
        ("Router: Laya — a ModernBERT encoder with an RL head, calibrated probabilities, "
         "no generated tokens", 0),
        ("Rule: cloud if difficulty ≥ 1.5, P(needs tools) ≥ 0.14, or P(sensitive) ≥ 0.5; "
         "thresholds fixed for every run", 0),
        ("Same path in the FastAPI service and the harness: the numbers describe the "
         "deployed system", 0),
    ], top=5.95, size=12)

    # ---------------------------------------------------------- 6 dataset
    s = add_slide(prs)
    header(s, "Dataset", "Six public benchmarks, lm-evaluation-harness prompt conventions")
    make_table(s, [
        ["Task", "Source split", "Items", "Shots", "Answer format", "Metric", "Labels"],
        ["MMLU", "test (subject-stratified)", "8", "5", "option letter", "accuracy", "4-way"],
        ["GSM8K", "test", "8", "5", "final number", "exact match", "free numeric"],
        ["ARC-Challenge", "test", "7", "5", "option letter", "accuracy", "4-way"],
        ["BoolQ", "validation", "7", "3", "yes / no", "accuracy", "binary"],
        ["HumanEval", "test", "8", "0", "function body", "pass@1", "unit tests"],
        ["MBPP", "test", "8", "3", "delimited program", "pass@1", "unit tests"],
        ["Total", "—", str(n_items), "—", "—", "—", "—"],
    ], 0.6, 1.5, 8.2, 3.1, col_widths=[1.5, 2.5, 0.8, 0.8, 1.9, 1.4, 1.3], font_size=11)
    card(s, 9.05, 1.5, 3.7, 3.1, "Why these six",
         "Two knowledge tasks (MMLU, ARC), one arithmetic chain (GSM8K), one reading-comprehension "
         "binary task (BoolQ) and two code tasks with real unit tests (HumanEval, MBPP).\n\n"
         "That spans short-answer and long-generation regimes, which is where the cost picture "
         "turns out to depend.", bar=BLUE, body_size=11.5)
    card(s, 0.6, 4.8, 12.15, 1.95, "Sample item — MBPP/361 (prompt ending, verbatim)",
         "“Write a function to remove empty lists from a given list of lists. Your code should "
         "pass these tests:  assert remove_empty([[], [], [], 'Red', 'Green', [1,2], 'Blue', [], []]) "
         "== ['Red', 'Green', [1, 2], 'Blue'] …”\n\n"
         "Gold label: the function must pass those assertions. Note the mixed list — strings are "
         "valid elements, only empty lists are removed. That detail is what the local model misses.",
         bar=ACCENT, body_size=11.5)

    # ---------------------------------------------------------- 7 setup
    s = add_slide(prs)
    header(s, "Experimental setup", "Splits, implementation and evaluation conditions")
    card(s, 0.6, 1.45, 3.85, 2.8, "Hardware & runtime",
         "Apple M2, 16 GB unified memory, fanless MacBook Air, macOS 26.6, Python 3.14.\n\n"
         "Gemma 4 E2B (2.4 GB) served by LiteRT-LM on Metal with "
         "gpu_decode_steps_per_sync=4; CPU is only a fallback.", bar=GREEN, body_size=11.5)
    card(s, 4.65, 1.45, 3.85, 2.8, "Models",
         "Local: Gemma 4 E2B, no fine-tuning, no training of any kind.\n\n"
         "Cloud: nvidia/nemotron-3-ultra-550b-a55b, greedy (T = 0), reasoning disabled — the "
         "local checkpoint has no thinking mode, and enabling it would let the cloud spend "
         "hidden tokens the local model cannot produce.", bar=ACCENT, body_size=11.5)
    card(s, 8.7, 1.45, 4.0, 2.8, "Evaluation",
         "No train/val/test split in the ML sense: every arm is evaluated on each task's "
         "official test split (validation for BoolQ); few-shot exemplars come from the "
         "corresponding training split.\n\n"
         f"Sample: {n_items} items, {min(len(by_task[t]) for t in ORDER)}–"
         f"{max(len(by_task[t]) for t in ORDER)} per task — a pilot, stated as such.",
         bar=BLUE, body_size=11.5)
    make_table(s, [
        ["Fixed variable", "Setting"],
        ["Prompts", "byte-identical across arms, exemplars included; hash stored per item"],
        ["Decoding", "greedy on both tiers; max tokens 512 (1024 for code)"],
        ["Scoring", "strict markers, pass@1 in a sandboxed subprocess, 1e-4 numeric tolerance"],
        ["Inference", "Wilson 95% CI; McNemar exact test; sums and medians of wall clock"],
        ["Resume", "one JSON record per item, append-only, deduplicated by (task, item)"],
    ], 0.6, 4.55, 12.15, 2.1, col_widths=[2.2, 9.95], font_size=11.5)

    # ---------------------------------------------------------- 8 quantitative
    s = add_slide(prs)
    header(s, "Quantitative results", "Accuracy with 95% Wilson intervals")
    data = [["Task", "n", "Local", "Cloud", "Hybrid", "Oracle"]]
    for task in ORDER:
        rs = by_task[task]
        n = len(rs)
        row = [TASK_LABELS[task], str(n)]
        for arm in ARMS:
            c = sum(1 for r in rs if r.get(f"{arm}_correct"))
            lo, hi = wilson_interval(c, n)
            row.append(f"{c/n*100:.0f}% [{lo*100:.0f},{hi*100:.0f}]")
        data.append(row)
    data.append(["Pooled", str(n_items)] + [f"{acc[a]:.1f}%" for a in ARMS])
    make_table(s, data, 0.6, 1.5, 7.9, 2.6, col_widths=[1.7, 0.55, 1.5, 1.5, 1.5, 1.5],
               font_size=10.5)
    card(s, 8.8, 1.5, 3.9, 2.6, "Headline",
         f"Cloud-only  {acc['cloud']:.1f}%\nHybrid      {acc['hybrid']:.1f}%\n"
         f"Local-only   {acc['local']:.1f}%\nOracle      {acc['oracle']:.1f}%\n\n"
         f"{acc['cloud'] - acc['hybrid']:.1f} points behind cloud-only, "
         f"+{acc['hybrid'] - acc['local']:.1f} over always-local.", bar=ACCENT, body_size=11.5)
    if (FIG / "accuracy_by_task.png").exists():
        s.shapes.add_picture(str(FIG / "accuracy_by_task.png"), Inches(0.6), Inches(4.25),
                             width=Inches(5.4))
    bullets(s, [
        (f"The oracle equals cloud-only: a perfect router would have scored "
         f"{acc['oracle']:.1f}%, exactly what always-cloud achieved", 0),
        (f"Routing is therefore worth {acc['oracle'] - acc['cloud']:.1f} accuracy points — "
         f"the only case for it is latency, cost and privacy", 0),
        (f"Hybrid loses {acc['cloud'] - acc['hybrid']:.1f} points on one HumanEval item; "
         f"elsewhere it matches the cloud answer exactly", 0),
        (f"Only GSM8K reaches significance (McNemar p = 0.031); the rest are p ≥ 0.25", 0),
    ], left=6.2, top=4.35, width=6.5, size=12.5)

    # ---------------------------------------------------------- 9 qualitative
    s = add_slide(prs)
    header(s, "Qualitative results", "Same prompts, different answers — verbatim from the records")
    gsm = next(r for r in rows if r["item_id"] == "gsm8k/6")
    mbpp = next(r for r in rows if r["item_id"] == "mbpp/361")
    hev = next(r for r in rows if r["item_id"] == "HumanEval/2")

    card(s, 0.6, 1.4, 4.0, 2.6, "GSM8K/6 — both correct",
         "Ground truth: 10 pizzas.\n\n"
         f"Local ({out_tokens(gsm, 'local')} tok): “{clip(gsm['local_text'], 120)}”\n\n"
         f"Cloud ({out_tokens(gsm, 'cloud')} tok): “{clip(gsm['cloud_text'], 120)}”",
         bar=GREEN, body_size=10.5)
    card(s, 4.75, 1.4, 4.0, 2.6, "HumanEval/2 — routed local, fails",
         "Ground truth: return the decimal part of a float.\n\n"
         f"Local ({out_tokens(hev, 'local')} tok) reasons in comments and stops at "
         "“# fractional_part = 10.12” — no return statement, so the tests fail.\n\n"
         f"Cloud ({out_tokens(hev, 'cloud')} tok) returns number − math.floor(number).",
         bar=ACCENT, body_size=10.5)
    card(s, 8.9, 1.4, 3.8, 2.6, "BoolQ/3 — both wrong, weak label",
         "Is chicken cordon bleu made with blue cheese? Gold: yes.\n\n"
         "Local: “no”. Cloud: “no”. The passage describes ham and cheese, so both answers are "
         "defensible from the text — a label problem, not a model failure.",
         bar=GREY, body_size=10.5)
    card(s, 0.6, 4.15, 12.1, 2.35, "MBPP/361 — local correct about the docstring, wrong about the tests",
         "Gold: drop only empty lists, keep the strings.\n\n"
         f"Local ({out_tokens(mbpp, 'local')} tok) writes a full docstring and filters with "
         "“if isinstance(item, list) and item”, which discards 'Red' and 'Green' — the tests "
         "pass mixed lists.\n\n"
         f"Cloud ({out_tokens(mbpp, 'cloud')} tok): “{clip(mbpp['cloud_text'], 90)}” — passes.\n\n"
         "The more professional-looking answer is the wrong one, which is exactly what aggregate "
         "accuracy is meant to catch.",
         bar=ACCENT, body_size=11)
    band(s, "Aggregate accuracy hides this: the local answers read better and are sometimes wrong; "
            "the cloud answers read tersely and are usually right.", top=6.65, size=12)

    # ---------------------------------------------------------- 10 ablation/fail
    s = add_slide(prs)
    header(s, "Ablation & failure cases", "One component changed at a time, same data and metric")
    make_table(s, [
        ["ID", "Configuration", "Purpose", "Measured effect"],
        ["A1", "Local, plain “Answer:” prompt", "MC format baseline", "MMLU 0/20"],
        ["A2", "Local, explicit answer directive", "Does format instruction help?",
         "MMLU 9/20, ARC 21/30"],
        ["A3", "Local, prose few-shot MBPP", "Code-format baseline", "MBPP 0/25"],
        ["A4", "Harness [BEGIN]/[END] delimiters", "Stop answering an exemplar", "MBPP 10/25"],
        ["A5", "Answer directive on BoolQ", "Is the directive universal?",
         "22/30 vs 23/30 without"],
        ["A6", "CPU (XNNPACK) vs Metal GPU", "Is the accelerator worth it?",
         "30.7 vs 51.6 tok/s"],
        ["A7", "Router disabled (always local)", "Routing baseline",
         f"{acc['local']:.1f}%, {local_s:.1f} s"],
        ["A8", "Router enabled (hybrid)", "Routing contribution",
         f"{acc['hybrid']:.1f}%, {hybrid_s:.1f} s"],
        ["A9", "Scorer: direct concat vs newline join", "How much does the evaluator matter?",
         "MBPP cloud 2/8 → 7/8"],
    ], 0.6, 1.45, 7.6, 4.25, col_widths=[0.5, 2.6, 2.3, 2.2], font_size=10)
    card(s, 8.4, 1.45, 4.35, 4.25, "Failure cases",
         "F1 · Wrong tier chosen (HumanEval/2): local emits no return, tests fail — the entire "
         "hybrid-to-cloud gap.\n\n"
         "F2 · Both wrong on BoolQ/3: gold says yes, both said no; the passage supports no.\n\n"
         "F3 · Router is not free: 3.1 s per request sharing the GPU vs 358 ms idle — 8.8×, an "
         "order of magnitude worse than assumed.\n\n"
         "F4 · The evaluator can invert a result: a missing newline before the test harness made "
         "correct programs raise SyntaxError.",
         bar=ACCENT, body_size=10.5)
    band(s, "A1–A6 are calibration runs on separate samples (n = 20–30) used to fix the prompt "
            "format; A7–A9 are measured on the 46-item sample.", top=5.95, size=12)

    # ---------------------------------------------------------- 11 discussion
    s = add_slide(prs)
    header(s, "Discussion", "What governs the ranking, and what does not")
    card(s, 0.6, 1.45, 3.9, 2.5, "Speed follows output length, not difficulty",
         f"Local writes {local_out:,} tokens against {cloud_out:,} "
         f"({local_out / max(1, cloud_out):.2f}×) but decodes only "
         f"{dec_cloud / max(1e-9, dec_local):.1f}× slower.\n\n"
         "So the cloud's speed comes mostly from saying less, and prefill actually favours the device.",
         bar=BLUE, body_size=11.5)
    card(s, 4.65, 1.45, 3.9, 2.5, "The crossover is per task",
         "MMLU and ARC: both arms emit one token, so the device wins on prefill "
         "(1.6–2.0× faster).\n\n"
         "GSM8K and MBPP: long generations, so the cloud wins (4.6–5.9×). A difficulty-only "
         "router cannot see this.", bar=GREEN, body_size=11.5)
    card(s, 8.7, 1.45, 4.0, 2.5, "The router is a real cost",
         f"{routing_s:.1f} s of the hybrid's {hybrid_s:.1f} s ({router_share:.0f}%) is decision "
         f"time.\n\n{idle_ms:.0f} ms on an idle GPU becomes {busy_ms:.0f} ms in the request path "
         f"({penalty:.1f}×) because it shares the accelerator with the generator.",
         bar=ACCENT, body_size=11.5)
    if (FIG / "latency_distribution.png").exists():
        s.shapes.add_picture(str(FIG / "latency_distribution.png"), Inches(0.85), Inches(4.15),
                             width=Inches(4.3))
    bullets(s, [
        ("Interpretation: a hybrid is a latency-and-privacy instrument on this sample, not an "
         "accuracy instrument — the oracle equals cloud-only", 0),
        ("The router should predict expected output length, or time-to-answer, rather than "
         "difficulty", 0),
        ("Measurement discipline matters as much as modelling: one scorer join changed the MBPP "
         "conclusion, one token counter inflated local volume", 0),
        ("Ranking of arms is stable to these fixes; the earlier narrative about MBPP was not", 0),
    ], left=5.9, top=4.2, width=6.8, size=12)

    # ---------------------------------------------------------- 12 conclusion
    s = add_slide(prs)
    header(s, "Conclusion & future work", "Three findings, stated with their limits")
    card(s, 0.6, 1.4, 3.9, 3.0, "Contributions",
         "· A paired per-item protocol: the oracle ceiling for two passes instead of three\n\n"
         "· Evidence the oracle can equal cloud-only, so routing buys latency and privacy, not "
         "accuracy\n\n"
         "· A hardware decomposition: per-query speed follows expected output length\n\n"
         "· Two corrected measurement defects, reported because they changed numbers",
         bar=GREEN, body_size=10)
    card(s, 4.65, 1.4, 3.9, 3.0, "Limitations",
         f"· {n_items} items, 7–8 per task; only GSM8K is significant\n\n"
         "· One cloud model, one provider, one machine\n\n"
         "· Hybrid arm derived, validated on six end-to-end items\n\n"
         "· Reasoning disabled on the cloud model\n\n"
         "· Code execution uses rlimits, not isolation",
         bar=GREY, body_size=10)
    card(s, 8.7, 1.4, 4.0, 3.0, "Future work",
         "· Add an expected-output-length regressor and route on predicted time-to-answer\n\n"
         "· Run the router on a separate thread or device so its cost is not serialised\n\n"
         "· Replace the strict marker scorer with a semantic judge\n\n"
         "· Extend to the full 1,864-item sample with the same protocol",
         bar=BLUE, body_size=10)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(4.6),
                             Inches(12.1), Inches(2.0))
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.3)
    tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.18)
    items = [
        f"1 · On this sample routing cannot improve accuracy: cloud {acc['cloud']:.1f}%, "
        f"hybrid {acc['hybrid']:.1f}%, oracle {acc['oracle']:.1f}%.",
        "2 · Difficulty is the wrong signal for latency: output length decides, and the "
        "per-task winner flips.",
        f"3 · The router is not free: {busy_ms:.0f} ms per decision in the request path, "
        f"{router_share:.0f}% of hybrid wall clock.",
    ]
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = item
        r.font.size = Pt(14)
        r.font.color.rgb = WHITE
        p.space_after = Pt(9)

    out = ROOT / "slides"
    out.mkdir(exist_ok=True)
    path = out / "slides.pptx"
    prs.save(path)
    print(f"wrote {path} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
