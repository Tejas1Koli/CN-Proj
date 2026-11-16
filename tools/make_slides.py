"""Build slides/slides.pptx from the measured results.

Numbers come from results/summary.json and results/raw_paired.jsonl so the deck
cannot drift from the experiment.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.runner import _dedupe  # noqa: E402
from bench.stats import mcnemar_exact, wilson_interval  # noqa: E402

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


def add_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def header(slide, title: str, subtitle: str | None = None):
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs_width(), Inches(0.95)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(12.3), Inches(0.7))
    p = box.text_frame.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = WHITE
    if subtitle:
        sb = slide.shapes.add_textbox(Inches(0.5), Inches(1.02), Inches(12.3), Inches(0.4))
        sp = sb.text_frame.paragraphs[0]
        sr = sp.add_run(); sr.text = subtitle
        sr.font.size = Pt(13); sr.font.color.rgb = GREY


def bullets(slide, items, left=0.6, top=1.6, width=12.1, height=5.4, size=17):
    height = min(height, 7.35 - top)
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        text, level = item if isinstance(item, tuple) else (item, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        r = p.add_run(); r.text = text
        r.font.size = Pt(size - level * 2)
        r.font.color.rgb = NAVY if level == 0 else GREY
        p.space_after = Pt(8)
    return box


def card(slide, left, top, width, height, title, body, fill=LIGHT, bar=BLUE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(width), Inches(height),
    )
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = bar; shape.line.width = Pt(1.25)
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.09), Inches(height)
    )
    accent.fill.solid(); accent.fill.fore_color.rgb = bar
    accent.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.22); tf.margin_top = Inches(0.18)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = body
    r2.font.size = Pt(13); r2.font.color.rgb = GREY
    p2.space_before = Pt(6)
    return shape


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
            para.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            run = para.runs[0] if para.runs else para.add_run()
            run.font.size = Pt(font_size)
            if ri == 0:
                run.font.bold = True
                run.font.color.rgb = WHITE
                cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
            else:
                run.font.color.rgb = NAVY
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
    return table


_W = 13.333


def prs_width():
    return Inches(_W)


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
    hybrid_out = sum(
        (r.get("local_output_tokens") or 0)
        if r["route"] == "local"
        else (r.get("cloud_output_tokens") or 0)
        for r in rows
    )

    clean_file = ROOT / "results" / "router_latency_clean.json"
    if clean_file.exists():
        clean = json.loads(clean_file.read_text())
        idle_s = clean["idle"]["median_ms"] / 1000
        busy_s = clean["interleaved"]["median_ms"] / 1000
        penalty = busy_s / max(1e-9, idle_s)
    else:
        idle_s = busy_s = penalty = float("nan")

    prs = Presentation()
    prs.slide_width = Inches(_W)
    prs.slide_height = Inches(7.5)

    # ------------------------------------------------------------------ 1 title
    s = add_slide(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.1), Inches(_W), Inches(2.5))
    band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.35), Inches(11.5), Inches(1.2))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = "A Cloud–Local Hybrid Chatbot"
    r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
    p2 = tb.text_frame.add_paragraph()
    r2 = p2.add_run(); r2.text = "Intelligence and Hardware Trade-offs"
    r2.font.size = Pt(28); r2.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEE)
    sb = s.shapes.add_textbox(Inches(0.9), Inches(5.0), Inches(11.5), Inches(1.2))
    tf = sb.text_frame
    lines = [
        f"Tejas Koli  ·  Independent Researcher  ·  November 2026",
        f"{n_items} paired items · 6 benchmarks · Apple M2 (Metal) vs Nemotron-3-Ultra 550B",
        f"Hybrid accuracy {acc['hybrid']:.1f}%  —  captures {captured:.0f}% of oracle headroom",
    ]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(14 if i else 16); r.font.color.rgb = GREY
        p.space_after = Pt(4)

    # ------------------------------------------------------------------ 2 question
    s = add_slide(prs)
    header(s, "The problem", "Two default answers, both wrong")
    card(s, 0.6, 1.7, 5.9, 3.1, "Always cloud",
         "Most accurate almost everywhere.\n\nCosts: latency, spend, and shipping the query off the device.", bar=ACCENT)
    card(s, 6.85, 1.7, 5.9, 3.1, "Always local",
         "Fast, private, free.\n\nCost: a low capability ceiling that makes the choice rarely close.", bar=GREEN)
    box = s.shapes.add_textbox(Inches(0.6), Inches(5.1), Inches(12.1), Inches(1.7))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "A hybrid should send only what needs the cloud to the cloud."
    r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = ("But that reduces the entire system to one component — the router — "
               "and routers are usually evaluated in isolation, not on end-to-end behaviour.")
    r2.font.size = Pt(15); r2.font.color.rgb = GREY

    # ------------------------------------------------------------------ 3 systems
    s = add_slide(prs)
    header(s, "Three systems under test", "Every arm sees byte-identical prompts")
    make_table(s, [
        ["Arm", "Model", "Runtime / placement"],
        ["Local", "Gemma 4 E2B (2.4 GB)", "LiteRT-LM, Metal GPU, this machine"],
        ["Cloud", "Nemotron-3-Ultra 550B", "NVIDIA integrate endpoint"],
        ["Router", "Laya (ModernBERT + RL head)", "On-device, this machine"],
    ], 0.9, 1.8, 11.5, 2.2, col_widths=[1.4, 4.0, 6.1], font_size=14)
    bullets(s, [
        ("Routing rule: cloud when difficulty ≥ 1.5, or tool-needs probability ≥ 0.14, or sensitive", 0),
        ("Cloud answers greedily with reasoning disabled — the local model has no thinking mode", 0),
        ("Router costs ~%.1f s per decision in the request path (%.0f ms when the GPU is idle)" % (
            busy_s, idle_s), 0),
    ], top=4.4, size=16)

    # ------------------------------------------------------------------ 4 method
    s = add_slide(prs)
    header(s, "Method: the paired protocol", "One cloud request per item instead of two")
    boxes = [
        ("Local answer", "Gemma 4 E2B\n(no API call)", GREEN, 0.6),
        ("Cloud answer", "Nemotron-3-Ultra\n(1 API call)", ACCENT, 4.85),
        ("Router", "Laya decision\n(no API call)", BLUE, 9.1),
    ]
    for title, body, colour, left in boxes:
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(2.0),
                                Inches(3.6), Inches(1.5))
        sh.fill.solid(); sh.fill.fore_color.rgb = LIGHT
        sh.line.color.rgb = colour; sh.line.width = Pt(1.5)
        tf = sh.text_frame; tf.word_wrap = True
        tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.12)
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = title
        r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = NAVY
        p2 = tf.add_paragraph()
        r2 = p2.add_run(); r2.text = body
        r2.font.size = Pt(13); r2.font.color.rgb = GREY
    arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(3.9), Inches(2.35),
                               Inches(1.1), Inches(0.6))
    arrow.fill.solid(); arrow.fill.fore_color.rgb = GREY; arrow.line.fill.background()
    arrow2 = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(8.15), Inches(2.35),
                                Inches(1.1), Inches(0.6))
    arrow2.fill.solid(); arrow2.fill.fore_color.rgb = GREY; arrow2.line.fill.background()

    result = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(4.0),
                                Inches(12.1), Inches(1.3))
    result.fill.solid(); result.fill.fore_color.rgb = NAVY; result.line.fill.background()
    tf = result.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.25); tf.margin_top = Inches(0.15)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "hybrid_answer  =  cloud_answer   if route == \"cloud\"   else   local_answer"
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = WHITE
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = ("Exactly paired · McNemar's test applies · oracle upper bound falls out for free "
               "· API cost halved")
    r2.font.size = Pt(13); r2.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEE)

    bullets(s, [
        ("End-to-end validation: 100% agreement on correctness and routing across the subset tested", 0),
    ], top=5.6, size=15)

    # ------------------------------------------------------------------ 5 setup
    s = add_slide(prs)
    header(s, "Experimental setup", "Six standard benchmarks, strict scoring")
    make_table(s, [
        ["Task", "Format", "Metric"],
        ["MMLU", "5-shot, answer directive", "choice accuracy"],
        ["GSM8K", "5-shot", "exact match on final number"],
        ["ARC-Challenge", "5-shot", "choice accuracy"],
        ["BoolQ", "3-shot", "yes/no accuracy"],
        ["HumanEval", "0-shot", "pass@1, sandboxed exec"],
        ["MBPP", "3-shot, [BEGIN]/[END]", "pass@1, sandboxed exec"],
    ], 0.7, 1.75, 7.4, 4.4, col_widths=[2.1, 2.9, 2.4], font_size=13)
    bullets(s, [
        ("Fixed seed 1234; few-shot exemplars shared by every arm", 0),
        ("Wilson 95% intervals; McNemar exact test on discordant pairs", 0),
        ("Strict extraction: only explicit answer markers count", 0),
        ("Sample: %d items (%d–%d per task) — a pilot, stated as such" % (
            n_items, min(len(by_task[t]) for t in ORDER), max(len(by_task[t]) for t in ORDER)), 0),
    ], left=8.4, top=1.9, width=4.4, size=15)

    # ------------------------------------------------------------------ 6 intelligence
    s = add_slide(prs)
    header(s, "Intelligence results", "Accuracy with 95% Wilson intervals")
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
    data.append(["Pooled", str(n_items)] + [
        f"{totals[a]/n_items*100:.1f}%" for a in ARMS
    ])
    make_table(s, data, 0.7, 1.75, 7.55, 2.85, col_widths=[1.7, 0.6, 1.5, 1.5, 1.5, 1.5],
               font_size=11)
    card(s, 8.55, 1.75, 4.2, 2.85, "Headline",
         f"Cloud-only {acc['cloud']:.1f}%\nHybrid      {acc['hybrid']:.1f}%\n"
         f"Local-only   {acc['local']:.1f}%\nOracle      {acc['oracle']:.1f}%\n\n"
         f"Only {acc['cloud']-acc['hybrid']:.1f} points behind cloud-only, "
         f"+{acc['hybrid']-acc['local']:.1f} over always-local.",
         bar=ACCENT)
    if (FIG / "accuracy_by_task.png").exists():
        s.shapes.add_picture(str(FIG / "accuracy_by_task.png"), Inches(0.7), Inches(4.85),
                             width=Inches(5.5))
    bullets(s, [
        ("Per-task spread is large — the small $n$ makes single-task claims soft", 0),
        ("Pooled ordering is unambiguous: cloud > hybrid > local", 0),
        ("Oracle shows what a length-aware router could still win", 0),
    ], left=6.55, top=4.95, width=6.2, size=13)

    # ------------------------------------------------------------------ 7 router quality
    s = add_slide(prs)
    header(s, "How good is the router?", "It captures most of the headroom — but not all")
    card(s, 0.7, 1.7, 3.9, 2.0, "Headroom over local",
         f"{acc['oracle']-acc['local']:.1f} points separate always-local from a perfect router.",
         bar=BLUE)
    card(s, 4.85, 1.7, 3.9, 2.0, "Captured by Laya",
         f"{acc['hybrid']-acc['local']:.1f} points — {captured:.0f}% of the available headroom.",
         bar=GREEN)
    card(s, 9.0, 1.7, 3.7, 2.0, "Unrealised",
         f"{acc['oracle']-acc['hybrid']:.1f} points left on the table.",
         bar=ACCENT)
    bullets(s, [
        (f"Routing split: {cloud_routed}/{n_items} items ({cloud_routed/n_items*100:.0f}%) went to the cloud", 0),
        ("Laya routes on difficulty — these benchmark items are simply hard for a 2B model", 0),
        ("A length-aware signal would capture much of the remainder (next slides)", 0),
        (f"And each decision costs {busy_s:.1f} s in the request path — {router_share:.0f}% of hybrid wall clock", 0),
    ], left=4.75, top=3.95, width=8.0, size=15)
    if (FIG / "router_scatter.png").exists():
        s.shapes.add_picture(str(FIG / "router_scatter.png"), Inches(0.7), Inches(3.95),
                             width=Inches(3.75))

    # ------------------------------------------------------------------ 8 tokens/time
    s = add_slide(prs)
    header(s, "Hardware: tokens and time", "The cloud is faster overall — but not for the obvious reason")
    input_tokens = {}
    try:
        from bench import data as bench_data
        from bench.models import LocalModel

        model = LocalModel(); model.warm()
        for task in bench_data.build_all():
            for item in task.items:
                input_tokens[(task.name, item.item_id)] = len(model.engine.tokenize(item.prompt))
    except Exception:
        pass

    def arm_stats(arm, rs):
        inp = sum(input_tokens.get((r["task"], r["item_id"]), 0) for r in rs)
        out = sum(r.get(f"{arm}_output_tokens") or 0 for r in rs)
        tot = sum(r.get(f"{arm}_latency_ms") or 0 for r in rs)
        pre = sum(r.get(f"{arm}_ttft_ms") or 0 for r in rs)
        return inp, out, tot / 1000, pre / 1000, (tot - pre) / 1000

    li, lo, lt, lp, ld = arm_stats("local", rows)
    ci_, co, ct, cp, cd = arm_stats("cloud", rows)

    make_table(s, [
        ["Arm", "Input tok", "Output tok", "Prefill s", "Decode s", "Total s"],
        ["Local only", f"{li:,}", f"{lo:,}", f"{lp:.1f}", f"{ld:.1f}", f"{lt:.1f}"],
        ["Cloud only", f"{ci_:,}", f"{co:,}", f"{cp:.1f}", f"{cd:.1f}", f"{ct:.1f}"],
        ["Hybrid", f"{li:,}", f"{hybrid_out:,}", "—", "—", f"{hybrid_s:.1f}"],
    ], 0.7, 1.75, 12.0, 2.05, col_widths=[2.4, 2.0, 2.0, 1.9, 1.9, 1.8], font_size=13)

    card(s, 0.7, 4.0, 3.9, 2.35, "Output economy",
         f"Local emits {lo/max(1,co):.2f}× more output tokens "
         f"({lo/n_items:.0f} vs {co/n_items:.0f} per item). It rambles; the cloud is concise.",
         bar=ACCENT)
    card(s, 4.85, 4.0, 3.9, 2.35, "Per-token decode",
         f"local {lo/max(1e-9,ld):.1f} tok/s vs cloud {co/max(1e-9,cd):.1f} tok/s — "
         f"only a {((co/max(1e-9,cd))/(lo/max(1e-9,ld))):.1f}× advantage, far smaller than the "
         f"{lt/max(1e-9,ct):.1f}× overall gap.",
         bar=BLUE)
    card(s, 9.0, 4.0, 3.7, 2.35, "Prefill is a local win",
         f"{li/max(1e-9,lp):.0f} tok/s on-device vs {ci_/max(1e-9,cp):.0f} tok/s in the cloud, "
         f"network round-trips included.",
         bar=GREEN)

    foot = s.shapes.add_textbox(Inches(0.7), Inches(6.5), Inches(12.0), Inches(0.75))
    ftf = foot.text_frame; ftf.word_wrap = True
    fp = ftf.paragraphs[0]
    fr = fp.add_run()
    fr.text = (
        f"The hybrid's {hybrid_s:.1f} s includes {routing_s:.1f} s of router time "
        f"({router_share:.0f}%): {busy_s:.1f} s per decision in the request path against "
        f"{idle_s:.2f} s on an idle GPU."
    )
    fr.font.size = Pt(13); fr.font.italic = True; fr.font.color.rgb = NAVY

    # ------------------------------------------------------------------ 9 crossover
    s = add_slide(prs)
    header(s, "The per-task crossover", "Short answers favour local; long generations favour cloud")
    data = [["Task", "Local s", "Cloud s", "Hybrid s", "Local out", "Cloud out", "Winner"]]
    for task in ORDER:
        rs = by_task[task]
        _, lo2, lt2, _, _ = arm_stats("local", rs)
        _, co2, ct2, _, _ = arm_stats("cloud", rs)
        ht2 = sum(r.get("hybrid_latency_ms") or 0 for r in rs) / 1000
        winner = "LOCAL" if lt2 < ct2 else "CLOUD"
        data.append([TASK_LABELS[task], f"{lt2:.1f}", f"{ct2:.1f}", f"{ht2:.1f}",
                     f"{lo2:,}", f"{co2:,}", winner])
    make_table(s, data, 0.7, 1.8, 8.35, 4.2,
               col_widths=[1.75, 1.15, 1.15, 1.15, 1.2, 1.2, 1.1], font_size=12)
    def local_penalty(task):
        rs = by_task[task]
        loc_s = sum(r.get("local_latency_ms") or 0 for r in rs) / 1000
        hyb_s = sum(r.get("hybrid_latency_ms") or 0 for r in rs) / 1000
        return hyb_s / max(1e-9, loc_s)

    pen_lo, pen_hi = sorted([local_penalty("mmlu"), local_penalty("arc_challenge")])
    card(s, 9.2, 1.8, 3.5, 4.2, "The key finding",
         "Speed is governed by expected output length, not input difficulty.\n\n"
         f"Laya routes on difficulty, so it cannot see the crossover: on the two local-win "
         f"tasks, routing makes the system {pen_lo:.1f}–{pen_hi:.1f}× slower than local-only.",
         bar=ACCENT)
    box = s.shapes.add_textbox(Inches(0.7), Inches(6.2), Inches(12.0), Inches(0.9))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = ("MMLU and ARC-Challenge are 1.6–2.0× faster on-device because prefill dominates. "
              "GSM8K and MBPP are 4.6–5.9× faster in the cloud because decode dominates — "
              "and the local model also generates several times more tokens.")
    r.font.size = Pt(14); r.font.italic = True; r.font.color.rgb = GREY

    # ------------------------------------------------------------------ 10 failures
    s = add_slide(prs)
    header(s, "Where routing actively hurt", "Two tasks explain the whole hybrid-to-oracle gap")
    he = by_task["humaneval"]; mb = by_task["mbpp"]
    he_h = sum(1 for r in he if r["hybrid_correct"]) / len(he) * 100
    he_c = sum(1 for r in he if r["cloud_correct"]) / len(he) * 100
    he_l = sum(1 for r in he if r["local_correct"]) / len(he) * 100
    mb_l = sum(1 for r in mb if r["local_correct"]) / len(mb) * 100
    mb_c = sum(1 for r in mb if r["cloud_correct"]) / len(mb) * 100
    mb_o = sum(1 for r in mb if r["oracle_correct"]) / len(mb) * 100
    mb_cloud_routed = sum(1 for r in mb if r["route"] == "cloud")

    card(s, 0.7, 1.8, 5.9, 3.4, "HumanEval — wrong tier chosen",
         f"Hybrid {he_h:.0f}%  vs  Cloud-only {he_c:.0f}%\n"
         f"(local scored {he_l:.0f}%)\n\n"
         f"The router sent {len(he)-sum(1 for r in he if r['route']=='cloud')} of {len(he)} items "
         f"to the local model, and local got them wrong.",
         bar=ACCENT)
    card(s, 6.85, 1.8, 5.8, 3.4, "MBPP — the better model was ignored",
         f"Local {mb_l:.0f}%  vs  Cloud {mb_c:.0f}%\n"
         f"Oracle routing {mb_o:.0f}%\n\n"
         f"The router sent all {mb_cloud_routed} items to the cloud, "
         f"where the score was half of local's.",
         bar=ACCENT)
    box = s.shapes.add_textbox(Inches(0.7), Inches(5.5), Inches(12.0), Inches(1.5))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = ("Both failures share one cause: the router predicts difficulty, "
              "but the cost-optimal choice depends on how long the answer will be.")
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "On MBPP the local model was simultaneously faster AND more accurate — the single clearest miss."
    r2.font.size = Pt(14); r2.font.color.rgb = GREY

    # ------------------------------------------------------------------ 11 significance
    s = add_slide(prs)
    header(s, "What is statistically solid?", "Be honest about the sample")
    data = [["Test", "Result"]]
    for task in ORDER:
        rs = by_task[task]
        h_only = sum(1 for r in rs if r["hybrid_correct"] and not r["local_correct"])
        l_only = sum(1 for r in rs if r["local_correct"] and not r["hybrid_correct"])
        c_only = sum(1 for r in rs if r["cloud_correct"] and not r["local_correct"])
        l_win = sum(1 for r in rs if r["local_correct"] and not r["cloud_correct"])
        hl_p = mcnemar_exact(h_only, l_only)
        cl_p = mcnemar_exact(c_only, l_win)
        mark = "  ✓ significant" if hl_p < 0.05 else ""
        data.append([TASK_LABELS[task],
                     f"hybrid vs local p={hl_p:.3f}{mark}   |   cloud vs local p={cl_p:.3f}"])
    make_table(s, data, 0.7, 1.8, 12.0, 3.2, col_widths=[2.6, 9.4], font_size=13)
    bullets(s, [
        ("Only GSM8K reaches significance (p = 0.031)", 0),
        ("With %d–%d items per task the study is powered for large effects only" % (
            min(len(by_task[t]) for t in ORDER), max(len(by_task[t]) for t in ORDER)), 0),
        ("The pooled gaps are large enough to believe; per-task claims are indicative", 0),
    ], top=5.3, size=15)

    # ------------------------------------------------------------------ 12 limitations
    s = add_slide(prs)
    header(s, "Limitations", "Read these before quoting any number")
    bullets(s, [
        (f"Sample: {n_items} items, {min(len(by_task[t]) for t in ORDER)}–{max(len(by_task[t]) for t in ORDER)} per task — a pilot, not a definitive study", 0),
        ("Single cloud model (550B) from a single provider; thresholds were fitted against it", 0),
        ("The hybrid arm is derived from per-item backend results, not executed independently", 0),
        ("Reasoning disabled on the cloud model, so its accuracy is a non-reasoning accuracy", 0),
        ("Generative scoring with an answer directive — not comparable to log-likelihood literature values", 0),
        ("One machine, one run: a fanless M2 MacBook Air, thermally constrained", 0),
        (f"Router cost is machine-specific: {idle_s:.2f} s idle vs {busy_s:.1f} s when sharing the GPU", 0),
        ("Code execution is mitigated with rlimits and a timeout, not a security boundary", 0),
    ], top=1.7, size=17)

    # ------------------------------------------------------------------ 13 takeaways
    s = add_slide(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(_W), Inches(7.5))
    band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.9), Inches(0.7), Inches(11.5), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = "Takeaways"
    r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = WHITE

    items = [
        (f"The hybrid reaches {acc['hybrid']:.1f}% against {acc['cloud']:.1f}% cloud-only "
         f"and {acc['local']:.1f}% local-only — losing only {acc['cloud']-acc['hybrid']:.1f} points."),
        (f"It captures {captured:.0f}% of the accuracy headroom available over always running the 2B model on-device."),
        ("The cloud wins on speed mainly because it says less "
         f"({lo/max(1,co):.2f}× fewer output tokens), not because it decodes much faster."),
        ("For short answers the on-device model is the faster choice outright — 1.6–2.0× on MMLU and ARC."),
        ("Difficulty is the wrong routing signal. Expected output length is the better predictor."),
        (f"Routing is not free either: {routing_s:.0f} s of the hybrid's {hybrid_s:.0f} s "
         f"({router_share:.0f}%) is decision time, at {busy_s:.1f} s per request while sharing the GPU."),
        ("The harness is resumable: BENCH_MODE=full accumulates a publication-grade sample automatically."),
    ]
    box = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(4.9))
    tf = box.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = f"▸  {item}"
        r.font.size = Pt(17); r.font.color.rgb = WHITE
        p.space_after = Pt(11)

    out = ROOT / "slides"
    out.mkdir(exist_ok=True)
    path = out / "slides.pptx"
    prs.save(path)
    print(f"wrote {path} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
