"""Draw the system architecture diagram used by the report and deck.

Rendered with matplotlib so it stays reproducible and needs no external
diagram tool.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures" / "architecture.png"

NAVY = "#1F3355"
BLUE = "#2E5C8A"
GREEN = "#2E7D5B"
ORANGE = "#D1652A"
GREY = "#5B6470"
LIGHT = "#F3F6FA"


def box(ax, x, y, w, h, title, body="", edge=BLUE, fill=LIGHT, title_size=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 0.075, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=NAVY, zorder=3)
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.045, body, ha="center", va="center",
                fontsize=8.4, color=GREY, zorder=3, linespacing=1.45)


def arrow(ax, p0, p1, colour=GREY, label=None, lx=0.0, ly=0.0):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                        linewidth=1.2, color=colour, zorder=1,
                        shrinkA=2, shrinkB=2)
    )
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label,
                ha="center", va="center", fontsize=7.8, color=colour,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.92), zorder=4)


def main() -> int:
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.028, 0.40, 0.125, 0.20, "Query", "one user\nmessage", edge=GREY)
    box(ax, 0.175, 0.315, 0.185, 0.37, "Laya router",
        "ModernBERT encoder\n+ RL head\nno generated tokens\n\n"
        "difficulty  P(tools)  sensitive", edge=NAVY)
    box(ax, 0.475, 0.60, 0.30, 0.24, "Local tier",
        "Gemma 4 E2B  ·  2.4 GB\nLiteRT-LM on Metal GPU\n"
        "no network, no spend", edge=GREEN)
    box(ax, 0.475, 0.145, 0.30, 0.24, "Cloud tier",
        "Nemotron-3-Ultra 550B\nNVIDIA integrate API\n"
        "greedy, reasoning off", edge=ORANGE)
    box(ax, 0.825, 0.35, 0.16, 0.30, "Response",
        "answer + route\n+ timings", edge=GREY)

    arrow(ax, (0.155, 0.50), (0.173, 0.50))
    arrow(ax, (0.362, 0.62), (0.472, 0.70), GREEN)
    ax.text(0.417, 0.685, "local", ha="center", va="bottom", fontsize=8.0, color=GREEN)
    arrow(ax, (0.362, 0.38), (0.472, 0.28), ORANGE)
    ax.text(0.417, 0.352, "cloud", ha="center", va="bottom", fontsize=8.0, color=ORANGE)
    arrow(ax, (0.777, 0.70), (0.822, 0.55), GREY)
    arrow(ax, (0.777, 0.26), (0.822, 0.45), GREY)

    ax.text(0.5, 0.045,
            "Thresholds: cloud if difficulty $\\geq$ 1.5,  P(tools) $\\geq$ 0.14, "
            "or the query is sensitive.  The hybrid arm reuses the same two answers.",
            ha="center", fontsize=8.2, color=GREY, style="italic")
    ax.text(0.255, 0.145, "router cost: 358 ms idle  ·  2.6 s sharing the GPU",
            ha="center", va="top", fontsize=7.8, color=NAVY)

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
