"""Figures for Paper B that replace tables or add a picture the text lacked.

  fig_lead_crowding.png   the crowding reading of lead: a filing's own main
                          theme, as a share of its class's filings, in the five
                          years before and after it was filed, by lead fifth
  fig_ladder.png          the capital-markets ladder (funded / SEC reporting /
                          IPO) by fifth of lead and of atypicality
  fig_staged.png          every stage's lead and atypicality coefficient on one
                          scale: percent change in the stage's rate per SD
  fig_reversal.png        the 2000-2004 reversal, cell by cell

Reads paper/results/{lead_crowding,sec_event_ladder,staged_outcomes,
reversal_theme_decomp}.json. Colors: lead #c0392b, atypicality #2b6cb0 (the
papers' existing pair; CVD separation dE 20-31 in OKLab x100).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
LEAD, ATYP, INK, INK2, GRID = "#c0392b", "#2b6cb0", "#1a1a1a", "#555555", "#dddddd"
WEB, OTHER = "#2b6cb0", "#8a8f98"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def crowding() -> None:
    d = json.loads((RES / "lead_crowding.json").read_text())["by_lead_fifth"]
    q = np.arange(1, 6)
    g = np.array([r["share_growing"] for r in d]) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.bar(q, g, width=0.62, color=LEAD)
    ax.axhline(50, color=INK2, lw=1, ls=(0, (4, 3)))
    for x, v in zip(q, g):
        ax.text(x, v + 1.8, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(q, ["1\nmost lagging", "2", "3", "4", "5\nmost leading"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of registrations whose main theme\nwas more common afterwards")
    ax.set_xlabel("fifth of lead, within Nice class and registration year", color=INK2)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(RES / "fig_lead_crowding.png", dpi=200)
    plt.close(fig)


def ladder() -> None:
    d = json.loads((RES / "sec_event_ladder.json").read_text())
    n = d["n_debuts"]
    rungs = [("funded", "Raised a priced round"), ("reporting", "In SEC reporting"), ("ipo", "Listed (IPO marker)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    q = np.arange(1, 6)
    for ax, (k, title) in zip(axes, rungs):
        for var, col, lab, off in (("L_length_held", LEAD, "by fifth of lead", -0.08),
                                   ("A_length_held", ATYP, "by fifth of atypicality", 0.08)):
            p = np.array(d["contrasts"][k][var]["quintiles"])
            se = np.sqrt(p * (1 - p) / (n / 5))
            ax.errorbar(q + off, 100 * p, yerr=196 * se, fmt="o-", color=col, lw=1.8, ms=5,
                        elinewidth=1, capsize=0, label=lab)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks(q, ["1\nlow", "2", "3", "4", "5\nhigh"])
        ax.set_ylim(bottom=0)
        tidy(ax)
    axes[0].set_ylabel("% of debut owners")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.supxlabel("fifth of the debut filing's lead or atypicality (within class, debut year and length fifth)",
                  fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(RES / "fig_ladder.png", dpi=200)
    plt.close(fig)


def staged() -> None:
    d = json.loads((RES / "staged_outcomes.json").read_text())["stages"]
    labels = [f'{s["stage"]}\n({s["conditioned_on"]})' for s in d]
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for key, col, lab, off in (("z_lean", LEAD, "lead", 0.14), ("z_level", ATYP, "atypicality", -0.14)):
        b = np.array([100 * s[key]["b_pp"] / s["base_rate_pct"] for s in d])
        se = np.array([100 * s[key]["se_pp"] / s["base_rate_pct"] for s in d])
        ax.errorbar(b, y + off, xerr=1.96 * se, fmt="o", color=col, ms=6, elinewidth=1.2,
                    capsize=0, label=lab)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlabel("change in the stage's rate for a one-SD increase, % of the base rate")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    tidy(ax, "x")
    fig.tight_layout()
    fig.savefig(RES / "fig_staged.png", dpi=200)
    plt.close(fig)


def reversal() -> None:
    o = json.loads((RES / "reversal_theme_decomp.json").read_text())
    cells = [c for c in o["focus_theme_web_cells"].values() if c["n"] >= 5000 and c["contrast"]]
    cells.sort(key=lambda c: c["contrast"]["lift"])
    names = {9: "Computer, software & info services", 18: "Media & entertainment goods",
             0: "Electronic apparatus & control", 12: "Retail store services",
             35: "Education, training & events", 2: "Design & technical development",
             13: "Research, scientific & diagnostic", 14: "Clothing, bags & footwear",
             3: "Health & wellness services", 42: "Construction & engineering services"}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(cells))
    for i, c in enumerate(cells):
        lift, se = 100 * c["contrast"]["lift"], 100 * c["contrast"]["se"]
        col = WEB if c["web"] else OTHER
        ax.errorbar(lift, i, xerr=1.96 * se, fmt="o", color=col, ms=7, elinewidth=1.3, capsize=0)
    ax.set_yticks(y, [f'{names.get(c["theme"], "theme " + str(c["theme"]))}'
                      f'{" (internet)" if c["web"] else ""}' for c in cells], fontsize=8.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("failure at the five-year proof, most leading minus most lagging fifth (pp)")
    ax.text(0.01, 1.02, "← leading filings survived more      leading filings failed more →",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker="o", ls="", color=WEB), Line2D([], [], marker="o", ls="", color=OTHER)],
              ["names the internet", "does not"], frameon=False, fontsize=8.5, loc="lower right")
    tidy(ax, "x")
    fig.tight_layout()
    fig.savefig(RES / "fig_reversal.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ladder()
    staged()
    reversal()
    if (RES / "lead_crowding.json").exists():
        crowding()
    print("figures written")
