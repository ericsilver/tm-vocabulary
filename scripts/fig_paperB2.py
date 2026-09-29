"""Figures for Paper B (v2).

fig_b_headline      survival by atypicality decile and by lead decile
fig_b_blueocean     the value of atypicality by lead, by class growth, by co-location
fig_b_magnitudes    each factor alone: its survival difference, and how it changes the
                    value of atypicality and of lead (95% intervals, Holm stars)
fig_b_protect       survival by atypicality for filings with and without patents, counsel,
                    platform vocabulary; and by lead for the same, plus registration period
fig_b_industries    class lead effects at the proof and at renewal, named

All survival figures are adjusted for class x registration-year fixed effects
(the cell mean is removed and the overall mean added back), so lines compare
filings of the same class and year. x axes are percentiles within class and year.

Output: paper/figures/fig_b_*.pdf (+ .png)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import paperB_models as pm  # noqa: E402
from class_profiles import NAMES  # noqa: E402

RES, FIG = REPO / "paper" / "results", REPO / "paper" / "figures"
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#8C8C8C", "#009E73"
X = np.arange(10) * 10 + 5
XLAB_A = "Atypicality percentile within class and year\n(0 = most typical language, 100 = most unusual)"
XLAB_L = "Lead percentile within class and year\n(0 = most lagging language, 100 = most leading)"
YLAB = "Passed the five-year proof (%)"


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.set_xticks([5, 25, 50, 75, 95]); ax.set_xlim(0, 100)


def line(ax, rows, color, label, ls="-"):
    m = np.array([r["m"] for r in rows]); se = np.array([r["s"] / math.sqrt(r["n"]) for r in rows])
    ax.fill_between(X[:len(m)], m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15, linewidth=0)
    ax.plot(X[:len(m)], m, color=color, linewidth=2, marker="o", markersize=3.2, label=label, linestyle=ls)


def curves(d: pl.DataFrame, by: list[str], dec: str) -> dict:
    g = d.group_by(by + [dec]).agg(pl.col("adj").mean().alias("m"), pl.col("adj").std().alias("s"),
                                   pl.len().alias("n")).sort(by + [dec])
    out = {}
    for key, sub in g.group_by(by, maintain_order=True):
        out[key if len(by) > 1 else key[0]] = sub.to_dicts()
    return out


def p_of(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2))


def holm(ps):
    items = sorted(ps.items(), key=lambda kv: kv[1]); m = len(items); run = 0.0; out = {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main() -> int:
    R = json.loads((RES / "paperB_models.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    d = pm.load()
    d = d.with_columns(
        (pl.col("surv") - pl.col("surv").mean().over("cell") + pl.col("surv").mean()).alias("adj"),
        (((pl.col("atyp") + 0.5) * 10).floor().clip(0, 9)).cast(pl.Int8).alias("adec"),
        (((pl.col("lead") + 0.5) * 10).floor().clip(0, 9)).cast(pl.Int8).alias("ldec"),
        (((pl.col("lead") + 0.5) * 3).floor().clip(0, 2)).cast(pl.Int8).alias("lthird"))
    third = lambda c: (pl.when(pl.col(c) >= d[c].quantile(2 / 3)).then(pl.lit(2))
                       .when(pl.col(c) <= d[c].quantile(1 / 3)).then(pl.lit(0)).otherwise(pl.lit(1))).cast(pl.Int8)
    d = d.with_columns(third("mkt_pace").alias("mthird"), third("colocation").alias("cthird"),
                       pl.when(pl.col("reg_year") <= 2007).then(pl.lit(0)).when(pl.col("reg_year") <= 2012)
                       .then(pl.lit(1)).otherwise(pl.lit(2)).cast(pl.Int8).alias("period"))

    # ---- 1. headline
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    a = d.group_by("adec").agg(pl.col("adj").mean().alias("m"), pl.col("adj").std().alias("s"), pl.len().alias("n")).sort("adec").to_dicts()
    l_ = d.group_by("ldec").agg(pl.col("adj").mean().alias("m"), pl.col("adj").std().alias("s"), pl.len().alias("n")).sort("ldec").to_dicts()
    line(axes[0], a, BLUE, None); line(axes[1], l_, ORANGE, None)
    axes[0].set_title("Unusual language survives more often", fontsize=9.5, loc="left", color=INK)
    axes[1].set_title("Early language survives less often", fontsize=9.5, loc="left", color=INK)
    axes[0].set_xlabel(XLAB_A, fontsize=7.5, color=MUTED); axes[1].set_xlabel(XLAB_L, fontsize=7.5, color=MUTED)
    axes[0].set_ylabel(YLAB, fontsize=8, color=MUTED)
    for ax in axes:
        style(ax)
    fig.tight_layout(); fig.savefig(FIG / "fig_b_headline.pdf"); fig.savefig(FIG / "fig_b_headline.png", dpi=160)
    plt.close(fig)

    # ---- 2. blue ocean and competition
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True)
    specs = [("lthird", "By the filing's own lead", {0: ("Most lagging third", GREY), 2: ("Most leading third", ORANGE)}),
             ("mthird", "By growth in the class's filing volume",
              {0: ("Slowest-growing third of class-years", GREY), 2: ("Fastest-growing third", ORANGE)}),
             ("cthird", "By the geographic concentration of the theme",
              {0: ("Most dispersed third of themes", GREY), 2: ("Most concentrated third", ORANGE)})]
    for ax, (col, title, lev) in zip(axes, specs):
        cv = curves(d, [col], "adec")
        for k, (lab, colr) in lev.items():
            line(ax, cv[k], colr, lab)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xlabel(XLAB_A, fontsize=7, color=MUTED); style(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel(YLAB, fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / "fig_b_blueocean.pdf"); fig.savefig(FIG / "fig_b_blueocean.png", dpi=160)
    plt.close(fig)

    # ---- 3. magnitudes (each factor alone)
    facs = R["factors"]; A = R["alone"]
    GROUP = {"counsel": "record", "patents": "record", "debut": "record", "established": "record", "itu": "record",
             "foreign": "record", "china": "record", "site": "forced", "contract": "forced", "platform": "forced",
             "gpt": "forced", "invention": "forced", "consumer": "forced", "imported": "emergent",
             "volatility": "emergent", "new_demand": "emergent", "tech_pace": "emergent", "colocation": "geography",
             "has_hub": "geography", "in_hub": "geography", "mkt_pace": "time", "trend": "time", "boom": "time",
             "bust": "time"}
    for f in facs:
        f["source"] = GROUP[f["key"]]
    order = [f for src in ("record", "forced", "emergent", "geography", "time") for f in facs if f["source"] == src]
    keys = [f["key"] for f in order]
    panels = [("main", lambda k: k, "Survival difference\nwith the factor (points)"),
              ("atyp", lambda k: f"{k}:atyp", "Change in the value of\natypicality (points)"),
              ("lead", lambda k: f"{k}:lead", "Change in the value of\nlead (points)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 9.6), sharey=True, gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ypos = np.arange(len(keys))[::-1]
    src_col = {"record": INK, "forced": BLUE, "emergent": GREEN, "geography": ORANGE, "time": GREY}
    for ax, (tag, term, title) in zip(axes, panels):
        est = {k: A[k][term(k)] for k in keys}
        ph = holm({k: p_of(*v) for k, v in est.items()})
        for y, f in zip(ypos, order):
            b, se = est[f["key"]]
            c = src_col[f["source"]]
            ax.errorbar(b, y, xerr=1.96 * se, fmt="o", color=c, markersize=4, elinewidth=1.2, capsize=0)
            ax.text(b + (1.96 * se if b >= 0 else -1.96 * se), y + 0.28, f"{b:+.1f}{stars(ph[f['key']])}",
                    fontsize=6, color=MUTED, ha="left" if b >= 0 else "right")
        ax.axvline(0, color="#999999", linewidth=0.8)
        lo_ = min(b - 1.96 * se for b, se in est.values()); hi_ = max(b + 1.96 * se for b, se in est.values())
        pad = 0.22 * (hi_ - lo_)
        ax.set_xlim(lo_ - pad, hi_ + pad)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=MUTED)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels([f["label"] for f in order], fontsize=7.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=s)
               for s, c in (("Owner and filing (record)", INK), ("Offering (forced vocabulary)", BLUE),
                            ("Theme (emergent)", GREEN), ("Geography", ORANGE), ("Class and year", GREY))]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig_b_magnitudes.pdf"); fig.savefig(FIG / "fig_b_magnitudes.png", dpi=160)
    plt.close(fig)

    # ---- 4. what protects the blue ocean, and what changes the cost of leading
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 6.6))
    rowspec = [
        ("adec", XLAB_A, [("patents", "Owner files patents"), ("counsel", "Represented by counsel"),
                          ("platform", "Platform or network offering"), ("gpt", "General-purpose-technology vocabulary")]),
        ("ldec", XLAB_L, [("counsel", "Represented by counsel"), ("platform", "Platform or network offering"),
                          ("gpt", "General-purpose-technology vocabulary"), ("period", "Registration period")])]
    for r_, (dec, xlab, items) in enumerate(rowspec):
        for c_, (k, title) in enumerate(items):
            ax = axes[r_][c_]
            cv = curves(d, [k], dec)
            if k == "period":
                for lv, lab, colr in ((0, "Registered 2002-07", GREY), (1, "2008-12", BLUE), (2, "2013-18", ORANGE)):
                    line(ax, cv[lv], colr, lab)
            else:
                line(ax, cv[0.0] if 0.0 in cv else cv[0], GREY, "Without")
                line(ax, cv[1.0] if 1.0 in cv else cv[1], BLUE, "With")
            ax.set_title(title, fontsize=8.5, loc="left", color=INK)
            ax.set_xlabel(xlab, fontsize=6.5, color=MUTED); style(ax)
            ax.legend(fontsize=6.5, frameon=False)
        axes[r_][0].set_ylabel(YLAB, fontsize=7.5, color=MUTED)
    fig.text(0.005, 0.985, "a. The value of unusual language", fontsize=9, color=INK, va="top")
    fig.text(0.005, 0.49, "b. The cost of early language", fontsize=9, color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    fig.savefig(FIG / "fig_b_protect.pdf"); fig.savefig(FIG / "fig_b_protect.png", dpi=160)
    plt.close(fig)

    # ---- 5. industries: proof vs renewal lead effects, named
    W = json.loads((RES / "where_it_accrues.json").read_text())["by_class"]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    labels = []
    for c, v in W.items():
        if "lead" not in v["proof"] or "lead" not in v["renewal"]:
            continue
        x_, y_ = v["proof"]["lead"][0], v["renewal"]["lead"][0]
        col = ORANGE if (x_ < -2 and y_ < -2) else BLUE if (x_ > 1.5 and y_ > 1.5) else GREY
        ax.scatter(x_, y_, s=10 + 150 * v["proof"]["n"] / 450000, color=col, alpha=0.85, edgecolors="white", linewidths=0.6)
        if abs(x_) > 3 or abs(y_) > 3.5:
            labels.append([x_, y_, y_, NAMES[int(c)]])
    labels.sort(key=lambda r: -r[2])
    placed = []
    for lab in labels:
        for _ in range(40):
            clash = [p for p in placed if abs(p[0] - lab[0]) < 2.6 and abs(p[2] - lab[2]) < 0.38]
            if not clash:
                break
            lab[2] -= 0.4
        placed.append(lab)
        ax.annotate(lab[3], (lab[0], lab[1]), xytext=(lab[0] + 0.25, lab[2] + 0.12), fontsize=6.8, color=INK,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", linewidth=0.5) if abs(lab[2] - lab[1]) > 0.2 else None)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8); ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xlabel("Effect of lead on passing the five-year proof\n(points, most leading minus most lagging filing)",
                  fontsize=8, color=MUTED)
    ax.set_ylabel("Effect of lead on renewal at year ten, among marks that passed the proof (points)",
                  fontsize=8, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    fig.tight_layout(); fig.savefig(FIG / "fig_b_industries.pdf"); fig.savefig(FIG / "fig_b_industries.png", dpi=160)
    plt.close(fig)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
