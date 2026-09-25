"""Grid: five-year survival by lead, with and without each attribute that changes the cost of leading.

Panels are the factors (and class groups) whose with-minus-without difference in
the lead effect is significant after Holm adjustment in combined_model.json.
Each panel plots survival of the five-year proof by lead decile (within class and
registration year), one line for registrations with the attribute and one for
those without (continuous factors: top third against bottom third; boom and bust
against years that are neither). Survival is adjusted for class x registration-
year fixed effects (cell mean removed, overall mean added back), which is the
comparison the model makes; bands are 95% intervals of the bin means.

Output: paper/figures/fig_lead_by_factor.pdf (+ .png)
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
import combined_model as cm  # noqa: E402

RES = REPO / "paper" / "results"
FIG = REPO / "paper" / "figures"
WITH_C, WITHOUT_C = "#0072B2", "#8C8C8C"      # Okabe-Ito blue against neutral grey
INK, MUTED = "#222222", "#666666"
CONT = {k for k, _, _ in cm.CONTINUOUS}
YEAR = {"boom", "bust"}
SHORT = {"counsel": "Represented by counsel", "platform": "Platform or network offering",
         "gpt": "General-purpose-technology vocabulary", "mkt_pace": "Class filing-volume growth",
         "trend": "Filing year", "tech_pace": "Class theme-mix turnover", "site": "Site-based offering",
         "itu": "Intent-to-use filing"}


def p_of(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2)) if se else 1.0


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def main() -> int:
    R = json.loads((RES / "combined_model.json").read_text())
    J, lab = R["joint"], R["labels"]
    fac = [k for k, _, _ in cm.BINARY] + [k for k, _, _ in cm.CONTINUOUS]
    ph = holm({k: p_of(*J[k]["inter"][:2]) for k in fac})
    panels = [("factor", k) for k in sorted(fac, key=lambda k: ph[k]) if ph[k] < 0.05]
    Gv = R["class_group_vs_other"]
    pg = holm({k: p_of(*v["inter"]) for k, v in Gv.items()})
    panels += [("group", k) for k in sorted(Gv, key=lambda k: pg[k]) if pg[k] < 0.05]
    cm.log(f"panels: {[p[1] for p in panels]}")

    d = cm.frame()
    d = d.with_columns(
        (pl.col("surv") - pl.col("surv").mean().over("cell") + pl.col("surv").mean()).alias("adj"),
        (((pl.col("lead") + 0.5) * 10).floor().clip(0, 9)).cast(pl.Int8).alias("dec"),
        pl.col("reg_year"))
    grp_of = {c: g for g, (name, members) in enumerate(R["class_groups"].items()) for c in members}
    gnames = list(R["class_groups"].keys())
    cg_expr = {lbl: e for _, lbl, e in cm.CLASS_GROUPS}

    def split(kind, k):
        """Return (frame, with-mask expr, without-mask expr, with label, without label)."""
        if kind == "group":
            base = k.replace(" (held-out half)", "")
            if base in gnames:
                gi = gnames.index(base)
                dd = d.filter(pl.col("half") == 1)
                e = pl.col("cls").replace_strict(grp_of, return_dtype=pl.Int32) == gi
                return dd, e, ~e, "in these classes", "all other classes"
            e = cg_expr[k]
            return d, e, ~e, "in these classes", "all other classes"
        if k in YEAR:
            other = (YEAR - {k}).pop()
            return d, pl.col(k) == 1, (pl.col(k) == 0) & (pl.col(other) == 0), f"{k} years", "neither"
        if k in CONT:
            lo, hi = d[k].quantile(1 / 3), d[k].quantile(2 / 3)
            return d, pl.col(k) >= hi, pl.col(k) <= lo, "top third", "bottom third"
        return d, pl.col(k) == 1, pl.col(k) == 0, "with", "without"

    n = len(panels)
    ncol = 3 if n > 4 else n
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow), sharex=True, squeeze=False)
    summary = {}
    x = np.arange(10) * 10 + 5
    for ax, (kind, k) in zip(axes.flat, panels):
        dd, ew, eo, lw, lo_ = split(kind, k)
        rows = {}
        for tag, e in (("with", ew), ("without", eo)):
            s = dd.filter(e).group_by("dec").agg(pl.col("adj").mean().alias("m"), pl.col("adj").std().alias("s"),
                                                  pl.len().alias("n")).sort("dec")
            m = s["m"].to_numpy(); se = s["s"].to_numpy() / np.sqrt(s["n"].to_numpy())
            rows[tag] = (m, se, int(s["n"].sum()))
        ends = {t: rows[t][0][-1] for t in rows}
        span = max(max(rows[t][0].max() for t in rows) - min(rows[t][0].min() for t in rows), 1e-9)
        gap = ends["with"] - ends["without"]
        nudge = {"with": 0.0, "without": 0.0}
        if abs(gap) < 0.08 * span:          # end labels would collide: push them apart
            s_ = 1 if gap >= 0 else -1
            nudge = {"with": s_ * 0.05 * span, "without": -s_ * 0.05 * span}
        for tag, col, lbl in (("without", WITHOUT_C, lo_), ("with", WITH_C, lw)):
            m, se, nn = rows[tag]
            ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.18, linewidth=0)
            ax.plot(x, m, color=col, linewidth=2, marker="o", markersize=3.5)
            ax.annotate(lbl, (x[-1], m[-1] + nudge[tag]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK, va="center")
        title = SHORT.get(k, lab.get(k, k[0].upper() + k[1:]))
        if kind == "group":
            title = k[0].upper() + k[1:]
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_xlim(0, 118)
        ax.set_xticks([5, 25, 50, 75, 95])
        summary[k] = {t: {"survival_by_decile": [float(v) for v in rows[t][0]], "n": rows[t][2]} for t in rows}
    for ax in axes.flat[n:]:
        ax.axis("off")
    # the lowest panel in each column carries the x axis
    for c in range(ncol):
        col_axes = [axes[r][c] for r in range(nrow) if r * ncol + c < n]
        if col_axes:
            col_axes[-1].tick_params(labelbottom=True)
            col_axes[-1].set_xlabel("Lead percentile within class and year", fontsize=7.5, color=MUTED)
    for row in axes:
        row[0].set_ylabel("Survival at five-year proof (%)", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_lead_by_factor.pdf")
    fig.savefig(FIG / "fig_lead_by_factor.png", dpi=160)
    (RES / "fig_lead_by_factor.json").write_text(json.dumps(summary, indent=1))
    cm.log(f"wrote {FIG / 'fig_lead_by_factor.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
