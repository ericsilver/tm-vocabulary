"""Themes that rose from under 0.5% to over 5% of filings: survival by filing year.

A theme qualifies within an industry when its share of the Nice class's filings
(3-year moving average, primary theme under the production T = 50 model) fell
below 0.5% and later rose above 5%. Classes 43-45 are excluded: they were split
from class 42 in the 2002 Nice revision, so their earlier shares reflect
reclassification, not growth. Corpus-wide no theme crosses both thresholds; the
nearest, "Downloadable and online software" (0.1% of all filings in 1988, 4.7% in
2022), is drawn as the first panel across all classes.

Outcome: share of registrations, by filing year, that passed the five-year proof
of continued use (no Section 8/71 cancellation at registration age 4.0-8.5).
Only registrations from 2018 or earlier, whose proof window has elapsed, count;
filing years with fewer than 30 such registrations are not drawn. The grey line
is the same rate for every registration in the class (all classes in panel 1).

Output: paper/figures/fig_riser_survival.png (+ .pdf), paper/results/fig_riser_survival.json
"""
from __future__ import annotations

import glob
import json
import re
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
FIG = REPO / "paper" / "figures"
LO, HI, MIN_N = 0.005, 0.05, 30
YEARS = (1985, 2024)
THEME_C, BASE_C, INK, MUTED = "#0072B2", "#8C8C8C", "#222222", "#666666"
CLASS_NAMES = {"02": "Paints, inks", "04": "Fuels, candles", "09": "Electronics, software",
               "35": "Advertising, retail", "38": "Telecommunications", "41": "Education, entertainment",
               "42": "Technology services"}


def labels() -> dict:
    src = (REPO / "scripts" / "themes_t50_page.py").read_text(encoding="utf-8")
    m = re.search(r"LABELS\s*=\s*(\{.*?\n\})", src, re.S) or re.search(r"LABELS\s*=\s*(\[.*?\n\])", src, re.S)
    L = ast.literal_eval(m.group(1))
    return {k: L[k] for k in range(50)}


def main() -> int:
    lab = labels()
    th = pl.concat([pl.read_parquet(f, columns=["serial_number", "fy", "top_theme"]).with_columns(
        pl.lit(Path(f).stem[-2:]).alias("c")) for f in glob.glob(str(PROC / "theme_full" / "theme_class*.parquet"))]
    ).filter(pl.col("fy").is_between(*YEARS))
    # qualifying class-theme pairs
    s = th.group_by("c", "fy", "top_theme").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over(["c", "fy"])).alias("share"))
    yrs = np.arange(YEARS[0], YEARS[1] + 1)
    cases = []
    for (c, k), g in s.group_by(["c", "top_theme"]):
        if c in ("43", "44", "45"):
            continue
        d = dict(zip(g["fy"].to_list(), g["share"].to_list()))
        v = np.convolve([d.get(y, 0.0) for y in yrs], np.ones(3) / 3, mode="valid")
        y3 = yrs[1:-1]
        lo = np.where(v < LO)[0]
        if not len(lo):
            continue
        hi = np.where((v > HI) & (np.arange(len(v)) > lo[0]))[0]
        if len(hi):
            cases.append({"c": c, "k": int(k), "lo_year": int(y3[lo[lo < hi[0]].max()]),
                          "hi_year": int(y3[hi[0]]), "peak": float(v.max())})
    cases.sort(key=lambda r: (r["k"] != 5, r["hi_year"]))
    # corpus-wide near-miss: theme 5 across all classes
    cs = th.group_by("fy", "top_theme").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over("fy")).alias("share")).filter(pl.col("top_theme") == 5)
    cd = dict(zip(cs["fy"].to_list(), cs["share"].to_list()))
    cv = np.convolve([cd.get(y, 0.0) for y in yrs], np.ones(3) / 3, mode="valid")
    corpus = {"c": "all", "k": 5, "lo_year": int(yrs[1:-1][np.where(cv < LO)[0].max()]),
              "hi_year": int(yrs[1:-1][np.where(cv > 0.04)[0][0]]) if (cv > 0.04).any() else None,
              "peak": float(cv.max())}

    po = pl.read_parquet(REPO / "data" / "release" / "proof_outcomes.parquet",
                         columns=["serial_number", "registration_date", "failed_proof_window"]).with_columns(
        pl.col("registration_date").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32, strict=False).alias("ry")
    ).filter(pl.col("ry") <= 2018)
    d = th.join(po.select("serial_number", "failed_proof_window"), on="serial_number", how="inner").with_columns(
        (100 * (1 - pl.col("failed_proof_window").cast(pl.Float64))).alias("surv"))

    def series(df):
        g = df.group_by("fy").agg(pl.col("surv").mean().alias("m"), pl.len().alias("n")).sort("fy").filter(
            pl.col("n") >= MIN_N)
        m = g["m"].to_numpy()
        se = np.sqrt(m * (100 - m) / g["n"].to_numpy())
        return g["fy"].to_numpy(), m, se, g["n"].to_numpy()

    panels = [corpus] + cases
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), sharey=True, squeeze=False)
    out = []
    for ax, p in zip(axes.flat, panels):
        sub = d if p["c"] == "all" else d.filter(pl.col("c") == p["c"])
        bx, bm, _, _ = series(sub)
        tx, tm, tse, tn = series(sub.filter(pl.col("top_theme") == p["k"]))
        ax.plot(bx, bm, color=BASE_C, lw=1.6, label="all registrations in the class")
        ax.fill_between(tx, tm - 1.96 * tse, tm + 1.96 * tse, color=THEME_C, alpha=0.18, lw=0)
        ax.plot(tx, tm, color=THEME_C, lw=2, marker="o", ms=2.5, label="this theme")
        note = []
        for yr, txt in ((p["lo_year"], "<0.5%"), (p["hi_year"], ">5%")):
            if yr and yr <= 2017:          # inside the drawn range: mark it
                ax.axvline(yr, color=MUTED, lw=0.8, ls=":")
                ax.text(yr, 1.0, f" {txt}", transform=ax.get_xaxis_transform(), fontsize=6.5, color=MUTED,
                        va="top")
            elif yr:                       # crossed after the last drawn year: say so in the title
                note.append(f"{txt} in {yr}")
        if p["c"] == "all":
            note.append(f"peak {100 * p['peak']:.1f}% of all filings")
        where = "All classes" if p["c"] == "all" else f"Class {int(p['c'])} ({CLASS_NAMES.get(p['c'], '')})"
        extra = (" — " + "; ".join(note)) if note else ""
        ax.set_title(f"{lab[p['k']]}\n{where}{extra}", fontsize=8, color=INK, loc="left")
        ax.set_xlim(1985, 2017)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7, colors=MUTED)
        out.append({**p, "label": lab[p["k"]], "theme": {"fy": tx.tolist(), "surv": tm.tolist(), "n": tn.tolist()},
                    "class": {"fy": bx.tolist(), "surv": bm.tolist()}})
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for r in range(nrow):
        axes[r][0].set_ylabel("Passed five-year proof (%)", fontsize=7.5, color=MUTED)
    for c in range(ncol):
        col = [axes[r][c] for r in range(nrow) if r * ncol + c < len(panels)]
        if col:
            col[-1].set_xlabel("Filing year", fontsize=7.5, color=MUTED)
            col[-1].tick_params(labelbottom=True)
    h, l_ = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h[::-1], l_[::-1], loc="upper center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_riser_survival.png", dpi=160)
    fig.savefig(FIG / "fig_riser_survival.pdf")
    (RES / "fig_riser_survival.json").write_text(json.dumps(out, indent=1))
    for p in out:
        t = dict(zip(p["theme"]["fy"], p["theme"]["surv"]))
        b = dict(zip(p["class"]["fy"], p["class"]["surv"]))
        common = [y for y in t if y in b]
        gap = np.mean([t[y] - b[y] for y in common]) if common else float("nan")
        print(f"{p['c']:>3} theme {p['k']:2d} {p['label'][:34]:34s} <0.5% {p['lo_year']} >5% {p['hi_year']} "
              f"years drawn {len(common)}  mean gap vs class {gap:+.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
