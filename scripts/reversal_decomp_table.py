"""Emit paper/results/reversal_decomp.tex from reversal_theme_decomp.json.

Table for Paper B, section "What carried the reversal": theme x internet-flag
cells of the 2000-2004 technology pool, with each cell's size, its share of
the era's pooled leading fifth, its failure rate, and the within-cell
top-minus-bottom lead-quintile contrast; below, the pooled statements the
text quotes (baseline, services theme excluded, web/non-web within cells).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"

MIN_N = 5_000  # every theme x flag cell at least this large is shown


def fmt(r: dict | None) -> str:
    return f"${100*r['lift']:+.2f}$ ({100*r['se']:.2f})" if r else "---"


def main() -> int:
    o = json.loads((RES / "reversal_theme_decomp.json").read_text())
    cells, words = o["focus_theme_web_cells"], o["theme_words"]
    loo, split = o["focus_leave_one_out"], o["era_web_split"]["2000-2004"]

    rows = [r"\begin{tabular}{p{5.6cm}rrrrr}", r"\toprule",
            r"Cell & $n$ & Share & Of top fifth & Failure & Q5$-$Q1 \\",
            r"\midrule"]
    shown = sorted((c for c in cells.values() if c["n"] >= MIN_N),
                   key=lambda c: -c["n"])
    for c in shown:
        name = ", ".join(words[str(c["theme"])][:4])
        tag = "internet" if c["web"] else "other"
        rows.append(
            f"{name} ({tag}) & {c['n']:,} & {100*c['share_all']:.1f}\\% & "
            f"{100*c['share_top']:.1f}\\% & {100*c['base']:.1f}\\% & "
            f"{fmt(c['contrast'])} \\\\")
    rows += [r"\midrule",
             f"All filings, pooled quintiles & {loo['baseline']['n']:,} & "
             f"100\\% & 100\\% & {100*loo['baseline']['base']:.1f}\\% & "
             f"{fmt(loo['baseline'])} \\\\",
             f"\\quad excluding the services theme & {loo['drop_t9']['n']:,} "
             f"& & & & {fmt(loo['drop_t9'])} \\\\",
             f"Internet-bearing, within class$\\times$cohort & "
             f"{split['web_within']['n']:,} & & & "
             f"{100*split['web_within']['base']:.1f}\\% & "
             f"{fmt(split['web_within'])} \\\\",
             f"Other filings, within class$\\times$cohort & "
             f"{split['nonweb_within']['n']:,} & & & "
             f"{100*split['nonweb_within']['base']:.1f}\\% & "
             f"{fmt(split['nonweb_within'])} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    (RES / "reversal_decomp.tex").write_text("\n".join(rows) + "\n")
    print("wrote", RES / "reversal_decomp.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
