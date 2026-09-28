"""Class table for the where-it-accrues summary: lead and atypicality at the proof and at renewal.

Classes are sorted into patterns by their lead effects at the two stages
(points, most leading minus most lagging; a 2-point threshold marks an effect
as material):
  penalty persists     proof <= -2 and renewal <= -2
  penalty fades        proof <= -2 and renewal > -1
  penalty appears late proof > -2 and renewal <= -3
  benefit persists     proof >= 1.5 and renewal >= 1.5
  benefit fades        proof >= 2 and renewal < 1
  small                everything else
Output: paper/split/B_brief/where_table.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from class_profiles import NAMES  # noqa: E402

RES = REPO / "paper" / "results"
OUT = REPO / "paper" / "split" / "B_brief" / "where_table.tex"
ORDER = ["penalty persists", "penalty appears late", "penalty fades", "benefit persists", "benefit fades", "small"]


def pattern(p, r):
    if p <= -2 and r <= -2:
        return "penalty persists"
    if p <= -2 and r > -1:
        return "penalty fades"
    if p > -2 and r <= -3:
        return "penalty appears late"
    if p >= 1.5 and r >= 1.5:
        return "benefit persists"
    if p >= 2 and r < 1:
        return "benefit fades"
    return "small"


def n(x):
    if round(x, 1) == 0:
        return "0.0"
    return f"{x:+.1f}".replace("-", "$-$")


def main() -> int:
    R = json.loads((RES / "where_it_accrues.json").read_text())
    rows = []
    for c, v in R["by_class"].items():
        P, Rn = v["proof"], v["renewal"]
        if "lead" not in P or "lead" not in Rn:
            continue
        rows.append({"cls": int(c), "n": P["n"], "lp": P["lead"], "lr": Rn["lead"], "ap": P["atyp"], "ar": Rn["atyp"],
                     "pat": pattern(P["lead"][0], Rn["lead"][0])})
    L = [r"{\small\begin{longtable}{r>{\raggedright\arraybackslash}p{3.7cm}rrrrr}",
         r"\toprule & & & \multicolumn{2}{c}{Lead} & \multicolumn{2}{c}{Atypicality} \\"
         r" \cmidrule(lr){4-5}\cmidrule(l){6-7} Class & & Registrations & Proof & Renewal & Proof & Renewal \\ \midrule \endhead"]
    for pat in ORDER:
        rs = sorted([r for r in rows if r["pat"] == pat], key=lambda r: r["lp"][0])
        if not rs:
            continue
        L.append(r"\addlinespace\multicolumn{7}{l}{\emph{Lead: " + pat + r"}} \\")
        for r in rs:
            L.append(f"{r['cls']} & {NAMES[r['cls']]} & {r['n']:,} & {n(r['lp'][0])} ({r['lp'][1]:.1f}) & "
                     f"{n(r['lr'][0])} ({r['lr'][1]:.1f}) & {n(r['ap'][0])} & {n(r['ar'][0])} \\\\")
    L.append(r"\bottomrule\end{longtable}}")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    counts = {p: sum(r["pat"] == p for r in rows) for p in ORDER}
    print(f"wrote {OUT}; {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
