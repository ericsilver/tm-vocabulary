"""LaTeX tables from paper/results/hypothesis_battery.json and geo_cluster_validation.json.

Output: paper/split/B_brief/battery_results.tex (input by hypotheses.tex)
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
OUT = REPO / "paper" / "split" / "B_brief" / "battery_results.tex"


def esc(s: str) -> str:
    return (s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
            .replace("[", "{[}").replace("]", "{]}"))


def f(x, d=2):
    return "--" if x is None else f"{x:+.{d}f}"


def main() -> int:
    H = json.loads((RES / "hypothesis_battery.json").read_text())
    G = json.loads((RES / "geo_cluster_validation.json").read_text())
    L = []
    h0 = H["H0"]["groups"]["all"]
    L.append(r"\section*{Results}")
    L.append(
        "Penalty = failure rate at the five-year proof of the most leading fifth minus the most lagging "
        f"fifth, in percentage points (all registrations 2002--2018: {h0['penalty_pp']:+.2f}, "
        f"s.e.\\ {h0['se_pp']:.2f}). Each row's two penalties come from one model with the lead fifths "
        "interacted with the group; the difference is the first minus the second. Predicted: the sign of "
        "that difference stated in the catalogue (? = no prediction). Holm $p$ adjusts across all "
        "pre-stated contrasts. Agrees: whether the sign matches the prediction, given only where "
        "Holm $p < 0.10$.\n")
    L.append(r"{\small")
    L.append(r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.7cm}>{\raggedright\arraybackslash}p{4.6cm}"
             r">{\raggedright\arraybackslash}p{2.9cm}rrrcr}")
    L.append(r"\toprule & Groups compared & Penalty (pp) & Diff. & $t$ & Holm $p$ & Pred. & Agrees \\ \midrule \endhead")
    order = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10", "H11", "H12", "H13", "H14", "H15",
             "H17", "H18", "H20", "H22", "H23", "H24", "H25", "H26", "H27", "H28", "H29", "H30", "H31",
             "H32", "H33", "H34"]
    for hid in order:
        h = H.get(hid)
        if not h or "diff_pp" not in h:
            continue
        a, b = [x.strip() for x in h["contrast"].split(" minus ")]
        gr = h["groups"]
        def pen(k):
            v = gr.get(k, {})
            return v.get("penalty_pp", v.get("excess_pp"))
        pens = f"{esc(a)} {f(pen(a))}; {esc(b)} {f(pen(b))}"
        # verdict only where the contrast survives the adjustment (Holm p < 0.10)
        ag = ("" if h.get("agrees") is None else "--" if h.get("p_holm", 1) >= 0.10
              else ("yes" if h["agrees"] else "no"))
        L.append(f"{hid[1:]} & {esc(h['title'])} & {pens} & {f(h['diff_pp'])} & {h['t']:.1f} & "
                 f"{h.get('p_holm', 1):.3f} & {h['predicted']} & {ag} \\\\")
    L.append(r"\bottomrule\end{longtable}}")
    L.append("\nRows 15, 26 and 27 report excess failure (pp) over other filings in the same class and "
             "registration year, not a lead penalty.\n")
    # pace quadrants
    q = H["H8-11"]["groups"]
    L.append(r"\paragraph{Pace of change (hypotheses 8--11).} Penalty by quadrant: " + "; ".join(
        f"{esc(k)} {f(v['penalty_pp'])} (n = {v['n']:,})" for k, v in q.items()) + ".")
    # verticals
    for key, title in (("H19", "Golder--Tellis cases (hypothesis 19)"),
                       ("H21", "Blue-ocean exemplar verticals (hypothesis 21)")):
        parts = []
        for k, v in H[key]["groups"].items():
            nm = k.split("_", 1)[1].replace("_", " ")
            if "penalty_pp" in v:
                parts.append(f"{nm} {f(v['penalty_pp'])} (s.e.\\ {v['se_pp']:.2f}, n = {v['n']:,})")
            else:
                parts.append(f"{nm}: too few")
        L.append(r"\paragraph{" + title + ".} Penalty within each: " + "; ".join(parts) + ".")
    ex = H["H21"].get("exemplar_firms", {})
    if ex:
        L.append(r"\paragraph{The exemplar firms' own marks.} Registrations 2002--2018 by owner; mean lead "
                 "fifth (1 = most lagging, 5 = most leading), share in the most leading fifth, and proof "
                 "failure rate: " + "; ".join(
                     f"{esc(k)} {v['n_marks']} marks, fifth {v['mean_lead_fifth']:.1f}, "
                     f"{100*v['share_top_fifth']:.0f}\\% leading, {v['fail_rate']:.0f}\\% failed"
                     for k, v in ex.items()) + ".")
    hm = H["H29"].get("hub_main_effect_pp")
    if hm:
        L.append(r"\paragraph{Being in a hub.} Within themes that have a hub, filers in the hub fail "
                 f"{f(hm['b'])} points more often than filers outside it (s.e.\\ {hm['se']:.2f}), "
                 "whatever their lead.")
    # geography validation
    L.append(r"\section*{The clustering measure}")
    L.append(
        "For a group of $N$ US filers placed at their ZIP centroids, the co-location index is the "
        "average over pairs of distinct filers of $k(d) = \\max(0, 1 - d/200\\text{ miles})$: two filers "
        "in the same place score 1, two 100 miles apart 0.5, two more than 200 miles apart 0. It is the "
        "chance that two filers drawn at random are neighbours, weighted by how near, and it allows any "
        "number of hubs. Filers crowd into populous places whatever they sell, so the index is set "
        "against the same index $B$ for all filers in the class and years: $L = (C - B)/(1 - B)$, which is "
        "0 for a group placed like its class and 1 when every filer is in one place. This is the "
        "distance-based comparison of \\citet{DurantonOverman2005} summarized at one bandwidth. A hub is a "
        "place where at least 10\\% of the group's filers are nearby (kernel-weighted) and at least 1.2 "
        "times the class's share; hubs are picked greedily and named by the commonest owner city within "
        "50 miles. For the battery the groups are each class's filers of one primary theme in a five-year "
        "window around the filing year, with at least 30 US filers.\n")
    L.append(r"{\small\begin{longtable}{>{\raggedright\arraybackslash}p{5.2cm}rrrr>{\raggedright\arraybackslash}p{4.6cm}}")
    L.append(r"\toprule Group & $N$ & $C$ & $B$ & $L$ & Hubs (share nearby vs class) \\ \midrule \endhead")
    for k, v in G.items():
        hubs = "; ".join(f"{esc(h['name'])} {100*h['share_nearby']:.0f}\\% vs {100*h['class_share_nearby']:.0f}\\%"
                         for h in v["hubs"]) or "none"
        L.append(f"{esc(k)} & {v['n']:,} & {v['C']:.3f} & {v['B']:.3f} & {v['L']:.3f} & {hubs} \\\\")
    L.append(r"\bottomrule\end{longtable}}")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
