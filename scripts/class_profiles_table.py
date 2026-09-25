"""LaTeX for class_profiles.json: suggested clusters with their metrics, for review.

Output: paper/split/B_brief/class_profiles.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
OUT = REPO / "paper" / "split" / "B_brief" / "class_profiles.tex"

import sys  # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from class_profiles import NAMES, METRICS  # noqa: E402

LABEL = {"survival": "Survival (\\%)", "survival_sd": "SD across years (pp)", "lead_mean": "Lead, mean",
         "lead_var": "Lead, variance", "froth": "Froth"}


def m(x):
    return f"{x:+.1f}".replace("-", "$-$")


def main() -> int:
    R = json.loads((RES / "class_profiles.json").read_text())
    C, S = R["classes"], {s["cluster"]: s for s in R["clusters"]}
    # describe each cluster by the metrics on which it departs most from the all-class mean
    X = np.array([[c[k] for k in METRICS] for c in C])
    mu, sd = X.mean(0), X.std(0)

    def describe(s):
        z = {k: (s[k] - mu[i]) / sd[i] for i, k in enumerate(METRICS)}
        words = {"survival": ("high survival", "low survival"), "survival_sd": ("volatile survival", "steady survival"),
                 "lead_mean": ("leading on average", "lagging on average"),
                 "lead_var": ("widely spread lead", "narrow lead"), "froth": ("frothy", "calm")}
        top = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:2]
        return ", ".join(words[k][0 if v > 0 else 1] for k, v in top if abs(v) > 0.4) or "near the average"

    # precision-weighted correlation of each class's lead effect with each metric
    b = np.array([c["lead_effect"] for c in C])
    w = 1 / np.array([c["lead_effect_se"] for c in C]) ** 2

    def wcorr(x):
        xm, bm = (w * x).sum() / w.sum(), (w * b).sum() / w.sum()
        return float((w * (x - xm) * (b - bm)).sum() / np.sqrt((w * (x - xm) ** 2).sum() * (w * (b - bm) ** 2).sum()))

    corr = {k: wcorr(X[:, i]) for i, k in enumerate(METRICS)}
    L = [r"\section*{Classes that look alike}"]
    L.append(
        "Each Nice class is described by five metrics computed on its 2002--2018 registrations: the share "
        "passing the five-year proof; the SD of that share across registration years; the mean and variance "
        "of the raw lead score; and froth, the class's average five-year turnover in its theme mix (half the "
        "summed absolute change in theme shares between the five filing years before and the five after; 0 = "
        "no change, 1 = complete). The metrics were standardized and clustered by Ward linkage; "
        f"$k = {R['k']}$ has the highest silhouette ({R['silhouette'][str(R['k'])]:.2f}; values near 0.25 "
        "indicate weak separation, so the clusters are a suggestion for review, not a finding). The last column, "
        "each class's own lead effect (survival of its most leading minus its most lagging filing, class "
        "$\\times$ year fixed effects and the paper's controls), played no part in forming the clusters.\n")
    L.append(
        f"The clusters account for {100 * R['Q_between'] / R['Q_total']:.0f}\\% of the precision-weighted "
        f"variation in class lead effects ($\\chi^2_{{{R['k'] - 1}}} = {R['Q_between']:.0f}$, "
        f"$p < 10^{{{int(np.floor(np.log10(max(R['p_between'], 1e-300))))+1}}}$). Across classes, the lead effect "
        "correlates (precision-weighted) with " + "; ".join(
            f"{LABEL[k].lower().replace(' (\\%)', '').replace(' (pp)', '')} {corr[k]:+.2f}".replace("-", "$-$")
            for k in METRICS) + ".\n")
    L.append(r"{\small\begin{longtable}{r>{\raggedright\arraybackslash}p{3.6cm}rrrrrr}")
    L.append(r"\toprule Class & & Survival & SD yrs & Lead mean & Lead var. & Froth & Lead effect \\ \midrule \endhead")
    for g in sorted(S):
        s = S[g]
        noun = "class" if s["classes"] == 1 else "classes"
        head = (r"\textbf{Cluster " + str(g) + ": " + describe(s) + "} "
                + f"({s['classes']} {noun}; pooled lead effect {m(s['lead_effect_pooled'])}, "
                + f"s.e.\\ {s['lead_effect_pooled_se']:.2f})")
        L.append(r"\addlinespace\multicolumn{8}{l}{" + head + r"} \\")
        for c in sorted([c for c in C if c["cluster"] == g], key=lambda c: c["survival"]):
            L.append(f"{int(c['cls'])} & {NAMES[int(c['cls'])]} & {c['survival']:.1f} & {c['survival_sd']:.1f} & "
                     f"{c['lead_mean']:+.3f} & {c['lead_var']:.3f} & {c['froth']:.3f} & "
                     f"{m(c['lead_effect'])} ({c['lead_effect_se']:.1f}) \\\\".replace("+0.", "0.").replace("-0.", "$-$0."))
    L.append(r"\bottomrule\end{longtable}}")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for k, v in corr.items():
        print(f"  corr(lead effect, {k}) = {v:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
