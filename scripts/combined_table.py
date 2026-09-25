"""LaTeX for the combined model (paper/results/combined_model.json).

Output: paper/split/B_brief/combined_results.tex
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
OUT = REPO / "paper" / "split" / "B_brief" / "combined_results.tex"

BINARY = ["counsel", "patents", "itu", "debut", "established", "foreign", "china", "site", "contract",
          "platform", "invention", "gpt", "consumer", "imported", "has_hub", "in_hub", "boom", "bust"]
CONT = ["volatility", "new_demand", "colocation", "tech_pace", "mkt_pace", "trend"]
CG = ["services", "pharma", "cpg", "manufactured", "regulated"]
YEAR = ["boom", "bust"]
SECTIONS = [("The owner and the filing", ["counsel", "patents", "debut", "established", "foreign", "china", "itu"]),
            ("The offering", ["site", "contract", "platform", "invention", "gpt", "consumer", "imported"]),
            ("The theme and its geography", ["volatility", "new_demand", "colocation", "has_hub", "in_hub"]),
            ("The class and the year (boom and bust against years that are neither)",
             ["tech_pace", "mkt_pace", "trend", "boom", "bust"])]


def p_of(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2)) if se else 1.0


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def st(p):
    return r"$^{***}$" if p < 0.001 else r"$^{**}$" if p < 0.01 else r"$^{*}$" if p < 0.05 else ""


def num(b, p=None, d=2):
    s = f"{b:+.{d}f}".replace("-", "$-$")
    return s + (st(p) if p is not None else "")


def est(b, se):
    return f"{num(b)} ({se:.2f})"


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%")


def main() -> int:
    R = json.loads((RES / "combined_model.json").read_text())
    J, lab, sh = R["joint"], R["labels"], R["shares"]
    lead_b, lead_se = J["lead"][0], J["lead"][1]
    keys = BINARY + CONT + CG
    vn = R["joint_V"]["names"]
    V = np.array(R["joint_V"]["V"])
    ix = {n: i for i, n in enumerate(vn)}
    coef = {"lead": lead_b, **{f"{k}:lead": J[k]["inter"][0] for k in keys}}

    def comb(w: dict) -> tuple[float, float]:
        a = np.zeros(len(vn))
        for k, wk in w.items():
            a[ix[k]] = wk
        return float(sum(wk * coef[k] for k, wk in w.items())), float(math.sqrt(max(a @ V @ a, 0)))

    mean = J["means"]
    withv, without, diff = {}, {}, {}
    for k in keys:
        if k in YEAR:
            other = [y for y in YEAR if y != k][0]
            base = {"lead": 1.0, f"{other}:lead": -mean[other]}
            withv[k] = comb({**base, f"{k}:lead": 1 - mean[k]})
            without[k] = comb({**base, f"{k}:lead": -mean[k]})        # years that are neither
            diff[k] = tuple(J[k]["inter"][:2])
        elif k in CONT:
            withv[k] = comb({"lead": 1.0, f"{k}:lead": 1.0})
            without[k] = comb({"lead": 1.0, f"{k}:lead": -1.0})
            diff[k] = (2 * J[k]["inter"][0], 2 * J[k]["inter"][1])
        else:
            withv[k] = comb({"lead": 1.0, f"{k}:lead": 1 - mean[k]})
            without[k] = comb({"lead": 1.0, f"{k}:lead": -mean[k]})
            diff[k] = tuple(J[k]["inter"][:2])
    main_keys = [k for _, ks in SECTIONS for k in ks]
    pm = holm({k: p_of(*J[k]["main"][:2]) for k in main_keys if J[k]["main"]})
    pd_ = holm({k: p_of(*diff[k]) for k in main_keys})
    L = [r"\section*{Where being leading costs: one model}"]
    L.append(
        "One linear probability model of surviving the five-year proof (percentage points), with class "
        "$\\times$ registration-year fixed effects and standard errors clustered by owner; "
        f"{R['n']:,} registrations, 2002--2018. \\emph{{Lead}} is the filing's percentile of lead within its "
        "class and year, so a lead effect is the survival difference between the most leading and the most "
        "lagging filing (negative: leading costs). Every factor enters as a main effect and as an interaction "
        "with lead, all at once. \\emph{Survival} is the factor's association with survival at median lead. "
        "\\emph{With} and \\emph{without} are the lead effects among filings that have and lack the factor, "
        "with standard errors (continuous factors: one SD above and below the mean; boom and bust: against "
        "years that are neither). \\emph{Difference} is with minus without; it is the test of whether the "
        "factor changes the cost of leading. Stars (Survival and Difference only): Holm-adjusted $p$ within "
        "the column, $^{*}<0.05$, $^{**}<0.01$, $^{***}<0.001$. The with and without estimates carry no stars "
        "because they answer a different question -- is leading costly within this subgroup? -- and a small "
        "subgroup can be indistinguishable from zero while the large remainder, whose estimate sits near the "
        "average of $-1.1$, is not. Controls: description length, owner filing count, foreign filing basis, "
        "missing-data flags.\n")

    def share_of(k):
        if k not in sh:
            return "SD"
        return f"{100*sh[k]:.1f}\\%" if sh[k] < 0.01 else f"{100*sh[k]:.0f}\\%"

    L.append(r"{\small\begin{longtable}{>{\raggedright\arraybackslash}p{4.9cm}rrrrr}")
    L.append(r"\toprule & & & \multicolumn{3}{c}{Lead effect} \\ \cmidrule(l){4-6}"
             r" Factor & Share & Survival & With & Without & Difference \\ \midrule \endhead")
    L.append(f"\\textbf{{Lead, average filing}} & & & \\multicolumn{{2}}{{c}}{{{est(lead_b, lead_se)}}} & \\\\")
    L.append(f"\\quad lead alone, no factors & & & \\multicolumn{{2}}{{c}}{{{est(R['lead_only']['b'], R['lead_only']['se'])}}} & \\\\")
    for title, ks in SECTIONS:
        L.append(f"\\addlinespace\\multicolumn{{6}}{{l}}{{\\emph{{{esc(title)}}}}} \\\\")
        for k in ks:
            main = num(J[k]["main"][0], pm[k]) if J[k]["main"] else "--"
            L.append(f"{esc(lab[k])} & {share_of(k)} & {main} & {est(*withv[k])} & {est(*without[k])} & "
                     f"{num(diff[k][0], pd_[k])} \\\\")
    L.append(r"\bottomrule\end{longtable}}")

    # class groups against all other classes
    Gv = R["class_group_vs_other"]
    pg = holm({k: p_of(*v["inter"]) for k, v in Gv.items()})
    ps = holm({k: p_of(*v["survival"]) for k, v in Gv.items()})
    L.append(r"\paragraph{Class groups against all other classes.} Class fixed effects absorb a class "
             "group's survival level, so the groups are compared here with all other classes in a model with "
             "registration-year fixed effects only, the factors above and their lead interactions held. "
             "Survival is the group's difference from all other classes; the lead effects are within the "
             "group and within all other classes. The three composite groups (next section) are estimated "
             "on the half of owners that played no part in forming them.\n")
    L.append(r"{\small\begin{longtable}{>{\raggedright\arraybackslash}p{4.9cm}rrrrr}")
    L.append(r"\toprule & & & \multicolumn{3}{c}{Lead effect} \\ \cmidrule(l){4-6}"
             r" Class group & Share & Survival & In group & Other classes & Difference \\ \midrule \endhead")
    for k, v in Gv.items():
        L.append(f"{esc(k[0].upper() + k[1:])} & {100*v['share']:.0f}\\% & {num(v['survival'][0], ps[k])} & "
                 f"{est(*v['lead_in'])} & {est(*v['lead_out'])} & {num(v['inter'][0], pg[k])} \\\\")
    L.append(r"\bottomrule\end{longtable}}")

    # one-at-a-time comparison
    A = R["alone"]
    pma = holm({k: p_of(*A[k]["main"][:2]) for k in main_keys if A[k]["main"]})
    pia = holm({k: p_of(*A[k]["inter"][:2]) for k in main_keys})
    L.append(r"\paragraph{Each factor alone.} The survival and lead-interaction coefficients from models with "
             "lead, one factor and the controls (boom and bust together with the trend), against the joint "
             "model (Holm stars within column).")
    L.append(r"{\small\begin{longtable}{>{\raggedright\arraybackslash}p{6.0cm}rrrr}")
    L.append(r"\toprule & \multicolumn{2}{c}{Survival} & \multicolumn{2}{c}{Lead $\times$ factor} \\"
             r" \cmidrule(lr){2-3}\cmidrule(l){4-5} Factor & Alone & Joint & Alone & Joint \\ \midrule \endhead")
    pmj = holm({k: p_of(*J[k]["main"][:2]) for k in main_keys if J[k]["main"]})
    pij = holm({k: p_of(*J[k]["inter"][:2]) for k in main_keys})
    for k in main_keys:
        ma = num(A[k]["main"][0], pma[k]) if A[k]["main"] else "--"
        mj = num(J[k]["main"][0], pmj[k]) if J[k]["main"] else "--"
        L.append(f"{esc(lab[k])} & {ma} & {mj} & {num(A[k]['inter'][0], pia[k])} & "
                 f"{num(J[k]['inter'][0], pij[k])} \\\\")
    L.append(r"\bottomrule\end{longtable}}")

    # class groups
    G, S = R["class_groups"], R["class_slopes_halfA"]
    L.append(r"\section*{Classes that behave alike}")
    L.append(
        "Class-specific lead effects were estimated on a random half of owners (split by owner name), shrunk "
        "toward the pooled effect in proportion to their noise, and grouped into three by $k$-means. The "
        "groups' lead effects below are estimated on the other half of owners, which played no part in "
        "forming them.\n")
    L.append(r"{\small\begin{longtable}{>{\raggedright\arraybackslash}p{4.2cm}>{\raggedright\arraybackslash}p{7.0cm}r}")
    L.append(r"\toprule Group & Nice classes (shrunk lead effect, first half) & Lead effect, second half \\ \midrule \endhead")
    for g, (name, members) in enumerate(G.items()):
        hb = R["class_groups_halfB"][name]
        mem = ", ".join(f"{int(c)} ({S[c]['shrunk']:+.1f})".replace("-", "$-$") for c in
                        sorted(members, key=lambda c: S[c]['shrunk']))
        L.append(f"{esc(name.capitalize())} & {mem} & {num(hb['b'], p_of(hb['b'], hb['se']))} "
                 f"(s.e.\\ {hb['se']:.2f}) \\\\")
    L.append(r"\bottomrule\end{longtable}}")

    # lasso
    La, P = R["lasso"], R["post_lasso_top"]["terms"]
    top = len(P)
    L.append(r"\section*{Which factors explain where leading costs (LASSO)}")
    L.append(
        "With fixed effects, main effects and controls partialled out, a LASSO was fitted over every lead "
        "interaction: the factors above, the 45 classes and the three class groups (all standardized), on "
        f"{La['n_rows']:,} registrations from a random 40\\% of owners. The penalty was chosen by five-fold "
        "cross-validation with folds grouped by owner. Terms are listed in the order they enter as the "
        "penalty is relaxed, the usual ranking of explanatory value. At the cross-validated penalty "
        f"{len(La['selected_min'])} of the {len(keys) + 45 + 3} candidate terms are kept "
        f"({len(La['entry_order'])} enter anywhere on the path); the stricter one-standard-error "
        "rule keeps none, because for a pass/fail outcome the gain in out-of-sample fit is within the "
        f"fold-to-fold noise. The last column re-estimates the first {top} terms to enter, jointly, on all "
        "registrations with clustered errors (after selection, so the stars are optimistic).\n")
    L.append(r"{\small\begin{longtable}{r>{\raggedright\arraybackslash}p{8.2cm}r}")
    L.append(r"\toprule Order & Lead interaction & Re-estimated (pp) \\ \midrule \endhead")
    pp = holm({k: p_of(*v[:2]) for k, v in P.items()})
    for i, (k, _) in enumerate(La["entry_order"][:20], 1):
        nm = esc(lab.get(k, k[0].upper() + k[1:]))
        if k.startswith("class "):
            nm = f"Nice class {int(k[6:])}"
        v = P.get(k)
        cell = (num(v[0], pp[k]) + f" ({v[1]:.2f})") if v else ""
        L.append(f"{i} & {nm} & {cell} \\\\")
    L.append(r"\bottomrule\end{longtable}}")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
