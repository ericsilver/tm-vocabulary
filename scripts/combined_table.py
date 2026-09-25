"""LaTeX for the combined model (paper/results/combined_model.json).

Output: paper/split/B_brief/combined_results.tex
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
OUT = REPO / "paper" / "split" / "B_brief" / "combined_results.tex"

BINARY = ["counsel", "patents", "itu", "debut", "established", "foreign", "china", "site", "contract",
          "platform", "invention", "gpt", "consumer", "imported", "has_hub", "in_hub", "boom", "bust"]
CONT = ["volatility", "new_demand", "colocation", "tech_pace", "mkt_pace"]
CG = ["services", "pharma", "cpg", "manufactured", "regulated"]
SECTIONS = [("The owner and the filing", ["counsel", "patents", "debut", "established", "foreign", "china", "itu"]),
            ("The offering", ["site", "contract", "platform", "invention", "gpt", "consumer", "imported"]),
            ("The theme and its geography", ["volatility", "new_demand", "colocation", "has_hub", "in_hub"]),
            ("The class and the year", ["tech_pace", "mkt_pace", "boom", "bust"]),
            ("Class groups (main effect absorbed by class fixed effects)", CG)]


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


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%")


def main() -> int:
    R = json.loads((RES / "combined_model.json").read_text())
    J, lab, sh = R["joint"], R["labels"], R["shares"]
    lead_b, lead_se = J["lead"][0], J["lead"][1]
    keys = BINARY + CONT + CG
    # lead effect for filings with the factor
    def lead_at(k, w):
        bi, si, ci = J[k]["inter"]
        v = lead_se ** 2 + w ** 2 * si ** 2 + 2 * w * ci
        return (lead_b + w * bi, math.sqrt(max(v, 0)))

    # lead effect among filings with / without the factor (continuous: +1 / -1 SD)
    cond = {k: lead_at(k, (1 - J["means"][k]) if k in BINARY + CG else 1.0) for k in keys}
    uncond = {k: lead_at(k, -J["means"][k] if k in BINARY + CG else -1.0) for k in keys}
    pm = holm({k: p_of(*J[k]["main"][:2]) for k in keys if J[k]["main"]})
    pi = holm({k: p_of(*J[k]["inter"][:2]) for k in keys})
    pc = holm({k: p_of(*cond[k]) for k in keys})
    pu = holm({k: p_of(*uncond[k]) for k in keys})
    L = [r"\section*{Where being leading costs: one model}"]
    L.append(
        "One linear probability model of surviving the five-year proof (percentage points), with class "
        "$\\times$ registration-year fixed effects and standard errors clustered by owner; "
        f"{R['n']:,} registrations, 2002--2018. \\emph{{Lead}} is the filing's percentile of lead within its "
        "class and year, so a lead coefficient is the survival difference between the most leading and the "
        "most lagging filing (negative: leading costs). Every factor enters as a main effect and as an "
        "interaction with lead, all at once. \\emph{Survival} is the factor's association with survival at "
        "median lead. \\emph{Lead $\\times$ factor} is how much the factor changes the lead effect. "
        "\\emph{With} and \\emph{without} give the lead effect among filings that have and lack the factor "
        "(continuous factors: one SD above and below the mean). Stars: Holm-adjusted $p$ within each column, "
        "$^{*}<0.05$, $^{**}<0.01$, $^{***}<0.001$. Controls: description length, owner filing count, "
        "foreign filing basis, missing-data flags.\n")
    def share_of(k):
        if k not in sh:
            return "SD"
        return f"{100*sh[k]:.1f}\\%" if sh[k] < 0.01 else f"{100*sh[k]:.0f}\\%"

    L.append(r"{\small\begin{longtable}{>{\raggedright\arraybackslash}p{5.4cm}rrrrr}")
    L.append(r"\toprule & & & & \multicolumn{2}{c}{Lead effect} \\ \cmidrule(l){5-6}"
             r" Factor & Share & Survival & Lead $\times$ factor & With & Without \\ \midrule \endhead")
    L.append(f"\\textbf{{Lead, average filing}} & & & & \\multicolumn{{2}}{{c}}{{{num(lead_b, p_of(lead_b, lead_se))}}} \\\\")
    L.append(f"\\quad (lead alone, no factors) & & & & \\multicolumn{{2}}{{c}}{{{num(R['lead_only']['b'], p_of(R['lead_only']['b'], R['lead_only']['se']))}}} \\\\")
    for title, ks in SECTIONS:
        L.append(f"\\addlinespace\\multicolumn{{6}}{{l}}{{\\emph{{{esc(title)}}}}} \\\\")
        for k in ks:
            main = num(J[k]["main"][0], pm[k]) if J[k]["main"] else "--"
            L.append(f"{esc(lab[k])} & {share_of(k)} & {main} & {num(J[k]['inter'][0], pi[k])} & "
                     f"{num(cond[k][0], pc[k])} & {num(uncond[k][0], pu[k])} \\\\")
    L.append(r"\bottomrule\end{longtable}}")

    # one-at-a-time comparison
    A = R["alone"]
    pma = holm({k: p_of(*A[k]["main"][:2]) for k in keys if A[k]["main"]})
    pia = holm({k: p_of(*A[k]["inter"][:2]) for k in keys})
    L.append(r"\paragraph{Each factor alone.} The same two coefficients from models with lead, one factor and "
             "the controls, against the joint model above (Holm stars within column).")
    L.append(r"{\small\begin{longtable}{>{\raggedright\arraybackslash}p{6.0cm}rrrr}")
    L.append(r"\toprule & \multicolumn{2}{c}{Survival} & \multicolumn{2}{c}{Lead $\times$ factor} \\"
             r" \cmidrule(lr){2-3}\cmidrule(l){4-5} Factor & Alone & Joint & Alone & Joint \\ \midrule \endhead")
    for k in keys:
        ma = num(A[k]["main"][0], pma[k]) if A[k]["main"] else "--"
        mj = num(J[k]["main"][0], pm[k]) if J[k]["main"] else "--"
        L.append(f"{esc(lab[k])} & {ma} & {mj} & {num(A[k]['inter'][0], pia[k])} & "
                 f"{num(J[k]['inter'][0], pi[k])} \\\\")
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
