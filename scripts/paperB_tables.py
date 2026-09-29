"""Tables for Paper B (v2): the regression table, the factor definitions, the geography validation.

Output: paper/split/B_v2/tab_regression.tex, tab_factors.tex, tab_geo.tex
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
OUT = REPO / "paper" / "split" / "B_v2"

DEFS = {
    "counsel": "An attorney of record on the application.",
    "patents": "The owner, matched by name, holds US patents (PatentsView).",
    "debut": "The owner's first filing in the record.",
    "established": "The owner had filed 25 or more marks before this one.",
    "itu": "Filed on an intent-to-use basis rather than on use in commerce.",
    "foreign": "Owner domiciled outside the US, other than China.",
    "china": "Owner domiciled in China.",
    "site": "Description names a physical site: store, restaurant, hotel, clinic, salon, gym and similar.",
    "contract": "Description names a subscription, membership, account, insurance, banking or leasing.",
    "platform": "Description names a platform, marketplace, social network, online community or peer-to-peer service.",
    "gpt": "Internet, mobile-app, cloud, social-media, artificial-intelligence or blockchain vocabulary.",
    "invention": "3D printing, drone, virtual-reality, electric-vehicle, gene-therapy or nanotechnology vocabulary.",
    "consumer": "E-cigarette, energy-drink, kombucha, cold-brew, plant-milk, hard-seltzer or CBD vocabulary.",
    "imported": "The filing's primary theme held under 1\\% of its class in the five prior years and at least 5\\% of another class.",
    "volatility": "Year-to-year variability of the primary theme's share of its class, over the filing year $\\pm 5$.",
    "new_demand": "Share of the primary theme's filings in its class over the next five years made by first-time filers.",
    "colocation": "Co-location index of the class's US filers of the same primary theme within two years of the filing (Section 7).",
    "has_hub": "That group of filers has at least one hub (Section 7).",
    "in_hub": "The filer is located in such a hub.",
    "tech_pace": "Change in the class's theme mix between the five years before and after the filing year.",
    "mkt_pace": "Growth in the class's filing volume, five years after against five years before.",
    "trend": "Filing year.",
    "boom": "Filed in 1998--2000 or 2005--07.",
    "bust": "Filed in 2001--02 or 2008--09.",
}
SRC = {"counsel": "Record", "patents": "Record", "debut": "Record", "established": "Record", "itu": "Record",
       "foreign": "Record", "china": "Record", "site": "Forced", "contract": "Forced", "platform": "Forced",
       "gpt": "Forced", "invention": "Forced", "consumer": "Forced", "imported": "Emergent",
       "volatility": "Emergent", "new_demand": "Emergent", "tech_pace": "Emergent", "colocation": "Geography",
       "has_hub": "Geography", "in_hub": "Geography", "mkt_pace": "Record", "trend": "Record", "boom": "Record",
       "bust": "Record"}
ORDER = ["counsel", "patents", "debut", "established", "itu", "foreign", "china", "site", "contract", "platform",
         "gpt", "invention", "consumer", "imported", "volatility", "new_demand", "tech_pace", "colocation",
         "has_hub", "in_hub", "mkt_pace", "trend", "boom", "bust"]


def st(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def cell(m, k):
    if k not in m["coef"]:
        return ""
    b, se = m["coef"][k]
    return f"{b:+.2f}".replace("-", "$-$") + st(m["holm"].get(k, 1)) + f" ({se:.2f})"


def main() -> int:
    R = json.loads((RES / "paperB_models.json").read_text())
    lab = {f["key"]: f["label"] for f in R["factors"]}
    share = {f["key"]: f["share"] for f in R["factors"]}
    M = [R["restricted"], R["significant"], R["full"]]
    L = [r"\begin{small}", r"\begin{longtable}{>{\raggedright\arraybackslash}p{5.9cm}rrr}",
         r"\caption{Passing the five-year proof: atypicality, lead and the factors that change what each is worth}\label{tab:reg}\\",
         r"\toprule & (1) Atypicality & (2) With & (3) With all \\ & and lead only & significant factors & factors \\ \midrule",
         r"\endfirsthead",
         r"\toprule & (1) & (2) & (3) \\ \midrule \endhead",
         r"\multicolumn{4}{l}{\textbf{A. Survival difference (percentage points)}} \\",
         f"Atypicality (most unusual $-$ most typical) & {cell(M[0],'atyp')} & {cell(M[1],'atyp')} & {cell(M[2],'atyp')} \\\\",
         f"Lead (most leading $-$ most lagging) & {cell(M[0],'lead')} & {cell(M[1],'lead')} & {cell(M[2],'lead')} \\\\"]
    for k in ORDER:
        L.append(f"\\quad {lab[k]} & & {cell(M[1], k)} & {cell(M[2], k)} \\\\")
    L.append(r"\addlinespace\multicolumn{4}{l}{\textbf{B. Change in the value of atypicality (points)}} \\")
    L.append(f"Lead & {cell(M[0],'atyp:lead')} & {cell(M[1],'atyp:lead')} & {cell(M[2],'atyp:lead')} \\\\")
    for k in ORDER:
        L.append(f"\\quad {lab[k]} & & {cell(M[1], k + ':atyp')} & {cell(M[2], k + ':atyp')} \\\\")
    L.append(r"\addlinespace\multicolumn{4}{l}{\textbf{C. Change in the value of lead (points)}} \\")
    for k in ORDER:
        L.append(f"\\quad {lab[k]} & & {cell(M[1], k + ':lead')} & {cell(M[2], k + ':lead')} \\\\")
    L.append(r"\addlinespace\midrule")
    L.append(f"Registrations & {M[0]['n']:,} & {M[1]['n']:,} & {M[2]['n']:,} \\\\")
    L.append(f"Factors & 0 & {len(R['significant_factors'])} & {len(ORDER)} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\multicolumn{4}{p{14.2cm}}{\footnotesize Linear probability models of passing the five-year proof of "
             r"continued use, registrations 2002--2018. Atypicality and lead are percentiles within Nice class and "
             r"registration year, centred, so a coefficient in panel A is the difference between the top and the bottom "
             r"of the class-and-year distribution. Factors are centred (binary factors 0/1, continuous factors in SD "
             r"units), so the atypicality and lead rows are the effects for a registration of average composition. Panel B "
             r"reports how much a factor adds to or subtracts from the atypicality effect; panel C, from the lead effect. "
             r"Column 2 keeps the factors with at least one term significant in column 3. All models: class $\times$ "
             r"registration-year fixed effects; log description length, log owner filing count and foreign-basis flags; "
             r"standard errors clustered by owner in parentheses. Stars: Holm-adjusted $p$ within each panel of each "
             r"column; $^{*}<0.05$, $^{**}<0.01$, $^{***}<0.001$.} \\")
    L.append(r"\end{longtable}")
    L.append(r"\end{small}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_regression.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    F = [r"\begin{small}", r"\begin{longtable}{>{\raggedright\arraybackslash}p{4.3cm}lr>{\raggedright\arraybackslash}p{6.6cm}}",
         r"\caption{The factors, where each comes from, and how common it is}\label{tab:factors}\\",
         r"\toprule Factor & Source & Share & Definition \\ \midrule \endfirsthead",
         r"\toprule Factor & Source & Share & Definition \\ \midrule \endhead"]
    for k in ORDER:
        s = share[k]
        sh = "SD" if k in ("volatility", "new_demand", "colocation", "tech_pace", "mkt_pace", "trend") else (
            f"{100*s:.1f}\\%" if s < 0.01 else f"{100*s:.0f}\\%")
        F.append(f"{lab[k]} & {SRC[k]} & {sh} & {DEFS[k]} \\\\")
    F.append(r"\bottomrule")
    F.append(r"\multicolumn{4}{p{14.2cm}}{\footnotesize Record: a field of the filing or the owner's filing history. "
             r"Forced: a curated list of terms matched in the description (Appendix~\ref{app:vocab}). Emergent: derived "
             r"from the fifty themes the topic model found without guidance. Geography: from the applicant's ZIP code. "
             r"Share: percentage of the 3.10 million registrations with the factor; SD marks continuous factors.} \\")
    F.append(r"\end{longtable}\end{small}")
    (OUT / "tab_factors.tex").write_text("\n".join(F) + "\n", encoding="utf-8")

    G = json.loads((RES / "geo_cluster_validation.json").read_text())
    T = [r"\begin{table}[htbp]\centering\small",
         r"\caption{The co-location measure on groups with known geography}\label{tab:geo}",
         r"\begin{tabular}{>{\raggedright\arraybackslash}p{4.8cm}rrr>{\raggedright\arraybackslash}p{5.4cm}}",
         r"\toprule Filers using the vocabulary of & Filers & $L$ & & Hubs: share of the group nearby / share of all filers \\ \midrule"]
    for k, v in G.items():
        hubs = "; ".join(f"{h['name']} {100*h['share_nearby']:.0f}\\% / {100*h['class_share_nearby']:.0f}\\%"
                         for h in v["hubs"]) or "none"
        T.append(f"{k.replace('&', '')} & {v['n']:,} & {v['L']:.3f} & & {hubs} \\\\")
    T.append(r"\bottomrule\end{tabular}")
    T.append(r"\par\footnotesize $L$: co-location of the group net of all filers in the same classes and years (0 = "
             r"placed like the classes, 1 = all in one place). Nearby: kernel-weighted within 200 miles.")
    T.append(r"\end{table}")
    (OUT / "tab_geo.tex").write_text("\n".join(T) + "\n", encoding="utf-8")
    print("tables written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
