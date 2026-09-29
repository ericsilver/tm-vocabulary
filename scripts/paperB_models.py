"""Paper B regression family: atypicality, lead, and the factors that change what each is worth.

Outcome: passing the five-year proof of continued use (percentage points),
registrations 2002-2018, one row per serial. Class x registration-year fixed
effects; standard errors clustered by owner (streaming estimator from
combined_model.py). Design controls in every column: log description length, log owner filing count,
foreign filing basis (44e, 66a).

Atypicality and lead enter as percentiles within class and registration year,
centred at the median, so a coefficient is the survival difference between the
top and the bottom of the class-year distribution. Atypicality is the global
T = 50 score (average KL divergence from the class's past and future theme
mix); under a global model it registers themes imported from other classes
(Paper A, validation section).

Columns
  (1) restricted     atypicality, lead, atypicality x lead
  (2) significant    (1) + the factors with at least one Holm-significant term in (3)
  (3) full           (1) + every factor: main effect, x atypicality, x lead
Every factor is centred, so the atypicality and lead coefficients are the
effects for a registration of average composition. Binary factors are 0/1;
continuous factors are in SD units. Stars: Holm-adjusted p within each panel of
each column.

Also: each factor alone (atypicality, lead, atypicality x lead, the factor and
its two interactions) for the magnitude figure; and renewal at year ten for
column (1).

Output: paper/results/paperB_models.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import combined_model as cm  # noqa: E402

PROC, RES = cm.PROC, cm.RES
DESIGN = ["log_len", "log_owner_n", "basis_44e", "basis_66a"]

# (key, label, source) -- source: record field / forced vocabulary / emergent theme / geography / time
FACTORS = [
    ("counsel", "Represented by counsel", "record"),
    ("patents", "Owner files patents", "record (PatentsView)"),
    ("debut", "Owner's first-ever filing", "record"),
    ("established", "Owner has 25+ earlier filings", "record"),
    ("itu", "Intent-to-use filing", "record"),
    ("foreign", "Foreign owner (not China)", "record"),
    ("china", "Owner in China", "record"),
    ("site", "Site-based offering (stores, restaurants, clinics)", "forced vocabulary"),
    ("contract", "Contract or subscription offering", "forced vocabulary"),
    ("platform", "Platform or network offering", "forced vocabulary"),
    ("gpt", "General-purpose-technology vocabulary", "forced vocabulary"),
    ("invention", "New-invention vocabulary", "forced vocabulary"),
    ("consumer", "Emergent consumer-category vocabulary", "forced vocabulary"),
    ("imported", "Primary theme imported from another class", "emergent theme"),
    ("volatility", "Theme-share volatility (per SD)", "emergent theme"),
    ("new_demand", "Theme growth from first-time filers (per SD)", "emergent theme"),
    ("colocation", "Theme's geographic co-location (per SD)", "geography"),
    ("has_hub", "Theme has a geographic hub", "geography"),
    ("in_hub", "Filer located in a hub", "geography"),
    ("tech_pace", "Class theme-mix turnover (per SD)", "emergent theme"),
    ("mkt_pace", "Class filing-volume growth (per SD)", "record"),
    ("trend", "Filing year (per SD, about 5 years)", "time"),
    ("boom", "Filed in a boom (1998-2000, 2005-07)", "time"),
    ("bust", "Filed in a bust (2001-02, 2008-09)", "time"),
]


def load() -> pl.DataFrame:
    d = cm.frame()
    keys = d.select("serial_number", "cls")
    parts = []
    for c in sorted(d["cls"].unique().to_list()):
        a = pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                            columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"])
        a = a.join(keys.filter(pl.col("cls") == c), on="serial_number", how="semi").unique("serial_number")
        parts.append(a.with_columns(pl.lit(c).alias("cls"),
                                    ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2)
                                    .alias("atyp_raw")).select("serial_number", "cls", "atyp_raw"))
    d = d.join(pl.concat(parts), on=["serial_number", "cls"], how="left")
    st = []
    for c in sorted(d["cls"].unique().to_list()):
        s = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "status_code"])
        st.append(s.join(keys.filter(pl.col("cls") == c), on="serial_number", how="semi"))
    d = d.join(pl.concat(st).unique("serial_number"), on="serial_number", how="left")
    return d.with_columns(
        ((pl.col("atyp_raw").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5)
        .fill_null(0.0).alias("atyp"),
        pl.when(pl.col("status_code") == "800").then(100.0)
        .when(pl.col("status_code").is_in(["710", "900"])).then(0.0).otherwise(None).alias("renew"))


def design(d: pl.DataFrame, facs: list[str]) -> tuple[np.ndarray, list[str], dict]:
    lead = d["lead"].to_numpy().astype(np.float64)
    atyp = d["atyp"].to_numpy().astype(np.float64)
    cols = [atyp, lead, atyp * lead]
    names = ["atyp", "lead", "atyp:lead"]
    means = {}
    for f in facs:
        v = d[f].to_numpy().astype(np.float64)
        means[f] = float(v.mean())
        vc = v - means[f]
        cols += [vc, vc * atyp, vc * lead]
        names += [f, f"{f}:atyp", f"{f}:lead"]
    for c in DESIGN + cm.MISSING:
        cols.append(d[c].cast(pl.Float64).fill_null(0).to_numpy()); names.append(c)
    return np.column_stack(cols).astype(np.float32), names, means


def p_of(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2)) if se else 1.0


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def fit(des, y, d, facs) -> dict:
    X, names, means = design(d, facs)
    r = des.ols(y, X, names)
    out = {"n": d.height, "means": means,
           "coef": {k: list(r[k][:2]) for k in names if k in r and k not in DESIGN + cm.MISSING}}
    # Holm within panels: main effects / x atyp / x lead (the three headline terms join their panels)
    panels = {"main": ["atyp", "lead"] + facs, "x_atyp": ["atyp:lead"] + [f"{f}:atyp" for f in facs],
              "x_lead": [f"{f}:lead" for f in facs]}
    out["holm"] = {}
    for pn, ks in panels.items():
        ks = [k for k in ks if k in out["coef"]]
        if ks:
            out["holm"].update(holm({k: p_of(*out["coef"][k]) for k in ks}))
    return out


def main() -> int:
    d = load()
    cm.log(f"[frame] {d.height:,}; peak {cm.peak_gb():.2f} GB")
    y = d["surv"].to_numpy()
    des = cm.Design(d)
    keys = [k for k, _, _ in FACTORS]
    R = {"factors": [{"key": k, "label": l, "source": s, "share": float(d[k].mean())} for k, l, s in FACTORS]}
    R["restricted"] = fit(des, y, d, [])
    cm.log("  (1) " + ", ".join(f"{k} {v[0]:+.2f} ({v[1]:.2f})" for k, v in R["restricted"]["coef"].items()))
    R["full"] = fit(des, y, d, keys)
    sig = [k for k in keys if any(R["full"]["holm"].get(t, 1) < 0.05 for t in (k, f"{k}:atyp", f"{k}:lead"))]
    R["significant_factors"] = sig
    R["significant"] = fit(des, y, d, sig)
    cm.log(f"  (3) full: atyp {R['full']['coef']['atyp'][0]:+.2f} lead {R['full']['coef']['lead'][0]:+.2f} "
           f"atyp:lead {R['full']['coef']['atyp:lead'][0]:+.2f}; significant factors: {sig}")
    # each factor alone
    alone = {}
    for k in keys:
        r = fit(des, y, d, [k])
        alone[k] = {t: r["coef"].get(t) for t in (k, f"{k}:atyp", f"{k}:lead")}
    R["alone"] = alone
    # renewal, restricted
    dr = d.filter((pl.col("surv") == 100) & pl.col("reg_year").is_between(2002, 2013) & pl.col("renew").is_not_null())
    R["renewal_restricted"] = fit(cm.Design(dr), dr["renew"].to_numpy(), dr, [])
    cm.log("  renewal (1): " + ", ".join(f"{k} {v[0]:+.2f}" for k, v in R["renewal_restricted"]["coef"].items()))
    # deciles for figures: survival adjusted for class x year, by atypicality decile x lead third
    d = d.with_columns(
        (pl.col("surv") - pl.col("surv").mean().over("cell") + pl.col("surv").mean()).alias("adj"),
        (((pl.col("atyp") + 0.5) * 10).floor().clip(0, 9)).cast(pl.Int8).alias("adec"),
        (((pl.col("lead") + 0.5) * 10).floor().clip(0, 9)).cast(pl.Int8).alias("ldec"),
        (((pl.col("lead") + 0.5) * 3).floor().clip(0, 2)).cast(pl.Int8).alias("lthird"))
    agg = lambda by: d.group_by(by).agg(pl.col("adj").mean().alias("m"), pl.col("adj").std().alias("s"),
                                        pl.len().alias("n")).sort(by).to_dicts()
    R["curves"] = {"atyp": agg(["adec"]), "lead": agg(["ldec"]), "atyp_by_leadthird": agg(["lthird", "adec"])}
    mp = d["mkt_pace"]
    d = d.with_columns(pl.when(pl.col("mkt_pace") >= mp.quantile(2 / 3)).then(pl.lit(2))
                       .when(pl.col("mkt_pace") <= mp.quantile(1 / 3)).then(pl.lit(0)).otherwise(pl.lit(1))
                       .cast(pl.Int8).alias("mthird"))
    R["curves"]["atyp_by_growth"] = agg(["mthird", "adec"])
    for k in ("patents", "counsel", "platform"):
        R["curves"][f"atyp_by_{k}"] = agg([k, "adec"])
    cq = d["colocation"]
    d = d.with_columns(pl.when(pl.col("colocation") >= cq.quantile(2 / 3)).then(pl.lit(2))
                       .when(pl.col("colocation") <= cq.quantile(1 / 3)).then(pl.lit(0)).otherwise(pl.lit(1))
                       .cast(pl.Int8).alias("cthird"))
    R["curves"]["atyp_by_colocation"] = agg(["cthird", "adec"])
    # year ranges behind the filing-year thirds (for labels)
    fy = d["fy"]
    R["fy_thirds"] = [int(fy.quantile(1 / 3)), int(fy.quantile(2 / 3))]
    (RES / "paperB_models.json").write_text(json.dumps(R, indent=1))
    cm.log(f"[done] peak {cm.peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
