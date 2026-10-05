"""Trademark entry and producer-price inflation (descriptive).

Step 2. Industry-year panels for the good/partial PPI mappings in
paper/results/bls_concordance.csv:
  * class panel   : filings and first-time filers per Nice class and filing year
                    (data/processed/tm_class{cls}.parquet), 1985-2025
  * segment panel : the same per product segment (data/processed/theme_segments.parquet,
                    registrations 2002-2018 only -> filing years 2003-2016 used)
  pi_{i,t}  = log change of the annual-average PPI
  g_{i,t}   = log change of filings (or of first-time filers)
  Forward : pi_{i,t+h} = b g_{i,t} + a_i + d_t   (h = 1, 2, 3; and the 3-year average)
  Reverse : g_{i,t}    = b pi_{i,t-h} + a_i + d_t
  SEs clustered by industry (unit). Year effects absorb economy-wide inflation, so
  nominal and real inflation give identical slopes.

Step 3. Powder coatings: filings whose goods/services match
  powder[- ]coat|powder paint|coating powder  (classes 001, 002, 040)
  against paint-and-coatings PPIs deflated by CPI-U, 1995-2024.

Output: paper/results/bls_entry_prices.json, paper/results/bls_entry_panel.csv,
        paper/results/bls_powder_coatings.csv
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "paper" / "results"

SUFFIXES = (r"\b(INC|INCORPORATED|LLC|L\.L\.C|LTD|LIMITED|CORP|CORPORATION|"
            r"COMPANY|CO|LP|LLP|PLC|GMBH|SA|NV|BV|AB|AG|AS|OY|SPA|SRL|PTY|"
            r"KK|KABUSHIKI KAISHA)\b")
POWDER_RX = r"(?i)powder[- ]coat|powder paint|coating powder"
POWDER_RX_WIDE = r"(?i)powder[- ]?coat|powder paint|coating powder"
LAST_YEAR = 2025


def norm_owner(col: str) -> pl.Expr:
    return (pl.col(col).str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9 ]", " ")
            .str.replace_all(SUFFIXES, "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars())


def read_class(cls: str, extra: tuple[str, ...] = ()) -> pl.DataFrame:
    return (pl.scan_parquet(PROC / f"tm_class{cls}.parquet")
            .select("serial_number", "filing_date", "owner_name", *extra)
            .with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"))
            .filter(pl.col("year").is_between(1870, LAST_YEAR))
            .unique("serial_number")
            .with_columns(norm_owner("owner_name").alias("owner"))
            .with_columns(pl.when(pl.col("owner") == "").then(None).otherwise(pl.col("owner")).alias("owner"))
            .collect())


def entry_counts(df: pl.DataFrame, unit_col: str) -> pl.DataFrame:
    """Filings and first-time filers (owners with no earlier filing in the unit) per unit-year."""
    n = df.group_by(unit_col, "year").agg(pl.len().alias("filings"))
    first = (df.drop_nulls("owner").group_by(unit_col, "owner").agg(pl.col("year").min())
             .group_by(unit_col, "year").agg(pl.len().alias("first_filers")))
    return n.join(first, on=[unit_col, "year"], how="left").with_columns(pl.col("first_filers").fill_null(0))


def add_growth(p: pl.DataFrame) -> pl.DataFrame:
    p = p.sort("unit", "year")
    lg = lambda c: pl.when(pl.col(c) > 0).then(pl.col(c).cast(pl.Float64).log()).otherwise(None)
    p = p.with_columns(lg("filings").alias("lf"), lg("first_filers").alias("lff"), pl.col("ppi").log().alias("lp"))
    # contiguous years only
    prev_ok = (pl.col("year") - pl.col("year").shift(1).over("unit")) == 1
    d = lambda c: pl.when(prev_ok).then(pl.col(c) - pl.col(c).shift(1).over("unit")).otherwise(None)
    p = p.with_columns(d("lf").alias("g_filings"), d("lff").alias("g_first"), d("lp").alias("pi"))
    for h in (1, 2, 3):
        p = p.with_columns(
            pl.col("pi").shift(-h).over("unit").alias(f"pi_f{h}"),
            pl.col("pi").shift(h).over("unit").alias(f"pi_l{h}"),
        )
    p = p.with_columns(((pl.col("lp").shift(-3).over("unit") - pl.col("lp")) / 3).alias("pi_f123"),
                       pl.col("g_filings").shift(1).over("unit").alias("g_filings_l1"),
                       pl.col("g_first").shift(1).over("unit").alias("g_first_l1"))
    return p


def fe_ols(df: pl.DataFrame, y: str, x: str, controls: tuple[str, ...] = ()) -> dict:
    cols = ["unit", "year", y, x, *controls]
    d = df.select(cols).drop_nulls().to_pandas()
    d = d[np.isfinite(d[[y, x, *controls]]).all(axis=1)]
    if len(d) < 30 or d["unit"].nunique() < 5:
        return {"y": y, "x": x, "n": len(d), "note": "too few obs"}
    rhs = " + ".join([x, *controls, "C(unit)", "C(year)"])
    m = smf.ols(f"{y} ~ {rhs}", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["unit"].astype("category").cat.codes})
    sd = float(d[x].std())
    return {"y": y, "x": x, "controls": list(controls), "b": float(m.params[x]), "se": float(m.bse[x]),
            "p": float(m.pvalues[x]), "n": int(m.nobs), "units": int(d["unit"].nunique()),
            "years": [int(d["year"].min()), int(d["year"].max())], "sd_x": sd,
            "effect_1sd_pp": float(m.params[x]) * sd * 100}


def run_panel(p: pl.DataFrame, t_range: tuple[int, int]) -> dict:
    p = p.filter(pl.col("year").is_between(*t_range))
    out: dict = {"forward": [], "forward_controls": [], "forward_winsor": [], "reverse": [], "contemporaneous": []}
    # winsorized growth (1st/99th pct pooled)
    for g in ("g_filings", "g_first"):
        lo, hi = p[g].quantile(0.01), p[g].quantile(0.99)
        p = p.with_columns(pl.col(g).clip(lo, hi).alias(f"{g}_w"))
    for g in ("g_filings", "g_first"):
        for y in ("pi_f1", "pi_f2", "pi_f3", "pi_f123"):
            out["forward"].append(fe_ols(p, y, g))
            out["forward_controls"].append(fe_ols(p, y, g, ("pi", f"{g}_l1")))
            out["forward_winsor"].append(fe_ols(p, y, f"{g}_w"))
        for h in (1, 2, 3):
            out["reverse"].append(fe_ols(p, g, f"pi_l{h}"))
        out["contemporaneous"].append(fe_ols(p, g, "pi"))
    return out


def class_panel(conc: pl.DataFrame, ppi: pl.DataFrame) -> pl.DataFrame:
    prim = conc.filter((pl.col("unit_type") == "class") & (pl.col("role") == "primary"))
    parts = []
    for r in prim.iter_rows(named=True):
        df = read_class(r["unit"]).with_columns(pl.lit(r["unit"]).alias("unit"))
        e = entry_counts(df, "unit").filter(pl.col("year") >= 1980)
        parts.append(e.with_columns(pl.lit(r["series_id"]).alias("series_id"), pl.lit(r["fit"]).alias("fit")))
        del df
    e = pl.concat(parts)
    return add_growth(e.join(ppi, on=["series_id", "year"], how="inner"))


def segment_panel(conc: pl.DataFrame, ppi: pl.DataFrame) -> pl.DataFrame:
    segm = conc.filter((pl.col("unit_type") == "segment") & (pl.col("series_id") != ""))
    seg2name = {}
    for r in segm.iter_rows(named=True):
        for s in str(r["segments"]).split(";"):
            seg2name[int(s)] = r["unit"]
    ts = pl.read_parquet(PROC / "theme_segments.parquet").filter(pl.col("seg").is_in(list(seg2name)))
    ts = ts.with_columns(pl.col("seg").replace_strict(seg2name, return_dtype=pl.Utf8).alias("unit"))
    parts = []
    for cls in sorted(ts["cls"].unique().to_list()):
        sub = ts.filter(pl.col("cls") == cls).select("serial_number", "unit")
        df = read_class(cls).select("serial_number", "year", "owner").join(sub, on="serial_number", how="inner")
        parts.append(df)
    df = pl.concat(parts).unique(["serial_number", "unit"])
    e = entry_counts(df, "unit")
    meta = segm.select("unit", "series_id", "fit")
    e = e.join(meta, on="unit")
    return add_growth(e.join(ppi, on=["series_id", "year"], how="inner"))


def powder(ppi: pl.DataFrame) -> dict:
    parts, totals = [], []
    for cls in ("001", "002", "040"):
        df = read_class(cls, ("goods_services",))
        totals.append(df.select("serial_number", "year"))
        parts.append(df.filter(pl.col("goods_services").str.contains(POWDER_RX_WIDE))
                     .with_columns(pl.col("goods_services").str.contains(POWDER_RX).alias("strict"),
                                   pl.lit(cls).alias("cls")))
        del df
    allf = pl.concat(parts)
    by_cls = allf.filter("strict").group_by("cls").agg(pl.col("serial_number").n_unique().alias("n")).sort("cls")
    tot = pl.concat(totals).unique("serial_number").group_by("year").agg(pl.len().alias("all_filings_001_002_040"))
    wide_extra = allf.unique("serial_number").height - allf.filter("strict").unique("serial_number").height
    pw = allf.filter("strict").unique("serial_number")
    first = (pw.drop_nulls("owner").group_by("owner").agg(pl.col("year").min())
             .group_by("year").agg(pl.len().alias("first_filers")))
    ts = (pw.group_by("year").agg(pl.len().alias("filings"))
          .join(first, on="year", how="full", coalesce=True)
          .join(tot, on="year", how="right", coalesce=True)
          .fill_null(0).sort("year").filter(pl.col("year").is_between(1985, LAST_YEAR)))
    series = {"PCU325510325510": "paint_coating_mfg", "WPU06210201": "oem_finishes",
              "PCU332812332812": "metal_coating_services", "CUUR0000SA0": "cpi"}
    w = (ppi.filter(pl.col("series_id").is_in(list(series)))
         .with_columns(pl.col("series_id").replace_strict(series))
         .pivot(on="series_id", index="year", values="ppi"))
    ts = ts.join(w, on="year", how="left").sort("year")
    ts = ts.with_columns(
        (pl.col("filings").log() - pl.col("filings").shift(1).log()).alias("g_filings"),
        (pl.col("first_filers").log() - pl.col("first_filers").shift(1).log()).alias("g_first"),
        ((pl.col("filings") / pl.col("all_filings_001_002_040")).log()
         - (pl.col("filings") / pl.col("all_filings_001_002_040")).shift(1).log()).alias("g_share"),
        pl.col("cpi").log().diff().alias("cpi_infl"),
    )
    for s in ("paint_coating_mfg", "oem_finishes", "metal_coating_services"):
        ts = ts.with_columns((pl.col(s).log().diff() - pl.col("cpi_infl")).alias(f"real_{s}"))
        ts = ts.with_columns(
            pl.col(f"real_{s}").shift(-1).alias(f"real_{s}_f1"),
            ((pl.col(s).log().shift(-3) - pl.col(s).log()) / 3
             - (pl.col("cpi").log().shift(-3) - pl.col("cpi").log()) / 3).alias(f"real_{s}_f123"))
    ts.write_csv(RES / "bls_powder_coatings.csv")

    win = ts.filter(pl.col("year").is_between(1995, 2024))
    res: dict = {"regex": POWDER_RX, "filings_by_class": by_cls.to_dicts(),
                 "extra_filings_if_powdercoat_unspaced_included": wide_extra,
                 "total_powder_filings_1995_2024": int(win["filings"].sum()),
                 "total_first_filers_1995_2024": int(win["first_filers"].sum()),
                 "surge": {}, "ts_regression": {}}
    d = win.to_pandas().replace([np.inf, -np.inf], np.nan)
    for gvar in ("g_filings", "g_first", "g_share"):
        thr = float(d[gvar].quantile(0.75))
        d["surge"] = d[gvar] >= thr
        res["surge"][gvar] = {"threshold_log_growth": thr,
                              "surge_years": [int(y) for y in d.loc[d["surge"], "year"]]}
        for s in ("paint_coating_mfg", "oem_finishes", "metal_coating_services"):
            for y in (f"real_{s}_f1", f"real_{s}_f123"):
                dd = d[["year", "surge", y]].dropna()
                a, b = dd.loc[dd["surge"], y], dd.loc[~dd["surge"], y]
                from scipy import stats
                t = stats.ttest_ind(a, b, equal_var=False)
                res["surge"][gvar][y] = {"mean_after_surge_pp": float(a.mean() * 100), "n_surge": int(len(a)),
                                         "mean_other_pp": float(b.mean() * 100), "n_other": int(len(b)),
                                         "diff_pp": float((a.mean() - b.mean()) * 100), "welch_p": float(t.pvalue)}
                dr = d[["year", gvar, y]].dropna()
                X = sm.add_constant(dr[[gvar]])
                m = sm.OLS(dr[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
                res["ts_regression"].setdefault(gvar, {})[y] = {
                    "b": float(m.params[gvar]), "se": float(m.bse[gvar]), "p": float(m.pvalues[gvar]),
                    "n": int(m.nobs), "years": [int(dr["year"].min()), int(dr["year"].max())]}
    # robustness: drop entry years 2008 and 2021, whose following years (2009, 2022) carry
    # economy-wide commodity/inflation shocks to paint input costs; Spearman rank correlation
    from scipy import stats as _st
    for gvar in ("g_filings", "g_first"):
        for s in ("paint_coating_mfg", "oem_finishes"):
            y = f"real_{s}_f1"
            dr = d.loc[~d["year"].isin([2008, 2021]), ["year", gvar, y]].dropna()
            m = sm.OLS(dr[y], sm.add_constant(dr[[gvar]])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            sp = _st.spearmanr(dr[gvar], dr[y])
            res.setdefault("ts_regression_excl_2008_2021", {}).setdefault(gvar, {})[y] = {
                "b": float(m.params[gvar]), "se": float(m.bse[gvar]), "p": float(m.pvalues[gvar]), "n": int(m.nobs),
                "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue)}
    # reverse timing: powder entry growth on lagged real paint inflation
    for s in ("paint_coating_mfg", "oem_finishes"):
        dr = d[["year", "g_filings", f"real_{s}"]].copy()
        dr["lag"] = dr[f"real_{s}"].shift(1)
        dr = dr.dropna()
        m = sm.OLS(dr["g_filings"], sm.add_constant(dr[["lag"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        res.setdefault("reverse_ts", {})[s] = {"b": float(m.params["lag"]), "se": float(m.bse["lag"]),
                                               "p": float(m.pvalues["lag"]), "n": int(m.nobs)}
    res["cumulative_real_growth_1995_2024_pct"] = {
        s: float((math.log(win.filter(pl.col("year") == 2024)[s][0] / win.filter(pl.col("year") == 1995)[s][0])
                  - math.log(win.filter(pl.col("year") == 2024)["cpi"][0] / win.filter(pl.col("year") == 1995)["cpi"][0])) * 100)
        for s in ("paint_coating_mfg", "oem_finishes", "metal_coating_services")}
    res["table"] = win.select("year", "filings", "first_filers", "all_filings_001_002_040", "g_filings",
                              "paint_coating_mfg", "oem_finishes", "metal_coating_services", "cpi",
                              "real_paint_coating_mfg", "real_oem_finishes", "real_metal_coating_services").to_dicts()
    return res


def main() -> None:
    conc = pl.read_csv(RES / "bls_concordance.csv", infer_schema_length=0)
    conc = conc.filter(pl.col("fit").is_in(["good", "partial"]))
    ppi = pl.read_parquet(PROC / "bls_ppi_annual.parquet")
    out: dict = {"notes": "Descriptive. Two-way FE (unit, year), SEs clustered by unit. pi = log change of annual-average PPI; "
                          "g = log change of filings / first-time filers. effect_1sd_pp = b * sd(x) * 100 (percentage points)."}

    cp = class_panel(conc, ppi)
    cp.write_csv(RES / "bls_entry_panel.csv")
    out["class_panel"] = {"units": cp["unit"].n_unique(), "t_range": [1990, 2022],
                          "all": run_panel(cp, (1990, 2022)),
                          "good_only": run_panel(cp.filter(pl.col("fit") == "good"), (1990, 2022)),
                          "goods_classes_001_034": run_panel(cp.filter(pl.col("unit") <= "034"), (1990, 2022))}
    print("class panel done", flush=True)
    sp = segment_panel(conc, ppi)
    sp.with_columns(pl.lit("segment").alias("panel")).write_csv(RES / "bls_entry_panel_segments.csv")
    out["segment_panel"] = {"units": sp["unit"].n_unique(), "t_range": [2003, 2016],
                            "all": run_panel(sp, (2003, 2016))}
    print("segment panel done", flush=True)
    out["powder"] = powder(ppi)
    (RES / "bls_entry_prices.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    def show(lst):
        for r in lst:
            if "b" in r:
                print(f"  {r['y']:>10} ~ {r['x']:<12} b={r['b']:+.4f} se={r['se']:.4f} p={r['p']:.3f} n={r['n']} "
                      f"units={r['units']} 1sd={r['effect_1sd_pp']:+.2f}pp")
            else:
                print("  ", r)
    for k in ("class_panel", "segment_panel"):
        for sub in [s for s in out[k] if isinstance(out[k][s], dict)]:
            print(f"== {k} / {sub}")
            for blk in ("forward", "forward_controls", "reverse", "contemporaneous"):
                print(f" [{blk}]"); show(out[k][sub][blk])
    pw = out["powder"]
    print(json.dumps({k: v for k, v in pw.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    main()
