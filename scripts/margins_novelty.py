"""Test 1. Do firms whose trademark filings are unusual or early earn higher margins?

Panel: SEC-matched trademark owners (uspto_sec_crosswalk, owner names normalized
with gate_decisive_regression.norm_owner on both sides), firm (CIK) x fiscal year t.

  Filing scores (rolling_surprise_class{cls}): atypicality A = (kl_past + kl_future)/2,
  lead L = kl_past - kl_future (= topic_dkl). Each is converted to a percentile within
  class x filing year among ALL scored filings in that cell, (rank - 0.5)/n in (0, 1).
  A serial filed in several classes is averaged over its class rows.
  Portfolio for (cik, t): mean percentile over the firm's scored filings dated t-3..t,
  and the number of those filings.
  Outcome: gross margin in t, in percentage points (rows with gross margin outside
  [-1, 1] or revenue <= 0 dropped); second outcome operating margin =
  operating_income / revenue, same [-1, 1] trim.

Fiscal years: 2009-2019 in the main sample (scores end with filing year 2019, so later
windows are partly unscored); 2009-2022 reported as a check.

Specs (OLS, firm-clustered SEs, regressors A_pct and L_pct in 0-1 units, controls
log revenue and log filings in window):
  a  SIC2 x year fixed effects
  b  SIC2 x year + firm fixed effects (within-firm changes)
Fixed effects are absorbed by alternating-projection demeaning; firm FE are nested in
the clusters, so no df correction is needed for them.

Check: firm-years with 5+ filings in the window (t 2009-2019).

Output: paper/results/margins_novelty.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from gate_decisive_regression import norm_owner  # noqa: E402

CLASSES = [f"{i:03d}" for i in range(1, 46)]
FILING_LO = 2006            # earliest filing year that enters a t >= 2009 window
T_LO, T_HI_MAIN, T_HI_EXT = 2009, 2019, 2022


def free_gb() -> float:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2**30
    except Exception:
        return 99.0


def mem_wait(min_gb: float = 4.0) -> None:
    while (g := free_gb()) < min_gb:
        print(f"[mem] {g:.1f} GB free, waiting", file=sys.stderr, flush=True)
        time.sleep(60)


def crosswalk() -> pl.DataFrame:
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet", columns=["owner_name", "cik"]).with_columns(
        norm_owner(pl.col("owner_name")).alias("okey")).filter(pl.col("okey") != "")
    amb = cw.group_by("okey").agg(pl.col("cik").n_unique().alias("k")).filter(pl.col("k") > 1)
    return cw.join(amb, on="okey", how="anti").select("okey", "cik").unique()


def filings(cw: pl.DataFrame) -> pl.DataFrame:
    """One row per (serial, class) for SEC-matched owners: cik, year, A_pct, L_pct."""
    parts = []
    for c in CLASSES:
        mem_wait()
        rs = pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                             columns=["serial_number", "year", "topic_kl_vs_past", "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite() & pl.col("topic_kl_vs_future").is_finite())
        rs = rs.with_columns(((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
                             (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"))
        rs = rs.with_columns(
            ((pl.col("A").rank("average").over("year") - 0.5) / pl.len().over("year")).alias("A_pct"),
            ((pl.col("L").rank("average").over("year") - 0.5) / pl.len().over("year")).alias("L_pct"),
        ).filter(pl.col("year") >= FILING_LO).select("serial_number", "year", "A_pct", "L_pct")
        tm = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "owner_name"]).unique(
            "serial_number").with_columns(norm_owner(pl.col("owner_name")).alias("okey")).join(
            cw, on="okey", how="inner").select("serial_number", "cik")
        j = rs.join(tm, on="serial_number", how="inner").with_columns(pl.lit(c).alias("cls"))
        parts.append(j)
        print(f"[class {c}] scored {rs.height:,} matched {j.height:,}", file=sys.stderr, flush=True)
        del rs, tm
        gc.collect()
    d = pl.concat(parts)
    # a multi-class serial counts once; average its class-specific percentiles
    return d.group_by("serial_number", "cik").agg(pl.col("year").min(), pl.col("A_pct").mean(),
                                                   pl.col("L_pct").mean(), pl.len().alias("n_cls"))


def panel(f: pl.DataFrame) -> pd.DataFrame:
    fy = f.group_by("cik", "year").agg(pl.len().alias("n"), pl.col("A_pct").sum().alias("sA"),
                                       pl.col("L_pct").sum().alias("sL"))
    # window t-3..t: expand each filing year to the four fiscal years it feeds
    win = pl.concat([fy.with_columns((pl.col("year") + k).alias("t")) for k in range(4)]).group_by(
        "cik", "t").agg(pl.col("n").sum(), pl.col("sA").sum(), pl.col("sL").sum()).with_columns(
        (pl.col("sA") / pl.col("n")).alias("A_pct"), (pl.col("sL") / pl.col("n")).alias("L_pct"))
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet",
                          columns=["cik", "sic", "fy", "revenue", "gross_margin", "operating_income"]).filter(
        pl.col("fy").is_between(T_LO, T_HI_EXT) & (pl.col("revenue") > 0)).unique(["cik", "fy"]).rename(
        {"fy": "t"})
    p = win.join(sec, on=["cik", "t"], how="inner").with_columns(
        (pl.col("operating_income") / pl.col("revenue")).alias("op_margin"),
        pl.col("sic").str.slice(0, 2).alias("sic2"),
        pl.col("revenue").log().alias("log_rev"), pl.col("n").cast(pl.Float64).log().alias("log_n"))
    return p.select("cik", "t", "sic2", "n", "A_pct", "L_pct", "log_rev", "log_n", "revenue",
                    "gross_margin", "op_margin").to_pandas()


def demean(df: pd.DataFrame, cols: list[str], fes: list[str], tol: float = 1e-10, it: int = 500) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    if len(fes) == 1:
        return out - out.groupby(df[fes[0]]).transform("mean")
    for _ in range(it):
        prev = out.copy()
        for fe in fes:
            out = out - out.groupby(df[fe]).transform("mean")
        if float(np.max(np.abs(out.values - prev.values))) < tol:
            break
    return out


def fit(p: pd.DataFrame, y: str, fes: list[str]) -> dict:
    d = p.dropna(subset=[y]).copy()
    d = d[(d[y] >= -1) & (d[y] <= 1)].copy()
    d["y_pp"] = 100 * d[y]
    d["sy"] = d["sic2"].astype(str) + "_" + d["t"].astype(str)
    # drop singletons iteratively (they carry no within-FE information)
    for _ in range(20):
        n0 = len(d)
        for fe in fes:
            d = d[d.groupby(fe)[fe].transform("size") > 1]
        if len(d) == n0:
            break
    X = ["A_pct", "L_pct", "log_rev", "log_n"]
    dm = demean(d, ["y_pp", *X], fes)
    m = sm.OLS(dm["y_pp"], dm[X]).fit(cov_type="cluster", cov_kwds={"groups": d["cik"].values})
    r = {"n_firm_years": int(len(d)), "n_firms": int(d["cik"].nunique()),
         "mean_y_pp": float(d["y_pp"].mean()), "sd_y_pp": float(d["y_pp"].std())}
    for x in X:
        r[x] = {"b": float(m.params[x]), "se": float(m.bse[x]), "p": float(m.pvalues[x])}
    return r


def main() -> int:
    cw = crosswalk()
    f = filings(cw)
    print(f"[filings] {f.height:,} matched serials, {f['cik'].n_unique():,} ciks", file=sys.stderr, flush=True)
    p = panel(f)
    del f
    gc.collect()
    out = {"description": __doc__.strip().splitlines()[0],
           "crosswalk_keys": cw.height, "crosswalk_ciks": int(cw["cik"].n_unique()),
           "samples": {}}
    for tag, hi in (("t2009_2019", T_HI_MAIN), ("t2009_2022", T_HI_EXT)):
        s = p[p["t"] <= hi]
        res = {}
        for y in ("gross_margin", "op_margin"):
            res[y] = {"a_sic2xyear_fe": fit(s, y, ["sy"]),
                      "b_plus_firm_fe": fit(s, y, ["sy", "cik"])}
        out["samples"][tag] = res
    # check: portfolios of 5+ filings (a one-filing portfolio's percentile is very noisy)
    s5 = p[(p["t"] <= T_HI_MAIN) & (p["n"] >= 5)]
    out["samples"]["t2009_2019_min5filings"] = {
        y: {"a_sic2xyear_fe": fit(s5, y, ["sy"]), "b_plus_firm_fe": fit(s5, y, ["sy", "cik"])}
        for y in ("gross_margin", "op_margin")}
    # descriptive: margin by atypicality / lead quintile (main sample, gross margin)
    s = p[(p["t"] <= T_HI_MAIN) & p["gross_margin"].between(-1, 1)].copy()
    desc = {}
    for v in ("A_pct", "L_pct"):
        q = pd.qcut(s[v], 5, labels=False)
        g = s.groupby(q).agg(gm=("gross_margin", "mean"), n=("cik", "size"), filings=("n", "median"))
        desc[v] = [{"quintile": int(k) + 1, "mean_gross_margin_pp": 100 * float(r.gm), "n": int(r.n),
                    "median_filings": float(r.filings)} for k, r in g.iterrows()]
    out["quintile_means_gross_margin"] = desc
    out["corr_A_L_firmyear"] = float(s[["A_pct", "L_pct"]].corr().iloc[0, 1])
    out["portfolio_spread"] = {v: {"sd": float(s[v].std()), "p10": float(s[v].quantile(.1)),
                                   "p90": float(s[v].quantile(.9)),
                                   "within_firm_sd": float((s[v] - s.groupby("cik")[v].transform("mean")).std())}
                               for v in ("A_pct", "L_pct")}
    out["median_filings_in_window"] = float(s["n"].median())
    RES.joinpath("margins_novelty.json").write_text(json.dumps(out, indent=1))
    for tag, res in out["samples"].items():
        for y, specs in res.items():
            for k, r in specs.items():
                print(f"{tag} {y:12s} {k:16s} firms {r['n_firms']:5d} fy {r['n_firm_years']:6d}  "
                      f"A {r['A_pct']['b']:+6.2f} ({r['A_pct']['se']:.2f})  L {r['L_pct']['b']:+6.2f} "
                      f"({r['L_pct']['se']:.2f})  logrev {r['log_rev']['b']:+.2f} logn {r['log_n']['b']:+.2f}")
    print(json.dumps(desc, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
