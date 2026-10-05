"""Test 2. Does competition in a product segment go with lower producer margins?
Test 3. Powder coatings: who among SEC filers files powder-coating marks.

Test 2. Each SEC-matched firm (uspto_sec_crosswalk, owner names normalized with
norm_owner on both sides) is spread over the 60 product segments by the shares of its
registrations (2002-2018, theme_segments.parquet) in each segment. Firm gross margin is
the mean over fiscal years 2009-2023 (rows with gross margin outside [-1, 1] or
revenue <= 0 dropped); firm revenue is the mean over the same years.
  segment margin, unweighted   = sum_f s_fg * gm_f / sum_f s_fg
  segment margin, rev-weighted = sum_f s_fg * rev_f * gm_f / sum_f s_fg * rev_f
Kept: segments with at least 15 matched firms holding any registration there.
Effective firms = sum_f s_fg is reported alongside.
Check: modal assignment (each firm to its largest-share segment, unweighted mean,
segments with 15+ firms).
Segment margin is related to (a) the record-based scores in segment_scores.parquet and
(b) the mean of the three blind ratings (paper/ratings/ratings_*.json) by Pearson and
Spearman correlation and by WLS of margin (pp) on the standardized score, weights =
effective firms, HC1 SEs. With ~60 segments this is descriptive.
Per-segment table: margins next to five-year survival (theme_segments.json) and the
adjusted survival and lead effect in segment_scores.

Test 3. Filings in classes 001, 002, 040 whose goods/services match
powder[- ]coat|powder paint|coating powder, owners matched to SEC filers, listed with
SIC and median gross margin (2009+), grouped as formulators (SIC 2851), other
chemicals (28xx), equipment makers (35xx-38xx), coating services (347x), other.
Known large formulators are looked up by owner-name pattern to show which are missed
and why.

Output: paper/results/margins_competition.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from gate_decisive_regression import norm_owner  # noqa: E402
from margins_novelty import crosswalk, mem_wait  # noqa: E402

MIN_FIRMS = 15
FY_LO, FY_HI = 2009, 2023
SCORES = ["entry_debut", "entry_owner_growth", "rivalry_owners", "rivalry_hhi", "substitutes",
          "substitutes_nn3_dist", "buyer_power", "platform_power", "hhi_in_use", "hhi_in_use_trend",
          "est_gap", "no_counsel_share", "no_patent_share", "incumbent_share", "regulated_share",
          "surv_adj", "lead_b"]
RATINGS = ["entry_threat", "rivalry", "substitutes", "buyer_power", "supplier_power", "scale_economies",
           "imitability", "complementary_assets", "incumbents_hold_assets"]
POWDER = r"powder[- ]coat|powder paint|coating powder"
KNOWN = {  # name pattern in filings -> SEC registrant CIK if it files financial statements
    "AkzoNobel": (r"AKZO", None),
    "Axalta": (r"AXALTA|DU ?PONT", 1616862),
    "PPG": (r"\bPPG\b", 79879),
    "Sherwin-Williams (incl. Valspar)": (r"SHERWIN|SWIMC|VALSPAR", 89800),
    "RPM": (r"\bRPM\b|RUST OLEUM|CARBOLINE|TREMCO|TCI\b|TCI POWDER", 110621),
    "Tiger": (r"TIGER COATINGS|TIGER DRYLAC", None),
}


def firm_margins() -> pl.DataFrame:
    return pl.read_parquet(PROC / "sec_firm_year.parquet",
                           columns=["cik", "name", "sic", "fy", "revenue", "gross_margin"]).filter(
        pl.col("fy").is_between(FY_LO, FY_HI) & (pl.col("revenue") > 0)
        & pl.col("gross_margin").is_between(-1, 1)).group_by("cik").agg(
        pl.col("name").sort_by("fy").last(), pl.col("sic").sort_by("fy").last(),
        pl.col("gross_margin").mean().alias("gm"), pl.col("gross_margin").median().alias("gm_med"),
        pl.col("revenue").mean().alias("rev"), pl.len().alias("n_years"))


def firm_segment_shares(cw: pl.DataFrame) -> pl.DataFrame:
    seg = pl.read_parquet(PROC / "theme_segments.parquet")
    parts = []
    for c in sorted(seg["cls"].unique().to_list()):
        mem_wait()
        s = seg.filter(pl.col("cls") == c)
        tm = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "owner_name"]).join(
            s.select("serial_number"), on="serial_number", how="semi").unique("serial_number").with_columns(
            norm_owner(pl.col("owner_name")).alias("okey")).join(cw, on="okey", how="inner")
        parts.append(tm.join(s, on="serial_number").select("cik", "seg"))
        del tm, s
        gc.collect()
    d = pl.concat(parts)
    del seg
    return d.group_by("cik", "seg").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over("cik")).alias("share"),
        pl.col("len").sum().over("cik").alias("firm_regs"))


def wls(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> dict:
    z = (x - x.mean()) / x.std()
    m = sm.WLS(y, sm.add_constant(z), weights=w).fit(cov_type="HC1")
    return {"b_per_sd": float(m.params[1]), "se": float(m.bse[1]), "p": float(m.pvalues[1]), "n": int(len(y))}


def relate(tab: pl.DataFrame, cols: list[str]) -> dict:
    out = {}
    for ycol in ("gm_unw", "gm_rw"):
        out[ycol] = {}
        for c in cols:
            t = tab.drop_nulls([c, ycol]).filter(pl.col(c).is_finite())
            y, x, w = (t[ycol].to_numpy() * 100, t[c].to_numpy().astype(float), t["eff_firms"].to_numpy())
            out[ycol][c] = {"pearson": float(stats.pearsonr(x, y)[0]),
                            "spearman": float(stats.spearmanr(x, y)[0]),
                            "spearman_p": float(stats.spearmanr(x, y)[1]), **wls(y, x, w)}
    return out


def test2(cw: pl.DataFrame, fm: pl.DataFrame) -> dict:
    sh = firm_segment_shares(cw).join(fm.select("cik", "gm", "rev"), on="cik", how="inner")
    seg = sh.group_by("seg").agg(
        pl.len().alias("n_firms"), pl.col("share").sum().alias("eff_firms"),
        ((pl.col("share") * pl.col("gm")).sum() / pl.col("share").sum()).alias("gm_unw"),
        ((pl.col("share") * pl.col("rev") * pl.col("gm")).sum() / (pl.col("share") * pl.col("rev")).sum())
        .alias("gm_rw"))
    # top three revenue contributors per segment (who drives the weighted mean)
    top = sh.with_columns((pl.col("share") * pl.col("rev")).alias("wr")).join(
        fm.select("cik", "name"), on="cik").sort("wr", descending=True).group_by("seg", maintain_order=True).agg(
        pl.col("name").head(3).alias("top_rev_firms"),
        (pl.col("wr").head(3).sum() / pl.col("wr").sum()).alias("top3_rev_weight"))
    modal = sh.sort("share", descending=True).unique("cik", keep="first").group_by("seg").agg(
        pl.len().alias("modal_n_firms"), pl.col("gm").mean().alias("modal_gm"))
    sc = pl.read_parquet(PROC / "segment_scores.parquet")
    rt = []
    for f in sorted((REPO / "paper" / "ratings").glob("ratings_*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            rt.append({"seg": int(r["segment"][1:]), **{f"r_{k}": float(r[k]) for k in RATINGS}})
    rt = pl.DataFrame(rt).group_by("seg").agg(pl.all().mean()).with_columns(pl.col("seg").cast(pl.Int16))
    surv = pl.DataFrame([{"seg": s["seg"], "survival": s["survival"], "lead_effect": s["lead_effect"]}
                         for s in json.loads((RES / "theme_segments.json").read_text())["segments"]]).with_columns(
        pl.col("seg").cast(pl.Int16))
    seg = seg.with_columns(pl.col("seg").cast(pl.Int16))
    full = seg.join(sc, on="seg", how="left").join(rt, on="seg", how="left").join(surv, on="seg", how="left").join(
        top.with_columns(pl.col("seg").cast(pl.Int16)), on="seg", how="left").join(
        modal.with_columns(pl.col("seg").cast(pl.Int16)), on="seg", how="left")
    kept = full.filter(pl.col("n_firms") >= MIN_FIRMS)
    rel_scores = relate(kept, SCORES)
    rel_ratings = relate(kept, [f"r_{k}" for k in RATINGS])
    rel_surv = relate(kept, ["survival"])
    km = kept.filter(pl.col("modal_n_firms") >= MIN_FIRMS)
    modal_check = {}
    for c in SCORES + [f"r_{k}" for k in RATINGS] + ["survival"]:
        t = km.drop_nulls([c, "modal_gm"])
        modal_check[c] = {"spearman": float(stats.spearmanr(t[c], t["modal_gm"])[0]),
                          "spearman_p": float(stats.spearmanr(t[c], t["modal_gm"])[1]), "n": t.height}
    rows = kept.sort("gm_unw", descending=True).select(
        "seg", "name", "n_firms", "eff_firms", "gm_unw", "gm_rw", "modal_n_firms", "modal_gm", "survival",
        "surv_adj", "lead_effect", "top_rev_firms", "top3_rev_weight").to_dicts()
    return {"n_firms_with_margin_and_segment": int(sh["cik"].n_unique()),
            "n_segments_total": full.height, "n_segments_kept": kept.height,
            "min_firms": MIN_FIRMS, "n_segments_modal_check": km.height,
            "margin_spread_pp": {"unw_min": float(kept["gm_unw"].min() * 100), "unw_max": float(kept["gm_unw"].max() * 100),
                                 "unw_sd": float(kept["gm_unw"].std() * 100),
                                 "rw_sd": float(kept["gm_rw"].std() * 100)},
            "record_scores": rel_scores, "blind_ratings": rel_ratings, "survival": rel_surv,
            "modal_assignment_check": modal_check, "segments": rows}


def classify(sic: str | None) -> str:
    s = int(sic) if sic and sic.isdigit() else -1
    if s == 2851:
        return "formulator (2851 paints)"
    if 2800 <= s < 2900:
        return "other chemicals (28xx)"
    if 3470 <= s < 3480:
        return "coating services (347x)"
    if 3500 <= s < 3900:
        return "equipment / machinery (35xx-38xx)"
    return "other"


def test3(cw: pl.DataFrame, fm: pl.DataFrame) -> dict:
    parts = []
    for c in ("001", "002", "040"):
        parts.append(pl.read_parquet(PROC / f"tm_class{c}.parquet",
                                     columns=["serial_number", "filing_date", "owner_name", "goods_services"])
                     .filter(pl.col("goods_services").fill_null("").str.to_lowercase().str.contains(POWDER)))
    d = pl.concat(parts).unique("serial_number").with_columns(
        norm_owner(pl.col("owner_name")).alias("okey"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"))
    owners = d.group_by("okey").agg(pl.len().alias("filings"))
    m = d.join(cw, on="okey", how="inner")
    firms = m.group_by("cik").agg(pl.len().alias("powder_filings"), pl.col("fy").min().alias("first"),
                                  pl.col("fy").max().alias("last"), pl.col("owner_name").first()).join(
        fm, on="cik", how="left").with_columns(
        pl.col("sic").map_elements(classify, return_dtype=pl.Utf8).alias("group"))
    with_margin = firms.filter(pl.col("gm_med").is_not_null())
    grp = with_margin.group_by("group").agg(pl.len().alias("n_firms"), pl.col("gm_med").median().alias("median_gm"),
                                            pl.col("powder_filings").sum()).sort("n_firms", descending=True)
    sec_names = pl.read_parquet(PROC / "sec_firm_year.parquet", columns=["cik", "name", "sic"]).unique(
        "cik", keep="last")
    cw_full = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet", columns=["owner_name", "cik"])
    known = {}
    for firm, (pat, cik) in KNOWN.items():
        k = d.filter(pl.col("okey").str.contains(pat))
        names = k.group_by("owner_name").len().sort("len", descending=True)
        matched = k.join(cw, on="okey", how="inner")
        reg = sec_names.filter(pl.col("cik") == cik).to_dicts() if cik else []
        cw_names = cw_full.filter(pl.col("cik") == cik)["owner_name"].unique().to_list() if cik else []
        known[firm] = {"powder_filings": k.height,
                       "filing_owner_names": [{"owner": o, "n": int(n)} for o, n in names.head(6).iter_rows()],
                       "matched_powder_filings": matched.height,
                       "sec_registrant": reg[0] if reg else None,
                       "crosswalk_names_for_registrant": cw_names[:8]}
    out = {"regex": POWDER, "powder_filings": d.height, "powder_owners": owners.height,
           "matched_owner_keys": int(m["okey"].n_unique()), "matched_firms": firms.height,
           "matched_firms_with_margin": with_margin.height,
           "matched_powder_filings": m.height,
           "formulator_share_of_matched_firms": float((with_margin["group"] == "formulator (2851 paints)").mean())
           if with_margin.height else None,
           "by_group": grp.to_dicts(),
           "firms": firms.sort("powder_filings", descending=True).select(
               "cik", "name", "owner_name", "sic", "group", "powder_filings", "first", "last", "gm_med",
               "n_years").to_dicts(),
           "known_formulators": known}
    return out


def main() -> int:
    cw = crosswalk()
    fm = firm_margins()
    out = {"test2_segments": test2(cw, fm)}
    gc.collect()
    out["test3_powder"] = test3(cw, fm)
    RES.joinpath("margins_competition.json").write_text(json.dumps(out, indent=1, default=str))
    t2 = out["test2_segments"]
    print(f"segments kept {t2['n_segments_kept']} of {t2['n_segments_total']}; firms {t2['n_firms_with_margin_and_segment']}"
          f"; spread {t2['margin_spread_pp']}")
    for block in ("record_scores", "blind_ratings", "survival"):
        print(f"== {block}")
        for c, r in t2[block]["gm_unw"].items():
            rw = t2[block]["gm_rw"][c]
            print(f"  {c:24s} unw: r {r['pearson']:+.2f} rho {r['spearman']:+.2f} b/sd {r['b_per_sd']:+5.2f} "
                  f"({r['se']:.2f}) | rw: rho {rw['spearman']:+.2f} b/sd {rw['b_per_sd']:+5.2f} ({rw['se']:.2f})")
    print("== modal check", {k: round(v["spearman"], 2) for k, v in t2["modal_assignment_check"].items()},
          t2["n_segments_modal_check"])
    for r in t2["segments"]:
        print(f"  S{r['seg']:02d} {r['name'][:40]:40s} firms {r['n_firms']:4d} eff {r['eff_firms']:6.1f} "
              f"gm {100 * r['gm_unw']:5.1f} rw {100 * r['gm_rw']:5.1f} surv {r['survival']:5.1f} "
              f"top {r['top_rev_firms'][:2]} {r['top3_rev_weight']:.2f}")
    t3 = out["test3_powder"]
    print(f"== powder: filings {t3['powder_filings']} owners {t3['powder_owners']} matched firms "
          f"{t3['matched_firms']} (with margin {t3['matched_firms_with_margin']}), formulator share "
          f"{t3['formulator_share_of_matched_firms']}")
    print(t3["by_group"])
    for f in t3["firms"]:
        print("  ", f)
    for k, v in t3["known_formulators"].items():
        print("  ", k, v["powder_filings"], v["matched_powder_filings"], v["sec_registrant"],
              [x["owner"] for x in v["filing_owner_names"]][:4], v["crosswalk_names_for_registrant"][:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
