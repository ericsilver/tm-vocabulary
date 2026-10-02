"""Test 4. Record-based industry scores for the 60 product segments, and a check
of whether they line up with segment outcomes.

Population: the frame's registrations (2002-2018), assigned to the 60 theme-mix
segments (theme_segments.parquet). Segment labels exist only for registrations,
so "filings" below are registered filings; the frame's filing years run from
about 2000 (earlier filing years are truncated by the 2002 registration start).

Pre-window scores. For each segment and filing-year cohort t in 2006..2017 a
score is computed over the segment's registrations filed in t-5..t-1 (so no
window reaches before 2001); the segment's score is the cohort-size-weighted mean
over cohorts.
  five forces
    entry_debut        share of filings from owners' first-ever filing (prior == 0)
    entry_owner_growth slope of log distinct owners per year over the window
    rivalry_owners     log number of distinct owners active in the window
    rivalry_hhi        HHI of filings across owners in the window
    substitutes        number of other segment centroids within Hellinger distance
                       h* (h* = 10th percentile of all centroid pairs); also the
                       mean distance to the 3 nearest centroids (static, theme_mix_theta)
    buyer_power        share naming retail/wholesale/distribution channels
    platform_power     share naming platforms they depend on
  scale / winner-take-most (outcome-side, by cohort)
    hhi_in_use         HHI of owners among the cohort's registrations still in use
                       at year five; mean over cohorts 2002-2016 and its slope per year
    est_gap            survival of owners with 25+ prior filings minus first-time
                       filers in the segment (class x reg-year FE, owner-clustered)
  Teece
    imitability: no_counsel_share, no_patent_share, debut share (= entry_debut)
    complementary assets: incumbent_share (25+ prior), regulated_share
                       (classes 5, 10, 33, 34, 36), counsel_share (= 1 - no_counsel)
  outcomes
    surv_adj   mean survival minus class x reg-year mean (plus frame mean), with an
               owner-clustered SE
    lead_b/se  segment lead effect (Design model within the segment, controls)
Validation (60 segments, standardized scores, WLS with HC1 SEs):
  (a) surv_adj on each force score alone and jointly, weights 1/se(surv_adj)^2
  (b) lead_b on each Teece / scale score alone and jointly, weights 1/se(lead)^2

Output: data/processed/segment_scores.parquet, paper/results/teece_segment_scores.json
"""
from __future__ import annotations

import gc
import json
import math

import numpy as np
import polars as pl

from teece_common import (CLASSES, CONTROLS, PROC, REPO, Design, load_frame, log, mem_wait, save, wls)

BUYER = r"retail store|wholesale|distributorship|online marketplace|for others"
PLATFORM = r"downloadable|mobile app|application software for|app store|via (a|the) website|hosted|cloud|third.party"
REGULATED = ["005", "010", "033", "034", "036"]
COHORTS = list(range(2006, 2018))


def text_flags(serials: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for c in CLASSES:
        t = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "goods_services"]) \
            .join(serials.filter(pl.col("cls") == c).select("serial_number"), on="serial_number", how="semi") \
            .unique("serial_number")
        t = t.select("serial_number",
                     pl.col("goods_services").fill_null("").str.to_lowercase().str.contains(BUYER).alias("buyer"),
                     pl.col("goods_services").fill_null("").str.to_lowercase().str.contains(PLATFORM).alias("platform"))
        parts.append(t)
        del t
        gc.collect()
    return pl.concat(parts)


def centroid_scores() -> pl.DataFrame:
    th = pl.read_parquet(PROC / "theme_mix_theta.parquet")
    seg = pl.read_parquet(PROC / "theme_segments.parquet")
    tcols = [f"t{k}" for k in range(50)]
    th = th.join(seg, on=["serial_number", "cls"], how="inner")
    C = th.group_by("seg").agg([pl.col(t).clip(0, None).sqrt().mean() for t in tcols]).sort("seg")
    del th
    gc.collect()
    S = C.select(tcols).to_numpy().astype(np.float64)
    # renormalize so each centroid is the square root of a probability vector
    P = S ** 2
    P = P / P.sum(1, keepdims=True)
    R = np.sqrt(P)
    D = np.sqrt(np.maximum(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1), 0)) / math.sqrt(2)
    iu = np.triu_indices(len(R), 1)
    hstar = float(np.quantile(D[iu], 0.10))
    Dn = D + np.eye(len(R)) * 9
    within = (Dn <= hstar).sum(1)
    nn3 = np.sort(Dn, 1)[:, :3].mean(1)
    return pl.DataFrame({"seg": C["seg"], "substitutes": within.astype(np.float64), "substitutes_nn3_dist": nn3}), hstar


def hhi(counts: pl.Expr) -> pl.Expr:
    return ((counts / counts.sum()) ** 2).sum()


def main() -> int:
    mem_wait("start")
    cent, hstar = centroid_scores()
    log(f"[substitutes] h*={hstar:.3f}; mean within {cent['substitutes'].mean():.2f}")
    mem_wait("frame")
    d = load_frame(["prior", "patenter"]).join(pl.read_parquet(PROC / "theme_segments.parquet"),
                                               on=["serial_number", "cls"], how="inner")
    tf = text_flags(d.select("serial_number", "cls"))
    d = d.join(tf, on="serial_number", how="left").with_columns(pl.col("buyer").fill_null(False),
                                                                pl.col("platform").fill_null(False))
    del tf
    gc.collect()
    g_all = float(d["surv"].mean())
    d = d.with_columns(
        (pl.col("surv") - pl.col("surv").mean().over("cell") + g_all).alias("adj"),
        (pl.col("prior") == 0).alias("debut"), (pl.col("prior") >= 25).alias("est"),
        (pl.col("has_attorney") == 0).alias("no_counsel"), (~pl.col("patenter").fill_null(False)).alias("no_patent"),
        pl.col("cls").is_in(REGULATED).alias("regulated"),
        pl.col("owner_key").cast(pl.Categorical).to_physical().cast(pl.Int64).alias("oid"))
    log(f"[frame] {d.height:,} rows with segments")

    # ---------- per (seg, fy) sums, then 5-year pre-windows for cohorts
    flags = ["debut", "no_counsel", "no_patent", "est", "regulated", "buyer", "platform"]
    sy = d.group_by("seg", "fy").agg([pl.len().alias("n")] + [pl.col(f).sum().alias(f) for f in flags]
                                     + [pl.col("oid").n_unique().alias("owners_y")])
    win_rows = []
    so = d.select("seg", "fy", "oid")
    for t in COHORTS:
        w = so.filter(pl.col("fy").is_between(t - 5, t - 1))
        oc = w.group_by("seg", "oid").len()
        a = oc.group_by("seg").agg(pl.len().alias("owners_w"), hhi(pl.col("len")).alias("hhi_w"))
        s = sy.filter(pl.col("fy").is_between(t - 5, t - 1)).group_by("seg").agg(
            [pl.col("n").sum().alias("n_w")] + [pl.col(f).sum().alias(f + "_w") for f in flags])
        gr = []
        for (sg,), gg in sy.filter(pl.col("fy").is_between(t - 5, t - 1)).group_by(["seg"]):
            x = gg["fy"].to_numpy().astype(float); y = np.log(gg["owners_y"].to_numpy().astype(float))
            gr.append({"seg": sg, "growth": float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 else None})
        a = a.join(s, on="seg").join(pl.DataFrame(gr).with_columns(pl.col("seg").cast(pl.Int16)), on="seg", how="left")
        coh = sy.filter(pl.col("fy") == t).select("seg", pl.col("n").alias("coh_n"))
        win_rows.append(a.join(coh, on="seg", how="left").with_columns(pl.lit(t).alias("t")))
    W = pl.concat(win_rows, how="diagonal").with_columns(pl.col("coh_n").fill_null(0))
    del so, win_rows
    W = W.with_columns(
        *[(pl.col(f + "_w") / pl.col("n_w")).alias(f) for f in flags],
        pl.col("owners_w").log().alias("rivalry_owners"), pl.col("hhi_w").alias("rivalry_hhi"),
        pl.col("growth").alias("entry_owner_growth"))
    score_cols = ["debut", "no_counsel", "no_patent", "est", "regulated", "buyer", "platform",
                  "rivalry_owners", "rivalry_hhi", "entry_owner_growth"]
    S = W.group_by("seg").agg([((pl.col(c) * pl.col("coh_n")).sum() / pl.col("coh_n").sum()).alias(c)
                               for c in score_cols]).rename(
        {"debut": "entry_debut", "buyer": "buyer_power", "platform": "platform_power", "est": "incumbent_share",
         "regulated": "regulated_share", "no_counsel": "no_counsel_share", "no_patent": "no_patent_share"})
    S = S.with_columns((1 - pl.col("no_counsel_share")).alias("counsel_share"))

    # ---------- scale: HHI among in-use marks by cohort (fy 2002-2016) and its trend
    alive = d.filter((pl.col("surv") > 50) & pl.col("fy").is_between(2002, 2016)).group_by("seg", "fy", "oid").len() \
        .group_by("seg", "fy").agg(hhi(pl.col("len")).alias("h"), pl.col("len").sum().alias("m"))
    hrow = []
    for (sg,), gg in alive.group_by(["seg"]):
        x = gg["fy"].to_numpy().astype(float); h = gg["h"].to_numpy(); m = gg["m"].to_numpy().astype(float)
        hrow.append({"seg": sg, "hhi_in_use": float(np.average(h, weights=m)),
                     "hhi_in_use_trend": float(np.polyfit(x, h, 1)[0]) if len(x) >= 5 else None})
    S = S.join(pl.DataFrame(hrow).with_columns(pl.col("seg").cast(pl.Int16)), on="seg", how="left")
    S = S.join(cent.with_columns(pl.col("seg").cast(pl.Int16)), on="seg", how="left")

    # ---------- outcomes and within-segment regressions
    rows = []
    for (sg,), g in d.group_by(["seg"]):
        n = g.height
        # owner-clustered SE of the cell-adjusted mean
        e = g.select(pl.col("oid"), (pl.col("adj") - pl.col("adj").mean()).alias("e")).group_by("oid").agg(pl.col("e").sum())
        se_m = float(math.sqrt((e["e"].to_numpy() ** 2).sum()) / n)
        des = Design(g)
        X = np.column_stack([g["lead"].to_numpy()] + [g[c].to_numpy() for c in CONTROLS])
        r = des.ols(g["surv"].to_numpy(), X, ["lead"] + CONTROLS)
        gg = g.filter(pl.col("debut") | pl.col("est"))
        if gg.filter(pl.col("est")).height >= 200 and gg.filter(pl.col("debut")).height >= 200:
            d2 = Design(gg)
            r2 = d2.ols(gg["surv"].to_numpy(), gg["est"].cast(pl.Float64).to_numpy()[:, None], ["est"])
            gap, gap_se = r2["est"][0], r2["est"][1]
            del d2
        else:
            gap, gap_se = None, None
        rows.append({"seg": sg, "n": n, "owners": des.G, "surv_raw": float(g["surv"].mean()),
                     "surv_adj": float(g["adj"].mean()), "surv_adj_se": se_m,
                     "lead_b": r["lead"][0], "lead_se": r["lead"][1], "est_gap": gap, "est_gap_se": gap_se})
        del des, X, g
        gc.collect()
    O = pl.DataFrame(rows).with_columns(pl.col("seg").cast(pl.Int16))
    names = {r["seg"]: r["name"] for r in json.loads((REPO / "paper" / "results" / "theme_segments.json")
                                                     .read_text(encoding="utf-8"))["segments"]}
    T = O.join(S, on="seg", how="left").with_columns(
        pl.col("seg").map_elements(lambda s: names.get(int(s), ""), return_dtype=pl.Utf8).alias("name")).sort("seg")
    T.write_parquet(PROC / "segment_scores.parquet")
    log(f"[table] {T.height} segments written")
    del d
    gc.collect()

    # ---------- validation
    def z(col):
        v = T[col].to_numpy().astype(float)
        return (v - np.nanmean(v)) / np.nanstd(v)

    forces = ["entry_debut", "entry_owner_growth", "rivalry_owners", "rivalry_hhi", "substitutes",
              "substitutes_nn3_dist", "buyer_power", "platform_power"]
    teece = ["no_counsel_share", "no_patent_share", "entry_debut", "incumbent_share", "regulated_share",
             "hhi_in_use", "hhi_in_use_trend", "est_gap"]
    y_s = T["surv_adj"].to_numpy(); w_s = 1 / T["surv_adj_se"].to_numpy() ** 2
    y_l = T["lead_b"].to_numpy(); w_l = 1 / T["lead_se"].to_numpy() ** 2
    val = {"a_survival_on_forces": {"alone": {}, "alone_unweighted": {}},
           "b_lead_on_teece_scale": {"alone": {}, "alone_unweighted": {}}}

    def run(y, w, cols, key):
        Xs = np.column_stack([z(c) for c in cols])
        ok = np.isfinite(Xs).all(1) & np.isfinite(y)
        return wls(y[ok], Xs[ok], w[ok], cols)

    for c in forces:
        val["a_survival_on_forces"]["alone"][c] = run(y_s, w_s, [c], "a")[c] | {"n": int(np.isfinite(z(c)).sum())}
        val["a_survival_on_forces"]["alone_unweighted"][c] = run(y_s, np.ones_like(w_s), [c], "a")[c]
    joint_f = [c for c in forces if c != "substitutes_nn3_dist"]
    val["a_survival_on_forces"]["joint"] = run(y_s, w_s, joint_f, "a")
    for c in teece:
        val["b_lead_on_teece_scale"]["alone"][c] = run(y_l, w_l, [c], "b")[c] | {"n": int(np.isfinite(z(c)).sum())}
        val["b_lead_on_teece_scale"]["alone_unweighted"][c] = run(y_l, np.ones_like(w_l), [c], "b")[c]
    val["b_lead_on_teece_scale"]["joint"] = run(y_l, w_l, teece, "b")
    # the lead effect on the force scores too, and survival on Teece scores, for completeness
    val["c_lead_on_forces_alone"] = {c: run(y_l, w_l, [c], "c")[c] for c in forces}
    val["d_survival_on_teece_alone"] = {c: run(y_s, w_s, [c], "d")[c] for c in teece}
    # pooled precision-weighted mean lead effect and heterogeneity
    mu = float((w_l * y_l).sum() / w_l.sum())
    Q = float((w_l * (y_l - mu) ** 2).sum())
    val["lead_heterogeneity"] = {"weighted_mean": mu, "Q": Q, "df": len(y_l) - 1,
                                 "tau2_DL": max(0.0, (Q - (len(y_l) - 1)) / (w_l.sum() - (w_l ** 2).sum() / w_l.sum()))}
    for k_, v in val["a_survival_on_forces"]["alone"].items():
        log(f"  surv ~ {k_:22s} {v['b']:+.2f} ({v['se']:.2f})")
    for k_, v in val["b_lead_on_teece_scale"]["alone"].items():
        log(f"  lead ~ {k_:22s} {v['b']:+.2f} ({v['se']:.2f})")
    out = {"n_segments": T.height, "hellinger_threshold": hstar, "cohorts": [COHORTS[0], COHORTS[-1]],
           "frame_mean_survival": g_all, "validation": val,
           "table": T.to_dicts(),
           "score_correlations": {a: {b: float(np.corrcoef(z(a)[np.isfinite(z(a)) & np.isfinite(z(b))],
                                                            z(b)[np.isfinite(z(a)) & np.isfinite(z(b))])[0, 1])
                                      for b in forces + teece} for a in forces + teece}}
    save("segment_scores", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
