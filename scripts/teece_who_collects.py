"""Test 3. Who collects when a theme surges: pioneers, followers, late entrants, incumbents?

Surge: a class x primary-theme whose share of the class's filings in years
t0+1..t0+5 is at least 1.5x its share in t0-5..t0-1, with at least 200 filings of
the theme in t0+1..t0+5. Consecutive qualifying years form one run; the onset t0
is the first qualifying year of a run, and a new episode in the same class-theme
must start at least 10 years after the previous onset. Onsets 1998-2008 (so the
filings of years t0+1..t0+10 fall mostly inside the 2002-2018 registration frame).
Shares are computed from every scored filing (theme_full), as in the battery.

Groups (registrations in the frame with that class and primary theme):
  pioneers  fy in t0+1..t0+2       followers fy in t0+3..t0+5
  late      fy in t0+6..t0+10      pre       fy in t0-4..t0 (reference)
  incumbents: owner with 25+ earlier filings (any timing, overlaps the above)
Reported: survival (cell-adjusted: survival minus the class x registration-year
mean, plus the frame mean), a regression on group dummies with cell FE and
controls, and each group's share of the theme's registrations and of those still
in use after the first maintenance deadline (pooled and averaged over episodes).

Acquisition: from the USPTO Trademark Assignment Dataset (as in
acquisition_rung.py): a registration is ACQUIRED when a full assignment or merger,
recorded after filing, is made by its owner (normalized names equal) to a
different, non-individual company (different normalized name and first word),
not distressed; ESTABLISHED BUYER when the assignee holds 25+ marks in the corpus.
The flag is computed for every frame registration and cached in
data/processed/teece_acquired.parquet.

Output: paper/results/teece_who_collects.json
"""
from __future__ import annotations

import gc
import importlib.util

import numpy as np
import polars as pl

from teece_common import CLASSES, CONTROLS, PROC, REPO, cell_ols, load_frame, log, mem_wait, save

RAW = REPO / "data" / "raw" / "tm_assignment"
DISTRESS = r"bankrupt|foreclos|court order|receiver|trustee|sheriff|liquidat|benefit of creditors"
ESTABLISHED = 25
ONSET = (1998, 2008)


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_map(names: pl.Series, normalize) -> pl.DataFrame:
    u = names.drop_nulls().unique()
    return pl.DataFrame({"raw": u, "norm": [normalize(s) for s in u.to_list()]})


def acquired_flags() -> pl.DataFrame:
    cache = PROC / "teece_acquired.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    normalize = load_module("sl", "sec_link.py").normalize
    frame = pl.read_parquet(PROC / "battery_frame.parquet", columns=["serial_number"])
    own, fd = [], []
    for c in CLASSES:
        t = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "owner_name", "filing_date"])
        own.append(t.select("serial_number", "owner_name"))
        del t
    own = pl.concat(own).unique("serial_number").drop_nulls("owner_name")
    counts = own.group_by("owner_name").len()
    onm = norm_map(counts["owner_name"], normalize).rename({"raw": "owner_name", "norm": "nn"})
    size = counts.join(onm, on="owner_name").group_by("nn").agg(pl.col("len").sum().alias("marks"))
    log(f"[acq] {onm.height:,} owner names normalized")
    # filing dates and owner for the frame
    fr = []
    for c in CLASSES:
        t = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "owner_name", "filing_date"]) \
            .join(frame, on="serial_number", how="semi")
        fr.append(t)
    fr = pl.concat(fr).unique("serial_number").join(onm, on="owner_name", how="left").with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fdd"),
        pl.col("serial_number").cast(pl.Utf8).str.strip_chars().alias("serial"),
        pl.col("nn").str.split(" ").list.first().alias("nfirst")).select("serial_number", "serial", "nn", "nfirst", "fdd")
    del own, counts
    gc.collect()
    mem_wait("assignment csv")
    DOC = pl.scan_csv(RAW / "tm_docid.csv", infer_schema_length=0).select("rf_id", "serial") \
        .filter(pl.col("serial").is_not_null()).join(fr.lazy().select("serial"), on="serial", how="semi").collect()
    rf = DOC.select("rf_id").unique()
    A = pl.scan_csv(RAW / "tm_assignment.csv", infer_schema_length=0).select("rf_id", "record_dt", "convey_text") \
        .join(rf.lazy(), on="rf_id", how="semi").collect()
    C = pl.read_csv(RAW / "tm_convey.csv", infer_schema_length=0)
    T = A.join(C, on="rf_id").filter(pl.col("conv_group").is_in(["assignment", "merger"])).with_columns(
        pl.col("convey_text").fill_null("").str.to_lowercase().alias("ct"),
        pl.col("record_dt").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rdt")).filter(
        ~pl.col("ct").str.contains("undivided")).with_columns(pl.col("ct").str.contains(DISTRESS).alias("distressed")) \
        .select("rf_id", "rdt", "distressed")
    del A, C
    OR = pl.scan_csv(RAW / "tm_assignor.csv", infer_schema_length=0).select("rf_id", "or_name", "or_legal_entity_text") \
        .join(T.lazy().select("rf_id"), on="rf_id", how="semi").collect()
    EE = pl.scan_csv(RAW / "tm_assignee.csv", infer_schema_length=0).select("rf_id", "ee_name", "ee_legal_entity_text") \
        .join(T.lazy().select("rf_id"), on="rf_id", how="semi").collect()
    orn = norm_map(OR["or_name"], normalize).rename({"raw": "or_name", "norm": "or_norm"})
    een = norm_map(EE["ee_name"], normalize).rename({"raw": "ee_name", "norm": "ee_norm"})
    T = T.join(DOC, on="rf_id").join(OR.join(orn, on="or_name"), on="rf_id").join(EE.join(een, on="ee_name"), on="rf_id")
    log(f"[acq] {T.height:,} transfer rows touching frame serials")
    x = T.join(fr.select("serial", "nn", "nfirst", "fdd"), on="serial").filter(
        (pl.col("rdt") > pl.col("fdd")) & (pl.col("or_norm") == pl.col("nn")) & (pl.col("ee_norm") != pl.col("nn"))
        & (pl.col("ee_norm").str.split(" ").list.first() != pl.col("nfirst"))
        & (pl.col("or_legal_entity_text").fill_null("") != "INDIVIDUAL")
        & (pl.col("ee_legal_entity_text").fill_null("") != "INDIVIDUAL")
        & ~pl.col("distressed") & ~pl.col("or_name").str.to_lowercase().str.contains(DISTRESS))
    x = x.join(size.rename({"nn": "ee_norm"}), on="ee_norm", how="left").with_columns(
        (pl.col("marks").fill_null(0) >= ESTABLISHED).alias("est"))
    acq = x.group_by("serial").agg(pl.len().alias("_n"), pl.col("est").any().alias("acq_established"),
                                   pl.col("rdt").min().alias("acq_date"))
    out = fr.select("serial_number", "serial").join(acq, on="serial", how="left").select(
        "serial_number", pl.col("_n").is_not_null().alias("acquired"),
        pl.col("acq_established").fill_null(False), "acq_date")
    out.write_parquet(cache)
    log(f"[acq] acquired {out['acquired'].mean():.4f}, established buyer {out['acq_established'].mean():.4f}")
    del T, x, OR, EE, DOC
    gc.collect()
    return out


def theme_panel() -> pl.DataFrame:
    th = pl.concat([pl.read_parquet(PROC / "theme_full" / f"theme_class{c}.parquet", columns=["fy", "top_theme"])
                    .with_columns(pl.lit(c).alias("cls")) for c in CLASSES]).filter(pl.col("fy") >= 1980)
    cy = th.group_by("cls", "fy").agg(pl.len().alias("n_c"))
    ck = th.group_by("cls", "top_theme", "fy").agg(pl.len().alias("n_k"))
    del th
    years = list(range(1980, 2027))
    grid = cy.select("cls").unique().join(pl.DataFrame({"top_theme": list(range(50))}, schema={"top_theme": pl.Int16}),
                                          how="cross").join(pl.DataFrame({"fy": years}, schema={"fy": pl.Int32}), how="cross")
    p = grid.join(cy, on=["cls", "fy"], how="left").join(ck.with_columns(pl.col("top_theme").cast(pl.Int16)),
                                                        on=["cls", "top_theme", "fy"], how="left").with_columns(
        pl.col("n_k").fill_null(0), pl.col("n_c").fill_null(0)).sort(["cls", "top_theme", "fy"]).with_columns(
        pl.when(pl.col("n_c") > 0).then(pl.col("n_k") / pl.col("n_c")).alias("s"))
    ov = ["cls", "top_theme"]
    p = p.with_columns(
        pl.mean_horizontal([pl.col("s").shift(k).over(ov) for k in range(1, 6)]).alias("pre"),
        pl.mean_horizontal([pl.col("s").shift(-k).over(ov) for k in range(1, 6)]).alias("post"),
        pl.sum_horizontal([pl.col("n_k").shift(-k).over(ov) for k in range(1, 6)]).alias("n_post"))
    return p


def episodes(p: pl.DataFrame) -> pl.DataFrame:
    q = p.with_columns(((pl.col("post") >= 1.5 * pl.col("pre")) & (pl.col("pre") > 0) & (pl.col("n_post") >= 200)
                        & (pl.col("fy") >= 1990) & (pl.col("fy") <= 2020)).fill_null(False).alias("qual"))
    rows = []
    for (c, k), g in q.filter(pl.col("qual")).group_by(["cls", "top_theme"]):
        last = -999
        for fy, pre, post, n in g.sort("fy").select("fy", "pre", "post", "n_post").iter_rows():
            if fy - last >= 10:
                rows.append({"cls": c, "top_theme": k, "t0": fy, "pre": pre, "post": post, "n_post": n})
                last = fy
    e = pl.DataFrame(rows).with_columns(pl.col("top_theme").cast(pl.Int16), pl.col("t0").cast(pl.Int32))
    return e


def main() -> int:
    mem_wait("start")
    p = theme_panel()
    e = episodes(p)
    del p
    gc.collect()
    out = {"episodes_all_onsets": e.height}
    e = e.filter(pl.col("t0").is_between(*ONSET)).with_row_index("ep")
    out["episodes"] = e.height
    out["episode_onset_years"] = e.group_by("t0").len().sort("t0").to_dicts()
    out["episode_ratio_median"] = float((e["post"] / e["pre"]).median())
    log(f"[episodes] {e.height} with onset {ONSET}")

    acq = acquired_flags()
    mem_wait("frame")
    d = load_frame(["top_theme", "prior"])
    g_all = float(d["surv"].mean())
    d = d.with_columns((pl.col("surv") - pl.col("surv").mean().over("cell") + g_all).alias("adj"))
    out["frame_acquired_rate"] = float(acq["acquired"].mean())
    out["frame_acq_established_rate"] = float(acq["acq_established"].mean())
    s = d.join(e.select("ep", "cls", "top_theme", "t0"), on=["cls", "top_theme"], how="inner").with_columns(
        (pl.col("fy") - pl.col("t0")).alias("k")).filter(pl.col("k").is_between(-4, 10))
    # a registration could fall in two episodes of the same class-theme only if they are <15 years apart;
    # keep the earliest episode
    s = s.sort("t0").unique("serial_number", keep="first")
    s = s.with_columns(
        pl.when(pl.col("k") <= 0).then(pl.lit("pre")).when(pl.col("k") <= 2).then(pl.lit("pioneer"))
        .when(pl.col("k") <= 5).then(pl.lit("follower")).otherwise(pl.lit("late")).alias("timing"),
        (pl.col("prior") >= ESTABLISHED).alias("incumbent"),
        (pl.col("prior") == 0).alias("debut")).join(acq.select("serial_number", "acquired", "acq_established"),
                                                    on="serial_number", how="left").with_columns(
        pl.col("acquired").fill_null(False), pl.col("acq_established").fill_null(False))
    out["n_registrations"] = s.height

    def summ(f):
        return {"n": f.height, "surv_raw": float(f["surv"].mean()), "surv_cell_adj": float(f["adj"].mean()),
                "acquired_pct": 100 * float(f["acquired"].mean()),
                "acq_by_established_pct": 100 * float(f["acq_established"].mean()),
                "incumbent_share": float(f["incumbent"].mean()), "debut_share": float(f["debut"].mean()),
                "mean_lead": float(f["lead"].mean())}
    groups = {}
    for t in ("pre", "pioneer", "follower", "late"):
        groups[t] = summ(s.filter(pl.col("timing") == t))
        groups[t + "_non_incumbent"] = summ(s.filter((pl.col("timing") == t) & ~pl.col("incumbent")))
    groups["incumbent"] = summ(s.filter(pl.col("incumbent")))
    groups["non_incumbent"] = summ(s.filter(~pl.col("incumbent")))
    groups["all_frame"] = {"surv_raw": g_all}
    out["groups"] = groups
    for k_, v in groups.items():
        if "n" in v:
            log(f"  {k_:24s} n={v['n']:>8,} surv {v['surv_raw']:.1f} adj {v['surv_cell_adj']:.1f} "
                f"acq {v['acquired_pct']:.2f}% est {v['acq_by_established_pct']:.2f}% lead {v['mean_lead']:+.3f}")

    # regression: group dummies vs pioneers, cell FE + controls (+ lead as a check)
    s = s.with_columns([(pl.col("timing") == t).cast(pl.Float64).alias(t) for t in ("pre", "follower", "late")]
                       + [pl.col("incumbent").cast(pl.Float64).alias("incumbent_f")])
    xs = ["pre", "follower", "late", "incumbent_f"]
    ctl_sets = {"no_controls": [], "controls_without_owner_size": [c for c in CONTROLS if c != "log_owner_n"],
                "controls_full": CONTROLS}
    out["regression_vs_pioneers"] = {"spec": "y ~ pre + follower + late + incumbent + controls | class x reg-year FE "
                                             "(pioneers omitted; incumbents overlap the timing groups)",
                                     "note": "log_owner_n (owner's registrations in the frame) is nearly collinear with "
                                             "incumbent status, so controls_without_owner_size is the main reading"}
    for lab, ctl in ctl_sets.items():
        r = cell_ols(s, "surv", xs + ctl)
        r2 = cell_ols(s, "surv", xs + ["lead"] + ctl)
        blk = {"surv": {k_: r[k_] for k_ in xs}, "surv_with_lead": {k_: r2[k_] for k_ in xs + ["lead"]}, "n": r["_n"]}
        for nm, col in (("acquired", "acquired"), ("acq_established", "acq_established")):
            s = s.with_columns((100 * pl.col(col).cast(pl.Float64)).alias("y_" + nm))
            ra = cell_ols(s, "y_" + nm, xs + ctl)
            blk[nm + "_pct"] = {k_: ra[k_] for k_ in xs}
        out["regression_vs_pioneers"][lab] = blk
        log(f"[reg {lab}] " + ", ".join(f"{k_} {r[k_]['b']:+.2f} ({r[k_]['se']:.2f})" for k_ in xs))

    # shares of the theme's registrations and of survivors, pooled and by episode
    s = s.with_columns((pl.col("surv") > 50).alias("alive"))
    def shares(f, label):
        tot_n, tot_a = f.height, int(f["alive"].sum())
        res = {}
        for t in ("pre", "pioneer", "follower", "late"):
            ff = f.filter(pl.col("timing") == t)
            res[t] = {"share_of_registrations": ff.height / tot_n, "share_of_in_use": int(ff["alive"].sum()) / tot_a}
        ff = f.filter(pl.col("incumbent"))
        res["incumbent"] = {"share_of_registrations": ff.height / tot_n, "share_of_in_use": int(ff["alive"].sum()) / tot_a}
        return res
    out["shares_pooled_k_minus4_to_10"] = shares(s, "pooled")
    out["shares_pooled_surge_entrants_only"] = shares(s.filter(pl.col("timing") != "pre"), "entrants")
    # episode-averaged (entrants only, episodes with >= 50 entrant registrations)
    eps = []
    for (ep,), f in s.filter(pl.col("timing") != "pre").group_by(["ep"]):
        if f.height >= 50 and f["alive"].sum() > 0:
            eps.append(shares(f, ep))
    out["shares_episode_mean_entrants_only"] = {
        g_: {m: float(np.mean([x[g_][m] for x in eps])) for m in ("share_of_registrations", "share_of_in_use")}
        for g_ in ("pioneer", "follower", "late", "incumbent")}
    out["shares_episode_n"] = len(eps)
    # end-of-surge stock: registrations filed by t0+5 (pre + pioneers + followers) still in use
    out["shares_end_of_surge_stock"] = shares(s.filter(pl.col("k") <= 5), "stock")
    log(f"[shares] pooled {out['shares_pooled_surge_entrants_only']}")
    log(f"[shares] stock {out['shares_end_of_surge_stock']}")
    save("who_collects", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
