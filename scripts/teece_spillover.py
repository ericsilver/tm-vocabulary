"""Test 5. Do failed pioneers supply language that survivors later use?

Within each Nice class, every filing's goods/services text (all filings, registered
or not) is lowercased and split into words ([a-z]+); its distinct two-word
sequences (bigrams) are hashed. A bigram is NEW in year t when no earlier filing
in the class used it (the class's record is scanned from its first year). Kept:
bigrams new in 2002..2015 that are used by at least 20 filings in later years.

For each kept bigram:
  first filers  = the filings of year t that used it; outcome from the frame
                  (registered 2002-2018): failed1 (cancelled at the first
                  maintenance deadline); also the share never in the frame
  later users   = filings of years > t that used it and are in the frame;
                  survival = 1 - failed1
Comparisons:
  - share of first filers that failed (bigram-level mean, and the share of bigrams
    whose single introducer failed) against the failure rate of all frame
    registrations of the same class and filing year
  - later users' survival, for bigrams introduced by failed vs surviving filings,
    against the same class-year baseline

Output: paper/results/teece_spillover.json
"""
from __future__ import annotations

import gc

import numpy as np
import polars as pl

from teece_common import CLASSES, PROC, log, mem_wait, save

T_LO, T_HI = 2002, 2015
MIN_LATER = 20


def bigram_rows(df: pl.DataFrame) -> pl.DataFrame:
    """df: serial_number, fy, gs -> distinct (serial_number, fy, h)."""
    return df.select(
        "serial_number", "fy",
        pl.col("gs").str.to_lowercase().str.extract_all(r"[a-z]+").list.eval(
            pl.concat_str([pl.element(), pl.element().shift(-1)], separator=" ")).alias("bg")
    ).explode("bg").drop_nulls("bg").select("serial_number", "fy", pl.col("bg").hash(seed=7).alias("h")).unique()


def main() -> int:
    frame = pl.read_parquet(PROC / "battery_frame.parquet", columns=["serial_number", "cls", "fy", "failed1"])
    base = frame.group_by("cls", "fy").agg(pl.col("failed1").mean().alias("base_fail"))
    fr = frame.select("serial_number", "failed1").unique("serial_number")
    del frame
    per_bigram, per_class = [], {}
    for c in CLASSES:
        mem_wait(f"class {c}")
        tm = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "filing_date", "goods_services"]) \
            .unique("serial_number").with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy")).drop_nulls("fy") \
            .rename({"goods_services": "gs"}).drop("filing_date").filter(pl.col("gs").is_not_null())
        years = sorted(tm["fy"].unique().to_list())
        chunks = [[y for y in years if y < 1990]] + [[y] for y in years if y >= 1990]
        B = None   # h -> first_year, n_later
        for ch in chunks:
            if not ch:
                continue
            r = bigram_rows(tm.filter(pl.col("fy").is_in(ch)))
            cnt = r.group_by("h").agg(pl.col("fy").min().alias("first"), pl.len().alias("n"))
            del r
            if B is None:
                B = cnt.select("h", "first", pl.lit(0, dtype=pl.UInt32).alias("n_later"))
                continue
            known = cnt.join(B.select("h"), on="h", how="semi").select("h", pl.col("n").cast(pl.UInt32).alias("add"))
            new = cnt.join(B.select("h"), on="h", how="anti").select("h", "first", pl.lit(0, dtype=pl.UInt32).alias("n_later"))
            B = B.join(known, on="h", how="left").with_columns(
                (pl.col("n_later") + pl.col("add").fill_null(0)).alias("n_later")).drop("add")
            B = pl.concat([B, new])
            del cnt, known, new
        # note: within a pre-1990 chunk, uses after the first year inside the chunk are not counted (irrelevant:
        # only bigrams first seen in 2002+ are kept)
        keep = B.filter(pl.col("first").is_between(T_LO, T_HI) & (pl.col("n_later") >= MIN_LATER))
        nb_total_new = B.filter(pl.col("first").is_between(T_LO, T_HI)).height
        del B
        gc.collect()
        if keep.height == 0:
            continue
        # second pass: uses of kept bigrams from T_LO on
        uses = []
        for y in [y for y in years if y >= T_LO]:
            r = bigram_rows(tm.filter(pl.col("fy") == y)).join(keep.select("h", "first"), on="h", how="inner")
            uses.append(r)
        del tm
        U = pl.concat(uses).join(fr, on="serial_number", how="left").with_columns(pl.lit(c).alias("cls"))
        del uses
        U = U.with_columns((pl.col("fy") == pl.col("first")).alias("is_first"))
        b = U.group_by("h").agg(
            pl.col("first").first(), pl.col("cls").first(),
            pl.col("is_first").sum().alias("n_first"),
            (pl.col("is_first") & pl.col("failed1").is_not_null()).sum().alias("n_first_reg"),
            pl.col("failed1").filter(pl.col("is_first")).cast(pl.Float64).mean().alias("first_fail"),
            (~pl.col("is_first")).sum().alias("n_later"),
            (~pl.col("is_first") & pl.col("failed1").is_not_null()).sum().alias("n_later_reg"),
            (1 - pl.col("failed1").filter(~pl.col("is_first")).cast(pl.Float64)).mean().alias("later_surv"),
            pl.col("fy").filter(~pl.col("is_first") & pl.col("failed1").is_not_null()).mean().alias("later_fy_mean"))
        per_bigram.append(b)
        per_class[c] = {"new_bigrams_2002_2015": nb_total_new, "kept": keep.height}
        log(f"  [{c}] new {nb_total_new:,} kept {keep.height:,}")
        del U, keep
        gc.collect()
    P = pl.concat(per_bigram).join(base.rename({"fy": "first"}), on=["cls", "first"], how="left")
    # later users' class-year baseline: survival of frame registrations in the same class, mean over the later
    # users' years is approximated by the class baseline at the later users' mean filing year (rounded)
    P = P.with_columns(pl.col("later_fy_mean").round(0).cast(pl.Int32).alias("lfy")).join(
        base.rename({"fy": "lfy", "base_fail": "later_base_fail"}), on=["cls", "lfy"], how="left")
    out = {"classes": per_class, "n_bigrams": P.height, "t_range": [T_LO, T_HI], "min_later_uses": MIN_LATER}
    R = P.filter(pl.col("n_first_reg") > 0)
    out["with_registered_first_filer"] = R.height
    out["first_filers"] = {
        "mean_n_first_filings": float(P["n_first"].mean()),
        "share_first_filings_never_in_frame": float(1 - P["n_first_reg"].sum() / P["n_first"].sum()),
        "bigram_mean_first_filer_failure": float(R["first_fail"].mean()),
        "bigram_mean_class_year_baseline": float(R["base_fail"].mean()),
        "filing_weighted_first_filer_failure": float((R["first_fail"] * R["n_first_reg"]).sum() / R["n_first_reg"].sum()),
        "filing_weighted_baseline": float((R["base_fail"] * R["n_first_reg"]).sum() / R["n_first_reg"].sum()),
    }
    S = R.filter(pl.col("n_first") == 1)
    out["single_introducer"] = {"n_bigrams": S.height,
                                "share_introduced_by_failed_filing": float(S["first_fail"].mean()),
                                "class_year_failure_rate": float(S["base_fail"].mean())}
    L = R.filter(pl.col("n_later_reg") >= 5).with_columns((pl.col("first_fail") >= 0.5).alias("intro_failed"))
    out["later_users"] = {}
    for lab, f in (("all", L), ("introducer_failed", L.filter(pl.col("intro_failed"))),
                   ("introducer_survived", L.filter(~pl.col("intro_failed")))):
        w = f["n_later_reg"].cast(pl.Float64)
        out["later_users"][lab] = {
            "n_bigrams": f.height, "later_registrations": int(w.sum()),
            "weighted_later_survival": float((f["later_surv"] * w).sum() / w.sum()),
            "weighted_baseline_survival": float(((1 - f["later_base_fail"]) * w).sum() / w.sum()),
            "bigram_mean_later_survival": float(f["later_surv"].mean()),
            "bigram_mean_baseline_survival": float((1 - f["later_base_fail"]).mean())}
    # by year of introduction
    out["by_first_year"] = R.group_by("first").agg(
        pl.len().alias("n"), pl.col("first_fail").mean().alias("first_fail"),
        pl.col("base_fail").mean().alias("base_fail")).sort("first").to_dicts()
    log(f"[spillover] {out['first_filers']}  single {out['single_introducer']}")
    log(f"[later] {out['later_users']}")
    save("spillover", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
