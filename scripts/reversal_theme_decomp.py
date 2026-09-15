"""Does one internet theme carry the 2000-2004 reversal?

Eric's question: can we pull out an internet theme that carries the weight of
the mid-sample reversal (leading tech filings made 2000-2004 failing the
five-year proof LESS than lagging ones), so the paper can talk about how that
business theme differed?

Design, on the production T=50 scoring, settled registrations in 009/035/038/
042, filing era 2000-2004 (with 1995-1999 and 2008-2014 as comparison eras):

1. Reproduce the era baselines (raw top-minus-bottom lead-quintile contrast).
2. Split each era by the curated internet text flag (internet_breakout.py
   pattern): contrast within web and within non-web, raw and within
   class x cohort cells.
3. Theme x web cells for 2000-2004: per-cell contrast, base failure, and each
   cell's share of the era's leading fifth vs its share of all era filings.
4. Leave-one-out: era contrast excluding each large theme (quintiles
   reassigned on the remainder); excluding all web filings; excluding each
   theme's web half only. If dropping one theme (or one theme's web half)
   erases the negative contrast, that theme carries the reversal.
5. T=50 themes whose top-25 words contain internet-ish vocabulary, for naming.

Output: paper/results/reversal_theme_decomp.json (+ stderr log).
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
TECH = ["009", "035", "038", "042"]
EDGE = "2026-04-02"
RESOLVED_AGE = 9.0
WIDE_LO, WIDE_HI = 4.0, 8.5
ERAS = [("1995-1999", 1995, 1999), ("2000-2004", 2000, 2004),
        ("2008-2014", 2008, 2014)]
FOCUS = "2000-2004"
T = 50
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
MIN_CELL = 3_000
LOO_MIN = 30_000     # era filings a theme needs for leave-one-out

PATTERN = json.loads((RES / "internet_breakout.json").read_text())["pattern"]
WEBBY = ("internet", "online", "on-line", "web", "website", "websites",
         "electronic", "downloadable", "network", "computer")


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def contrast_arrays(q: np.ndarray, y: np.ndarray) -> dict | None:
    m1, m5 = q == 0, q == 4
    n1, n5 = int(m1.sum()), int(m5.sum())
    if n1 < 200 or n5 < 200:
        return None
    p1, p5 = float(y[m1].mean()), float(y[m5].mean())
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(len(y)), "base": float(y.mean()), "lift": p5 - p1,
            "se": se, "p1": p1, "p5": p5}


def contrast(df: pl.DataFrame, min_n: int = MIN_CELL) -> dict | None:
    if df.height < min_n:
        return None
    s = df.sort(["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal") - 1) * 5 // pl.len())
        .cast(pl.Int8).alias("q"))
    return contrast_arrays(s["q"].to_numpy(), s["failed"].to_numpy())


def contrast_within(df: pl.DataFrame, cells: list[str],
                    min_n: int = MIN_CELL) -> dict | None:
    if df.height < min_n:
        return None
    s = df.sort(cells + ["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal").over(cells) - 1) * 5
         // pl.len().over(cells)).cast(pl.Int8).alias("q"))
    return contrast_arrays(s["q"].to_numpy(), s["failed"].to_numpy())


def load_events() -> pl.DataFrame:
    return pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d",
                                                  strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())


def load_class(c: str, ev: pl.DataFrame, lda, vec) -> pl.DataFrame:
    tm = pl.read_parquet(
        PROC / f"tm_class{c}.parquet",
        columns=["serial_number", "filing_date", "registration_date",
                 "goods_services"]).filter(
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        & pl.col("goods_services").is_not_null())
    sc = pl.read_parquet(PROC / f"{SRC}_surprise_class{c}.parquet",
                         columns=["serial_number", "topic_dkl"]).filter(
        pl.col("topic_dkl").is_finite())
    d = tm.join(sc, on="serial_number", how="inner").unique(
        subset="serial_number")
    del tm, sc
    edge = pl.lit(EDGE).str.strptime(pl.Date, "%Y-%m-%d")
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d",
                                                 strict=False).alias("rd"),
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d",
                                           strict=False).alias("fd"),
    ).drop_nulls(["rd", "fd"]).filter(
        pl.col("rd").dt.offset_by(f"{int(RESOLVED_AGE * 365.25)}d") <= edge)
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
        pl.col("rd").dt.year().alias("ry"),
        pl.col("fd").dt.year().alias("fy"),
        pl.lit(c).alias("cls"),
        pl.col("goods_services").str.to_lowercase().str.contains(PATTERN)
        .alias("web"),
    ).with_columns(
        ((pl.col("age") >= WIDE_LO) & (pl.col("age") < WIDE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"))
    texts = d["goods_services"].to_list()
    dom = np.empty(len(texts), dtype=np.int16)
    for s in range(0, len(texts), 200_000):
        th = lda.transform(vec.transform(texts[s:s + 200_000]))
        dom[s:s + 200_000] = th.argmax(axis=1)
        del th
    d = d.with_columns(pl.Series("theme", dom)).drop(
        "goods_services", "filing_date", "registration_date")
    log(f"  [{c}] {d.height:,} settled ({100 * d['web'].mean():.1f}% web)")
    del texts
    gc.collect()
    return d


def main() -> int:
    m = joblib.load(PROC / "topic_model.joblib")
    lda, vocab = m["lda"], m["vocabulary"]
    vec = CountVectorizer(vocabulary=vocab, lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    words = json.loads((PROC / "topic_lda_meta.json").read_text())["top_words"]
    ev = load_events()
    d = pl.concat([load_class(c, ev, lda, vec) for c in TECH])
    del ev, m, lda, vec
    gc.collect()
    log(f"[frame] {d.height:,} settled technology registrations")

    out = {"scoring": SRC, "pattern": PATTERN, "n": int(d.height),
           "webby_themes": {}, "era_baseline": {}, "era_web_split": {},
           "focus_theme_web_cells": {}, "focus_leave_one_out": {},
           "focus_composition": {}, "theme_words": {}}

    # 5. Internet-ish themes by top words, for naming.
    for t, ws in words.items():
        hits = [w for w in ws[:25] if any(w == v or w.startswith(v + " ")
                or w.endswith(" " + v) for v in WEBBY)]
        if len(hits) >= 3:
            out["webby_themes"][t] = {"top6": ws[:6], "webby_hits": hits[:8]}

    # 1 + 2. Era baselines and the web split.
    for lab, lo, hi in ERAS:
        de = d.filter(pl.col("fy").is_between(lo, hi))
        out["era_baseline"][lab] = contrast(de)
        out["era_web_split"][lab] = {
            "web_share": float(de["web"].mean()),
            "web_raw": contrast(de.filter(pl.col("web"))),
            "web_within": contrast_within(de.filter(pl.col("web")),
                                          ["cls", "ry"]),
            "nonweb_raw": contrast(de.filter(~pl.col("web"))),
            "nonweb_within": contrast_within(de.filter(~pl.col("web")),
                                             ["cls", "ry"]),
        }
        b, w = out["era_baseline"][lab], out["era_web_split"][lab]
        log(f"  {lab}: all {100 * b['lift']:+.2f}  "
            f"web {100 * w['web_raw']['lift']:+.2f}  "
            f"nonweb {100 * w['nonweb_raw']['lift']:+.2f}  "
            f"(web share {100 * w['web_share']:.1f}%)")

    # Focus era frame.
    lo, hi = [(l, h) for lab, l, h in ERAS if lab == FOCUS][0]
    df = d.filter(pl.col("fy").is_between(lo, hi))
    del d
    gc.collect()

    # 3. Theme x web cells, and composition of the era's leading fifth.
    s = df.sort(["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal") - 1) * 5 // pl.len())
        .cast(pl.Int8).alias("q"))
    n_top = s.filter(pl.col("q") == 4).height
    comp = (s.group_by(["theme", "web"]).agg(
        pl.len().alias("n_all"),
        (pl.col("q") == 4).sum().alias("n_top"),
        pl.col("failed").mean().alias("base"))
        .with_columns((pl.col("n_all") / s.height).alias("share_all"),
                      (pl.col("n_top") / n_top).alias("share_top"))
        .sort("n_all", descending=True))
    for r in comp.iter_rows(named=True):
        key = f"t{int(r['theme'])}_{'web' if r['web'] else 'plain'}"
        cell = df.filter((pl.col("theme") == r["theme"])
                         & (pl.col("web") == r["web"]))
        out["focus_theme_web_cells"][key] = {
            "theme": int(r["theme"]), "web": bool(r["web"]),
            "n": int(r["n_all"]), "base": r["base"],
            "share_all": r["share_all"], "share_top": r["share_top"],
            "contrast": contrast(cell),
        }
        out["theme_words"][str(int(r["theme"]))] = words[str(int(r["theme"]))][:6]
    del s, comp
    gc.collect()

    # 4. Leave-one-out on the era contrast.
    big = [t for t, in df.group_by("theme").agg(pl.len().alias("n"))
           .filter(pl.col("n") >= LOO_MIN).select("theme").iter_rows()]
    out["focus_leave_one_out"]["baseline"] = contrast(df)
    out["focus_leave_one_out"]["drop_web_all"] = contrast(
        df.filter(~pl.col("web")))
    out["focus_leave_one_out"]["keep_web_only"] = contrast(
        df.filter(pl.col("web")))
    for t in big:
        out["focus_leave_one_out"][f"drop_t{t}"] = contrast(
            df.filter(pl.col("theme") != t))
        out["focus_leave_one_out"][f"drop_t{t}_webhalf"] = contrast(
            df.filter(~((pl.col("theme") == t) & pl.col("web"))))
    for k, r in out["focus_leave_one_out"].items():
        if r:
            log(f"  LOO {k}: {100 * r['lift']:+.2f} ({100 * r['se']:.2f})"
                f"  n {r['n']:,}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "reversal_theme_decomp.json").write_text(json.dumps(out, indent=1))
    log("[done] reversal_theme_decomp.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
