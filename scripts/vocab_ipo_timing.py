"""New-vocabulary debuts: funded, but listed? And is the road to listing lengthening?

The brief notes that AI and blockchain debuts raise private rounds about five
times as often as the average debut but list at the average rate. Two readings
compete. (a) Something specific to those vocabularies. (b) Right-censoring and
a lengthening road to listing: AI/blockchain debuts cluster in the last debut
years, which have had the least time to list. This separates them.

Frame: owners whose first registered, scored filing was made 2009-2018 (as in
curated_ladder.py), one row per owner. Outcomes: a Form D round after the
debut, appearance in SEC reporting, an IPO marker at any time, and an IPO
dated after the debut (with years from debut to IPO).

1. For each curated vocabulary, actual rates against EXPECTED rates: the
   all-debut rate in each debut year, weighted by the vocabulary's own
   distribution over debut years. Ratios above 1 mean the vocabulary out-does
   same-vintage debuts; this removes any difference in follow-up time.
2. The road to listing, by debut year: share of debuts listing within 3 and 5
   years of their debut, and the median years from debut to IPO among those
   that listed after debuting.

Output: paper/results/vocab_ipo_timing.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]

PATTERNS = {
    "internet": r"\binternet\b|\bonline\b|\bon-line\b|\bweb ?sites?\b|\bweb pages?\b|\bwebsites?\b|\bworld wide web\b|\be-?commerce\b|\belectronic commerce\b|\bweb portals?\b",
    "online retail": r"online retail|online store|e-?commerce|electronic commerce|online marketplace",
    "mobile apps": r"mobile app|mobile application|smartphone app|mobile device application|downloadable mobile",
    "cloud / SaaS": r"cloud computing|cloud-based|cloud based|software as a service|\bsaas\b|platform as a service",
    "social media": r"social media|social network|social networking",
    "ai": r"artificial intelligence|machine learning|deep learning|neural network|natural language processing|computer vision|predictive analytics|chatbots?",
    "blockchain": r"blockchain|cryptocurrenc|crypto asset|crypto token|\bbitcoin\b|non-fungible token|distributed ledger|digital currency|virtual currency|smart contract",
    "3d printing": r"3d print|three-dimensional print|additive manufactur",
    "drones": r"\bdrones?\b|unmanned aerial",
    "electronic cigarettes": r"electronic cigarette|e-cigarette|vaporizer|vape\b|vaping",
    "energy drinks": r"energy drink",
    "kombucha": r"kombucha",
    "cold brew": r"cold brew",
    "plant-based milk": r"plant-based milk|plant based milk|almond milk|oat milk|non-dairy milk|nondairy milk",
    "hard seltzer": r"hard seltzer|alcoholic seltzer|spiked seltzer",
    "cbd / hemp": r"cannabidiol|\bcbd\b|hemp-derived|hemp derived",
}


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet",
                         columns=["owner_name", "first_formd_date", "in_sec", "in_fsds",
                                  "in_8a", "ipo_date"])
    parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "owner_name", "filing_date",
                                          "registration_date", "goods_services"]).filter(
            pl.col("owner_name").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        tm = tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs"))
        tm = tm.with_columns([pl.col("gs").str.contains(p).alias(n) for n, p in PATTERNS.items()])
        parts.append(tm.select(["owner_name", "filing_date"] + list(PATTERNS)))
        del tm, sc
        gc.collect()
    d = pl.concat(parts)
    del parts
    debut = d.group_by("owner_name").agg(pl.col("filing_date").min().alias("dd"))
    first = d.join(debut, on="owner_name").filter(pl.col("filing_date") == pl.col("dd"))
    first = first.group_by("owner_name").agg(
        [pl.col("filing_date").min().alias("fd")] + [pl.col(n).any() for n in PATTERNS]
    ).with_columns(pl.col("fd").str.slice(0, 4).cast(pl.Int32).alias("fy")).filter(
        pl.col("fy").is_between(2009, 2018))
    x = first.join(fm, on="owner_name", how="left").with_columns(
        pl.col("fd").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fdd"),
        pl.col("ipo_date").cast(pl.Date, strict=False).alias("ipod"),
        pl.col("first_formd_date").cast(pl.Date, strict=False).alias("fmd"),
    ).with_columns(
        (pl.col("fmd").is_not_null() & (pl.col("fmd") >= pl.col("fdd"))).alias("funded"),
        (pl.col("in_sec").fill_null(0) + pl.col("in_fsds").fill_null(0) >= 1).alias("reporting"),
        ((pl.col("in_8a").fill_null(0) == 1) | pl.col("ipod").is_not_null()).alias("ipo"),
        (pl.col("ipod").is_not_null() & (pl.col("ipod") > pl.col("fdd"))).alias("ipo_after"),
        ((pl.col("ipod") - pl.col("fdd")).dt.total_days() / 365.25).alias("yrs_to_ipo"),
    )
    log(f"[frame] {x.height:,} debut owners 2009-2018")

    outs = ["funded", "reporting", "ipo", "ipo_after"]
    by_year = x.group_by("fy").agg([pl.col(o).mean().alias(o) for o in outs] + [pl.len().alias("n")]).sort("fy")
    base = {r["fy"]: r for r in by_year.iter_rows(named=True)}
    res = {"n": x.height, "by_debut_year": by_year.to_dicts(), "vocab": {}}
    for name in PATTERNS:
        g = x.filter(pl.col(name))
        if g.height < 30:
            continue
        w = g.group_by("fy").len()
        wt = {r["fy"]: r["len"] for r in w.iter_rows(named=True)}
        tot = sum(wt.values())
        row = {"n": g.height, "mean_debut_year": float(g["fy"].mean())}
        for o in outs:
            act = float(g[o].mean())
            exp = sum(base[y][o] * k for y, k in wt.items()) / tot
            row[o] = act
            row[o + "_expected"] = exp
            row[o + "_ratio"] = act / exp if exp > 0 else None
            row[o + "_count"] = int(g[o].sum())
        f = g.filter(pl.col("funded"))
        row["ipo_after_given_funded"] = float(f["ipo_after"].mean()) if f.height else None
        res["vocab"][name] = row
        log(f"  {name:22s} n={g.height:6,} yr={row['mean_debut_year']:.1f} "
            f"funded x{row['funded_ratio']:.1f}  report x{row['reporting_ratio']:.2f}  "
            f"ipo x{row['ipo_ratio']:.2f} ({row['ipo_count']})  ipo_after x{(row['ipo_after_ratio'] or 0):.2f} ({row['ipo_after_count']})")

    # the road to listing, by debut year
    road = []
    for y in range(2009, 2019):
        g = x.filter(pl.col("fy") == y)
        la = g.filter(pl.col("ipo_after"))
        road.append({"debut_year": y, "n": g.height,
                     "list_within_3y": float((g["ipo_after"] & (g["yrs_to_ipo"] <= 3)).fill_null(False).mean()),
                     "list_within_5y": float((g["ipo_after"] & (g["yrs_to_ipo"] <= 5)).fill_null(False).mean()),
                     "listed_after_debut": la.height,
                     "median_years_to_ipo": float(la["yrs_to_ipo"].median()) if la.height else None})
        r = road[-1]
        log(f"  debut {y}: n={g.height:,} within3={100*r['list_within_3y']:.3f}% within5={100*r['list_within_5y']:.3f}% "
            f"listed_after={la.height} median_yrs={r['median_years_to_ipo']}")
    res["road_to_listing"] = road
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "vocab_ipo_timing.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
