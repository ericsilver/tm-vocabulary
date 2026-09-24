"""Lead read as crowding: how much did a filing's own main theme grow afterwards?

A filing leads when its class's later filings look more like it than its
class's earlier filings did. In plain terms, the kind of offering it described
became more common among later entrants -- its competition grew after it
entered. This puts a number on that reading.

For each class: a sample of scored registrations 2002-2018 is given its
dominant theme under the production model; the class's theme shares by filing
year (share of filings whose dominant theme is k) are estimated from a sample
of each year's filings. For each registration, its dominant theme's mean
share over the five filing years before its own and the five after gives the
theme's growth around the filing. Registrations are cut into lead fifths
within class and registration year, and growth is averaged by fifth.

Output: paper/results/lead_crowding.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
REG_PER_CLASS = 12_000
PER_YEAR = 1_500
W = 5


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    m = joblib.load(PROC / "topic_model.joblib")
    lda = m["lda"]
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    dom = lambda texts: lda.transform(vec.transform(texts)).argmax(axis=1)
    rng = np.random.default_rng(3)
    rows = []
    for p in sorted(PROC.glob("tm_class*.parquet")):
        c = p.stem.replace("tm_class", "")
        sp = PROC / f"rolling_surprise_class{c}.parquet"
        if not c.isdigit() or not sp.exists():
            continue
        d = pl.read_parquet(p, columns=["serial_number", "filing_date", "registration_date",
                                        "goods_services"]).filter(
            pl.col("goods_services").is_not_null() & (pl.col("goods_services").str.len_chars() > 0)
        ).with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
                       pl.col("registration_date").fill_null("").str.slice(0, 4)
                       .cast(pl.Int32, strict=False).alias("ry")).unique("serial_number")
        # class theme shares by filing year
        shares = {}
        for y in range(1990, 2025):
            g = d.filter(pl.col("fy") == y)["goods_services"]
            if len(g) < 200:
                continue
            idx = rng.choice(len(g), size=min(PER_YEAR, len(g)), replace=False)
            k = dom(g[np.sort(idx)].to_list())
            shares[y] = np.bincount(k, minlength=lda.n_components) / len(k)
        # scored registrations
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        r = d.filter(pl.col("ry").is_between(2002, 2018)).join(sc, on="serial_number")
        if r.height < 2_000:
            continue
        r = r.sample(n=min(REG_PER_CLASS, r.height), seed=7)
        k = dom(r["goods_services"].to_list())
        fy = r["fy"].to_numpy()
        before, after = np.full(len(k), np.nan), np.full(len(k), np.nan)
        for i, (kk, y) in enumerate(zip(k, fy)):
            b = [shares[t][kk] for t in range(y - W, y) if t in shares]
            a = [shares[t][kk] for t in range(y + 1, y + W + 1) if t in shares]
            if len(b) == W and len(a) == W:
                before[i], after[i] = np.mean(b), np.mean(a)
        rows.append(r.select("serial_number", "ry", "topic_dkl").with_columns(
            pl.lit(c).alias("cls"), pl.Series("before", before), pl.Series("after", after)))
        log(f"[{c}] {r.height:,} registrations")
    df = pl.concat(rows).filter(pl.col("before").is_not_nan() & pl.col("after").is_not_nan()
                                ).filter(pl.col("before") > 0).sort(
        ["cls", "ry", "topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal").over(["cls", "ry"]) - 1) * 5
         // pl.len().over(["cls", "ry"])).cast(pl.Int8).alias("q")).with_columns(
        (pl.col("after") - pl.col("before")).alias("change_pts"),
        (pl.col("after") / pl.col("before")).alias("ratio"))
    by_q = df.group_by("q").agg(
        pl.len().alias("n"),
        pl.col("before").mean().alias("share_before"),
        pl.col("after").mean().alias("share_after"),
        pl.col("change_pts").mean().alias("change_pts"),
        pl.col("ratio").median().alias("median_ratio"),
        (pl.col("after") > pl.col("before")).mean().alias("share_growing")).sort("q")
    out = {"n": df.height, "by_lead_fifth": by_q.to_dicts()}
    for r_ in out["by_lead_fifth"]:
        log(f"  Q{r_['q']+1}: before {100*r_['share_before']:.1f}% after {100*r_['share_after']:.1f}% "
            f"change {100*r_['change_pts']:+.2f}pts median ratio {r_['median_ratio']:.2f} "
            f"growing {100*r_['share_growing']:.0f}%")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lead_crowding.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
