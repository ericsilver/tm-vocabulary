"""Measure-section facts recomputed on the production scoring.

Paper A's measure section quotes a few descriptive correlations that were
computed on earlier builds (the T = 200 theme build, or class 009 alone). This
recomputes each on the production build -- 50 global themes, per-filing
windows (rolling_surprise_class*.parquet) -- and, for the term-scored side, on
the per-filing term build (termroll_surprise_class*.parquet), so the paper can
quote numbers from the scoring it actually uses.

  - corr(A, log distinct terms), term-scored and theme-scored, classes 009,
    035 and all classes pooled (identical filings in each comparison)
  - corr(K-, K+) and corr(A, L), theme-scored, 009, 035 and pooled
  - lead magnitude's spread relative to the levels' spread

Output: paper/results/measure_facts_production.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"


def load(cls: str) -> pl.DataFrame | None:
    th, te = PROC / f"rolling_surprise_class{cls}.parquet", PROC / f"termroll_surprise_class{cls}.parquet"
    if not (th.exists() and te.exists()):
        return None
    a = pl.read_parquet(th, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"]).rename(
        {"topic_kl_vs_past": "kp", "topic_kl_vs_future": "kf"})
    b = pl.read_parquet(te, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "n_terms"]).rename(
        {"topic_kl_vs_past": "tkp", "topic_kl_vs_future": "tkf"})
    d = a.join(b, on="serial_number", how="inner").filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in ("kp", "kf", "tkp", "tkf")])
        & (pl.col("n_terms") > 0)).with_columns(
        ((pl.col("kp") + pl.col("kf")) / 2).alias("A"), (pl.col("kp") - pl.col("kf")).alias("L"),
        ((pl.col("tkp") + pl.col("tkf")) / 2).alias("tA"),
        pl.col("n_terms").cast(pl.Float64).log().alias("logk"), pl.lit(cls).alias("cls"))
    return d.select("cls", "kp", "kf", "A", "L", "tA", "logk")


def facts(d: pl.DataFrame) -> dict:
    c = lambda x, y: float(d.select(pl.corr(x, y)).item())
    return {"n": d.height,
            "r_termA_logk": c("tA", "logk"), "r_themeA_logk": c("A", "logk"),
            "r_Kminus_Kplus": c("kp", "kf"), "r_A_L": c("A", "L"),
            "sd_L_over_sd_levels": float(d["L"].std() / ((d["kp"].std() + d["kf"].std()) / 2))}


def main() -> int:
    out, parts = {}, []
    for i in range(1, 46):
        cls = f"{i:03d}"
        d = load(cls)
        if d is None:
            continue
        if cls in ("009", "035"):
            out[cls] = facts(d)
            print(cls, out[cls])
        parts.append(d)
    pooled = pl.concat(parts)
    out["pooled"] = facts(pooled)
    # pooled figures computed within class (demeaned by class) as well, since
    # every estimate in the paper is within class
    w = pooled.with_columns([(pl.col(c) - pl.col(c).mean().over("cls")).alias(c)
                             for c in ("kp", "kf", "A", "L", "tA", "logk")])
    out["pooled_within_class"] = facts(w)
    print("pooled", out["pooled"]); print("within-class", out["pooled_within_class"])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "measure_facts_production.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
