"""Pace-of-change quadrants (hypotheses 8-11) with and without a filing-year trend.

The battery estimated the lead penalty by quadrant of class-year technology pace
(theme-mix turnover) and market pace (filing-volume growth) without a trend. The
lead penalty grows over time, and fast-growth class-years are not spread evenly
over time, so this re-estimates the quadrant lead effects with the filing-year
trend x lead term held (same survival outcome, fixed effects, controls and
owner clustering as combined_model.py).

Output: paper/results/pace_quadrants_trend.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import combined_model as cm  # noqa: E402

LEVELS = ["calm", "technology leads", "market leads", "rough"]


def main() -> int:
    d = cm.frame()
    raw = pl.read_parquet(cm.PROC / "battery_frame.parquet", columns=["tech_pace", "mkt_pace"])
    d = d.with_columns(raw["tech_pace"].alias("tp_raw"), raw["mkt_pace"].alias("mp_raw"))
    tp, mp = d["tp_raw"].median(), d["mp_raw"].median()
    d = d.with_columns(
        pl.when(pl.col("tp_raw").is_null() | pl.col("mp_raw").is_null()).then(None)
        .when((pl.col("tp_raw") < tp) & (pl.col("mp_raw") < mp)).then(pl.lit("calm"))
        .when((pl.col("tp_raw") >= tp) & (pl.col("mp_raw") < mp)).then(pl.lit("technology leads"))
        .when((pl.col("tp_raw") < tp) & (pl.col("mp_raw") >= mp)).then(pl.lit("market leads"))
        .otherwise(pl.lit("rough")).alias("quad")).filter(pl.col("quad").is_not_null())
    for i, lv in enumerate(LEVELS):
        d = d.with_columns((pl.col("quad") == lv).cast(pl.Float64).alias(f"qd{i}"))
    des = cm.Design(d)
    y = d["surv"].to_numpy()
    out = {}
    for tag, extra_fac in (("no trend", []), ("with trend", ["trend"])):
        # quadrant-specific lead effects: lead x each quadrant dummy (no pooled lead term)
        X, nm, _ = cm.build_X(d, ["qd1", "qd2", "qd3"] + extra_fac, extra_fac)
        lead = X[:, 0].astype(np.float64)
        Q = np.column_stack([d[f"qd{i}"].to_numpy() * lead for i in range(4)])
        X2 = np.column_stack([X[:, 1:], Q]).astype(np.float32)
        nm2 = nm[1:] + [f"lead_{lv}" for lv in LEVELS]
        r = des.ols(y, X2, nm2)
        out[tag] = {lv: r[f"lead_{lv}"][:2] for lv in LEVELS}
        cm.log(f"  {tag}: " + "; ".join(f"{lv} {r[f'lead_{lv}'][0]:+.2f} ({r[f'lead_{lv}'][1]:.2f})"
                                        for lv in LEVELS))
    (cm.RES / "pace_quadrants_trend.json").write_text(json.dumps(out, indent=1))
    cm.log(f"peak {cm.peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
