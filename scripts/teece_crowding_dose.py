"""Test 2. Does realized later entry explain the cost of leading?

Realized entry into a registration's primary theme in its class after filing:
  (a) entry_share = log((s_post + eps) / (s_pre + eps)), the theme's share of the
      class's filings in years fy+1..fy+5 against fy-5..fy-1 (eps = 0.001; 0.005 as a check)
  (b) entry_owners = log(1 + number of owners debuting in the class in fy+1..fy+5
      whose debut-year filings in the class carry the theme as primary theme).
      Owner = normalized owner name (norm_owner); class debut = owner's first
      filing year in that class among all scored filings (registered or not).
      Sensitivity: entry_owners_growth = log(1+N post) - log(1+N in fy-5..fy-1).
Models (survival in points, controls, class x registration-year FE, owner-clustered):
  surv ~ entry ; surv ~ lead ; surv ~ lead + entry. Mediation = change in the lead
  coefficient when entry is held. Plus cell-adjusted survival by deciles of each
  entry measure.

Output: paper/results/teece_crowding_dose.json
"""
from __future__ import annotations

import gc

import numpy as np
import polars as pl

from gate_decisive_regression import norm_owner
from teece_common import CLASSES, CONTROLS, PROC, cell_ols, load_frame, log, mem_wait, save

EPS = 0.001


def new_owner_counts() -> pl.DataFrame:
    """(cls, top_theme, fy) -> number of owners whose class debut year is fy and whose
    debut-year filings in the class include that primary theme."""
    parts = []
    for c in CLASSES:
        th = pl.read_parquet(PROC / "theme_full" / f"theme_class{c}.parquet", columns=["serial_number", "fy", "top_theme"])
        tm = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "owner_name"]).unique("serial_number")
        x = th.join(tm, on="serial_number", how="inner").with_columns(norm_owner(pl.col("owner_name")).alias("ok")) \
            .filter(pl.col("ok") != "").drop("owner_name")
        deb = x.group_by("ok").agg(pl.col("fy").min().alias("dy"))
        x = x.join(deb, on="ok").filter(pl.col("fy") == pl.col("dy"))
        cnt = x.group_by("top_theme", "fy").agg(pl.col("ok").n_unique().alias("n_new")).with_columns(pl.lit(c).alias("cls"))
        parts.append(cnt)
        del th, tm, x, deb
        gc.collect()
    return pl.concat(parts)


def main() -> int:
    mem_wait("start")
    cnt = new_owner_counts()
    log(f"[new owners] {cnt.height:,} class-theme-years, {int(cnt['n_new'].sum()):,} debut owner-theme pairs")
    years = list(range(1975, 2027))
    grid = cnt.select("cls").unique().join(pl.DataFrame({"top_theme": list(range(50))}, schema={"top_theme": pl.Int16}),
                                           how="cross").join(pl.DataFrame({"fy": years}, schema={"fy": pl.Int32}), how="cross")
    p = grid.join(cnt.with_columns(pl.col("top_theme").cast(pl.Int16), pl.col("fy").cast(pl.Int32)),
                  on=["cls", "top_theme", "fy"], how="left").with_columns(pl.col("n_new").fill_null(0)) \
        .sort(["cls", "top_theme", "fy"])
    ov = ["cls", "top_theme"]
    p = p.with_columns(
        pl.sum_horizontal([pl.col("n_new").shift(-k).over(ov) for k in range(1, 6)]).alias("new_post"),
        pl.sum_horizontal([pl.col("n_new").shift(k).over(ov) for k in range(1, 6)]).alias("new_pre"),
    ).select("cls", "top_theme", "fy", "new_post", "new_pre")

    mem_wait("frame")
    d = load_frame(["top_theme", "s_pre", "s_post"]).join(p, on=["cls", "top_theme", "fy"], how="left")
    d = d.with_columns(
        (((pl.col("s_post") + EPS) / (pl.col("s_pre") + EPS)).log()).alias("entry_share"),
        (((pl.col("s_post") + 0.005) / (pl.col("s_pre") + 0.005)).log()).alias("entry_share_eps005"),
        (pl.col("new_post").cast(pl.Float64) + 1).log().alias("entry_owners"),
        ((pl.col("new_post").cast(pl.Float64) + 1).log() - (pl.col("new_pre").cast(pl.Float64) + 1).log())
        .alias("entry_owners_growth"),
    ).drop_nulls(["entry_share", "entry_owners"])
    measures = ["entry_share", "entry_owners", "entry_owners_growth", "entry_share_eps005"]
    # standardize so coefficients are per SD
    d = d.with_columns([((pl.col(m) - pl.col(m).mean()) / pl.col(m).std()).alias(m + "_sd") for m in measures])
    out = {"n": d.height, "eps": EPS,
           "describe": {m: {"mean": float(d[m].mean()), "sd": float(d[m].std()),
                            "p10": float(d[m].quantile(0.1)), "p50": float(d[m].quantile(0.5)),
                            "p90": float(d[m].quantile(0.9))} for m in measures},
           "corr_with_lead": {m: float(np.corrcoef(d["lead"].to_numpy(), d[m].to_numpy())[0, 1]) for m in measures},
           "spec": "surv (pts) ~ lead_pct and/or entry (per SD) + controls | class x reg-year FE; owner-clustered"}
    # within-cell correlation (what the FE models use)
    for m in measures:
        dm = d.select((pl.col("lead") - pl.col("lead").mean().over("cell")).alias("a"),
                      (pl.col(m) - pl.col(m).mean().over("cell")).alias("b"))
        out["corr_with_lead"][m + "_within_cell"] = float(np.corrcoef(dm["a"].to_numpy(), dm["b"].to_numpy())[0, 1])
    log(f"corr {out['corr_with_lead']}")

    base = cell_ols(d, "surv", ["lead"] + CONTROLS)
    out["lead_only"] = base["lead"]
    out["models"] = {}
    for m in measures:
        mem_wait(m)
        r1 = cell_ols(d, "surv", [m + "_sd"] + CONTROLS)
        r2 = cell_ols(d, "surv", ["lead", m + "_sd"] + CONTROLS)
        out["models"][m] = {"entry_only": r1[m + "_sd"], "both_lead": r2["lead"], "both_entry": r2[m + "_sd"],
                            "lead_shrink_share": 1 - r2["lead"]["b"] / base["lead"]["b"]}
        log(f"{m}: entry alone {r1[m+'_sd']}  both: lead {r2['lead']} entry {r2[m+'_sd']}")
    # all three main measures together
    r3 = cell_ols(d, "surv", ["lead", "entry_share_sd", "entry_owners_sd"] + CONTROLS)
    out["models"]["lead_plus_share_plus_owners"] = {k: r3[k] for k in ("lead", "entry_share_sd", "entry_owners_sd")}

    # deciles, cell-adjusted survival (surv - cell mean + grand mean)
    g = float(d["surv"].mean())
    d = d.with_columns((pl.col("surv") - pl.col("surv").mean().over("cell") + g).alias("adj"))
    out["deciles"] = {}
    for m in ("entry_share", "entry_owners"):
        dd = d.with_columns(((pl.col(m).rank("ordinal") - 1) * 10 // pl.len()).alias("dec")).group_by("dec").agg(
            pl.col(m).mean().alias("mean_entry"), pl.col("adj").mean().alias("surv_adj"),
            pl.col("lead").mean().alias("mean_lead"), pl.len().alias("n")).sort("dec")
        out["deciles"][m] = dd.to_dicts()
        log(f"{m} deciles: " + ", ".join(f"{r['surv_adj']:.1f}" for r in out["deciles"][m]))
    save("crowding_dose", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
