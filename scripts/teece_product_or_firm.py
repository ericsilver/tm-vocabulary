"""Test 1. Is the cost of leading in the product or in the firm?

The lead effect (survival, most leading minus most lagging filing in the same
class x registration-year cell, with the paper's controls) is estimated on owners
with 2+ registrations (and 5+) twice: with class x year FE only, and with owner FE
absorbed jointly with class x year FE (alternating projections; coefficients
change by <1e-4 at convergence). If the effect survives owner FE, it is a
property of the filing (the product), not of who files it.

Within-owner pairs: owners with at least one leading (top two lead fifths) and
one lagging (bottom two fifths) registration; difference of the owner's mean
survival in the two groups, raw and after subtracting each filing's
class x year cell mean survival.

Output: paper/results/teece_product_or_firm.json
"""
from __future__ import annotations

import gc
import math

import numpy as np
import polars as pl

from teece_common import CONTROLS, cell_ols, codes, load_frame, log, mem_wait, save, twoway_ols


def main() -> int:
    mem_wait("start")
    d = load_frame()
    d = d.with_columns(pl.len().over("owner_key").alias("own_n"),
                       pl.col("surv").mean().over("cell").alias("cell_mean"))
    xs = ["lead"] + CONTROLS
    out = {"n_frame": d.height, "spec": "surv ~ lead_pct + controls | FE, SE clustered by owner"}
    out["full_frame_cell_fe"] = cell_ols(d, "surv", xs)["lead"]
    log(f"full frame lead {out['full_frame_cell_fe']}")
    for k in (2, 5):
        mem_wait(f"owners {k}+")
        s = d.filter(pl.col("own_n") >= k)
        r_cell = cell_ols(s, "surv", xs)
        y = s["surv"].to_numpy().astype(np.float64)
        X = np.column_stack([s[c].cast(pl.Float64).to_numpy() for c in xs])
        own = codes(s["owner_key"])
        r_two = twoway_ols(y, X, xs, codes(s["cell"]), own, own, tol=1e-4)
        out[f"owners_{k}plus"] = {
            "n": s.height, "owners": int(own.max()) + 1,
            "cell_fe": {"lead": r_cell["lead"], "n": r_cell["_n"], "owners": r_cell["_owners"]},
            "cell_and_owner_fe": {"lead": r_two["lead"], "n": r_two["_n"], "owners": r_two["_owners"],
                                  "iterations": r_two["_iterations"]},
            "controls_owner_fe": {c: r_two[c] for c in CONTROLS},
        }
        log(f"[{k}+] n={s.height:,} cellFE {r_cell['lead']}  +ownerFE {r_two['lead']} it={r_two['_iterations']}")
        del s, X, y, own
        gc.collect()

    # within-owner pairs
    d = d.with_columns((pl.col("surv") - pl.col("cell_mean")).alias("adj"),
                       pl.when(pl.col("q") >= 3).then(pl.lit("lead")).when(pl.col("q") <= 1)
                       .then(pl.lit("lag")).alias("grp"))
    g = d.filter(pl.col("grp").is_not_null()).group_by("owner_key", "grp").agg(
        pl.col("surv").mean().alias("s"), pl.col("adj").mean().alias("a"), pl.len().alias("m"))
    w = g.pivot(on="grp", index="owner_key", values=["s", "a", "m"]).drop_nulls()
    pairs = {}
    for lab, col in (("raw", "s"), ("cell_adjusted", "a")):
        diff = (w[f"{col}_lead"] - w[f"{col}_lag"]).to_numpy()
        hm = (2 / (1 / w["m_lead"].to_numpy() + 1 / w["m_lag"].to_numpy()))
        wm = float(np.sum(hm * diff) / hm.sum())
        # weighted SE (owners independent)
        wse = float(math.sqrt(np.sum(hm ** 2 * (diff - wm) ** 2)) / hm.sum())
        pairs[lab] = {"mean_diff_lead_minus_lag": float(diff.mean()),
                      "se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
                      "weighted_mean_diff": wm, "weighted_se": wse}
    pairs["owners"] = w.height
    pairs["registrations_leading"] = int(w["m_lead"].sum())
    pairs["registrations_lagging"] = int(w["m_lag"].sum())
    pairs["mean_survival_lagging"] = float(w["s_lag"].mean())
    pairs["note"] = ("leading = lead fifths 4-5, lagging = fifths 1-2 within cell; equal-weight mean over owners; "
                     "weighted = harmonic mean of the owner's leading and lagging counts")
    out["within_owner_pairs"] = pairs
    log(f"pairs {pairs}")
    # reconcile: the same leading-vs-lagging contrast as a binary regressor, on the pair owners'
    # leading and lagging rows only, with owner + cell FE, without and with controls
    mem_wait("pair regression")
    s = d.filter(pl.col("grp").is_not_null()).join(w.select("owner_key"), on="owner_key", how="semi").with_columns(
        (pl.col("grp") == "lead").cast(pl.Float64).alias("leading"))
    y = s["surv"].to_numpy().astype(np.float64)
    own = codes(s["owner_key"]); cel = codes(s["cell"])
    for lab, xs2 in (("no_controls", ["leading"]), ("controls", ["leading"] + CONTROLS)):
        X = np.column_stack([s[c].cast(pl.Float64).to_numpy() for c in xs2])
        r = twoway_ols(y, X, xs2, cel, own, own)
        pairs[f"owner_cell_fe_{lab}"] = {"leading_minus_lagging": r["leading"], "n": r["_n"], "owners": r["_owners"]}
        log(f"pair regression {lab}: {r['leading']}")
    del s, X, y
    save("product_or_firm", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
