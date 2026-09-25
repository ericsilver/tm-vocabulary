"""Post-LASSO re-estimates on all registrations.

combined_model.py ranks the lead interactions by the order they enter the LASSO
path. Its one-standard-error penalty keeps no term (the out-of-sample gain is
within fold-to-fold noise for a binary outcome), so the parsimonious model here
re-estimates the first TOP terms to enter; the cross-validated set is
re-estimated as well. Same outcome, fixed effects, controls and clustering as
combined_model.py; p-values after selection are optimistic.

Output: adds post_lasso_top and post_lasso_cvmin to paper/results/combined_model.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import combined_model as cm  # noqa: E402

TOP = 10


def main() -> int:
    R = json.loads((cm.RES / "combined_model.json").read_text())
    d = cm.frame()
    y = d["surv"].to_numpy()
    des = cm.Design(d)
    fac = [k for k, _, _ in cm.BINARY] + [k for k, _, _ in cm.CONTINUOUS]
    grp_of = {c: g for g, (name, members) in enumerate(R["class_groups"].items()) for c in members}
    gnames = list(R["class_groups"].keys())
    cls = d["cls"].to_numpy()
    gcode = d["cls"].replace_strict(grp_of, return_dtype=pl.Int32).to_numpy()
    Xm, nmm, _ = cm.build_X(d, fac, [])
    lead = Xm[:, 0]

    def term(k):
        if k.startswith("class "):
            v = (cls == k[6:]).astype(float)
        elif k in gnames:
            v = (gcode == gnames.index(k)).astype(float)
        else:
            v = d[k].to_numpy().astype(float)
        return (v - v.mean()) * lead

    order = [k for k, _ in R["lasso"]["entry_order"]]
    for tag, terms in (("post_lasso_top", order[:TOP]), ("post_lasso_cvmin", R["lasso"]["selected_min"])):
        X = np.column_stack([Xm] + [term(k) for k in terms])
        r = des.ols(y, X, nmm + [f"{k}:lead" for k in terms])
        R[tag] = {"lead": r["lead"], "terms": {k: r[f"{k}:lead"] for k in terms if f"{k}:lead" in r},
                  "mains": {f: r.get(f) for f in fac}}
        cm.log(f"  {tag}: lead {r['lead'][0]:+.2f}; " + "; ".join(
            f"{k} {r[f'{k}:lead'][0]:+.2f} ({r[f'{k}:lead'][1]:.2f})" for k in terms[:TOP] if f"{k}:lead" in r))
        del X
        gc.collect()
    (cm.RES / "combined_model.json").write_text(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
