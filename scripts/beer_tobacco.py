"""Why is leading rewarded in class 32 (beer, soft drinks) but penalized in 34 (tobacco)?

Same survival model as combined_model.py (class x registration-year FE, owner-
clustered SEs, lead percentile within cell, the paper's controls), run inside
classes 32, 33 and 34: overall, by registration period, and split by the
product vocabulary the filing uses. Also lists the themes most over-represented
in each class's most leading fifth.

Output: paper/results/beer_tobacco.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import combined_model as cm  # noqa: E402

PROC, RES = cm.PROC, cm.RES
SPLITS = {
    "032": {"beer": r"\bbeers?\b|\bales?\b|\blagers?\b|stouts?\b|\bipa\b",
            "energy / sports drinks": r"energy drink|sports drink|isotonic",
            "water / seltzer": r"\bwaters?\b|seltzer|sparkling",
            "juice / soft drinks": r"juice|soft drink|soda|carbonated"},
    "033": {"wine": r"\bwines?\b", "spirits": r"whisk|vodka|\bgin\b|\brum\b|tequila|liqueur|spirits|bourbon",
            "hard seltzer / cider": r"seltzer|cider"},
    "034": {"e-cigarettes / vaping": r"electronic cigarette|e-cigarette|vapor|vaping|\bvape|e-liquid|liquid nicotine",
            "cigars": r"\bcigars?\b", "cigarettes": r"\bcigarettes?\b", "hookah / shisha": r"hookah|shisha|molasses tobacco"},
}
PERIODS = [(2002, 2007), (2008, 2012), (2013, 2018)]


def fit(d: pl.DataFrame, extra: dict[str, np.ndarray]) -> dict:
    des = cm.Design(d)
    lead = d["lead"].to_numpy()
    cols, names = [lead], ["lead"]
    for k, v in extra.items():
        vc = v - v.mean()
        cols += [vc, vc * lead]
        names += [k, f"{k}:lead"]
    for c in ["has_attorney", "itu", "log_len", "log_owner_n", "dom_us", "dom_cn", "basis_44e", "basis_66a"]:
        cols.append(d[c].cast(pl.Float64).fill_null(0).to_numpy()); names.append(c)
    r = des.ols(d["surv"].to_numpy(), np.column_stack(cols), names)
    out = {"n": d.height, "lead": r["lead"][:2]}
    for k, v in extra.items():
        m = float(v.mean())
        if f"{k}:lead" in r:
            out[k] = {"share": m, "lead_with": cm.lincomb(r, {"lead": 1, f"{k}:lead": 1 - m}),
                      "diff": r[f"{k}:lead"][:2], "survival": r[k][:2]}
    return out


def main() -> int:
    f = pl.read_parquet(PROC / "battery_frame.parquet").filter(pl.col("cls").is_in(list(SPLITS)))
    f = f.with_columns(
        (100 * (1 - pl.col("failed1").cast(pl.Float64))).alias("surv"),
        ((pl.col("z").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5).alias("lead"),
        pl.col("has_attorney").cast(pl.Float64), pl.col("itu").cast(pl.Float64))
    gs = pl.concat([pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "goods_services"])
                    .with_columns(pl.lit(c).alias("cls")) for c in SPLITS]).unique(["serial_number", "cls"])
    f = f.join(gs, on=["serial_number", "cls"], how="left").with_columns(
        pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs"))
    m = joblib.load(PROC / "topic_model.joblib")
    vocab = np.array(sorted(m["vocabulary"], key=m["vocabulary"].get))
    top_words = {k: ", ".join(vocab[np.argsort(-m["lda"].components_[k])[:6]]) for k in range(50)}
    out = {}
    for c, splits in SPLITS.items():
        d = f.filter(pl.col("cls") == c)
        res = {"overall": fit(d, {})}
        res["by_period"] = {f"{lo}-{hi}": fit(d.filter(pl.col("reg_year").is_between(lo, hi)), {})
                            for lo, hi in PERIODS}
        res["by_product"] = fit(d, {k: d["gs"].str.contains(p).cast(pl.Float64).to_numpy()
                                    for k, p in splits.items()})
        # product mix by period, and themes over-represented among leading filings
        res["product_share_by_period"] = {
            f"{lo}-{hi}": {k: float(d.filter(pl.col("reg_year").is_between(lo, hi))["gs"].str.contains(p).mean())
                           for k, p in splits.items()} for lo, hi in PERIODS}
        top = d.filter(pl.col("q") == 4).group_by("top_theme").len().rename({"len": "n_top"})
        allk = d.group_by("top_theme").len()
        ov = allk.join(top, on="top_theme", how="left").with_columns(
            (pl.col("n_top").fill_null(0) / pl.col("n_top").sum() / (pl.col("len") / pl.col("len").sum()))
            .alias("ratio")).filter(pl.col("len") >= 500).sort("ratio", descending=True).head(4)
        res["leading_themes"] = [{"theme": int(t), "words": top_words[int(t)], "ratio": float(r_)}
                                 for t, r_ in ov.select("top_theme", "ratio").iter_rows()]
        out[c] = res
        cm.log(f"class {c}: lead {res['overall']['lead'][0]:+.2f} ({res['overall']['lead'][1]:.2f}); periods "
               + ", ".join(f"{k} {v['lead'][0]:+.1f}" for k, v in res["by_period"].items()))
        for k in splits:
            v = res["by_product"].get(k)
            if v:
                cm.log(f"   {k:24s} share {100*v['share']:.0f}%  lead effect {v['lead_with'][0]:+.1f} "
                       f"({v['lead_with'][1]:.1f})  survival {v['survival'][0]:+.1f}")
        cm.log("   shares by period: " + json.dumps({p: {k: round(100 * x) for k, x in s.items()}
                                                      for p, s in res["product_share_by_period"].items()}))
        for t in res["leading_themes"]:
            cm.log(f"   leading theme {t['theme']} x{t['ratio']:.1f}: {t['words']}")
    (RES / "beer_tobacco.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
