"""Which Nice classes are alike? Cluster on descriptive profiles, then look at lead.

Five metrics per class, all from registrations 2002-2018 (the gate sample):
  survival      share passing the five-year proof of continued use (%)
  survival_sd   SD across registration years of the class's survival rate (pp)
  lead_mean     mean raw lead score (KL vs past minus KL vs future, T = 50)
  lead_var      variance of the raw lead score
  froth         mean five-year theme turnover of the class: half the summed
                absolute change in its theme mix between the five filing years
                before and the five after (0 = no change, 1 = complete)
The metrics are standardized and clustered (Ward linkage); k is chosen by the
silhouette over 3-7. The class's own lead effect (survival difference, most
leading minus most lagging filing, class x year FE, the paper's controls) is
shown beside the clusters but plays no part in forming them, so the table shows
whether classes that look alike also treat leading alike.

Output: paper/results/class_profiles.json
        paper/split/B_brief/class_profiles.tex
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

PROC, RES = cm.PROC, cm.RES
OUT_TEX = REPO / "paper" / "split" / "B_brief" / "class_profiles.tex"
NAMES = {1: "Chemicals", 2: "Paints", 3: "Cosmetics, cleaning", 4: "Lubricants, fuels", 5: "Pharmaceuticals",
         6: "Metal goods", 7: "Machinery", 8: "Hand tools", 9: "Electronics, software", 10: "Medical devices",
         11: "Lighting, heating", 12: "Vehicles", 13: "Firearms", 14: "Jewelry", 15: "Musical instruments",
         16: "Paper, printed matter", 17: "Rubber, plastics", 18: "Leather goods", 19: "Building materials",
         20: "Furniture", 21: "Housewares", 22: "Rope, fibers", 23: "Yarns, threads", 24: "Fabrics",
         25: "Clothing", 26: "Lace, ribbons", 27: "Floor coverings", 28: "Toys, sporting goods",
         29: "Meat, dairy, processed", 30: "Coffee, bakery, staples", 31: "Agricultural produce",
         32: "Beer, soft drinks", 33: "Wine, spirits", 34: "Tobacco, vaping", 35: "Advertising, business",
         36: "Insurance, finance", 37: "Construction, repair", 38: "Telecommunications", 39: "Transport, storage",
         40: "Treatment of materials", 41: "Education, entertainment", 42: "Technology, science services",
         43: "Hotels, restaurants", 44: "Medical, beauty services", 45: "Legal, personal services"}
METRICS = ["survival", "survival_sd", "lead_mean", "lead_var", "froth"]


def main() -> int:
    f = pl.read_parquet(PROC / "battery_frame.parquet",
                        columns=["serial_number", "cls", "reg_year", "cell", "owner_key", "failed1", "z",
                                 "tech_pace", "has_attorney", "itu", "log_len", "log_owner_n", "dom_us",
                                 "dom_cn", "basis_44e", "basis_66a"])
    raw = pl.concat([pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                                     columns=["serial_number", "topic_dkl"]).with_columns(pl.lit(c).alias("cls"))
                     for c in sorted(f["cls"].unique().to_list())]).unique(["serial_number", "cls"])
    f = f.join(raw, on=["serial_number", "cls"], how="left").with_columns(
        (100 * (1 - pl.col("failed1").cast(pl.Float64))).alias("surv"),
        ((pl.col("z").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5).alias("lead"))
    by_year = f.group_by("cls", "reg_year").agg(pl.col("surv").mean().alias("sy"))
    prof = f.group_by("cls").agg(
        pl.len().alias("n"), pl.col("surv").mean().alias("survival"),
        pl.col("topic_dkl").mean().alias("lead_mean"), pl.col("topic_dkl").var().alias("lead_var"),
        pl.col("tech_pace").mean().alias("froth"),
    ).join(by_year.group_by("cls").agg(pl.col("sy").std().alias("survival_sd")), on="cls").sort("cls")
    # each class's own lead effect (display only)
    eff = {}
    for c in prof["cls"].to_list():
        d = f.filter(pl.col("cls") == c)
        des = cm.Design(d)
        cols = [d["lead"].to_numpy()] + [d[x].cast(pl.Float64).fill_null(0).to_numpy() for x in
                                         ["has_attorney", "itu", "log_len", "log_owner_n", "dom_us", "dom_cn",
                                          "basis_44e", "basis_66a"]]
        r = des.ols(d["surv"].to_numpy(), np.column_stack(cols),
                    ["lead", "a", "b", "c", "d", "e", "f", "g", "h"])
        eff[c] = r["lead"][:2]
    prof = prof.with_columns(pl.col("cls").replace_strict({c: v[0] for c, v in eff.items()},
                                                          return_dtype=pl.Float64).alias("lead_effect"),
                             pl.col("cls").replace_strict({c: v[1] for c, v in eff.items()},
                                                          return_dtype=pl.Float64).alias("lead_effect_se"))
    X = prof.select(METRICS).to_numpy()
    Z = (X - X.mean(0)) / X.std(0)
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score
    L = linkage(Z, "ward")
    sil = {k: float(silhouette_score(Z, fcluster(L, k, "maxclust"))) for k in range(3, 8)}
    k = max(sil, key=sil.get)
    lab = fcluster(L, k, "maxclust")
    prof = prof.with_columns(pl.Series("cluster", lab))
    # does lead effect differ across descriptive clusters? precision-weighted ANOVA
    b, se = prof["lead_effect"].to_numpy(), prof["lead_effect_se"].to_numpy()
    w = 1 / se ** 2
    grand = (w * b).sum() / w.sum()
    Q_between = sum((w[lab == g].sum()) * ((w[lab == g] * b[lab == g]).sum() / w[lab == g].sum() - grand) ** 2
                    for g in np.unique(lab))
    from scipy.stats import chi2
    p_between = float(chi2.sf(Q_between, k - 1))
    Q_total = float((w * (b - grand) ** 2).sum())
    summary = prof.group_by("cluster").agg(
        pl.len().alias("classes"), *[pl.col(m).mean() for m in METRICS],
        ((pl.col("lead_effect") / pl.col("lead_effect_se") ** 2).sum() / (1 / pl.col("lead_effect_se") ** 2).sum())
        .alias("lead_effect_pooled"),
        (1 / (1 / pl.col("lead_effect_se") ** 2).sum()).sqrt().alias("lead_effect_pooled_se")).sort("cluster")
    out = {"k": k, "silhouette": sil, "Q_between": float(Q_between), "Q_total": Q_total, "p_between": p_between,
           "classes": prof.to_dicts(), "clusters": summary.to_dicts()}
    (RES / "class_profiles.json").write_text(json.dumps(out, indent=1))
    cm.log(f"k={k} silhouette {sil}; lead effect between clusters Q={Q_between:.1f} of {Q_total:.1f}, p={p_between:.3g}")
    for s in summary.to_dicts():
        cm.log(f"  cluster {s['cluster']}: {s['classes']} classes surv {s['survival']:.1f} sd {s['survival_sd']:.1f} "
               f"lead {s['lead_mean']:.3f} var {s['lead_var']:.3f} froth {s['froth']:.3f} -> lead effect "
               f"{s['lead_effect_pooled']:+.2f} ({s['lead_effect_pooled_se']:.2f})")
    for c in prof.sort(["cluster", "survival"]).to_dicts():
        cm.log(f"    [{c['cluster']}] {int(c['cls']):2d} {NAMES[int(c['cls'])]:26s} surv {c['survival']:.1f} "
               f"sd {c['survival_sd']:.1f} lead {c['lead_mean']:+.3f} var {c['lead_var']:.3f} "
               f"froth {c['froth']:.3f} eff {c['lead_effect']:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
