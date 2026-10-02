"""Product segments from the full theme mix, across Nice classes.

A Nice class is a legal category: class 9 holds software, phones and sunglasses.
A segment here is a cluster of registrations with similar theme mixes, wherever
they were filed. Clustering uses the square root of each 50-theme mix (so that
Euclidean distance is the Hellinger distance between mixes) and mini-batch
k-means with K segments, fitted on a sample and applied to all registrations.
Each segment is named by the themes that carry most of its centre.

Questions:
  1. Fit. How much of the variation in five-year-proof survival do class x year
     cells explain, against segment x year cells, against both? (R^2 of cell means.)
  2. Composition. How each large class splits across segments (class 9 especially),
     and how many classes each segment spans.
  3. The cost of leading. The lead effect (survival of the most leading minus the
     most lagging filing within class and year) estimated within each segment and
     within each class, with the spread of each set after shrinking toward the
     pooled effect; a larger spread across segments means segments sort the cost of
     leading better than classes do.

Output: paper/results/theme_segments.json, data/processed/theme_segments.parquet
"""
from __future__ import annotations

import json
import math
import re
import ast
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from combined_model import Design  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
CONTROLS = ["has_attorney", "itu", "log_len", "log_owner_n", "dom_us", "dom_cn", "basis_44e", "basis_66a"]


def labels() -> dict:
    src = (REPO / "scripts" / "themes_t50_page.py").read_text(encoding="utf-8")
    m = re.search(r"LABELS\s*=\s*(\{.*?\n\})", src, re.S) or re.search(r"LABELS\s*=\s*(\[.*?\n\])", src, re.S)
    L = ast.literal_eval(m.group(1))
    return {k: L[k] for k in range(50)}


def shrink(b, s):
    w = 1 / s ** 2
    mu = (w * b).sum() / w.sum()
    tau2 = max(0.0, float(((b - mu) ** 2 - s ** 2).mean()))
    return mu + tau2 / (tau2 + s ** 2) * (b - mu), math.sqrt(tau2)


def main() -> int:
    lab = labels()
    th = pl.read_parquet(PROC / "theme_mix_theta.parquet")
    tcols = [f"t{k}" for k in range(50)]
    X = np.sqrt(np.clip(th.select(tcols).to_numpy().astype(np.float32), 0, None))
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(600_000, len(X)), replace=False)
    km = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=5, random_state=0).fit(X[idx])
    seg = km.predict(X)
    cent = km.cluster_centers_ ** 2
    cent = cent / cent.sum(1, keepdims=True)
    names = {}
    for s in range(K):
        o = np.argsort(-cent[s])
        parts = [lab[o[0]]] + ([lab[o[1]]] if cent[s, o[1]] >= 0.15 else [])
        names[s] = " + ".join(parts)
    segs = pl.DataFrame({"serial_number": th["serial_number"], "cls": th["cls"], "seg": seg.astype(np.int16)})
    segs.write_parquet(PROC / "theme_segments.parquet")
    del X, th
    print(f"[segments] K={K}", flush=True)

    f = pl.read_parquet(PROC / "battery_frame.parquet",
                        columns=["serial_number", "cls", "cell", "reg_year", "owner_key", "failed1", "z"] + CONTROLS)
    f = f.join(segs, on=["serial_number", "cls"], how="inner").with_columns(
        (100 * (1 - pl.col("failed1").cast(pl.Float64))).alias("surv"),
        ((pl.col("z").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5).alias("lead"),
        (pl.col("seg").cast(pl.Utf8) + "_" + pl.col("reg_year").cast(pl.Utf8)).alias("segcell"),
        (pl.col("cell") + "|" + pl.col("seg").cast(pl.Utf8)).alias("bothcell"),
        *[pl.col(c).cast(pl.Float64) for c in ["has_attorney", "itu", "dom_us", "dom_cn", "basis_44e", "basis_66a"]])
    y = f["surv"].to_numpy()
    out: dict = {"K": K, "n": f.height}

    # 1. fit: R^2 of cell means
    def r2(cellcol):
        m = f.group_by(cellcol).agg(pl.col("surv").mean().alias("_m"), pl.len().alias("_n"))
        g = f.select(cellcol).join(m, on=cellcol, how="left")["_m"].to_numpy()
        return float(1 - ((y - g) ** 2).sum() / ((y - y.mean()) ** 2).sum()), int(m.height)
    out["r2"] = {"class_x_year": r2("cell"), "segment_x_year": r2("segcell"), "class_x_segment_x_year": r2("bothcell")}
    print("  R2:", {k: round(v[0], 4) for k, v in out["r2"].items()}, flush=True)

    # 2. composition
    comp = f.group_by("cls", "seg").len()
    tot = comp.group_by("cls").agg(pl.col("len").sum().alias("N"))
    comp = comp.join(tot, on="cls").with_columns((pl.col("len") / pl.col("N")).alias("share"))
    out["class_composition"] = {}
    for c in ["009", "025", "035", "042", "041", "005"]:
        rows = comp.filter(pl.col("cls") == c).sort("share", descending=True).head(6)
        out["class_composition"][c] = [{"segment": names[int(s)], "share": float(sh)}
                                       for s, sh in rows.select("seg", "share").iter_rows()]
    span = comp.filter(pl.col("share") >= 0.0).group_by("seg").agg(
        pl.col("len").sum().alias("n"),
        (pl.col("len").max() / pl.col("len").sum()).alias("top_class_share"),
        pl.col("cls").sort_by("len", descending=True).first().alias("top_class"),
        (pl.col("len") >= 0.05 * pl.col("len").sum()).sum().alias("classes_over_5pct"))
    surv = f.group_by("seg").agg(pl.col("surv").mean().alias("surv"))

    # 3. lead effect within each segment and within each class (same controls, cell FE)
    def lead_effects(groupcol):
        res = {}
        for (gv,), d in f.group_by([groupcol]):
            if d.height < 20000:
                continue
            des = Design(d)
            X_ = np.column_stack([d["lead"].to_numpy()] + [d[c].cast(pl.Float64).fill_null(0).to_numpy()
                                                           for c in CONTROLS])
            r = des.ols(d["surv"].to_numpy(), X_, ["lead"] + CONTROLS)
            res[gv] = (r["lead"][0], r["lead"][1], d.height)
        return res
    se_ = lead_effects("seg")
    ce_ = lead_effects("cls")
    for name, eff in (("segment", se_), ("class", ce_)):
        b = np.array([v[0] for v in eff.values()]); s = np.array([v[1] for v in eff.values()])
        shr, tau = shrink(b, s)
        out[f"lead_effect_{name}"] = {"n_groups": len(eff), "tau": tau,
                                      "shrunk_sd": float(np.std(shr)), "raw_sd": float(np.std(b))}
        print(f"  lead effect across {name}s: {len(eff)} groups, true spread (tau) {tau:.2f} pp", flush=True)
    seg_rows = []
    for r in span.join(surv, on="seg").sort("n", descending=True).iter_rows(named=True):
        e = se_.get(r["seg"])
        seg_rows.append({"seg": int(r["seg"]), "name": names[int(r["seg"])], "n": int(r["n"]),
                         "survival": float(r["surv"]), "top_class": r["top_class"],
                         "top_class_share": float(r["top_class_share"]),
                         "classes_over_5pct": int(r["classes_over_5pct"]),
                         "lead_effect": e[0] if e else None, "lead_effect_se": e[1] if e else None})
    out["segments"] = seg_rows
    (RES / "theme_segments.json").write_text(json.dumps(out, indent=1))
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
