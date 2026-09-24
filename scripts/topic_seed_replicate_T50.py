"""Is the production 50-theme fit stable across seeds?

topic_seed_replicate.py established that at T=200 a second seed reproduces the
per-filing scores only at r = 0.79 on lead, so most of the apparent resolution
sensitivity is fitting noise. The production scoring uses T=50, and nothing so
far says how stable a 50-theme fit is. This answers that, apples to apples with
the T=200 test: same cached design matrix and vocabulary, same fitting routine
(topic_seed_replicate.fit), same class-009 scorer (topic_resolution_sweep).

Three comparisons:
  scorer check  production T=50 model, sweep scorer  vs  the production score
                file (should be ~1; confirms the scorer is not a confound)
  seed          production T=50 model  vs  a seed-7 refit, both sweep-scored:
                per-filing r on lead and atypicality, and the five-year-proof
                lead and atypicality contrasts on the identical sample
  themes        seed-42 vs seed-7 themes matched one-to-one (Hungarian on the
                cosine of their word distributions); the same for the existing
                T=200 pair, for comparison

Nothing existing is overwritten.

Output: paper/results/topic_seed_replicate_T50.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLS = "009"
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(m: str) -> None:
    print(m, flush=True, file=sys.stderr)


def scores(path: Path, tag: str) -> pl.DataFrame:
    return (pl.read_parquet(path, columns=["serial_number", "topic_kl_vs_past",
                                           "topic_kl_vs_future"])
            .filter(pl.col("topic_kl_vs_past").is_finite()
                    & pl.col("topic_kl_vs_future").is_finite())
            .with_columns(((pl.col("topic_kl_vs_past")
                            + pl.col("topic_kl_vs_future")) / 2).alias("A" + tag),
                          (pl.col("topic_kl_vs_past")
                           - pl.col("topic_kl_vs_future")).alias("L" + tag))
            .select("serial_number", "A" + tag, "L" + tag))


def gate_frame() -> pl.DataFrame:
    tm = pl.read_parquet(PROC / f"tm_class{CLS}.parquet",
                         columns=["serial_number", "registration_date"]).with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("rd")).drop_nulls("rd").with_columns(
        pl.col("rd").dt.year().alias("ry")).filter(pl.col("ry").is_between(REG_LO, REG_HI))
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("cd")).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    return tm.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                   .fill_null(False).cast(pl.Float64).alias("failed")
                   ).unique("serial_number").select("serial_number", "ry", "failed")


def contrast(df: pl.DataFrame, var: str) -> dict:
    s = df.sort(["ry", var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over("ry") - 1) * 5 // pl.len().over("ry"))
        .cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"),
                            pl.len().alias("n")).sort("q")
    p, n = [float(v) for v in g["p"]], [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"lift": p[4] - p[0], "se": se, "t": (p[4] - p[0]) / se}


def theme_match(path_a: Path, path_b: Path) -> dict:
    a = joblib.load(path_a)["lda"].components_
    b = joblib.load(path_b)["lda"].components_
    a = a / a.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = an @ bn.T
    r, c = linear_sum_assignment(-cos)
    m = cos[r, c]
    return {"T": int(a.shape[0]), "mean_cos": float(m.mean()),
            "median_cos": float(np.median(m)),
            "share_above_0_9": float((m > 0.9).mean()),
            "share_above_0_7": float((m > 0.7).mean()),
            "share_below_0_5": float((m < 0.5).mean())}


def main() -> int:
    rep = load_module("rep", "topic_seed_replicate.py")
    sweep = rep.sweep
    rep.fit(50, 7)

    prod_model = PROC / "topic_model.joblib"
    seed_model = PROC / "topic_model_T50_seed7.joblib"
    p_sweep42 = PROC / f"rolling_surprise_class{CLS}_T50_sweepscorer.parquet"
    p_seed7 = PROC / f"rolling_surprise_class{CLS}_T50_seed7.parquet"
    base_model, base_score = sweep.model_path, sweep.score_path
    for model, path in ((prod_model, p_sweep42), (seed_model, p_seed7)):
        sweep.model_path = lambda t, m=model: m
        sweep.score_path = lambda c, t, p=path: p
        sweep.score_class(CLS, 50)
    sweep.model_path, sweep.score_path = base_model, base_score

    prod = scores(PROC / f"rolling_surprise_class{CLS}.parquet", "p")
    a = scores(p_sweep42, "a")
    b = scores(p_seed7, "b")
    chk = prod.join(a, on="serial_number")
    ab = a.join(b, on="serial_number")
    g = ab.join(gate_frame(), on="serial_number", how="inner")

    out = {
        "class": CLS, "T": 50, "seed_a": 42, "seed_b": 7,
        "scorer_check": {"n": chk.height,
                         "r_lead": chk.select(pl.corr("Lp", "La")).item(),
                         "r_atyp": chk.select(pl.corr("Ap", "Aa")).item()},
        "seed": {"n": ab.height,
                 "r_lead": ab.select(pl.corr("La", "Lb")).item(),
                 "r_atyp": ab.select(pl.corr("Aa", "Ab")).item(),
                 "n_gate": g.height,
                 "seed42": {"lead": contrast(g, "La"), "atyp": contrast(g, "Aa")},
                 "seed7": {"lead": contrast(g, "Lb"), "atyp": contrast(g, "Ab")}},
        "themes": {"T50": theme_match(prod_model, seed_model),
                   "T200": theme_match(PROC / "topic_model_T200.joblib",
                                       PROC / "topic_model_T200_seed7.joblib")},
    }
    log(json.dumps(out, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "topic_seed_replicate_T50.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
