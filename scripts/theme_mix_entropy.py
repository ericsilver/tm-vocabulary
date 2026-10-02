"""Breadth of each registration's theme mix: entropy H and its top-two themes.

For every registration in the gate sample (battery_frame.parquet: unique serials,
registered 2002-2018, scored), the goods/services description is re-embedded
with the production T = 50 model and the full theme mix theta kept long enough
to compute:
    H        = -sum_k theta_k log theta_k        (nats; 0 = one theme, log 50 = uniform)
    eff_n    = exp(H)                            (effective number of themes)
    n_pres   = number of themes with theta >= 0.10
    top1, w1, top2, w2                           (heaviest two themes and their shares)

Output: data/processed/theme_mix_entropy.parquet
        data/processed/theme_mix_theta.parquet (serial, class, the full 50-theme mix)
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = PROC / "theme_mix_entropy_parts"
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
CHUNK = 40_000


def one_class(cls: str, serials: list[str]) -> str:
    warnings.filterwarnings("ignore")
    import joblib
    from sklearn.feature_extraction.text import CountVectorizer
    out = OUT / f"class{cls}.parquet"
    if out.exists():
        return f"{cls}: exists"
    m = joblib.load(PROC / "topic_model.joblib")
    lda = m["lda"]
    lda.n_jobs = 1
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True, token_pattern=TOKEN, ngram_range=(1, 2))
    d = pl.read_parquet(PROC / f"tm_class{cls}.parquet", columns=["serial_number", "goods_services"]).unique(
        "serial_number").join(pl.DataFrame({"serial_number": serials}), on="serial_number", how="inner")
    texts = d["goods_services"].fill_null("").to_list()
    cols = {k: [] for k in ("H", "n_pres", "top1", "w1", "top2", "w2")}
    thetas = []
    for s in range(0, len(texts), CHUNK):
        th = lda.transform(vec.transform(texts[s:s + CHUNK]))
        th = th / th.sum(1, keepdims=True)
        thetas.append(th.astype(np.float32))
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(np.where(th > 0, th * np.log(th), 0.0)).sum(1)
        o = np.argsort(-th, 1)[:, :2]
        r = np.arange(len(th))
        cols["H"].append(H.astype(np.float32))
        cols["n_pres"].append((th >= 0.10).sum(1).astype(np.int8))
        cols["top1"].append(o[:, 0].astype(np.int16)); cols["w1"].append(th[r, o[:, 0]].astype(np.float32))
        cols["top2"].append(o[:, 1].astype(np.int16)); cols["w2"].append(th[r, o[:, 1]].astype(np.float32))
    df = pl.DataFrame({"serial_number": d["serial_number"],
                       **{k: (np.concatenate(v) if v else np.array([])) for k, v in cols.items()}}).with_columns(
        pl.lit(cls).alias("cls"))
    # full mix, for clustering filings into multi-theme segments
    TH = np.concatenate(thetas) if thetas else np.zeros((0, 50), np.float32)
    pl.DataFrame({"serial_number": d["serial_number"], **{f"t{k}": TH[:, k] for k in range(TH.shape[1])}}
                 ).with_columns(pl.lit(cls).alias("cls")).write_parquet(OUT / f"theta_class{cls}.parquet")
    df.write_parquet(out)
    return f"{cls}: {df.height:,}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pl.read_parquet(PROC / "battery_frame.parquet", columns=["serial_number", "cls"])
    groups = {c: g["serial_number"].to_list() for (c,), g in fr.group_by(["cls"])}
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    order = sorted(groups, key=lambda c: -len(groups[c]))
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(one_class, c, groups[c]) for c in order]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(order)}] {f.result()}", flush=True)
    pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("class*.parquet"))]).write_parquet(
        PROC / "theme_mix_entropy.parquet")
    pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("theta_class*.parquet"))]).write_parquet(
        PROC / "theme_mix_theta.parquet")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
