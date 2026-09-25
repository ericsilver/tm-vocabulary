"""Primary theme (production T = 50 model) for every scored filing.

lead_crowding.py assigned themes to a sample; the hypothesis battery needs
every scored filing's primary theme, and each class-year's theme mix, so the
growth, turnover and geography of a theme can be measured on the full record.

For each class: the scored filings (rows of rolling_surprise_class{cls}) are
re-embedded with the production LDA model; each gets its primary theme (the
largest share of its mix) and that share. The class-year mean mix is kept
separately.

Output: data/processed/theme_full/theme_class{cls}.parquet
          serial_number, fy, top_theme, top_w
        data/processed/theme_full/classyear_mix.parquet
          nice, fy, n, theta_0 .. theta_49 (mean mix of the year's filings)
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
OUT = PROC / "theme_full"
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
CHUNK = 50_000


def one_class(cls: str) -> str:
    warnings.filterwarnings("ignore")
    import joblib
    from sklearn.feature_extraction.text import CountVectorizer
    out = OUT / f"theme_class{cls}.parquet"
    mix_out = OUT / f"mix_class{cls}.parquet"
    if out.exists() and mix_out.exists():
        return f"{cls}: exists"
    m = joblib.load(PROC / "topic_model.joblib")
    lda = m["lda"]
    lda.n_jobs = 1
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    sc = pl.read_parquet(PROC / f"rolling_surprise_class{cls}.parquet",
                         columns=["serial_number"]).unique()
    d = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                        columns=["serial_number", "filing_date", "goods_services"]).unique(
        "serial_number").join(sc, on="serial_number", how="inner").with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).fill_null(0).alias("fy"),
        pl.col("goods_services").fill_null(""))
    tops, ws = [], []
    K = lda.n_components
    fy = d["fy"].to_numpy()
    sums = {}
    texts = d["goods_services"].to_list()
    for s in range(0, len(texts), CHUNK):
        th = lda.transform(vec.transform(texts[s:s + CHUNK]))
        tops.append(th.argmax(1).astype(np.int16))
        ws.append(th.max(1).astype(np.float32))
        for y in np.unique(fy[s:s + CHUNK]):
            sel = fy[s:s + CHUNK] == y
            a = sums.setdefault(int(y), [0, np.zeros(K)])
            a[0] += int(sel.sum())
            a[1] += th[sel].sum(0)
    pl.DataFrame({"serial_number": d["serial_number"], "fy": d["fy"],
                  "top_theme": np.concatenate(tops) if tops else [],
                  "top_w": np.concatenate(ws) if ws else []}).write_parquet(out)
    rows = {"nice": [], "fy": [], "n": []}
    for k in range(K):
        rows[f"theta_{k}"] = []
    for y, (n, v) in sorted(sums.items()):
        rows["nice"].append(cls); rows["fy"].append(y); rows["n"].append(n)
        for k in range(K):
            rows[f"theta_{k}"].append(float(v[k] / n))
    pl.DataFrame(rows).write_parquet(mix_out)
    return f"{cls}: {d.height:,}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    classes = [f"{i:03d}" for i in range(1, 46)]
    # big classes first so the pool stays busy
    size = {c: (PROC / f"tm_class{c}.parquet").stat().st_size for c in classes}
    classes.sort(key=lambda c: -size[c])
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(one_class, c) for c in classes]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/45] {f.result()}", flush=True)
    pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("mix_class*.parquet"))]).write_parquet(
        OUT / "classyear_mix.parquet")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
