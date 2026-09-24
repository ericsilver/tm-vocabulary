"""How much does the equal-per-class fitting sample thin big-class vocabulary?

The production theme model is fitted on at most 10,000 descriptions per Nice
class. Classes differ in size by two orders of magnitude, so vocabulary that
lives mostly in the largest classes is under-represented in what the model
learns from. For a few probe vocabularies this reports the share of
descriptions containing any probe term, in the fitting sample and in the full
corpus of descriptions filed 1990-2024 (the years the sample is drawn from).

The sample share is read from the cached fitting matrix, so it counts the
analyzer's tokens (words and adjacent pairs present in the fitted vocabulary);
the corpus share is a case-insensitive whole-word search of the raw text. The
two can differ slightly in what they count as a match; the ratio is the point.

Output: paper/results/sample_thinning.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

PROBES = {
    "blockchain": ["blockchain", "blockchains", "cryptocurrency",
                   "cryptocurrencies", "bitcoin", "distributed ledger",
                   "digital currency", "non-fungible token"],
    "ai": ["artificial intelligence", "machine learning", "deep learning",
           "neural networks", "computer vision", "predictive analytics"],
    "cloud": ["cloud computing", "saas", "software as a service",
              "virtualization", "data center"],
    "internet": ["internet", "online", "website", "web site", "e-commerce"],
    "solar": ["solar", "solar cells", "solar energy", "photovoltaic"],
}


def main() -> int:
    X = sp.load_npz(PROC / "_lda_fit_matrix.npz").tocsc()
    vocab = joblib.load(PROC / "_lda_fit_vocab.joblib")
    n_sample = X.shape[0]
    out = {"n_sample": int(n_sample), "probes": {}}
    for name, terms in PROBES.items():
        idx = [vocab[t] for t in terms if t in vocab]
        has = (X[:, idx].sum(axis=1) > 0) if idx else np.zeros((n_sample, 1), bool)
        out["probes"][name] = {"sample_share": float(np.asarray(has).mean()),
                               "terms_in_vocab": [t for t in terms if t in vocab]}

    pats = {k: r"(?i)\b(?:" + "|".join(re.escape(t) for t in v) + r")\b"
            for k, v in PROBES.items()}
    tot = 0
    hits = {k: 0 for k in PROBES}
    for p in sorted(PROC.glob("tm_class*.parquet")):
        if not p.stem.replace("tm_class", "").isdigit():
            continue
        d = pl.read_parquet(p, columns=["filing_date", "goods_services"]).filter(
            pl.col("goods_services").is_not_null()
            & (pl.col("goods_services").str.len_chars() > 0)
            & pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False)
            .is_between(1990, 2024))
        tot += d.height
        for k, pat in pats.items():
            hits[k] += int(d.select(pl.col("goods_services").str.contains(pat).sum()).item())
    out["n_corpus"] = tot
    for k in PROBES:
        c = hits[k] / tot
        s = out["probes"][k]["sample_share"]
        out["probes"][k].update({"corpus_share": c,
                                 "ratio_corpus_to_sample": c / s if s else None})
        print(f"{k:11s} sample {100*s:5.2f}%  corpus {100*c:5.2f}%  "
              f"ratio {c/s if s else float('nan'):4.1f}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "sample_thinning.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
