"""How much does the scoring tokenizer differ from the fitting tokenizer?

The production theme model was fitted on descriptions tokenized by
novelty.dictionary._make_analyzer (tokens of two or more characters, sklearn
English + USPTO stopwords removed before forming word pairs, pairs only within
punctuation-delimited items). Filings are scored with a plain CountVectorizer
over the same vocabulary (tokens of three or more letters, no stopword removal,
pairs across punctuation). Every filing is scored by the same rule, so this is
a fixed feature of the measure rather than a source of bias between filings;
this script measures how far each filing's theme mix moves between the two.

On a random sample of descriptions from every class, for each description:
  - vocabulary-token overlap between the two tokenizations
  - total-variation distance between the theme mixes the two tokenizations give
and, pooled, the correlation of K-/K+-style KL against a common reference.

Output: paper/results/tokenizer_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
PER_CLASS = 1_000


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer

    m = joblib.load(PROC / "topic_model.joblib")
    lda, vocab = m["lda"], m["vocabulary"]
    fit_vec = CountVectorizer(analyzer=_make_analyzer(frozenset(STOPWORDS), (1, 2)),
                              vocabulary=vocab)
    score_vec = CountVectorizer(vocabulary=vocab, lowercase=True,
                                token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b",
                                ngram_range=(1, 2))
    rng = np.random.default_rng(1)
    docs, cls_of = [], []
    for p in sorted(PROC.glob("tm_class*.parquet")):
        c = p.stem.replace("tm_class", "")
        if not c.isdigit():
            continue
        g = pl.read_parquet(p, columns=["goods_services"]).filter(
            pl.col("goods_services").str.len_chars() > 0)["goods_services"]
        idx = rng.choice(len(g), size=min(PER_CLASS, len(g)), replace=False)
        docs.extend(g[np.sort(idx)].to_list())
        cls_of.extend([c] * len(idx))
    Xf = fit_vec.transform(docs)
    Xs = score_vec.transform(docs)
    both = np.asarray(Xf.minimum(Xs).sum(axis=1)).ravel()
    nf = np.asarray(Xf.sum(axis=1)).ravel()
    ns = np.asarray(Xs.sum(axis=1)).ravel()
    ok = (nf > 0) & (ns > 0)
    tf = lda.transform(Xf[ok])
    ts = lda.transform(Xs[ok])
    tv = 0.5 * np.abs(tf - ts).sum(axis=1)
    q = np.clip(ts.mean(axis=0), 1e-12, None)
    kf = (np.clip(tf, 1e-12, None) * (np.log(np.clip(tf, 1e-12, None)) - np.log(q))).sum(axis=1)
    ks = (np.clip(ts, 1e-12, None) * (np.log(np.clip(ts, 1e-12, None)) - np.log(q))).sum(axis=1)
    out = {
        "n_docs": int(ok.sum()),
        "fit_tokens_also_in_scoring_median": float(np.median((both / nf)[ok])),
        "scoring_tokens_also_in_fit_median": float(np.median((both / ns)[ok])),
        "theme_mix_tv_median": float(np.median(tv)),
        "theme_mix_tv_p90": float(np.percentile(tv, 90)),
        "dominant_theme_agree": float((tf.argmax(1) == ts.argmax(1)).mean()),
        "kl_vs_common_reference_r": float(np.corrcoef(kf, ks)[0, 1]),
    }
    print(json.dumps(out, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tokenizer_check.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
