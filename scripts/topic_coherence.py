"""Theme quality by resolution: NPMI coherence of each theme's leading words.

A standard diagnostic for a topic model is whether each theme's leading words
actually occur together in documents. For a theme's ten highest-probability
terms, normalized pointwise mutual information (NPMI) averages, over every pair,
log[p(w1,w2) / (p(w1)p(w2))] / -log p(w1,w2), with p counted as the share of
documents containing the term(s). It runs from -1 (never together) through 0
(independent) to 1 (always together). Computed on the model's own fitting sample
(448,437 descriptions), for each resolution fitted with the production recipe.

Also reported: how much of the sample's token mass each resolution's themes
spread over (effective number of themes per document, exp of the entropy of the
document's theme mix, on a 20,000-document subsample).

Output: paper/results/topic_coherence.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
TOPN = 10
MODELS = {50: "topic_model.joblib", 100: "topic_model_T100.joblib",
          200: "topic_model_T200.joblib", 500: "topic_model_T500.joblib"}


def log(m: str) -> None:
    print(m, flush=True, file=sys.stderr)


def main() -> int:
    X = sp.load_npz(PROC / "_lda_fit_matrix.npz").tocsc()
    fit_vocab = joblib.load(PROC / "_lda_fit_vocab.joblib")
    n_docs = X.shape[0]
    Xb = (X > 0).astype(np.float32).tocsc()
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(n_docs, size=20_000, replace=False))
    out = {"n_docs": int(n_docs), "topn": TOPN, "by_T": {}}
    for T, fname in MODELS.items():
        path = PROC / fname
        if not path.exists():
            log(f"[T={T}] missing {fname}")
            continue
        m = joblib.load(path)
        lda, vocab = m["lda"], m["vocabulary"]
        if vocab != fit_vocab:
            log(f"[T={T}] vocabulary differs from the cached fit matrix; skipped")
            out["by_T"][str(T)] = {"skipped": "vocabulary mismatch"}
            continue
        comps = lda.components_
        top = np.argsort(comps, axis=1)[:, ::-1][:, :TOPN]
        cols = np.unique(top)
        sub_b = Xb[:, cols]
        df = np.asarray(sub_b.sum(axis=0)).ravel() / n_docs
        co = (sub_b.T @ sub_b).toarray() / n_docs
        pos = {c: i for i, c in enumerate(cols)}
        npmi_topic = []
        for k in range(comps.shape[0]):
            idx = [pos[c] for c in top[k]]
            vals = []
            for i in range(TOPN):
                for j in range(i + 1, TOPN):
                    a, b = idx[i], idx[j]
                    pj = co[a, b]
                    if pj <= 0:
                        vals.append(-1.0)
                        continue
                    pmi = np.log(pj / (df[a] * df[b]))
                    vals.append(pmi / -np.log(pj))
            npmi_topic.append(float(np.mean(vals)))
        npmi_topic = np.array(npmi_topic)
        theta = lda.transform(X[sub].tocsr())
        theta = np.clip(theta, 1e-12, None)
        theta /= theta.sum(axis=1, keepdims=True)
        eff = np.exp(-(theta * np.log(theta)).sum(axis=1))
        out["by_T"][str(T)] = {
            "npmi_mean": float(npmi_topic.mean()),
            "npmi_median": float(np.median(npmi_topic)),
            "npmi_p10": float(np.percentile(npmi_topic, 10)),
            "share_themes_npmi_below_0": float((npmi_topic < 0).mean()),
            "eff_themes_per_doc_median": float(np.median(eff)),
        }
        log(f"[T={T}] {out['by_T'][str(T)]}")
        del m, lda, comps, theta
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "topic_coherence.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
