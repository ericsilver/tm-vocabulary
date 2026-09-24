"""Is word-scored "specificity" professional drafting, and does counsel predict
reaching SEC reporting better than the word-scored signal does?

Paper A's second hazard: word-scored lead appears to predict firm success
(reaching SEC reporting), but the signal is unsigned lexical specificity --
long descriptions using rare, specific words -- rather than vocabulary
position. This asks where that specificity comes from, on the paper's own
SEC-reporting sample (registered debut filings 1995-2018, one row per owner):

  1. How strongly counsel of record tracks each specificity measure:
     log distinct terms, share of tokens outside the theme model's vocabulary
     ("invisible share"), word-scored atypicality and lead, and, for contrast,
     theme-scored atypicality and lead. Raw and within class x filing year.
  2. Linear probability models of P(owner ever in SEC reporting), class x
     filing-year fixed effects, HC1 errors (one row per owner):
        a. word-scored lead alone
        b. counsel alone
        c. word-scored lead + counsel
        d. specificity (log distinct terms, invisible share) + word lead
        e. d + counsel
     with within-R^2 for each, so the question "which predicts better" has a
     direct answer, and the change in the word-lead coefficient when counsel
     enters shows how much of it counsel accounts for.

Reads   tm_class*.parquet, termroll_surprise_class*.parquet (per-filing word
        build), rolling_surprise_class*.parquet (production theme build),
        case_extras.parquet, uspto_sec_crosswalk.parquet, topic_model.joblib
Output: paper/results/specificity_counsel.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
FILE_LO, FILE_HI = 1995, 2018
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def build() -> pl.DataFrame:
    parts = []
    for c in classes():
        parts.append(pl.read_parquet(PROC / f"tm_class{c}.parquet",
                                     columns=["owner_name", "filing_date"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date")))
    debut = pl.concat(parts).group_by("owner_name").agg(pl.col("debut_date").min())
    del parts
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))
    att = pl.read_parquet(PROC / "case_extras.parquet", columns=["serial_number", "attorney_name"]
                          ).with_columns((pl.col("attorney_name").fill_null("").str.len_chars() > 0)
                                         .alias("counsel")).select("serial_number", "counsel")
    rows = []
    for c in classes():
        th, te = PROC / f"rolling_surprise_class{c}.parquet", PROC / f"termroll_surprise_class{c}.parquet"
        if not (th.exists() and te.exists()):
            continue
        tm = pl.read_parquet(PROC / f"tm_class{c}.parquet",
                             columns=["serial_number", "owner_name", "filing_date",
                                      "registration_date", "goods_services"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").str.len_chars() == 8)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
            & pl.col("goods_services").is_not_null()).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy")
        ).filter(pl.col("fy").is_between(FILE_LO, FILE_HI))
        a = pl.read_parquet(th, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"]
                            ).rename({"topic_kl_vs_past": "kp", "topic_kl_vs_future": "kf"})
        b = pl.read_parquet(te, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future",
                                         "n_terms"]).rename(
            {"topic_kl_vs_past": "tkp", "topic_kl_vs_future": "tkf"})
        j = tm.join(debut, on="owner_name").filter(pl.col("filing_date") == pl.col("debut_date")
        ).join(a, on="serial_number").join(b, on="serial_number").filter(
            pl.all_horizontal([pl.col(x).is_finite() for x in ("kp", "kf", "tkp", "tkf")])
            & (pl.col("n_terms") > 0)
        ).join(sec, on="owner_name", how="left").join(att, on="serial_number", how="left"
        ).with_columns(pl.col("in_sec").fill_null(False), pl.col("counsel").fill_null(False),
                       pl.lit(c).alias("cls"))
        rows.append(j.select("serial_number", "owner_name", "cls", "fy", "goods_services",
                             "kp", "kf", "tkp", "tkf", "n_terms", "in_sec", "counsel"))
        del tm, a, b, j
        gc.collect()
    df = pl.concat(rows).unique("serial_number", keep="first").sort("serial_number").unique(
        "owner_name", keep="first")
    log(f"[frame] {df.height:,} registered debut owners")
    return df


def invisible_share(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer
    vocab = joblib.load(PROC / "topic_model.joblib")["vocabulary"]
    uni = {k for k in vocab if " " not in k}
    all_vec = CountVectorizer(lowercase=True, token_pattern=TOKEN)
    out = np.zeros(len(texts))
    for s in range(0, len(texts), 200_000):
        X = all_vec.fit_transform(texts[s:s + 200_000])
        names = all_vec.get_feature_names_out()
        inv = np.array([n not in uni for n in names])
        tot = np.asarray(X.sum(axis=1)).ravel()
        miss = np.asarray(X[:, np.where(inv)[0]].sum(axis=1)).ravel()
        out[s:s + 200_000] = np.where(tot > 0, miss / np.maximum(tot, 1), np.nan)
    return out


def demean(pdf: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pdf[cols] - pdf.groupby("cell")[cols].transform("mean")


def lpm(pdf: pd.DataFrame, xcols: list[str], label: str) -> dict:
    Xd = demean(pdf, xcols).to_numpy()
    yd = demean(pdf, ["y"]).to_numpy().ravel()
    XtX = Xd.T @ Xd
    b = np.linalg.solve(XtX, Xd.T @ yd)
    e = yd - Xd @ b
    n, k = Xd.shape
    inv = np.linalg.inv(XtX)
    V = (n / (n - k)) * inv @ ((Xd * (e ** 2)[:, None]).T @ Xd) @ inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / (yd @ yd)
    res = {"label": label, "n": int(n), "within_r2": float(r2),
           "coef": {c: {"b_pp": float(100 * bi), "se_pp": float(100 * si), "t": float(bi / si)}
                    for c, bi, si in zip(xcols, b, se)}}
    log(f"  {label}: R2w={r2:.6f} " + " ".join(
        f"{c}={100*bi:+.3f}pp(t={bi/si:.1f})" for c, bi, si in zip(xcols, b, se)))
    return res


def main() -> int:
    df = build()
    inv = invisible_share(df["goods_services"].to_list())
    pdf = df.drop("goods_services").to_pandas()
    pdf["invisible"] = inv
    pdf = pdf[np.isfinite(pdf["invisible"])].copy()
    pdf["cell"] = pdf["cls"] + "_" + pdf["fy"].astype(str)
    pdf["y"] = pdf["in_sec"].astype(float)
    pdf["counsel"] = pdf["counsel"].astype(float)
    pdf["log_terms"] = np.log(pdf["n_terms"].astype(float))
    pdf["word_A"] = (pdf["tkp"] + pdf["tkf"]) / 2
    pdf["word_L"] = pdf["tkp"] - pdf["tkf"]
    pdf["word_absL"] = pdf["word_L"].abs()
    pdf["theme_A"] = (pdf["kp"] + pdf["kf"]) / 2
    pdf["theme_L"] = pdf["kp"] - pdf["kf"]
    z = lambda s: (s - s.mean()) / s.std()
    for c in ("log_terms", "invisible", "word_A", "word_L", "word_absL", "theme_A", "theme_L"):
        pdf[c + "_z"] = z(pdf[c])

    spec = ["log_terms", "invisible", "word_A", "word_L", "word_absL", "theme_A", "theme_L"]
    corr_raw = {c: float(np.corrcoef(pdf["counsel"], pdf[c])[0, 1]) for c in spec}
    dm = demean(pdf, ["counsel"] + spec)
    corr_within = {c: float(np.corrcoef(dm["counsel"], dm[c])[0, 1]) for c in spec}
    means = {c: {"counsel": float(pdf.loc[pdf.counsel == 1, c].mean()),
                 "self": float(pdf.loc[pdf.counsel == 0, c].mean())} for c in spec}
    rates = {"counsel": float(pdf.loc[pdf.counsel == 1, "y"].mean()),
             "self": float(pdf.loc[pdf.counsel == 0, "y"].mean()),
             "share_counsel": float(pdf["counsel"].mean())}
    log(f"[corr raw] {corr_raw}")
    log(f"[corr within] {corr_within}")
    log(f"[rates] {rates}")

    models = [
        lpm(pdf, ["word_L_z"], "a_wordlead"),
        lpm(pdf, ["counsel"], "b_counsel"),
        lpm(pdf, ["word_L_z", "counsel"], "c_wordlead_counsel"),
        lpm(pdf, ["log_terms_z", "invisible_z", "word_L_z"], "d_specificity_wordlead"),
        lpm(pdf, ["log_terms_z", "invisible_z", "word_L_z", "counsel"], "e_d_plus_counsel"),
        lpm(pdf, ["theme_L_z", "theme_A_z", "log_terms_z", "counsel"], "f_theme_plus_counsel"),
    ]
    out = {"n": int(len(pdf)), "base_rate": float(pdf["y"].mean()), "rates_by_counsel": rates,
           "corr_counsel_raw": corr_raw, "corr_counsel_within_class_year": corr_within,
           "means_by_counsel": means, "models": models}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "specificity_counsel.json").write_text(json.dumps(out, indent=1))
    log("[done] specificity_counsel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
