"""Per-theme use, lead and outcomes at three theme counts, for the themes page.

For T = 50 (production), 200 and 500 (robustness models) this assigns a sample
of filings to the theme that takes the largest share of each description and
aggregates, per theme:

  use       share of each filing year's applications (1990-2024) whose largest
            theme it is
  lead      mean lead and atypicality of those filings, standardized within
            Nice class and filing year on the full scored population, overall
            and by filing year (1995-2019)
  outcomes  each measured against the rate expected from the filings' own
            class and year, so a theme is not credited with its class's level:
              reached registration        (applications filed 1995-2020)
              passed the five-year proof  (registrations 1996-2018)
              owner in SEC reporting      (registrations 1996-2018)
              owner with an IPO marker    (registrations 1996-2018)

Outcome definitions match scripts/build_ipo_viewer_data.py. Owner outcomes are
properties of the owner, so a firm with many marks counts once per mark.

The sample is up to CAP applications per class (filing years 1990-2024), the
same applications at every T. Corpus-wide figures weight each class by the
ratio of its applications to its sample.

Usage:  python scripts/build_theme_resolution_data.py [CAP] [T ...]
Output: docs/online-appendix/themes_data/T{T}.json
        docs/online-appendix/themes_data/T{T}_c{NNN}.json   (one per class)
"""
from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
DOCS = REPO / "docs" / "online-appendix"
OUT = DOCS / "themes_data"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
TS = [int(a) for a in sys.argv[2:]] or [50, 200, 500]
ONLY = os.environ.get("THEME_ONLY", "").split(",") if os.environ.get("THEME_ONLY") else None  # test runs
SEED = 20261118
FY_LO, FY_HI = 1990, 2024        # use series
LY_LO, LY_HI = 1995, 2019        # lead by year
REGY_LO, REGY_HI = 1995, 2020    # applications judged on reaching registration
RY_LO, RY_HI = 1996, 2018        # registrations judged at the five-year proof
GATE_LO, GATE_HI = 4.0, 8.5
NY = FY_HI - FY_LO + 1
NLY = LY_HI - LY_LO + 1
MIN_CLASS = 40                   # a theme needs this many sampled filings to appear in a class file
MIN_YEAR_N = 20                  # filings needed for a yearly mean lead
NWORDS = 10


def log(m): print(time.strftime("%H:%M:%S"), m, file=sys.stderr, flush=True)


def suffix(T: int) -> str:
    return "" if T == 50 else f"_T{T}"


def class_frame(c: str, ev: pl.DataFrame, sec: pl.DataFrame, ipo: pl.DataFrame) -> pl.DataFrame:
    """One row per application in class c, with outcomes and class-year expectations."""
    tp = PROC / f"tm_class{c}.parquet"
    d = pl.scan_parquet(tp).select(
        "serial_number", "filing_date", "registration_date", "owner_name",
        pl.col("goods_services").fill_null("").str.len_chars().alias("glen"),
    ).collect().filter(pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    d = d.sort("serial_number").unique("serial_number", keep="first", maintain_order=True)
    d = d.with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
    ).filter(pl.col("fy").is_between(FY_LO, FY_HI))
    d = d.join(ev, on="serial_number", how="left").join(sec, on="owner_name", how="left").join(
        ipo, on="owner_name", how="left")
    d = d.with_columns(
        pl.col("rd").dt.year().alias("ry"),
        pl.col("rd").is_not_null().cast(pl.Float64).alias("reg"),
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
        pl.col("is_sec").fill_null(False), pl.col("is_ipo").fill_null(False),
    ).with_columns(
        (~((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False))
        .cast(pl.Float64).alias("passed"),
        (pl.col("is_sec") | pl.col("is_ipo")).cast(pl.Float64).alias("sec"),
        pl.col("is_ipo").cast(pl.Float64).alias("ipo"),
        pl.col("fy").is_between(REGY_LO, REGY_HI).alias("in_regframe"),
        (pl.col("rd").is_not_null() & pl.col("rd").dt.year().is_between(RY_LO, RY_HI)).alias("in_gate"),
    )
    # expectations from every application of the class that has a description
    # (the population the sample is drawn from), by filing year (registration)
    # and registration year (the rest)
    d = d.filter(pl.col("glen") > 0).with_columns(
        pl.col("reg").mean().over("fy").alias("e_reg"),
        pl.col("passed").mean().over("ry").alias("e_pass"),
        pl.col("sec").mean().over("ry").alias("e_sec"),
        pl.col("ipo").mean().over("ry").alias("e_ipo"),
    )
    return d.drop("filing_date", "registration_date", "owner_name", "cd", "age", "rd",
                  "is_sec", "is_ipo")


def scores(c: str, T: int, d: pl.DataFrame) -> pl.DataFrame:
    """Lead and atypicality standardized within filing year on every scored filing of the class."""
    s = pl.read_parquet(PROC / f"rolling_surprise_class{c}{suffix(T)}.parquet",
                        columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future",
                                 "topic_dkl"]).filter(pl.col("topic_dkl").is_finite())
    s = s.unique("serial_number", keep="first").join(
        d.select("serial_number", "fy"), on="serial_number", how="inner").with_columns(
        pl.col("topic_dkl").alias("L"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    s = s.with_columns(
        ((pl.col("L") - pl.col("L").mean().over("fy")) / pl.col("L").std().over("fy")).alias("zL"),
        ((pl.col("A") - pl.col("A").mean().over("fy")) / pl.col("A").std().over("fy")).alias("zA"))
    return s.select("serial_number", "zL", "zA")


def bsum(idx: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(idx, weights=w, minlength=n)[:n]


def run(T: int, ev, sec, ipo, names: dict, labels: dict, t50_shares: dict | None) -> None:
    m = joblib.load(PROC / ("topic_model.joblib" if T == 50 else f"topic_model_T{T}.joblib"))
    lda = m["lda"]
    lda.n_jobs = 1
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    words = json.loads((PROC / f"topic_lda_meta{suffix(T)}.json").read_text())["top_words"]

    # corpus-wide accumulators (class-weighted)
    G = {k: np.zeros(T) for k in (
        "n", "L", "Ln", "A", "reg_d", "reg_n", "pass_d", "pass_n",
        "sec_o", "sec_e", "ipo_o", "ipo_e")}
    G_cnt = np.zeros((T, NY)); G_tot = np.zeros(NY)
    G_ly = np.zeros((T, NLY)); G_lyn = np.zeros((T, NLY)); G_lyraw = np.zeros((T, NLY))
    raw = {k: np.zeros(T) for k in ("n", "Ln", "reg_n", "pass_n", "sec_o", "ipo_o")}  # unweighted counts
    mix = {}                                                                          # class -> mean theta
    sizes = {}

    for c in (ONLY or CLASSES):
        if not (PROC / f"tm_class{c}.parquet").exists():
            continue
        d = class_frame(c, ev, sec, ipo)
        d = d.join(scores(c, T, d), on="serial_number", how="left")
        pool = d
        n_all = pool.height
        smp = pool.sample(min(CAP, n_all), seed=SEED).sort("serial_number") if n_all > CAP else pool
        w = n_all / smp.height
        sizes[c] = {"applications": n_all, "sampled": smp.height}
        txt = pl.scan_parquet(PROC / f"tm_class{c}.parquet").select(
            "serial_number", "goods_services").filter(
            pl.col("serial_number").is_in(smp["serial_number"].to_list())).collect().unique(
            "serial_number", keep="first")
        smp = smp.join(txt, on="serial_number", how="inner").sort("serial_number")
        texts = smp["goods_services"].to_list()
        dom = np.empty(len(texts), dtype=np.int64)
        has = np.zeros(len(texts), dtype=bool)        # description has at least one model term
        msum = np.zeros(T)
        for s0 in range(0, len(texts), 25_000):
            X = vec.transform(texts[s0:s0 + 25_000])
            th = lda.transform(X)
            th /= th.sum(axis=1, keepdims=True)
            dom[s0:s0 + len(th)] = th.argmax(axis=1)
            has[s0:s0 + len(th)] = np.asarray(X.sum(axis=1)).ravel() > 0
            msum += th.sum(axis=0)
        mix[c] = msum / len(texts)
        del texts, txt
        # a description with none of the model's terms has no largest theme
        smp = smp.filter(pl.Series(has))
        dom = dom[has]

        fy = smp["fy"].to_numpy().astype(np.int64) - FY_LO
        zL = smp["zL"].to_numpy(); zA = smp["zA"].to_numpy()
        okL = np.isfinite(zL) & np.isfinite(zA) & (fy + FY_LO >= LY_LO) & (fy + FY_LO <= LY_HI)
        rf = smp["in_regframe"].to_numpy(); gf = smp["in_gate"].to_numpy()
        reg = smp["reg"].to_numpy(); e_reg = smp["e_reg"].to_numpy()
        pas = smp["passed"].to_numpy(); e_pas = smp["e_pass"].to_numpy()
        se = smp["sec"].to_numpy(); e_se = smp["e_sec"].to_numpy()
        ip = smp["ipo"].to_numpy(); e_ip = smp["e_ipo"].to_numpy()

        one = np.ones(len(dom))
        C = {
            "n": bsum(dom, one, T),
            "L": bsum(dom[okL], zL[okL], T), "Ln": bsum(dom[okL], one[okL], T),
            "A": bsum(dom[okL], zA[okL], T),
            "reg_d": bsum(dom[rf], (reg - e_reg)[rf], T), "reg_n": bsum(dom[rf], one[rf], T),
            "pass_d": bsum(dom[gf], (pas - e_pas)[gf], T), "pass_n": bsum(dom[gf], one[gf], T),
            "sec_o": bsum(dom[gf], se[gf], T), "sec_e": bsum(dom[gf], e_se[gf], T),
            "ipo_o": bsum(dom[gf], ip[gf], T), "ipo_e": bsum(dom[gf], e_ip[gf], T),
        }
        cnt = bsum(dom * NY + fy, one, T * NY).reshape(T, NY)
        tot = np.bincount(fy, minlength=NY)[:NY].astype(float)
        ly = fy[okL] - (LY_LO - FY_LO)
        lys = bsum(dom[okL] * NLY + ly, zL[okL], T * NLY).reshape(T, NLY)
        lyn = bsum(dom[okL] * NLY + ly, one[okL], T * NLY).reshape(T, NLY)

        for k in G:
            G[k] += w * C[k]
        for k in raw:
            raw[k] += C[k]
        G_cnt += w * cnt; G_tot += w * tot
        G_ly += w * lys; G_lyn += w * lyn; G_lyraw += lyn

        rows = []
        for k in np.argsort(-C["n"]):
            if C["n"][k] < MIN_CLASS:
                break
            rows.append(theme_row(int(k), {a: b[k] for a, b in C.items()},
                                  {a: b[k] for a, b in C.items()},
                                  cnt[k], tot, lys[k], lyn[k], lyn[k]))
        (OUT / f"T{T}_c{c}.json").write_text(json.dumps(
            {"T": T, "cls": c, "name": names.get(c, ""), "applications": n_all,
             "sampled": sizes[c]["sampled"], "rows": rows}, separators=(",", ":")))
        log(f"  [T{T} {c}] {sizes[c]['sampled']:,} of {n_all:,} sampled, {len(rows)} themes listed")
        del d, pool, smp, dom
        gc.collect()

    classes = sorted(mix)
    S = np.vstack([mix[c] for c in classes])                       # classes x themes
    wts = np.array([sizes[c]["applications"] for c in classes], float)
    corpus = (S * wts[:, None]).sum(axis=0) / wts.sum()
    rows = []
    for k in range(T):
        r = theme_row(k, {a: b[k] for a, b in G.items()}, {a: b[k] for a, b in raw.items()},
                      G_cnt[k], G_tot, G_ly[k], G_lyn[k], G_lyraw[k])
        if t50_shares:                                             # published T=50 shares, unchanged
            r["corpus"] = t50_shares[k]["corpus"]
            by = t50_shares[k]["by_class"]
        else:
            r["corpus"] = round(float(corpus[k]), 5)
            by = {classes[i]: round(float(S[i, k]), 5) for i in range(len(classes))}
        r["by_class"] = [by.get(c, 0.0) for c in CLASSES]
        top = sorted(by.items(), key=lambda kv: -kv[1])[:3]
        r["top"] = [[c, v] for c, v in top]
        r["words"] = words[str(k)][:NWORDS]
        if labels:
            r["label"] = labels[k]
        rows.append(r)
    (OUT / f"T{T}.json").write_text(json.dumps(
        {"T": T, "cap": CAP, "fy": [FY_LO, FY_HI], "ly": [LY_LO, LY_HI],
         "reg_years": [REGY_LO, REGY_HI], "gate_years": [RY_LO, RY_HI],
         "classes": [[c, names.get(c, "")] for c in CLASSES],
         "sizes": sizes, "rows": rows}, separators=(",", ":")))
    log(f"[done] T={T}: {sum(v['sampled'] for v in sizes.values()):,} filings assigned")
    del m, lda
    gc.collect()


def theme_row(k, a, n_raw, cnt, tot, lys, lyn, lyn_raw) -> dict:
    """a: (weighted) sums for one theme; n_raw: unweighted counts behind them."""
    def ratio(x, y, nd=3):
        return round(float(x / y), nd) if y > 0 else None
    share = np.divide(cnt, tot, out=np.zeros_like(cnt, dtype=float), where=tot > 0)
    lead_y = [int(round(100 * s / n)) if nr >= MIN_YEAR_N else None
              for s, n, nr in zip(lys, lyn, lyn_raw)]
    return {
        "k": k,
        "n": int(n_raw["n"]),
        "use": [int(round(1e5 * v)) for v in share],          # per 100,000 applications
        "lead_y": lead_y,                                      # mean z x 100
        "lead": ratio(a["L"], a["Ln"]), "atyp": ratio(a["A"], a["Ln"]), "lead_n": int(n_raw["Ln"]),
        "reg": ratio(100 * a["reg_d"], a["reg_n"], 2), "reg_n": int(n_raw["reg_n"]),
        "proof": ratio(100 * a["pass_d"], a["pass_n"], 2), "proof_n": int(n_raw["pass_n"]),
        "sec": ratio(a["sec_o"], a["sec_e"], 2), "sec_n": int(n_raw["sec_o"]),
        "sec_e": round(float(a["sec_e"] * (n_raw["pass_n"] / a["pass_n"])) if a["pass_n"] > 0 else 0.0, 1),
        "ipo": ratio(a["ipo_o"], a["ipo_e"], 2), "ipo_n": int(n_raw["ipo_o"]),
        "ipo_e": round(float(a["ipo_e"] * (n_raw["pass_n"] / a["pass_n"])) if a["pass_n"] > 0 else 0.0, 1),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import importlib.util
    def load(name):
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    names = load("online_appendix").NICE_NAMES
    labels = load("themes_t50_page").LABELS

    t50 = {}
    with open(DOCS / "themes_T50.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t50[int(r["theme"])] = {"corpus": float(r["corpus_share"]),
                                    "by_class": {c: float(r[f"class_{c}"]) for c in CLASSES
                                                 if f"class_{c}" in r}}

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet", columns=["owner_name"]).unique().with_columns(
        pl.lit(True).alias("is_sec"))
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet", columns=["owner_name", "in_8a", "ipo_date"])
    ipo = fm.filter((pl.col("in_8a").fill_null(0) == 1) | pl.col("ipo_date").is_not_null()).select(
        "owner_name").unique().with_columns(pl.lit(True).alias("is_ipo"))
    log(f"[setup] {ev.height:,} cancellation dates, {sec.height:,} SEC owners, {ipo.height:,} IPO owners; "
        f"cap {CAP:,}; T {TS}")
    for T in TS:
        run(T, ev, sec, ipo, names, labels if T == 50 else {}, t50 if T == 50 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
