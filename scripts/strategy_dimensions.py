"""Product and market characteristics from the strategy/IO literature, read from
trademark goods/services text, and their relation to survival and to the lead
effect.

Stage `flags`  : applies scripts/strategy_lexicons.py to every registration in
                 the battery frame (2002-2018), class file by class file, in
                 row-group batches; writes data/processed/strategy_flags.parquet
                 (one boolean column f_<key> per lexicon).
Stage `sample` : texts of 40,000 random frame registrations (data/processed/strategy_review_sample.parquet).
Stage `review` : prints 30 random matches and 30 random non-matches of a
                 lexicon from that sample, for reading by hand.
Stage `models` : for each characteristic, a within class x registration-year
                 regression of survival (100 = passed the five-year declaration)
                 on lead (within-cell percentile, centred at zero), the
                 characteristic (centred), their interaction, and the paper's
                 controls (counsel, intent-to-use, log description length, log
                 owner filing count, US and China domicile, 44(e) and 66(a)
                 bases); SEs clustered on the normalized owner. Holm adjustment
                 across characteristics, separately for main effects and for
                 interactions. Class-defined characteristics are constant within
                 a cell, so their main effect is reported from a registration-
                 year-FE model (descriptive) and only their interaction is
                 identified under the cell FE. A joint model enters all text
                 characteristics and interactions together.

Output: paper/results/strategy_dimensions_tests.json
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from strategy_lexicons import LEXICONS, CLASS_DEFINED, flag_exprs, class_flag_exprs  # noqa: E402

FLAGS = PROC / "strategy_flags.parquet"
REVIEW = PROC / "strategy_review_sample.parquet"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
NREV = 30
CONTROLS = ["has_attorney", "itu", "log_len", "log_owner_n", "dom_us", "dom_cn", "basis_44e", "basis_66a"]


def log(m):
    print(m, file=sys.stderr, flush=True)


def free_gb() -> float:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MS(); s.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / 2**30
    except Exception:
        return float("nan")


def wait_mem(need=4.0):
    import time
    while free_gb() < need:
        log(f"  [mem] {free_gb():.1f} GB free; waiting")
        time.sleep(60)


# ------------------------------------------------------------------ stage 1
def stage_flags():
    keys = pl.read_parquet(PROC / "battery_frame.parquet", columns=["serial_number", "cls"])
    K = list(LEXICONS)
    out = []
    for c in CLASSES:
        wait_mem()
        want = set(keys.filter(pl.col("cls") == c)["serial_number"].to_list())
        pf = pq.ParquetFile(PROC / f"tm_class{c}.parquet")
        seen = set()
        for rb in pf.iter_batches(batch_size=150_000, columns=["serial_number", "goods_services"]):
            b = pl.from_arrow(rb)
            b = b.filter(pl.col("serial_number").is_in(want) & ~pl.col("serial_number").is_in(seen))
            b = b.unique("serial_number")
            if b.height == 0:
                continue
            seen.update(b["serial_number"].to_list())
            b = b.with_columns(pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs"),
                               pl.lit(c).alias("cls"))
            f = b.with_columns(flag_exprs("gs"))
            out.append(f.select("serial_number", "cls", *[f"f_{k}" for k in K]))
            del b, f
        log(f"  [flags] {c}: {len(seen):,} of {len(want):,}")
        gc.collect()
    fl = pl.concat(out)
    fl.write_parquet(FLAGS)
    log(f"[flags] {fl.height:,} rows -> {FLAGS}")



def stage_review():
    """Print 30 random matches (with the matched span marked) and 30 random
    non-matches of one or more lexicons from the review sample, for reading.
    Usage: strategy_dimensions.py review <key[,key...]|all> [m|n|mn] [seed]"""
    import re
    import strategy_lexicons as SL
    d = pl.read_parquet(REVIEW).with_columns(
        pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs")).sort("serial_number")
    d = d.with_columns(SL.flag_exprs("gs"))
    keys = list(SL.LEXICONS) if sys.argv[2] == "all" else sys.argv[2].split(",")
    tags = sys.argv[3] if len(sys.argv) > 3 else "mn"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    for k in keys:
        for tag in tags:
            pool = d.filter(pl.col(f"f_{k}") == (tag == "m"))
            print(f"===== {k} {tag} (pool {pool.height} of {d.height})")
            for i, r in enumerate(pool.sample(min(NREV, pool.height), seed=seed).iter_rows(named=True)):
                s_ = (SL.first_match(r["gs"], k) or r["gs"][:160]) if tag == "m" else r["gs"][:170]
                print(f"{i:2d} c{r['cls']} {re.sub(chr(92) + 's+', ' ', s_)}")


# characteristic -> (identifiable, reference keys in paper/results/strategy_dimensions_refs.md)
TABLE_META = {
    "b2b": ("partly", "robinson1988; robinson1985"),
    "b2c": ("partly", "robinson1985; golder1993"),
    "local": ("yes", "bresnahan1991; syverson2004"),
    "broad_scope": ("partly", "syverson2004; sutton1991"),
    "bulky": ("partly", "syverson2004"),
    "services_cls": ("yes (class)", "song1999; zeithaml1985"),
    "credence": ("yes", "darby1973; dulleck2006"),
    "experience": ("yes", "nelson1970; nelson1974"),
    "search": ("yes", "nelson1970; nelson1974"),
    "durable": ("yes", "coase1972; bulow1982"),
    "consumable": ("yes", "coase1972; klemperer1995"),
    "customized": ("partly", "utterback1975; suarez1995"),
    "subscription": ("yes", "klemperer1987; farrell2007"),
    "switching": ("partly", "klemperer1987; klemperer1995; farrell2007"),
    "network": ("yes", "katz1985; rochet2003; suarez2007"),
    "complement": ("yes", "teece1986; adner2010"),
    "approval": ("yes", "grabowski1992; scottmorton1999"),
    "approval_cls": ("yes (class)", "grabowski1992; scottmorton1999"),
    "licensed": ("yes", "teece1986; bresnahan1991"),
    "ad_intensive": ("partly", "sutton1991; schmalensee1982; bronnenberg2009"),
    "ad_intensive_cls": ("partly (class)", "sutton1991; bronnenberg2009"),
    "tech": ("yes", "teece1986; christensen1996; suarez2007"),
    "tech_cls": ("yes (class)", "teece1986; suarez2007"),
    "digital": ("yes", "shapiro1999; bakos1999"),
    "intermediary": ("yes", "spulber1996; teece1986"),
    "direct": ("yes", "brynjolfsson2000; spulber1996"),
    "perishable": ("yes", "gallego1994"),
    "perishable_cls": ("partly (class)", "gallego1994"),
    "fad": ("partly", "bikhchandani1992; pesendorfer1995"),
    "fad_cls": ("partly (class)", "pesendorfer1995"),
    "luxury": ("partly", "bagwell1996; amaldoss2005"),
    "discount": ("no", "bagwell1996"),
}


def stage_table():
    """Markdown results table from strategy_dimensions_tests.json (printed)."""
    import strategy_lexicons as SL
    r = json.loads((RES / "strategy_dimensions_tests.json").read_text(encoding="utf-8"))

    def st(p):
        return "" if p is None else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    def label(k):
        return (SL.LEXICONS.get(k) or SL.CLASS_DEFINED[k])["label"]

    rows = ["| Characteristic | References | Identifiable | Precision; misses | Share | "
            "Survival difference, pp | Change in lead effect, pp | Lead effect if present | Joint model: change in lead effect |",
            "|---|---|---|---|---|---|---|---|---|"]
    order = list(TABLE_META)
    for k in order:
        e, j = r["chars"][k], r["joint"][k]
        if k in SL.PRECISION:
            tp, n, _, miss = SL.PRECISION[k]
            prec = f"{tp}/{n}; " + ("-" if miss is None else f"{miss}/30")
        else:
            prec = "class"
        m = e["main"]
        mtxt = (f"{m[0]:+.1f} ({m[1]:.1f}){st(e['holm_main_all'])}" if not e["class_defined"]
                else f"{m[0]:+.1f} ({m[1]:.1f}) [year FE]")
        it = e["inter"]
        rows.append(f"| {label(k)} | {TABLE_META[k][1]} | {TABLE_META[k][0]} | {prec} | {e['share']:.1%} | {mtxt} | "
                    f"{it[0]:+.1f} ({it[1]:.1f}){st(e['holm_inter_all'])} | {e['lead_if_flag']:+.1f} | "
                    f"{j['inter'][0]:+.1f} ({j['inter'][1]:.1f}){st(j['holm_inter'])} |")
    print("\n".join(rows))


def stage_sample():
    """Texts of a fixed uniform random sample of frame registrations (REVIEW_N,
    default 40,000, seed 20261005), for hand review of the lexicons with the
    `review` stage."""
    n = int(os.environ.get("REVIEW_N", 40000))
    keys = pl.read_parquet(PROC / "battery_frame.parquet", columns=["serial_number", "cls"]).sample(
        n, seed=20261005)
    parts = []
    for c in CLASSES:
        want = set(keys.filter(pl.col("cls") == c)["serial_number"].to_list())
        pf = pq.ParquetFile(PROC / f"tm_class{c}.parquet")
        for rb in pf.iter_batches(batch_size=150_000, columns=["serial_number", "goods_services"]):
            b = pl.from_arrow(rb).filter(pl.col("serial_number").is_in(want))
            if b.height:
                parts.append(b.with_columns(pl.lit(c).alias("cls")))
        log(f"  [sample] {c}")
    out = pl.concat(parts).unique("serial_number")
    out.write_parquet(REVIEW)
    log(f"[sample] {out.height:,} texts -> {REVIEW}")


# ------------------------------------------------------------------ stage 2
def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def pval(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2))


def stage_models():
    from combined_model import Design
    wait_mem(4.5)
    cols = ["serial_number", "cls", "cell", "reg_year", "owner_key", "failed1", "z"] + CONTROLS
    d = pl.read_parquet(PROC / "battery_frame.parquet", columns=cols)
    fl = pl.read_parquet(FLAGS)
    d = d.join(fl, on=["serial_number", "cls"], how="left").with_columns(class_flag_exprs("cls"))
    TK = list(LEXICONS)
    CK = list(CLASS_DEFINED)
    n_missing = int(d[f"f_{TK[0]}"].null_count())
    d = d.with_columns([pl.col(f"f_{k}").fill_null(False) for k in TK])
    d = d.with_columns(
        (100 * (1 - pl.col("failed1").cast(pl.Float64))).alias("surv"),
        ((pl.col("z").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5).alias("lead"),
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in CONTROLS],
    )
    log(f"[models] {d.height:,} rows; {n_missing:,} without text")
    y = d["surv"].to_numpy()
    lead = d["lead"].to_numpy()
    Xc = np.column_stack([d[c].to_numpy() for c in CONTROLS]).astype(np.float32)
    des = Design(d)
    desY = Design(d, fe="reg_year")
    res = {"n": d.height, "n_without_text": n_missing, "controls": CONTROLS, "chars": {}}

    r0 = des.ols(y, np.column_stack([lead, Xc]).astype(np.float32), ["lead"] + CONTROLS)
    res["lead_only"] = {"b": r0["lead"][0], "se": r0["lead"][1]}
    log(f"  lead only {r0['lead'][0]:+.2f} ({r0['lead'][1]:.2f})")
    base_surv = float(y.mean())

    def fit(k, dd):
        v = d[f"f_{k}"].cast(pl.Float64).to_numpy()
        mu = float(v.mean())
        X = np.column_stack([lead, v - mu, (v - mu) * lead, Xc]).astype(np.float32)
        names = ["lead", k, f"{k}:lead"] + CONTROLS
        r = dd.ols(y, X, names)
        return r, mu, v

    for k in TK + CK:
        is_cls = k in CLASS_DEFINED
        r, mu, v = fit(k, des)
        inter = r[f"{k}:lead"]
        e = {"share": mu, "class_defined": is_cls,
             "surv_with": float(y[v == 1].mean()) if mu > 0 else None,
             "surv_without": float(y[v == 0].mean()),
             "lead": r["lead"][:2], "inter": inter[:2], "p_inter": pval(*inter[:2])}
        # lead effect among flagged and unflagged filings
        b0, s0, c0 = r["lead"]
        bi, si, _ = inter
        e["lead_if_flag"] = b0 + bi * (1 - mu)
        e["lead_if_not"] = b0 - bi * mu
        if is_cls:
            ry, _, _ = fit(k, desY)
            e["main"] = ry[k][:2]
            e["main_fe"] = "registration year"
        else:
            e["main"] = r[k][:2]
            e["main_fe"] = "class x registration year"
        e["p_main"] = pval(*e["main"])
        res["chars"][k] = e
        log(f"  {k:16s} share {mu:6.3f}  main {e['main'][0]:+6.2f} ({e['main'][1]:.2f})  "
            f"x lead {inter[0]:+6.2f} ({inter[1]:.2f})")
        gc.collect()

    for tag, keyset in (("text", TK), ("all", TK + CK)):
        hm = holm({k: res["chars"][k]["p_main"] for k in keyset if not res["chars"][k]["class_defined"]})
        hi = holm({k: res["chars"][k]["p_inter"] for k in keyset})
        for k in keyset:
            res["chars"][k][f"holm_main_{tag}"] = hm.get(k)
            res["chars"][k][f"holm_inter_{tag}"] = hi[k]

    # joint model: all text characteristics together (class-defined interactions added)
    wait_mem(4.0)
    V = np.column_stack([d[f"f_{k}"].cast(pl.Float64).to_numpy() for k in TK + CK])
    mus = V.mean(axis=0)
    Vc = (V - mus).astype(np.float32)
    X = np.column_stack([lead[:, None].astype(np.float32), Vc[:, :len(TK)], Vc * lead[:, None].astype(np.float32),
                         Xc]).astype(np.float32)
    names = ["lead"] + TK + [f"{k}:lead" for k in TK + CK] + CONTROLS
    del V, Vc
    gc.collect()
    rj = des.ols(y, X, names)
    del X
    joint = {"lead": rj["lead"][:2]}
    for k in TK + CK:
        joint[k] = {"main": rj[k][:2] if k in rj else None, "inter": rj[f"{k}:lead"][:2]}
    pj = holm({k: pval(*joint[k]["inter"]) for k in TK + CK})
    pm = holm({k: pval(*joint[k]["main"]) for k in TK})
    for k in TK + CK:
        joint[k]["holm_inter"] = pj[k]
        joint[k]["holm_main"] = pm.get(k)
    res["joint"] = joint
    res["mean_survival"] = base_surv
    # pairwise overlap (phi) among text flags, for reading the joint model
    F = np.column_stack([d[f"f_{k}"].cast(pl.Float32).to_numpy() for k in TK])
    res["phi"] = {"keys": TK, "matrix": np.round(np.corrcoef(F, rowvar=False), 3).tolist()}
    del F
    import strategy_lexicons as SL
    res["lexicon_review"] = {k: dict(zip(["true_pos", "read", "seed", "misses_in_30_nonmatches"], v))
                             for k, v in SL.PRECISION.items()}
    res["labels"] = {k: (SL.LEXICONS.get(k) or SL.CLASS_DEFINED[k])["label"] for k in TK + CK}
    (RES / "strategy_dimensions_tests.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log("[models] wrote strategy_dimensions_tests.json")


if __name__ == "__main__":
    {"flags": stage_flags, "sample": stage_sample, "review": stage_review, "table": stage_table,
     "models": stage_models}[sys.argv[1]]()
