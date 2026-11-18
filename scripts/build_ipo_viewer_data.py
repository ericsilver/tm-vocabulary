"""Build the per-class data files behind the online IPO viewer.

One JSON per Nice class. Each row of the dot arrays is a registration
(registration years 1996-2018, scored on the production rolling measure),
positioned by lead and atypicality standardized within class and
registration year, and tiered by the latest gate its record reaches:

  0  registered, cancelled at the five-year proof of continued use
     (dated C8../C71T cancellation at registration age 4.0-8.5)
  1  passed the five-year proof
  2  owner appears in SEC financial reporting records (name-matched)
  3  owner carries an IPO marker (8-A registration or IPO date)

Tiers 2 and 3 are kept in full; tiers 0 and 1 are down-sampled so a class
file stays around a few hundred kilobytes. Coordinates are rounded to two
decimals and clipped to +/-3.5 standard deviations.

Three further blocks are computed on EVERY registration of the class, not on
the down-sampled dots, so the page's rates are exact:

  grid   counts by registration year and half-standard-deviation cell of the
         plane: registrations, passed the five-year proof (a property of the
         mark, whatever the owner's tier), owner in SEC reporting, owner with
         an IPO marker. The page sums the years in view and colors each cell
         by its rate against the class's.
  fit    per-registration-year sums from which the page computes the
         least-squares slope of each outcome on lead and on atypicality for
         any range of years.
  named  a handful of registrations to label, each with its description, its
         theme mix under the production model, and the class's average share
         of those themes in the five calendar years before and after filing.
         Marks named in the papers come first; the rest are chosen by rule
         from the house marks of the largest public owners (see pick_named).

Output: docs/online-appendix/ipo-viewer/data/class_{NNN}.json
        docs/online-appendix/ipo-viewer/data/manifest.json
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
OUT = REPO / "docs" / "online-appendix" / "ipo-viewer" / "data"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
RY_LO, RY_HI = 1996, 2018
GATE_LO, GATE_HI = 4.0, 8.5
CLIP = 3.5
GRID = 14                 # cells per axis: half a standard deviation each
MAX_BASE_DOTS = 22_000    # cap on tiers 0+1 per class
SEED = 20260825
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
N_NAMED = 8
TEXT_CHARS = 260

# Marks named in the papers' figures. A serial is used where the figure scripts
# assert one (fig_registration_only.py, fig_internet_1999_2x2.py); the class-32
# marks are resolved by mark text and filing year, as quadrant_whiteclaw_hires.py does.
PAPER_MARKS = {
    "009": [("75982871", None, None), ("86359718", None, None), ("77318565", None, None),
            ("75493408", None, None), ("75754414", None, None)],
    "035": [("75277670", None, None), ("76314811", None, None), ("85023193", None, None),
            ("75412591", None, None), ("75669553", None, None), ("75737216", None, None)],
    "042": [("75978469", None, None), ("75772855", None, None), ("75882670", None, None)],
    "032": [(None, "WHITE CLAW SELTZER WORKS", 2016), (None, "RED BULL", 1995),
            (None, "MONSTER ENERGY", 2002), (None, "KOMBUCHA", 1997),
            (None, "SMARTWATER", 1997), (None, "CELSIUS", 2004),
            (None, "ATHLETIC BREWING", 2017)],
}


# Rule picks set aside by hand: the record lists the CURRENT owner, so a mark
# bought after filing reads as the buyer's own filing. (class, mark)
SKIP_MARKS = {("009", "META")}


def log(m): print(m, file=sys.stderr, flush=True)


def first_token(s: str | None) -> str:
    toks = [t for t in re.split(r"[^A-Z0-9]+", (s or "").upper()) if t and t != "THE"]
    return toks[0] if toks else ""


def pick_named(d: pl.DataFrame, c: str) -> list[tuple[str, str]]:
    """Choose the registrations to label: (serial, reason).

    Paper-named marks first. Then, from the house marks (the mark's first word
    is the owner's first word) of SEC-reporting owners for which this class is
    one of their three largest and holds at least three of their registrations,
    ranked by the owner's peak annual revenue:
    the largest owner centered on the class, the most leading, the most lagging,
    the next largest, the most atypical, the largest whose house mark was
    cancelled at the five-year proof, and the third largest. The position picks
    are taken among the 30 largest owners, IPO-marker owners where there are
    at least eight.
    """
    picks: list[tuple[str, str]] = []
    for serial, mark, year in PAPER_MARKS.get(c, []):
        if serial:
            hit = d.filter(pl.col("serial_number") == serial)
        else:
            hit = d.filter((pl.col("fy") == year) & pl.col("mark").str.to_uppercase().str.contains(
                f"^{re.escape(mark)}$|^{re.escape(mark)} ")).sort("filing_date")
        if hit.is_empty():
            log(f"  [{c}] paper mark not in frame: {serial or mark}")
            continue
        # a serial is the papers' own filing; a name match is the earliest registered filing of that name
        picks.append((hit["serial_number"][0], "paper" if serial else "name"))

    pool = d.filter(pl.col("revenue").is_not_null() & pl.col("mark").is_not_null()
                    & pl.col("top3")
                    & (pl.col("mark").str.len_chars() <= 28)
                    & (pl.col("tok_m").str.len_chars() >= 3)
                    & (pl.col("tok_m") == pl.col("tok_o")))
    # one house mark per owner: the shortest mark text, then the earliest filing
    pool = pool.with_columns(pl.col("mark").str.len_chars().alias("mlen")).sort(
        ["cik", "mlen", "filing_date"]).unique("cik", keep="first", maintain_order=True).sort(
        "revenue", descending=True)
    taken = {s for s, _ in picks}
    taken_cik = set(d.filter(pl.col("serial_number").is_in(list(taken)))["cik"].drop_nulls().to_list())

    def take(frame: pl.DataFrame, reason: str) -> None:
        for r in frame.iter_rows(named=True):
            if r["serial_number"] in taken or r["cik"] in taken_cik or (c, r["mark"]) in SKIP_MARKS:
                continue
            picks.append((r["serial_number"], reason))
            taken.add(r["serial_number"]); taken_cik.add(r["cik"])
            return

    pool = pool.filter(pl.col("n_cls") >= 3)
    big = pool.head(30)
    ipo_big = pool.filter(pl.col("ipo")).head(30)
    ext = ipo_big if ipo_big.height >= 8 else big
    leaders = pool.filter(pl.col("modal"))
    steps = [
        (leaders, "largest"),
        (ext.sort("zL", descending=True), "leading"),
        (ext.sort("zL"), "lagging"),
        (leaders, "largest"),
        (ext.sort("zA", descending=True), "atypical"),
        (big.filter(pl.col("failed")), "lapsed"),
        (leaders, "largest"),
        (pool, "largest"),
        (pool, "largest"),
    ]
    for frame, reason in steps:
        if len(picks) >= N_NAMED:
            break
        take(frame, reason)
    # small classes: fill to four from any mark of an SEC-reporting owner with revenue
    if len(picks) < 4:
        rest = d.filter(pl.col("revenue").is_not_null() & pl.col("mark").is_not_null()
                        & (pl.col("mark").str.len_chars() <= 28)).sort(
            ["cik", "filing_date"]).unique("cik", keep="first", maintain_order=True).sort(
            "revenue", descending=True)
        while len(picks) < 4:
            before = len(picks)
            take(rest, "largest")
            if len(picks) == before:
                break
    return picks[:N_NAMED]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((RES / "per_industry_names.json").read_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location("t50", REPO / "scripts" / "themes_t50_page.py")
    t50 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t50)
    theme_labels = t50.LABELS

    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet",
                         columns=["owner_name", "norm", "cik"]).unique("owner_name")
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet",
                         columns=["owner_name", "in_8a", "ipo_date"])
    ipo_owners = fm.filter((pl.col("in_8a").fill_null(0) == 1)
                           | pl.col("ipo_date").is_not_null()).select("owner_name").unique().with_columns(
        pl.lit(True).alias("ipo"))
    rev = pl.read_parquet(PROC / "sec_firm_year.parquet", columns=["cik", "revenue"]).filter(
        pl.col("revenue").is_finite() & (pl.col("revenue") > 0)).group_by(
        "cik").agg(pl.col("revenue").max())
    log(f"[owners] SEC {cw.height:,}  IPO {ipo_owners.height:,}  with revenue {rev.height:,}")

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())

    # which classes each SEC-reporting owner's registrations sit in
    parts = []
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        if not tp.exists():
            continue
        t = pl.read_parquet(tp, columns=["serial_number", "registration_date", "owner_name"]).filter(
            pl.col("registration_date").fill_null("").str.slice(0, 4).cast(pl.Int32, strict=False)
            .is_between(RY_LO, RY_HI)).unique("serial_number").join(cw.select("owner_name", "cik"),
                                                                   on="owner_name", how="inner")
        parts.append(t.group_by("cik").len().with_columns(pl.lit(c).alias("cls")))
        del t
    oc = pl.concat(parts).with_columns(
        pl.col("len").rank("ordinal", descending=True).over("cik").alias("rk"))
    del parts
    gc.collect()

    m = joblib.load(PROC / "topic_model.joblib")
    lda = m["lda"]
    lda.n_jobs = 1
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    cym = pl.read_parquet(PROC / "theme_full" / "classyear_mix.parquet")
    T = lda.components_.shape[0]

    manifest = []
    only = [v for v in os.environ.get("VIEWER_ONLY", "").split(",") if v]   # test runs
    for c in (only or CLASSES):
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                          "owner_name", "mark_identification"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
        ).drop_nulls("rd").sort("serial_number").unique("serial_number", keep="first", maintain_order=True)
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite()).unique("serial_number", keep="first")
        d = tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.col("rd").dt.year().alias("ry"),
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            pl.col("mark_identification").alias("mark"),
            pl.col("topic_dkl").alias("L"),
            ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
        d = d.filter(pl.col("ry").is_between(RY_LO, RY_HI))
        d = d.join(ev, on="serial_number", how="left").with_columns(
            ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
            ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).alias("failed"))
        d = d.join(cw, on="owner_name", how="left").join(ipo_owners, on="owner_name", how="left").join(
            rev, on="cik", how="left").join(
            oc.filter(pl.col("cls") == c).select("cik", "rk", pl.col("len").alias("n_cls")),
            on="cik", how="left")
        d = d.with_columns(
            ((pl.col("L") - pl.col("L").mean().over("ry")) / pl.col("L").std().over("ry")).alias("zL"),
            ((pl.col("A") - pl.col("A").mean().over("ry")) / pl.col("A").std().over("ry")).alias("zA"),
            pl.col("cik").is_not_null().alias("sec"),
            pl.col("ipo").fill_null(False),
            (pl.col("rk") == 1).fill_null(False).alias("modal"),
            (pl.col("rk") <= 3).fill_null(False).alias("top3"),
            pl.col("mark").map_elements(first_token, return_dtype=pl.Utf8).alias("tok_m"),
            pl.col("norm").map_elements(first_token, return_dtype=pl.Utf8).alias("tok_o"),
        ).filter(pl.col("zL").is_finite() & pl.col("zA").is_finite())
        d = d.with_columns(
            pl.when(pl.col("ipo")).then(3)
              .when(pl.col("sec")).then(2)
              .when(~pl.col("failed")).then(1)
              .otherwise(0).cast(pl.Int8).alias("g"),
            pl.col("zL").clip(-CLIP, CLIP).alias("x"),
            pl.col("zA").clip(-CLIP, CLIP).alias("y"))

        # ---- exact rates: grid and regression sums over every registration ----
        cell = 2 * CLIP / GRID
        gd = d.with_columns(
            ((pl.col("x") + CLIP) / cell).floor().clip(0, GRID - 1).cast(pl.Int32).alias("bx"),
            ((pl.col("y") + CLIP) / cell).floor().clip(0, GRID - 1).cast(pl.Int32).alias("by"),
            (~pl.col("failed")).cast(pl.Float64).alias("o_p"),
            (pl.col("sec") | pl.col("ipo")).cast(pl.Float64).alias("o_s"),
            pl.col("ipo").cast(pl.Float64).alias("o_i"))
        g = gd.group_by(["ry", "bx", "by"]).agg(
            pl.len().alias("n"), pl.col("o_p").sum().alias("p"),
            pl.col("o_s").sum().alias("s"), pl.col("o_i").sum().alias("i")).sort(["ry", "by", "bx"])
        grid = {"ry": g["ry"].to_list(),
                "c": (g["by"] * GRID + g["bx"]).to_list(),
                "n": g["n"].to_list(),
                "p": [int(v) for v in g["p"]], "s": [int(v) for v in g["s"]],
                "i": [int(v) for v in g["i"]]}
        aggs = [pl.len().alias("n"), pl.col("x").sum().alias("sx"), (pl.col("x") ** 2).sum().alias("sxx"),
                pl.col("y").sum().alias("sy"), (pl.col("y") ** 2).sum().alias("syy")]
        for k in "psi":
            aggs += [pl.col(f"o_{k}").sum().alias(f"{k}_o"),
                     (pl.col(f"o_{k}") * pl.col("x")).sum().alias(f"{k}_x"),
                     (pl.col(f"o_{k}") * pl.col("y")).sum().alias(f"{k}_y")]
        f = gd.group_by("ry").agg(aggs).sort("ry")
        fit = {col: [round(float(v), 3) for v in f[col]] if col != "ry" else f[col].to_list()
               for col in f.columns}
        del gd, g, f

        # ---- named registrations ----
        picks = pick_named(d, c)
        named = []
        if picks:
            serials = [s for s, _ in picks]
            txt = pl.scan_parquet(tp).select("serial_number", "goods_services").filter(
                pl.col("serial_number").is_in(serials)).collect().unique("serial_number", keep="first")
            tmap = dict(zip(txt["serial_number"], txt["goods_services"]))
            cm = cym.filter(pl.col("nice") == c)
            for serial, reason in picks:
                r = d.filter(pl.col("serial_number") == serial).row(0, named=True)
                text = " ".join((tmap.get(serial) or "").split())
                th = lda.transform(vec.transform([text]))[0]
                th = th / th.sum()
                fy = r["fy"]

                def window(lo, hi):
                    w = cm.filter(pl.col("fy").is_between(lo, hi))
                    if w.is_empty() or w["n"].sum() == 0:
                        return None
                    n = w["n"].to_numpy().astype(float)
                    M = w.select([f"theta_{k}" for k in range(T)]).to_numpy()
                    return (M * n[:, None]).sum(axis=0) / n.sum()
                before, after = window(fy - 5, fy - 1), window(fy + 1, fy + 5)
                mix = []
                for k in np.argsort(-th)[:3]:
                    if th[k] < 0.05:
                        break
                    mix.append([int(k), round(float(th[k]), 3),
                                None if before is None else round(float(before[k]), 3),
                                None if after is None else round(float(after[k]), 3)])
                fd, rd = r["filing_date"], r["rd"]
                named.append({
                    "s": serial, "m": r["mark"] or "(design mark)", "o": r["owner_name"],
                    "fd": f"{fd[:4]}-{fd[4:6]}-{fd[6:8]}", "rd": rd.isoformat(), "ry": int(r["ry"]),
                    "x": round(float(r["x"]), 2), "y": round(float(r["y"]), 2),
                    "zl": round(float(r["zL"]), 2), "za": round(float(r["zA"]), 2),
                    "g": int(r["g"]), "f": bool(r["failed"]),
                    "sec": bool(r["sec"] or r["ipo"]), "ipo": bool(r["ipo"]),
                    "t": text[:TEXT_CHARS] + ("…" if len(text) > TEXT_CHARS else ""),
                    "th": mix, "why": reason})

        # ---- dots ----
        top = d.filter(pl.col("g") >= 2)
        base = d.filter(pl.col("g") < 2)
        if base.height > MAX_BASE_DOTS:
            base = base.sample(MAX_BASE_DOTS, seed=SEED)
        keep = pl.concat([base, top]).sort("g", maintain_order=True)  # ascending; the page draws in array order
        rec = {
            "cls": c,
            "name": names.get(c, c),
            "n_total": int(d.height),
            "n_by_tier": {str(t): int((d["g"] == t).sum()) for t in range(4)},
            "base_sampled": int(base.height),
            "ry": [int(v) for v in keep["ry"]],
            "x": [round(float(v), 2) for v in keep["x"]],
            "y": [round(float(v), 2) for v in keep["y"]],
            "g": [int(v) for v in keep["g"]],
            "grid": grid, "fit": fit, "named": named,
        }
        (OUT / f"class_{c}.json").write_text(json.dumps(rec, separators=(",", ":"), ensure_ascii=False),
                                             encoding="utf-8")
        manifest.append({"cls": c, "name": rec["name"], "n_total": rec["n_total"],
                         "n_by_tier": rec["n_by_tier"]})
        log(f"[{c}] {d.height:,} regs  tiers {rec['n_by_tier']}  dots {keep.height:,}  "
            f"named: " + "; ".join(f"{n['m']} ({n['why']})" for n in named))
        del tm, sc, d, top, base, keep
        gc.collect()
    (OUT / "manifest.json").write_text(json.dumps(
        {"ry_lo": RY_LO, "ry_hi": RY_HI, "clip": CLIP, "grid": GRID,
         "themes": {str(k): v for k, v in theme_labels.items()},
         "classes": manifest}, indent=1), encoding="utf-8")
    log("[done] ipo-viewer data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
