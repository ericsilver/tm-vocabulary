"""An acquisition rung for the ladder, from the USPTO Trademark Assignment Dataset.

Listing is one exit; acquisition is the other, and in consumer categories the
usual one. The record's own outcome cannot see it: a brand retired after its
owner was bought counts as a product that failed. The USPTO Trademark
Assignment Dataset (2023 release, transactions recorded through April 2024)
records every transfer of trademark ownership with its conveyance type, the
assignor and assignee, and the serial numbers it covers.

Frame: owners whose first registered, scored filing was made 2009-2018 (the
ladder's frame), one row per owner, keyed on the debut filing's serial.

A debut is ACQUIRED when its serial appears in a transfer that is
  - an assignment of the entire interest or a merger (conveyance groups
    'assignment' and 'merger'; partial "undivided part" assignments excluded),
  - recorded after the debut filing date,
  - made by the debut owner (assignor's normalized name equals the owner's),
  - between companies (neither party recorded as an individual, which drops
    founders moving a mark into their own new company and person-to-person
    brand trading),
  - to an unrelated party (assignee's normalized name differs from the
    owner's and does not share its first word, which screens out most moves
    between a firm and its own holding company),
  - and not by court order, bankruptcy, foreclosure or receivership, which are
    counted separately as DISTRESSED transfers.
Flags on acquired debuts:
  - ESTABLISHED BUYER: the assignee owns at least 25 marks in the corpus.
  - PUBLIC ACQUIRER: the assignee's normalized name is an SEC registrant in
    the project's crosswalk.
  - KEPT IN USE: the debut mark was never cancelled for non-use (no Section 8 /
    71 cancellation through April 2026) -- a going concern rather than a brand
    bought to be retired.

Outputs rates overall, by fifth of lead and of atypicality (fifths cut within
Nice class, debut year and fifth of description length, as in the ladder),
and for the curated vocabularies against same-debut-year baselines.

Output: paper/results/acquisition_rung.json
"""
from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw" / "tm_assignment"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
ESTABLISHED = 25
DISTRESS = r"bankrupt|foreclos|court order|receiver|trustee|sheriff|liquidat|benefit of creditors"


def log(m): print(m, file=sys.stderr, flush=True)


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_map(names: pl.Series, normalize) -> pl.DataFrame:
    u = names.drop_nulls().unique()
    return pl.DataFrame({"raw": u, "norm": [normalize(s) for s in u.to_list()]})


def main() -> int:
    normalize = load_module("sl", "sec_link.py").normalize
    vocab = load_module("vt", "vocab_ipo_timing.py").PATTERNS

    # ---- debut frame -------------------------------------------------------
    parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "owner_name", "filing_date",
                                          "registration_date", "goods_services"]).filter(
            pl.col("owner_name").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"])
        tm = tm.join(sc, on="serial_number").filter(
            pl.col("topic_kl_vs_past").is_finite() & pl.col("topic_kl_vs_future").is_finite()
        ).with_columns(pl.lit(c).alias("cls"),
                       pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs"))
        tm = tm.with_columns([pl.col("gs").str.contains(p).alias(n) for n, p in vocab.items()])
        parts.append(tm.select(["serial_number", "owner_name", "filing_date", "cls",
                                "topic_kl_vs_past", "topic_kl_vs_future", "gs"] + list(vocab)))
        del tm, sc
        gc.collect()
    d = pl.concat(parts)
    del parts
    debut = d.group_by("owner_name").agg(pl.col("filing_date").min().alias("dd"))
    f = d.join(debut, on="owner_name").filter(pl.col("filing_date") == pl.col("dd")).sort(
        "serial_number").unique("owner_name", keep="first").with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("fy"),
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fdd"),
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
        pl.col("gs").str.len_chars().alias("glen")).filter(pl.col("fy").is_between(2009, 2018)).drop("gs")
    del d
    gc.collect()
    om = norm_map(f["owner_name"], normalize).rename({"raw": "owner_name", "norm": "owner_norm"})
    f = f.join(om, on="owner_name", how="left").with_columns(
        pl.col("owner_norm").str.split(" ").list.first().alias("owner_first"),
        pl.col("serial_number").cast(pl.Utf8).str.strip_chars().alias("serial"))
    log(f"[frame] {f.height:,} debut owners 2009-2018")

    # ---- transfers ---------------------------------------------------------
    A = pl.read_csv(RAW / "tm_assignment.csv", infer_schema_length=0,
                    columns=["rf_id", "record_dt", "convey_text"])
    C = pl.read_csv(RAW / "tm_convey.csv", infer_schema_length=0)
    OR = pl.read_csv(RAW / "tm_assignor.csv", infer_schema_length=0, columns=["rf_id", "or_name", "or_legal_entity_text"])
    EE = pl.read_csv(RAW / "tm_assignee.csv", infer_schema_length=0, columns=["rf_id", "ee_name", "ee_legal_entity_text"])
    DOC = pl.read_csv(RAW / "tm_docid.csv", infer_schema_length=0, columns=["rf_id", "serial"]).filter(
        pl.col("serial").is_not_null())
    T = A.join(C, on="rf_id").filter(pl.col("conv_group").is_in(["assignment", "merger"])).with_columns(
        pl.col("convey_text").fill_null("").str.to_lowercase().alias("ct"),
        pl.col("record_dt").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rdt")).filter(
        ~pl.col("ct").str.contains("undivided"))
    T = T.with_columns(pl.col("ct").str.contains(DISTRESS).alias("distressed")).select(
        "rf_id", "rdt", "conv_group", "distressed")
    # only transactions touching a debut serial
    DOC = DOC.join(f.select("serial"), on="serial", how="semi")
    T = T.join(DOC, on="rf_id")
    orn = norm_map(OR["or_name"], normalize).rename({"raw": "or_name", "norm": "or_norm"})
    een = norm_map(EE["ee_name"], normalize).rename({"raw": "ee_name", "norm": "ee_norm"})
    T = T.join(OR.join(orn, on="or_name"), on="rf_id").join(EE.join(een, on="ee_name"), on="rf_id")
    log(f"[transfers] {T.height:,} assignor/assignee rows touching debut serials")

    x = T.join(f.select("serial", "owner_norm", "owner_first", "fdd"), on="serial").filter(
        (pl.col("rdt") > pl.col("fdd")) & (pl.col("or_norm") == pl.col("owner_norm"))
        & (pl.col("ee_norm") != pl.col("owner_norm"))
        & (pl.col("ee_norm").str.split(" ").list.first() != pl.col("owner_first"))
        & (pl.col("or_legal_entity_text").fill_null("") != "INDIVIDUAL")
        & (pl.col("ee_legal_entity_text").fill_null("") != "INDIVIDUAL"))
    # established buyer: the assignee owns at least ESTABLISHED marks in the corpus
    own = []
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        if tp.exists():
            own.append(pl.read_parquet(tp, columns=["serial_number", "owner_name"]))
    own = pl.concat(own).unique("serial_number").drop_nulls("owner_name")
    own_counts = own.group_by("owner_name").len()
    onm = norm_map(own_counts["owner_name"], normalize).rename({"raw": "owner_name", "norm": "ee_norm"})
    size = own_counts.join(onm, on="owner_name").group_by("ee_norm").agg(pl.col("len").sum().alias("ee_marks"))
    del own, own_counts, onm
    x = x.with_columns((pl.col("distressed") | pl.col("or_name").str.to_lowercase().str.contains(DISTRESS)).alias("distressed"))
    x = x.join(size, on="ee_norm", how="left").with_columns(
        (pl.col("ee_marks").fill_null(0) >= ESTABLISHED).alias("established"))
    sec = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")["owner_name"].drop_nulls().to_list())
    sec_norm = {normalize(s) for s in sec}
    x = x.with_columns(pl.col("ee_norm").is_in(list(sec_norm)).alias("public_acq"))
    first = x.sort("rdt").group_by("serial").agg(
        pl.col("rdt").first().alias("acq_date"),
        pl.col("distressed").first().alias("acq_distressed"),
        pl.col("public_acq").any().alias("acq_public"),
        pl.col("established").any().alias("acq_established_any"),
        pl.col("conv_group").first().alias("acq_type"))
    cancels = pl.read_parquet(REPO / "data" / "release" / "proof_outcomes.parquet").with_columns(
        pl.col("serial_number").cast(pl.Utf8).str.strip_chars().alias("serial")).select("serial", "cancel_date")
    f = f.join(first, on="serial", how="left").join(cancels, on="serial", how="left").with_columns(
        (pl.col("acq_date").is_not_null() & ~pl.col("acq_distressed").fill_null(False)).alias("acquired"),
        (pl.col("acq_date").is_not_null() & pl.col("acq_distressed").fill_null(False)).alias("distressed"),
        (pl.col("acq_date").is_not_null() & ~pl.col("acq_distressed").fill_null(False)
         & pl.col("acq_public").fill_null(False)).alias("acq_public"),
        (pl.col("acq_date").is_not_null() & ~pl.col("acq_distressed").fill_null(False)
         & pl.col("acq_established_any").fill_null(False)).alias("acq_established"),
    ).with_columns(
        (pl.col("acquired") & pl.col("cancel_date").is_null()).alias("acq_kept"))

    fm = pl.read_parquet(PROC / "funding_owner_match.parquet",
                         columns=["owner_name", "first_formd_date", "in_sec", "in_fsds", "in_8a", "ipo_date"])
    f = f.join(fm, on="owner_name", how="left").with_columns(
        (pl.col("first_formd_date").cast(pl.Date, strict=False) >= pl.col("fdd")).fill_null(False).alias("funded"),
        ((pl.col("in_8a").fill_null(0) == 1) | pl.col("ipo_date").is_not_null()).alias("ipo"))

    # fifths within class x debut year x length fifth
    f = f.with_columns(((pl.col("glen").rank("ordinal").over(["cls", "fy"]) - 1) * 5
                        // pl.len().over(["cls", "fy"])).alias("lenq"))
    cell = ["cls", "fy", "lenq"]
    for v in ("L", "A"):
        f = f.sort(cell + [v, "serial"]).with_columns(
            ((pl.col(v).rank("ordinal").over(cell) - 1) * 5 // pl.len().over(cell)).cast(pl.Int8).alias("q" + v))
    outs = ["acquired", "acq_established", "acq_public", "acq_kept", "distressed", "funded", "ipo"]
    res = {"n": f.height, "overall": {o: float(f[o].mean()) for o in outs},
           "counts": {o: int(f[o].sum()) for o in outs}}
    for v in ("L", "A"):
        g = f.group_by("q" + v).agg([pl.col(o).mean().alias(o) for o in outs] + [pl.len().alias("n")]).sort("q" + v)
        res["by_" + v] = g.to_dicts()
        for r in res["by_" + v]:
            log(f"  {v} Q{r['q'+v]+1}: acquired {100*r['acquired']:.2f}%  established {100*r['acq_established']:.3f}%  public {100*r['acq_public']:.3f}%  "
                f"kept {100*r['acq_kept']:.2f}%  distressed {100*r['distressed']:.3f}%  ipo {100*r['ipo']:.3f}%")
    by_year = {r["fy"]: r for r in f.group_by("fy").agg([pl.col(o).mean().alias(o) for o in outs]).iter_rows(named=True)}
    res["by_debut_year"] = [by_year[y] for y in sorted(by_year)]
    res["vocab"] = {}
    for name in vocab:
        g = f.filter(pl.col(name))
        if g.height < 30:
            continue
        wt = {r["fy"]: r["len"] for r in g.group_by("fy").len().iter_rows(named=True)}
        tot = sum(wt.values())
        row = {"n": g.height}
        for o in outs:
            act = float(g[o].mean())
            exp = sum(by_year[y][o] * k for y, k in wt.items()) / tot
            row[o] = act
            row[o + "_ratio"] = act / exp if exp > 0 else None
            row[o + "_count"] = int(g[o].sum())
        res["vocab"][name] = row
        log(f"  {name:22s} n={g.height:6,} acquired x{row['acquired_ratio']:.2f} ({row['acquired_count']})  "
            f"established x{(row['acq_established_ratio'] or 0):.2f} ({row['acq_established_count']})  "
            f"public x{(row['acq_public_ratio'] or 0):.2f} ({row['acq_public_count']})  "
            f"funded x{row['funded_ratio']:.1f}  ipo x{(row['ipo_ratio'] or 0):.2f} ({row['ipo_count']})")
    log(f"[overall] {res['overall']}  counts {res['counts']}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "acquisition_rung.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
