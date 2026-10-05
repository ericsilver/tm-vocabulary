"""Case study: powder coatings in the trademark record, against other coatings.

Groups (regex on the goods/services description, classes 1, 2, 40):
  powder        powder coating(s) / powder paint(s)
  architectural paint ... interior|exterior|house|wall|ceiling|deck|primer for buildings
  industrial    industrial coating(s) / coatings for metal|industrial use (not powder)
  automotive    automotive|vehicle refinish / automobile paint (not powder)
For each group and filing period: filings, distinct owners, share of filings by the
top five owners, marks per owner, first-time owners, foreign owners, counsel,
business-to-business wording, and continued use at five years (registrations
2002-2018). For owners matched to SEC filers: their mean gross margin, 2009-2023.

Output: paper/results/powder_case.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
sys.path.insert(0, str(REPO / "scripts"))
from gate_decisive_regression import norm_owner  # noqa: E402

GROUPS = {
    "powder": r"powder[- ]coat|powder paint|coating powder|powdered coating",
    "architectural": r"(interior|exterior|house|wall|ceiling|deck|masonry|drywall) (paint|primer|stain|coating)|paints? for (interior|exterior|buildings|walls)|architectural (paint|coating)",
    "industrial": r"industrial (coating|paint|finish)|coatings? for (metal|industrial use|use in industry)|protective coatings?",
    "automotive": r"(automotive|vehicle|automobile|car) (refinish|paint|coating)|refinish(ing)? paint",
}
B2B = r"for industrial|for use in (the )?manufactur|industrial use|for use by|for original equipment|oem|for application to metal"
PERIODS = [(1985, 1994), (1995, 2004), (2005, 2014), (2015, 2024)]


def main() -> int:
    parts = []
    for c in ("001", "002", "040"):
        parts.append(pl.read_parquet(PROC / f"tm_class{c}.parquet",
                                     columns=["serial_number", "filing_date", "registration_date", "owner_name",
                                              "goods_services"]).with_columns(pl.lit(c).alias("cls")))
    d = pl.concat(parts).unique("serial_number").with_columns(
        pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        norm_owner(pl.col("owner_name")).alias("okey"))
    d = d.with_columns([pl.col("gs").str.contains(p).alias(g) for g, p in GROUPS.items()])
    d = d.with_columns(pl.when(pl.col("powder")).then(False).otherwise(pl.col("industrial")).alias("industrial"),
                       pl.when(pl.col("powder")).then(False).otherwise(pl.col("automotive")).alias("automotive"))
    # owner history (first filing anywhere in these classes) and extras
    first = d.group_by("okey").agg(pl.col("fy").min().alias("first_fy"))
    d = d.join(first, on="okey", how="left")
    ex = pl.read_parquet(PROC / "case_extras.parquet", columns=["serial_number", "attorney_name"])
    geo = pl.read_parquet(PROC / "owner_geo.parquet", columns=["serial_number", "owner_country"])
    po = pl.read_parquet(REPO / "data" / "release" / "proof_outcomes.parquet",
                         columns=["serial_number", "registration_date", "failed_proof_window"])
    d = d.join(ex, on="serial_number", how="left").join(geo, on="serial_number", how="left").join(
        po.select("serial_number", "failed_proof_window"), on="serial_number", how="left").with_columns(
        (pl.col("attorney_name").fill_null("") != "").alias("counsel"),
        (~pl.col("owner_country").fill_null("US").is_in(["US", ""])).alias("foreign"),
        pl.col("gs").str.contains(B2B).alias("b2b"),
        pl.col("registration_date").fill_null("").str.slice(0, 4).cast(pl.Int32, strict=False).alias("ry"))
    out = {"groups": {}}
    for g in GROUPS:
        sub = d.filter(pl.col(g))
        rows = []
        for lo, hi in PERIODS:
            s = sub.filter(pl.col("fy").is_between(lo, hi))
            if s.height == 0:
                continue
            by_owner = s.group_by("okey").len().sort("len", descending=True)
            reg = s.filter(pl.col("ry").is_between(2002, 2018) & pl.col("failed_proof_window").is_not_null())
            rows.append({
                "period": f"{lo}-{hi}", "filings": s.height, "owners": by_owner.height,
                "top5_share": float(by_owner.head(5)["len"].sum() / s.height),
                "marks_per_owner": s.height / by_owner.height,
                "first_time_owner_share": float((s["fy"] == s["first_fy"]).mean()),
                "foreign_share": float(s["foreign"].mean()), "counsel_share": float(s["counsel"].mean()),
                "b2b_share": float(s["b2b"].mean()),
                "in_use_at_5y": (float(1 - reg["failed_proof_window"].cast(pl.Float64).mean()) if reg.height else None),
                "n_registered_2002_2018": reg.height,
                "top_owners": [{"owner": o, "filings": int(n)} for o, n in by_owner.head(6).iter_rows()]})
        out["groups"][g] = rows
    # margins of SEC-matched owners, by group (owner counted once per group)
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet", columns=["owner_name", "cik"]).with_columns(
        norm_owner(pl.col("owner_name")).alias("okey")).unique("okey")
    fin = pl.read_parquet(PROC / "sec_firm_year.parquet", columns=["cik", "name", "sic", "fy", "revenue",
                                                                    "gross_margin"]).filter(
        pl.col("gross_margin").is_between(-1, 1) & (pl.col("revenue") > 0))
    firm = fin.group_by("cik").agg(pl.col("name").last(), pl.col("sic").last(),
                                   pl.col("gross_margin").median().alias("gm"),
                                   pl.col("revenue").median().alias("rev"))
    out["margins"] = {}
    for g in GROUPS:
        ow = d.filter(pl.col(g) & (pl.col("fy") >= 2000)).select("okey").unique().join(cw, on="okey").join(
            firm, on="cik", how="inner").unique("cik")
        out["margins"][g] = {"n_firms": ow.height,
                             "median_gross_margin": float(ow["gm"].median()) if ow.height else None,
                             "firms": [{"name": n, "sic": s, "gross_margin": float(m)}
                                       for n, s, m in ow.sort("rev", descending=True).head(10).select(
                                           "name", "sic", "gm").iter_rows()]}
    (REPO / "paper" / "results" / "powder_case.json").write_text(json.dumps(out, indent=1, default=str))
    for g, rows in out["groups"].items():
        print(f"\n== {g}")
        for r in rows:
            u = f"{100 * r['in_use_at_5y']:.0f}%" if r["in_use_at_5y"] is not None else "--"
            print(f"  {r['period']}: filings {r['filings']:5d} owners {r['owners']:4d} top5 {r['top5_share']:.0%} "
                  f"marks/owner {r['marks_per_owner']:.1f} first-time {r['first_time_owner_share']:.0%} "
                  f"foreign {r['foreign_share']:.0%} counsel {r['counsel_share']:.0%} b2b {r['b2b_share']:.0%} "
                  f"in use@5y {u} (n={r['n_registered_2002_2018']})")
        print("   top owners 2005-14:", [o['owner'] for o in next((r for r in rows if r['period'] == '2005-2014'),
                                                                   rows[-1])['top_owners']])
        m = out["margins"][g]
        print(f"   SEC-matched firms {m['n_firms']}, median gross margin "
              f"{m['median_gross_margin'] if m['median_gross_margin'] is None else round(m['median_gross_margin'], 3)}:",
              [(f['name'][:22], round(f['gross_margin'], 2)) for f in m['firms'][:6]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
