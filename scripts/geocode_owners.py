"""Latitude and longitude for US owners at filing.

Owner postcodes come from extract_owner_geo.py. Each US filing is placed at
the centroid of its ZIP Code Tabulation Area (Census 2020 gazetteer). ZIPs
that are not ZCTAs (post-office boxes, single-firm ZIPs) fall back to the
mean centroid of the ZCTAs sharing their first three digits; filings with no
usable postcode fall back to the city's Census place centroid.

Output: data/processed/owner_ll.parquet
  serial_number, lat, lon, geo_src ('zcta' | 'zip3' | 'place'),
  owner_city, owner_state, zip5
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
GEO = REPO / "data" / "raw" / "geo"
STATES = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
          "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
          "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
          "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"}


def gaz(name: str) -> pl.DataFrame:
    d = pl.read_csv(GEO / name, separator="\t", infer_schema_length=0)
    d.columns = [c.strip() for c in d.columns]
    return d


def norm_city(e: pl.Expr) -> pl.Expr:
    return (e.fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z ]", " ")
            .str.replace_all(r"\b(SAINT)\b", "ST").str.replace_all(r"\s+", " ").str.strip_chars())


def main() -> int:
    z = gaz("2020_Gaz_zcta_national.txt").select(
        pl.col("GEOID").alias("zip5"),
        pl.col("INTPTLAT").str.strip_chars().cast(pl.Float64).alias("lat"),
        pl.col("INTPTLONG").str.strip_chars().cast(pl.Float64).alias("lon"))
    z3 = z.with_columns(pl.col("zip5").str.slice(0, 3).alias("zip3")).group_by("zip3").agg(
        pl.col("lat").mean(), pl.col("lon").mean())
    pl_ = gaz("2020_Gaz_place_national.txt").select(
        pl.col("USPS").alias("owner_state"),
        norm_city(pl.col("NAME").str.replace(
            r"\s+(city|town|village|CDP|borough|municipality|city and borough|"
            r"consolidated government|metropolitan government|unified government)(\s.*)?$", ""))
        .alias("city_n"),
        pl.col("INTPTLAT").str.strip_chars().cast(pl.Float64).alias("lat"),
        pl.col("INTPTLONG").str.strip_chars().cast(pl.Float64).alias("lon"),
        pl.col("ALAND").cast(pl.Float64).alias("aland"),
    ).sort("aland", descending=True).unique(["owner_state", "city_n"], keep="first").drop("aland")

    d = pl.read_parquet(PROC / "owner_geo.parquet").filter(
        pl.col("owner_state").is_in(list(STATES))
        & pl.col("owner_country").is_in(["US", ""])).select(
        "serial_number", "owner_city", "owner_state", "zip5").with_columns(
        pl.col("zip5").str.slice(0, 3).alias("zip3"),
        norm_city(pl.col("owner_city").str.split(",").list.first()).alias("city_n"))
    a = d.join(z, on="zip5", how="left").with_columns(
        pl.when(pl.col("lat").is_not_null()).then(pl.lit("zcta")).alias("geo_src"))
    a = a.join(z3.rename({"lat": "lat3", "lon": "lon3"}), on="zip3", how="left").with_columns(
        pl.when(pl.col("lat").is_null() & (pl.col("zip5") != "") & pl.col("lat3").is_not_null())
        .then(pl.lit("zip3")).otherwise(pl.col("geo_src")).alias("geo_src"),
        pl.coalesce(pl.col("lat"), pl.when(pl.col("zip5") != "").then(pl.col("lat3"))).alias("lat"),
        pl.coalesce(pl.col("lon"), pl.when(pl.col("zip5") != "").then(pl.col("lon3"))).alias("lon"),
    ).drop("lat3", "lon3")
    a = a.join(pl_.rename({"lat": "latp", "lon": "lonp"}), on=["owner_state", "city_n"],
               how="left").with_columns(
        pl.when(pl.col("lat").is_null() & pl.col("latp").is_not_null())
        .then(pl.lit("place")).otherwise(pl.col("geo_src")).alias("geo_src"),
        pl.coalesce("lat", "latp").alias("lat"), pl.coalesce("lon", "lonp").alias("lon"),
    ).drop("latp", "lonp", "zip3", "city_n").drop_nulls("lat")
    a.write_parquet(PROC / "owner_ll.parquet")
    print(f"US owners {d.height:,}; placed {a.height:,} ({a.height/d.height:.1%})")
    print(a.group_by("geo_src").len())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
