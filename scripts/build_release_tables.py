"""Assemble the public data release that the papers' Access sections describe.

The repository holds code only; the derived tables are too large for git. This
writes the release assets into data/release/:

  proof_outcomes.parquet   one row per registered serial: registration date,
                           the date of its first Section 8 / Section 71
                           cancellation, registration age at that date, and
                           whether it falls in the five-year-proof window
                           (4.0-8.5 years) -- the event-dated outcome the papers
                           use, directly reusable without the text measure
  scores_T50.zip           per-filing K-, K+, lead and atypicality under the
                           production scoring (50 themes, per-filing windows),
                           one parquet per Nice class
  theme_model_T50.zip      the fitted theme model, its vocabulary, and meta
  case_events.parquet      242 million dated prosecution events
  case_extras.parquet      counsel of record, filing basis, declaration flags
  event_code_dict.parquet  USPTO event codes with modal descriptions
  owner_links.zip          USPTO owner -> SEC CIK crosswalk and Form D matches
  MANIFEST.txt             file list with sizes and SHA-256 digests

Usage: python scripts/build_release_tables.py
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "data" / "release"
GATE_LO, GATE_HI = 4.0, 8.5


def proof_outcomes() -> None:
    regs = []
    for p in sorted(PROC.glob("tm_class*.parquet")):
        if not p.stem.replace("tm_class", "").isdigit():
            continue
        regs.append(pl.read_parquet(p, columns=["serial_number", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8))
    r = pl.concat(regs).unique("serial_number").with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("registration_date")).drop_nulls("registration_date")
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("cancel_date")).drop_nulls("cancel_date").group_by("serial_number").agg(
        pl.col("cancel_date").min())
    out = r.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cancel_date") - pl.col("registration_date")).dt.total_days() / 365.25)
        .alias("cancel_age_years")).with_columns(
        ((pl.col("cancel_age_years") >= GATE_LO) & (pl.col("cancel_age_years") < GATE_HI))
        .fill_null(False).alias("failed_proof_window"))
    out.write_parquet(OUT / "proof_outcomes.parquet")
    print(f"proof_outcomes: {out.height:,} registrations, "
          f"{int(out['failed_proof_window'].sum()):,} in-window cancellations")


def zip_files(name: str, files: list[Path]) -> None:
    with zipfile.ZipFile(OUT / name, "w", compression=zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    print(f"{name}: {len(files)} files")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    proof_outcomes()
    zip_files("scores_T50.zip", sorted(PROC.glob("rolling_surprise_class[0-9][0-9][0-9].parquet")))
    zip_files("theme_model_T50.zip", [PROC / "topic_model.joblib", PROC / "topic_lda_meta.json"])
    zip_files("owner_links.zip", [PROC / "uspto_sec_crosswalk.parquet",
                                  PROC / "funding_owner_match.parquet"])
    for f in ("case_events.parquet", "case_extras.parquet", "event_code_dict.parquet"):
        shutil.copy2(PROC / f, OUT / f)
    lines = []
    for f in sorted(OUT.iterdir()):
        if f.name == "MANIFEST.txt":
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{f.name}\t{f.stat().st_size}\t{h.hexdigest()}")
    (OUT / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
