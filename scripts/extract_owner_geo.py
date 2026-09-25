"""Owner location at filing from the USPTO trademark annual archives.

The earlier extract (extract_owner_addresses.py) kept state and country only.
This pass keeps what a spatial measure needs: city, state, postcode and
country of the applicant as filed (party-type 10 when present, otherwise the
lowest party-type, i.e. the earliest owner role recorded).

Reads the local archives in data/raw (no download), one worker per part.

Output: data/processed/owner_geo.parquet
  serial_number, filing_date, party_type, owner_city, owner_state,
  owner_postcode (as filed), zip5 (US 5-digit, '' otherwise), owner_country
"""
from __future__ import annotations

import gzip
import io
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
SHARDS = REPO / "data" / "processed" / "owner_geo_shards"
OUT = REPO / "data" / "processed" / "owner_geo.parquet"

SCHEMA = pa.schema([(c, pa.string()) for c in (
    "serial_number", "filing_date", "party_type", "owner_city", "owner_state",
    "owner_postcode", "zip5", "owner_country")])
ZIP5 = re.compile(r"^\s*(\d{5})(?:[-\s]?\d{4})?\s*$")


def _t(e, tag):
    n = e.find(tag)
    return (n.text or "").strip() if n is not None and n.text else ""


def one_part(zpath: str) -> str:
    zpath = Path(zpath)
    shard = SHARDS / f"{zpath.stem}.parquet"
    if shard.exists():
        return f"{zpath.name}: exists"
    rows = {c: [] for c in SCHEMA.names}
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            if not (member.endswith(".xml") or member.endswith(".xml.gz")):
                continue
            with zf.open(member) as fh:
                if member.endswith(".gz"):
                    fh = gzip.GzipFile(fileobj=io.BufferedReader(fh))
                for _, el in etree.iterparse(fh, events=("end",), tag="case-file", recover=True):
                    serial = _t(el, "serial-number")
                    hdr = el.find("case-file-header")
                    fdate = _t(hdr, "filing-date") if hdr is not None else ""
                    best, best_pt = None, 10**6
                    owners = el.find("case-file-owners")
                    if owners is not None:
                        for o in owners.findall("case-file-owner"):
                            try:
                                pt = int(_t(o, "party-type") or 999)
                            except ValueError:
                                pt = 999
                            if pt < best_pt:
                                best, best_pt = o, pt
                    if serial:
                        pc = _t(best, "postcode") if best is not None else ""
                        ctry = _t(best, "country") if best is not None else ""
                        st = _t(best, "state") if best is not None else ""
                        m = ZIP5.match(pc)
                        us = (ctry in ("", "US")) and bool(st)
                        rows["serial_number"].append(serial)
                        rows["filing_date"].append(fdate)
                        rows["party_type"].append("" if best is None else str(best_pt))
                        rows["owner_city"].append(_t(best, "city") if best is not None else "")
                        rows["owner_state"].append(st)
                        rows["owner_postcode"].append(pc)
                        rows["zip5"].append(m.group(1) if (m and us) else "")
                        rows["owner_country"].append(ctry or ("US" if st else ""))
                    el.clear()
                    while el.getprevious() is not None:
                        del el.getparent()[0]
    tmp = shard.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=SCHEMA), tmp, compression="zstd")
    tmp.replace(shard)
    return f"{zpath.name}: {len(rows['serial_number']):,}"


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    parts = sorted(RAW.glob("apc*.zip"))
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(one_part, str(p)) for p in parts]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(parts)}] {f.result()}", flush=True)
    with pq.ParquetWriter(OUT, SCHEMA, compression="zstd") as w:
        for s in sorted(SHARDS.glob("*.parquet")):
            w.write_table(pq.read_table(s))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
