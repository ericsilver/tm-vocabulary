"""Deposit the data-v1 release tables at Zenodo (draft only; publishing is done by hand).

Creates a draft deposition with metadata and a reserved DOI, checks each file's
SHA-256 against data/release/MANIFEST.txt, and uploads the files to the draft's
bucket. Re-running reuses the draft recorded in paper/results/zenodo_deposit.json
and skips files already uploaded. Reads ZENODO_TOKEN from .env.

Output: paper/results/zenodo_deposit.json (deposition id, reserved DOI, links)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
REL = REPO / "data" / "release"
STATE = REPO / "paper" / "results" / "zenodo_deposit.json"
API = "https://zenodo.org/api/deposit/depositions"

DESCRIPTION = """<p>Derived data for <em>An Event-Dated Corpus of US Trademark Prosecution and a Two-Sided
Measure of Vocabulary Position</em> (SSRN 7520758) and a companion paper on early entry, built from the
complete USPTO trademark record.</p>
<ul>
<li><code>proof_outcomes.parquet</code>: one row per registered serial: registration date, date of the first
Section 8 / Section 71 cancellation, registration age at that date, and whether it falls in the five-year-proof
window (4.0-8.5 years). Usable as a product-survival outcome without the text measure.</li>
<li><code>scores_T50.zip</code>: per-filing past- and future-facing divergence (K-, K+) and lead
(<code>topic_dkl</code>) under the production scoring (50 themes, per-filing 1,826-day windows, filings
1995-2019), one parquet per Nice class. Atypicality = (K- + K+)/2.</li>
<li><code>theme_model_T50.zip</code>: the fitted theme model (scikit-learn LDA, joblib) with its 62,168-term
vocabulary and metadata.</li>
<li><code>case_events.parquet</code>: 242 million dated prosecution events across 13.99 million USPTO case
files (serial, code, type, date, sequence).</li>
<li><code>case_extras.parquet</code>: counsel of record, statutory filing basis, post-registration declaration
flags, abandonment and status dates.</li>
<li><code>event_code_dict.parquet</code>: USPTO event codes with their modal descriptions.</li>
<li><code>owner_links.zip</code>: normalized-name links from USPTO owners to SEC registrants (CIK) and to
Form D issuers.</li>
<li><code>MANIFEST.txt</code>: file sizes and SHA-256 digests.</li>
</ul>
<p>Built from public USPTO, SEC and PatentsView sources by <code>scripts/build_release_tables.py</code> in
<a href="https://github.com/ericsilver/tm-vocabulary">github.com/ericsilver/tm-vocabulary</a>, where every
table rebuilds from source. The underlying government data are public domain; PatentsView-derived material
carries PatentsView's CC-BY-4.0 attribution requirement. An interactive online appendix is at
<a href="https://aporia.institute/tm-vocabulary/">aporia.institute/tm-vocabulary</a>.</p>"""

METADATA = {
    "title": "tm-vocabulary: Event-dated US trademark prosecution corpus, outcomes, theme model "
             "and lead/atypicality scores",
    "upload_type": "dataset",
    "description": DESCRIPTION,
    "creators": [{"name": "Silver, Eric", "orcid": "0000-0003-3351-1109",
                  "affiliation": "Independent researcher"}],
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": "data-v1",
    "keywords": ["trademarks", "USPTO", "innovation indicators", "text as data", "topic models",
                 "product survival", "administrative data"],
    "related_identifiers": [
        {"identifier": "10.2139/ssrn.7520758", "relation": "isSupplementTo", "scheme": "doi",
         "resource_type": "publication-preprint"},
        {"identifier": "https://github.com/ericsilver/tm-vocabulary", "relation": "isSupplementedBy",
         "scheme": "url", "resource_type": "software"},
    ],
    "prereserve_doi": True,
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    env = dict(l.split("=", 1) for l in (REPO / ".env").read_text().splitlines() if "=" in l)
    tok = os.environ.get("ZENODO_TOKEN") or env["ZENODO_TOKEN"].strip()
    H = {"Authorization": f"Bearer {tok}"}
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("id"):
        r = requests.get(f"{API}/{state['id']}", headers=H, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"reusing draft {dep['id']}", flush=True)
    else:
        r = requests.post(API, headers=H, json={}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"created draft {dep['id']}", flush=True)
    r = requests.put(f"{API}/{dep['id']}", headers=H, json={"metadata": METADATA}, timeout=60)
    if r.status_code >= 400:
        print(r.text, file=sys.stderr)
        r.raise_for_status()
    dep = r.json()
    doi = dep["metadata"].get("prereserve_doi", {}).get("doi")
    state = {"id": dep["id"], "doi": doi, "bucket": dep["links"]["bucket"],
             "html": dep["links"]["html"], "state": dep.get("state"), "submitted": dep.get("submitted")}
    STATE.write_text(json.dumps(state, indent=1))
    print(f"reserved DOI {doi}\ndraft page {state['html']}", flush=True)

    manifest = {}
    for line in (REL / "MANIFEST.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            manifest[parts[0]] = (int(parts[1]), parts[2])
    have = {f["filename"]: f["filesize"] for f in requests.get(f"{API}/{dep['id']}/files", headers=H,
                                                                timeout=60).json()}
    names = sorted(manifest, key=lambda n: manifest[n][0]) + ["MANIFEST.txt"]
    for name in names:
        p = REL / name
        if name in manifest:
            size, digest = manifest[name]
            if p.stat().st_size != size or sha256(p) != digest:
                raise SystemExit(f"{name} does not match MANIFEST.txt; not uploading")
        if have.get(name) == p.stat().st_size:
            print(f"  {name}: already uploaded", flush=True)
            continue
        with open(p, "rb") as f:
            r = requests.put(f"{state['bucket']}/{name}", headers=H, data=f, timeout=3600)
        r.raise_for_status()
        print(f"  {name}: uploaded ({p.stat().st_size/1e6:.1f} MB)", flush=True)
    print("[done] draft ready for review; not published", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
