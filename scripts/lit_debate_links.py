"""Do the first-mover literatures answer each other? Citation links from OpenAlex.

For the key works on each side, finds each work's OpenAlex record and checks
which of the others it cites (its referenced_works), plus how many later works
cite both a pro-first-mover and an anti-first-mover anchor. Also scans the
most-cited works using the phrase "first-mover advantage" for the industries
they study, so the paper's hypotheses can be instantiated where that
literature looked.

Output: paper/results/lit_debate_links.json
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "paper" / "results"
MAILTO = "eric.silver@aporal.com"
API = "https://api.openalex.org/works"
# OpenAlex now meters anonymous requests against a small daily budget shared by
# everyone on the network's IP address. A free personal key has its own budget:
# put OPENALEX_API_KEY=... in the repo's .env (gitignored).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
import os
KEY = os.environ.get("OPENALEX_API_KEY", "")

ANCHORS = {
    "LiebermanMontgomery1988": ("first-mover advantages", 1988),
    "LiebermanMontgomery1998": ("first-mover (dis)advantages: retrospective and link with the resource-based view", 1998),
    "GolderTellis1993": ("pioneer advantage: marketing logic or marketing legend", 1993),
    "SuarezLanzolla2007": ("role of environmental dynamics in building a first mover advantage theory", 2007),
    "Teece1986": ("profiting from technological innovation", 1986),
    "KimMauborgne2004": ("blue ocean strategy", 2004),
    "Klepper1997": ("industry life cycles", 1997),
    "RobinsonFornell1985": ("sources of market pioneer advantages in consumer goods industries", 1985),
    "SemadeniAnderson2010": ("follower's dilemma", 2010),
}

INDUSTRY_TERMS = {
    "internet / e-commerce": r"internet|e-commerce|online|dot-com|web\b|website",
    "software / platforms": r"software|platform|app\b|apps\b",
    "telecommunications / wireless": r"telecom|wireless|mobile|cellular",
    "consumer packaged goods": r"consumer goods|packaged goods|grocery|brand",
    "pharmaceuticals / drugs": r"pharmaceutic|drug|biotech",
    "electronics / hardware": r"electronic|semiconductor|computer|hardware|personal computer",
    "automobiles": r"automobile|automotive|\bcar\b|cars\b|vehicle",
    "airlines / transport": r"airline|aviation|transport",
    "retail": r"retail|store",
    "financial services / banking": r"bank|financial|fintech|payment",
    "network industries": r"network effect|network externalit|standards?\b|installed base",
    "emerging markets / entry abroad": r"emerging market|foreign market|international|china|country",
    "high-tech generally": r"high-tech|high technology|technolog",
    "services": r"service",
    "food and beverage": r"food|beverage|drink|restaurant",
}


def get(url: str) -> dict:
    if KEY:
        url += ("&" if "?" in url else "?") + f"api_key={KEY}"
    for attempt in range(8):
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"    retry {attempt+1}: {e}")
            time.sleep(10 * (attempt + 1))
    return {}


def find(title: str, year: int) -> dict | None:
    q = urllib.parse.quote(title)
    d = get(f"{API}?filter=title.search:{q},publication_year:{year}&per-page=5&sort=cited_by_count:desc&mailto={MAILTO}")
    res = d.get("results", [])
    return res[0] if res else None


def abstract(w: dict) -> str:
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    return " ".join(t for _, t in pos)


def main() -> int:
    recs = {}
    for k, (title, yr) in ANCHORS.items():
        w = find(title, yr)
        if w:
            recs[k] = {"id": w["id"], "title": w["title"], "year": w["publication_year"],
                       "cited_by": w.get("cited_by_count"), "refs": set(w.get("referenced_works") or [])}
            print(f"{k}: {w['title']} ({w['publication_year']}) cites={w.get('cited_by_count')} refs={len(recs[k]['refs'])}")
        else:
            print(f"{k}: NOT FOUND")
    links = {a: [b for b in recs if b != a and recs[b]["id"] in recs[a]["refs"]] for a in recs}
    for a, bs in links.items():
        print(f"  {a} cites: {bs}")
    # co-citation: later works citing both a pro and an anti anchor
    co = {}
    for pro in ("LiebermanMontgomery1988", "KimMauborgne2004"):
        for anti in ("GolderTellis1993", "Teece1986"):
            if pro in recs and anti in recs:
                a, b = recs[pro]["id"].split("/")[-1], recs[anti]["id"].split("/")[-1]
                d = get(f"{API}?filter=cites:{a},cites:{b}&per-page=1&mailto={MAILTO}")
                co[f"{pro}+{anti}"] = d.get("meta", {}).get("count")
                print(f"  works citing both {pro} and {anti}: {co[f'{pro}+{anti}']}")
    # first-mover-advantage literature: industries studied
    works, cursor = [], "*"
    while len(works) < 600 and cursor:
        d = get(f"{API}?search=%22first-mover%20advantage%22&filter=has_abstract:true&sort=cited_by_count:desc"
                f"&per-page=200&cursor={cursor}&mailto={MAILTO}")
        works += d.get("results", [])
        cursor = d.get("meta", {}).get("next_cursor")
    tally = Counter()
    examples = {k: [] for k in INDUSTRY_TERMS}
    for w in works:
        text = ((w.get("title") or "") + " " + abstract(w)).lower()
        for k, pat in INDUSTRY_TERMS.items():
            if re.search(pat, text):
                tally[k] += 1
                if len(examples[k]) < 5:
                    examples[k].append(f"{w['title']} ({w['publication_year']})")
    print(f"\nFMA literature scan: {len(works)} most-cited works with abstracts")
    for k, n in tally.most_common():
        print(f"  {k:32s} {n:4d}  ({100*n/len(works):.0f}%)")
    out = {"anchors": {k: {kk: vv for kk, vv in v.items() if kk != "refs"} for k, v in recs.items()},
           "cites": links, "co_citation_counts": co,
           "fma_scan": {"n_works": len(works), "tally": dict(tally), "examples": examples}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lit_debate_links.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
