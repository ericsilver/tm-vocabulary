"""Do the first-mover literatures answer each other? Crossref version.

OpenAlex and Semantic Scholar were rate-limiting this connection, so this
uses Crossref: each anchor work's deposited reference list (where the
publisher deposits one) shows which of the other anchors it cites, and a
title search for "first-mover advantage" / "pioneer advantage" gives the
industries that literature studies (titles only; Crossref rarely carries
abstracts).

Output: paper/results/lit_debate_crossref.json
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
UA = {"User-Agent": "tm-vocabulary research (mailto:eric.silver@aporal.com)"}

ANCHORS = {
    "RobinsonFornell1985": ("Sources of market pioneer advantages in consumer goods industries", "Robinson", 1985),
    "LiebermanMontgomery1988": ("First-mover advantages", "Lieberman", 1988),
    "GolderTellis1993": ("Pioneer advantage: marketing logic or marketing legend?", "Golder", 1993),
    "LiebermanMontgomery1998": ("First-mover (dis)advantages: retrospective and link with the resource-based view", "Lieberman", 1998),
    "KimMauborgne2004": ("Blue ocean strategy", "Kim", 2004),
    "SuarezLanzolla2007": ("The role of environmental dynamics in building a first mover advantage theory", "Suarez", 2007),
    "Teece1986": ("Profiting from technological innovation", "Teece", 1986),
    "Klepper1997": ("Industry life cycles", "Klepper", 1997),
    "SemadeniAnderson2010": ("The follower's dilemma: innovation and imitation in the professional services industry", "Semadeni", 2010),
}

INDUSTRY_TERMS = {
    "internet / e-commerce": r"internet|e-commerce|online|dot-com|\bweb\b|website|e-business",
    "software / platforms": r"software|platform|\bapps?\b",
    "telecommunications / wireless": r"telecom|wireless|mobile|cellular",
    "consumer goods / brands": r"consumer goods|packaged goods|grocery|\bbrands?\b",
    "pharmaceuticals / biotech": r"pharmaceutic|\bdrugs?\b|biotech",
    "electronics / computers": r"electronic|semiconductor|computer|hardware",
    "automobiles": r"automobile|automotive|\bcars?\b",
    "airlines / transport": r"airline|aviation|shipping",
    "retail": r"retail",
    "financial services": r"\bbank|financial|payment",
    "network effects / standards": r"network effect|network externalit|standard|installed base",
    "international entry / emerging markets": r"emerging market|foreign|international|china|transition econom",
    "services": r"service",
    "food and beverage": r"\bfood|beverage|drink|restaurant",
}


def get(url: str) -> dict:
    for attempt in range(6):
        time.sleep(0.5)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"    retry {attempt+1}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return {}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def main() -> int:
    recs = {}
    for k, (title, author, yr) in ANCHORS.items():
        q = urllib.parse.quote(title)
        d = get(f"https://api.crossref.org/works?query.bibliographic={q}&query.author={author}"
                f"&filter=from-pub-date:{yr-1},until-pub-date:{yr+1}&rows=5")
        items = d.get("message", {}).get("items", [])
        best = next((w for w in items if norm(title)[:25] in norm((w.get("title") or [""])[0])), items[0] if items else None)
        if not best:
            print(f"{k}: not found"); continue
        refs = best.get("reference") or []
        ref_text = [norm(" ".join(str(v) for v in r.values())) for r in refs]
        recs[k] = {"doi": best.get("DOI"), "title": (best.get("title") or [""])[0],
                   "n_refs_deposited": len(refs), "ref_text": ref_text}
        print(f"{k}: {recs[k]['title'][:70]} | doi {recs[k]['doi']} | refs deposited {len(refs)}", flush=True)
    cites = {}
    for a, ra in recs.items():
        hit = []
        for b, (title, author, yr) in ANCHORS.items():
            if b == a or b not in recs:
                continue
            key_t = norm(title)[:30]
            key_a = author.lower()
            if any((key_t and key_t in t) or (key_a in t and str(yr) in t) for t in ra["ref_text"]):
                hit.append(b)
        cites[a] = hit
        print(f"  {a} cites: {hit}" + ("" if ra["n_refs_deposited"] else "  (no reference list deposited)"))
    # literature scan by title
    titles = []
    for phrase in ("first-mover advantage", "first mover advantage", "pioneer advantage", "first-mover disadvantage", "late mover advantage"):
        cursor = "*"
        while cursor and len(titles) < 1500:
            d = get(f"https://api.crossref.org/works?query.title={urllib.parse.quote(phrase)}&rows=1000"
                    f"&select=title,DOI,is-referenced-by-count,issued&cursor={urllib.parse.quote(cursor)}")
            m = d.get("message", {})
            items = m.get("items", [])
            if not items:
                break
            for w in items:
                t = (w.get("title") or [""])[0]
                if re.search(r"first[- ]mover|pioneer|late[- ]mover|early entr", t, re.I):
                    titles.append((t, w.get("is-referenced-by-count", 0), w.get("DOI")))
            cursor = None  # first page of 1000 per phrase is enough for a title tally
    seen, uniq = set(), []
    for t in titles:
        if t[2] not in seen:
            seen.add(t[2]); uniq.append(t)
    tally, ex = Counter(), {k: [] for k in INDUSTRY_TERMS}
    for t, c, _ in sorted(uniq, key=lambda x: -x[1]):
        for k, pat in INDUSTRY_TERMS.items():
            if re.search(pat, t, re.I):
                tally[k] += 1
                if len(ex[k]) < 6:
                    ex[k].append(f"{t} [{c} cites]")
    print(f"\nTitle scan: {len(uniq)} works with first-mover / pioneer terms in the title")
    for k, n in tally.most_common():
        print(f"  {k:40s} {n:4d}")
    out = {"anchors": {k: {kk: vv for kk, vv in v.items() if kk != "ref_text"} for k, v in recs.items()},
           "cites": cites, "title_scan": {"n": len(uniq), "tally": dict(tally), "examples": ex}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lit_debate_crossref.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
