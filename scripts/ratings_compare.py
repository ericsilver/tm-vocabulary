"""Blind expert-style ratings of the 60 segments against the record.

Reads every paper/ratings/ratings_*.json (one per rater), checks agreement
between raters, averages them, and relates the ratings to two outcomes for each
segment (from theme_segments.json): mean survival at the five-year declaration
(the attractiveness question) and the segment's lead effect, the survival
difference between its most leading and most lagging registrations (the
who-profits question). If data/processed/segment_scores.parquet exists, the
ratings are also correlated with the record-based scores.

Precision weights (1/se^2) are used for the lead effect; with 60 segments the
estimates are descriptive.

Output: paper/results/ratings_compare.json
"""
from __future__ import annotations

import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
ITEMS = ["entry_threat", "rivalry", "substitutes", "buyer_power", "supplier_power", "scale_economies",
         "imitability", "complementary_assets", "incumbents_hold_assets", "attractiveness", "pioneer_advantage"]


def wcorr(x, y, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sqrt(np.sum(w * (x - xm) ** 2) * np.sum(w * (y - ym) ** 2)))


def main() -> int:
    raters = {}
    for f in sorted(glob.glob(str(REPO / "paper" / "ratings" / "ratings_*.json"))):
        name = Path(f).stem.replace("ratings_", "")
        raters[name] = {r["segment"]: r for r in json.loads(Path(f).read_text(encoding="utf-8"))}
    seg = json.loads((RES / "theme_segments.json").read_text())
    outc = {f"S{s['seg']:02d}": s for s in seg["segments"] if s.get("lead_effect") is not None}
    ids = sorted(set.intersection(*[set(r) for r in raters.values()]) & set(outc))
    out = {"raters": list(raters), "n_segments": len(ids), "agreement": {}, "vs_outcomes": {}}
    for it in ITEMS:
        pairs = [spearmanr([raters[a][s][it] for s in ids], [raters[b][s][it] for s in ids]).statistic
                 for a, b in combinations(raters, 2)]
        out["agreement"][it] = float(np.mean(pairs)) if pairs else None
    surv = np.array([outc[s]["survival"] for s in ids])
    lead = np.array([outc[s]["lead_effect"] for s in ids])
    w = 1 / np.array([outc[s]["lead_effect_se"] for s in ids]) ** 2
    size = np.array([outc[s]["n"] for s in ids], float)
    for it in ITEMS:
        mean = np.array([np.mean([raters[r][s][it] for r in raters]) for s in ids])
        per = {r: {"survival": wcorr(np.array([raters[r][s][it] for s in ids], float), surv, size),
                   "lead_effect": wcorr(np.array([raters[r][s][it] for s in ids], float), lead, w)} for r in raters}
        out["vs_outcomes"][it] = {"mean_rating_vs_survival": wcorr(mean, surv, size),
                                  "mean_rating_vs_lead_effect": wcorr(mean, lead, w), "by_rater": per}
    ss = REPO / "data" / "processed" / "segment_scores.parquet"
    if ss.exists():
        import polars as pl
        sc = pl.read_parquet(ss)
        key = "seg" if "seg" in sc.columns else sc.columns[0]
        sc = {f"S{int(r[key]):02d}": r for r in sc.to_dicts()}
        num = [c for c, v in next(iter(sc.values())).items() if isinstance(v, (int, float)) and c != key]
        common = [s for s in ids if s in sc]
        out["ratings_vs_record_scores"] = {
            it: {c: float(spearmanr([np.mean([raters[r][s][it] for r in raters]) for s in common],
                                    [sc[s][c] for s in common]).statistic) for c in num}
            for it in ITEMS}
    (RES / "ratings_compare.json").write_text(json.dumps(out, indent=1))
    print(f"{len(raters)} raters, {len(ids)} segments")
    print(f"{'item':24s} {'agree':>6s} {'vs surv':>8s} {'vs lead':>8s}")
    for it in ITEMS:
        v = out["vs_outcomes"][it]
        print(f"{it:24s} {out['agreement'][it] if out['agreement'][it] is not None else float('nan'):6.2f} "
              f"{v['mean_rating_vs_survival']:+8.2f} {v['mean_rating_vs_lead_effect']:+8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
