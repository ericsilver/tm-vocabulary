"""Blind packet describing the 60 product segments, for expert-style ratings.

Each segment is described only by what its registrations sell: the themes that
make up its centre (with their shares and leading words) and a random sample of
goods/services descriptions from its registrations. Nothing about outcomes
(survival, lead, funding, listing), sizes over time, or dates is included, so a
rater cannot score with hindsight from the record.

Output: paper/ratings/segment_packet.json, paper/ratings/segment_packet.md
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import joblib
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "paper" / "ratings"
N_EX = 10


def main() -> int:
    src = (REPO / "scripts" / "themes_t50_page.py").read_text(encoding="utf-8")
    m = re.search(r"LABELS\s*=\s*(\{.*?\n\})", src, re.S) or re.search(r"LABELS\s*=\s*(\[.*?\n\])", src, re.S)
    L = ast.literal_eval(m.group(1))
    model = joblib.load(PROC / "topic_model.joblib")
    vocab = np.array(sorted(model["vocabulary"], key=model["vocabulary"].get))
    words = {k: ", ".join(vocab[np.argsort(-model["lda"].components_[k])[:8]]) for k in range(50)}

    seg = pl.read_parquet(PROC / "theme_segments.parquet")
    th = pl.read_parquet(PROC / "theme_mix_theta.parquet")
    d = seg.join(th, on=["serial_number", "cls"])
    tcols = [f"t{k}" for k in range(50)]
    cent = d.group_by("seg").agg([pl.col(c).mean() for c in tcols]).sort("seg")
    rng = np.random.default_rng(7)
    texts = {}
    for (s,), g in seg.group_by(["seg"]):
        pick = g.sample(min(400, g.height), seed=int(s)).group_by("cls").head(200)
        parts = []
        for (c,), gg in pick.group_by(["cls"]):
            t = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "goods_services"]).join(
                gg.select("serial_number"), on="serial_number", how="inner")
            parts.append(t)
        tt = pl.concat(parts).filter(pl.col("goods_services").fill_null("").str.len_chars() > 30)
        tt = tt.sample(min(N_EX, tt.height), seed=int(s))
        texts[int(s)] = [x[:280] + ("..." if len(x) > 280 else "") for x in tt["goods_services"].to_list()]
    segs = []
    for r in cent.iter_rows(named=True):
        w = np.array([r[c] for c in tcols])
        o = np.argsort(-w)[:5]
        segs.append({"segment": f"S{int(r['seg']):02d}",
                     "themes": [{"theme": L[int(k)], "share": round(float(w[k]), 3), "leading_words": words[int(k)]}
                                for k in o if w[k] >= 0.03],
                     "example_descriptions": texts.get(int(r["seg"]), [])})
    rng.shuffle(segs)   # order carries no information
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "segment_packet.json").write_text(json.dumps(segs, indent=1))
    md = ["# Product segments (blind packet)\n"]
    for s in segs:
        md.append(f"## {s['segment']}\n")
        md.append("Themes: " + "; ".join(f"{t['theme']} ({t['share']:.0%}; words: {t['leading_words']})"
                                       for t in s["themes"]) + "\n")
        md.append("Example goods/services descriptions:\n" + "\n".join(f"- {x}" for x in s["example_descriptions"]) + "\n")
    (OUT / "segment_packet.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {len(segs)} segments to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
