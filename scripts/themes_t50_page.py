"""The online appendix's page for the production theme model (T = 50).

Both papers point readers here for "every theme with its leading words and
its weight in each class". This builds that page: all 50 themes, a short
plain-language label for each, its leading words, its share of the whole
corpus, the classes where it is heaviest, and -- through a class selector --
its share of any one of the 45 Nice classes. Shares are the average theme mix
of up to SAMPLE_PER_CLASS scored filings per class (filing years 1995-2019),
computed with the production model and the production scoring tokenizer.
The page opens with a plain-language explanation of what a theme is, how the
themes were found, and why there are fifty.

Output: docs/online-appendix/themes_T50.csv   (theme x class shares)

The page (themes_T50.html) is written by scripts/themes_page.py, which adds
each theme's use over time, lead and outcomes and the 200- and 500-theme
models from scripts/build_theme_resolution_data.py. This script supplies the
labels and the T = 50 shares both of those read.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "docs" / "online-appendix"
SAMPLE_PER_CLASS = 20_000
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"

LABELS = {
    0: "Electronic apparatus and control systems", 1: "Accessories and cables",
    2: "Design and technical development services", 3: "Health and wellness services",
    4: "Plants, seeds and agriculture", 5: "Downloadable and online software",
    6: "Metal and building hardware", 7: "Pharmaceutical preparations",
    8: "Processed foods", 9: "Computer, software and information services",
    10: "Cleaning preparations and coatings", 11: "Components sold as a unit (drafting formula)",
    12: "Retail store services", 13: "Research, scientific and diagnostic",
    14: "Clothing, bags and footwear", 15: "Building and construction materials",
    16: "Nutritional supplements", 17: "Toys, games and sporting goods",
    18: "Media and entertainment goods", 19: "Oil, gas and metals",
    20: "Pets and animal feed", 21: "Bicycles, vehicle rental and tobacco (mixed)",
    22: "Art, home and interior decoration", 23: "Musical and surgical instruments",
    24: "Hair and skin care", 25: "Essential oils and candles",
    26: "Automotive vehicles and lubricants", 27: "Precious metal and household glassware",
    28: "Energy, transport and storage services", 29: "Marketing of consumer goods",
    30: "Electronic cigarettes and tobacco", 31: "Industrial chemicals",
    32: "Plastics, textiles and packaging materials", 33: "Paper, books and stationery",
    34: "Furniture and lighting", 35: "Education, training, travel and events",
    36: "Legal services, weapons and printer supplies (mixed)", 37: "Land vehicles and parts",
    38: "Medical and healthcare services", 39: "Heating and air apparatus",
    40: "Industrial machines and parts", 41: "Repair, maintenance and rental of machines",
    42: "Construction, engineering and waste services",
    43: "Medical-purpose apparatus and textile yarn (mixed)", 44: "Beverages and dairy",
    45: "Water treatment", 46: "Mats, floor coverings and printing",
    47: "Hand tools and knives", 48: "Veterinary and medical preparations",
    49: "Dental, industrial gases and fundraising (mixed)",
}

def _nice_names() -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("oa", REPO / "scripts" / "online_appendix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NICE_NAMES


CLASS_NAMES = _nice_names()


def main() -> int:
    m = joblib.load(PROC / "topic_model.joblib")
    lda = m["lda"]
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    words = json.loads((PROC / "topic_lda_meta.json").read_text())["top_words"]
    rng = np.random.default_rng(0)
    shares, sizes = {}, {}
    for p in sorted(PROC.glob("tm_class*.parquet")):
        c = p.stem.replace("tm_class", "")
        if not c.isdigit():
            continue
        d = pl.read_parquet(p, columns=["filing_date", "goods_services"]).filter(
            pl.col("goods_services").is_not_null()
            & (pl.col("goods_services").str.len_chars() > 0)
            & pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False)
            .is_between(1995, 2019))
        sizes[c] = d.height
        g = d["goods_services"]
        idx = np.sort(rng.choice(len(g), size=min(SAMPLE_PER_CLASS, len(g)), replace=False))
        th = lda.transform(vec.transform(g[idx].to_list()))
        th /= th.sum(axis=1, keepdims=True)
        shares[c] = th.mean(axis=0)
        print(c, d.height, flush=True)
    classes = sorted(shares)
    S = np.vstack([shares[c] for c in classes])            # classes x themes
    w = np.array([sizes[c] for c in classes], float)
    corpus = (S * w[:, None]).sum(axis=0) / w.sum()

    rows = []
    for k in range(S.shape[1]):
        top = np.argsort(S[:, k])[::-1][:3]
        rows.append({"k": k, "label": LABELS[k], "words": words[str(k)][:8],
                     "corpus": float(corpus[k]),
                     "top": [[classes[i], CLASS_NAMES.get(classes[i], ""), float(S[i, k])] for i in top],
                     "by_class": {classes[i]: float(S[i, k]) for i in range(len(classes))}})
    rows.sort(key=lambda r: -r["corpus"])

    # CSV
    lines = ["theme,label,leading_words,corpus_share," + ",".join(f"class_{c}" for c in classes)]
    for r in sorted(rows, key=lambda r: r["k"]):
        lines.append(f'{r["k"]},"{r["label"]}","{" ".join(r["words"])}",{r["corpus"]:.5f},'
                     + ",".join(f'{r["by_class"][c]:.5f}' for c in classes))
    (OUT / "themes_T50.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("wrote themes_T50.csv")
    # The page itself is written by themes_page.py from themes_data/, which
    # build_theme_resolution_data.py fills (it reads the CSV written above for
    # the T = 50 shares). Re-run that script after this one, then themes_page.py.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
