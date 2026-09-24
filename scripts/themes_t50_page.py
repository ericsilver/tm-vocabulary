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

Output: docs/online-appendix/themes_T50.html
        docs/online-appendix/themes_T50.csv   (theme x class shares)
"""
from __future__ import annotations

import html
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

    data = json.dumps({"rows": rows, "classes": [[c, CLASS_NAMES.get(c, "")] for c in classes]})
    opts = "".join(f'<option value="{c}">{c} {html.escape(CLASS_NAMES.get(c, ""))}</option>' for c in classes)
    page = PAGE.replace("__DATA__", data).replace("__OPTS__", opts)
    (OUT / "themes_T50.html").write_text(page, encoding="utf-8")
    print("wrote themes_T50.html and themes_T50.csv")
    return 0


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The fifty themes</title><style>
:root{--bg:#fff;--fg:#1a202c;--mut:#5a6a7a;--line:#e2e8f0;--acc:#2b6cb0;--bar:#bcd3ee}
@media(prefers-color-scheme:dark){:root{--bg:#14181d;--fg:#e6edf3;--mut:#9aa8b6;--line:#2b333c;--acc:#6ba7e6;--bar:#2c4a6e}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:1.5rem 1rem;max-width:72rem;margin-inline:auto}
nav{font-size:.9rem;margin-bottom:1rem}nav a{color:var(--acc);margin-right:1rem}
h1{font-size:1.5rem;margin:.2rem 0 .6rem}h2{font-size:1.1rem;margin:1.6rem 0 .4rem}
p{max-width:48rem}.mut{color:var(--mut)}
.controls{display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin:1rem 0}
select,input{padding:.45rem .6rem;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
input{width:18rem;max-width:100%}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:.4rem .55rem;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;cursor:pointer}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.w{color:var(--mut);font-size:13px}.lab{font-weight:600}
.barc{position:relative;min-width:7rem}.bar{position:absolute;left:0;top:.35rem;bottom:.35rem;background:var(--bar);border-radius:3px;z-index:0}
.barc span{position:relative;z-index:1}
</style></head><body>
<nav><a href="index.html">Online appendix</a><a href="../index.html">Project home</a><a href="ipo-viewer/">IPO viewer</a></nav>
<h1>The fifty themes behind the lead and atypicality scores</h1>

<h2>What a theme is</h2>
<p>A theme is a list of words that tend to turn up in the same goods/services
descriptions &mdash; <i>video, computer, electronic, music, games</i>, say, or
<i>store, retail, store services, featuring</i>. Each trademark description is
re-described as a mix of themes that sums to one: Amazon's 1997 filing, for
example, is about 41% media and entertainment goods, 15% computer and
information services, and 13% retail store services. Lead and atypicality
compare a filing's theme mix with the average mix of its own Nice class in the
five years before and the five years after its filing date.</p>

<h2>How the themes were found</h2>
<p>Nobody wrote these lists. They come from latent Dirichlet allocation, a
standard statistical method that starts from a random guess about which words
belong together and adjusts the lists, pass after pass, until each description
can be explained by a few themes and each theme uses a consistent set of words.
The model was fitted once, on a sample of 10,000 descriptions from each of the
45 Nice classes (448,437 in all), using every word and adjacent word pair that
appears in at least 50 sampled descriptions (62,168 terms), and then applied to
every filing. The labels in the table below were added afterwards by reading
each theme's leading words; the model itself only produces numbered word lists.</p>

<h2>Why fifty</h2>
<p>Fifty is coarse &mdash; roughly one theme per Nice class &mdash; and chosen
on purpose. At fifty themes every theme's leading words genuinely occur
together (average normalized pointwise mutual information 0.48, none below
zero); at 200 and 500 themes, a tenth and a fifth of themes are word lists whose
words rarely co-occur. The themes also come back when the model is refitted with
a different random seed far more faithfully at fifty than at 200. The papers'
findings do not depend on the choice: the five-year-proof lead penalty holds at
50, 200 and 500 themes and with fifty themes fitted separately inside each class.
The cost of coarseness is visible below: a few themes merge unrelated
vocabularies (marked <i>mixed</i>), and fast-growing vocabularies concentrated in
the largest classes &mdash; artificial intelligence, blockchain, cloud computing
&mdash; have no theme of their own, because the equal-per-class sample
under-represents them.</p>

<h2>The themes</h2>
<div class="controls">
<label>Show share in class <select id="cls"><option value="">(corpus-wide only)</option>__OPTS__</select></label>
<input id="q" placeholder="filter by word or label, e.g. retail, software, pet">
</div>
<p class="mut" id="note">Sorted by share of the whole corpus. "Heaviest in" lists the three classes where the theme carries the largest share of the class's language.</p>
<div class="wrap"><table><thead><tr>
<th>#</th><th>Theme (label added by reading its words)</th><th>Corpus share</th><th id="clsh">Class share</th><th>Heaviest in</th>
</tr></thead><tbody id="tb"></tbody></table></div>
<p class="mut">Shares are the average theme mix of up to 20,000 scored filings per class
(filing years 1995&ndash;2019), weighted by class size for the corpus column. The
full theme-by-class table is <a href="themes_T50.csv">themes_T50.csv</a>. A
500-theme model used as a robustness check has its own
<a href="themes/index.html">theme explorer</a>.</p>
<script>
const D=__DATA__;
const tb=document.getElementById('tb'),sel=document.getElementById('cls'),q=document.getElementById('q');
const pct=x=>(100*x).toFixed(1)+'%';
function draw(){
  const c=sel.value, f=q.value.trim().toLowerCase();
  document.getElementById('clsh').style.display=c?'':'none';
  let rows=D.rows.slice();
  if(c) rows.sort((a,b)=>b.by_class[c]-a.by_class[c]);
  const mx=Math.max(...rows.map(r=>c?r.by_class[c]:r.corpus));
  tb.innerHTML=rows.filter(r=>!f||r.label.toLowerCase().includes(f)||r.words.join(' ').includes(f)).map(r=>{
    const v=c?r.by_class[c]:r.corpus;
    const top=r.top.map(t=>t[0]+' '+t[1]+' ('+pct(t[2])+')').join('; ');
    return `<tr><td class="n">${r.k}</td><td><div class="lab">${r.label}</div><div class="w">${r.words.join(', ')}</div></td>`+
      `<td class="n barc"><div class="bar" style="width:${c?0:100*r.corpus/mx}%"></div><span>${pct(r.corpus)}</span></td>`+
      (c?`<td class="n barc"><div class="bar" style="width:${100*v/mx}%"></div><span>${pct(v)}</span></td>`:'<td style="display:none"></td>')+
      `<td class="w">${top}</td></tr>`}).join('');
  document.getElementById('note').textContent=c?('Sorted by share of class '+c+'.'):'Sorted by share of the whole corpus. "Heaviest in" lists the three classes where the theme carries the largest share of the class\'s language.';
}
sel.onchange=draw;q.oninput=draw;draw();
</script>
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
