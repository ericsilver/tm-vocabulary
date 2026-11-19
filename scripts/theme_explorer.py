"""A browsable page of every theme in a fitted model, for judging the partition.

The theme count is the one modelling choice a reader is asked to take entirely
on trust: the paper says a filing's language is represented by its weight over
T fitted themes, and then never shows what those themes are. This writes a
static page listing all of them with their leading words, how much of the
corpus each carries, and -- for one Nice class -- how many filings it is the
dominant theme for. That last column is what makes the partition judgeable:
a theme that is dominant for no filing in a class is not resolving structure
in that class, whatever it looks like in the abstract.

The word search reads an index of every word in the model and the themes that hold
it (theme_word_index.py), so a word that leads no theme can still be found. At
T=500 the page shares themes/words.js with the explorer that theme_pages.py writes;
at any other T this script writes themes_T{T}_words.js beside the page.

Usage:  python scripts/theme_explorer.py [T] [CLASS] [N_DOCS]
Output: docs/online-appendix/themes_T{T}.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]

# The published pages take the look of aporia.institute: a stylesheet link and the wordmark header.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_site_style import apply as apply_site_style  # noqa: E402
from theme_word_index import SEARCH_JS, build_index, write_js  # noqa: E402
PROC = REPO / "data" / "processed"
OUT = REPO / "docs" / "online-appendix"

T = int(sys.argv[1]) if len(sys.argv) > 1 else 500
CLS = sys.argv[2] if len(sys.argv) > 2 else "009"
NDOCS = int(sys.argv[3]) if len(sys.argv) > 3 else 150_000
NWORDS = 12
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"

# The search box and its count line, the style of the line naming matched words, and the script.
SEARCH_BOX = ('<input id="q" placeholder="find a word in any theme, e.g. blockchain, insurance, kombucha">'
              '<span id="cnt" style="color:var(--mut);font-size:13px;margin-left:.8rem"></span>')
SEARCH_CSS = ".hit{color:var(--mut);font-size:13px;margin-top:.25rem}.hit b{color:var(--fg);font-weight:600}"
SEARCH_PAGE_JS = SEARCH_JS + """
const q=document.getElementById('q'),cnt=document.getElementById('cnt'),tb=document.getElementById('tb');
const rows=[...tb.querySelectorAll('tr')];
let W=null,ws=0;   // word index: 0 not asked for, 1 loading, 2 ready, 3 failed
function load(){
  if(ws) return;
  ws=1;
  const s=document.createElement('script');
  s.src='__WORDS__';
  s.onload=()=>{W=window.THEME_WORDS||null;ws=W?2:3;apply();};
  s.onerror=()=>{ws=3;apply();};
  document.head.appendChild(s);
}
function apply(){
  const v=q.value.toLowerCase().trim().replace(/\\s+/g,' '),on=v.length>=2;
  let res=null,shown=0;
  if(on){ if(!ws) load(); if(ws===2) res=themeWordSearch(W,v); }
  const vis=[];
  rows.forEach(r=>{
    const old=r.children[1].querySelector('.hit');
    if(old) old.remove();
    let ok=true;
    if(on&&res){
      const o=res.by[r.children[0].textContent];
      ok=!!o;
      if(o){
        const d=document.createElement('div'),pc=100*o.m/res.total;
        d.className='hit';
        d.innerHTML=themeWordHits(o)+': '+(pc>=10?pc.toFixed(0):pc.toFixed(1))+'% of the search word';
        r.children[1].appendChild(d);
      }
    } else if(on){
      ok=r.dataset.words.toLowerCase().includes(v);
    }
    r.style.display=ok?'':'none';
    if(ok) vis.push(r);
  });
  // while a word is searched the theme that holds most of it comes first; otherwise the page's own order
  const order=(on&&res)?vis.slice().sort((a,b)=>res.by[b.children[0].textContent].m-res.by[a.children[0].textContent].m):rows;
  order.forEach(r=>tb.appendChild(r));
  shown=vis.length;
  if(on&&res) cnt.textContent=shown?shown+(shown===1?' theme holds':' themes hold')+' \u201c'+v+'\u201d':'No theme holds a word starting \u201c'+v+'\u201d';
  else if(on&&ws===1) cnt.textContent='Loading the word index\u2026';
  else if(on) cnt.textContent=shown+' themes list \u201c'+v+'\u201d among their leading words (the full word index did not load)';
  else cnt.textContent='';
}
q.addEventListener('focus',load);
q.addEventListener('input',apply);
"""


def words_file(model, out: Path, t: int) -> str:
    """Path, relative to the page, of the word index for this model; written here unless the explorer has it."""
    if t == 500 and (out / "themes" / "words.js").exists():
        return "themes/words.js"
    blob, _ = build_index(model)
    write_js(out / f"themes_T{t}_words.js", blob)
    return f"themes_T{t}_words.js"


def main() -> int:
    mp = PROC / ("topic_model.joblib" if T == 50 else f"topic_model_T{T}.joblib")
    m = joblib.load(mp)
    lda, vocab = m["lda"], m["vocabulary"]
    OUT.mkdir(parents=True, exist_ok=True)
    page_js = SEARCH_PAGE_JS.replace("__WORDS__", words_file(m, OUT, T))
    inv = {v: k for k, v in vocab.items()}
    comp = lda.components_
    top = [[inv[j] for j in np.argsort(comp[k])[::-1][:NWORDS]] for k in range(T)]

    raw = pl.read_parquet(
        PROC / f"tm_class{CLS}.parquet",
        columns=["filing_date", "goods_services"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
    texts = raw["goods_services"].head(NDOCS).to_list()
    vec = CountVectorizer(vocabulary=vocab, lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    th = lda.transform(vec.transform(texts))
    np.clip(th, 1e-12, None, out=th)
    th /= th.sum(axis=1, keepdims=True)
    mass = th.mean(axis=0)
    dom = np.bincount(th.argmax(axis=1), minlength=T)
    order = np.argsort(mass)[::-1]

    rows = []
    for k in order:
        words = ", ".join(top[k])
        flag = ' class="empty"' if dom[k] == 0 else ""
        rows.append(
            f'<tr{flag} data-words="{html.escape(words)}">'
            f'<td class="n">{k}</td>'
            f'<td class="w">{html.escape(words)}</td>'
            f'<td class="n">{100*mass[k]:.2f}%</td>'
            f'<td class="n">{dom[k]:,}</td></tr>')

    used = int((mass >= 0.01).sum())
    zero = int((dom == 0).sum())
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Themes at T={T}</title><style>
:root{{--bg:#fff;--fg:#1a202c;--mut:#5a6a7a;--line:#e2e8f0;--warn:#c05621;--acc:#2b6cb0}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#14181d;--fg:#e6edf3;--mut:#9aa8b6;--line:#2b333c;--warn:#e08a4f;--acc:#6ba7e6}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:2rem}}
h1{{font-size:1.4rem;margin:0 0 .3rem}}p.sub{{color:var(--mut);margin:0 0 1.2rem;max-width:60rem}}
.stats{{display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1.2rem}}
.stat b{{display:block;font-size:1.5rem}}.stat span{{color:var(--mut);font-size:.85rem}}
input{{padding:.5rem .7rem;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);width:22rem;max-width:100%}}
.wrap{{overflow-x:auto;margin-top:1rem}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}}
th{{position:sticky;top:0;background:var(--bg);color:var(--mut);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.w{{color:var(--fg)}}tr.empty td{{color:var(--warn)}}
tr.empty td.w::after{{content:" — dominant for no filing in this class";color:var(--warn);font-style:italic}}
{SEARCH_CSS}</style></head><body>
<h1>Themes at T&nbsp;=&nbsp;{T}</h1>
<p class="sub">Every theme in the fitted model, with its leading words, the share
of Nice class {CLS} it carries, and how many of {len(texts):,} class-{CLS} filings
it is the single most-weighted theme for. Sorted by mass in this class. Themes
dominant for no filing here are marked; they may still carry another class.</p>
<div class="stats">
<div class="stat"><b>{T}</b><span>themes fitted</span></div>
<div class="stat"><b>{used}</b><span>carry &ge;1% of this class</span></div>
<div class="stat"><b>{zero}</b><span>dominant for no filing here</span></div>
<div class="stat"><b>{len(vocab):,}</b><span>vocabulary terms</span></div>
</div>
{SEARCH_BOX}
<div class="wrap"><table><thead><tr>
<th>#</th><th>Leading words</th><th>Mass in class {CLS}</th><th>Filings dominant</th>
</tr></thead><tbody id="tb">
{chr(10).join(rows)}
</tbody></table></div>
<script>{page_js}</script></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"themes_T{T}.html"
    p.write_text(apply_site_style(doc), encoding="utf-8")
    print(f"[explorer] T={T}: {used} themes >=1% of class {CLS}, "
          f"{zero} dominant for none -> {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
