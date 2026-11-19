"""A word index for a fitted theme model, so a theme page can be searched by any word.

A theme page lists only the dozen or so words that lead each theme. A word such as
"blockchain" leads no theme at T=500, where it is the 68th word of a large software
theme, so a search over the leading words alone either finds nothing or finds only
a theme that happens to list a phrase containing it. The index records, for every
word and two-word phrase in the model's vocabulary, which themes hold it.

The quantity is the fitted topic-word weight with the prior removed, which is the
model's count of how often it assigned the word to the theme. For each word the
index keeps the themes that hold at least MIN_SHARE of that count, the share each
holds, and the word's position in the theme when its words are ordered by weight.

Format, one line per word, in a single string that starts and ends with a newline:

    word|total|theme:share:position,theme:share:position

`total` is the word's count over all themes, `share` a whole percent. The string is
written to a .js file that sets one global, so a page can load it with a script tag
from a local file as well as from a server. SEARCH_JS is the matching reader.

Usage (from another script):
    from theme_word_index import build_index, write_js, SEARCH_JS
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_COUNT = 5.0     # a word, or a theme's holding of it, below this count is noise
MIN_SHARE = 0.05    # a theme is listed for a word when it holds at least this share
MAX_HOLDERS = 4


def build_index(model: dict) -> tuple[str, dict]:
    """Return the packed index for a model saved as {"vocabulary", "lda"}, with counts."""
    vocab, lda = model["vocabulary"], model["lda"]
    comp = lda.components_
    n_themes, n_words = comp.shape
    terms = sorted(vocab, key=vocab.get)
    bad = [t for t in terms if "|" in t or "\n" in t or ":" in t or "," in t]
    if bad:
        raise ValueError("vocabulary terms clash with the index separators: %r" % bad[:5])
    count = np.clip(comp - lda.topic_word_prior_, 0, None)

    # position of every word within every theme, by weight (1 = the theme's leading word)
    pos = np.empty((n_themes, n_words), dtype=np.int32)
    ranks = np.arange(1, n_words + 1, dtype=np.int32)
    for k in range(n_themes):
        pos[k, np.argsort(-comp[k], kind="stable")] = ranks

    total = count.sum(axis=0)
    lines, holders = [], 0
    for j, term in enumerate(terms):
        tot = float(total[j])
        if tot < MIN_COUNT:
            continue
        col = count[:, j]
        best = np.argsort(-col, kind="stable")[:MAX_HOLDERS]
        keep = [int(k) for k in best if col[k] >= MIN_COUNT and col[k] / tot >= MIN_SHARE]
        if not keep:
            keep = [int(best[0])]
        holders += len(keep)
        lines.append("%s|%d|%s" % (term, round(tot), ",".join(
            "%d:%d:%d" % (k, max(1, round(100.0 * col[k] / tot)), pos[k, j]) for k in keep)))
    blob = "\n" + "\n".join(lines) + "\n"
    return blob, {"themes": n_themes, "vocabulary": n_words, "words": len(lines), "holders": holders}


def write_js(path: Path, blob: str, var: str = "THEME_WORDS") -> None:
    Path(path).write_text("window.%s=%s;\n" % (var, json.dumps(blob)), encoding="utf-8")


# themeWordSearch(index, query) -> {total, by: {theme: {m, hits: [[word, position, count], ...]}}}
# The query is matched at the start of a word, so "insur" finds "insurance" and "health insurance".
# When the query is itself a whole word of the vocabulary, only whole-word matches count, so
# "car" does not also return "carbon". A phrase the vocabulary lacks falls back to themes that
# hold every one of its words.
SEARCH_JS = r"""
function themeWordSearch(B,q){
  function scan(needle,whole){
    var by={},total=0,i=0,n=needle.length;
    while((i=B.indexOf(needle,i))>-1){
      var p=B.charCodeAt(i-1);
      if(p!==10&&p!==32){i++;continue;}
      var ls=B.lastIndexOf('\n',i)+1,b1=B.indexOf('|',ls);
      if(i>b1){i++;continue;}
      if(whole){var e=B.charCodeAt(i+n);if(e!==32&&e!==124){i++;continue;}}
      var b2=B.indexOf('|',b1+1),le=B.indexOf('\n',b2);
      var term=B.slice(ls,b1),tot=+B.slice(b1+1,b2),hs=B.slice(b2+1,le).split(',');
      for(var h=0;h<hs.length;h++){
        var a=hs[h].split(':'),m=tot*(+a[1])/100,o=by[a[0]]||(by[a[0]]={m:0,hits:[]});
        o.m+=m;o.hits.push([term,+a[2],m]);total+=m;
      }
      i=le;
    }
    return {total:total,by:by};
  }
  function best(needle){var r=scan(needle,true);return r.total?r:scan(needle,false);}
  var r=best(q),toks=q.split(' ');
  if(r.total||toks.length<2) return r;
  var parts=toks.map(best),by={},total=0;
  for(var k in parts[0].by){
    var m=Infinity,hits=[],ok=true;
    for(var t=0;t<parts.length;t++){
      var o=parts[t].by[k];
      if(!o||!parts[t].total){ok=false;break;}
      m=Math.min(m,o.m/parts[t].total);hits=hits.concat(o.hits);
    }
    if(ok){by[k]={m:m,hits:hits};total+=m;}
  }
  return {total:total,by:by};
}
// themeWordHits(entry) -> the line a row shows for a match: the matched words and where each ranks in the theme
function themeWordHits(o){
  function ord(n){var s=['th','st','nd','rd'],v=n%100;return n.toLocaleString('en-US')+(s[(v-20)%10]||s[v]||s[0]);}
  var h=o.hits.slice().sort(function(a,b){return b[2]-a[2];});
  var out=h.slice(0,4).map(function(x){
    return '<b>'+x[0].replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</b> ('+ord(x[1])+' word)';}).join(', ');
  if(h.length>4) out+=' and '+(h.length-4)+' more';
  return 'Holds '+out;
}
"""
