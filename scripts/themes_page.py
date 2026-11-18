"""Write the online appendix's themes page.

The page explains what a theme is and lists the themes of the production
model (T = 50) with, for each: its share of the corpus or of one Nice class,
its use over time, the mean lead of the filings it dominates, and how those
filings fared against their class and year. A switch reloads the same table
for the 200- and 500-theme robustness models, and appears only for the models
whose data files exist.

All figures come from scripts/build_theme_resolution_data.py
(docs/online-appendix/themes_data/). The T = 50 data are written into the page
so it opens without a server; the other models and the per-class files are
fetched when asked for.

Usage:  python scripts/themes_page.py
Output: docs/online-appendix/themes_T50.html
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "online-appendix"


def main() -> int:
    data = json.loads((OUT / "themes_data" / "T50.json").read_text(encoding="utf-8"))
    # the switch lists only the models whose data have been built
    avail = [t for t in (50, 200, 500) if (OUT / "themes_data" / f"T{t}.json").exists()]
    seg, sentence = "", ""
    if len(avail) > 1:
        seg = ('<span class="seg" role="group" aria-label="Number of themes">' + "".join(
            f'<button type="button" data-t="{t}" aria-pressed="{"true" if t == 50 else "false"}">'
            f'{t}{" themes" if t == 50 else ""}</button>' for t in avail) + "</span>")
        others = " and ".join(f"{t}-" for t in avail if t != 50)
        sentence = (f" The table can be switched to the {others}theme "
                    f"model{'s' if len(avail) > 2 else ''}, which split{'' if len(avail) > 2 else 's'} "
                    "the same vocabulary more finely and carry no labels.")
    page = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace(
        "__SEG__", seg).replace("__SWITCH__", sentence).replace("__AVAIL__", json.dumps(avail))
    (OUT / "themes_T50.html").write_text(page, encoding="utf-8")
    print("wrote themes_T50.html")
    return 0


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>The fifty themes</title><style>
:root{--bg:#fff;--fg:#1a202c;--mut:#5a6a7a;--line:#e2e8f0;--acc:#2b6cb0;--bar:#bcd3ee;--card:#fff;--pos:#256abf;--neg:#c9302f;--mid:#eceae6;--shadow:rgba(0,0,0,.18)}
@media(prefers-color-scheme:dark){:root{--bg:#14181d;--fg:#e6edf3;--mut:#9aa8b6;--line:#2b333c;--acc:#6ba7e6;--bar:#2c4a6e;--card:#1b2129;--pos:#5598e7;--neg:#e66767;--mid:#2a3038;--shadow:rgba(0,0,0,.5)}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:1.5rem 1rem;max-width:78rem;margin-inline:auto}
nav{font-size:.9rem;margin-bottom:1rem}nav a{color:var(--acc);margin-right:1rem}a{color:var(--acc)}
h1{font-size:1.5rem;margin:.2rem 0 .6rem}h2{font-size:1.1rem;margin:1.6rem 0 .4rem}
p{max-width:48rem}.mut{color:var(--mut)}
.controls{display:flex;gap:.7rem 1rem;flex-wrap:wrap;align-items:center;margin:1rem 0 .6rem}
select,input{padding:.45rem .6rem;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);font:inherit}
input{width:17rem;max-width:100%}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{font:inherit;padding:.45rem .8rem;border:0;background:var(--card);color:var(--mut);cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
button:focus-visible,select:focus-visible,input:focus-visible,th:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.key{display:flex;flex-wrap:wrap;gap:.3rem 1.4rem;font-size:13px;color:var(--mut);margin:.2rem 0 .8rem;max-width:none}
.key i{display:inline-block;width:64px;height:9px;border-radius:2px;vertical-align:0;margin:0 .35rem;background:linear-gradient(to right,var(--neg),var(--mid),var(--pos))}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:.4rem .55rem;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:.74rem;text-transform:uppercase;letter-spacing:.03em;white-space:nowrap;vertical-align:bottom}
th.s{cursor:pointer;user-select:none}th.s:hover{color:var(--fg)}
th.s[data-dir]::after{content:" \2193"}th.s[data-dir=asc]::after{content:" \2191"}
th small{display:block;text-transform:none;letter-spacing:0;font-weight:400;font-size:.72rem}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.w{color:var(--mut);font-size:13px}.lab{font-weight:600}.hv{color:var(--mut);font-size:12px;margin-top:.15rem}
.barc{position:relative;min-width:5rem}.bar{position:absolute;left:0;top:.35rem;height:1.25rem;background:var(--bar);border-radius:3px;z-index:0}
.barc span{position:relative;z-index:1}
td.use{min-width:208px}td.use svg{display:block}td.use .pk{font-size:11.5px;color:var(--mut);white-space:nowrap}
.d{display:inline-block;width:52px;height:9px;position:relative;margin-left:.4rem;vertical-align:1px;background:linear-gradient(var(--line),var(--line)) center/1px 100% no-repeat}
.d b{position:absolute;top:0;bottom:0;border-radius:2px}
.ns{opacity:.45}
#tip{position:fixed;z-index:9;display:none;pointer-events:none;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:.35rem .55rem;font-size:12.5px;box-shadow:0 6px 20px var(--shadow);white-space:nowrap}
#tip b{font-variant-numeric:tabular-nums}
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
under-represents them.__SWITCH__</p>

<h2 id="themes">The themes</h2>
<p>Each row also shows when the theme was in use, whether the filings built on
it led or lagged their class, and how those filings fared. A filing is counted
under the theme that takes the largest share of its description. Outcomes are
measured against the rate expected from each filing's own class and year, so a
theme is not credited with the level of the class it sits in.</p>
<div class="controls">
__SEG__
<label>Class <select id="cls"><option value="">all classes</option></select></label>
<input id="q" placeholder="filter by word or label, e.g. retail, software, pet">
</div>
<p class="mut" id="note"></p>
<p class="key"><span>Use: share of each year's applications whose largest theme this is, 1990&ndash;2024; each line is scaled to its own peak. (Share is the theme's average weight in a description, so the two differ.)</span>
<span>Strip under the line: mean lead of the theme's filings by filing year, 1995&ndash;2019: lagging<i></i>leading (&plusmn;1 SD).</span>
<span>Bars beside the figures run to &plusmn;0.5 SD of lead, &plusmn;6 points, and half to twice the expected share. Faded figures are within about two standard errors of no difference.</span></p>
<div class="wrap"><table><thead><tr id="hd"></tr></thead><tbody id="tb"></tbody></table></div>
<p class="mut" id="foot"></p>
<p class="mut">A 500-theme model used as a robustness check has its own <a href="themes/index.html">theme explorer</a>.</p>
<div id="tip"></div>
<script>
const DATA={50:__DATA__};
const CLS={};                      // per-class files, keyed T + '_' + class
const tb=document.getElementById('tb'),sel=document.getElementById('cls'),q=document.getElementById('q'),tip=document.getElementById('tip');
let T=50,sortKey=null,sortDir='desc';
const pct=(x,d=1)=>(100*x).toFixed(d)+'%';
const sg=(v,d=1)=>(v>0?'+':v<0?'−':'')+Math.abs(v).toFixed(d);
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e};
const NS='http://www.w3.org/2000/svg';
const sv=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e};
const COLS=[
 {k:'k',h:'#',n:1,tt:'Theme number in the fitted model'},
 {k:null,h:'Theme'},
 {k:'share',h:'Share',n:1,tt:'Average share of a description taken by this theme'},
 {k:'peak',h:'Use over time',sub:'and lead by year',tt:'Sort by the year of peak use'},
 {k:'lead',h:'Lead',sub:'mean, SD',n:1,tt:'Mean lead of the filings this theme dominates, in standard deviations within class and filing year. Positive: their class later moved toward them.'},
 {k:'reg',h:'Registered',sub:'points vs expected',n:1,tt:'Applications filed 1995–2020 that reached registration, in percentage points against the rate of their class and filing year'},
 {k:'proof',h:'Five-year proof',sub:'points vs expected',n:1,tt:'Registrations of 1996–2018 that passed the five-year proof of continued use, in percentage points against the rate of their class and registration year'},
 {k:'sec',h:'SEC owner',sub:'× expected',n:1,tt:'Registrations whose owner is name-matched to SEC financial reporting, as a multiple of the share expected from their class and registration year'},
 {k:'ipo',h:'IPO owner',sub:'× expected',n:1,tt:'Registrations whose owner carries an IPO marker, as a multiple of the share expected from their class and registration year'}];

function rows(){
  const D=DATA[T],c=sel.value,ci=c?D.classes.findIndex(x=>x[0]===c):-1;
  if(!c) return D.rows.map(r=>Object.assign({},r,{share:r.corpus}));
  const C=CLS[T+'_'+c]; if(!C) return [];
  const base=new Map(D.rows.map(r=>[r.k,r]));
  return C.rows.map(r=>{const b=base.get(r.k);return Object.assign({},b,r,{share:b.by_class[ci]})});
}
const peak=r=>{let m=0,j=0;r.use.forEach((v,i)=>{if(v>m){m=v;j=i}});return{y:DATA[T].fy[0]+j,v:m/1e5}};

function spark(r){
  const D=DATA[T],W=200,H=38,S=9,n=r.use.length,mx=Math.max(...r.use,1),x=i=>2+i*(W-4)/(n-1),y=v=>H-2-(v/mx)*(H-5);
  const s=sv('svg',{width:W,height:H+S+2,viewBox:`0 0 ${W} ${H+S+2}`,role:'img'});
  s.append(sv('line',{x1:0,x2:W,y1:H-1.5,y2:H-1.5,stroke:'var(--line)','stroke-width':1}));
  s.append(sv('path',{d:r.use.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(''),fill:'none',stroke:'var(--acc)','stroke-width':1.8,'stroke-linejoin':'round'}));
  const off=D.ly[0]-D.fy[0],cw=(W-4)/(n-1);
  r.lead_y.forEach((z,i)=>{if(z===null)return;const t=Math.max(-1,Math.min(1,z/100));
    s.append(sv('rect',{x:(x(i+off)-cw/2).toFixed(1),y:H+2,width:(cw-.6).toFixed(1),height:S,fill:`color-mix(in srgb, var(${t<0?'--neg':'--pos'}) ${Math.round(100*Math.abs(t))}%, var(--mid))`}))});
  s.addEventListener('pointermove',e=>{const b=s.getBoundingClientRect(),i=Math.max(0,Math.min(n-1,Math.round((e.clientX-b.left-2)/(b.width-4)*(n-1)))),z=r.lead_y[i-off];
    tip.replaceChildren(el('b',null,String(D.fy[0]+i)),document.createTextNode(': '+pct(r.use[i]/1e5,2)+' of applications'+(z==null?'':'; mean lead '+sg(z/100,2)+' SD')));
    tip.style.display='block';tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY+14)+'px'});
  s.addEventListener('pointerleave',()=>{tip.style.display='none'});
  return s;
}
function num(td,v,txt,scale,faded,title){
  if(v===null||v===undefined){td.textContent='–';td.classList.add('ns');if(title)td.title=title;return}
  td.append(document.createTextNode(txt));
  const d=el('span','d'),b=document.createElement('b'),t=Math.max(-1,Math.min(1,scale));
  b.style.background=t<0?'var(--neg)':'var(--pos)';b.style.width=(50*Math.abs(t))+'%';b.style[t<0?'right':'left']='50%';
  d.append(b);td.append(d);if(faded)td.classList.add('ns');if(title)td.title=title;
}
function draw(){
  const D=DATA[T],c=sel.value,f=q.value.trim().toLowerCase();
  let R=rows().filter(r=>!f||(r.label||'').toLowerCase().includes(f)||r.words.join(' ').toLowerCase().includes(f));
  R.forEach(r=>{r.peak=peak(r).y});
  const key=sortKey||'share',dir=sortKey?sortDir:'desc';
  R.sort((a,b)=>{const x=a[key],y=b[key];if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return-1;return dir==='asc'?x-y:y-x});
  const hd=document.getElementById('hd');hd.replaceChildren();
  COLS.forEach(cn=>{const th=el('th',(cn.n?'n ':'')+(cn.k?'s':''),cn.h);if(cn.sub)th.append(el('small',null,cn.sub));if(cn.tt)th.title=cn.tt;
    if(cn.k){th.tabIndex=0;if(cn.k===key)th.dataset.dir=dir;const go=()=>{if(sortKey===cn.k)sortDir=sortDir==='desc'?'asc':'desc';else{sortKey=cn.k;sortDir=cn.k==='k'||cn.k==='peak'?'asc':'desc'}draw()};
      th.onclick=go;th.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}}}
    hd.append(th)});
  const mx=Math.max(...R.map(r=>r.share),1e-9),names=new Map(D.classes);
  tb.replaceChildren();
  for(const r of R){
    const tr=el('tr');tr.append(el('td','n',String(r.k)));
    const th=el('td');
    if(r.label)th.append(el('div','lab',r.label));
    th.append(el('div',r.label?'w':'',r.words.join(', ')));
    const hv=el('div','hv','Heaviest in '+r.top.map(t=>+t[0]+' '+(names.get(t[0])||'')+' ('+pct(t[1])+')').join('; '));
    if(T===500){hv.append(document.createTextNode(' · '));const a=el('a',null,'theme page');a.href='themes/theme-'+String(r.k).padStart(4,'0')+'.html';hv.append(a)}
    th.append(hv);tr.append(th);
    const sh=el('td','n barc'),bar=el('div','bar');bar.style.width=(100*r.share/mx)+'%';sh.append(bar,el('span',null,pct(r.share,r.share<.001?2:1)));tr.append(sh);
    const u=el('td','use'),p=peak(r);u.append(spark(r),el('div','pk','peak '+p.y+' · '+pct(p.v,p.v<.001?3:2)+' · '+r.n.toLocaleString()+' sampled'));tr.append(u);
    const a=el('td','n');num(a,r.lead_n>=50?r.lead:null,r.lead==null?'':sg(r.lead,2),(r.lead||0)/0.5,Math.abs(r.lead||0)<2/Math.sqrt(Math.max(r.lead_n,1)),r.lead_n.toLocaleString()+' filings');tr.append(a);
    const g=el('td','n');num(g,r.reg_n>=100?r.reg:null,r.reg==null?'':sg(r.reg),(r.reg||0)/6,Math.abs(r.reg||0)<100/Math.sqrt(Math.max(r.reg_n,1)),r.reg_n.toLocaleString()+' applications');tr.append(g);
    const pf=el('td','n');num(pf,r.proof_n>=100?r.proof:null,r.proof==null?'':sg(r.proof),(r.proof||0)/6,Math.abs(r.proof||0)<100/Math.sqrt(Math.max(r.proof_n,1)),r.proof_n.toLocaleString()+' registrations');tr.append(pf);
    for(const k of['sec','ipo']){const e=r[k+'_e'],o=r[k+'_n'],td=el('td','n');
      num(td,e>=5?r[k]:null,r[k]==null?'':r[k].toFixed(2)+'×',Math.log2(Math.max(r[k]||0,.05)),Math.abs(o-e)<2*Math.sqrt(Math.max(e,1)),o.toLocaleString()+' registrations, '+e.toFixed(1)+' expected');tr.append(td)}
    tb.append(tr);
  }
  const N=Object.values(D.sizes).reduce((s,v)=>s+v.sampled,0);
  document.getElementById('note').textContent=(c?R.length+' themes with at least 40 sampled filings in class '+(+c)+' '+(names.get(c)||'')+'; every column describes that class alone.':R.length+' of '+D.rows.length+' themes, all classes together.')+(sortKey?'':' Sorted by share.');
  document.getElementById('foot').replaceChildren(document.createTextNode('Shares are the average theme mix of a sample of each class’s filings, weighted by class size for all classes together. Every other column is computed on '+N.toLocaleString()+' applications filed 1990–2024, up to '+D.cap.toLocaleString()+' sampled per class and weighted back to class size. SEC reporting and the IPO marker are properties of the owner, so a firm with many marks counts once per mark. Data: '),
    Object.assign(el('a',null,'themes_data/T'+T+'.json'),{href:'themes_data/T'+T+'.json'}),document.createTextNode(T===50?'; theme-by-class shares: ':'.'),
    ...(T===50?[Object.assign(el('a',null,'themes_T50.csv'),{href:'themes_T50.csv'}),document.createTextNode('.')]:[]));
}
async function need(){
  if(!DATA[T]) DATA[T]=await (await fetch('themes_data/T'+T+'.json')).json();
  const c=sel.value; if(c&&!CLS[T+'_'+c]) CLS[T+'_'+c]=await (await fetch('themes_data/T'+T+'_c'+c+'.json')).json();
}
async function update(){tb.style.opacity=.5;try{await need();draw()}catch(e){document.getElementById('note').textContent='Could not load the data for this view (the 200- and 500-theme tables and the per-class views need the page to be served over http).'}tb.style.opacity=1}
for(const [c,n] of DATA[50].classes){const o=el('option',null,(+c)+' '+n);o.value=c;sel.append(o)}
document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>{T=+b.dataset.t;document.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed',x===b));update()});
sel.onchange=update;q.oninput=draw;
const h=location.hash.match(/T=(\d+)/);if(h&&__AVAIL__.includes(+h[1])&&+h[1]!==50){T=+h[1];document.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed',+x.dataset.t===T))}
update();
</script>
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
