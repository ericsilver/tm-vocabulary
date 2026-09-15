"""Assemble the two split papers from the v3_rp / v3 section files.

Run from paper/split:  python assemble.py [--no-build]

Nothing under v3_rp or v3 is modified. Generated files are written as
gen_*.tex inside A_corpus/ and B_lead/; hand-written files (abstract,
intro, bridges) live beside them and are never overwritten here.
"""
import io, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.join(HERE, "..", "v3_rp")
V3 = os.path.join(HERE, "..", "v3")

def rd(*p):
    return io.open(os.path.join(*p), encoding="utf-8").read()

def wr(path, s):
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)

SUB = re.compile(r"^\\subsection\{", re.M)
SEC = re.compile(r"^\\section\{", re.M)

def split_subs(text):
    """Return (head, [(title, body_with_heading)]) split on \\subsection."""
    idx = [m.start() for m in SUB.finditer(text)]
    if not idx:
        return text, []
    head = text[:idx[0]]
    parts = []
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(text)
        chunk = text[s:e]
        title = re.match(r"\\subsection\{(.*?)\}", chunk).group(1)
        parts.append((title, chunk))
    return head, parts

def split_secs(text):
    idx = [m.start() for m in SEC.finditer(text)]
    parts = []
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(text)
        chunk = text[s:e]
        title = re.match(r"\\section\{(.*?)\}", chunk).group(1)
        parts.append((title, chunk))
    return parts

def sub(parts, key):
    for t, c in parts:
        if key.lower() in t.lower():
            return c
    raise KeyError(key)

def sec(parts, key):
    return sub(parts, key)

def patch(text, pairs, name=""):
    for old, new in pairs:
        if old not in text:
            print(f"  [warn] patch not found in {name}: {old[:70]!r}")
        text = text.replace(old, new)
    return text

# ---- sources -------------------------------------------------------------
measure = rd(RP, "rp_measure.tex")
m_head, m_subs = split_subs(measure)
related = rd(RP, "rp_related.tex")
r_head, r_subs = split_subs(related)
sec2 = rd(RP, "sec2_registration.tex")
s2_head, s2_subs = split_subs(sec2)
sec3 = rd(RP, "sec3_gate.tex")
s3_head, s3_subs = split_subs(sec3)
sec4 = rd(RP, "sec4_waves.tex")
sec5 = rd(RP, "sec5_success.tex")
s5_head, s5_subs = split_subs(sec5)
disc = rd(RP, "rp_discussion.tex")
d_head, d_subs = split_subs(disc)
robust_rp = rd(RP, "sec5_robustness.tex")
robust_v3 = rd(V3, "sec5_robustness.tex")
rv_head, rv_subs = split_subs(robust_v3)
appx = rd(V3, "appendices.tex")
ap_secs = split_secs(appx)

COMPANION = "the companion paper"

# =========================================================================
# PAPER A - corpus and measure
# =========================================================================
A = os.path.join(HERE, "A_corpus")

# --- record: gates subsection, representation, refiling ---
gates = sub(m_subs, "gates, the samples")
gates_A = patch(gates, [
    ("\\subsection{The gates, the samples, and the conventions}\\label{sec:gates}",
     "\\subsection{The steps, the samples, and the conventions}\\label{sec:gates}"),
    ("The outcome sections that follow are written to be read independently, so\nthe record's steps, the outcome names, and the counting conventions are\nset out once, here.",
     "The record's steps, the outcome names, and the counting conventions are\nset out once, here, and every later section uses them."),
    ("(Section~\\ref{sec:registration})", "(Section~\\ref{sec:val_registration})"),
    ("(Sections~\\ref{sec:gate} and~\\ref{sec:waves})", "(Section~\\ref{sec:val_gate})"),
    ("(Section~\\ref{sec:success})", "(Section~\\ref{sec:links})"),
    ("(Appendix~\\ref{app:data})", "(Section~\\ref{sec:burnin})"),
    ("--- a bound that matters, because carried back to the first\nscoreable cohorts the penalty was larger than it has ever been since\n(Section~\\ref{sec:eras}).",
     "--- a bound that matters, because carried back to the first\nscoreable cohorts the lead penalty was larger than it has ever been\nsince, as " + COMPANION + " shows."),
], "gates_A")
wr(os.path.join(A, "gen_record_gates.tex"), gates_A)

wr(os.path.join(A, "gen_record_representation.tex"), sub(s2_subs, "Self-filed"))

amend = sub(s2_subs, "Office actions and refiling")
refile_app = sec(ap_secs, "How refiled descriptions")
refile_body = re.sub(r"^\\section\{.*?\}\\label\{app:refile\}\n", "", refile_app)
wr(os.path.join(A, "gen_record_refile.tex"), amend + "\n" + refile_body)

# --- measure ---
rep_app = sec(ap_secs, "Term scoring measures length")
rep_app = rep_app.replace("\\section{", "\\subsection{", 1)
exhibit = sec(ap_secs, "How three filings encode")
exhibit = exhibit.replace("\\section{", "\\subsection{", 1)
burnin_sec = sec(ap_secs, "Corpus, scoring, and burn-in")
_, bi_subs = split_subs(burnin_sec)
burnin = sub(bi_subs, "interpretable from 1995")
burnin = burnin.replace("\\subsection{Scores are interpretable from 1995}",
                        "\\subsection{Scores are interpretable from 1995}\\label{sec:burnin}")
measure_A = ("\\section{The measure}\\label{sec:measure}\\label{sec:construct}\n\n"
             + sub(m_subs, "Two readers") + sub(m_subs, "What is measured")
             + sub(m_subs, "From a description") + sub(m_subs, "Within class")
             + rep_app + exhibit + burnin)
wr(os.path.join(A, "gen_measure.tex"), measure_A)

# --- nomenclature / neighbours ---
related_A = ("\\section{Neighbours and nomenclature}\\label{sec:related}\n\n"
             + sub(r_subs, "Trademarks as innovation data")
             + sub(r_subs, "Text measures of novelty")
             + sub(r_subs, "Nomenclature"))
wr(os.path.join(A, "gen_related.tex"), related_A)

# --- validation ---
reg_rates = sub(s2_subs, "Registration rises with atypicality")
reg_rates = reg_rates.replace("\\label{sec:reg_rates}", "\\label{sec:reg_rates}\\label{sec:val_registration}", 1)
reg_app = sec(ap_secs, "Examination rewards a middling specificity")
reg_app = reg_app.replace("\\section{", "\\subsection{", 1)
reg_app = patch(reg_app, [
    ("strongly\n(Table~\\ref{tab:staged}), but monotonically",
     "strongly\n(harmonized estimates in " + COMPANION + "), but monotonically"),
], "reg_app")
wr(os.path.join(A, "gen_val_registration.tex"), reg_rates + reg_app)

s3_head_A = patch(s3_head, [
    ("\\section{Persistence in commerce: the five-year proof of continued use}\\label{sec:gate}",
     "\\subsection{The five-year proof of continued use: leading marks fail it more often}\\label{sec:gate}\\label{sec:val_gate}"),
    ("How the\npenalty found here moved across industries and decades is\nSection~\\ref{sec:waves}; what became of the firms is\nSection~\\ref{sec:success}.",
     "How the\npenalty moved across industries and decades, and what became of the\nfirms, is the subject of " + COMPANION + "."),
], "s3_head_A")
def demote(chunk):
    return chunk.replace("\\subsection{", "\\subsubsection{", 1)
gate_headline = patch(sub(s3_subs, "Leading marks fail"), [
    ("Section~\\ref{sec:waves} takes that variation as its subject.",
     COMPANION.capitalize() + " takes that variation as its subject."),
], "gate_headline_A")
renewal = patch(sub(s3_subs, "year-ten renewal"), [
    ("(Table~\\ref{tab:staged},\nboth stated on survival)", "(harmonized estimates in " + COMPANION + ",\nboth stated on survival)"),
], "renewal_A")
gate_A = (s3_head_A + demote(sub(s3_subs, "conditions on registration")) + demote(gate_headline)
          + demote(sub(s3_subs, "three theme scorings")) + demote(renewal)
          + demote(sub(s3_subs, "What the five-year proof shows")))
wr(os.path.join(A, "gen_val_gate.tex"), gate_A)

wr(os.path.join(A, "gen_val_patent.tex"), sub(s5_subs, "not patenting"))

edgar = sub(s5_subs, "Leading debuts reach public reporting")
edgar = edgar.replace("\\subsection{Leading debuts reach public reporting less often; unusual ones no more often}",
                      "\\subsection{SEC reporting: the lead profile survives the design; the atypicality gradient is composition}", 1)
edgar = patch(edgar, [
    ("The outcome here is whether an owner ever appears",
     "The outcome here is the firm-level one that the representation hazard\nof Section~\\ref{sec:hazard} turns on: whether an owner ever appears"),
], "edgar_A")
wr(os.path.join(A, "gen_val_edgar.tex"), edgar)

hazard = sub(rv_subs, "Term scoring at the firm level")
hazard = re.sub(r"^\\subsection\{.*?\}\\label\{sec:robust_hazard\}\n", lambda m: "\\label{sec:robust_hazard}\n", hazard)
wr(os.path.join(A, "gen_val_hazard.tex"), hazard)

# --- robustness (full v3 treatment, minus what moved) ---
keep = [c for t, c in rv_subs if not any(k in t for k in
        ["Term scoring at the firm level", "Changing the time period", "What remains unresolved"])]
robust_A = rv_head + "".join(keep)
robust_A = patch(robust_A, [
    ("Each new analysis in Sections~\\ref{sec:registration}--\\ref{sec:success} was rerun\nunder its natural parameter toggles, on the identical data.",
     "Each analysis in this paper and in " + COMPANION + " was rerun under its\nnatural parameter toggles, on the identical data."),
    ("which is Section~\\ref{sec:eras}'s era swing seen\nfrom the other side.",
     "which is the era swing " + COMPANION + " documents, seen\nfrom the other side."),
    ("The raw quintile contrasts of\nSections~\\ref{sec:gate}--\\ref{sec:waves} --- the era, theme, internet\nand surge tables --- are not:",
     "The raw quintile contrasts of\n" + COMPANION + " --- the era, theme, internet\nand surge tables --- are not:"),
    ("and Section~\\ref{sec:registration}\nreports only the unsigned version.",
     "and the registration analysis\nreports only the unsigned version."),
    ("Each result above is stated under one formulation chosen for legibility",
     "Each result in Section~\\ref{sec:validation} is stated under one formulation chosen for legibility"),
], "robust_A")
wr(os.path.join(A, "gen_robust.tex"), robust_A)

# --- practices ---
impl = sub(d_subs, "Implications for trademark-based measurement")
impl = impl.replace("\\subsection{Implications for trademark-based measurement}\\label{sec:implications}",
                    "\\section{Practices for trademark-text research}\\label{sec:practices}", 1)
wr(os.path.join(A, "gen_practices.tex"), impl)

main_A = r"""% Paper A: corpus + measure. Assembled by ../assemble.py; hand-written
% files: abstract, intro, record_head, owner_links, validation_head,
% hazard_head, access. gen_*.tex are generated from ../../v3_rp and ../../v3.
\input{../../v3_rp/preamble}
\newcommand{\onlineappendixnote}{\url{https://aporia.institute/tm-vocabulary/online-appendix/themes_T50.html}}
\graphicspath{{../../}}
\makeatletter\def\input@path{{./}{../../}{../../v3_rp/}}\makeatother
\title{\bfseries An Event-Dated Corpus of US Trademark Prosecution\\
\large and a Two-Sided Measure of Vocabulary Position}
\author{Eric Silver\thanks{Independent researcher; formerly a PhD candidate
at Carnegie Mellon University.}}
\date{Draft, September 2026 --- Paper A of two}

\begin{document}
\maketitle
\input{abstract}
\input{intro}
\input{record_head}
\input{gen_record_gates}
\input{gen_record_representation}
\input{gen_record_refile}
\input{owner_links}
\input{gen_measure}
\input{gen_related}
\input{validation_head}
\input{gen_val_registration}
\input{gen_val_gate}
\input{gen_val_patent}
\input{gen_val_edgar}
\input{hazard_head}
\input{gen_val_hazard}
\input{gen_robust}
\input{gen_practices}
\input{access}
\input{../../v3_rp/backmatter}
\input{../../v3_rp/bibliography_rp}
\end{document}
"""
wr(os.path.join(A, "main.tex"), main_A)

# =========================================================================
# PAPER B - the lead penalty
# =========================================================================
B = os.path.join(HERE, "B_lead")

nomen = sub(r_subs, "Nomenclature")
table = re.search(r"\\begin\{table\}.*?\\end\{table\}", nomen, re.S).group(0)
related_B = ("\\section{Pioneering, conformity, and the trademark record}\\label{sec:related}\n\n"
             + sub(r_subs, "Pioneering and optimal distinctiveness")
             + sub(r_subs, "Trademarks as innovation data")
             + sub(r_subs, "Text measures of novelty")
             + "\\input{related_bridge}\n\n" + table + "\n")
wr(os.path.join(B, "gen_related.tex"), related_B)

gates_B = patch(gates, [
    ("(Appendix~\\ref{app:data})", "(" + COMPANION + ")"),
], "gates_B")
wr(os.path.join(B, "gen_gates.tex"), gates_B)

wr(os.path.join(B, "gen_sec2.tex"), sec2)
wr(os.path.join(B, "gen_sec3.tex"), sec3)
wr(os.path.join(B, "gen_sec4.tex"), sec4)

s5_keep = [c for t, c in s5_subs if "not patenting" not in t]
sec5_B = s5_head + "".join(s5_keep)
sec5_B = patch(sec5_B, [
    ("Being unusual is at worst free and at best mildly rewarded;",
     "Within the firms that do both, vocabulary position and patenting\nmove independently (every coefficient under $0.06$ standard deviations;\n" + COMPANION + "), so none of this is patenting seen through a\ndifferent record. Being unusual is at worst free and at best mildly rewarded;"),
], "sec5_B")
wr(os.path.join(B, "gen_sec5.tex"), sec5_B)

disc_B = d_head + "\\input{discussion_tail}\n"
wr(os.path.join(B, "gen_discussion.tex"), disc_B)

robust_B = patch(robust_rp, [
    ("the term-scored replication and its firm-level hazard,", "the term-scored replication,"),
], "robust_B")
wr(os.path.join(B, "gen_robust.tex"), robust_B)

main_B = r"""% Paper B: the lead penalty. Assembled by ../assemble.py; hand-written
% files: abstract, intro, measure_brief, related_bridge, discussion_tail.
% gen_*.tex are generated from ../../v3_rp.
\input{../../v3_rp/preamble}
\newcommand{\onlineappendixnote}{\url{https://aporia.institute/tm-vocabulary/online-appendix/themes_T50.html}}
\graphicspath{{../../}}
\makeatletter\def\input@path{{./}{../../}{../../v3_rp/}}\makeatother
\title{\bfseries Arrows in Their Backs:\\
\large Vocabulary Lead and Product Survival in the US Trademark Record}
\author{Eric Silver\thanks{Independent researcher; formerly a PhD candidate
at Carnegie Mellon University.}}
\date{Draft, September 2026 --- Paper B of two}

\begin{document}
\maketitle
\input{abstract}
\input{intro}
\input{gen_related}
\input{measure_brief}
\input{gen_gates}
\input{gen_sec2}
\input{gen_sec3}
\input{gen_sec4}
\input{gen_sec5}
\input{gen_discussion}
\input{gen_robust}
\input{../../v3_rp/backmatter}
\input{../../v3_rp/bibliography_rp}
\end{document}
"""
wr(os.path.join(B, "main.tex"), main_B)

# =========================================================================
# build
# =========================================================================
def build(d):
    for i in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=d, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(f"[{os.path.basename(d)}] pdflatex FAILED on pass {i+1}")
            log = rd(d, "main.log")
            m = re.search(r"^! .*?(?=\n\n)", log, re.S | re.M)
            print(m.group(0) if m else r.stdout[-2000:])
            return False
    log = rd(d, "main.log")
    und = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
    cit = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", log)))
    pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
    print(f"[{os.path.basename(d)}] built: {pages.group(1) if pages else '?'} pages; "
          f"undefined refs: {und or 'none'}; undefined cites: {cit or 'none'}")
    return True

if "--no-build" not in sys.argv:
    build(A); build(B)
