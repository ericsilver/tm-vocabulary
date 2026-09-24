# Vocabulary position in the US trademark record

This repository builds an event-dated corpus of all 13.99 million USPTO trademark case files and defines a filing-level text measure on it. Each filing's goods/services description is reduced to a mix of fifty themes and compared with what the filing's own Nice class filed in the 1,826 days *before* it and the 1,826 days *after* it. The average of the two comparisons is **atypicality** (how unusual the language is for its industry); their difference is **lead** (whether the language ran ahead of or behind where the industry's language was moving). Because the corpus carries every prosecution event with its date, a filing's language can be followed to what happened to the mark: registration, the five-year proof of continued use (the sworn §8 declaration due in years five to six), the year-ten renewal, and the owner's appearance in SEC reporting, Form D rounds and listing.

## The papers

| | Paper A | Paper B |
|---|---|---|
| Title | *An Event-Dated Corpus of US Trademark Prosecution and a Two-Sided Measure of Vocabulary Position* | *Arrows in Their Backs: Vocabulary Lead and Product Survival in the US Trademark Record* |
| What it does | Builds the corpus and the measure, validates the measure by rescoring under every alternative construction, documents two hazards for trademark-text research | Asks whether being early pays: the five-year proof, the era swing and its themes, the internet, surges, funding and listing |
| Source | `paper/split/A_qss/` (`main.tex`, supplement `supp.tex`) | `paper/split/B_lead/` (`main.tex`, supplement `supp.tex`) |
| PDF | [SSRN 7520758](https://ssrn.com/abstract=7520758); `paper/split/A_qss/submission/manuscript.pdf`, `supplementary_material.pdf` | `paper/split/B_lead/main.pdf`, `supp.pdf` |
| Build | `cd paper/split/A_qss && pdflatex main` (×3), then `pdflatex supp` (×2) | `cd paper/split/B_lead && pdflatex main` (×3), then `pdflatex supp` (×2) |

Both develop the earlier combined working paper, *Business Themes in the Trademark Record* ([SSRN 6954598](https://ssrn.com/abstract=6954598)). `paper/split/A_corpus/` is a longer version of Paper A kept for reference. Everything else under `paper/` — `v3/`, `v3_rp/`, `frozen/`, `_legacy/`, and the older `.tex` files at the top level — is earlier work, kept for provenance only; it does not describe the current measure or results. (`paper/v3_rp/` still supplies the shared preamble, back matter and bibliography the two papers `\input`.)

## The measure in one screen

For filing *i* made on date *d*, with theme mix *P<sub>i</sub>*:

| Quantity | Definition | Column | Name in the papers | Murdock/Barron name |
|---|---|---|---|---|
| Surprise against the past | KL(*P<sub>i</sub>* ‖ *Q*⁻), *Q*⁻ = mean theme mix of same-class filings in [*d*−1826, *d*) | `topic_kl_vs_past` | past-facing surprise *K*⁻ | novelty |
| Surprise against the future | KL(*P<sub>i</sub>* ‖ *Q*⁺), *Q*⁺ over (*d*, *d*+1826] | `topic_kl_vs_future` | future-facing surprise *K*⁺ | transience |
| Average | *A* = ½(*K*⁻ + *K*⁺) | formed downstream | **atypicality** | — |
| Signed difference | *L* = *K*⁻ − *K*⁺ | `topic_dkl` | **lead** (leading / lagging) | resonance |

- **Positive lead means the class moved toward the filing**: unusual against what came before, ordinary against what came after.
- Both windows are anchored on the filing's own date and exclude it; every application in the class in the window counts equally, registered or not; a filing is scored only when each window holds at least 500 filings.
- *K*⁻ and *K*⁺ correlate at about 0.99, which is why the papers use their average and difference (a sum and a difference, nothing estimated) rather than the pair. *L* is a small residual of two large quantities and correspondingly noisy.
- Everything is estimated **within class and year**. Across classes atypicality levels are not comparable: a class where the USPTO ID Manual supplies dense standard language has a compressed distribution for reasons unrelated to innovation.
- Scoring covers filings made 1995–2019 (`SURPRISE_SRC=rolling`, the default everywhere).

### The theme model

- Latent Dirichlet allocation (scikit-learn, online variational Bayes, default priors, 8 passes, seed 42), **50 themes**, fitted once on a sample of 10,000 descriptions per Nice class (every description for smaller classes; 448,437 in all; filings 1990–2024), then applied to every filing.
- Vocabulary: words and adjacent word pairs appearing in at least 50 sampled descriptions, stopwords removed — 62,168 terms (`novelty.dictionary._make_analyzer`). At scoring time filings are read with a simpler tokenizer (3+ letter tokens, no stopword removal); `scripts/tokenizer_check.py` measures the difference (same dominant theme for 88% of descriptions; median theme-mix change 0.07 in total variation).
- Why fifty: `scripts/topic_coherence.py` (every theme coherent at 50; a tenth incoherent at 200, a fifth at 500), `scripts/topic_seed_replicate_T50.py` (themes reproduce across seeds far better at 50 than at 200; per-filing lead does not), and the robustness scorings (500 global themes; 50 themes fitted per class, `perclass_lda_rescore.py`; 200 themes), under all of which the headline holds.
- The equal-per-class sample under-represents vocabulary concentrated in the largest classes (`scripts/sample_thinning.py`): internet terms appear in 7.4% of sampled descriptions against 14.7% of the corpus, cloud computing 0.4% against 2.4%. No theme is dedicated to AI, blockchain or cloud computing; analyses of those waves use curated word lists.
- The fifty themes, with labels and their share of every class: [online appendix](https://aporia.institute/tm-vocabulary/online-appendix/themes_T50.html) (`scripts/themes_t50_page.py`).

## Data

The repository holds **code only**; `data/` is gitignored. Two routes to the data:

1. **Data release** (`gh release` tag `data-v1`, built by `scripts/build_release_tables.py`): the event-dated proof outcomes for every registration (`proof_outcomes.parquet` — reusable without the text measure), per-filing scores under the production scoring (`scores_T50.zip`), the fitted theme model (`theme_model_T50.zip`), the 242 million dated prosecution events (`case_events.parquet`), counsel/basis/declaration fields (`case_extras.parquet`), the event-code dictionary, and the owner links (`owner_links.zip`), with `MANIFEST.txt` giving SHA-256 digests.
2. **Rebuild from public sources** with the pipeline below.

`data_publish/firm_year_dkl.csv` and `firm_year_patents_and_dkl.csv` are an early **word-scored** firm-year panel from a retired build. They are not the papers' measure, and Paper A documents why word-scored firm-level correlations are unreliable; they are kept only for provenance.

## Data pipeline

**1. Bulk XML → per-class records.** `scripts/download_all_classes.py` streams the USPTO TRTYRAP backfile (83 archives, 1884–2025, ~12 GB) once and writes a slim parquet per Nice class. A filing declaring several classes is written into each of their parquets. → `data/processed/tm_class{NNN}.parquet`

**2. Bulk XML → prosecution events.** `scripts/events_full_build.py` re-parses the same backfile for the dated sequence of events behind each case's status: 242 million events across 13.99 million case files. → `case_events.parquet`, `case_extras.parquet`, `event_code_dict.parquet`

**3. Theme model.** `scripts/topic_p_scorer_all.py` fits the LDA (`TOPIC_T=50` default; 200/500 for the robustness scorings) and writes `topic_model*.joblib` and `topic_lda_meta*.json`. (It also writes an older calendar-year scoring, `topic_surprise_*`, which is retired.)

**4. Production scores on per-filing windows.** `scripts/rolling_rescore_all.py` reuses the fitted model and scores every class against references anchored on each filing's date; then `scripts/rolling_add_year.py` backfills the `year` column. → `rolling_surprise_class{NNN}[_T{T}].parquet`. Word-scored comparisons: `scripts/term_rescore_rolling.py` → `termroll_surprise_*`.

**5. External records.** `scripts/download_sec_fsds.py` + `scripts/sec_extract.py` build the SEC financial-statement panel; `scripts/sec_link.py` resolves owner names to CIKs; `scripts/persist_funding_match.py` resolves Regulation D (Form D) issuers. PatentsView assignees are matched in `scripts/wsC_within_firm_patents.py`. → `sec_firm_year.parquet`, `uspto_sec_crosswalk.parquet`, `funding_owner_match.parquet`

**6. Analysis.** The scripts in the tables below write JSON, figures and `.tex` fragments into `paper/results/`, which the papers read directly.

## Reproducing

Requires Python 3.11, a TeX install, and a free USPTO Open Data Portal API key (<https://data.uspto.gov>, My ODP → My API Key).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install statsmodels matplotlib rapidfuzz
cp .env.example .env          # then add USPTO_ODP_API_KEY=<your key>
make tm                       # stages 1-2: ~12 GB streamed once, multi-hour
make sec crosswalk            # stage 5
```

`make tm` is the corpus build and is the target to trust; the `analysis` and `paper` targets in the `Makefile` are stale. The theme fit and a full rescore take roughly an hour per resolution on the full corpus. Most analysis scripts run in minutes once the parquets exist.

## Where each exhibit comes from

Run everything with the default `SURPRISE_SRC=rolling`. "Typed" means the numbers were typed into the `.tex` from the artifact; the rest are generated files the papers read directly.

### Paper A

| Exhibit | Script | Artifact (in `paper/results/`) |
|---|---|---|
| Funnel table (Table 2) | `topic_debut.py`, `event_gates_all.py` | `debut_outcome_topic.json`, `event_gates_all.json` (typed) |
| Why fifty themes (§3): coherence, seed stability, sample thinning, tokenizer | `topic_coherence.py`, `topic_seed_replicate_T50.py`, `sample_thinning.py`, `tokenizer_check.py` | `topic_coherence.json`, `topic_seed_replicate_T50.json`, `sample_thinning.json`, `tokenizer_check.json` |
| Measure facts on the production build (§3) | `measure_facts_production.py` | `measure_facts_production.json` |
| Registration deciles (Fig. 1) | `fig_registration_only.py` | `fig_registration_deciles.png`, `fig_registration_only.json` |
| Five-year-proof LPM (Table 3) | `gate_decisive_regression.py` | `gate_decisive_regression.json` (typed) |
| Timing / settledness checks | `gate_censoring_check.py` | `gate_duration.json` |
| Three scorings (Fig. 2) | `resolution_compare.py`, `fig_resolution_compare.py`, `perclass_lda_rescore.py` | `resolution_compare.{json,tex}`, `fig_resolution_compare.png` |
| Theme arrival (new to class / corpus) | `theme_novelty_origin.py` | `theme_novelty_origin.json` |
| Lead within atypicality fifths | `ceiling_check.py` | `ceiling_check.json` |
| Patents | `wsC_within_firm_patents.py`, `patent_complementarity_by_sector.py` | `wsC_within_firm_patents.json` (typed) |
| SEC reporting (Table 4) | `debut_edgar_substantiate.py` | `debut_edgar_substantiate.json` (typed) |
| Robustness summary (Table 5) and supplement §S4 | `scoring_robustness` runs, `topic_resolution_sweep.py`, `topic_seed_gate.py`, `decay_gate_check.py`, `window_choice_all.py`, `window_mix_rolling.py`, `registration_and_unconditional.py`, `build_dup_flags.py`, `variants_runner.py`, `gate_curve_shapes.py`, `decompose_L_shape.py` | the matching `.json` files; `paper/v3/_eval/*.json` |
| Supplement §S1 (registration detail, flow, refiling) | `quintile_profiles.py`, `sankey_registration.py`, `refile_text_change.py`, `refile_prepost_chart.py` | `quintile_profiles.png`, `fig_registration_sankey.png`, `refile_prepost.png` |
| Supplement §S2 (representation, exhibit, pairs) | `representation_appendix.py`, `exhibit_encodings.py`, `pair_lead_corpus.py`, `combination_measures.py` | `representation_appendix.png`, `exhibit_encodings.json`, `pair_lead_corpus.{json,tex}` |
| Supplement §S3 (burn-in) | `burnin_optimization.py` | `burnin.png`, `burnin_by_class.json` |

### Paper B

| Exhibit | Script | Artifact (in `paper/results/`) |
|---|---|---|
| Five-year-proof LPM, forest | `gate_decisive_regression.py`, `event_gates_all.py` | `gate_decisive_regression.json`, `event_gate_forest.png` |
| Per-cohort coefficients | `fig_cohort_slopes.py` | `fig_cohort_slopes.{png,json}` |
| Technology vs other eras | `gate_era_profile.py` | `gate_era_profile.png` |
| Technology classes and themes | `gate_era_tech_themes.py` | `gate_era_tech_classes.tex`, `gate_era_tech_themes.{tex,png,json}` |
| What carried the 2000–2004 reversal | `reversal_theme_decomp.py`, `reversal_decomp_table.py` | `reversal_theme_decomp.json`, `reversal_decomp.tex` |
| Internet by class, scatter, event time, cohort split | `internet_breakout.py`, `fig_internet_scatter.py`, `internet_convergence.py`, `fig_cohort_slopes_internet.py` | `internet_breakout.{json,tex}`, `fig_internet_scatter.png`, `internet_convergence.png`, `fig_cohort_slopes_internet.png` |
| Surges | `theme_surge.py`, `theme_surge_class.py`, `wave_timing.py`, `fig_surges.py` | `theme_surge{,_class}.json`, `wave_timing.json`, `fig_surges.png` |
| Ladder, unfunded listings, value tiers, curated vocabularies | `sec_event_ladder.py`, `unfunded_ipo.py`, `value_concentration.py`, `curated_ladder.py` | matching `.json` (ladder typed) |
| Harmonized staged outcomes (discussion) | `staged_outcomes_table.py` | `staged_outcomes_table.tex` (50 themes; the earlier 200-theme version is `staged_outcomes_table_T200.tex`) |

## Online appendix

Published from `docs/` at <https://aporia.institute/tm-vocabulary/>.

- `online-appendix/index.html` — per-class breakouts, the cross-industry forest and scatters, and `per_class_estimates.csv` (`scripts/online_appendix.py`).
- `online-appendix/themes_T50.html` and `themes_T50.csv` — the fifty production themes (`scripts/themes_t50_page.py`).
- `online-appendix/ipo-viewer/` — every class's registrations on the lead/atypicality plane (`scripts/build_ipo_viewer_data.py`).
- `online-appendix/themes/` — explorer for the 500-theme robustness model (`scripts/theme_explorer.py`, `theme_pages.py`).

## Known limitations

**The SEC crosswalk is normalized-exact-match only.** The committed crosswalk has 19,889 owner strings and zero fuzzy matches. Match rates favour formally constituted entities with stable legal names, and every listing and financing estimate is conditional on matchability. Only about half the matched CIKs carry an exchange ticker — the outcome is SEC *reporting*, not exchange listing.

**Word scoring measures description length.** Word-scored atypicality correlates with log distinct-term count at −0.64 within class (−0.65 in class 009); theme scoring takes that to between −0.10 and +0.10 (`measure_facts_production.json`). Do not build firm-performance claims on word-scored measures; Paper A shows the firm-level correlations they produce dissolve under theme scoring.

**Word-level scores have no per-filing-window equivalent for every analysis.** Some word-scored material (worked examples, phrase transit, era turbulence) is on the older calendar-year references, and two theme-scored analyses (the asymmetric-window decomposition, the response-latency split) likewise.

**Per-filing lead is noisy.** A refit under a second seed reproduces lead at r = 0.72 at 50 themes (0.79 at 200). The noise attenuates estimates toward zero; averaging scores over several refits reduces it.

**Filings are not independent draws.** Identical text carries identical scores (47% of registrations share their exact normalized description with another filing), and outcomes cluster within owner (within-owner correlation 0.38 at the five-year proof, mechanically 1.0 for listing). Proof regressions cluster on normalized owner; firm-level specifications carry one row per owner.

**Registration is selected on language.** The main results condition on grant; the unconditional version is reported separately and diverges.

**The measure has no momentum channel, and the theme basis looks ahead.** References are flat pooled aggregates, and the theme model is fitted on filings from 1990–2024, so a vintage refit is owed.

**Legacy names.** Many scripts still say "gate" for the five-year proof (`gate_decisive_regression.py`, `event_gates_all.py`, `gate_*` columns) and `dkl` for lead (`topic_dkl`). `scripts/migrate_kl_column_names.py` rewrites older panels into the current column names.

## Repository layout

```
.
├── paper/
│   ├── split/A_qss/        Paper A  ← start here for the corpus and measure
│   ├── split/B_lead/       Paper B  ← start here for the findings
│   ├── split/A_corpus/     longer version of Paper A (reference)
│   ├── results/            JSON metrics, figures, .tex fragments both papers read
│   ├── v3_rp/              shared preamble, back matter, bibliography (+ earlier manuscript)
│   └── v3/, frozen/, _legacy/, *.tex   earlier work, provenance only
├── docs/                   the published site and online appendix
├── scripts/                the analysis chain (see the exhibit tables)
├── src/novelty/            the Python package: dictionary, surprise, firm_year, survival
├── data_publish/           early word-scored firm-year panel (retired; see Data)
└── LICENSE                 GPL-3.0
```

## Citing

> Silver, E. (2026). *An event-dated corpus of US trademark prosecution and a two-sided measure of vocabulary position*. SSRN Working Paper 7520758. https://ssrn.com/abstract=7520758
>
> Silver, E. (2026). *Arrows in their backs: Vocabulary lead and product survival in the US trademark record*. Working paper.
>
> Earlier combined version: Silver, E. (2026). *Business themes in the trademark record: Language signals of product survival, funding, and listing*. SSRN Working Paper 6954598. https://doi.org/10.2139/ssrn.6954598

## Author

Eric Silver — `epsilver@gmail.com`, ORCID 0000-0003-3351-1109. Independent researcher. The author's current employment is unrelated to this work, and the views expressed are the author's alone.

## License

GPL-3.0 (see `LICENSE`).
