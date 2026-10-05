# Producer margins, filing novelty and segment competition

Sources: SEC Financial Statement Data Sets (sec_firm_year.parquet, fiscal years 2009+), USPTO trademark case files, rolling-window filing scores, 60 product segments (registrations 2002-2018). Owners are linked to SEC filers by normalized exact name (uspto_sec_crosswalk.parquet, re-keyed with norm_owner). Scripts: scripts/margins_novelty.py, scripts/margins_competition.py. Full output: margins_novelty.json, margins_competition.json.

## Test 1. Firms with unusual or early filings do not earn higher gross margins

Question: do firms whose trademark filings are atypical (A) or early (lead, L) earn higher gross margins?

Specification: gross margin in t (pp) on the firm's mean class x year percentile of A and of L over its filings in t-3..t (0-1 scale), log revenue, log filings; (a) SIC2 x year FE, (b) plus firm FE; firm-clustered SEs; t = 2009-2019 (scores end with filing year 2019).

| Outcome | Spec | Firms | Firm-years | Atypicality b (SE) | Lead b (SE) |
|---|---|---|---|---|---|
| Gross margin | (a) SIC2 x year | 2,811 | 12,332 | -5.80 (1.98) | -3.31 (1.97) |
| Gross margin | (b) + firm FE | 2,282 | 11,799 | +2.31 (1.94) | -0.99 (1.65) |
| Operating margin | (a) SIC2 x year | 2,795 | 12,947 | +0.64 (1.70) | -0.92 (1.71) |
| Operating margin | (b) + firm FE | 2,293 | 12,436 | -3.90 (2.02) | -1.37 (1.74) |

Checks. Extending t to 2022 (windows partly unscored) gives gross-margin coefficients of A -5.30 (1.85) and L -4.26 (1.79) in (a), and A +3.96 (1.78) and L +0.36 (1.52) in (b). Restricting to firm-years with 5 or more filings (1,599 firms, 6,866 firm-years) gives A -3.09 (3.12) and L -9.00 (3.15) in (a), and A +6.76 (3.52) and L -0.86 (2.53) in (b). Raw gross margin by quintile falls from 42.3% to 37.2% across atypicality quintiles and from 43.0% to 38.9% across lead quintiles.

Scale. Portfolio percentiles are averages over a median of 6 filings, so they do not span 0-1: the 10th-90th percentile range is 0.25-0.78 for A and 0.20-0.71 for L. The within-firm SD is 0.09 for A and 0.11 for L. A realistic p10-p90 contrast is therefore about half the tabled coefficient, and a typical within-firm change is about a tenth of it.

Plain reading: within an industry and year, firms that file atypical or early marks have gross margins about 3 pp lower (p10 vs p90). Within a firm over time, changes in filing novelty do not move margins in a consistent direction. Caveats: the sample covers only owners whose filing name matches an SEC registrant (6,098 CIKs carry scored filings; IP-holding-company filers are missed). Conglomerate margins average across many product lines. Gross margin also depends on how each industry books cost of goods sold, which the SIC2 x year effects absorb only at the 2-digit level.

## Test 2. Segment competition measures do not predict producer margins; survival runs opposite to margins

Question: do segments with more competition have lower producer margins?

Specification: firm gross margin (mean over FY 2009-2023) is spread over segments by the firm's share of 2002-2018 registrations in each segment. Segment margin is the share-weighted mean (unweighted, and also weighted by share x revenue). This margin is related to standardized segment scores by Spearman correlation and by WLS with weights = effective firms and HC1 SEs. All 60 segments have 15 or more matched firms (3,254 firms in total).

Segment margins range from 19.4% (processed foods + beverages) to 61.4% (pharmaceutical preparations), with an SD of 9.6 pp. Coefficients are pp of unweighted margin per SD of the score.

| Score | Spearman | WLS b/SD (SE) | Rev-weighted Spearman |
|---|---|---|---|
| Record: rivalry, log owners | +0.15 | +0.93 (1.84) | +0.12 |
| Record: rivalry, HHI | -0.12 | +2.38 (2.09) | -0.02 |
| Record: entry, debut share | +0.30 | -0.94 (2.08) | +0.10 |
| Record: entry, owner growth | +0.28 | +0.92 (1.67) | +0.16 |
| Record: substitutes, centroid count | +0.43 | +2.78 (2.12) | +0.37 |
| Record: substitutes, nn3 distance | -0.55 | -3.76 (1.32) | -0.44 |
| Record: buyer power (channel wording) | +0.27 | +0.05 (1.34) | +0.02 |
| Record: platform power | +0.59 | +4.13 (0.99) | +0.46 |
| Record: established-owner survival gap | -0.16 | -4.47 (1.13) | -0.25 |
| Record: regulated share | +0.15 | +4.22 (1.36) | +0.12 |
| Rating: entry threat | +0.07 | -1.26 (1.83) | -0.09 |
| Rating: rivalry | +0.06 | +0.06 (1.48) | -0.04 |
| Rating: substitutes | +0.21 | -0.18 (1.73) | +0.11 |
| Rating: buyer power | -0.23 | -2.87 (1.30) | -0.18 |
| Rating: supplier power | +0.03 | -0.17 (1.53) | +0.00 |
| Rating: scale economies | +0.07 | +1.67 (1.48) | +0.12 |
| Rating: imitability | -0.06 | -2.41 (1.60) | -0.22 |
| Rating: complementary assets | +0.02 | +1.34 (2.01) | +0.18 |
| Rating: incumbents hold assets | +0.05 | +1.53 (2.02) | +0.14 |
| Five-year mark survival | -0.54 | -5.37 (1.28) | -0.44 |

Modal-assignment check (each firm counted only in its largest segment; 47 segments with 15+ firms): survival -0.58, rated rivalry +0.10, rated buyer power -0.12, record rivalry (owners) +0.17.

The highest-margin segments are pharmaceuticals (61%), downloadable media (55%) and software/information services (45-55%). Five-year mark survival in these segments is 37-48%. The lowest-margin segments are metal and building hardware (21%), construction and engineering services (26%), plastics and packaging (26%), vehicles (27%) and processed foods (19-28%). Survival there is 51-59%. The per-segment table, including the three largest revenue contributors, is in margins_competition.json under test2_segments.segments.

Plain reading: neither the record-based nor the blind-rated rivalry and entry measures relate to segment gross margins. Rated buyer power is the only competitive force with a modest negative association, at about -3 pp per SD. Segments where marks survive longest have the lowest producer margins. Caveats: there are only 60 segments, and they are labeled for registrations rather than revenue. Segment margins are averages of whole-firm margins, so a conglomerate's margin is spread over every segment it files in. The 3 largest firms carry 15-73% of the revenue weight in each segment. Cross-industry differences in what counts as cost of goods sold (software and pharmaceuticals versus manufacturing) probably account for much of the margin ranking, including the negative relation with survival.

## Test 3. SEC matching reaches almost none of the powder-coating formulators

The regex covers classes 001, 002 and 040 and finds 991 powder-coating filings from 537 owners. Of these, 18 filings from 10 owners match an SEC filer, and 8 of those firms have gross margins.

| Firm | SIC | Group | Powder filings (years) | Median gross margin |
|---|---|---|---|---|
| Valmont Industries | 3440 | other (fabricated metal; runs a galvanizing/coatings segment) | 4 (2002-2016) | 27.6% |
| Ferro | 2851 | formulator | 3 (1995-1998) | 26.3% |
| Nordson | 3569 | equipment (application systems) | 2 (2001-2009) | 54.8% |
| Eastman Chemical | 2821 | other chemicals (resins) | 2 (1994) | 23.3% |
| Covia Holdings | 1400 | other (minerals) | 2 (2017) | -40.5% (1 year) |
| AZZ | 3470 | coating services | 1 (2015) | 27.4% |
| Valspar | 2851 | formulator | 1 (1994) | 32.4% |
| Stepan | 2840 | other chemicals | 1 (1989) | 17.5% |
| Smith International, Linde | - | no gross margin in the SEC extract | 1 each | - |

Two of the 8 firms with margins (25%) are formulators (SIC 2851), and both qualify only through filings from the 1990s. The equipment maker Nordson has the highest margin, at 55%, against 26-32% for the formulators. Large formulators are missed for the following reasons:

- **AkzoNobel** (36 powder filings, as Akzo Nobel N.V. and Akzo Nobel Coatings International B.V.) does not file financial statements with the SEC.
- **Tiger Coatings** (9 filings) is a private Austrian firm.
- **PPG** (14 filings) files as PPG Industries Ohio, Inc., an IP-holding subsidiary. The crosswalk links only "PPG Industries, Inc."
- **Sherwin-Williams** (13 filings including Valspar) files as SWIMC LLC and as Valspar Sourcing / Valspar Solutions. Only one 1994 Valspar Corporation filing matches. Sherwin-Williams' own SEC SIC is 5200 (retail building materials), so it would not be classified as a formulator even if matched.
- **Axalta** (4 filings) files as Axalta Coating Systems IP Co. LLC. The predecessor's filing under E. I. du Pont de Nemours and Company is also unlinked, because DuPont is absent from the crosswalk.
- **RPM** (3 filings) files through its subsidiary TCI Powder Coatings / TCI, Inc.

Plain reading: name matching to SEC filers recovers less than 2% of powder-coating filings and almost none of the formulators that dominate the record, so this case cannot speak to formulator profits. The firms it does reach are equipment makers, resin and chemical suppliers, and coating-service firms whose margins reflect their whole business.
