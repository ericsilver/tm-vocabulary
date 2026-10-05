# Trademark entry and producer prices (BLS PPI)

## Entry surges are not followed by lower producer-price inflation

Across 41 Nice classes with a good or partial PPI match (1990-2022), growth in trademark filings or first-time filers in year t has no detectable association with PPI inflation over the next three years. A one-standard-deviation entry surge (18 log points in filings) moves average annual inflation over t+1..t+3 by +0.07 percentage points (p = 0.09). The residual standard deviation of annual PPI inflation after industry and year effects is 3.3 points, so the estimate is small as well as statistically weak. The point estimate is slightly positive, which is the opposite of the sign that "entry lowers prices" would predict. Prices do not predict entry either. Powder coatings, the case study, show the same result. The broad paint index grows a little more slowly in the year after powder-coating filings rise. The OEM-finishes cell that contains powder coatings shows no such effect, and neither paint series shows a difference over three years.

## Data sources

- Producer Price Index, BLS flat files at download.bls.gov/pub/time.series/: industry database `pc` (NAICS-based; pc.series, pc.industry, pc.product and the 60 sector data files), commodity database `wp` (wp.series, wp.item and the sector data files). Pulled 2026-10-05, data through 2026-08. Annual values are BLS annual averages (M13), or the mean of 12 monthly values where M13 is not published. 2026 is excluded. The `pc.data.0.Current` file starts in 1998, so full histories come from the sector files.
- Deflator for the powder-coatings case: CPI-U all items, US city average, NSA (`CUUR0000SA0`, cu.data.1.AllItems).
- Trademark filings: `data/processed/tm_class{cls}.parquet` (USPTO case files; serial_number, filing_date, owner_name, goods_services). Product segments: `data/processed/theme_segments.parquet` (registrations 2002-2018).
- Raw downloads: `data/raw/bls/`. Annual PPI extract: `data/processed/bls_ppi_annual.parquet`.

## Concordance coverage: 14 classes good, 27 partial, 4 poor

`bls_concordance.csv` / `.md` map each class to one primary series, with alternates. Thirty-five primary series run from 1995 or earlier through 2024.

| Fit | Classes |
|---|---|
| good (14) | 001 WPU061, 002 PCU325510325510, 003 PCU3256--3256--, 005 PCU325412325412, 008 WPU1042, 010 PCU339112339112, 013 WPU1514, 014 WPU1594, 015 PCU339992339992, 020 PCU337---337---, 023 PCU3131--3131--, 025 WPU0381, 027 WPU1231, 034 PCU3122--3122-- |
| partial (27) | 004, 006, 007, 009 (from 2003), 011, 012, 016, 017, 019, 021, 022, 024, 026, 028-033, 035, 036, 038, 039, 040, 042 (from 2000), 044, 045 |
| poor (4) | 018 leather goods (no live luggage/handbag PPI), 037 construction (nonresidential repair only, from 2009), 041 education/entertainment (no PPI), 043 restaurants (no food-service PPI; accommodation only) |

None of the services classes (35-45) is graded good. Where a services class does have a series, it covers only part of what that class sells: advertising agencies for class 35, property and casualty insurance for 36, wired telecom for 38, air transport for 39, metal coating for 40, data processing and hosting for 42, physicians' offices for 44 and law offices for 45. Segments: 57 distinct segment names cover the 60 segment ids. Of these, 8 are good, 22 partial and 27 poor; the poor group includes every mixed two-theme segment.

Paint and coatings: PCU325510325510 (paint and coating manufacturing, NAICS 325510, 1984-2025) and WPU0621 (prepared paint, 1926-). **No PPI series prices powder coatings.** The only series in pc/wp that mentions them is "Transportation finishes, except powdered and high-solids coatings" (PCU32551032551041, from 2012), and it excludes them. The nearest long series is OEM product finishes excluding marine, WPU06210201 (1983-). Since 2012 BLS has split that series, and powder coatings fall in "all other OEM product finishes" (PCU32551032551042). The class-40 counterpart, powder-coating job shops, falls under PCU332812332812, metal coating and nonprecious engraving (1984-).

## Panel results: forward and reverse timing are both near zero

Model: inflation_{i,t+h} = b x entry growth_{i,t} + industry FE + year FE. Year effects absorb economy-wide inflation, so nominal and real give the same slope. SEs are clustered by industry. Entry growth is the log change in filings, or in first-time filers (owners with no earlier filing in that class).

**Class panel, 41 industries, t = 1990-2022, n ≈ 1,285**

| Outcome | b (filings) | SE | p | b (first-time filers) | SE | p |
|---|---|---|---|---|---|---|
| inflation t+1 | -0.0041 | 0.0067 | 0.54 | -0.0013 | 0.0072 | 0.86 |
| inflation t+2 | +0.0060 | 0.0044 | 0.17 | +0.0050 | 0.0062 | 0.42 |
| inflation t+3 | +0.0085 | 0.0059 | 0.15 | +0.0104 | 0.0059 | 0.08 |
| avg t+1..t+3 | +0.0035 | 0.0021 | 0.09 | +0.0047 | 0.0027 | 0.08 |

The SD of entry growth is 0.18 (filings) and 0.19 (first-time filers). A 1-SD surge therefore implies between -0.08 and +0.20 points of annual inflation.

- Controlling for current inflation and lagged entry growth, the 3-year-average slopes fall to +0.0026 (p = 0.27) and +0.0032 (p = 0.18). Winsorizing entry growth at the 1st/99th percentiles gives +0.0061 (p = 0.12) and +0.0088 (p = 0.10).
- Good-fit classes only (14 industries, n = 462): t+1 is -0.051 (p = 0.04), t+3 is +0.041 (p = 0.01), and the 3-year average is -0.007 (p = 0.45). The sign flips between horizons, and the average is zero.
- Goods classes 001-034 only (33 industries, n = 1,074): the 3-year average is -0.002 (p = 0.72).
- Segment panel (30 single-theme segments, filing years 2003-2016, n = 383-412): every forward slope has p > 0.14. The 3-year average is -0.004 (p = 0.65) for filings and +0.006 (p = 0.56) for first-time filers.

**Reverse timing (entry growth_t on inflation_{t-h}), class panel:** h = 1: +0.052 (p = 0.66); h = 2: +0.011 (p = 0.93); h = 3: +0.081 (p = 0.23). First-time filers give the same picture, with every p above 0.5. The segment panel is also null (p > 0.66). Neither direction of timing carries a reliable association.

## Powder coatings: little or no price decline after filing surges

Filings matching `powder[- ]coat|powder paint|coating powder` in classes 001, 002 and 040: 846 filings and 443 first-time filers over 1995-2024. Annual filings rose from 13-26 in 1995-2003 to 45-58 in 2019-2024. Adding "powdercoat" without a space contributes 3 more filings. The year-by-year table is in `bls_powder_coatings.csv`, and the `table` field of the JSON has the same rows.

Real price growth over 1995-2024, deflated by CPI-U: paint and coating manufacturing +34%, OEM finishes +1%, metal coating services -15%.

Surge years are those with filing growth in the top quartile (at least 33 log points): 1998, 2001, 2009, 2011, 2013, 2014, 2016 and 2019. Mean real price growth after surge years compared with other years:

| Series | Next year: surge / other (pp) | p | Avg next 3 yrs: surge / other (pp) | p |
|---|---|---|---|---|
| Paint and coating mfg (PCU325510325510) | 0.1 / 1.5 | 0.24 | 0.8 / 1.4 | 0.56 |
| OEM finishes (WPU06210201) | -0.1 / 0.1 | 0.85 | -0.0 / 0.3 | 0.76 |
| Metal coating services (PCU332812332812) | -0.5 / -0.4 | 0.93 | -0.4 / -0.4 | 0.99 |

The table uses Welch t-tests, with 8 surge years against 20-22 other years. First-time-filer surges give similar results.

Time-series slopes (real growth in t+1 on filing growth in t, HAC SEs, n = 30):

- Paint and coating manufacturing: -0.029 (p = 0.05). On first-time-filer growth: -0.023 (p = 0.03).
- Dropping entry years 2008 and 2021 removes the 2009 and 2022 input-cost shocks. The filings slope then falls to -0.011, with a rank correlation of -0.21 (p = 0.29). The first-time-filer slope is -0.013, with a rank correlation of -0.36 (p = 0.06).
- OEM finishes, the cell that contains powder coatings: -0.015 (p = 0.23), and 0.001 (p = 0.90) without 2008 and 2021.
- All three series show no 3-year-average effect (p ≥ 0.22).
- Reverse timing (filing growth on lagged real paint inflation) is null (p ≥ 0.66).

## Reading

The trademark record gives no evidence that entry surges pass through to lower producer prices within three years, either across industries or in powder coatings. The estimates are precise enough to rule out large effects. In the class panel, the upper end of the 95% interval for the 3-year average is a decline of about 0.01 points of annual inflation per 1-SD surge. The powder-coatings data allow only a weak, single-year negative correlation in the broad paint index. That correlation does not appear in the series closest to powder coatings, and it does not last beyond one year.

Caveats:

1. The results are descriptive. Entry and prices both respond to demand, input costs and regulation. Year effects remove only the economy-wide part.
2. Industry PPIs price whole NAICS industries. A new kind of offering is usually a small share of its mapped industry, so any price effect is diluted, and the services classes, where much new-category entry happens, are mapped only partially or poorly.
3. PPIs are producer prices, net of quality adjustment. Buyer gains that come as new variety or quality, rather than lower list prices, are not measured.
4. First-time filers are identified by normalized owner names, so name changes and assignments add noise.
5. Powder-coating filings average about 28 a year, so year-to-year growth is noisy. With 30 years and many specifications, an isolated p ≈ 0.05 is consistent with chance.
6. The segment panel covers only filing years 2003-2016, from registrations, so it misses abandoned applications.

## Files

- `scripts/bls_concordance.py` writes `paper/results/bls_concordance.csv`, `paper/results/bls_concordance.md` and `data/processed/bls_ppi_annual.parquet`.
- `scripts/bls_entry_prices.py` writes `paper/results/bls_entry_prices.json`, `paper/results/bls_entry_panel.csv`, `paper/results/bls_entry_panel_segments.csv` and `paper/results/bls_powder_coatings.csv`.
