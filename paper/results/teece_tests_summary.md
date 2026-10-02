# Teece tests: who profits when a new kind of offering appears

Common conventions: outcome = survival of the first maintenance deadline in points (100 × (1 − failed1)); lead = within-cell percentile of z centred at zero, so a coefficient is the survival difference between the most lagging and the most leading filing; controls = has_attorney, itu, log_len, log_owner_n, dom_us, dom_cn, basis_44e, basis_66a; class × registration-year fixed effects; standard errors clustered by owner_key. Frame: 3,104,534 registrations, 2002–2018. On the full frame this specification gives a lead coefficient of −1.62 (SE 0.20).

## 1. Product or firm (`teece_product_or_firm.json`)

**Question.** Is the cost of leading tied to the filing or to the firm that files it?

**Specification.** surv ~ lead + controls, first with class × year fixed effects only and then with owner fixed effects added. Both sets of fixed effects are absorbed jointly by alternating projections, which converged in 125–140 sweeps with coefficient changes below 1e-4.

| Sample | n | Owners | Class×year FE | + Owner FE |
|---|---|---|---|---|
| Owners with 2+ registrations | 2,271,592 | 435,235 | −1.22 (0.25) | −0.82 (0.21) |
| Owners with 5+ registrations | 1,442,078 | 101,830 | −0.60 (0.34) | −0.87 (0.25) |


**Within-owner pairs.** 144,202 owners hold at least one leading registration (top two fifths) and one lagging registration (bottom two fifths). The equal-weighted difference in their mean survival is +0.50 (0.12) raw and +0.24 (0.12) after subtracting each cell's mean; with harmonic-count weights it is −0.00 (0.13). Run as a regression on those owners' leading and lagging rows with owner and class × year fixed effects, the difference is −0.30 (0.13) without controls and −0.36 (0.13) with controls.

**Reading.** Within the same firm, a leading registration survives about 0.8–0.9 points less than a lagging one across the full lead range. That is roughly half the pooled −1.62, so part of the cost belongs to the filing and part reflects which firms lead. Simple paired averages come out near zero or slightly positive because they do not hold the class-year fixed, so the owner-fixed-effects regression is the estimate to use. It still compares filings a firm chose to make, which is a selection on the firm's own portfolio.

## 2. Crowding dose (`teece_crowding_dose.json`)

**Question.** Does the entry that actually followed a filing account for the cost of leading?

**Specification.** surv ~ entry (per SD), surv ~ lead, and surv ~ lead + entry, each with controls and class × year fixed effects. Entry is measured four ways:
- (a) entry_share = log((s_post + 0.001) / (s_pre + 0.001));
- (b) entry_owners = log(1 + number of owners who first entered the class in fy+1..fy+5 with debut filings in the theme);
- growth version of (b): post minus pre;
- (a) with eps = 0.005 as a check.

| Entry measure | Corr. with lead (within cell) | Entry alone | Lead, with entry held | Entry, with lead held | Lead shrinks by |
|---|---|---|---|---|---|
| entry_share | 0.56 | −0.57 (0.05) | −0.77 (0.23) | −0.44 (0.05) | 53% |
| entry_owners (level) | 0.09 | −1.50 (0.09) | −1.32 (0.20) | −1.45 (0.09) | 19% |
| entry_owners growth | 0.50 | −0.71 (0.08) | −1.15 (0.24) | −0.45 (0.10) | 29% |
| entry_share, eps = 0.005 | 0.57 | −0.55 (0.05) | −0.81 (0.23) | −0.42 (0.05) | 50% |

With entry_share and entry_owners both entered, lead falls to −0.56 (0.23).

**Deciles of realized entry, cell-adjusted survival.**
- entry_share, deciles 1 to 10: 48.1, 48.2, 48.0, 47.7, 47.4, 47.3, 47.8, 48.7, 47.8, 45.9. The curve is flat until the top decile, where survival drops about 2 points.
- entry_owners, deciles 1 to 10: 49.1 falling to about 46–47 by deciles 7–10.

**Reading.** Holding realized entry fixed cuts the lead penalty by about a fifth to a half. The penalty is concentrated in the top tenth of themes, where the theme's share surged and survival falls about 2 points. However, entry_share and lead both measure the class moving toward the filing's theme (within-cell r = 0.56), so part of the "mediation" is mechanical. The level count of new owners mostly measures theme size: it hurts survival but barely overlaps with lead.

## 3. Who collects (`teece_who_collects.json`)

**Question.** In a theme surge, who ends up holding the marks still in use: pioneers, followers, late entrants or incumbents?

**Specification.** A surge is a class × theme whose share in t0+1..t0+5 is at least 1.5 times its share in t0−5..t0−1, with at least 200 filings in the surge years. Episodes are the first qualifying year of each run, spaced at least 10 years apart, with onsets in 1998–2008: 162 episodes with a median share ratio of 1.70 (491 episodes across all onset years).
- Timing groups: pioneers k = 1–2, followers k = 3–5, late k = 6–10, pre k = −4..0.
- Incumbents: owners with 25+ earlier filings, at any timing.
- Survival is cell-adjusted (survival minus the class-year mean, plus the frame mean of 47.7).

| Group | n | Cell-adj. survival | Mean lead | Acquired | Bought by established firm (25+ marks) |
|---|---|---|---|---|---|
| Pre-surge | 19,128 | 50.3 | +0.25 | 2.14% | 0.63% |
| Pioneers | 19,378 | 49.4 | +0.36 | 1.40% | 0.37% |
| Followers | 46,163 | 47.0 | +0.28 | 1.09% | 0.34% |
| Late | 105,217 | 45.7 | +0.09 | 0.65% | 0.20% |
| Incumbents | 31,856 | 54.0 | +0.17 | 1.64% | 0.73% |
| Non-incumbents | 158,030 | 45.4 | +0.18 | 0.85% | 0.21% |

Frame-wide, 0.99% of registrations were acquired and 0.31% were bought by an established firm.

**Share of marks still in use against share of registrations.**
- Among surge entrants, pooled: pioneers 12.4% vs 11.3%; followers 27.7% vs 27.0%; late 59.8% vs 61.6%; incumbents 19.4% vs 16.6%.
- Averaged over 141 episodes: pioneers 12.6% vs 11.9%; incumbents 19.0% vs 16.7%.
- End-of-surge stock (filed by t0+5): pioneers 23.8% vs 22.9%; incumbents 19.9% vs 17.8%.

**Regression against pioneers** (class × year fixed effects; controls without log_owner_n, which is nearly collinear with incumbent status): incumbents +6.2 (0.7). Timing coefficients flip sign inside this sample (late +2.8, 0.7) because the fixed effects compare late filers of earlier episodes with pioneers of later episodes in the same class-year. With log_owner_n included, the incumbent coefficient becomes −15.3, a collinearity artefact.

**Reading.** In surging themes, pioneers survive slightly better than followers and late entrants relative to their class-year peers, about 3.7 points above late entrants. Incumbents survive best (+8.6 points over non-incumbents) and hold a larger share of the marks still in use than of the filings: 19–20% against 17–18%. Marks from earlier groups are bought more often, and established buyers take incumbents' marks at about twice the rate they take entrants'. Caveats:
- Acquisition rates are not adjusted for time at risk: earlier filings have had longer to be bought.
- Incumbent status overlaps with firm size.
- Timing comparisons depend on how cells are drawn, as the regression shows.

## 4. Segment scores (`teece_segment_scores.json`, `data/processed/segment_scores.parquet`)

**Question.** Do record-based industry scores (five forces, scale, Teece) line up with segment survival and with the segment's lead effect?

**Specification.** Scores come from 5-year pre-windows (t−5..t−1) for filing-year cohorts 2006–2017 and are averaged over cohorts with cohort-size weights. Segment labels exist only for registrations, so the scores describe registered filings 2001+.
- Substitutes = number of other segment centroids within a Hellinger distance of 0.734, the 10th percentile of centroid pairs.
- Outcomes: mean survival relative to the class-year mean, with an owner-clustered SE; segment lead effect from the Design model with controls.
- Validation is WLS on 60 segments with standardized scores, HC1 SEs, and weights of 1/se².

The 60 segment lead effects average −1.82 when precision-weighted and are heterogeneous: Q = 227 on 59 df, τ ≈ 2.4 points.

**(a) Segment survival on force scores, one at a time, per SD** (weighted; unweighted in brackets):

| Score | Coefficient |
|---|---|
| entry_debut | −0.59 (0.53) [−0.64 (0.51)] |
| entry_owner_growth | −1.57 (0.54) [−1.50 (0.38)] |
| rivalry_owners | −0.21 (0.47) |
| rivalry_hhi | −0.17 (0.89) |
| substitutes | −0.56 (0.56) |
| distance to the 3 nearest centroids | +0.49 (0.52) [+1.03 (0.42)] |
| buyer_power | −0.68 (0.39) |
| platform_power | −1.81 (0.27) [−1.53 (0.32)] |

Joint model (R² 0.51): owner growth −1.21 (0.42), HHI −2.19 (1.07), buyer −0.92 (0.44), platform −1.49 (0.31); the other scores are not distinguishable from zero.

**(b) Segment lead effect on Teece and scale scores, alone, per SD** (weighted; unweighted in brackets):

| Score | Coefficient |
|---|---|
| no_counsel_share | −0.43 (0.38) [−0.95 (0.36)] |
| no_patent_share | −0.51 (0.30) [−0.78 (0.33)] |
| debut share | −0.95 (0.39) [−1.22 (0.34)] |
| incumbent_share | +0.91 (0.39) [+0.96 (0.37)] |
| regulated_share | +0.15 (0.45) |
| hhi_in_use | +0.00 (0.40) |
| hhi_in_use_trend | +1.67 (0.76) |
| est_gap | −0.45 (0.37) |

Joint model (R² 0.30): no single coefficient is clearly nonzero. Debut share and incumbent share are close mirror images of each other.

**Reading.** Segments whose pre-window had fast owner growth and many platform-dependent descriptions have lower survival. Where filers are more often debut, unrepresented or patentless (easy to imitate), and incumbents are scarce, leading costs about 1 point more per SD. This is consistent with Teece's prediction that innovators lose where imitation is easy and complementary assets are not needed. With 60 segments, correlated scores and one-at-a-time tests, these are descriptive patterns rather than estimates. Platform share is concentrated in a few software segments, and the hhi_in_use_trend result rests on a handful of segments.

## 5. Spillover (`teece_spillover.json`)

**Question.** Do failed pioneers supply language that survivors later use?

**Specification.**
- Within each class, take bigrams (two-word sequences of lowercased [a-z]+ words) that first appear in year t (2002–2015) after a scan of the full filing record, and that are used by at least 20 later filings.
- First filers' outcomes come from the frame; 35% of first filings never entered the frame, meaning they were not registered in 2002–2018.
- This gives 636,131 bigrams with a registered first filer.

**Who introduces new language.**
- The bigram-mean failure rate of first filers is 52.1%, against a class-year baseline of 51.3%. Weighted by filings it is 51.40% against 51.42%.
- For bigrams with a single introducer (504,761), 51.8% were introduced by a filing that later failed, against a 51.2% class-year failure rate.

**Who uses it later** (registrations weighted, against the class baseline at the users' mean filing year):
- All bigrams: later users survive 45.9% against a 46.8% baseline.
- Introducer failed: later users survive 44.6% against 46.6%.
- Introducer survived: later users survive 47.6% against 46.9%.

**Reading.** New language enters through failed and surviving filings at the base failure rate. Firms that adopt language first used by a failed filing also do worse (about −2 points), so the pattern is a shared fate rather than a transfer of value from failed pioneers to survivors. Still, about 45% of those later users' registrations survive, so failed pioneers' words do end up in many surviving registrations. Caveats:
- "New bigram" includes rewordings as well as new product terms.
- A later filing counts once for each bigram it uses.
- The later-user baseline is matched only approximately, by mean filing year.
