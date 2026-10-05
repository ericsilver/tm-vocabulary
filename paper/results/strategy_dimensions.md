# Product and market characteristics in trademark descriptions, and how they relate to survival and the cost of leading

Registrations 2002-2018, one row per serial (N = 3,104,534). Survival is passing the five-year declaration of use (100 = passed; mean 47.7). Lead is the registration's within-cell percentile of the lead score, centred at zero, so its coefficient is the survival difference in percentage points between the most leading and the most lagging registration of the same class and registration year. With class x registration-year fixed effects, owner-clustered standard errors and the controls (counsel, intent-to-use, log description length, log owner filing count, US and China domicile, 44(e) and 66(a) bases), the lead effect is -1.6 pp (SE 0.2).

Results: `strategy_dimensions_tests.json`. References: `strategy_dimensions_refs.md` (keys below). Lexicons: `scripts/strategy_lexicons.py`. Estimation: `scripts/strategy_dimensions.py`.

## Summary

- Of 18 characteristics the literature treats as decisive, 12 can be read from goods/services text or the Nice class, with lexicon precision of 83-100% in hand-read samples. Six are identified only in part, because descriptions state them only sometimes: buyer type (B2B or B2C), customization, switching costs, advertising intensity, fads and luxury. Within geographic scope, local site-based offerings are identified; national reach and transport costs only in part. Discount positioning is effectively absent from descriptions (0.1% of registrations).
- Four characteristics make leading markedly more costly. Platform or network offerings: the lead effect is -6.4 pp against -1.4 pp for other registrations (interaction -5.0, SE 0.8). Complements to another product: -5.3 against -1.5 (interaction -3.8, SE 0.9). Digital goods: -4.6 against -1.4 (interaction -3.2, SE 0.5). Electronics and IT classes 9 and 42: -5.2 against -0.9 (interaction -4.3, SE 0.4). Consumables add a smaller penalty (interaction -1.7, SE 0.5).
- Two characteristics reverse the sign of the lead effect. For mass-advertised packaged consumer goods, leading raises survival by 1.9 pp (interaction +3.7, SE 0.7). For bulky, low value-to-weight goods it raises survival by 4.6 pp (interaction +6.3, SE 1.9; 0.5% of registrations).
- Survival levels differ most for platforms (-9.4 pp), digital goods (-4.8), R&D-intensive technology (-4.2), local site-based offerings (+3.5), credence services (+2.9) and intermediaries (+2.5), all within class and year.
- Experience versus search goods, subscriptions, premarket approval read from text, fashion and novelty cues, and luxury positioning show no association with survival or with the lead effect after the Holm adjustment.

## 1. Characteristics the strategy and IO literatures link to value capture and entry order

The first-mover literature sets out the mechanisms (technological leadership, pre-emption of scarce assets, buyer switching costs) and the conditions under which they fail: free-riding by later entrants, technological and market uncertainty, and incumbent inertia (lieberman1988; lieberman1998; kerin1992). Survival-adjusted samples show that many pioneers fail (golder1993). Pioneering raises sales but can lower profits (boulding2003). Innovative late entrants can outsell pioneers (shankar1998). The pace of technology and market change conditions whether the early entrant's advantage persists (suarez2007). Shakeouts follow from innovation dynamics over the product life cycle (klepper1996, UNVERIFIED). The characteristics below are those this literature, and the IO work it draws on, treats as deciding who profits.

| Characteristic | What the literature links it to | Key references |
|---|---|---|
| Sold to consumers vs businesses | Market pioneers hold share advantages in both consumer and industrial goods, through different routes: consumer learning and brand preference versus buyer specifications and switching costs | robinson1985; robinson1988; schmalensee1982 (UNVERIFIED) |
| Geographic scope and transport costs | Transport costs split markets into local ones. The number of firms a local market supports rises with its size. Local markets keep low-productivity producers in business | bresnahan1991; syverson2004 |
| Goods vs services | Services are intangible, produced and consumed together, and perishable. Pioneering advantages differ between manufacturing and service industries | zeithaml1985; song1999 |
| Search, experience and credence goods | Product quality is judged by inspection (search), by use (experience), or not even after use (credence). This shapes advertising, reputation and the room for seller opportunism | nelson1970; nelson1974; darby1973; dulleck2006 |
| Durables vs consumables | A durable-goods monopolist competes with its own future sales. Repeat-purchase goods create switching costs | coase1972; bulow1982; klemperer1995 |
| Standardized vs customized | Before a dominant design, products are varied and customized; after it, competition turns to process and cost, and entrants that come after the dominant design survive less often | utterback1975; suarez1995 |
| Subscription and recurring contracts | Recurring relationships create switching costs: firms compete for new customers and harvest locked-in ones | klemperer1987; farrell2007 |
| Switching costs | As above; switching costs raise the value of early market share | klemperer1987; klemperer1995; farrell2007 |
| Network effects and platforms | Compatibility and installed base decide competition. Two-sided platforms set prices across sides. Network effects can strengthen or reverse first-mover advantages | katz1985 (UNVERIFIED); rochet2003; suarez2007; shapiro1999 |
| Complementary goods and assets | Owners of complementary assets, not the innovator, often capture the value of an innovation. Bottlenecks in complements decide when leading pays | teece1986; adner2010 |
| Regulatory approval | Approval delays entry and protects pioneer brands. Brand-name drugs keep share after generic entry | grabowski1992; scottmorton1999 |
| Brand and advertising intensity | In advertising-intensive industries, sunk advertising outlays escalate with market size and concentration stays high (endogenous sunk costs). Early-entrant brand shares persist for decades | sutton1991 (UNVERIFIED); schmalensee1982 (UNVERIFIED); bronnenberg2009 |
| Technology intensity | The appropriability regime and complementary assets decide whether innovators profit. Leading firms fail when new technologies serve customers they do not yet have | teece1986; christensen1996; cohen1990 |
| Information and digital goods | Near-zero marginal cost, bundling and lock-in shape pricing and competition | shapiro1999; bakos1999 |
| Intermediated vs direct channels | Intermediaries set prices and match buyers and sellers. Internet channels change price levels and dispersion | spulber1996; brynjolfsson2000 |
| Perishability | Perishable inventory is priced dynamically. Perishability also limits the reach of a market | gallego1994; syverson2004 |
| Fashion and fad dynamics | Informational cascades produce fads. Fashion cycles follow from design innovation by firms | bikhchandani1992; pesendorfer1995 |
| Luxury vs mass | Conspicuous goods carry status value. Social effects change the pricing of luxury goods | bagwell1996; amaldoss2005 |

UNVERIFIED marks the four references that neither Crossref nor Semantic Scholar would confirm on 2026-10-05: Sutton 1991 (book), and Katz and Shapiro 1985, Schmalensee 1982 and Klepper 1996 (pre-DOI American Economic Review articles; Semantic Scholar returned HTTP 429). The other 41 are verified; see the refs file.

## 2. Which characteristics a goods/services description identifies

The description states what is offered. It identifies product category, delivery mode and some contract forms well. It identifies the buyer, geographic reach, price position and intensity measures (switching costs, network size, advertising spending) only when the applicant writes them. Most applicants do not.

**Identified by the Nice class.** Goods versus services (classes 35-45). Pharmaceuticals and medical devices (5, 10). Packaged consumer goods (3, 29-33). Fashion goods (14, 18, 25). Electronics, software and IT services (9, 42). Fresh and processed foods (29-31). Under class x year fixed effects these are constant within a cell. Their survival differences below therefore come from a registration-year-FE model and are descriptive. Their interaction with lead is identified.

**Identified by text** (`scripts/strategy_lexicons.py`). Each lexicon is a regular expression over the lowercased description. A second expression lists known false-positive phrases ("platform shoes", "spas, hot tubs", "non-medicated", "software for ..." clauses), and these are blanked out before matching. Two lexicons (bulky, perishable) require the material or food to head a list item, so that "coatings for concrete" or "fish sauce" do not count. Four are restricted to goods classes 1-34 (durable, consumable, approval, advertising-intensive), and two to the classes that sell the goods named (bulky: 1, 4, 19, 20, 31, 32; perishable: 29-32, 35, 43).

**Precision.** For each lexicon, 30 random matches and 30 random non-matches were read from a uniform sample of 40,000 frame registrations (`data/processed/strategy_review_sample.parquet`; reproducible with `strategy_dimensions.py review <key> m|n <seed>`). Each lexicon was revised until its false positives had no common pattern left. The counts below are from the last reading; some of the false positives it found were excluded afterwards, so the counts are conservative. "Misses" counts non-matches that plainly have the characteristic. With 30 draws, a count of 0 bounds the miss rate below about 10% of non-matches and says little about recall for rare characteristics.

| Lexicon | Precision | Misses in 30 non-matches | False positives in the last reading |
|---|---|---|---|
| B2B | 28/30 (93%) | 7 | "addresses for businesses" in map data |
| B2C | 27/30 (90%) | 10 | fulfilment or analytics services about consumers |
| Local, site-based | 26/30 (87%) | 3 | goods named for a site (barber smocks), in-room hotel networks |
| National/international trade | 26/30 (87%) | 0 | publications in the field of international trade |
| Bulky goods | 29/30 (97%) | 0 | chemicals listed for use on concrete |
| Credence | 27/30 (90%) | 1 | education of financial advisors; repair tools |
| Experience | 29/30 (97%) | 3 | cosmetic bags |
| Search | 26/30 (87%) | 1 | medical belts; cleaning of carpets |
| Durable | 25/30 (83%) | 4 | inputs named after a durable ("printing ink vehicles", "appliance cleaner") |
| Consumable | 25/30 (83%) | 4 | containers for food; storage batteries |
| Customized | 30/30 (100%) | 0 | none found |
| Subscription | 29/30 (97%) | 0 | promotional "club services" |
| Switching costs | 25/30 (83%) | 1 | insurance as a topic (claims coding, investigations) |
| Platform/network | 25/30 (83%) | 0 | hardware "platforms", topic mentions |
| Complement | 27/30 (90%) | 1 | "accessories for infants"; "covers for the foodservice industry" |
| Premarket approval | 26/30 (87%) | 0 | consulting or media about pharmaceuticals |
| Licensed trade | 27/30 (90%) | 1 | publications about wine; nursing homes as a setting |
| Advertising-intensive CPG | 26/30 (87%) | 2 | containers and applicators for the goods |
| R&D technology | 27/30 (90%) | 1 | satellite broadcasting; advertising over wireless networks |
| Digital | 29/30 (97%) | 1 | database servers (hardware) |
| Intermediary | 29/30 (97%) | 1 | consulting about retail |
| Direct channel | 25/30 (83%) | 0 | vending machines as goods; e-commerce software |
| Perishable | 27/30 (90%) | 0 | fruit-based beverages; rice cakes |
| Fashion/novelty | 28/30 (93%) | 1 | "in a linear fashion"; fashion as a topic |
| Luxury | 27/30 (90%) | 0 | "fine art" as a field of teaching or insurance |
| Discount | 12/15 | - | 15 matches in 40,000; not usable |

The B2B and B2C lexicons find explicit cues only: 7 and 10 of 30 non-matches plainly sell to businesses or consumers without saying so. Their recall is roughly one in four, so these flags mark registrations that state the buyer, not all registrations that serve that buyer.

**Not identified by text.** Standardization (never stated; only its absence, customization, is). The size of switching costs or of a network (text names the relationship or the platform, not its strength). Advertising spending (text gives the category). Freight distances. The timing of fads, which needs filing time series, not single descriptions. Discount positioning.

Two properties of the text matter for the estimates. The goods_services field holds the whole application's description across all its classes, while the frame keeps one row per serial at its lowest class. A flag can therefore come from goods in another class of the same application. Bracketed text (goods deleted after registration) is kept. The flags describe the offering as filed for passed and failed registrations alike, because deleting brackets would remove text only from registrations that filed the declaration.

## 3. Associations with survival and with the cost of leading

Each row is a separate regression of survival on lead, the characteristic (centred), their interaction, and the controls, with class x registration-year fixed effects and owner-clustered SEs. "Survival difference" is the characteristic's coefficient. "Change in lead effect" is the interaction: the difference, among registrations with the characteristic, in the survival gap between the most leading and the most lagging filing; the baseline lead effect is -1.6. "Lead effect if present" is lead + interaction x (1 - share). Stars use Holm-adjusted p-values: across the 26 text characteristics for survival differences, and across all 32 rows for interactions (* < 0.05, ** < 0.01, *** < 0.001). Class-defined rows report the survival difference from a registration-year-FE model ([year FE]), with no stars. The joint model enters all 26 text characteristics, all 32 interactions and the controls together (joint lead effect -1.8, SE 0.2), with its own Holm adjustment. Standard errors in parentheses; pp = percentage points.

| Characteristic | References | Identifiable | Precision; misses | Share | Survival difference, pp | Change in lead effect, pp | Lead effect if present | Joint model: change in lead effect |
|---|---|---|---|---|---|---|---|---|
| Sold to businesses | robinson1988; robinson1985 | partly | 28/30; 7/30 | 8.5% | +1.0 (0.2)*** | +0.6 (0.5) | -1.0 | -0.5 (0.5) |
| Sold to end consumers | robinson1985; golder1993 | partly | 27/30; 10/30 | 11.2% | -1.7 (0.6)* | +2.9 (1.2) | +1.1 | +2.2 (1.6) |
| Local, site-based offering | bresnahan1991; syverson2004 | yes | 26/30; 3/30 | 7.4% | +3.5 (0.2)*** | -1.4 (0.5) | -3.1 | -3.6 (0.6)*** |
| Explicitly national or international trade scope | syverson2004; sutton1991 | partly | 26/30; 0/30 | 0.3% | -1.9 (0.7) | -4.0 (2.1) | -5.6 | -5.3 (2.1) |
| Bulky, low value-to-weight goods | syverson2004 | partly | 29/30; 0/30 | 0.5% | -1.7 (0.7) | +6.3 (1.9)* | +4.6 | +6.7 (2.0)* |
| Services (Nice classes 35-45) | song1999; zeithaml1985 | yes (class) | class | 37.4% | -3.2 (0.1) [year FE] | +0.7 (0.4) | -1.2 | -0.1 (0.5) |
| Credence offering (quality hard to judge even after use) | darby1973; dulleck2006 | yes | 27/30; 1/30 | 7.4% | +2.9 (0.3)*** | -0.8 (0.6) | -2.3 | -0.2 (0.6) |
| Experience good (food, drink, personal care, entertainment, travel) | nelson1970; nelson1974 | yes | 29/30; 3/30 | 22.9% | -0.2 (0.2) | +0.2 (0.4) | -1.5 | +1.0 (0.4) |
| Search good (apparel, furniture, hardware, inspectable before purchase) | nelson1970; nelson1974 | yes | 26/30; 1/30 | 14.0% | +0.9 (0.4) | +0.2 (0.4) | -1.5 | -0.9 (0.5) |
| Durable good (machines, appliances, vehicles, furniture) | coase1972; bulow1982 | yes | 25/30; 4/30 | 14.9% | +0.6 (0.3) | -1.2 (0.5) | -2.6 | -0.8 (0.5) |
| Consumable, repeat-purchase good | coase1972; klemperer1995 | yes | 25/30; 4/30 | 9.7% | -1.7 (0.3)*** | -1.7 (0.5)* | -3.1 | -4.1 (0.7)*** |
| Customized or made-to-order | utterback1975; suarez1995 | partly | 30/30; 0/30 | 1.7% | -1.5 (0.3)*** | -2.9 (1.0) | -4.5 | -1.3 (1.0) |
| Subscription, membership or recurring contract | klemperer1987; farrell2007 | yes | 29/30; 0/30 | 2.3% | +0.3 (0.4) | +0.7 (1.0) | -0.9 | +0.5 (0.9) |
| Ongoing account relationship (switching costs) | klemperer1987; klemperer1995; farrell2007 | partly | 25/30; 1/30 | 4.0% | -2.2 (0.4)*** | +0.7 (0.9) | -1.1 | +3.8 (1.0)** |
| Platform or network offering | katz1985; rochet2003; suarez2007 | yes | 25/30; 0/30 | 2.0% | -9.4 (0.3)*** | -5.0 (0.8)*** | -6.4 | -4.3 (0.8)*** |
| Complement to another product (for use with, adapted for) | teece1986; adner2010 | yes | 27/30; 1/30 | 4.0% | +0.2 (0.3) | -3.8 (0.9)*** | -5.3 | -3.3 (0.8)** |
| Requires premarket regulatory approval (drugs, devices, pesticides) | grabowski1992; scottmorton1999 | yes | 26/30; 0/30 | 3.2% | -0.6 (0.5) | -0.5 (1.8) | -2.2 | -0.5 (1.4) |
| Pharmaceuticals and medical devices (classes 5, 10) | grabowski1992; scottmorton1999 | yes (class) | class | 5.3% | +1.0 (0.6) [year FE] | -1.8 (1.4) | -3.3 | -3.9 (1.0)** |
| Licensed or regulated trade (alcohol, tobacco, firearms, finance, gaming, health) | teece1986; bresnahan1991 | yes | 27/30; 1/30 | 9.4% | +0.9 (0.3)* | -0.3 (0.5) | -1.9 | -2.8 (0.6)*** |
| Mass-advertised packaged consumer good (Sutton's endogenous sunk-cost categories) | sutton1991; schmalensee1982; bronnenberg2009 | partly | 26/30; 2/30 | 5.9% | -2.1 (0.4)*** | +3.7 (0.7)*** | +1.9 | +5.6 (0.8)*** |
| Packaged consumer goods (classes 3, 29-33) | sutton1991; bronnenberg2009 | partly (class) | class | 11.8% | +1.6 (0.4) [year FE] | -0.6 (0.6) | -2.2 | -0.9 (1.0) |
| R&D-intensive technology | teece1986; christensen1996; suarez2007 | yes | 27/30; 1/30 | 15.4% | -4.2 (0.2)*** | -1.8 (0.6)* | -3.6 | +3.2 (0.8)** |
| Electronics, software and IT services (classes 9, 42) | teece1986; suarez2007 | yes (class) | class | 17.8% | -1.5 (0.2) [year FE] | -4.3 (0.4)*** | -5.2 | -5.6 (0.6)*** |
| Digital or information good (near-zero marginal cost) | shapiro1999; bakos1999 | yes | 29/30; 1/30 | 14.1% | -4.8 (0.2)*** | -3.2 (0.5)*** | -4.6 | -2.3 (0.7)** |
| Intermediary (retail, wholesale, distribution, brokerage) | spulber1996; teece1986 | yes | 29/30; 1/30 | 4.9% | +2.5 (0.2)*** | +1.7 (0.6) | -0.1 | +1.3 (0.8) |
| Direct-to-buyer channel (mail order, direct selling, online ordering) | brynjolfsson2000; spulber1996 | yes | 25/30; 0/30 | 2.5% | -1.0 (0.3)** | +1.2 (0.8) | -0.4 | -1.5 (1.2) |
| Perishable good | gallego1994 | yes | 27/30; 0/30 | 1.8% | +1.5 (0.5)* | -3.1 (1.3) | -4.7 | -1.7 (1.3) |
| Fresh and processed foods (classes 29-31) | gallego1994 | partly (class) | class | 5.2% | +1.8 (0.4) [year FE] | -2.5 (0.8) | -4.0 | -3.5 (1.2) |
| Fashion, novelty, seasonal or collectible | bikhchandani1992; pesendorfer1995 | partly | 28/30; 1/30 | 1.7% | +0.1 (0.5) | +0.5 (1.0) | -1.1 | -0.4 (1.3) |
| Fashion goods (classes 14, 18, 25) | pesendorfer1995 | partly (class) | class | 9.1% | -4.3 (0.2) [year FE] | +1.1 (0.4) | -0.6 | -0.4 (0.9) |
| Luxury or premium positioning | bagwell1996; amaldoss2005 | partly | 27/30; 0/30 | 1.3% | +0.0 (0.4) | -0.6 (1.0) | -2.2 | -1.0 (1.0) |
| Discount or value positioning | bagwell1996 | no | 12/15; - | 0.1% | +3.9 (5.1) | -3.5 (10.9) | -5.2 | -3.3 (10.1) |

### Platforms, complements and digital goods carry the largest lead penalties

Network and platform offerings have the lowest survival of any characteristic (-9.4 pp). Leading costs them 6.4 pp, four times the baseline. Complements to another firm's product survive at the average rate, yet leading costs them 5.3 pp. Digital goods (-4.6) and the electronics and IT classes (-5.2) follow. All four interactions hold in the joint model. These are the settings where the literature locates value capture outside the innovating firm: in the installed base (katz1985; rochet2003) or with the owner of the complementary asset (teece1986; adner2010).

### Leading pays in advertising-intensive consumer goods and bulky materials

For mass-advertised packaged goods the lead effect turns positive (+1.9 pp; interaction +3.7, joint +5.6). This is the setting in which early-entrant brand shares persist (bronnenberg2009) and advertising outlays act as sunk costs (sutton1991). Bulky, low value-to-weight goods show the same reversal (+4.6 pp) on 0.5% of registrations, the local-market setting of syverson2004. Text flags drive both results. The class-defined packaged-goods group (classes 3, 29-33) shows no change in the lead effect (-0.6, SE 0.6). The advantage therefore attaches to the named mass-advertised categories, not to the classes as a whole.

### Local services, consumables and licensed trades add a penalty once other characteristics are held fixed

Local site-based offerings survive more often (+3.5 pp). Their interaction is -1.4 alone (Holm p = 0.11) and -3.6 in the joint model. Consumables add -1.7 alone and -4.1 jointly. Licensed trades add -2.8 jointly (-0.3 alone). Switching-cost relationships go the other way in the joint model (+3.8). They overlap with licensed trades (phi = 0.42; banking and insurance appear in both lexicons), so the two joint coefficients split one group of financial services.

### R&D technology and digital goods cannot be separated in the joint model

The R&D-technology and digital flags share most of their matches through "software" (phi = 0.68). Alone, each raises the lead penalty (-1.8 and -3.2). In the joint model, R&D technology changes sign (+3.2) while digital keeps a penalty (-2.3). The joint coefficients for these two rows split one software effect and are not separate estimates. The class-defined row (classes 9, 42: -4.3 alone, -5.6 jointly) is the cleaner measure of technology intensity.

### Characteristics with no measurable relation

Experience versus search goods, subscriptions, premarket approval read from text, fashion and novelty cues, and luxury positioning show neither a survival difference nor a change in the lead effect after the Holm adjustment. The precision of these lexicons is 87-100%. Approval (3.2% of registrations), fashion (1.7%), luxury (1.3%) and subscriptions (2.3%) are rare, so their interaction SEs are 1.0-1.8 pp and moderate effects cannot be ruled out.

### Relation to the earlier battery lexicons

The battery frame already carries three earlier lexicons (`r_site`, `r_switching`, `r_network` from `scripts/hypothesis_battery.py`), used in `combined_model.py`. The reviewed versions here agree in sign. The platform interaction is smaller here (-5.0 against -8.4 in `combined_model.json`). The reviewed lexicon drops the hardware and furniture "platforms" that the earlier one counted, and this model adds the counsel, intent-to-use and domicile controls.

## Reproduction

```
PYTHONIOENCODING=utf-8 python scripts/strategy_dimensions.py flags    # data/processed/strategy_flags.parquet (about 12 min)
PYTHONIOENCODING=utf-8 python scripts/strategy_dimensions.py sample   # data/processed/strategy_review_sample.parquet
PYTHONIOENCODING=utf-8 python scripts/strategy_dimensions.py review network m 12   # hand-review draws
PYTHONIOENCODING=utf-8 python scripts/strategy_dimensions.py models   # paper/results/strategy_dimensions_tests.json
PYTHONIOENCODING=utf-8 python scripts/strategy_dimensions.py table    # the results table above
```

