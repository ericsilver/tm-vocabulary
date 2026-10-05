"""Nice class / theme segment -> BLS Producer Price Index concordance.

Maps each of the 45 Nice classes (and the single-theme product segments in
paper/results/theme_segments.json) to the closest PPI industry (pc, NAICS-based)
or commodity (wp) series, preferring series with annual data from 1995 or earlier
through 2024. Coverage (first/last full year) is computed from the downloaded
BLS flat files, not from the series metadata.

Inputs  : data/raw/bls/{pc,wp}.series, {pc,wp}.data.<sector> files, cu.data.1.AllItems
          (download.bls.gov/pub/time.series/, UA with contact email)
Outputs : paper/results/bls_concordance.csv, paper/results/bls_concordance.md,
          data/processed/bls_ppi_annual.parquet (annual averages of every series used)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bls"
RES = ROOT / "paper" / "results"
PROC = ROOT / "data" / "processed"

NICE = {
    "001": "Industrial chemicals", "002": "Paints, varnishes, colorants",
    "003": "Cosmetics and cleaning preparations", "004": "Industrial oils, lubricants, fuels, candles",
    "005": "Pharmaceuticals and supplements", "006": "Common metals and hardware",
    "007": "Machines and machine tools", "008": "Hand tools and cutlery",
    "009": "Scientific/electronic apparatus, computers, software", "010": "Medical apparatus",
    "011": "Lighting, heating, cooking, refrigeration, sanitary", "012": "Vehicles",
    "013": "Firearms, ammunition, fireworks", "014": "Jewelry, precious metals, watches",
    "015": "Musical instruments", "016": "Paper goods and printed matter",
    "017": "Rubber, plastics (semi-finished), insulation", "018": "Leather goods and luggage",
    "019": "Non-metallic building materials", "020": "Furniture",
    "021": "Household utensils, glassware, cookware", "022": "Ropes, tents, bags, fibres",
    "023": "Yarns and threads", "024": "Textiles and household linens",
    "025": "Clothing, footwear, headwear", "026": "Lace, buttons, notions, artificial flowers",
    "027": "Carpets and floor coverings", "028": "Toys, games, sporting goods",
    "029": "Meat, fish, dairy, preserved foods", "030": "Coffee, bakery, confectionery, staples",
    "031": "Agricultural products, live animals, feed", "032": "Beer and non-alcoholic beverages",
    "033": "Wine and spirits", "034": "Tobacco and smokers' articles",
    "035": "Advertising, business management, retail", "036": "Insurance, finance, real estate",
    "037": "Construction, repair, installation", "038": "Telecommunications",
    "039": "Transport, travel, storage", "040": "Treatment of materials",
    "041": "Education, entertainment, sport, culture", "042": "Scientific, technology, software services",
    "043": "Restaurants and accommodation", "044": "Medical, beauty, agricultural services",
    "045": "Legal, security, personal services",
}

# (class, series_id, naics, fit, justification, [alternates (series_id, naics, note)])
CLASS_MAP = [
    ("001", "WPU061", "3251", "good", "Industrial chemicals commodity index covers the core of class 1 (industrial/agricultural chemicals); fertilizers and adhesives only partly.", [("PCU3251--3251--", "3251", "basic chemical mfg, 1984-")]),
    ("002", "PCU325510325510", "325510", "good", "Paint and coating manufacturing is the industry that makes class 2 goods.", [("WPU0621", "325510", "prepared paint commodity, 1926-"), ("WPU06210201", "325510", "OEM product finishes (where powder coatings sit), 1983-")]),
    ("003", "PCU3256--3256--", "3256", "good", "Soap, cleaning compound and toilet preparation mfg spans both halves of class 3 (cosmetics and cleaners).", [("PCU325620325620", "325620", "toilet preparations only, 1980-")]),
    ("004", "WPU0576", "324191", "partial", "Finished lubricants match the oils/greases core; fuels and candles are not covered.", []),
    ("005", "PCU325412325412", "325412", "good", "Pharmaceutical preparation mfg; dietary supplements and veterinary products only partly.", [("PCU325411325411", "325411", "medicinal and botanical mfg (supplements), 1982-")]),
    ("006", "PCU332510332510", "332510", "partial", "Hardware mfg covers locks/fittings; base metals, metal building materials and safes are outside it.", [("WPU1041", "3325", "hardware n.e.c. commodity, 1947-")]),
    ("007", "WPU114", "3339", "partial", "General purpose machinery; class 7 also includes engines, agricultural and kitchen machines.", [("PCU333---333---", "333", "machinery mfg, 2003- only")]),
    ("008", "WPU1042", "332216", "good", "Hand and edge tools commodity index; cutlery and razors partly.", [("PCU332216332216", "332216", "industry series, 2011- only")]),
    ("009", "PCU334---334---", "334", "partial", "Computer and electronic product mfg; class 9 is dominated by downloadable software and media, which this does not price. Starts 2003.", [("PCU3341--3341--", "3341", "computers and peripherals, 1995-"), ("PCU513210513210", "513210", "software publishers, 1997-")]),
    ("010", "PCU339112339112", "339112", "good", "Surgical and medical instrument mfg.", [("WPU1563", "339113", "medical and surgical appliances, 1983-")]),
    ("011", "PCU333415333415", "333415", "partial", "HVAC and refrigeration equipment; lighting, cooking appliances and sanitary fixtures are separate industries.", [("WPU124", "3352", "household appliances, 1947-"), ("WPU1083", "3351", "lighting fixtures, 1960-")]),
    ("012", "PCU336110336110", "336110", "partial", "Light vehicle assembly; parts, tyres, bicycles, boats and aircraft also sit in class 12.", [("PCU3363--3363--", "3363", "motor vehicle parts, 2003-")]),
    ("013", "WPU1514", "332994", "good", "Small arms commodity index; ammunition and fireworks not separately priced.", []),
    ("014", "WPU1594", "339910", "good", "Jewelry and jewelry products; watches/clocks not covered.", []),
    ("015", "PCU339992339992", "339992", "good", "Musical instrument mfg.", []),
    ("016", "WPU0915", "3222", "partial", "Converted paper (stationery, envelopes, bags); printed matter (books, periodicals) is priced separately.", [("WPU3311", "513130", "book publishing sales, 1980-")]),
    ("017", "PCU326---326---", "326", "partial", "Plastics and rubber products; includes finished plastic goods outside class 17.", []),
    ("018", "PCU316210316210", "316210", "poor", "No long PPI for luggage/handbags/leather goods (wp group 04 discontinued); footwear (2011-) is the nearest live leather series but belongs to class 25.", []),
    ("019", "PCU327---327---", "327", "partial", "Nonmetallic mineral products (cement, concrete, glass, clay); lumber and asphalt building materials are outside it.", []),
    ("020", "PCU337---337---", "337", "good", "Furniture and related product mfg.", []),
    ("021", "WPU126101", "327110", "partial", "Table and kitchenware pottery; cookware, glassware, brushes and containers are only partly covered.", [("PCU327212327212", "327212", "pressed and blown glassware, 1983-")]),
    ("022", "WPU0383", "3149", "partial", "Industrial and other fabricated textile products (tents, cordage, bags); raw fibres not covered.", []),
    ("023", "PCU3131--3131--", "3131", "good", "Fiber, yarn and thread mills.", []),
    ("024", "WPU0382", "3141", "partial", "Textile house furnishings (linens, curtains); piece-goods fabrics priced separately.", [("WPU034", "3133", "finished fabrics, 1975-")]),
    ("025", "WPU0381", "315", "good", "Apparel commodity index; footwear and headwear partly.", []),
    ("026", "WPU153201", "339993", "partial", "Fasteners, zippers, buttons, needles, pins; lace, ribbons and artificial flowers not covered. Ends 2025.", [("WPU0347", "3132", "embroideries and lace, 1985-")]),
    ("027", "WPU1231", "314110", "good", "Carpets and rugs.", []),
    ("028", "PCU339920339920", "339920", "partial", "Sporting and athletic goods; toys and games have no live PPI.", []),
    ("029", "WPU022", "3116", "partial", "Meats, poultry and fish; dairy, preserved produce and snack foods are priced in other series.", [("WPU023", "3115", "dairy products, 1926-")]),
    ("030", "WPU021", "3118", "partial", "Cereal and bakery products; coffee, confectionery and condiments are separate.", []),
    ("031", "WPU01", "111", "partial", "Farm products (crops and livestock); live plants, seeds and animal feed only partly.", []),
    ("032", "PCU3121--3121--", "3121", "partial", "Beverage mfg (soft drinks, water, beer) but also includes wine and spirits (class 33).", [("PCU312111312111", "312111", "soft drinks, 1981-"), ("PCU312120312120", "312120", "breweries, 1982-")]),
    ("033", "PCU312130312130", "312130", "partial", "Wineries; spirits are a separate industry.", [("PCU312140312140", "312140", "distilleries, 1975-")]),
    ("034", "PCU3122--3122--", "3122", "good", "Tobacco mfg; smokers' articles and e-cigarette devices only partly.", []),
    ("035", "PCU541810541810", "541810", "partial", "Advertising agencies; business management and retail-store services (the bulk of class 35) are not priced here (retail margins only from 2006).", [("PCUARETTRARETTR", "44-45", "total retail trade margins, 2006-")]),
    ("036", "PCU524126524126", "524126", "partial", "Property and casualty insurance; banking (2003-) and real estate are separate.", []),
    ("037", "PCU2381MR2381MR", "2381", "poor", "Only nonresidential building maintenance and repair, from 2009; no residential construction or installation services PPI.", []),
    ("038", "PCU517311517311", "517311", "partial", "Wired telecom carriers; broadcasting, streaming and online messaging services are separate.", [("PCU517312517312", "517312", "wireless carriers, 1999-")]),
    ("039", "PCU481---481---", "481", "partial", "Air transportation; trucking, travel arrangement and storage are separate.", []),
    ("040", "PCU332812332812", "332812", "partial", "Metal coating and engraving services (incl. powder-coating job shops); class 40 also covers custom manufacturing, printing and food processing.", []),
    ("041", "PCU713940713940", "713940", "poor", "No PPI for education or entertainment content; fitness centres (2004-) are a sliver of class 41.", []),
    ("042", "PCU518210518210", "518210", "partial", "Data processing and hosting (SaaS); engineering, design and research services are separate.", [("PCU541330541330", "541330", "engineering services, 1996-"), ("PCU513210513210", "513210", "software publishers, 1997-")]),
    ("043", "PCU721---721---", "721", "poor", "Accommodation only; restaurants and food service, the bulk of class 43, have no PPI.", []),
    ("044", "PCU621111621111", "621111", "partial", "Physicians' offices; beauty, spa, veterinary and landscaping services not covered.", [("PCU622110622110", "622110", "general hospitals, 1992-")]),
    ("045", "PCU541110541110", "541110", "partial", "Offices of lawyers; security, dating, funeral and religious services not covered.", []),
]

# Segment name -> (series_id, naics, fit, justification). Mixed segments ("A + B")
# have no single industry and are graded poor without a series.
SEG_MAP = {
    "Computer, software and information services": ("PCU518210518210", "518210", "partial", "Data processing/hosting; software publishing and IT consulting priced separately."),
    "Clothing, bags and footwear": ("WPU0381", "315", "good", "Apparel commodity index."),
    "Precious metal and household glassware": ("WPU1594", "339910", "partial", "Jewelry; household glassware not covered."),
    "Automotive vehicles and lubricants": ("PCU336110336110", "336110", "partial", "Light vehicles; lubricants separate."),
    "Water treatment": (None, "", "poor", "No PPI for water treatment equipment as a group."),
    "Metal and building hardware": ("WPU1041", "3325", "good", "Hardware n.e.c."),
    "Hair and skin care": ("PCU325620325620", "325620", "good", "Toilet preparation mfg."),
    "Land vehicles and parts": ("PCU336110336110", "336110", "partial", "Light vehicles; parts separate."),
    "Nutritional supplements": ("PCU325411325411", "325411", "partial", "Medicinal and botanical mfg."),
    "Toys, games and sporting goods": ("PCU339920339920", "339920", "partial", "Sporting goods; toys not priced."),
    "Beverages and dairy": ("PCU3121--3121--", "3121", "partial", "Beverage mfg; dairy separate."),
    "Research, scientific and diagnostic": (None, "", "poor", "Mix of research services and diagnostics; no single series."),
    "Retail store services": ("PCUARETTRARETTR", "44-45", "partial", "Total retail trade margins; from 2006 only."),
    "Plants, seeds and agriculture": ("WPU01", "111", "partial", "Farm products."),
    "Furniture and lighting": ("PCU337---337---", "337", "partial", "Furniture; lighting separate."),
    "Construction, engineering and waste services": ("PCU541330541330", "541330", "partial", "Engineering services; construction and waste separate."),
    "Energy, transport and storage services": (None, "", "poor", "Energy, transport and storage have no common series."),
    "Paper, books and stationery": ("WPU0915", "3222", "partial", "Converted paper; books separate."),
    "Health and wellness services": ("PCU713940713940", "713940", "partial", "Fitness centres; from 2004."),
    "Plastics, textiles and packaging materials": ("PCU326---326---", "326", "partial", "Plastics and rubber products."),
    "Industrial machines and parts": ("WPU114", "3339", "good", "General purpose machinery."),
    "Education, training, travel and events": (None, "", "poor", "No PPI for education or events."),
    "Pets and animal feed": ("PCU311111311111", "311111", "good", "Dog and cat food mfg."),
    "Electronic apparatus and control systems": ("WPU117", "335", "partial", "Electrical machinery and equipment."),
    "Musical and surgical instruments": ("PCU339992339992", "339992", "partial", "Musical instruments; surgical separate."),
    "Cleaning preparations and coatings": ("PCU3256--3256--", "3256", "partial", "Soap and cleaning compounds; coatings separate."),
    "Media and entertainment goods": (None, "", "poor", "Recorded media and entertainment content have no long PPI."),
    "Processed foods": ("PCU311---311---", "311", "partial", "Food mfg sub-sector."),
    "Hand tools and knives": ("WPU1042", "332216", "good", "Hand and edge tools."),
    "Pharmaceutical preparations": ("PCU325412325412", "325412", "good", "Pharmaceutical preparation mfg."),
    "Essential oils and candles": ("PCU325620325620", "325620", "partial", "Toilet preparations; candles separate."),
    "Design and technical development services": ("PCU541310541310", "541310", "partial", "Architectural services; design and product development only partly."),
    "Medical and healthcare services": ("PCU622110622110", "622110", "partial", "General hospitals."),
    "Mats, floor coverings and printing": ("WPU1231", "314110", "partial", "Carpets and rugs; printing separate."),
    "Heating and air apparatus": ("PCU333415333415", "333415", "good", "HVAC and refrigeration equipment."),
}

EXTRA = ["CUUR0000SA0", "WPU00000000", "WPU0621", "WPU062", "WPU06210201", "WPU062102012",
         "PCU3255103255104", "PCU32551032551042", "PCU332812332812"]


def read_series_meta() -> pl.DataFrame:
    out = []
    for db in ("pc", "wp"):
        df = pl.read_csv(RAW / f"{db}.series", separator="\t", infer_schema_length=0,
                         quote_char=None, truncate_ragged_lines=True)
        df = df.rename({c: c.strip() for c in df.columns})
        df = df.select(pl.col("series_id").str.strip_chars(), pl.col("base_date").str.strip_chars(),
                       pl.col("series_title").str.strip_chars(), pl.lit(db).alias("db"))
        out.append(df)
    return pl.concat(out)


def load_monthly(ids: set[str]) -> pl.DataFrame:
    files = sorted(p for p in RAW.glob("*.data.*") if ".0.Current" not in p.name)
    parts = []
    for f in files:
        lf = pl.scan_csv(f, separator="\t", infer_schema_length=0, quote_char=None,
                         truncate_ragged_lines=True)
        cols = lf.collect_schema().names()
        lf = lf.rename({c: c.strip() for c in cols})
        df = (lf.select(pl.col("series_id").str.strip_chars(), pl.col("year").str.strip_chars(),
                        pl.col("period").str.strip_chars(), pl.col("value").str.strip_chars())
              .filter(pl.col("series_id").is_in(list(ids))).collect())
        if df.height:
            parts.append(df)
    df = pl.concat(parts).unique(["series_id", "year", "period"])
    return df.with_columns(pl.col("year").cast(pl.Int32),
                           pl.col("value").cast(pl.Float64, strict=False)).drop_nulls("value")


def annualize(m: pl.DataFrame) -> pl.DataFrame:
    """Annual average: BLS M13 when published, else the mean of 12 monthly values."""
    m13 = m.filter(pl.col("period") == "M13").select("series_id", "year", pl.col("value").alias("m13"))
    mon = (m.filter(pl.col("period").str.contains(r"^M(0[1-9]|1[0-2])$"))
           .group_by("series_id", "year").agg(pl.col("value").mean().alias("mm"), pl.len().alias("nm")))
    a = mon.join(m13, on=["series_id", "year"], how="full", coalesce=True)
    a = a.with_columns(pl.when(pl.col("m13").is_not_null()).then(pl.col("m13"))
                       .when(pl.col("nm") == 12).then(pl.col("mm")).otherwise(None).alias("ppi"))
    return a.drop_nulls("ppi").select("series_id", "year", "ppi").sort("series_id", "year")


def coverage(ann: pl.DataFrame) -> dict[str, tuple[int, int, int]]:
    out = {}
    for sid, g in ann.group_by("series_id"):
        ys = sorted(g["year"].to_list())
        # longest contiguous run ending at the last year
        last = ys[-1]
        first = last
        s = set(ys)
        while first - 1 in s:
            first -= 1
        out[sid[0]] = (first, last, last - first + 1)
    return out


def main() -> None:
    segs = json.loads((RES / "theme_segments.json").read_text(encoding="utf-8"))["segments"]
    seg_ids: dict[str, list[int]] = {}
    seg_n: dict[str, int] = {}
    for s in segs:
        seg_ids.setdefault(s["name"], []).append(s["seg"])
        seg_n[s["name"]] = seg_n.get(s["name"], 0) + s["n"]

    ids = {r[1] for r in CLASS_MAP} | {a[0] for r in CLASS_MAP for a in r[5]}
    ids |= {v[0] for v in SEG_MAP.values() if v[0]} | set(EXTRA)

    meta = read_series_meta()
    cpi = pl.read_csv(RAW / "cu.data.1.AllItems", separator="\t", infer_schema_length=0,
                      quote_char=None, truncate_ragged_lines=True)
    cpi = cpi.rename({c: c.strip() for c in cpi.columns}).select(
        pl.col("series_id").str.strip_chars(), pl.col("year").str.strip_chars(),
        pl.col("period").str.strip_chars(), pl.col("value").str.strip_chars()
    ).filter(pl.col("series_id") == "CUUR0000SA0").with_columns(
        pl.col("year").cast(pl.Int32), pl.col("value").cast(pl.Float64, strict=False))
    monthly = pl.concat([load_monthly(ids - {"CUUR0000SA0"}), cpi.drop_nulls("value")])
    ann = annualize(monthly)
    ann = ann.filter(pl.col("year") <= 2025)  # 2026 incomplete
    missing = ids - set(ann["series_id"].unique().to_list())
    if missing:
        raise SystemExit(f"series without data: {sorted(missing)}")
    PROC.mkdir(parents=True, exist_ok=True)
    ann.write_parquet(PROC / "bls_ppi_annual.parquet")
    cov = coverage(ann)
    title = {r["series_id"]: (r["series_title"], r["base_date"]) for r in meta.iter_rows(named=True)}
    title["CUUR0000SA0"] = ("CPI-U, all items, US city average, not seasonally adjusted", "198284")

    def clean(t: str) -> str:
        return re.sub(r"^PPI (industry|Commodity) (sub-sector |group )?data for |, not seasonally adjusted$", "", t)

    rows = []
    for cls, sid, naics, fit, just, alts in CLASS_MAP:
        for role, s, n, j, f in [("primary", sid, naics, just, fit)] + [("alternate", a[0], a[1], a[2], "") for a in alts]:
            t, b = title[s]
            c = cov[s]
            rows.append(dict(unit_type="class", unit=cls, unit_label=NICE[cls], segments="", role=role,
                             series_id=s, database=s[:2].lower() if s.startswith("PC") else "wp",
                             series_title=clean(t), naics=n, base_date=b, first_year=c[0], last_year=c[1],
                             n_years=c[2], fit=f, justification=j))
    for name, sids in sorted(seg_ids.items(), key=lambda kv: min(kv[1])):
        if " + " in name or "(mixed)" in name:
            m = (None, "", "poor", "Mixed segment (two themes); no single industry.")
        else:
            m = SEG_MAP[name]
        s = m[0]
        t, b = title.get(s, ("", "")) if s else ("", "")
        c = cov.get(s, (None, None, None)) if s else (None, None, None)
        rows.append(dict(unit_type="segment", unit=name, unit_label=f"n={seg_n[name]:,}",
                         segments=";".join(str(x) for x in sorted(sids)), role="primary" if s else "none",
                         series_id=s or "", database=("pc" if s and s.startswith("PC") else ("wp" if s else "")),
                         series_title=clean(t), naics=m[1], base_date=b, first_year=c[0], last_year=c[1],
                         n_years=c[2], fit=m[2], justification=m[3]))
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(RES / "bls_concordance.csv")

    # Markdown
    prim = df.filter((pl.col("unit_type") == "class") & (pl.col("role") == "primary"))
    segp = df.filter(pl.col("unit_type") == "segment")
    cnt = prim.group_by("fit").len().sort("fit")
    scnt = segp.group_by("fit").len().sort("fit")
    long_ = prim.filter((pl.col("first_year") <= 1995) & (pl.col("last_year") >= 2024)).height
    L = ["# Nice class -> BLS Producer Price Index concordance", "",
         "Source: BLS PPI flat files, download.bls.gov/pub/time.series/pc (industry, NAICS-based) and /wp (commodity), "
         "pulled 2026-10-05 (data through 2026-08). Years are the contiguous run of complete annual averages "
         "(BLS M13 or 12-month mean) ending at the last complete year (2025). Fit grades: good = the series prices most of "
         "what the class or segment sells; partial = it prices an identifiable part; poor = no PPI covers the bulk of it.", "",
         "## Coverage", "",
         f"- Classes, primary mapping: " + ", ".join(f"{r['fit']} {r['len']}" for r in cnt.iter_rows(named=True))
         + f" (of 45). {long_} primary series run from 1995 or earlier through 2024.",
         f"- Distinct segment names ({segp.height}, covering the 60 segment ids): "
         + ", ".join(f"{r['fit']} {r['len']}" for r in scnt.iter_rows(named=True))
         + ". Mixed two-theme segments are graded poor and left unmapped.",
         "- Services classes 35-45: none graded good; 37, 41 and 43 are poor (no PPI for residential construction, "
         "education/entertainment, or restaurants).", "",
         "## Paint, coatings and powder coatings", "",
         "- No PPI series prices powder coatings. A search of pc.product, pc.series, wp.item and wp.series for "
         "\"powder\" returns only metal powders, abrasives, flavoring powders, and \"Transportation finishes, except powdered "
         "and high-solids coatings\" (PCU32551032551041 / WPU062102011, from 2012), which excludes them.",
         "- Powder coatings are sold mainly as OEM industrial finishes, so the nearest long series is "
         "WPU06210201 / PCU3255103255104 (OEM product finishes excluding marine, 1983-). Since 2012 BLS splits it into "
         "transportation finishes (excluding powder) and \"all other OEM product finishes\" (PCU32551032551042 / WPU062102012), "
         "the cell that contains powder coatings.",
         "- Paint and coating manufacturing as a whole: PCU325510325510 (NAICS 325510, 1983-); prepared paint "
         "WPU0621 (1926-). The class-40 service counterpart (powder-coating job shops) is PCU332812332812, "
         "metal coating and nonprecious engraving (1984-).", "",
         "## Classes", "",
         "| Class | Content | Role | Series | Title | NAICS | Base | Years | Fit | Justification |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.filter(pl.col("unit_type") == "class").iter_rows(named=True):
        L.append(f"| {r['unit']} | {r['unit_label'] if r['role']=='primary' else ''} | {r['role']} | {r['series_id']} | "
                 f"{r['series_title']} | {r['naics']} | {r['base_date']} | {r['first_year']}-{r['last_year']} | "
                 f"{r['fit']} | {r['justification']} |")
    L += ["", "## Product segments (theme_segments.json; registrations 2002-2018)", "",
          "| Segment | Seg ids | Series | Title | NAICS | Years | Fit | Justification |", "|---|---|---|---|---|---|---|---|"]
    for r in segp.iter_rows(named=True):
        yrs = f"{r['first_year']}-{r['last_year']}" if r["first_year"] else ""
        L.append(f"| {r['unit']} | {r['segments']} | {r['series_id']} | {r['series_title']} | {r['naics']} | {yrs} | "
                 f"{r['fit']} | {r['justification']} |")
    (RES / "bls_concordance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(cnt, scnt, f"long={long_}", sep="\n")
    print(prim.select("unit", "series_id", "first_year", "last_year", "fit"))


if __name__ == "__main__":
    main()
