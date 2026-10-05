# Nice class -> BLS Producer Price Index concordance

Source: BLS PPI flat files, download.bls.gov/pub/time.series/pc (industry, NAICS-based) and /wp (commodity), pulled 2026-10-05 (data through 2026-08). Years are the contiguous run of complete annual averages (BLS M13 or 12-month mean) ending at the last complete year (2025). Fit grades: good = the series prices most of what the class or segment sells; partial = it prices an identifiable part; poor = no PPI covers the bulk of it.

## Coverage

- Classes, primary mapping: good 14, partial 27, poor 4 (of 45). 35 primary series run from 1995 or earlier through 2024.
- Distinct segment names (57, covering the 60 segment ids): good 8, partial 22, poor 27. Mixed two-theme segments are graded poor and left unmapped.
- Services classes 35-45: none graded good; 37, 41 and 43 are poor (no PPI for residential construction, education/entertainment, or restaurants).

## Paint, coatings and powder coatings

- No PPI series prices powder coatings. A search of pc.product, pc.series, wp.item and wp.series for "powder" returns only metal powders, abrasives, flavoring powders, and "Transportation finishes, except powdered and high-solids coatings" (PCU32551032551041 / WPU062102011, from 2012), which excludes them.
- Powder coatings are sold mainly as OEM industrial finishes, so the nearest long series is WPU06210201 / PCU3255103255104 (OEM product finishes excluding marine, 1983-). Since 2012 BLS splits it into transportation finishes (excluding powder) and "all other OEM product finishes" (PCU32551032551042 / WPU062102012), the cell that contains powder coatings.
- Paint and coating manufacturing as a whole: PCU325510325510 (NAICS 325510, 1983-); prepared paint WPU0621 (1926-). The class-40 service counterpart (powder-coating job shops) is PCU332812332812, metal coating and nonprecious engraving (1984-).

## Classes

| Class | Content | Role | Series | Title | NAICS | Base | Years | Fit | Justification |
|---|---|---|---|---|---|---|---|---|---|
| 001 | Industrial chemicals | primary | WPU061 | Chemicals and allied products-Industrial chemicals | 3251 | 198200 | 1926-2025 | good | Industrial chemicals commodity index covers the core of class 1 (industrial/agricultural chemicals); fertilizers and adhesives only partly. |
| 001 |  | alternate | PCU3251--3251-- | Basic chemical mfg | 3251 | 198412 | 1985-2025 |  | basic chemical mfg, 1984- |
| 002 | Paints, varnishes, colorants | primary | PCU325510325510 | Paint and coating manufacturing | 325510 | 198306 | 1984-2025 | good | Paint and coating manufacturing is the industry that makes class 2 goods. |
| 002 |  | alternate | WPU0621 | Chemicals and allied products-Prepared paint | 325510 | 198200 | 1926-2025 |  | prepared paint commodity, 1926- |
| 002 |  | alternate | WPU06210201 | Chemicals and allied products-OEM finishes excluding marine coatings | 325510 | 198306 | 1984-2025 |  | OEM product finishes (where powder coatings sit), 1983- |
| 003 | Cosmetics and cleaning preparations | primary | PCU3256--3256-- | Soap, cleaners, and toilet preparation mfg | 3256 | 198412 | 1985-2025 | good | Soap, cleaning compound and toilet preparation mfg spans both halves of class 3 (cosmetics and cleaners). |
| 003 |  | alternate | PCU325620325620 | Toilet preparation manufacturing | 325620 | 198003 | 1981-2025 |  | toilet preparations only, 1980- |
| 004 | Industrial oils, lubricants, fuels, candles | primary | WPU0576 | Fuels and related products and power-Finished lubricants | 324191 | 198200 | 1974-2025 | partial | Finished lubricants match the oils/greases core; fuels and candles are not covered. |
| 005 | Pharmaceuticals and supplements | primary | PCU325412325412 | Pharmaceutical preparation manufacturing | 325412 | 198106 | 1982-2025 | good | Pharmaceutical preparation mfg; dietary supplements and veterinary products only partly. |
| 005 |  | alternate | PCU325411325411 | Medicinal and botanical manufacturing | 325411 | 198206 | 1983-2025 |  | medicinal and botanical mfg (supplements), 1982- |
| 006 | Common metals and hardware | primary | PCU332510332510 | Hardware mfg | 332510 | 198506 | 1986-2025 | partial | Hardware mfg covers locks/fittings; base metals, metal building materials and safes are outside it. |
| 006 |  | alternate | WPU1041 | Metals and metal products-Hardware, n.e.c. | 3325 | 198200 | 1947-2025 |  | hardware n.e.c. commodity, 1947- |
| 007 | Machines and machine tools | primary | WPU114 | Machinery and equipment-General purpose machinery and equipment | 3339 | 198200 | 1939-2025 | partial | General purpose machinery; class 7 also includes engines, agricultural and kitchen machines. |
| 007 |  | alternate | PCU333---333--- | Machinery manufacturing | 333 | 200312 | 2004-2025 |  | machinery mfg, 2003- only |
| 008 | Hand tools and cutlery | primary | WPU1042 | Metals and metal products-Hand and edge tools | 332216 | 198200 | 1947-2025 | good | Hand and edge tools commodity index; cutlery and razors partly. |
| 008 |  | alternate | PCU332216332216 | Saw blade, handsaw, and hand and edge tool mfg | 332216 | 201112 | 2012-2025 |  | industry series, 2011- only |
| 009 | Scientific/electronic apparatus, computers, software | primary | PCU334---334--- | Computer & electronic product mfg | 334 | 200312 | 2004-2025 | partial | Computer and electronic product mfg; class 9 is dominated by downloadable software and media, which this does not price. Starts 2003. |
| 009 |  | alternate | PCU3341--3341-- | Computer & peripheral equipment mfg | 3341 | 200612 | 1995-2025 |  | computers and peripherals, 1995- |
| 009 |  | alternate | PCU513210513210 | Software publishers | 513210 | 199712 | 1998-2025 |  | software publishers, 1997- |
| 010 | Medical apparatus | primary | PCU339112339112 | Surgical and medical instrument mfg | 339112 | 198206 | 1983-2025 | good | Surgical and medical instrument mfg. |
| 010 |  | alternate | WPU1563 | Miscellaneous products-Medical and surgical appliances and supplies | 339113 | 198306 | 1984-2025 |  | medical and surgical appliances, 1983- |
| 011 | Lighting, heating, cooking, refrigeration, sanitary | primary | PCU333415333415 | Air-conditioning, refrigeration, and forced air heating equipment mfg | 333415 | 198212 | 1978-2025 | partial | HVAC and refrigeration equipment; lighting, cooking appliances and sanitary fixtures are separate industries. |
| 011 |  | alternate | WPU124 | Furniture and household durables-Household appliances | 3352 | 198200 | 1947-2025 |  | household appliances, 1947- |
| 011 |  | alternate | WPU1083 | Metals and metal products-Lighting fixtures | 3351 | 198200 | 1960-2025 |  | lighting fixtures, 1960- |
| 012 | Vehicles | primary | PCU336110336110 | Automobile, light truck and utility vehicle mfg | 336110 | 198206 | 1976-2025 | partial | Light vehicle assembly; parts, tyres, bicycles, boats and aircraft also sit in class 12. |
| 012 |  | alternate | PCU3363--3363-- | Motor vehicle parts manufacturing | 3363 | 200312 | 2004-2025 |  | motor vehicle parts, 2003- |
| 013 | Firearms, ammunition, fireworks | primary | WPU1514 | Miscellaneous products-Small arms | 332994 | 198200 | 1947-2024 | good | Small arms commodity index; ammunition and fireworks not separately priced. |
| 014 | Jewelry, precious metals, watches | primary | WPU1594 | Miscellaneous products-Jewelry and jewelry products | 339910 | 198200 | 1979-2025 | good | Jewelry and jewelry products; watches/clocks not covered. |
| 015 | Musical instruments | primary | PCU339992339992 | Musical instrument mfg | 339992 | 198506 | 1979-2025 | good | Musical instrument mfg. |
| 016 | Paper goods and printed matter | primary | WPU0915 | Pulp, paper, and allied products-Converted paper and paperboard products | 3222 | 198200 | 1947-2025 | partial | Converted paper (stationery, envelopes, bags); printed matter (books, periodicals) is priced separately. |
| 016 |  | alternate | WPU3311 | Publishing sales, excluding software-Sales of books | 513130 | 198200 | 1981-2025 |  | book publishing sales, 1980- |
| 017 | Rubber, plastics (semi-finished), insulation | primary | PCU326---326--- | Plastics and rubber products mfg | 326 | 198412 | 1985-2025 | partial | Plastics and rubber products; includes finished plastic goods outside class 17. |
| 018 | Leather goods and luggage | primary | PCU316210316210 | Footwear manufacturing | 316210 | 201112 | 2012-2025 | poor | No long PPI for luggage/handbags/leather goods (wp group 04 discontinued); footwear (2011-) is the nearest live leather series but belongs to class 25. |
| 019 | Non-metallic building materials | primary | PCU327---327--- | Nonmetallic mineral product manufacturing | 327 | 198412 | 1985-2025 | partial | Nonmetallic mineral products (cement, concrete, glass, clay); lumber and asphalt building materials are outside it. |
| 020 | Furniture | primary | PCU337---337--- | Furniture & related product mfg | 337 | 198412 | 1985-2025 | good | Furniture and related product mfg. |
| 021 | Household utensils, glassware, cookware | primary | WPU126101 | Furniture and household durables-Vitreous china, porcelain, and earthenware table and kitchenware and other pottery products | 327110 | 198200 | 1975-2025 | partial | Table and kitchenware pottery; cookware, glassware, brushes and containers are only partly covered. |
| 021 |  | alternate | PCU327212327212 | Other pressed and blown glass and glassware | 327212 | 198306 | 1984-2025 |  | pressed and blown glassware, 1983- |
| 022 | Ropes, tents, bags, fibres | primary | WPU0383 | Textile products and apparel-Industrial and other fabricated products | 3149 | 198200 | 1978-2025 | partial | Industrial and other fabricated textile products (tents, cordage, bags); raw fibres not covered. |
| 023 | Yarns and threads | primary | PCU3131--3131-- | Fiber, yarn, and thread mills | 3131 | 198412 | 1985-2025 | good | Fiber, yarn and thread mills. |
| 024 | Textiles and household linens | primary | WPU0382 | Textile products and apparel-Textile house furnishings | 3141 | 198200 | 1947-2025 | partial | Textile house furnishings (linens, curtains); piece-goods fabrics priced separately. |
| 024 |  | alternate | WPU034 | Textile products and apparel-Finished fabrics | 3133 | 198200 | 1976-2025 |  | finished fabrics, 1975- |
| 025 | Clothing, footwear, headwear | primary | WPU0381 | Textile products and apparel-Apparel | 315 | 198200 | 1947-2025 | good | Apparel commodity index; footwear and headwear partly. |
| 026 | Lace, buttons, notions, artificial flowers | primary | WPU153201 | Miscellaneous products-Fasteners, zippers, buttons, needles, pins, and buckles | 339993 | 198200 | 1986-2024 | partial | Fasteners, zippers, buttons, needles, pins; lace, ribbons and artificial flowers not covered. Ends 2025. |
| 026 |  | alternate | WPU0347 | Textile products and apparel-Screen printed textile materials, embroideries, and lace goods | 3132 | 198506 | 1986-2025 |  | embroideries and lace, 1985- |
| 027 | Carpets and floor coverings | primary | WPU1231 | Furniture and household durables-Carpets and rugs | 314110 | 198200 | 1947-2025 | good | Carpets and rugs. |
| 028 | Toys, games, sporting goods | primary | PCU339920339920 | Sporting and athletic goods mfg | 339920 | 198512 | 1986-2025 | partial | Sporting and athletic goods; toys and games have no live PPI. |
| 029 | Meat, fish, dairy, preserved foods | primary | WPU022 | Processed foods and feeds-Meats, poultry, and fish | 3116 | 198200 | 1926-2025 | partial | Meats, poultry and fish; dairy, preserved produce and snack foods are priced in other series. |
| 029 |  | alternate | WPU023 | Processed foods and feeds-Dairy products | 3115 | 198200 | 1926-2025 |  | dairy products, 1926- |
| 030 | Coffee, bakery, confectionery, staples | primary | WPU021 | Processed foods and feeds-Cereal and bakery products | 3118 | 198200 | 1926-2025 | partial | Cereal and bakery products; coffee, confectionery and condiments are separate. |
| 031 | Agricultural products, live animals, feed | primary | WPU01 | Farm products | 111 | 198200 | 1913-2025 | partial | Farm products (crops and livestock); live plants, seeds and animal feed only partly. |
| 032 | Beer and non-alcoholic beverages | primary | PCU3121--3121-- | Beverage mfg | 3121 | 198412 | 1985-2025 | partial | Beverage mfg (soft drinks, water, beer) but also includes wine and spirits (class 33). |
| 032 |  | alternate | PCU312111312111 | Soft drink manufacturing | 312111 | 198106 | 1982-2025 |  | soft drinks, 1981- |
| 032 |  | alternate | PCU312120312120 | Breweries | 312120 | 198206 | 2010-2025 |  | breweries, 1982- |
| 033 | Wine and spirits | primary | PCU312130312130 | Wineries | 312130 | 198312 | 1984-2025 | partial | Wineries; spirits are a separate industry. |
| 033 |  | alternate | PCU312140312140 | Distilleries | 312140 | 198306 | 1976-2025 |  | distilleries, 1975- |
| 034 | Tobacco and smokers' articles | primary | PCU3122--3122-- | Tobacco mfg | 3122 | 198412 | 1985-2025 | good | Tobacco mfg; smokers' articles and e-cigarette devices only partly. |
| 035 | Advertising, business management, retail | primary | PCU541810541810 | Advertising agencies | 541810 | 199506 | 1996-2025 | partial | Advertising agencies; business management and retail-store services (the bulk of class 35) are not priced here (retail margins only from 2006). |
| 035 |  | alternate | PCUARETTRARETTR | Total retail trade industries | 44-45 | 200612 | 2007-2025 |  | total retail trade margins, 2006- |
| 036 | Insurance, finance, real estate | primary | PCU524126524126 | Direct property and casualty insurers | 524126 | 199806 | 1999-2025 | partial | Property and casualty insurance; banking (2003-) and real estate are separate. |
| 037 | Construction, repair, installation | primary | PCU2381MR2381MR | Nonresidential building maintenance & repair | 2381 | 200904 | 2010-2025 | poor | Only nonresidential building maintenance and repair, from 2009; no residential construction or installation services PPI. |
| 038 | Telecommunications | primary | PCU517311517311 | Wired telecommunications carriers | 517311 | 200312 | 1996-2025 | partial | Wired telecom carriers; broadcasting, streaming and online messaging services are separate. |
| 038 |  | alternate | PCU517312517312 | Wireless telecommunications carriers | 517312 | 199906 | 2000-2025 |  | wireless carriers, 1999- |
| 039 | Transport, travel, storage | primary | PCU481---481--- | Air transportation | 481 | 199212 | 1993-2025 | partial | Air transportation; trucking, travel arrangement and storage are separate. |
| 040 | Treatment of materials | primary | PCU332812332812 | Metal coating and nonprecious engraving | 332812 | 198412 | 1985-2025 | partial | Metal coating and engraving services (incl. powder-coating job shops); class 40 also covers custom manufacturing, printing and food processing. |
| 041 | Education, entertainment, sport, culture | primary | PCU713940713940 | Fitness and recreational sports centers | 713940 | 200412 | 2005-2025 | poor | No PPI for education or entertainment content; fitness centres (2004-) are a sliver of class 41. |
| 042 | Scientific, technology, software services | primary | PCU518210518210 | Data processing, hosting and related services | 518210 | 200012 | 2001-2025 | partial | Data processing and hosting (SaaS); engineering, design and research services are separate. |
| 042 |  | alternate | PCU541330541330 | Engineering services | 541330 | 199612 | 1997-2025 |  | engineering services, 1996- |
| 042 |  | alternate | PCU513210513210 | Software publishers | 513210 | 199712 | 1998-2025 |  | software publishers, 1997- |
| 043 | Restaurants and accommodation | primary | PCU721---721--- | Accommodation | 721 | 199612 | 1997-2025 | poor | Accommodation only; restaurants and food service, the bulk of class 43, have no PPI. |
| 044 | Medical, beauty, agricultural services | primary | PCU621111621111 | Offices of physicians, except mental health | 621111 | 199312 | 1994-2025 | partial | Physicians' offices; beauty, spa, veterinary and landscaping services not covered. |
| 044 |  | alternate | PCU622110622110 | General medical and surgical hospitals | 622110 | 199212 | 1993-2025 |  | general hospitals, 1992- |
| 045 | Legal, security, personal services | primary | PCU541110541110 | Offices of lawyers | 541110 | 199612 | 1997-2025 | partial | Offices of lawyers; security, dating, funeral and religious services not covered. |

## Product segments (theme_segments.json; registrations 2002-2018)

| Segment | Seg ids | Series | Title | NAICS | Years | Fit | Justification |
|---|---|---|---|---|---|---|---|
| Education, training, travel and events + Downloadable and online software | 0 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Computer, software and information services | 1 | PCU518210518210 | Data processing, hosting and related services | 518210 | 2001-2025 | partial | Data processing/hosting; software publishing and IT consulting priced separately. |
| Clothing, bags and footwear | 2;57 | WPU0381 | Textile products and apparel-Apparel | 315 | 1947-2025 | good | Apparel commodity index. |
| Retail store services + Beverages and dairy | 3 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Precious metal and household glassware | 4;33 | WPU1594 | Miscellaneous products-Jewelry and jewelry products | 339910 | 1979-2025 | partial | Jewelry; household glassware not covered. |
| Education, training, travel and events + Media and entertainment goods | 5 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Automotive vehicles and lubricants | 6 | PCU336110336110 | Automobile, light truck and utility vehicle mfg | 336110 | 1976-2025 | partial | Light vehicles; lubricants separate. |
| Water treatment | 7 |  |  |  |  | poor | No PPI for water treatment equipment as a group. |
| Paper, books and stationery + Media and entertainment goods | 8 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Processed foods + Beverages and dairy | 9 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Media and entertainment goods + Electronic apparatus and control systems | 10 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Media and entertainment goods + Education, training, travel and events | 11 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Metal and building hardware | 12 | WPU1041 | Metals and metal products-Hardware, n.e.c. | 3325 | 1947-2025 | good | Hardware n.e.c. |
| Hair and skin care | 13;52 | PCU325620325620 | Toilet preparation manufacturing | 325620 | 1981-2025 | good | Toilet preparation mfg. |
| Land vehicles and parts | 14 | PCU336110336110 | Automobile, light truck and utility vehicle mfg | 336110 | 1976-2025 | partial | Light vehicles; parts separate. |
| Nutritional supplements | 15 | PCU325411325411 | Medicinal and botanical manufacturing | 325411 | 1983-2025 | partial | Medicinal and botanical mfg. |
| Education, training, travel and events + Computer, software and information services | 16 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Toys, games and sporting goods | 17 | PCU339920339920 | Sporting and athletic goods mfg | 339920 | 1986-2025 | partial | Sporting goods; toys not priced. |
| Beverages and dairy | 18 | PCU3121--3121-- | Beverage mfg | 3121 | 1985-2025 | partial | Beverage mfg; dairy separate. |
| Research, scientific and diagnostic | 19 |  |  |  |  | poor | Mix of research services and diagnostics; no single series. |
| Retail store services + Education, training, travel and events | 20 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Plants, seeds and agriculture | 21 | WPU01 | Farm products | 111 | 1913-2025 | partial | Farm products. |
| Furniture and lighting | 22 | PCU337---337--- | Furniture & related product mfg | 337 | 1985-2025 | partial | Furniture; lighting separate. |
| Construction, engineering and waste services | 23 | PCU541330541330 | Engineering services | 541330 | 1997-2025 | partial | Engineering services; construction and waste separate. |
| Pharmaceutical preparations + Medical-purpose apparatus and textile yarn (mixed) | 24 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Bicycles, vehicle rental and tobacco (mixed) | 25 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Computer, software and information services + Retail store services | 26 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Energy, transport and storage services | 27 |  |  |  |  | poor | Energy, transport and storage have no common series. |
| Paper, books and stationery | 28 | WPU0915 | Pulp, paper, and allied products-Converted paper and paperboard products | 3222 | 1947-2025 | partial | Converted paper; books separate. |
| Health and wellness services | 29 | PCU713940713940 | Fitness and recreational sports centers | 713940 | 2005-2025 | partial | Fitness centres; from 2004. |
| Plastics, textiles and packaging materials | 30 | PCU326---326--- | Plastics and rubber products mfg | 326 | 1985-2025 | partial | Plastics and rubber products. |
| Industrial machines and parts | 31 | WPU114 | Machinery and equipment-General purpose machinery and equipment | 3339 | 1939-2025 | good | General purpose machinery. |
| Education, training, travel and events | 32 |  |  |  |  | poor | No PPI for education or events. |
| Computer, software and information services + Downloadable and online software | 34 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Computer, software and information services + Health and wellness services | 35 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Pets and animal feed | 36 | PCU311111311111 | Dog and cat food manufacturing | 311111 | 1986-2025 | good | Dog and cat food mfg. |
| Electronic apparatus and control systems | 37 | WPU117 | Machinery and equipment-Electrical machinery and equipment | 335 | 1939-2025 | partial | Electrical machinery and equipment. |
| Musical and surgical instruments | 38 | PCU339992339992 | Musical instrument mfg | 339992 | 1979-2025 | partial | Musical instruments; surgical separate. |
| Retail store services | 39 | PCUARETTRARETTR | Total retail trade industries | 44-45 | 2007-2025 | partial | Total retail trade margins; from 2006 only. |
| Cleaning preparations and coatings | 40 | PCU3256--3256-- | Soap, cleaners, and toilet preparation mfg | 3256 | 1985-2025 | partial | Soap and cleaning compounds; coatings separate. |
| Media and entertainment goods | 41 |  |  |  |  | poor | Recorded media and entertainment content have no long PPI. |
| Computer, software and information services + Education, training, travel and events | 42 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Processed foods | 43 | PCU311---311--- | Food mfg | 311 | 1985-2025 | partial | Food mfg sub-sector. |
| Hand tools and knives | 44 | WPU1042 | Metals and metal products-Hand and edge tools | 332216 | 1947-2025 | good | Hand and edge tools. |
| Pharmaceutical preparations | 45 | PCU325412325412 | Pharmaceutical preparation manufacturing | 325412 | 1982-2025 | good | Pharmaceutical preparation mfg. |
| Education, training, travel and events + Health and wellness services | 46 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Essential oils and candles | 47 | PCU325620325620 | Toilet preparation manufacturing | 325620 | 1981-2025 | partial | Toilet preparations; candles separate. |
| Design and technical development services | 48 | PCU541310541310 | Architectural services | 541310 | 1997-2025 | partial | Architectural services; design and product development only partly. |
| Medical and healthcare services | 49 | PCU622110622110 | General medical and surgical hospitals | 622110 | 1993-2025 | partial | General hospitals. |
| Media and entertainment goods + Downloadable and online software | 50 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Computer, software and information services + Media and entertainment goods | 51 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Retail store services + Marketing of consumer goods | 53 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Mats, floor coverings and printing | 54 | WPU1231 | Furniture and household durables-Carpets and rugs | 314110 | 1947-2025 | partial | Carpets and rugs; printing separate. |
| Heating and air apparatus | 55 | PCU333415333415 | Air-conditioning, refrigeration, and forced air heating equipment mfg | 333415 | 1978-2025 | good | HVAC and refrigeration equipment. |
| Computer, software and information services + Electronic apparatus and control systems | 56 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Medical-purpose apparatus and textile yarn (mixed) | 58 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
| Cleaning preparations and coatings + Industrial chemicals | 59 |  |  |  |  | poor | Mixed segment (two themes); no single industry. |
