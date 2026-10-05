"""Lexicons for product and market characteristics named in the strategy and IO
literatures, applied to trademark goods/services descriptions.

Each text lexicon is a pair of regular expressions over the lowercased
description:

  exclude : phrases that contain an include term but carry another meaning
            ("platform shoes", "spas, hot tubs"); they are blanked out first
  include : the terms that mark the characteristic

A description is flagged when `include` matches the text left after `exclude`
has been blanked. Patterns use the Rust regex dialect that polars runs (no
look-around), so exclusions are done by blanking rather than by negative
look-ahead.

Two caveats on the text:
  * the goods_services field holds the whole application's description, across
    every class it names, so a flag on a class-35 row can come from that
    application's class-9 goods;
  * square brackets mark goods deleted after registration. They are kept, so the
    flag describes the offering as filed for passed and failed registrations
    alike (deleting them would remove text only from registrations that filed the
    five-year declaration).

Characteristics fixed by the Nice class itself are in CLASS_DEFINED; within a
class x registration-year cell they do not vary, so only their interaction with
lead is identified under the paper's fixed effects.

Precision of each lexicon (30 random matches read by hand, registrations
2002-2018) is recorded in paper/results/strategy_dimensions.md.
"""
from __future__ import annotations

import polars as pl

# key -> dict(label, dimension, include, exclude)
LEXICONS: dict[str, dict] = {
    # ------------------------------------------------------------ customer type
    "b2b": dict(
        dimension="Customer: businesses (B2B)",
        label="Sold to businesses",
        include=(r"\bfor (?:industrial|commercial|professional|business|institutional|laboratory|scientific|"
                 r"manufacturing|agricultural|military|hospital|veterinary|dental) "
                 r"(?:use|purposes|applications|users|settings)\b"
                 r"|\bfor use (?:by|in) (?:the )?(?:manufactur|industr|businesses|commercial|hospitals|laborator|"
                 r"professionals|construction|oil|mining|restaurants)"
                 r"|\b(?:to|for) (?:businesses|companies|corporations|enterprises|manufacturers|retailers|"
                 r"wholesalers|employers|contractors|healthcare providers|hospitals|merchants)\b"
                 r"|\bwholesale\b|\bb2b\b|business-to-business"
                 r"|\benterprise (?:software|resource|management|solutions|applications|data|networks?)"
                 r"|\bindustrial (?:machines|machinery|equipment|chemicals|robots|process|automation|gases|"
                 r"lubricants|adhesives|coatings)"
                 r"|\bbusiness (?:consulting|management|administration|advisory|process outsourcing|"
                 r"intelligence|analytics)"
                 r"|\bpayroll|\bhuman resources? (?:consulting|management|services)|\bpersonnel (?:placement|recruitment)"
                 r"|\bstaffing services|\bsupply chain|\blogistics (?:services|management)|\bbusiness consultation"
                 r"|\bdistributorships?\b|\bmarket(?:ing)? (?:research|studies|analysis)|\badvertising agenc"
                 r"|\bpromoting the (?:goods|services|goods and services|brands) of others|\bfactoring\b"
                 r"|\bpayment processing|\bwarehous\w* (?:storage|services)|\bfreight (?:forwarding|brokerage)"
                 r"|\bweb ?site (?:development|design) (?:services )?for others|\bfor others in the field of"),
        exclude=r"not for (?:industrial|commercial|professional|scientific|laboratory|business) (?:use|purposes)",
    ),
    "b2c": dict(
        dimension="Customer: end consumers (B2C)",
        label="Sold to end consumers",
        include=(r"\bconsumers?\b|\bfor (?:personal|household|home|domestic|family|private) (?:use|purposes)"
                 r"|\bhousehold (?:use|purposes|products|cleaning|appliances|furniture)"
                 r"|\bretail (?:store|stores|outlet|outlets|shop|shops|department store|bakery|pharmacy)"
                 r"|\bfor (?:babies|infants|children|kids|toddlers|teens|pets|dogs|cats|women|men)\b"
                 r"|\bpersonal care\b|\bcosmetics?\b|\btoys?\b|\bgames and playthings"),
        exclude=(r"consumer (?:research|surveys?|marketing|market research|behaviou?r|insights?|data|analytics|"
                 r"product testing|reporting|credit reporting|information management|purchas\w*|spending|buying)"
                 r"|(?:in|for) the manufacture of [\w ,]{0,60}"
                 r"|predicting (?:of )?consumers|fulfillment services[^;]{0,80}|(?:market|marketing) research (?:on|of|about) consumers?"),
    ),
    # ------------------------------------------------------------ geography
    "local": dict(
        dimension="Geographic scope: local, site-based",
        label="Local, site-based offering",
        include=(r"\brestaurants?\b|\bcafes?\b|\bcafeterias?\b|\bcoffee (?:shop|house)s?|\bbar services|\btaverns?\b"
                 r"|\bpubs?\b|\bhotels?\b|\bmotels?\b|bed and breakfast|\bresort (?:hotel|lodging)|\bsalons?\b|\bbarber"
                 r"|\bday spas?\b|\bspa services|\bhealth spa|\bfitness (?:center|centre|club|facilit|studio)"
                 r"|\bhealth clubs?\b|\bgym services|\bgyms\b|\bcar wash|\bday ?care\b|\bchild ?care\b|\bdental (?:services|clinic|"
                 r"practice)|\bdentistry\b|\bveterinary (?:services|clinic|hospital)|\bclinics?\b"
                 r"|\bretail (?:store|stores|outlet|outlets|shop|shops|department store|bakery|pharmacy)"
                 r"|\bgrocery stores?|\bconvenience stores?|\bsupermarkets?|\bbakery (?:services|shop)|\bcatering\b"
                 r"|\bdry cleaning|\blaundry services|\bplumbing (?:services|contractor)|\blandscap\w* (?:services|"
                 r"design)|\blawn care|\breal estate (?:agency|brokerage|agent)|\bmoving services|\btowing services"
                 r"|\b(?:auto|automobile|automotive|vehicle) (?:repair|service station|maintenance)|\bjanitorial"
                 r"|\bpest control services|\bextermination services|\bfuneral|\bnursing homes?|\bassisted living|\bhospital services|\boptometr"
                 r"|\bchiropract|\bphysical therapy services|\btutoring|\bpreschools?\b|\bsummer camps?|\bbowling (?:alley|center)"
                 r"|\bamusement parks?|\bmovie theat|\bnight ?clubs?"),
        exclude=(r"\bspas?,? (?:and )?hot tubs|hot tubs,? (?:and )?spas?|swimming pools? and spas?|pool and spa"
                 r"|\bspa (?:covers|pumps|filters|heaters|chemicals|equipment|parts|tubs|jets|baths)|portable spas?"
                 r"|whirlpool spas?|on-?line (?:wholesale and |and )?retail stores?|retail stores? services (?:available|provided|featuring "
                 r"\w+ )?(?:by|via|through) (?:computer|telecommunication|the internet|a global|interactive|mail)"
                 r"|(?:internet|electronic|virtual|web-based) retail stores?|retail store services in the field of (?:computer|downloadable)"
                 r"|restaurant (?:equipment|supply|supplies|furniture|reservation|consulting|management software|"
                 r"point of sale)|(?:for|to) (?:use in )?restaurants|hotel (?:reservation|booking)s?"
                 r"|\bclinical|software[^;]{0,120}|business management of [^;]{0,60}|(?:revenues?|performance|productivity|locating)[^;]{0,60}|wastewater|towels and barber"
                 r"|\bin the fields? of [^;]{0,80}|\b(?:about|regarding|concerning|featuring|of) "
                 r"(?:wineries, )?restaurants|\w+ industry|repair shop (?:customer|management)|in-home|\bgym (?:bags?|shorts|shoes|clothing|wear|equipment|mats?|towels?)"
                 r"|\bsalon (?:and|chairs?|equipment|furniture|dryers?|use|quality|products)|for (?:use in )?salons?"
                 r"|machines? for dry cleaning|dental practice (?:management|start-up|consult\w*)"
                 r"|(?:information|databases?|reviews?|guides?|director(?:y|ies)|ratings?) [\w ,]{0,40}"
                 r"(?:restaurants?|hotels?)|(?:for|from) (?:homes, offices, )?(?:hotels|grocery stores)"),
    ),
    "broad_scope": dict(
        dimension="Geographic scope: national or international",
        label="Explicitly national or international trade scope",
        include=(r"\bimport(?:s|ing)?(?:-| and | & )export|\bexport(?:ing|ation)? (?:services|agenc\w*|of|trading|"
                 r"consult\w*|promotion|management|brokerage)|\bimport (?:agenc\w*|services|brokerage|of)"
                 r"|\binternational (?:shipping|freight|delivery|courier|trade|trading|distribution|transport\w*|"
                 r"moving|logistics|sales|marketing|business|franchis\w*|money transfers?|wire transfers?|calling)"
                 r"|\bworld-?wide (?:shipping|delivery|distribution)|\bnationwide (?:shipping|delivery|distribution|"
                 r"network|chain)|\bcross-border\b|\bforeign (?:trade|markets)|\boverseas (?:shipping|markets|trade|"
                 r"distribution)|\bmulti-?national\b|\bglobal (?:shipping|logistics|distribution|trade|freight|"
                 r"sourcing|supply chain)"),
        exclude=r"in the fields? of[^;]{0,80}|multinational telecommunication|(?:information|newsletters?|papers?) (?:on|about|regarding|concerning)[^;]{0,60}",
    ),
    "bulky": dict(
        dimension="Transport costs: low value-to-weight goods",
        label="Bulky, low value-to-weight goods",
        # restricted to classes whose goods these materials are (chemicals and fertilizers, fuels,
        # building materials, pallets, agricultural products, waters)
        classes=["001", "004", "019", "020", "031", "032"],
        # the material must head a list item: after a delimiter (or "namely"/"and") and up to three
        # modifiers, and followed by a delimiter or a use phrase
        include=(r"(?:^|[;:,\[\(]|\bnamely\b|\band\b)\s*(?:[a-z-]+ ){0,3}(?:cement|concrete|ready-?mix(?:ed)? concrete"
                 r"|asphalt|gravel|sand|crushed stone|crushed rock|bricks?|lumber|timber|gypsum|drywall|wallboard"
                 r"|roofing shingles|fertili[sz]ers?|potting soil|top ?soil|mulch|coal|coke|scrap metal|corrugated "
                 r"(?:boxes|cartons)|pallets|bottled (?:drinking |spring |mineral )?water|packaged ice|bagged ice"
                 r"|building stones?|paving (?:stones|blocks)|aggregates?)\s*(?:$|[;,\]\)\.]|\band\b|\bfor "
                 r"(?:agricultural|domestic|commercial|industrial|horticultural|building|construction|use))"),
        exclude=(r"sand ?paper|sand (?:toys|art|box|pails|timers|dollars)|sand-?blast|\bcoal tar"
                 r"|concrete (?:ideas|results|steps)|in concrete|brick and mortar|brick-and-mortar"
                 r"|\b(?:for|on|to|of|into|from|in|through|with) (?:the )?(?:metal and )?(?:asphalt|concrete|cement|coal|"
                 r"sand|timber|lumber|bricks?|gravel|drywall|fertili[sz]ers?)\b"
                 r"|\b(?:asphalt|concrete|cement|coal|sand|gravel|bricks?|lumber|timber|drywall) (?:stoves?|fired|"
                 r"burning|cleaning|cutting|saws?|mixers?|machines?|tools|coatings?|sealers?|sealants?|additives|"
                 r"admixtures|surfaces|road surfaces|floors?|ramps|evaluation|appraisal|filtration|paving|"
                 r"construction|nails|screws|contractors?|contractor services|installation|repair|brokerage)"
                 r"|(?:pipe|rubber|contact|dental|bone|tile|plastic|fixing|temporary) cements?"
                 r"|(?:made|figures) of [\w ,]{0,20}(?:stone|concrete)|decorative sand|sand (?:media|filters?|yachts?)"
                 r"|aromatic sand|toy sand|packaged ice cream|(?:forest|timber) industr\w*"),
    ),
    # ------------------------------------------------------------ information
    "credence": dict(
        dimension="Credence goods and services",
        label="Credence offering (quality hard to judge even after use)",
        include=(r"\bmedical (?:services|care|treatment|diagnos\w* services|consultation|clinic)|\bhealth ?care services"
                 r"|\bdiagnostic (?:services|testing services|imaging services)|\bdental (?:services|care|treatment)"
                 r"|\bdentistry\b|\bsurg(?:ery|ical) services|\bphysician services|\bnursing (?:care|services)"
                 r"|\bpsycholog\w* (?:services|counsel|testing|therapy)|\bpsychiatr\w* (?:services|care|treatment|counsel\w*)|\bcounsel(?:l)?ing\b"
                 r"|\btherapy services|\bphysical therapy services|\bchiropract|\bveterinary services"
                 r"|\blegal services|\blaw firm|\battorney services|\baccounting\b|\bauditing\b|\btax (?:preparation|"
                 r"consult\w*|advisory|planning)|\bfinancial (?:advi[cs]\w*|planning|consult\w*)|\binvestment (?:advi[cs]\w*|"
                 r"management|counsel\w*)|\bwealth management|\b(?:auto|automobile|automotive|vehicle) (?:repair|"
                 r"maintenance)|\brepair (?:services|of)\b|\brepair and maintenance|\bmaintenance and repair"
                 r"|\b(?:home|building|property) inspection|\bdietary supplements?|\bnutritional supplements?"
                 r"|\bherbal supplements?|\bvitamins?\b|\borganic\b|\bhomeopathic"),
        exclude=(r"(?:excluding|except|not including)[^;]{0,80}|repair tools|conferences?[^;]{0,60}|software[^;]{0,120}|(?:training|seminars?|courses?|classes|instruction|educational)[^;]{0,60}"
                 r"|organic (?:material|matter|transparenc\w*|chemicals?|compounds?|light|led|oled|solvents?|pigments?|acids?|peroxides?|"
                 r"semiconductor|synthesis|electronic|electroluminescent|fibers?|polymers?|growth|search|traffic|"
                 r"results|reach|marketing|social)|counsel(?:l)?ing (?:and|or) (?:advice|consult)\w* (?:in|on) "
                 r"(?:business|marketing|advertising)|accounting software|software for accounting"),
    ),
    "experience": dict(
        dimension="Experience goods",
        label="Experience good (food, drink, personal care, entertainment, travel)",
        include=(r"\bfoods?\b|\bbeverages?\b|\bsnacks?\b|\bwines?\b|\bbeers?\b|\bliqueurs?\b|\bcoffee\b|\btea\b"
                 r"|\bcandy\b|\bconfectionery\b|\bchocolates?\b|\bsauces?\b|\bcosmetics?\b|\bfragrances?\b"
                 r"|\bperfumes?\b|\bskin ?care\b|\bhair ?care\b|\bshampoos?\b|\brestaurants?\b|\bhotels?\b"
                 r"|\bentertainment\b|\bmovies?\b|\bmotion pictures?\b|\bmusical (?:sound recordings|performances)"
                 r"|\bconcerts?\b|\bvideo games?\b|\bcomputer games?\b|\bbooks?\b|\bmagazines?\b|\bnovels?\b"
                 r"|\btravel (?:services|agency|tours)|\btours?\b|\bvacations?\b"),
        exclude=(r"cosmetics? (?:bags|cases|brushes|pouches)|(?:for|of|serving|packaging|in the fields? of) (?:\w+ )?(?:foods?|food and drinks?|beverages?)\b|tours of (?:residential|real estate|homes|propert)|food (?:processors?|containers?|storage|processing (?:machines|equipment)|slicers?|warmers?|"
                 r"service equipment|trays|safety|packaging|grade)|beverage (?:containers?|dispensers?|coolers?|"
                 r"ware|glassware|holders?|cans?)|wine (?:glasses|racks?|coolers?|openers?|bottles?|accessories|"
                 r"cellars?|aerators?|stoppers?)|beer (?:mugs?|steins?|glasses|kegs|taps|dispensers?)"
                 r"|tea (?:kettles?|pots?|cups?|sets?|towels?|strainers?|infusers?|lights?|balls?)|tea-?lights?"
                 r"|coffee (?:makers?|machines?|grinders?|mugs?|cups?|filters?|tables?|pots?)"
                 r"|(?:account|address|note|check|cheque|log|sketch|appointment|exercise|composition) books?"
                 r"|book ?(?:ends|marks|binding|cases?|keeping|shelves)|bookkeeping|books? of account"
                 r"|online booking|\bbook(?:ing)? (?:of|travel|reservations)"),
    ),
    "search": dict(
        dimension="Search goods",
        label="Search good (apparel, furniture, hardware, inspectable before purchase)",
        include=(r"\bclothing\b|\bapparel\b|\bfootwear\b|\bshoes\b|\bboots\b|\bshirts?\b|\bt-?shirts?\b|\bpants\b"
                 r"|\bjackets?\b|\bdresses\b|\bhats?\b|\bcaps\b|\bsocks\b|\bfurniture\b|\brugs?\b|\bcarpets?\b"
                 r"|\bhand tools\b|\bhardware\b|\bfasteners?\b|\bhandbags?\b|\bluggage\b|\bwallets?\b|\bbelts?\b"
                 r"|\bcurtains?\b|\bbedding\b|\btowels?\b|\btableware\b|\bcookware\b"),
        exclude=(r"cleaning[^;]{0,40}carpets|caps and clips|galvanic belts|(?:cleaning|laundering|alteration|repair|storage|rental) of clothing|towels of paper|control hardware|hardware (?:security|wallets?|keys?)|computer hardware|hardware (?:and|&) software|software (?:and|&) hardware|hardware for "
                 r"(?:computers|networking)|networking hardware|hardware design|hardware consult\w*"
                 r"|conveyor belts?|timing belts?|drive belts?|seat ?belts?|v-belts?|belts? for (?:machines|"
                 r"engines|conveyors|motors)|caps (?:for|being)|bottle caps|hub ?caps|valve caps|end caps"
                 r"|(?:for|on) (?:hats|caps|shirts|t-shirts|clothing|apparel)"),
    ),
    # ------------------------------------------------------------ durability
    "durable": dict(
        dimension="Durable goods",
        label="Durable good (machines, appliances, vehicles, furniture)",
        classes=[f"{i:03d}" for i in range(1, 35)],   # goods classes only
        include=(r"\bmachines?\b|\bmachinery\b|\bapparatus\b|\bequipment\b|\bappliances?\b|\bfurniture\b"
                 r"|\bvehicles?\b|\bautomobiles?\b|\bengines?\b|\bhand tools\b|\bpower tools\b|\bcomputers\b"
                 r"|\bcomputer hardware\b|\blaptops?\b|\btablet computers\b|\bbicycles?\b|\bboats?\b|\bwatches\b"
                 r"|\bclocks?\b|\brefrigerators?\b|\bwashing machines\b|\bcameras?\b|\btelevision (?:sets|receivers|monitors)|\btelevisions\b"
                 r"|\bmattresses\b|\bcookware\b|\bluggage\b|\bfirearms?\b|\bmusical instruments\b|\btelephones?\b"
                 r"|\bsmart ?phones\b|\bjewel(?:le)?ry\b"),
        exclude=(r"ink vehicles|appliance cleaner|toy musical|organization of[^;]{0,40}|transport services?[^;]{0,40}|luggage transport|(?:software|retail|wholesale|store services|none of the foregoing|not including|excluding)[^;]{0,150}"
                 r"|toy (?:vehicles|cars|boats|bicycles|furniture|watches)|engine (?:treatments|oils?|additives|coolants?)|equipment of others|search engines?|game engines?|machine learning|machine-?readable|machine translation"
                 r"|machine (?:vision|intelligence)|(?:rental|leasing|repair|maintenance|installation|"
                 r"cleaning|inspection|overhaul|moving|servicing) of [\w ,-]{0,60}?(?:machines|machinery|apparatus|"
                 r"equipment|appliances|vehicles|furniture|computers|engines)|(?:between|for|of) (?:\w+ ){0,2}"
                 r"(?:computers|machinery|furniture)\b|(?:installation|instruction and) (?:of )?(?:machinery|equipment)"
                 r"|information technology equipment|equipment (?:rental|leasing)|vehicle (?:insurance|financing|"
                 r"leasing|rental)|software for (?:use with |use on |operating |controlling )?(?:machines|"
                 r"equipment|vehicles|computers)|virtual machines?|slot machines?|sports equipment"),
    ),
    "consumable": dict(
        dimension="Consumables and repeat purchases",
        label="Consumable, repeat-purchase good",
        classes=[f"{i:03d}" for i in range(1, 35)],   # goods classes only
        include=(r"\bdisposable\b|\bsingle-?use\b|\brefills?\b|\breplacement (?:cartridges|filters|blades|heads|pads|"
                 r"bags|brushes)|\b(?:ink|toner|printer|filter|replacement|razor|vaporizer|e-cigarette|electronic cigarette|gas|co2) cartridges?\b|\bink\b|\btoner\b|\bbatteries\b|\bdetergents?\b|\bcleaning "
                 r"preparations\b|\bsoaps?\b|\bshampoos?\b|\btoothpaste\b|\bdiapers?\b|\btoilet (?:paper|tissue)"
                 r"|\bpaper towels\b|\bfacial tissues?\b|\btissue paper\b|\bpaper napkins\b|\bnapkins of paper\b|\bcandles?\b|\bfoods?\b|\bbeverages?\b|\bsnacks?\b"
                 r"|\bfuels?\b|\blubricants?\b|\bcosmetics\b|\bvitamins?\b|\bdietary supplements?\b|\bpet food"
                 r"|\bfertili[sz]ers?\b|\bseeds\b"),
        exclude=(r"(?:bags|containers|coolers|sacks) for (?:food|beverages)|provision of food|led candles|soap bubbles|storage batteries|energy accumulators|providing (?:of )?food|food (?:or|and) agricultural technology|food (?:technology|science|research)"
                 r"|for spraying[^;]{0,80}|batteries for (?:powering )?(?:electric )?vehicles|electric batteries for|diaper bags?|beverage (?:glassware|accessories|ware)|ink[- ]resistant|(?:chargers?|holders?|cases?) for (?:\w+ )?batteries|batteries,? electric,? for vehicles|vehicle batteries"
                 r"|fuel (?:cells?|injectors?|pumps?|tanks?|filters?|lines|systems|dispensers?|nozzles?|cards?|"
                 r"management|efficiency)|food (?:processors?|containers?|storage containers|processing machines|"
                 r"slicers?|warmers?|service equipment|trays)|beverage (?:containers?|dispensers?|coolers?)"
                 r"|soap (?:dispensers?|dishes|holders?)|candle (?:holders?|sticks?)|candlesticks?"
                 r"|(?:air|oil|water|fuel) filters? (?:for|being parts)|filters for (?:cameras|lenses|"
                 r"photography)|ink ?jet printers|battery (?:chargers?|packs? for)"),
    ),
    # ------------------------------------------------------------ customization
    "customized": dict(
        dimension="Standardized vs customized",
        label="Customized or made-to-order",
        include=(r"\bcustom[- ]?(?:made|designed|built|fitted|manufactur\w*|fabricat\w*|tailor\w*|printed|printing|"
                 r"embroider\w*|framing|jewel\w*|software|programming|design|formulat\w*|blend\w*|engineered|"
                 r"home|homes|furniture|cabinet\w*|apparel|clothing|packaging)\b|\bcustomi[sz](?:ed|ation|ing|able)\b"
                 r"|\bmade[- ]to[- ](?:order|measure)\b|\bbespoke\b|\bto the order and specification of others"
                 r"|\b(?:according|to) (?:the )?(?:customer|client)'?s?'? (?:specifications|requirements|orders?)"
                 r"|\btailor[- ]made\b|\btailored to\b|\btailoring\b|\bpersonali[sz](?:ed|ation)\b|\bmonogramm\w*"),
        exclude=r"",
    ),
    # ------------------------------------------------------------ contracting
    "subscription": dict(
        dimension="Subscription and recurring contracts",
        label="Subscription, membership or recurring contract",
        include=(r"\bsubscriptions?\b|\bsubscription-based\b|\bmembership\b|\bmonthly\b|\brecurring\b"
                 r"|\bcontinuity programs?\b|\bclub services\b|\bsoftware as a service\b|\bsaas\b"
                 r"|\bplatform as a service\b|\binfrastructure as a service\b|\bleasing\b"
                 r"|\b(?:maintenance|service) (?:contracts|agreements|plans)\b|\bextended warrant(?:y|ies)\b"),
        exclude=r"",
    ),
    "switching": dict(
        dimension="Switching costs and lock-in",
        label="Ongoing account relationship (switching costs)",
        include=(r"\baccounts?\b|\bbanking\b|\binsurance\b|\btelecommunications? services|\b(?:cellular|mobile|"
                 r"wireless) (?:telephone |phone )?(?:services|communication services)|\binternet (?:access|service "
                 r"provider)|\benterprise resource planning\b|\berp\b|\bdata (?:storage|hosting|migration)"
                 r"|\b(?:web|website|server|cloud|data|application|software) hosting|\bhosting (?:of )?(?:web ?sites?|websites?|servers?|software|applications|data|digital content|the websites)|\bpayroll\b|\bcredit cards?\b|\bdebit cards?\b|\bmortgages?\b|\bpensions?\b"
                 r"|\bretirement (?:plans|accounts)|\butility services|\belectricity (?:supply|distribution)"
                 r"|\bnatural gas (?:supply|distribution)|\bcable television (?:services|transmission)"
                 r"|\bpayment processing"),
        exclude=(r"in the fields? of[^;]{0,80}|promoting the interests of[^;]{0,80}|(?:insurance|banking) (?:claims|billing|information|industry|data)|related to insurance|(?:holding|holders? for|leashes)[^;]{0,60}credit cards|account books?|on account of|taking into account|accounts receivable|card cases|investment banking"
                 r"|(?:not including|excluding|except)[^;]{0,80}|software[^;]{0,120}"),
    ),
    # ------------------------------------------------------------ networks
    "network": dict(
        dimension="Network effects and platforms",
        label="Platform or network offering",
        include=(r"\bplatforms?\b|\bmarketplaces?\b|\bsocial network\w*|\bonline communit\w*|\bvirtual communit\w*"
                 r"|\bpeer-to-peer\b|\bp2p\b|\bconnecting (?:buyers|sellers|users|people|individuals|consumers|"
                 r"businesses|members|patients|drivers|riders|travel(?:l)?ers|employers|job seekers|"
                 r"service providers|customers)|\bmatching (?:buyers|sellers|users|people|individuals|consumers|"
                 r"employers|job seekers|riders|drivers|service providers)|\bdating services|\bride-?sharing"
                 r"|\bride-?hailing|\bcar-?sharing|\bchat rooms?|\bbulletin boards?|\bonline forums?|\buser-generated"
                 r"|\btwo-sided|\bmulti-sided|\bauction"),
        exclude=(r"in the fields? of[^;]{0,60}|computerized platform|sea platforms?|navigation platform|(?:any|computer|computing|gaming|video game|game|operating|hardware|software|mobile computing|technology|electronic) platforms?|platforms? as a service|platform solutions|bulletin board (?:backgrounds|borders|paper|sets)|platform (?:shoes|sandals|beds?|trucks?|ladders?|lifts?|scales?|rockers?|carts?|trailers?|"
                 r"tennis|diving|boots|heels|steps?|swings?|stages?|baskets?)|(?:aerial|work|lifting|drilling|oil|"
                 r"offshore|mobile|elevating|loading|hydraulic|viewing|observation|diving|truck|bed|metal|wooden|"
                 r"steel|portable|rolling|standing|access|scaffold\w*|vibration|swim) platforms?"
                 r"|platforms? (?:for|being) (?:lifting|vehicles|trucks|beds|scaffold\w*|parts)"
                 r"|advertising (?:services )?(?:via|through|on) social (?:media|network\w*)|metallic platforms|(?:mounting|ergonomic|adjustable) platforms"
                 r"|the marketplace|(?:via|across|on) (?:various|multiple|all|different) (?:media |digital )?platforms"),
    ),
    "complement": dict(
        dimension="Complementary goods",
        label="Complement to another product (for use with, adapted for)",
        include=(r"\bfor use with\b|\bcompatible with\b|\bspecially (?:adapted|designed|made|fitted) for\b"
                 r"|\baccessories for\b|\breplacement parts\b|\battachments for\b|\badapters? for\b|\brefills? for\b"
                 r"|\bcartridges for\b|\bcases for\b|\bcovers for\b|\bchargers? for\b|\bmounts? for\b"
                 r"|\bplug-?ins?\b|\badd-?ons?\b|\bupgrades? for\b|\bapps? for\b|\bskins for\b|\bstands for\b"),
        exclude=(r"plug-?in (?:connectors|relays|style|modules|units)|stands for others|accessories for the aforesaid|for use with (?:patients|children|infants|adults|people|students|clients|animals|pets)|accessories for (?:children|men|women)|skins for"),
    ),
    # ------------------------------------------------------------ regulation
    "approval": dict(
        dimension="Regulatory approval before sale",
        label="Requires premarket regulatory approval (drugs, devices, pesticides)",
        classes=[f"{i:03d}" for i in range(1, 35)],   # goods classes only
        include=(r"\bpharmaceutical|\bprescription\b|\bdrugs?\b|\bmedicines?\b|\bmedicated\b|\bvaccines?\b"
                 r"|\bbiologics?\b|\bmedical (?:devices?|apparatus|instruments|implants|equipment)|\bsurgical "
                 r"(?:apparatus|instruments|implants|devices)|\bimplants?\b|\bdiagnostic (?:preparations|reagents|"
                 r"kits|tests?|apparatus)|\bpesticides?|\bherbicides?|\binsecticides?|\bfungicides?\b"
                 r"|\bfood additives|\binfant formula|\bsunscreens?|\bcatheters?|\bstents?\b|\bcontact lenses"
                 r"|\bhearing aids"),
        exclude=(r"non medicated|(?:consult\w*|compact discs)[^;]{0,80}|incorporated into[^;]{0,40}|(?:except|excluding)[^;]{0,80}|\w+ (?:industry|market)\b|(?:books|publications|brochures|manuals|articles)[^;]{0,80}|(?:for use )?in (?:the )?manufacture[^;]{0,40}|submission[^;]{0,60}|patient and implant|in (?:pharmaceutical|hearing aids)[^;]{0,40}|(?:trays|holders|cases|chip cards) for [^;]{0,40}|information[^;]{0,40}|testing and research|prescription of|non-?medicated|non-?medical|in the fields? of[^;]{0,100}|fields of[^;]{0,100}|software[^;]{0,120}"
                 r"|(?:advertising|promoting|marketing|retail|wholesale)[^;]{0,100}|(?:drug|vaccine|pharmaceutical) (?:discovery|history|"
                 r"research|development|cost|utilization)|research (?:equipment|apparatus|instruments)|veterinary medicine"
                 r"|drug (?:stores?|rehabilitation|abuse|testing|treatment|counsel\w*|addiction|free|prevention|"
                 r"education|awareness)|drugstores?|pharmaceutical (?:consult\w*|marketing|advertising|packaging|"
                 r"research services|business)|anti-?drug|(?:prescription|drug) (?:discount|savings|benefit)"),
    ),
    "licensed": dict(
        dimension="Licensed or regulated trades",
        label="Licensed or regulated trade (alcohol, tobacco, firearms, finance, gaming, health)",
        include=(r"\balcoholic\b|\bbeers?\b|\bwines?\b|\bspirits\b|\bliquors?\b|\bvodka|\bwhiske?y|\brum\b|\bgin\b"
                 r"|\btequila|\bbourbon|\btobacco|\bcigars?\b|\bcigarettes?\b|\bfirearms?\b|\bammunition\b|\bguns?\b"
                 r"|\bexplosives?\b|\bfireworks\b|\bbanking\b|\binsurance\b|\bsecurities\b|\bbrokerage\b"
                 r"|\binvestment (?:banking|advi\w*|management|funds?)|\blending\b|\bloans?\b|\bmortgages?\b"
                 r"|\bcasinos?\b|\bgambling\b|\blotter(?:y|ies)\b|\bbetting\b|\bwagering\b|\bcannabis\b"
                 r"|\bmarijuana\b|\bpharmacy\b|\bpharmacies\b|\bmedical services|\bhospitals?\b|\bnursing homes?"
                 r"|\btelecommunications? services|\bair (?:transport|transportation|carrier)|\bairlines?\b|\btaxi\b"),
        exclude=(r"(?:excluding|except)[^;]{0,60}|in the fields? of[^;]{0,80}|residents in[^;]{0,80}|non-?alcoholic|cigar(?:ette)?s? (?:cases|lighters|boxes|holders)|(?:for|of) (?:wine|beer|liquor)s?\b"
                 r"|software[^;]{0,120}|(?:advertising|promoting|marketing)[^;]{0,100}|(?:in|for) (?:medical schools|hospitals)"
                 r"|(?:glue|spray|nail|heat|water|toy|massage|grease|paint|staple|caulking|soldering|squirt|"
                 r"tattoo|foam|bubble|nerf|radar|hot melt) guns?|cotton gin|mineral spirits|wine (?:glasses|racks?|"
                 r"coolers?|openers?|bottles?|accessories|cellars?|aerators?|stoppers?|making kits)"
                 r"|beer (?:mugs?|steins?|glasses|kegs|taps|dispensers?|making kits)|tobacco-?free|smokeless"
                 r"|(?:hospital|hospitals) (?:beds|gowns|equipment|furniture)|for (?:use in )?hospitals"),
    ),
    # ------------------------------------------------------------ brand intensity
    "ad_intensive": dict(
        dimension="Advertising-intensive consumer goods",
        label="Mass-advertised packaged consumer good (Sutton's endogenous sunk-cost categories)",
        classes=[f"{i:03d}" for i in range(1, 35)],   # goods classes only
        include=(r"\bsoft drinks\b|\bcarbonated (?:beverages|drinks|soft drinks)|\bsodas?\b|\bbeers?\b"
                 r"|\bsnack (?:foods|bars|chips|cakes)|\bpotato chips|\bcereals?\b|\bcandy\b|\bconfectionery\b"
                 r"|\bchocolates?\b|\bchewing gum|\bcookies\b|\bcrackers\b|\bdetergents?\b|\bsoaps?\b|\bshampoos?\b"
                 r"|\btoothpaste|\bdeodorants?|\bcosmetics\b|\bfragrances?\b|\bperfumes?\b|\bdiapers?\b"
                 r"|\bbottled water|\bjuices?\b|\bfrozen (?:foods|meals|pizzas?|dinners|entrees)|\bice cream\b"
                 r"|\bcondiments\b|\bketchup|\bmayonnaise|\bsalad dressings?|\bpet food"),
        exclude=(r"(?:applicators|containers?|boxes|holders?|dispensers?|bowls?) for [^;]{0,20}|soap boxes|fragrance candles|(?:except|excluding|other than|not including)[^;]{0,80}|(?:in the fields? of|featuring|consultation|trade shows?)[^;]{0,100}"
                 r"|candy boxes|cosmetic (?:spatulas|treatment|surgery)|chocolate (?:molds|fountains)|juice (?:extractors?|machines?|presses|dispensers?|bars? services)"
                 r"|juicers?|soap (?:dispensers?|dishes|holders?)|perfume (?:bottles|atomizers)|ice cream (?:makers?|"
                 r"freezers?|machines?|scoops?|parlou?rs?|shops?|stores?)|cereal bowls|candy (?:dispensers?|molds|"
                 r"machines?|stores?|shops?)|soda (?:fountains?|siphons?|machines?|ash|blasting)|baking soda"
                 r"|caustic soda|beer (?:mugs?|steins?|glasses|kegs|taps|dispensers?|gardens?|bars?|halls?)"
                 r"|cosmetic (?:surgery|dentistry|services)|cosmetics? (?:bags|cases|brushes)"),
    ),
    # ------------------------------------------------------------ technology
    "tech": dict(
        dimension="Technology intensity",
        label="R&D-intensive technology",
        include=(r"\bsoftware\b|\bsemiconductors?\b|\bintegrated circuits?\b|\bmicroprocessors?\b|\bmicro-?chips?\b"
                 r"|\bbiotechnolog\w*|\bgenetic\b|\bgenomic\w*|\bmolecular\b|\bnanotechnolog\w*|\bnanoparticles?"
                 r"|\bnanomaterials?|\blasers?\b|\boptical fib\w*|\brobot\w*|\bartificial intelligence|\bmachine learning"
                 r"|\bwireless (?:communication|network|technolog)\w*|\bscientific research|\bresearch and development"
                 r"|\bclinical trials?|\bmedical imaging|\bmedical devices?|\bsensors?\b|\bpharmaceutical\w*"
                 r"|\baerospace\b|\bsatellites?\b|\bfuel cells?|\bphotovoltaic|\bsolar (?:cells|panels|modules)"
                 r"|\bdata processing|\bcomputer (?:hardware|programming)|\belectronic (?:circuits|"
                 r"components|devices|control)|\bcloud computing|\bencryption|\bcyber ?security"),
        exclude=(r"satellite (?:broadcast\w*|television|transmission)|(?:via|by|cable,) satellite|(?:via|through) (?:public and private )?wireless networks|(?:for|with|of|avoid obstructions with) (?:\w+ )?(?:medical devices|pharmaceuticals|medicines)|laser tag|laser pointers?|(?:pharmaceutical|software) (?:marketing|advertising)"),
    ),
    "digital": dict(
        dimension="Information and digital goods",
        label="Digital or information good (near-zero marginal cost)",
        include=(r"\bdownloadable\b|\bsoftware\b|\bnon-?downloadable\b|\bdigital (?:media|content|music|video|files|"
                 r"images|books|publications|downloads|recordings)|\bstreaming\b|\bpodcasts?\b|\be-?books?\b"
                 r"|\belectronic (?:publications|books|newsletters|journals|magazines)|\bonline (?:publications|"
                 r"journals|magazines|newsletters|databases?|videos?|games?)|\bvideo games?\b|\bcomputer games?\b"
                 r"|\bmobile (?:apps|applications)|\bdatabases?\b|\bwebcasts?\b|\bwebinars?\b"),
        exclude=r"database servers|video game (?:facilities|arcades?)",
    ),
    # ------------------------------------------------------------ channels
    "intermediary": dict(
        dimension="Intermediated channels",
        label="Intermediary (retail, wholesale, distribution, brokerage)",
        include=(r"\bwholesale\b|\bdealerships?\b|\bdistributorships?\b|\bdistribution services|\bretail (?:store|stores|outlet|"
                 r"outlets|department store|services|sale|shops?)|\bdepartment stores?|\bimport(?:ing)? "
                 r"(?:and|&) export|\bimport agency|\bexport agency|\bbrokerage\b|\bbroker(?:s|ing)? services"
                 r"|\bsales agency|\bcommission agents?|\bfranchis\w*|\bauctioneering|\bonline marketplace"
                 r"|\bprocurement services for others|\bpurchasing agents?"),
        exclude=r"(?:on-?line|internet|electronic|computerized) (?:wholesale and |and )?retail[\w ]{0,20}|retail store shelving|(?:consult\w*|advisory)[^;]{0,40}franchis\w*",
    ),
    "direct": dict(
        dimension="Direct channels",
        label="Direct-to-buyer channel (mail order, direct selling, online ordering)",
        include=(r"\bmail order\b|\bmail-order\b|\bcatalog(?:ue)? (?:ordering|sales|shopping|services)"
                 r"|\bdirect (?:selling|sales)\b|\bdoor-to-door\b|\bhome parties\b|\bparty plan\b|\bmulti-?level marketing"
                 r"|\bnetwork marketing\b|\bonline ordering\b|\bonline retail store|\bon-?line retail\b"
                 r"|\be-?commerce\b|\bdirect-to-consumer\b|\bvending machine services|\btelevision shopping"
                 r"|\bhome shopping|\btelephone (?:ordering|shopping)"),
        exclude=(r"(?:facilitating|computer) e-?commerce|e-?commerce (?:software|hardware|platforms?|solutions|consult\w*)|software[^;]{0,100}"
                 r"|(?:in the fields? of|guides? in the field of)[^;]{0,80}|vending machines? (?:and|for|mechanisms)|automatic vending machines"),
    ),
    # ------------------------------------------------------------ perishability
    "perishable": dict(
        dimension="Perishability",
        label="Perishable good",
        # foods, agricultural products and drinks, plus retail and food service
        classes=["029", "030", "031", "032", "035", "043"],
        # the perishable food must head a list item (after a delimiter or "namely", up to two modifiers,
        # then a delimiter), so that "fish sauce" or "fruit-filled candy" do not count
        include=(r"\bperishable\b|\blive (?:animals|fish|plants|lobsters?|seafood|poultry|crabs|shellfish|bait|cattle|trees)"
                 r"|\bcut flowers|\bnatural flowers|\bfresh (?:fruits?|vegetables|produce|flowers|meats?|fish|seafood|"
                 r"poultry|herbs|juices?|pasta|bread|baked goods|salads?|sandwiches)|\bflorists?\b|\bbouquets?\b"
                 r"|(?:^|[;:,\[\(]|\bnamely\b|\band\b)\s*(?:[a-z-]+ ){0,2}(?:milk|cheeses?|yog(?:h)?urts?|meats?|poultry|"
                 r"seafood|fish|eggs|bread|breads|pastr(?:y|ies)|baked goods|fruits?|vegetables|sandwiches|salads|"
                 r"flowers|cakes|donuts|doughnuts|tortillas|sushi)\s*(?:$|[;,\]\)\.]|\band\b)"),
        exclude=(r"(?:fish|pepper|soy|oyster) sauce|preserves|fish meal|fruit-filled|milk ?shakes?|ice cream \w+|prepared meat|dairy-based|(?:canned|jarred|frozen|dried|dehydrated|preserved|processed|pickled|freeze-dried|powdered|candied|covered|"
                 r"cured|smoked|condensed|evaporated|shelf-stable|uht)[\w ,]{0,40}|excluding[^;]{0,80}"
                 r"|(?:fruit|vegetable|milk|meat|fish|cheese|flower)s?(?:-based| based| flavou?red)? (?:juices?|beverages?|drinks?|"
                 r"oils?|extracts?|fibers?|powders?|substitutes?|snacks?|chips|flavou?r\w*|teas?)|(?:seeds|planters?|pots?|"
                 r"holders?) for (?:\w+ )?(?:flowers|plants|fruits?|vegetables)|bread (?:improvers|crumbs|mixes)|flowers or leaves"
                 r"|vegetarian meat|meat (?:analogues|alternatives)|fresh (?:breath|scent\w*|fragrance|look|start|ideas|air|feel\w*)|breath fresh\w*|air fresh\w*"
                 r"|artificial (?:flowers|fruits?|plants)|(?:silk|paper|plastic|dried) flowers|flower (?:pots|vases)"
                 r"|milk (?:frothers?|bottles?|jugs?|of magnesia)|(?:cleansing|body|bath|hair|skin) milk"
                 r"|meat (?:grinders?|slicers?|thermometers?|tenderizers?|hooks?|substitutes?)|fish (?:tanks?|hooks?|"
                 r"food|finders?|lures?)|(?:easter|chocolate|toy) eggs|egg (?:cookers?|timers?)"
                 r"|fruit (?:flavou?red|scented)|fruit-flavou?red|bread (?:makers?|machines?|boxes?|knives)"),
    ),
    # ------------------------------------------------------------ fads
    "fad": dict(
        dimension="Fashion and fad dynamics",
        label="Fashion, novelty, seasonal or collectible",
        include=(r"\bfashion(?:s|able)?\b|\bnovelty\b|\bnovelties\b|\bcostumes?\b|\bseasonal\b|\bhalloween\b|\bchristmas\b"
                 r"|\bholiday (?:decorations|ornaments|gifts)|\bsouvenirs?\b|\bcollectibles?\b|\bcollectable\b"
                 r"|\btrading cards\b|\bfads?\b|\btrendy\b|\btrendsetting\b|\bdesigner (?:clothing|handbags|"
                 r"apparel|jewel\w*|goods|eyewear|sunglasses|fashions)"),
        exclude=(r", fashion,|(?:in the areas? of|areas of)[^;]{0,80}|costume stands?|(?:in the fields? of|related to|featuring|regarding|concerning|about|topics?)[^;]{0,80}"
                 r"|(?:bathing|folk|swimming|bath) costumes|novelty (?:search\w*|searching)|trend (?:analysis|forecast\w*|data|reports?)|market trends"),
    ),
    # ------------------------------------------------------------ positioning
    "luxury": dict(
        dimension="Luxury positioning",
        label="Luxury or premium positioning",
        include=(r"\bluxury\b|\bluxurious\b|\bpremium\b|\bgourmet\b|\bhaute couture\b|\bfine (?:jewel\w*|dining|"
                 r"wines?|china|art|chocolates?|fragrances?)|\b(?:made|articles|goods|jewel\w*|watches) (?:of|in) precious metals?|\bprecious (?:stones|gems)|\bdiamonds?\b"
                 r"|\bplatinum\b|\bcashmere\b|\bsilk\b|\bchampagne\b|\bcognac\b|\bcaviar\b|\btruffles\b"
                 r"|\byachts?\b|\bprivate (?:jets?|aviation|banking|clubs?)|\bconcierge\b|\bchauffeur\w*|\blimousines?\b"
                 r"|\bhandcrafted\b|\bhand-?made\b|\bartisan(?:al)?\b|\bboutiques?\b|\bhigh-?end\b|\bupscale\b"
                 r"|\bprestige\b"),
        exclude=(r"silk[- ]?screen\w*|silk thread|diamond wire|not (?:made )?of precious metals?|diamond (?:tools|saws?|blades|drill\w*|grinding|abrasives?|"
                 r"coatings?|wheels|bits)|for use in making diamond|luxury vinyl|unwrought[^;]{0,40}|(?:treatment|recycling|refining|"
                 r"sorting|recovery) of [^;]{0,60}precious metals|insurance premiums?|premium (?:financ\w*|payment|collection)|silk-?screen\w*|silk flowers"
                 r"|gold (?:colou?r\w*|tone|finish)|platinum (?:catalysts?|electrodes?)|chocolate truffles"),
    ),
    "discount": dict(
        dimension="Mass or discount positioning",
        label="Discount or value positioning",
        include=(r"\bdiscount (?:stores?|retail|department|variety|outlet|merchandise|brokerage|securities brokerage|"
                 r"travel|airline|tickets|grocery|supermarket|pharmacy)|\bvalue-?priced\b|\blow-?cost\b|\blow prices?\b"
                 r"|\bdollar stores?\b|\boff-?price\b|\bwarehouse clubs?\b|\bbargains?\b|\bclose-?outs?\b|\binexpensive\b"
                 r"|\bcheap\b|\bbudget (?:hotels?|motels?|accommodations?|lodging|travel|airlines?|car rental)"
                 r"|\beconomy (?:hotels?|motels?|lodging|car rental)|\baffordable (?:housing|homes|prices?)"),
        exclude=(r"budget(?:ing)? (?:planning|analysis|preparation|management|consult\w*|forecast\w*)|budgetary"
                 r"|(?:customs|security|medical|tax) clearance|economy (?:class|section)|sharing economy"
                 r"|economic|discount (?:rate|window)"),
    ),
}

# Characteristics fixed by the Nice class. Classes are three-digit strings as in
# the frame's `cls` column.
CLASS_DEFINED: dict[str, dict] = {
    "services_cls": dict(dimension="Goods vs services", label="Services (Nice classes 35-45)",
                         classes=[f"{i:03d}" for i in range(35, 46)]),
    "approval_cls": dict(dimension="Regulatory approval before sale",
                         label="Pharmaceuticals and medical devices (classes 5, 10)", classes=["005", "010"]),
    "ad_intensive_cls": dict(dimension="Advertising-intensive consumer goods",
                             label="Packaged consumer goods (classes 3, 29-33)",
                             classes=["003", "029", "030", "031", "032", "033"]),
    "fad_cls": dict(dimension="Fashion and fad dynamics", label="Fashion goods (classes 14, 18, 25)",
                    classes=["014", "018", "025"]),
    "tech_cls": dict(dimension="Technology intensity", label="Electronics, software and IT services (classes 9, 42)",
                     classes=["009", "042"]),
    "perishable_cls": dict(dimension="Perishability", label="Fresh and processed foods (classes 29-31)",
                           classes=["029", "030", "031"]),
}

# Characteristics judged not identifiable from goods/services text (see the .md)
# Hand review (scripts/strategy_dimensions.py review <key> m|n <seed>) on a uniform sample of
# 40,000 frame registrations: (true positives among the matches read, matches read, seed of the
# match draw, clear misses among 30 non-matches read with seed 21). Each lexicon was revised
# until its false positives had no common pattern left; the counts are from the last reading.
# After that reading only exclusions were added, except "dealerships" (intermediary).
PRECISION: dict[str, tuple[int, int, int, int | None]] = {
    "b2b": (28, 30, 14, 7), "b2c": (27, 30, 11, 10), "local": (26, 30, 12, 3),
    "broad_scope": (26, 30, 11, 0), "bulky": (29, 30, 11, 0), "credence": (27, 30, 12, 1),
    "experience": (29, 30, 11, 3), "search": (26, 30, 11, 1), "durable": (25, 30, 11, 4),
    "consumable": (25, 30, 11, 4), "customized": (30, 30, 11, 0), "subscription": (29, 30, 11, 0),
    "switching": (25, 30, 12, 1), "network": (25, 30, 12, 0), "complement": (27, 30, 12, 1),
    "approval": (26, 30, 15, 0), "licensed": (27, 30, 12, 1), "ad_intensive": (26, 30, 12, 2),
    "tech": (27, 30, 12, 1), "digital": (29, 30, 12, 1), "intermediary": (29, 30, 12, 1),
    "direct": (25, 30, 12, 0), "perishable": (27, 30, 13, 0), "fad": (28, 30, 15, 1),
    "luxury": (27, 30, 13, 0), "discount": (12, 15, 3, None),
}

NOT_IDENTIFIED = {
    "standardized": "Standardization is the default and is never stated; only its absence (customization) is.",
    "switching_cost_size": "Text names the relationship (an account, a contract), not the cost of leaving it.",
    "network_size": "Text names a platform; whether it shows network effects at scale is not observable.",
    "ad_spend": "Advertising intensity is an industry attribute; text gives the category, not spending.",
    "transport_cost": "Only the good's bulk is named; distances and freight costs are not.",
}


def flag_exprs(text_col: str = "gs", cls_col: str = "cls") -> list[pl.Expr]:
    """One boolean column per text lexicon, named f_<key>, over a lowercased text column.
    A lexicon with a `classes` entry also requires the registration's class to be listed."""
    out = []
    for k, lx in LEXICONS.items():
        t = pl.col(text_col)
        if lx["exclude"]:
            t = t.str.replace_all(lx["exclude"], " ")
        e = t.str.contains(lx["include"])
        if lx.get("classes"):
            e = e & pl.col(cls_col).is_in(lx["classes"])
        out.append(e.alias(f"f_{k}"))
    return out


def class_flag_exprs(cls_col: str = "cls") -> list[pl.Expr]:
    return [pl.col(cls_col).is_in(v["classes"]).alias(f"f_{k}") for k, v in CLASS_DEFINED.items()]


def first_match(text: str, key: str) -> str | None:
    """The first matched span with context, for hand review (Python re; same patterns)."""
    import re
    lx = LEXICONS[key]
    t = text.lower()
    if lx["exclude"]:
        t = re.sub(lx["exclude"], " ", t)
    m = re.search(lx["include"], t)
    if not m:
        return None
    a, b = max(0, m.start() - 70), min(len(t), m.end() + 70)
    return t[a:m.start()] + "<<" + t[m.start():m.end()] + ">>" + t[m.end():b]
