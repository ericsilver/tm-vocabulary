# Strategy-dimensions references (verified)

Verified 2026-10-05 against the Crossref REST API (`api.crossref.org/works`, query.bibliographic on title + first author + year, then the full record fetched by DOI), with Semantic Scholar (`api.semanticscholar.org/graph/v1/paper/search`) as fallback for items Crossref does not index. OpenAlex was not used. A reference is marked VERIFIED only when title, first author and year agree with the API record; fields (authors, title, journal, volume, number, pages, DOI) are as the API returned them, with any normalization stated in `note`. Where the same article has both a JSTOR (10.2307) and a publisher DOI, the publisher DOI is given. Pre-DOI American Economic Review articles and the two books are not in Crossref; those that Semantic Scholar did not return (HTTP 429 rate limiting, repeated on a second pass the same day) are marked UNVERIFIED: Sutton (1991), Katz and Shapiro (1985), Schmalensee (1982) and Klepper (1996). Shapiro and Varian (1999) was confirmed on the second Semantic Scholar pass.

```bibtex
@article{lieberman1988,
  author  = {Lieberman, Marvin B. and Montgomery, David B.},
  year    = {1988},
  title   = {First-mover advantages},
  journal = {Strategic Management Journal},
  volume  = {9},
  number  = {S1},
  pages   = {41--58},
  doi     = {10.1002/smj.4250090706},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{lieberman1998,
  author  = {Lieberman, Marvin B. and Montgomery, David B.},
  year    = {1998},
  title   = {First-mover (dis)advantages: retrospective and link with the resource-based view},
  journal = {Strategic Management Journal},
  volume  = {19},
  number  = {12},
  pages   = {1111--1125},
  doi     = {10.1002/(sici)1097-0266(1998120)19:12<1111::aid-smj21>3.0.co;2-w},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{robinson1985,
  author  = {Robinson, William T. and Fornell, Claes},
  year    = {1985},
  title   = {Sources of Market Pioneer Advantages in Consumer Goods Industries},
  journal = {Journal of Marketing Research},
  volume  = {22},
  number  = {3},
  pages   = {305--317},
  doi     = {10.1177/002224378502200306},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{robinson1988,
  author  = {Robinson, William T.},
  year    = {1988},
  title   = {Sources of Market Pioneer Advantages: The Case of Industrial Goods Industries},
  journal = {Journal of Marketing Research},
  volume  = {25},
  number  = {1},
  pages   = {87--94},
  doi     = {10.1177/002224378802500109},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{golder1993,
  author  = {Golder, Peter N. and Tellis, Gerard J.},
  year    = {1993},
  title   = {Pioneer Advantage: Marketing Logic or Marketing Legend?},
  journal = {Journal of Marketing Research},
  volume  = {30},
  number  = {2},
  pages   = {158--170},
  doi     = {10.1177/002224379303000203},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{song1999,
  author  = {Song, X. Michael and Di Benedetto, C. Anthony and Zhao, Yuzhen Lisa},
  year    = {1999},
  title   = {Pioneering advantages in manufacturing and service industries: empirical evidence from nine countries},
  journal = {Strategic Management Journal},
  volume  = {20},
  number  = {9},
  pages   = {811--835},
  doi     = {10.1002/(sici)1097-0266(199909)20:9<811::aid-smj52>3.0.co;2-#},
  note    = {VERIFIED via Crossref; Crossref splits the second author as given 'C. Anthony Di', family 'Benedetto'; rendered here as Di Benedetto}
}
```

```bibtex
@article{zeithaml1985,
  author  = {Zeithaml, Valarie A. and Parasuraman, A. and Berry, Leonard L.},
  year    = {1985},
  title   = {Problems and Strategies in Services Marketing},
  journal = {Journal of Marketing},
  volume  = {49},
  number  = {2},
  pages   = {33--46},
  doi     = {10.1177/002224298504900203},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{bresnahan1991,
  author  = {Bresnahan, Timothy F. and Reiss, Peter C.},
  year    = {1991},
  title   = {Entry and Competition in Concentrated Markets},
  journal = {Journal of Political Economy},
  volume  = {99},
  number  = {5},
  pages   = {977--1009},
  doi     = {10.1086/261786},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{syverson2004,
  author  = {Syverson, Chad},
  year    = {2004},
  title   = {Market Structure and Productivity: A Concrete Example},
  journal = {Journal of Political Economy},
  volume  = {112},
  number  = {6},
  pages   = {1181--1222},
  doi     = {10.1086/424743},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@book{sutton1991,
  author    = {Sutton, John},
  year      = {1991},
  title     = {Sunk Costs and Market Structure},
  publisher = {MIT Press},
  note      = {UNVERIFIED: book not indexed in Crossref (queries incl. publisher MIT Press returned only reviews of it); Semantic Scholar rate-limited (HTTP 429)}
}
```

```bibtex
@article{nelson1970,
  author  = {Nelson, Phillip},
  year    = {1970},
  title   = {Information and Consumer Behavior},
  journal = {Journal of Political Economy},
  volume  = {78},
  number  = {2},
  pages   = {311--329},
  doi     = {10.1086/259630},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{nelson1974,
  author  = {Nelson, Phillip},
  year    = {1974},
  title   = {Advertising as Information},
  journal = {Journal of Political Economy},
  volume  = {82},
  number  = {4},
  pages   = {729--754},
  doi     = {10.1086/260231},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{darby1973,
  author  = {Darby, Michael R. and Karni, Edi},
  year    = {1973},
  title   = {Free Competition and the Optimal Amount of Fraud},
  journal = {The Journal of Law and Economics},
  volume  = {16},
  number  = {1},
  pages   = {67--88},
  doi     = {10.1086/466756},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{dulleck2006,
  author  = {Dulleck, Uwe and Kerschbamer, Rudolf},
  year    = {2006},
  title   = {On Doctors, Mechanics, and Computer Specialists: The Economics of Credence Goods},
  journal = {Journal of Economic Literature},
  volume  = {44},
  number  = {1},
  pages   = {5--42},
  doi     = {10.1257/002205106776162717},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{coase1972,
  author  = {Coase, R. H.},
  year    = {1972},
  title   = {Durability and Monopoly},
  journal = {The Journal of Law and Economics},
  volume  = {15},
  number  = {1},
  pages   = {143--149},
  doi     = {10.1086/466731},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{bulow1982,
  author  = {Bulow, Jeremy I.},
  year    = {1982},
  title   = {Durable-Goods Monopolists},
  journal = {Journal of Political Economy},
  volume  = {90},
  number  = {2},
  pages   = {314--332},
  doi     = {10.1086/261058},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{suarez1995,
  author  = {Suárez, Fernando F. and Utterback, James M.},
  year    = {1995},
  title   = {Dominant designs and the survival of firms},
  journal = {Strategic Management Journal},
  volume  = {16},
  number  = {6},
  pages   = {415--430},
  doi     = {10.1002/smj.4250160602},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{utterback1975,
  author  = {Utterback, James M and Abernathy, William J},
  year    = {1975},
  title   = {A dynamic model of process and product innovation},
  journal = {Omega},
  volume  = {3},
  number  = {6},
  pages   = {639--656},
  doi     = {10.1016/0305-0483(75)90068-7},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{klemperer1987,
  author  = {Klemperer, Paul},
  year    = {1987},
  title   = {Markets with Consumer Switching Costs},
  journal = {The Quarterly Journal of Economics},
  volume  = {102},
  number  = {2},
  pages   = {375},
  doi     = {10.2307/1885068},
  note    = {VERIFIED via Crossref; Crossref gives start page only}
}
```

```bibtex
@article{klemperer1995,
  author  = {Klemperer, P.},
  year    = {1995},
  title   = {Competition when Consumers have Switching Costs: An Overview with Applications to Industrial Organization, Macroeconomics, and International Trade},
  journal = {The Review of Economic Studies},
  volume  = {62},
  number  = {4},
  pages   = {515--539},
  doi     = {10.2307/2298075},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@incollection{farrell2007,
  author    = {Farrell, Joseph and Klemperer, Paul},
  year      = {2007},
  title     = {Coordination and Lock-In: Competition with Switching Costs and Network Effects},
  booktitle = {Handbook of Industrial Organization},
  volume    = {3},
  pages     = {1967--2072},
  publisher = {Elsevier},
  isbn      = {9780444824356},
  doi       = {10.1016/s1573-448x(06)03031-7},
  note      = {VERIFIED via Crossref; Crossref title prefix 'Chapter 31' dropped; volume 3 from Crossref container title 'Handbook of Industrial Organization Volume 3'; editors not in Crossref record}
}
```

```bibtex
@article{katz1985,
  author  = {Katz, Michael L. and Shapiro, Carl},
  year    = {1985},
  title   = {Network Externalities, Competition, and Compatibility},
  journal = {American Economic Review},
  note    = {UNVERIFIED: not indexed in Crossref (pre-DOI AER); Semantic Scholar rate-limited (HTTP 429)}
}
```

```bibtex
@article{rochet2003,
  author  = {Rochet, Jean-Charles and Tirole, Jean},
  year    = {2003},
  title   = {Platform Competition in Two-Sided Markets},
  journal = {Journal of the European Economic Association},
  volume  = {1},
  number  = {4},
  pages   = {990--1029},
  doi     = {10.1162/154247603322493212},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{suarez2007,
  author  = {Suarez, Fernando F. and Lanzolla, Gianvito},
  year    = {2007},
  title   = {The Role of Environmental Dynamics in Building a First Mover Advantage Theory},
  journal = {Academy of Management Review},
  volume  = {32},
  number  = {2},
  pages   = {377--392},
  doi     = {10.5465/amr.2007.24349587},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{teece1986,
  author  = {Teece, David J.},
  year    = {1986},
  title   = {Profiting from technological innovation: Implications for integration, collaboration, licensing and public policy},
  journal = {Research Policy},
  volume  = {15},
  number  = {6},
  pages   = {285--305},
  doi     = {10.1016/0048-7333(86)90027-2},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{adner2010,
  author  = {Adner, Ron and Kapoor, Rahul},
  year    = {2010},
  title   = {Value creation in innovation ecosystems: how the structure of technological interdependence affects firm performance in new technology generations},
  journal = {Strategic Management Journal},
  volume  = {31},
  number  = {3},
  pages   = {306--333},
  doi     = {10.1002/smj.821},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{grabowski1992,
  author  = {Grabowski, Henry G. and Vernon, John M.},
  year    = {1992},
  title   = {Brand Loyalty, Entry, and Price Competition in Pharmaceuticals after the 1984 Drug Act},
  journal = {The Journal of Law and Economics},
  volume  = {35},
  number  = {2},
  pages   = {331--350},
  doi     = {10.1086/467257},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{scottmorton1999,
  author  = {Scott Morton, Fiona M.},
  year    = {1999},
  title   = {Entry Decisions in the Generic Pharmaceutical Industry},
  journal = {The RAND Journal of Economics},
  volume  = {30},
  number  = {3},
  pages   = {421},
  doi     = {10.2307/2556056},
  note    = {VERIFIED via Crossref; Crossref records the author as given 'Fiona M. Scott', family 'Morton'; rendered here as Scott Morton. Crossref gives start page only}
}
```

```bibtex
@article{schmalensee1982,
  author  = {Schmalensee, Richard},
  year    = {1982},
  title   = {Product Differentiation Advantages of Pioneering Brands},
  journal = {American Economic Review},
  note    = {UNVERIFIED: not indexed in Crossref (pre-DOI AER); Semantic Scholar rate-limited (HTTP 429)}
}
```

```bibtex
@article{bronnenberg2009,
  author  = {Bronnenberg, Bart J. and Dhar, Sanjay K. and Dubé, Jean‐Pierre H.},
  year    = {2009},
  title   = {Brand History, Geography, and the Persistence of Brand Shares},
  journal = {Journal of Political Economy},
  volume  = {117},
  number  = {1},
  pages   = {87--115},
  doi     = {10.1086/597301},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{christensen1996,
  author  = {Christensen, Clayton M. and Bower, Joseph L.},
  year    = {1996},
  title   = {Customer power, strategic investment, and the failure of leading firms},
  journal = {Strategic Management Journal},
  volume  = {17},
  number  = {3},
  pages   = {197--218},
  doi     = {10.1002/(sici)1097-0266(199603)17:3<197::aid-smj804>3.0.co;2-u},
  note    = {VERIFIED via Crossref; Crossref title and names are all-caps; case normalized here}
}
```

```bibtex
@article{cohen1990,
  author  = {Cohen, Wesley M. and Levinthal, Daniel A.},
  year    = {1990},
  title   = {Absorptive Capacity: A New Perspective on Learning and Innovation},
  journal = {Administrative Science Quarterly},
  volume  = {35},
  number  = {1},
  pages   = {128},
  doi     = {10.2307/2393553},
  note    = {VERIFIED via Crossref; Crossref gives start page only}
}
```

```bibtex
@article{spulber1996,
  author  = {Spulber, Daniel F},
  year    = {1996},
  title   = {Market Microstructure and Intermediation},
  journal = {Journal of Economic Perspectives},
  volume  = {10},
  number  = {3},
  pages   = {135--152},
  doi     = {10.1257/jep.10.3.135},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{brynjolfsson2000,
  author  = {Brynjolfsson, Erik and Smith, Michael D.},
  year    = {2000},
  title   = {Frictionless Commerce? A Comparison of Internet and Conventional Retailers},
  journal = {Management Science},
  volume  = {46},
  number  = {4},
  pages   = {563--585},
  doi     = {10.1287/mnsc.46.4.563.12061},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{gallego1994,
  author  = {Gallego, Guillermo and van Ryzin, Garrett},
  year    = {1994},
  title   = {Optimal Dynamic Pricing of Inventories with Stochastic Demand over Finite Horizons},
  journal = {Management Science},
  volume  = {40},
  number  = {8},
  pages   = {999--1020},
  doi     = {10.1287/mnsc.40.8.999},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{bikhchandani1992,
  author  = {Bikhchandani, Sushil and Hirshleifer, David and Welch, Ivo},
  year    = {1992},
  title   = {A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades},
  journal = {Journal of Political Economy},
  volume  = {100},
  number  = {5},
  pages   = {992--1026},
  doi     = {10.1086/261849},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{pesendorfer1995,
  author  = {Pesendorfer, Wolfgang},
  year    = {1995},
  title   = {Design Innovation and Fashion Cycles},
  journal = {The American Economic Review},
  volume  = {85},
  pages   = {771--792},
  note    = {VERIFIED via Semantic Scholar (CorpusId 153391201); no DOI (not indexed in Crossref); issue number not returned}
}
```

```bibtex
@article{bagwell1996,
  author  = {Bagwell, L. and Bernheim, B.},
  year    = {1996},
  title   = {Veblen Effects in a Theory of Conspicuous Consumption},
  journal = {The American Economic Review},
  volume  = {86},
  pages   = {349--373},
  note    = {VERIFIED via Semantic Scholar (CorpusId 153020226); no DOI (not indexed in Crossref); S2 gives initials only; issue number not returned}
}
```

```bibtex
@article{amaldoss2005,
  author  = {Amaldoss, Wilfred and Jain, Sanjay},
  year    = {2005},
  title   = {Pricing of Conspicuous Goods: A Competitive Analysis of Social Effects},
  journal = {Journal of Marketing Research},
  volume  = {42},
  number  = {1},
  pages   = {30--42},
  doi     = {10.1509/jmkr.42.1.30.56883},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@book{shapiro1999,
  author    = {Shapiro, Carl and Varian, Hal R.},
  year      = {1999},
  title     = {Information Rules: A Strategic Guide to the Network Economy},
  publisher = {Harvard Business School Press},
  pages     = {352},
  note      = {VERIFIED via Semantic Scholar (CorpusId 44521637; C. Shapiro, H. Varian, 1999, "Information rules - a strategic guide to the network economy", pp. I-X, 1-352). Not in Crossref. The DOI Semantic Scholar attaches (10.2307/1183273) is a JSTOR record, not the book, and is omitted}
}
```

```bibtex
@article{bakos1999,
  author  = {Bakos, Yannis and Brynjolfsson, Erik},
  year    = {1999},
  title   = {Bundling Information Goods: Pricing, Profits, and Efficiency},
  journal = {Management Science},
  volume  = {45},
  number  = {12},
  pages   = {1613--1630},
  doi     = {10.1287/mnsc.45.12.1613},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{kerin1992,
  author  = {Kerin, Roger A. and Varadarajan, P. Rajan and Peterson, Robert A.},
  year    = {1992},
  title   = {First-Mover Advantage: A Synthesis, Conceptual Framework, and Research Propositions},
  journal = {Journal of Marketing},
  volume  = {56},
  number  = {4},
  pages   = {33--52},
  doi     = {10.1177/002224299205600404},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{boulding2003,
  author  = {Boulding, William and Christen, Markus},
  year    = {2003},
  title   = {Sustainable Pioneering Advantage? Profit Implications of Market Entry Order},
  journal = {Marketing Science},
  volume  = {22},
  number  = {3},
  pages   = {371--392},
  doi     = {10.1287/mksc.22.3.371.17736},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{shankar1998,
  author  = {Shankar, Venkatesh and Carpenter, Gregory S. and Krishnamurthi, Lakshman},
  year    = {1998},
  title   = {Late Mover Advantage: How Innovative Late Entrants Outsell Pioneers},
  journal = {Journal of Marketing Research},
  volume  = {35},
  number  = {1},
  pages   = {54--70},
  doi     = {10.1177/002224379803500107},
  note    = {VERIFIED via Crossref}
}
```

```bibtex
@article{klepper1996,
  author  = {Klepper, Steven},
  year    = {1996},
  title   = {Entry, Exit, Growth, and Innovation over the Product Life Cycle},
  journal = {American Economic Review},
  note    = {UNVERIFIED: not indexed in Crossref (pre-DOI AER); Semantic Scholar rate-limited (HTTP 429)}
}
```

