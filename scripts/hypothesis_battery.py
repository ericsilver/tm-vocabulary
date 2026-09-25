"""Where does being early pay? One estimator, many groupings.

Every test is the paper's gate estimate run inside groups: first-gate failure
(the five-year proof of continued use) on lead quintiles cut within class x
registration-year cells, with the paper's controls (counsel, intent-to-use,
log description length, log owner filing count, owner domicile, foreign
basis), class x registration-year fixed effects, and standard errors
clustered on the normalized owner name. The quintile dummies are interacted
with the group, so each group gets its own "penalty" -- the failure-rate gap,
in percentage points, between the most leading and the most lagging fifth --
and the contrast between groups is a coefficient with a clustered SE.

Each hypothesis carries one prediction stated before estimation: the sign of
a single contrast. p-values for those contrasts are adjusted together (Holm).

Stage 1 builds the feature frame (cached); stage 2 runs the fits.

Output: paper/results/hypothesis_battery.json
        data/processed/battery_frame.parquet (cache)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from gate_decisive_regression import build_frame, norm_owner  # noqa: E402
import geo_cluster as gc_  # noqa: E402

PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CACHE = PROC / "battery_frame.parquet"
PRE = PROC / "battery_frame_pregeo.parquet"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
LAST_FY = 2025


def log(m):
    print(m, file=sys.stderr, flush=True)


# --------------------------------------------------------------------- vocab
VOCAB = {
    # general-purpose technologies (hypothesis 25)
    "internet": r"\binternet\b|\bonline\b|\bon-line\b|\bweb ?sites?\b|\bweb pages?\b|\bworld wide web\b|\be-?commerce\b|\belectronic commerce\b",
    "mobile apps": r"mobile app|mobile application|smartphone app|mobile device application|downloadable mobile",
    "cloud / SaaS": r"cloud computing|cloud-based|cloud based|software as a service|\bsaas\b|platform as a service",
    "social media": r"social media|social network",
    "ai": r"artificial intelligence|machine learning|deep learning|neural network|natural language processing|computer vision|chatbots?",
    "blockchain": r"blockchain|cryptocurrenc|\bbitcoin\b|non-fungible token|distributed ledger|virtual currency",
    # inventions (hypothesis 23)
    "3d printing": r"3d print|three-dimensional print|additive manufactur",
    "drones": r"\bdrones?\b|unmanned aerial",
    "virtual reality": r"virtual reality|augmented reality",
    "electric vehicles": r"electric vehicles?|electric cars?|ev charging|charging stations? for electric",
    "gene therapy": r"gene therapy|gene editing|crispr",
    "nanotechnology": r"nanotechnolog|nanoparticle|nanomaterial",
    # emergent consumer categories (hypothesis 27)
    "electronic cigarettes": r"electronic cigarette|e-cigarette|vaping|\bvape\b",
    "energy drinks": r"energy drink",
    "kombucha": r"kombucha",
    "cold brew": r"cold brew",
    "plant-based milk": r"plant-based milk|plant based milk|almond milk|oat milk|non-dairy milk|nondairy milk",
    "hard seltzer": r"hard seltzer|alcoholic seltzer|spiked seltzer",
    "cbd / hemp": r"cannabidiol|\bcbd\b|hemp-derived|hemp derived",
}
GPT = ["internet", "mobile apps", "cloud / SaaS", "social media", "ai", "blockchain"]
INVENTION = ["3d printing", "drones", "virtual reality", "electric vehicles", "gene therapy", "nanotechnology"]
CONSUMER = ["electronic cigarettes", "energy drinks", "kombucha", "cold brew", "plant-based milk",
            "hard seltzer", "cbd / hemp"]
ARRIVE_MIN = 5   # a vocabulary has arrived in a class in the first year it appears in >= 5 filings

REGEX = {
    "site": r"restaurant|\bhotels?\b|\bmotels?\b|retail store|store services|clinic|\bsalons?\b|\bspas?\b|fitness (center|club)|\bgyms?\b|\bbar services|\bcafes?\b|car wash|day care|child care|dental|veterinar|coffee shop|bakery services",
    "switching": r"subscription|membership|\baccounts?\b|insurance|banking|telecommunication|leasing|financing",
    "network": r"\bplatforms?\b|marketplace|social network|online communit|peer-to-peer|connecting (buyers|users|people)",
    # Golder-Tellis cases (hypothesis 19)
    "gt_diapers": r"diaper",
    "gt_video_recorders": r"video (cassette )?recorders?|videocassette recorders?|\bvcrs?\b|video tape recorders?",
    "gt_personal_computers": r"personal computers?|laptop|notebook computers?|desktop computers?",
    "gt_beer": r"\bbeers?\b",
    "gt_razors": r"razor",
    # blue-ocean exemplar verticals (hypothesis 21)
    "bo_wine": r"\bwines?\b",
    "bo_fitness": r"fitness|physical exercise|exercise (classes|instruction|programs)|\bgyms?\b",
    "bo_live_entertainment": r"circus|acrobat|live (entertainment|performances?)|theatrical",
    "bo_air_travel": r"air (transport|travel)|airline|charter flights?|fractional (ownership|aircraft)|aircraft (charter|leasing)",
    "bo_hotels": r"\bhotels?\b|\bmotels?\b|lodging",
    "bo_coffee": r"\bcoffee\b",
    "bo_home_improvement": r"home improvement|hardware stores?|building materials",
}
EXEMPLARS = {"[yellow tail] (Casella)": "CASELLA", "Curves": "CURVES INTERNATIONAL",
             "Cirque du Soleil": "CIRQUE DU SOLEIL", "Southwest Airlines": "SOUTHWEST AIRLINES",
             "NetJets": "NETJETS", "Accor (Formule 1)": "ACCOR", "Starbucks": "STARBUCKS",
             "Home Depot": "HOME DEPOT"}


# ------------------------------------------------------------------ stage 1
def owner_history() -> pl.DataFrame:
    """Every filing's owner and how many filings that owner had made before it."""
    parts = []
    for c in CLASSES:
        parts.append(pl.read_parquet(PROC / f"tm_class{c}.parquet",
                                     columns=["serial_number", "filing_date", "owner_name"]))
    a = pl.concat(parts).unique("serial_number").filter(pl.col("owner_name").is_not_null()).with_columns(
        norm_owner(pl.col("owner_name")).alias("okey")).filter(pl.col("okey") != "")
    a = a.sort(["okey", "filing_date", "serial_number"]).with_columns(
        (pl.col("serial_number").cum_count().over("okey") - 1).alias("prior"))
    return a.select("serial_number", "okey", "prior")


def vocab_pass(frame_keys: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Vocabulary flags for frame filings and arrival year of each vocabulary in each class."""
    flags, arrivals = [], []
    for c in CLASSES:
        d = pl.read_parquet(PROC / f"tm_class{c}.parquet",
                            columns=["serial_number", "filing_date", "goods_services"]).unique(
            "serial_number").with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs")).drop("goods_services")
        d = d.with_columns([pl.col("gs").str.contains(p).alias(f"v_{n}") for n, p in VOCAB.items()])
        for n in VOCAB:
            yr = d.filter(pl.col(f"v_{n}")).group_by("fy").len().filter(
                pl.col("len") >= ARRIVE_MIN)["fy"].min()
            arrivals.append({"cls": c, "vocab": n, "arrive": yr})
        fk = frame_keys.filter(pl.col("cls") == c).select("serial_number")
        sub = d.join(fk, on="serial_number", how="inner").with_columns(
            [pl.col("gs").str.contains(p).alias(f"r_{n}") for n, p in REGEX.items()])
        flags.append(sub.drop("gs", "filing_date").with_columns(pl.lit(c).alias("cls")))
        log(f"  [vocab] {c}")
        del d, sub
        gc.collect()
    return pl.concat(flags), pl.DataFrame(arrivals)


def theme_panel(hist: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Class x theme x year shares and derived theme-path variables for every (cls, theme, fy)."""
    th = pl.concat([pl.read_parquet(PROC / "theme_full" / f"theme_class{c}.parquet").with_columns(
        pl.lit(c).alias("cls")) for c in CLASSES]).filter(pl.col("fy") >= 1980)
    th = th.join(hist.select("serial_number", (pl.col("prior") == 0).alias("debut")),
                 on="serial_number", how="left").with_columns(pl.col("debut").fill_null(False))
    cy = th.group_by("cls", "fy").agg(pl.len().alias("n_c"), pl.col("debut").mean().alias("debut_c"))
    ck = th.group_by("cls", "top_theme", "fy").agg(pl.len().alias("n_k"),
                                                  pl.col("debut").sum().alias("debut_k"))
    years = list(range(1980, LAST_FY + 1))
    grid = cy.select("cls").unique().join(pl.DataFrame({"top_theme": list(range(50))}).cast(
        {"top_theme": pl.Int16}), how="cross").join(pl.DataFrame({"fy": years}).cast({"fy": pl.Int32}),
                                                    how="cross")
    p = grid.join(cy, on=["cls", "fy"], how="left").join(
        ck.with_columns(pl.col("top_theme").cast(pl.Int16)), on=["cls", "top_theme", "fy"], how="left"
    ).with_columns(pl.col("n_k").fill_null(0), pl.col("debut_k").fill_null(0), pl.col("n_c").fill_null(0)
    ).sort(["cls", "top_theme", "fy"]).with_columns(
        pl.when(pl.col("n_c") > 0).then(pl.col("n_k") / pl.col("n_c")).alias("s"))
    over = ["cls", "top_theme"]

    def win(col, a, b):  # mean of col over fy+a .. fy+b
        return pl.mean_horizontal([pl.col(col).shift(-k).over(over) for k in range(a, b + 1)])

    p = p.with_columns(
        win("s", -5, -1).alias("s_pre"), win("s", 1, 5).alias("s_post"), win("s", 6, 10).alias("s_late"),
        pl.concat_list([(pl.col("s").shift(-k).over(over) - pl.col("s").shift(-k + 1).over(over))
                        for k in range(-4, 6)]).list.std().alias("s_sd"),
        pl.sum_horizontal([pl.col("n_k").shift(-k).over(over) for k in range(0, 6)]).alias("nk_fwd"),
        pl.sum_horizontal([pl.col("debut_k").shift(-k).over(over) for k in range(0, 6)]).alias("dk_fwd"),
    ).with_columns(
        (pl.col("s_sd") / pl.mean_horizontal("s_pre", "s_post")).alias("volatility"),
        (pl.col("dk_fwd") / pl.col("nk_fwd")).alias("debut_share_fwd"),
        # theme's highest pre-window share in any OTHER class
    )
    other = p.group_by("top_theme", "fy").agg(pl.col("s_pre").max().alias("s_pre_max_any"))
    p = p.join(other, on=["top_theme", "fy"], how="left").with_columns(
        (pl.col("fy") + 5 > LAST_FY).alias("_c1"), (pl.col("fy") + 10 > LAST_FY).alias("_c2")
    ).with_columns(
        pl.when(~pl.col("_c1")).then(pl.col("s_post")).alias("s_post"),
        pl.when(~pl.col("_c2")).then(pl.col("s_late")).alias("s_late"),
        pl.when(~pl.col("_c1")).then(pl.col("volatility")).alias("volatility"),
        pl.when(~pl.col("_c1")).then(pl.col("debut_share_fwd")).alias("debut_share_fwd"),
    ).drop("_c1", "_c2")
    # class-level pace: mix turnover and volume growth around each year
    mix = pl.read_parquet(PROC / "theme_full" / "classyear_mix.parquet").rename({"nice": "cls"}).with_columns(
        pl.col("fy").cast(pl.Int32)).sort(
        ["cls", "fy"])
    tcols = [f"theta_{k}" for k in range(50)]
    pre = [pl.mean_horizontal([pl.col(t).shift(k).over("cls") for k in range(1, 6)]).alias(f"pre_{t}")
           for t in tcols]
    post = [pl.mean_horizontal([pl.col(t).shift(-k).over("cls") for k in range(1, 6)]).alias(f"post_{t}")
            for t in tcols]
    mix = mix.with_columns(pre + post).with_columns(
        (0.5 * pl.sum_horizontal([(pl.col(f"post_{t}") - pl.col(f"pre_{t}")).abs() for t in tcols]))
        .alias("tech_pace"),
        (pl.sum_horizontal([pl.col("n").shift(-k).over("cls") for k in range(1, 6)])
         / pl.sum_horizontal([pl.col("n").shift(k).over("cls") for k in range(1, 6)])).log().alias("mkt_pace"),
    ).with_columns(pl.when(pl.col("fy") + 5 <= LAST_FY).then(pl.col("tech_pace")).alias("tech_pace"),
                   pl.when(pl.col("fy") + 5 <= LAST_FY).then(pl.col("mkt_pace")).alias("mkt_pace")
    ).select("cls", "fy", "tech_pace", "mkt_pace")
    return p.drop("s_sd"), mix


def geo_pass(frame: pl.DataFrame) -> pl.DataFrame:
    """Co-location (L) of each (class, theme, 5-year window) and whether each frame filer sits in a hub."""
    ll = pl.read_parquet(PROC / "owner_ll.parquet")
    m, cells = gc_.cells_for(ll)
    K = gc_.kernel(cells)
    ncell = cells.height
    out = []
    for c in CLASSES:
        th = pl.read_parquet(PROC / "theme_full" / f"theme_class{c}.parquet").join(
            m.select("serial_number", "cell"), on="serial_number", how="inner").filter(
            pl.col("fy").is_between(1990, LAST_FY))
        y0, Y = 1990, LAST_FY - 1990 + 1
        r = np.array((th["top_theme"].cast(pl.Int32) * Y + (th["fy"] - y0)).to_numpy(), dtype=np.int64)
        tc = np.array(th["cell"].to_numpy(), dtype=np.int64)
        ty = np.array((th["fy"] - y0).to_numpy(), dtype=np.int64)
        M = sp.csr_matrix((np.ones(len(r)), (r, tc)), shape=(50 * Y, ncell))
        base = sp.csr_matrix((np.ones(len(r)), (ty, tc)),
                             shape=(Y, ncell))
        # +/- 2 year windows (shift within theme blocks)
        def window(A, block):
            A = A.tolil()
            W = sp.csr_matrix(A.shape)
            for s in range(-2, 3):
                S = sp.eye(A.shape[0], k=s, format="csr")
                # zero out shifts that cross a theme block
                idx = np.arange(A.shape[0])
                ok = ((idx + s) // block == idx // block) & (idx + s >= 0) & (idx + s < A.shape[0])
                S = sp.diags(ok.astype(float)) @ S
                W = W + S @ A.tocsr()
            return W.tocsr()
        Mw, Bw = window(M, Y), window(base, Y)
        C = gc_.co_location(Mw, K)
        B = gc_.co_location(Bw, K)
        Nk = np.asarray(Mw.sum(1)).ravel()
        Bk = np.tile(B, 50)
        L = gc_.localization(C, Bk)
        L[Nk < 30] = np.nan
        # hub status for frame filers of this class
        fr = frame.filter(pl.col("cls") == c).select("serial_number", "fy", "top_theme").join(
            m.select("serial_number", "cell"), on="serial_number", how="inner").filter(
            pl.col("fy").is_between(y0, LAST_FY) & pl.col("top_theme").is_not_null())
        rows = np.array((fr["top_theme"].cast(pl.Int32) * Y + (fr["fy"] - y0)).to_numpy(), dtype=np.int64)
        yrs = np.array((fr["fy"] - y0).to_numpy(), dtype=np.int64)
        cc = np.array(fr["cell"].to_numpy(), dtype=np.int64)
        MK = (Mw @ K).tocsr()
        BK = (Bw @ K).tocsr()
        NB = np.asarray(Bw.sum(1)).ravel()
        s_local = np.asarray(MK[rows, cc]).ravel() / np.maximum(Nk[rows], 1)
        b_local = np.asarray(BK[yrs, cc]).ravel() / np.maximum(NB[yrs], 1)
        in_hub = (s_local >= 0.10) & (s_local >= 1.2 * b_local)
        # does the group have any hub at all? (same rule as geo_cluster.hubs)
        S = MK.toarray() / np.maximum(Nk, 1)[:, None]
        Bd = BK.toarray() / np.maximum(NB, 1)[:, None]
        hubrow = ((S >= 0.10) & (S >= 1.2 * Bd[np.arange(S.shape[0]) % Y])).any(1) & (Nk >= 30)
        del S, Bd
        has_hub = {int(g): bool(hubrow[g]) for g in np.unique(rows) if Nk[g] >= 30}
        out.append(pl.DataFrame({
            "serial_number": fr["serial_number"], "geo_L": L[rows], "geo_N": Nk[rows],
            "in_hub": in_hub, "group_has_hub": [has_hub.get(g, None) for g in rows],
            "local_share": s_local, "local_class_share": b_local}))
        log(f"  [geo] {c}: {fr.height:,} frame filers placed")
        del th, M, Mw, MK, BK
        gc.collect()
    return pl.concat(out)


def patenters() -> pl.DataFrame:
    p = pl.read_csv(REPO / "data_publish" / "firm_year_patents_and_dkl.csv",
                    columns=["normalized_name", "uspto_owner_name", "n_patents"],
                    schema_overrides={"normalized_name": pl.Utf8}).filter(
        pl.col("normalized_name").fill_null("") != "").with_columns(
        norm_owner(pl.col("uspto_owner_name")).alias("owner_key")).group_by("owner_key").agg(
        pl.col("n_patents").sum()).filter(pl.col("n_patents") > 0)
    return p.select("owner_key", pl.lit(True).alias("patenter"))


def build() -> pl.DataFrame:
    if CACHE.exists() and not os.environ.get("REBUILD"):
        return pl.read_parquet(CACHE)
    if PRE.exists() and not os.environ.get("REBUILD"):
        df = pl.read_parquet(PRE)
        geo = geo_pass(df.select("serial_number", "cls", "fy", "top_theme"))
        df = df.join(geo, on="serial_number", how="left")
        df = df.join(patenters(), on="owner_key", how="left").with_columns(pl.col("patenter").fill_null(False))
        df.write_parquet(CACHE)
        return df
    df = build_frame().select("serial_number", "cls", "reg_year", "cell", "owner_key", "failed1", "q", "z",
                              "has_attorney", "itu", "basis_44e", "basis_66a", "dom_us", "dom_cn",
                              "log_len", "log_owner_n", "ctry")
    log(f"[frame] {df.height:,}")
    hist = owner_history()
    df = df.join(hist.select("serial_number", "prior"), on="serial_number", how="left")
    log("[history] joined")
    theme = pl.concat([pl.read_parquet(PROC / "theme_full" / f"theme_class{c}.parquet").with_columns(
        pl.lit(c).alias("cls")) for c in CLASSES])
    df = df.join(theme.select("serial_number", "cls", "fy", "top_theme"), on=["serial_number", "cls"],
                 how="left")
    flags, arrivals = vocab_pass(df.select("serial_number", "cls"))
    df = df.join(flags.drop("fy"), on=["serial_number", "cls"], how="left")
    arrivals.write_parquet(PROC / "battery_vocab_arrivals.parquet")
    for n in VOCAB:
        a = arrivals.filter(pl.col("vocab") == n).select("cls", pl.col("arrive").alias(f"arr_{n}"))
        df = df.join(a, on="cls", how="left")
    p, mix = theme_panel(hist)
    df = df.with_columns(pl.col("top_theme").cast(pl.Int16)).join(
        p.select("cls", "top_theme", "fy", "s_pre", "s_post", "s_late", "volatility", "debut_share_fwd",
                 "s_pre_max_any"), on=["cls", "top_theme", "fy"], how="left").join(
        mix, on=["cls", "fy"], how="left")
    del hist, theme, p
    gc.collect()
    df.write_parquet(PRE)
    geo = geo_pass(df.select("serial_number", "cls", "fy", "top_theme"))
    df = df.join(geo, on="serial_number", how="left")
    df = df.join(patenters(), on="owner_key", how="left").with_columns(pl.col("patenter").fill_null(False))
    df.write_parquet(CACHE)
    log(f"[cache] {CACHE}")
    return df


# ------------------------------------------------------------------ stage 2
CONTROLS = ["has_attorney", "itu", "log_len", "log_owner_n", "dom_us", "dom_cn", "basis_44e", "basis_66a"]


def lpm(d: pl.DataFrame, xcols: list[str]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """failed1 on xcols; cell FE by demeaning; owner-clustered covariance."""
    y = d["failed1"].cast(pl.Float64).to_numpy()
    X = d.select(xcols).cast(pl.Float64).to_numpy()
    cell = d["cell"].to_physical().to_numpy() if d["cell"].dtype == pl.Categorical else \
        d["cell"].cast(pl.Categorical).to_physical().to_numpy()
    cnt = np.bincount(cell)
    def dm(v):
        return v - (np.bincount(cell, weights=v, minlength=len(cnt)) / np.maximum(cnt, 1))[cell]
    yd = dm(y)
    Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    keep = np.abs(Xd).sum(0) > 1e-9
    Xd = Xd[:, keep]
    XtX = Xd.T @ Xd
    b = np.linalg.solve(XtX, Xd.T @ yd)
    e = yd - Xd @ b
    og = d["owner_key"].cast(pl.Categorical).to_physical().to_numpy()
    G = int(og.max()) + 1
    Sg = sp.csr_matrix((np.ones(len(og)), (og, np.arange(len(og)))), shape=(G, len(og))) @ (Xd * e[:, None])
    n, k = Xd.shape
    inv = np.linalg.inv(XtX)
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * inv @ (Sg.T @ Sg) @ inv
    bf = np.full(len(xcols), np.nan)
    Vf = np.full((len(xcols), len(xcols)), np.nan)
    idx = np.where(keep)[0]
    bf[idx] = b
    Vf[np.ix_(idx, idx)] = V
    return bf, Vf, n, G


def by_group(d: pl.DataFrame, gcol: str, levels: list, labels: dict | None = None) -> dict:
    """Penalty (Q5-Q1, pp) in each level of gcol, from one interacted model."""
    d = d.filter(pl.col(gcol).is_in(levels) & pl.col("q").is_not_null())
    xs, cols = [], {}
    for li, lv in enumerate(levels):
        ind = (pl.col(gcol) == lv).cast(pl.Float64)
        for qq in range(1, 5):
            nm = f"q{qq+1}_g{li}"
            xs.append(((pl.col("q") == qq).cast(pl.Float64) * ind).alias(nm))
            cols[nm] = None
        if li > 0:
            xs.append(ind.alias(f"g{li}"))
            cols[f"g{li}"] = None
    d = d.with_columns(xs)
    xcols = list(cols) + CONTROLS
    b, V, n, G = lpm(d, xcols)
    ix = {c: i for i, c in enumerate(xcols)}
    res = {"n": n, "n_owners": G, "levels": {}}
    for li, lv in enumerate(levels):
        j = ix[f"q5_g{li}"]
        sub = d.filter(pl.col(gcol) == lv)
        res["levels"][str(lv) if not labels else labels.get(lv, str(lv))] = {
            "penalty_pp": 100 * b[j], "se_pp": 100 * math.sqrt(V[j, j]) if V[j, j] == V[j, j] else None,
            "n": sub.height, "fail_rate": float(sub["failed1"].cast(pl.Float64).mean() or 0) * 100,
            "_j": j}
    res["_b"], res["_V"] = b, V
    return res


def contrast(res: dict, a: str, bname: str) -> dict:
    ja, jb = res["levels"][a]["_j"], res["levels"][bname]["_j"]
    b, V = res["_b"], res["_V"]
    diff = b[ja] - b[jb]
    se = math.sqrt(V[ja, ja] + V[jb, jb] - 2 * V[ja, jb])
    return {"diff_pp": 100 * diff, "se_pp": 100 * se, "t": diff / se}


def cohort_excess(d: pl.DataFrame, flag: str, cohort_col: str, levels: list) -> dict:
    """Excess first-gate failure of flagged filings in each cohort, relative to the
    same class x year cell's other filings (cohort dummies + controls, cell FE)."""
    xs = [((pl.col(flag)) & (pl.col(cohort_col) == lv)).fill_null(False).cast(pl.Float64).alias(f"c_{i}")
          for i, lv in enumerate(levels)]
    d2 = d.with_columns(xs)
    xcols = [f"c_{i}" for i in range(len(levels))] + CONTROLS
    b, V, n, G = lpm(d2, xcols)
    out = {"n": n, "levels": {}}
    for i, lv in enumerate(levels):
        out["levels"][str(lv)] = {"excess_pp": 100 * b[i], "se_pp": 100 * math.sqrt(V[i, i]),
                                  "n": int(d2[f"c_{i}"].sum()), "_j": i}
    out["_b"], out["_V"] = b, V
    return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main() -> int:
    df = build()
    df = df.with_columns(pl.col("cell").cast(pl.Categorical))
    log(f"[battery] frame {df.height:,}")
    H: dict = {}
    primary: list[tuple[str, dict, str]] = []   # (id, contrast, predicted sign '+'/'-'/'?')

    def two(hid, title, gexpr, a, bname, pred, sub=None, note=""):
        d = (sub if sub is not None else df).with_columns(gexpr.alias("_g"))
        r = by_group(d.filter(pl.col("_g").is_not_null()), "_g", [a, bname])
        cst = contrast(r, a, bname)
        H[hid] = {"title": title, "groups": strip(r["levels"]), "contrast": f"{a} minus {bname}",
                  "predicted": pred, **cst, "n": r["n"], "note": note}
        primary.append((hid, H[hid], pred))
        log(f"  {hid} {title}: {a} {100*r['_b'][r['levels'][a]['_j']]:+.2f} vs {bname} "
            f"{100*r['_b'][r['levels'][bname]['_j']]:+.2f}  diff {cst['diff_pp']:+.2f} (t={cst['t']:.1f})")

    g = pl.col
    # headline for reference
    r0 = by_group(df.with_columns(pl.lit("all").alias("_g")), "_g", ["all"])
    H["H0"] = {"title": "All registrations 2002-2018", "groups": strip(r0["levels"])}
    log(f"  H0 {r0['levels']['all']['penalty_pp']:+.2f}")

    # A. Lieberman & Montgomery mechanisms
    two("H1", "Owners who patent vs those who do not",
        pl.when(g("patenter")).then(pl.lit("patenter")).otherwise(pl.lit("no patents")),
        "patenter", "no patents", "-")
    two("H2", "Site-based offerings vs others",
        pl.when(g("r_site")).then(pl.lit("site-based")).otherwise(pl.lit("other")), "site-based", "other", "-")
    two("H3", "Contract / subscription offerings vs others",
        pl.when(g("r_switching")).then(pl.lit("switching costs")).otherwise(pl.lit("other")),
        "switching costs", "other", "-")
    two("H4", "Platform / network offerings vs others",
        pl.when(g("r_network")).then(pl.lit("network")).otherwise(pl.lit("other")), "network", "other", "-")
    easy = ["003", "014", "016", "018", "021", "024", "025", "026", "028", "030", "032"]
    heavy = ["001", "004", "006", "007", "011", "012", "013", "017", "019", "040"]
    two("H5", "Easy-to-imitate goods vs capital-heavy goods",
        pl.when(g("cls").is_in(easy)).then(pl.lit("easy to imitate")).when(g("cls").is_in(heavy))
        .then(pl.lit("capital heavy")), "easy to imitate", "capital heavy", "+")
    vt = df.filter(g("volatility").is_finite())["volatility"].quantile(2 / 3)
    vb = df.filter(g("volatility").is_finite())["volatility"].quantile(1 / 3)
    two("H6", "Volatile themes vs steady themes (top vs bottom third of share volatility)",
        pl.when(g("volatility") >= vt).then(pl.lit("volatile")).when(g("volatility") <= vb)
        .then(pl.lit("steady")), "volatile", "steady", "+")
    two("H7", "Debut owners vs established owners (25+ prior filings)",
        pl.when(g("prior") == 0).then(pl.lit("debut")).when(g("prior") >= 25).then(pl.lit("established")),
        "debut", "established", "+")

    # B. Pace of change quadrants
    tp_med = df["tech_pace"].median()
    mp_med = df["mkt_pace"].median()
    quad = (pl.when(g("tech_pace").is_null() | g("mkt_pace").is_null()).then(None)
            .when((g("tech_pace") < tp_med) & (g("mkt_pace") < mp_med)).then(pl.lit("calm"))
            .when((g("tech_pace") >= tp_med) & (g("mkt_pace") < mp_med)).then(pl.lit("technology leads"))
            .when((g("tech_pace") < tp_med) & (g("mkt_pace") >= mp_med)).then(pl.lit("market leads"))
            .otherwise(pl.lit("rough")))
    dq = df.with_columns(quad.alias("_q4")).filter(g("_q4").is_not_null())
    lv4 = ["calm", "technology leads", "market leads", "rough"]
    rq = by_group(dq, "_q4", lv4)
    H["H8-11"] = {"title": "Pace of change quadrants (class-year theme-mix turnover x filing-volume growth)",
                  "groups": strip(rq["levels"]), "n": rq["n"],
                  "medians": {"tech_pace": tp_med, "mkt_pace": mp_med}}
    for hid, a, bname, pred in (("H8", "technology leads", "calm", "+"),
                                ("H10", "market leads", "technology leads", "-"),
                                ("H11", "rough", "calm", "+")):
        cst = contrast(rq, a, bname)
        H[hid] = {"title": f"Pace: {a} vs {bname}", "contrast": f"{a} minus {bname}", "predicted": pred,
                  **cst, "groups": {k: v for k, v in strip(rq["levels"]).items() if k in (a, bname)}}
        primary.append((hid, H[hid], pred))
        log(f"  {hid}: {a} - {bname} {cst['diff_pp']:+.2f} (t={cst['t']:.1f})")

    # C. Appropriability
    two("H12", "Pharmaceuticals and medical devices (classes 5, 10) vs all other classes",
        pl.when(g("cls").is_in(["005", "010"])).then(pl.lit("pharma/devices")).otherwise(pl.lit("other")),
        "pharma/devices", "other", "-")
    two("H13", "Services (classes 35-45) vs goods",
        pl.when(g("cls").cast(pl.Int32) >= 35).then(pl.lit("services")).otherwise(pl.lit("goods")),
        "services", "goods", "+")
    cpg = ["003", "029", "030", "031", "032", "033"]
    two("H14", "Consumer packaged goods: debut owners vs established owners",
        pl.when(g("prior") == 0).then(pl.lit("debut")).when(g("prior") >= 25).then(pl.lit("established")),
        "debut", "established", "+", sub=df.filter(g("cls").is_in(cpg)))

    # D. Golder & Tellis / Klepper
    # H15: pioneers of a vocabulary in a class vs early followers (excess failure vs same cell)
    ords = []
    for n in VOCAB:
        ords.append(pl.when(g(f"v_{n}")).then(g("fy") - g(f"arr_{n}")))
    d15 = df.with_columns(pl.min_horizontal(ords).alias("yrs_since_arrival"),
                          pl.any_horizontal([g(f"v_{n}") for n in VOCAB]).alias("anyvocab")).with_columns(
        pl.when(g("yrs_since_arrival") <= 1).then(pl.lit("pioneer (arrival year or next)"))
        .when(g("yrs_since_arrival") <= 5).then(pl.lit("early follower (2-5 years in)"))
        .when(g("yrs_since_arrival") > 5).then(pl.lit("later (6+ years in)")).alias("cohort"))
    levs = ["pioneer (arrival year or next)", "early follower (2-5 years in)", "later (6+ years in)"]
    r15 = cohort_excess(d15, "anyvocab", "cohort", levs)
    c15 = {"diff_pp": 100 * (r15["_b"][0] - r15["_b"][1]),
           "se_pp": 100 * math.sqrt(r15["_V"][0, 0] + r15["_V"][1, 1] - 2 * r15["_V"][0, 1])}
    c15["t"] = c15["diff_pp"] / c15["se_pp"]
    H["H15"] = {"title": "Pioneers of a new vocabulary in a class vs early followers (excess failure over same class-year)",
                "groups": strip(r15["levels"]), "contrast": "pioneer minus early follower", "predicted": "+",
                **c15, "note": "vocabularies: " + ", ".join(VOCAB)}
    primary.append(("H15", H["H15"], "+"))
    log(f"  H15 pioneer-follower {c15['diff_pp']:+.2f} (t={c15['t']:.1f})")

    # H16: survivor conditioning -- registration stage handled in a separate frame below
    two("H17", "Consumer packaged goods vs industrial goods",
        pl.when(g("cls").is_in(["003", "029", "030", "032", "033"])).then(pl.lit("CPG"))
        .when(g("cls").is_in(["001", "006", "007", "017", "019"])).then(pl.lit("industrial")),
        "CPG", "industrial", "-")
    two("H18", "Manufactured goods (7, 9, 12) vs services (35-45)",
        pl.when(g("cls").is_in(["007", "009", "012"])).then(pl.lit("manufactured"))
        .when(g("cls").cast(pl.Int32) >= 35).then(pl.lit("services")), "manufactured", "services", "-")
    H["H19"] = {"title": "Golder-Tellis cases: penalty within each vertical", "groups": {}}
    for k in ["gt_diapers", "gt_video_recorders", "gt_personal_computers", "gt_beer", "gt_razors"]:
        sub = df.filter(g(f"r_{k}"))
        if sub.height < 2000:
            H["H19"]["groups"][k] = {"n": sub.height, "skipped": "too few"}
            continue
        r = by_group(sub.with_columns(pl.lit(k).alias("_g")), "_g", [k])
        H["H19"]["groups"][k] = strip(r["levels"][k])
        log(f"  H19 {k}: {r['levels'][k]['penalty_pp']:+.2f} (n={sub.height:,})")

    # E. Blue ocean
    imported = (g("s_pre") < 0.01) & (g("s_pre_max_any") >= 0.05)
    two("H20", "Theme imported from another class vs native theme",
        pl.when(g("s_pre").is_null()).then(None).when(imported).then(pl.lit("imported"))
        .otherwise(pl.lit("native")), "imported", "native", "-")
    H["H21"] = {"title": "Blue-ocean exemplar verticals: penalty within each", "groups": {}}
    for k in [x for x in REGEX if x.startswith("bo_")]:
        sub = df.filter(g(f"r_{k}"))
        r = by_group(sub.with_columns(pl.lit(k).alias("_g")), "_g", [k])
        H["H21"]["groups"][k] = strip(r["levels"][k])
        log(f"  H21 {k}: {r['levels'][k]['penalty_pp']:+.2f} (n={sub.height:,})")
    ex = {}
    for lab, pat in EXEMPLARS.items():
        s = df.filter(g("owner_key").str.contains(pat))
        if s.height:
            ex[lab] = {"n_marks": s.height, "mean_lead_fifth": float(s["q"].cast(pl.Float64).mean() + 1),
                       "share_top_fifth": float((s["q"] == 4).mean()),
                       "fail_rate": float(s["failed1"].cast(pl.Float64).mean() * 100)}
    H["H21"]["exemplar_firms"] = ex
    dt = df["debut_share_fwd"].drop_nulls()
    two("H22", "Themes growing through first-time filers vs through established filers (top vs bottom third)",
        pl.when(g("debut_share_fwd") >= dt.quantile(2 / 3)).then(pl.lit("new demand"))
        .when(g("debut_share_fwd") <= dt.quantile(1 / 3)).then(pl.lit("established demand")),
        "new demand", "established demand", "-")

    # F. Kinds of new
    inv = pl.any_horizontal([g(f"v_{n}") for n in INVENTION])
    gpt = pl.any_horizontal([g(f"v_{n}") for n in GPT])
    con = pl.any_horizontal([g(f"v_{n}") for n in CONSUMER])
    two("H23", "Invention vocabularies vs all other filings",
        pl.when(inv).then(pl.lit("invention")).otherwise(pl.lit("other")), "invention", "other", "-")
    surge = g("s_post") >= 1.3 * g("s_pre")
    fad = (pl.when(g("s_late").is_null() | ~surge).then(None)
           .when(g("s_late") <= g("s_pre")).then(pl.lit("fad (surge given back)"))
           .when(g("s_late") >= g("s_post")).then(pl.lit("surge held")))
    two("H24", "Fads vs surges that held", fad, "fad (surge given back)", "surge held", "+")
    gpt_yrs = pl.min_horizontal([pl.when(g(f"v_{n}")).then(g("fy") - g(f"arr_{n}")) for n in GPT])
    two("H25", "General-purpose technology filings: first 3 years in a class vs 6+ years",
        pl.when(gpt_yrs <= 2).then(pl.lit("early adopter")).when(gpt_yrs >= 6).then(pl.lit("late adopter")),
        "early adopter", "late adopter", "-")
    # H26: new to corpus (vocab's first class) vs new to class (later adopting class), within 5 yrs of arrival
    arr = pl.read_parquet(PROC / "battery_vocab_arrivals.parquet").drop_nulls("arrive")
    first_cls = arr.sort(["vocab", "arrive", "cls"]).group_by("vocab").agg(pl.col("arrive").first().alias("a0"))
    exprs = []
    for n in VOCAB:
        a0 = first_cls.filter(pl.col("vocab") == n)["a0"]
        if not len(a0):
            continue
        a0 = int(a0[0])
        exprs.append(pl.when(g(f"v_{n}") & (g("fy") - g(f"arr_{n}") <= 4))
                     .then(pl.when(g(f"arr_{n}") <= a0 + 1).then(pl.lit("new to corpus"))
                           .otherwise(pl.lit("new to class"))))
    d26 = df.with_columns(pl.coalesce(exprs).alias("novelty_kind"),
                          pl.any_horizontal([g(f"v_{n}") for n in VOCAB]).alias("anyvocab"))
    r26 = cohort_excess(d26, "anyvocab", "novelty_kind", ["new to class", "new to corpus"])
    c26 = {"diff_pp": 100 * (r26["_b"][0] - r26["_b"][1]),
           "se_pp": 100 * math.sqrt(r26["_V"][0, 0] + r26["_V"][1, 1] - 2 * r26["_V"][0, 1])}
    c26["t"] = c26["diff_pp"] / c26["se_pp"]
    H["H26"] = {"title": "Vocabulary new to its class vs new to the corpus (excess failure, first five years)",
                "groups": strip(r26["levels"]), "contrast": "new to class minus new to corpus",
                "predicted": "-", **c26}
    primary.append(("H26", H["H26"], "-"))
    log(f"  H26 {c26['diff_pp']:+.2f} (t={c26['t']:.1f})")
    con_yrs = pl.min_horizontal([pl.when(g(f"v_{n}")).then(g("fy") - g(f"arr_{n}")) for n in CONSUMER])
    d27 = df.with_columns(con.alias("anycon"), pl.when(con_yrs <= 2).then(pl.lit("first 3 years"))
                          .when(con_yrs > 2).then(pl.lit("later")).alias("ccoh"))
    r27 = cohort_excess(d27, "anycon", "ccoh", ["first 3 years", "later"])
    c27 = {"diff_pp": 100 * (r27["_b"][0] - r27["_b"][1]),
           "se_pp": 100 * math.sqrt(r27["_V"][0, 0] + r27["_V"][1, 1] - 2 * r27["_V"][0, 1])}
    c27["t"] = c27["diff_pp"] / c27["se_pp"]
    H["H27"] = {"title": "Emergent consumer categories: first three years vs later (excess failure)",
                "groups": strip(r27["levels"]), "contrast": "first 3 years minus later", "predicted": "?", **c27}
    primary.append(("H27", H["H27"], "?"))

    # G. Who and where
    gl = df.filter(g("geo_L").is_finite())["geo_L"]
    two("H28", "Geographically concentrated themes vs dispersed (top vs bottom third of co-location L)",
        pl.when(g("geo_L") >= gl.quantile(2 / 3)).then(pl.lit("concentrated"))
        .when(g("geo_L") <= gl.quantile(1 / 3)).then(pl.lit("dispersed")), "concentrated", "dispersed", "+")
    two("H29", "Within themes that have a hub: filers in a hub vs filers outside",
        pl.when(g("group_has_hub").fill_null(False).not_()).then(None).when(g("in_hub")).then(pl.lit("in hub"))
        .otherwise(pl.lit("outside hub")), "in hub", "outside hub", "?")
    # main effect of being in a hub (not the penalty)
    dh = df.filter(g("group_has_hub").fill_null(False)).with_columns(
        g("in_hub").cast(pl.Float64).alias("in_hub_f"))
    for qq in range(1, 5):
        dh = dh.with_columns((g("q") == qq).cast(pl.Float64).alias(f"q{qq+1}"))
    b, V, n, G = lpm(dh, ["in_hub_f", "q2", "q3", "q4", "q5"] + CONTROLS)
    H["H29"]["hub_main_effect_pp"] = {"b": 100 * b[0], "se": 100 * math.sqrt(V[0, 0]), "n": n}
    two("H30", "Foreign-domiciled owners vs US owners",
        pl.when(g("ctry") == "US").then(pl.lit("US")).when(g("ctry").is_in(["", "CN"]).not_())
        .then(pl.lit("foreign (not China)")), "foreign (not China)", "US", "+")
    two("H31", "Self-filed vs counsel-represented",
        pl.when(g("has_attorney").cast(pl.Float64) == 1.0).then(pl.lit("counsel")).otherwise(pl.lit("self-filed")),
        "self-filed", "counsel", "+")
    two("H32", "Intent-to-use vs use-based filings",
        pl.when(g("itu").cast(pl.Float64) == 1.0).then(pl.lit("intent to use")).otherwise(pl.lit("in use")),
        "intent to use", "in use", "+")
    two("H33", "Regulated classes (5, 10, 33, 34, 36) vs others",
        pl.when(g("cls").is_in(["005", "010", "033", "034", "036"])).then(pl.lit("regulated"))
        .otherwise(pl.lit("unregulated")), "regulated", "unregulated", "-")
    two("H34", "Filed in a boom (2004-07, 2014-17) vs a bust (2001-02, 2008-10)",
        pl.when(g("fy").is_in([2004, 2005, 2006, 2007, 2014, 2015, 2016, 2017])).then(pl.lit("boom"))
        .when(g("fy").is_in([2001, 2002, 2008, 2009, 2010])).then(pl.lit("bust")), "boom", "bust", "+")

    # Holm adjustment across the pre-stated contrasts (two-sided p; '?' kept two-sided)
    ps = []
    for hid, h, pred in primary:
        p = math.erfc(abs(h["t"]) / math.sqrt(2))
        ps.append((p, hid))
    ps.sort()
    m = len(ps)
    running = 0.0
    for i, (p, hid) in enumerate(ps):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        h = H[hid]
        h["p"] = p
        h["p_holm"] = adj
        sign = "+" if h["diff_pp"] > 0 else "-"
        h["agrees"] = None if h["predicted"] == "?" else (sign == h["predicted"])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "hypothesis_battery.json").write_text(json.dumps(strip(H), indent=1, default=float))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
