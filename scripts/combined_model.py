"""Where being leading costs: every factor in one model, then LASSO.

Outcome: survival of the five-year proof of continued use, in percentage points
(100 = the registration passed). Unit and sample as in the paper's gate estimate:
registrations 2002-2018, one row per serial, class x registration-year fixed
effects absorbed by demeaning, standard errors clustered on the normalized owner.

Lead enters as its percentile within the class x registration-year cell, centred
at the median (lead_pct - 0.5), so its coefficient is the survival difference
between the most lagging and the most leading filing of the same class and year.
Each factor enters twice: a main effect (does the factor go with survival?) and
an interaction with lead (does it change whether leading helps or hurts?).
Factors are centred, so the lead coefficient is the lead effect for a filing of
average composition; the lead effect for filings that have a binary factor is
lead + interaction x (1 - share). Continuous factors are in SD units
(winsorized at the 1st and 99th percentiles).

Models
  alone  : lead + one factor + its interaction + the paper's controls
  joint  : lead + all factors + all interactions + controls
  classes: class-specific lead effects estimated on half the owners (split by a
           hash of the owner name), shrunk toward the pooled effect, and grouped
           by k-means into composite class groups; the groups' lead effects are
           then estimated on the other half, which played no part in forming them
  lasso  : with main effects, controls and fixed effects partialled out, a LASSO
           over the lead interactions (all factors, the 45 classes, the composite
           groups); the penalty is chosen by 5-fold cross-validation grouped by
           owner, and the order in which terms enter the path ranks them
  post   : OLS with clustered SEs on the lasso-selected interactions (inference
           after selection is optimistic; read it as description)

Significance stars use Holm-adjusted p-values within each column.

Output: paper/results/combined_model.json
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from hypothesis_battery import GPT, INVENTION, CONSUMER  # noqa: E402


def log(m):
    print(m, file=sys.stderr, flush=True)


def wz(e: pl.Expr) -> pl.Expr:
    """Winsorize at 1/99 percentiles, then standardize; non-finite -> null."""
    e = pl.when(e.is_finite()).then(e)
    lo, hi = e.quantile(0.01), e.quantile(0.99)
    c = e.clip(lo, hi)
    return (c - c.mean()) / c.std()


BINARY = [  # (key, label, expression)
    ("counsel", "Represented by counsel", pl.col("has_attorney").cast(pl.Float64)),
    ("patents", "Owner files patents", pl.col("patenter").cast(pl.Float64)),
    ("itu", "Intent-to-use filing", pl.col("itu").cast(pl.Float64)),
    ("debut", "Owner's first-ever filing", (pl.col("prior") == 0).cast(pl.Float64)),
    ("established", "Owner has 25+ earlier filings", (pl.col("prior") >= 25).cast(pl.Float64)),
    ("foreign", "Foreign owner (not China)",
     (~pl.col("ctry").is_in(["US", "CN", ""])).cast(pl.Float64)),
    ("china", "Owner in China", pl.col("dom_cn").cast(pl.Float64)),
    ("site", "Site-based offering (stores, restaurants, clinics)", pl.col("r_site").cast(pl.Float64)),
    ("contract", "Contract or subscription offering", pl.col("r_switching").cast(pl.Float64)),
    ("platform", "Platform or network offering", pl.col("r_network").cast(pl.Float64)),
    ("invention", "Invention vocabulary",
     pl.any_horizontal([pl.col(f"v_{n}") for n in INVENTION]).cast(pl.Float64)),
    ("gpt", "General-purpose-technology vocabulary",
     pl.any_horizontal([pl.col(f"v_{n}") for n in GPT]).cast(pl.Float64)),
    ("consumer", "Emergent consumer-category vocabulary",
     pl.any_horizontal([pl.col(f"v_{n}") for n in CONSUMER]).cast(pl.Float64)),
    ("imported", "Theme imported from another class",
     ((pl.col("s_pre") < 0.01) & (pl.col("s_pre_max_any") >= 0.05)).fill_null(False).cast(pl.Float64)),
    ("has_hub", "Theme has a geographic hub", pl.col("group_has_hub").fill_null(False).cast(pl.Float64)),
    ("in_hub", "Filer located in a hub", pl.col("in_hub").fill_null(False).cast(pl.Float64)),
    # year types against "neither" years (1995-97, 2003-04, 2010-19): the two bubble
    # peaks and the two crashes that ended them
    ("boom", "Filed in a boom (1998-2000, 2005-07)",
     pl.col("fy").is_in([1998, 1999, 2000, 2005, 2006, 2007]).cast(pl.Float64)),
    ("bust", "Filed in a bust (2001-02, 2008-09)",
     pl.col("fy").is_in([2001, 2002, 2008, 2009]).cast(pl.Float64)),
]
CONTINUOUS = [
    ("volatility", "Theme-share volatility (per SD)", pl.col("volatility").log()),
    ("new_demand", "Theme growth from first-time filers (per SD)", pl.col("debut_share_fwd")),
    ("colocation", "Theme co-location L (per SD)", pl.col("geo_L")),
    ("tech_pace", "Class theme-mix turnover (per SD)", pl.col("tech_pace")),
    ("mkt_pace", "Class filing-volume growth (per SD)", pl.col("mkt_pace")),
    # the lead penalty has grown over time; without a trend, boom and bust years are
    # compared with "neither" years that are mostly 2010-19
    ("trend", "Filing year (per SD, about 5 years)", pl.col("fy").cast(pl.Float64)),
]
CLASS_GROUPS = [  # main effects absorbed by the class x year fixed effects
    ("services", "Services classes (35-45)", pl.col("cls").cast(pl.Int32) >= 35),
    ("pharma", "Pharmaceuticals and devices (5, 10)", pl.col("cls").is_in(["005", "010"])),
    ("cpg", "Consumer packaged goods (3, 29-33)", pl.col("cls").is_in(["003", "029", "030", "031", "032", "033"])),
    ("manufactured", "Machinery, electronics, vehicles (7, 9, 12)", pl.col("cls").is_in(["007", "009", "012"])),
    ("regulated", "Alcohol, tobacco, finance (33, 34, 36)", pl.col("cls").is_in(["033", "034", "036"])),
]
CONTROLS = ["log_len", "log_owner_n", "basis_44e", "basis_66a"]
MISSING = ["m_geo", "m_env"]
YEAR_TYPES = ("boom", "bust")
CHUNK = 250_000          # rows per streamed chunk in Design.ols


# ------------------------------------------------------------------ frame
FRAME_COLS = (["failed1", "z", "cell", "reg_year", "owner_key", "cls", "has_attorney", "patenter", "itu",
               "prior", "ctry", "dom_cn", "r_site", "r_switching", "r_network", "s_pre", "s_pre_max_any",
               "group_has_hub", "in_hub", "fy", "volatility", "debut_share_fwd", "geo_L", "tech_pace",
               "mkt_pace", "log_len", "log_owner_n", "basis_44e", "basis_66a"]
              + [f"v_{n}" for n in INVENTION + GPT + CONSUMER])


def peak_gb() -> float:
    """Peak working set of this process in GB (Windows), for the memory log."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e9
    except Exception:
        return float("nan")


def frame() -> pl.DataFrame:
    d = pl.read_parquet(PROC / "battery_frame.parquet", columns=FRAME_COLS)
    d = d.with_columns(
        (100 * (1 - pl.col("failed1").cast(pl.Float64))).alias("surv"),
        ((pl.col("z").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5).alias("lead"),
        *[e.fill_null(0.0).alias(k) for k, _, e in BINARY],
        *[wz(e).alias(k) for k, _, e in CONTINUOUS],
        *[e.cast(pl.Float64).alias(k) for k, _, e in CLASS_GROUPS],
        pl.col("geo_L").is_null().cast(pl.Float64).alias("m_geo"),
        (pl.col("volatility").is_finite().not_() | pl.col("mkt_pace").is_null()).fill_null(True)
        .cast(pl.Float64).alias("m_env"),
        pl.col("basis_44e").cast(pl.Float64), pl.col("basis_66a").cast(pl.Float64),
    ).with_columns([pl.col(k).fill_null(0.0) for k, _, _ in CONTINUOUS]
                   + [pl.col(c).fill_null(0.0) for c in ("log_len", "log_owner_n", "basis_44e", "basis_66a")])
    d = d.with_columns(
        pl.col("owner_key").map_elements(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 10,
                                         return_dtype=pl.Int8).alias("h10"))
    d = d.with_columns((pl.col("h10") % 2).alias("half"), (pl.col("h10") < 4).alias("lasso_rows"))
    keep = (["surv", "lead", "cell", "reg_year", "owner_key", "cls", "half", "lasso_rows"]
            + [k for k, _, _ in BINARY + CONTINUOUS + CLASS_GROUPS] + CONTROLS + MISSING)
    return d.select(keep)


# ------------------------------------------------------------------ estimator
class Design:
    """Holds FE and cluster codes for a row subset, and demeans columns."""

    def __init__(self, d: pl.DataFrame, fe: str = "cell"):
        self.cell = np.array(d[fe].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy(), dtype=np.int64)
        _, self.cell = np.unique(self.cell, return_inverse=True)
        self.cnt = np.bincount(self.cell)
        og = np.array(d["owner_key"].cast(pl.Categorical).to_physical().to_numpy(), dtype=np.int64)
        _, og = np.unique(og, return_inverse=True)
        self.G = int(og.max()) + 1
        self.n = len(og)
        # rows sorted by owner, cut into chunks that never split an owner, so the
        # clustered "meat" can be accumulated chunk by chunk
        self.order = np.argsort(og, kind="stable")
        ogs = og[self.order]
        starts = np.flatnonzero(np.r_[True, ogs[1:] != ogs[:-1]])
        self.owner_starts = starts
        cuts = [0]
        for target in range(CHUNK, self.n, CHUNK):
            j = int(np.searchsorted(starts, target))
            if j < len(starts) and starts[j] > cuts[-1]:
                cuts.append(int(starts[j]))
        cuts.append(self.n)
        self.cuts = sorted(set(cuts))

    def dm(self, v: np.ndarray) -> np.ndarray:
        return v - (np.bincount(self.cell, weights=v, minlength=len(self.cnt)) / np.maximum(self.cnt, 1))[self.cell]

    def ols(self, y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
        """Within-cell OLS with owner-clustered covariance, streamed in owner-aligned
        chunks so no demeaned copy of X is ever held whole."""
        y = np.asarray(y, dtype=np.float64)
        k = X.shape[1]
        C = len(self.cnt)
        cntc = np.maximum(self.cnt, 1)[:, None]
        M = np.column_stack([np.bincount(self.cell, weights=X[:, j], minlength=C) for j in range(k)]) / cntc
        my = np.bincount(self.cell, weights=y, minlength=C) / cntc[:, 0]
        XtX = np.zeros((k, k)); Xty = np.zeros(k)
        for a, b_ in zip(self.cuts[:-1], self.cuts[1:]):
            ix = self.order[a:b_]
            c = self.cell[ix]
            Xc = X[ix].astype(np.float64) - M[c]
            yc = y[ix] - my[c]
            XtX += Xc.T @ Xc
            Xty += Xc.T @ yc
        keep = np.diag(XtX) > 1e-9
        XtXk = XtX[np.ix_(keep, keep)]
        b = np.linalg.solve(XtXk, Xty[keep])
        meat = np.zeros((keep.sum(), keep.sum()))
        for a, b_ in zip(self.cuts[:-1], self.cuts[1:]):
            ix = self.order[a:b_]
            c = self.cell[ix]
            Xc = (X[ix].astype(np.float64) - M[c])[:, keep]
            e = (y[ix] - my[c]) - Xc @ b
            st = self.owner_starts[(self.owner_starts >= a) & (self.owner_starts < b_)] - a
            s = np.add.reduceat(Xc * e[:, None], st, axis=0)
            meat += s.T @ s
        inv = np.linalg.inv(XtXk)
        n, kk = self.n, int(keep.sum())
        V = (self.G / (self.G - 1)) * ((n - 1) / (n - kk)) * inv @ meat @ inv
        out, idx = {}, np.where(keep)[0]
        li = next((i for i, j in enumerate(idx) if names[j] == "lead"), None)
        for i, j in enumerate(idx):
            # (coefficient, SE, covariance with the lead coefficient)
            out[names[j]] = (float(b[i]), float(math.sqrt(V[i, i])),
                             float(V[i, li]) if li is not None else None)
        out["_V"] = (V, [names[j] for j in idx])
        return out


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def pval(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ------------------------------------------------------------------ models
def build_X(d: pl.DataFrame, factors: list[str], inter: list[str], extra_inter: dict | None = None):
    """Columns: lead, centred factor mains, centred factor x lead, controls, missing flags.
    Stored as float32 (sums are taken in float64 inside Design.ols)."""
    extra_inter = extra_inter or {}
    ncol = 1 + len(factors) + len(inter) + len(extra_inter) + len(CONTROLS) + len(MISSING)
    X = np.empty((d.height, ncol), dtype=np.float32)
    lead = d["lead"].to_numpy().astype(np.float64)
    X[:, 0] = lead
    names, means, j = ["lead"], {}, 1
    for f in factors:
        v = d[f].to_numpy().astype(np.float64)
        means[f] = float(v.mean())
        X[:, j] = v - means[f]; names.append(f); j += 1
    for f in inter:
        v = d[f].to_numpy().astype(np.float64)
        mu = means.get(f, float(v.mean()))
        means.setdefault(f, mu)
        X[:, j] = (v - mu) * lead; names.append(f"{f}:lead"); j += 1
    for kname, v in extra_inter.items():
        X[:, j] = v * lead; names.append(kname); j += 1
    for c in CONTROLS + MISSING:
        X[:, j] = d[c].cast(pl.Float64).to_numpy(); names.append(c); j += 1
    return X, names, means


def main() -> int:
    d = frame()
    log(f"[frame] {d.height:,}")
    y = d["surv"].to_numpy()
    des = Design(d)
    fac = [k for k, _, _ in BINARY] + [k for k, _, _ in CONTINUOUS]
    cg = [k for k, _, _ in CLASS_GROUPS]
    labels = {k: l for k, l, _ in BINARY + CONTINUOUS + CLASS_GROUPS}
    out: dict = {"n": d.height, "labels": labels,
                 "shares": {k: float(d[k].mean()) for k, _, _ in BINARY + CLASS_GROUPS}}

    PART = RES / "combined_model_partial.json"
    if PART.exists() and not os.environ.get("REBUILD"):
        out.update(json.loads(PART.read_text()))
        log("  [resume] alone + joint loaded")
    if "joint_V" not in out:
        stage_one(d, y, des, fac, cg, out)
        RES.mkdir(parents=True, exist_ok=True)
        PART.write_text(json.dumps({k: out[k] for k in ("lead_only", "alone", "joint", "joint_V")}, indent=1))
    gc.collect()
    FULL = RES / "combined_model.json"
    prev = json.loads(FULL.read_text()) if FULL.exists() else {}
    if os.environ.get("SKIP_STAGE2") and "lasso" in prev:
        for k in ("class_slopes_halfA", "class_groups", "class_groups_halfB", "lasso", "post_lasso",
                  "post_lasso_top", "post_lasso_cvmin"):
            if k in prev:
                out[k] = prev[k]
        log("  [resume] class groups + lasso loaded")
    else:
        stage_two(d, y, des, fac, cg, out)
    stage_groups(d, fac, out)
    FULL.write_text(json.dumps(out, indent=1, default=lambda o: None if isinstance(o, np.ndarray) else str(o)))
    log(f"[done] peak working set {peak_gb():.2f} GB")
    return 0


def stage_one(d, y, des, fac, cg, out):
    # lead alone
    X, nm, _ = build_X(d, [], [])
    r = des.ols(y, X, nm)
    out["lead_only"] = {"b": r["lead"][0], "se": r["lead"][1]}
    log(f"  lead alone: {r['lead'][0]:+.2f} ({r['lead'][1]:.2f})")

    # alone models
    alone = {}
    for f in fac + cg:
        mains = [f] if f not in cg else []
        inter = [f]
        if f in YEAR_TYPES:            # boom and bust against years that are neither, net of trend
            mains, inter = list(YEAR_TYPES) + ["trend"], list(YEAR_TYPES) + ["trend"]
        X, nm, means = build_X(d, mains, inter)
        r = des.ols(y, X, nm)
        alone[f] = {"lead": r["lead"], "inter": r.get(f"{f}:lead"), "main": r.get(f)}
        log(f"  alone {f:12s} main {r.get(f, (float('nan'),0))[0]:+.2f}  x lead {r[f'{f}:lead'][0]:+.2f} "
            f"({r[f'{f}:lead'][1]:.2f})")
    out["alone"] = alone

    # joint
    X, nm, means = build_X(d, fac, fac + cg)
    r = des.ols(y, X, nm)
    joint = {"lead": r["lead"], "means": means}
    for f in fac + cg:
        joint[f] = {"main": r.get(f), "inter": r.get(f"{f}:lead")}
    out["joint"] = joint
    V, vn = r["_V"]
    out["joint_V"] = {"names": vn, "V": V.tolist()}
    log(f"  joint lead {r['lead'][0]:+.2f} ({r['lead'][1]:.2f})")


def lincomb(r: dict, w: dict) -> tuple[float, float]:
    """Estimate and SE of sum_k w_k * b_k from an ols() result."""
    V, vn = r["_V"]
    ix = {n: i for i, n in enumerate(vn)}
    a = np.zeros(len(vn))
    b = 0.0
    for k, wk in w.items():
        a[ix[k]] = wk
        b += wk * r[k][0]
    return float(b), float(math.sqrt(max(a @ V @ a, 0)))


def stage_groups(d, fac, out):
    """Each class group against all other classes: registration-year fixed effects only
    (so the group's survival level is identified), the factors and their lead
    interactions held, the group's own survival difference and lead interaction."""
    res = {}
    grp_of = {c: g for g, (name, members) in enumerate(out["class_groups"].items()) for c in members}
    gnames = list(out["class_groups"].keys())
    specs = [(k, lab, d) for k, lab, _ in CLASS_GROUPS]
    B = d.filter(pl.col("half") == 1)
    for gi, gname in enumerate(gnames):
        col = f"_kg{gi}"
        B = B.with_columns((pl.col("cls").replace_strict(grp_of, return_dtype=pl.Int32) == gi)
                           .cast(pl.Float64).alias(col))
        specs.append((col, gname + " (held-out half)", B))
    for key, lab, dd in specs:
        des = Design(dd, fe="reg_year")
        X, nm, means = build_X(dd, fac + [key], fac + [key])
        r = des.ols(dd["surv"].to_numpy(), X, nm)
        m = means[key]
        res[lab] = {"share": m, "n": dd.height,
                    "survival": r[key][:2], "inter": r[f"{key}:lead"][:2],
                    "lead_in": lincomb(r, {"lead": 1.0, f"{key}:lead": 1 - m}),
                    "lead_out": lincomb(r, {"lead": 1.0, f"{key}:lead": -m})}
        log(f"  group {lab[:40]:40s} surv {r[key][0]:+.2f}  lead in {res[lab]['lead_in'][0]:+.2f} "
            f"out {res[lab]['lead_out'][0]:+.2f}")
        del X
        gc.collect()
    out["class_group_vs_other"] = res


def stage_two(d, y, des, fac, cg, out):
    # ---- composite class groups (cross-fit)
    cls_list = sorted(d["cls"].unique().to_list())
    A = d.filter(pl.col("half") == 0)
    desA = Design(A)
    XA, nmA, _ = build_X(A, fac, fac)
    cl = A["cls"].to_numpy()
    slopes_cols = [((cl == c).astype(float)) * XA[:, 0] for c in cls_list]
    XA2 = np.column_stack([XA[:, 1:]] + slopes_cols)   # drop pooled lead: class slopes span it
    nmA2 = nmA[1:] + [f"cls{c}:lead" for c in cls_list]
    rA = desA.ols(A["surv"].to_numpy(), XA2, nmA2)
    b = np.array([rA[f"cls{c}:lead"][0] for c in cls_list])
    s = np.array([rA[f"cls{c}:lead"][1] for c in cls_list])
    # empirical-Bayes shrinkage toward the precision-weighted mean
    w = 1 / s ** 2
    mu = float((w * b).sum() / w.sum())
    tau2 = max(0.0, float(((b - mu) ** 2 - s ** 2).mean()))
    shr = mu + (tau2 / (tau2 + s ** 2)) * (b - mu)
    from sklearn.cluster import KMeans
    K = 3
    km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(shr.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.ravel())        # 0 = most negative lead effect
    rank = {int(o): i for i, o in enumerate(order)}
    grp = {c: rank[int(g)] for c, g in zip(cls_list, km.labels_)}
    names_g = {0: "classes where leading costs most", 1: "middle classes", 2: "classes where leading costs least"}
    out["class_slopes_halfA"] = {c: {"b": float(bb), "se": float(ss), "shrunk": float(sh), "group": grp[c]}
                                 for c, bb, ss, sh in zip(cls_list, b, s, shr)}
    out["class_groups"] = {names_g[g]: [c for c in cls_list if grp[c] == g] for g in range(K)}
    del A, desA, XA, XA2, slopes_cols
    gc.collect()
    B = d.filter(pl.col("half") == 1)
    desB = Design(B)
    XB, nmB, _ = build_X(B, fac, fac)
    gB = B["cls"].replace_strict(grp, return_dtype=pl.Int32).to_numpy()
    gcols = [((gB == g).astype(float)) * XB[:, 0] for g in range(K)]
    XB2 = np.column_stack([XB[:, 1:]] + gcols)
    nmB2 = nmB[1:] + [f"group{g}:lead" for g in range(K)]
    rB = desB.ols(B["surv"].to_numpy(), XB2, nmB2)
    out["class_groups_halfB"] = {names_g[g]: {"b": rB[f"group{g}:lead"][0], "se": rB[f"group{g}:lead"][1],
                                              "n": int((gB == g).sum())} for g in range(K)}
    log("  class groups (half B): " + "; ".join(f"{names_g[g]} {rB[f'group{g}:lead'][0]:+.2f}"
                                                for g in range(K)))

    del B, desB, XB, XB2, gcols
    gc.collect()
    # ---- LASSO over lead interactions, mains/controls/FE partialled out.
    # Fitted on 40% of owners (owner-hash split) in float32 to fit in memory;
    # the selected terms are re-estimated on all registrations below.
    from sklearn.linear_model import Lasso, LassoCV, lasso_path
    from sklearn.model_selection import GroupKFold
    S_ = d.filter(pl.col("lasso_rows"))
    desS = Design(S_)
    Xm, nmm, _ = build_X(S_, fac, [])
    base = np.column_stack([desS.dm(Xm[:, j]) for j in range(Xm.shape[1])])
    lead = Xm[:, 0].copy()
    del Xm
    cls = S_["cls"].to_numpy()
    gall = S_["cls"].replace_strict(grp, return_dtype=pl.Int32).to_numpy()
    cand_names = list(fac + cg) + [f"class {c}" for c in cls_list] + [names_g[g] for g in range(K)]
    C = np.empty((S_.height, len(cand_names)), dtype=np.float32)
    for j, nmj in enumerate(cand_names):
        if nmj.startswith("class "):
            v = (cls == nmj[6:]).astype(float)
        elif nmj in names_g.values():
            v = (gall == [g for g in names_g if names_g[g] == nmj][0]).astype(float)
        else:
            v = S_[nmj].to_numpy().astype(float)
        C[:, j] = desS.dm((v - v.mean()) * lead)
    yd = desS.dm(S_["surv"].to_numpy())
    Q, _ = np.linalg.qr(base)                    # partial out lead, mains, controls
    yr = (yd - Q @ (Q.T @ yd)).astype(np.float32)
    for j in range(C.shape[1]):
        c = C[:, j].astype(float)
        c = c - Q @ (Q.T @ c)
        sdj = c.std()
        C[:, j] = c / (sdj if sdj > 0 else 1)
    del Q, base
    gc.collect()
    owners = np.array(S_["owner_key"].cast(pl.Categorical).to_physical().to_numpy(), dtype=np.int64)
    folds = list(GroupKFold(n_splits=5).split(C, yr, owners))
    lcv = LassoCV(alphas=40, cv=folds, n_jobs=1, fit_intercept=False, max_iter=5000,
                  precompute=False).fit(C, yr)
    mse = lcv.mse_path_.mean(1)
    se_ = lcv.mse_path_.std(1) / math.sqrt(lcv.mse_path_.shape[1])
    i_min = int(mse.argmin())
    i_1se = int(np.where(mse <= mse[i_min] + se_[i_min])[0].min())   # alphas decrease
    a_min, a_1se = float(lcv.alphas_[i_min]), float(lcv.alphas_[i_1se])
    alphas, coefs, _ = lasso_path(C, yr, alphas=np.geomspace(lcv.alphas_[0], a_min / 3, 60))
    entry = {}
    for j, nmj in enumerate(cand_names):
        nz = np.where(np.abs(coefs[j]) > 0)[0]
        if len(nz):
            entry[nmj] = float(alphas[nz.min()])
    ranked = sorted(entry.items(), key=lambda kv: -kv[1])
    sel = {}
    for tag, a_ in (("min", a_min), ("1se", a_1se)):
        m = Lasso(alpha=a_, fit_intercept=False, max_iter=10000).fit(C, yr)
        sel[tag] = [cand_names[j] for j in np.where(m.coef_ != 0)[0]]
    out["lasso"] = {"alpha_min": a_min, "alpha_1se": a_1se, "entry_order": ranked,
                    "selected_min": sel["min"], "selected_1se": sel["1se"], "n_rows": int(S_.height),
                    "r2_partial_cv_min": float(1 - mse[i_min] / yr.var())}
    log(f"  lasso: {len(sel['min'])} terms at CV min, {len(sel['1se'])} at 1-SE; first in: "
        + ", ".join(k for k, _ in ranked[:10]))
    del C, S_, desS
    gc.collect()

    # ---- post-lasso OLS on the 1-SE set
    Xm, nmm, _ = build_X(d, fac, [])
    leadF = Xm[:, 0]
    clsF = d["cls"].to_numpy()
    gF = d["cls"].replace_strict(grp, return_dtype=pl.Int32).to_numpy()
    extra = []
    for k in sel["1se"]:
        if k.startswith("class "):
            v = (clsF == k[6:]).astype(float)
        elif k in names_g.values():
            v = (gF == [g for g in names_g if names_g[g] == k][0]).astype(float)
        else:
            v = d[k].to_numpy().astype(float)
        extra.append((v - v.mean()) * leadF)
    Xp = np.column_stack([Xm] + extra)
    nmp = nmm + [f"{k}:lead" for k in sel["1se"]]
    del Xm, extra
    rp = des.ols(y, Xp, nmp)
    out["post_lasso"] = {"lead": rp["lead"], **{k: rp[f"{k}:lead"] for k in sel["1se"]},
                         "mains": {f: rp.get(f) for f in fac}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "combined_model.json").write_text(json.dumps(
        out, indent=1, default=lambda o: None if isinstance(o, np.ndarray) else str(o)))
    log("[done]")


if __name__ == "__main__":
    raise SystemExit(main())
