"""Geographic clustering of a group of filers, with more than one hub allowed.

Measure. For a group of N US filers, the co-location index is the average,
over all pairs of distinct filers, of a distance kernel

    C = sum_{i != j} k(d_ij) / (N (N - 1)),   k(d) = max(0, 1 - d / 200 miles)

so a pair in the same place scores 1, a pair 100 miles apart 0.5, and a pair
more than 200 miles apart 0. C is the chance that two filers drawn at random
are neighbours, weighted by how near. It needs no single centre: a group
split evenly between the Bay Area and New York scores about 0.5 x 0.5 x 2 =
0.5 of its within-hub value, while one spread evenly across the country
scores near zero. Every filer in one building scores 1.

Filers crowd into populous places whatever they sell, so C is compared with
the same index for all filers in the same class and years (the baseline B):

    L = (C - B) / (1 - B)

L is 0 when the group is placed like its class, 1 when every filer is in one
place, and negative when the group is more dispersed than its class. This is
the distance-based approach of Duranton and Overman (2005), who compare an
industry's pairwise-distance distribution with a counterfactual drawn from
all plants; here the comparison is summarized in one number at a 200-mile
bandwidth.

Hubs. The kernel-weighted share of a group's filers near a place is
s(x) = sum_j k(d_xj) / N. A hub is a place where s >= 10% and s is at least
1.2 times the class baseline there (the margin at which both the Bay Area and
New York register as hubs for social-networking filings, 2004-2012; at 1.5,
New York, where 14% of all filers sit, drops out); hubs are picked greedily (highest s
first, then everything within 200 miles of it is set aside) and named by the
commonest owner city within 50 miles.

Places are 0.1-degree grid cells (about 7 miles), each at the mean position of
its filers' ZIP centroids, so "the same place" means the same cell.

Run as a script for the validation table: internet-vocabulary filers
1998-2001 against three single-industry examples.
Output: paper/results/geo_cluster_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
EARTH_MI = 3958.8
BW = 200.0
GRID = 0.1


def cells_for(ll: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assign each serial a grid cell; return (serial->cell, cell table)."""
    ll = ll.with_columns(
        ((pl.col("lat") / GRID).floor().cast(pl.Int32).cast(pl.Utf8) + "_"
         + (pl.col("lon") / GRID).floor().cast(pl.Int32).cast(pl.Utf8)).alias("cell_key"))
    cells = ll.group_by("cell_key").agg(pl.col("lat").mean(), pl.col("lon").mean(),
                                         pl.len().alias("n_all")).sort("cell_key").with_row_index("cell")
    m = ll.join(cells.select("cell_key", "cell"), on="cell_key").select(
        "serial_number", "cell", "owner_city", "owner_state")
    return m, cells


def kernel(cells: pl.DataFrame) -> sp.csr_matrix:
    X = np.radians(cells.select("lat", "lon").to_numpy())
    tree = BallTree(X, metric="haversine")
    ind, dist = tree.query_radius(X, r=BW / EARTH_MI, return_distance=True)
    rows = np.repeat(np.arange(len(ind)), [len(i) for i in ind])
    cols = np.concatenate(ind)
    d = np.concatenate(dist) * EARTH_MI
    return sp.csr_matrix((1 - d / BW, (rows, cols)), shape=(len(ind), len(ind)))


def co_location(counts: sp.csr_matrix, K: sp.csr_matrix) -> np.ndarray:
    """C for each row of a (groups x cells) count matrix."""
    counts = sp.csr_matrix(counts, dtype=np.float64)
    q = np.asarray(counts.multiply(counts @ K).sum(1)).ravel()   # n'Kn incl. self pairs
    N = np.asarray(counts.sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (q - N) / (N * (N - 1))    # self pairs have k = 1; remove them


def localization(C: np.ndarray, B: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (C - B) / (1 - B)


def hubs(n: np.ndarray, base: np.ndarray, K: sp.csr_matrix, cells: pl.DataFrame,
         names: dict[int, str] | None = None, min_share=0.10, min_ratio=1.2) -> list[dict]:
    s = K @ n / n.sum()
    b = K @ base / base.sum()
    ok = (s >= min_share) & (s >= min_ratio * b)
    out = []
    s_left = np.where(ok, s, 0.0)
    while s_left.max() > 0:
        c = int(s_left.argmax())
        near = K.getrow(c).indices
        out.append({"cell": c, "share_nearby": float(s[c]), "class_share_nearby": float(b[c]),
                    "name": (names or {}).get(c, "")})
        s_left[near] = 0.0
    return out


def city_names(m: pl.DataFrame, cells: pl.DataFrame) -> dict[int, str]:
    top = m.group_by("cell", "owner_city", "owner_state").len().sort("len", descending=True).unique(
        "cell", keep="first")
    return {int(c): f"{ci.title()}, {st}" for c, ci, st in top.select("cell", "owner_city", "owner_state").iter_rows()}


def _validation() -> int:
    ll = pl.read_parquet(PROC / "owner_ll.parquet")
    m, cells = cells_for(ll)
    K = kernel(cells)
    print(f"cells {cells.height:,}; kernel nnz {K.nnz:,}", flush=True)
    names = city_names(m, cells)
    # name a hub by the commonest city within 50 miles (kernel >= 0.75)
    def hub_name(c, sub):
        near = K.getrow(c)
        near = near.indices[near.data >= 0.75]
        t = sub.filter(pl.col("cell").is_in(near.tolist())).group_by("owner_city", "owner_state").len().sort(
            "len", descending=True).head(1)
        return f"{t['owner_city'][0].title()}, {t['owner_state'][0]}" if t.height else names.get(c, "")

    tests = {
        "internet vocabulary, classes 9/35/38/42, 1998-2001": (["009", "035", "038", "042"], 1998, 2001,
            r"(?i)\binternet\b|\bonline\b|\bweb ?site|world wide web|e-commerce"),
        "semiconductors, class 9, 1995-2005": (["009"], 1995, 2005, r"(?i)semiconductor"),
        "wine, class 33, 2000-2010": (["033"], 2000, 2010, r"(?i)\bwines?\b"),
        "country music, class 41, 1995-2010": (["041"], 1995, 2010, r"(?i)country music"),
        "casino gaming, class 41, 1995-2010": (["041"], 1995, 2010, r"(?i)casino"),
        "biotechnology, classes 1/5/42, 1995-2010": (["001", "005", "042"], 1995, 2010,
            r"(?i)biotechnolog|monoclonal|genetic engineering"),
        "social networking, classes 38/42/45, 2004-2012": (["038", "042", "045"], 2004, 2012,
            r"(?i)social network"),
        "online advertising, classes 35/42, 2005-2015": (["035", "042"], 2005, 2015,
            r"(?i)online advertising|advertising.{0,40}(internet|online|mobile)|ad network"),
        "software as a service, class 42, 2008-2016": (["042"], 2008, 2016,
            r"(?i)software as a service|\bsaas\b"),
        "hip hop, classes 9/25/41, 1995-2010": (["009", "025", "041"], 1995, 2010, r"(?i)hip[- ]hop"),
    }
    out = {}
    for label, (classes, lo, hi, rx) in tests.items():
        d = pl.concat([pl.read_parquet(PROC / f"tm_class{c}.parquet",
                                       columns=["serial_number", "filing_date", "goods_services"])
                       for c in classes]).unique("serial_number").with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy")).filter(
            pl.col("fy").is_between(lo, hi)).join(m, on="serial_number", how="inner")
        g = d.filter(pl.col("goods_services").fill_null("").str.contains(rx))
        vec = lambda df: np.bincount(df["cell"].to_numpy(), minlength=cells.height).astype(float)
        n, base = vec(g), vec(d)
        C, B = co_location(sp.csr_matrix(np.vstack([n, base])), K)
        L = localization(np.array([C]), np.array([B]))[0]
        H = hubs(n, base, K, cells)
        for h in H:
            h["name"] = hub_name(h["cell"], g)
        out[label] = {"n": int(n.sum()), "n_class": int(base.sum()), "C": float(C), "B": float(B),
                      "L": float(L), "hubs": H}
        print(f"\n{label}: n={int(n.sum()):,}  C={C:.3f}  baseline={B:.3f}  L={L:.3f}")
        for h in H:
            print(f"   hub {h['name']:28s} {h['share_nearby']:.1%} of group nearby vs {h['class_share_nearby']:.1%} of class")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "geo_cluster_validation.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_validation())
