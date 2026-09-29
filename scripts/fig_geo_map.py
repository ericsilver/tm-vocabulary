"""Map of filer locations for four themes, with the hubs the co-location rule detects.

Each panel: grey = all US filers in the same classes and years (the baseline),
coloured = filers whose description uses the theme's vocabulary, circles = the
detected hubs (200-mile radius), labelled by the commonest owner city.

Output: paper/figures/fig_geo_map.pdf (+ .png)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import geo_cluster as gc_  # noqa: E402

PROC, FIG = gc_.PROC, REPO / "paper" / "figures"
PANELS = [
    ("Semiconductors (class 9, 1995-2005)", ["009"], 1995, 2005, r"(?i)semiconductor"),
    ("Social networking (classes 38, 42, 45, 2004-12)", ["038", "042", "045"], 2004, 2012, r"(?i)social network"),
    ("Biotechnology (classes 1, 5, 42, 1995-2010)", ["001", "005", "042"], 1995, 2010,
     r"(?i)biotechnolog|monoclonal|genetic engineering"),
    ("Casino gaming (class 41, 1995-2010)", ["041"], 1995, 2010, r"(?i)casino"),
]
ACCENT, BASE = "#0072B2", "#C9C9C9"


def main() -> int:
    ll = pl.read_parquet(PROC / "owner_ll.parquet")
    m, cells = gc_.cells_for(ll)
    K = gc_.kernel(cells)
    m = m.join(ll.select("serial_number", "lat", "lon"), on="serial_number")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
    for ax, (title, classes, lo, hi, rx) in zip(axes.flat, PANELS):
        d = pl.concat([pl.read_parquet(PROC / f"tm_class{c}.parquet",
                                       columns=["serial_number", "filing_date", "goods_services"])
                       for c in classes]).unique("serial_number").with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy")).filter(
            pl.col("fy").is_between(lo, hi)).join(m, on="serial_number", how="inner").filter(
            pl.col("lon").is_between(-125, -66) & pl.col("lat").is_between(24, 50))
        g = d.filter(pl.col("goods_services").fill_null("").str.contains(rx))
        vec = lambda df: np.bincount(df["cell"].to_numpy(), minlength=cells.height).astype(float)
        n, base = vec(g), vec(d)
        C, B = gc_.co_location(__import__("scipy").sparse.csr_matrix(np.vstack([n, base])), K)
        L = gc_.localization(np.array([C]), np.array([B]))[0]
        H = gc_.hubs(n, base, K, cells)
        bs = d.sample(min(d.height, 60000), seed=1)
        ax.scatter(bs["lon"], bs["lat"], s=1.2, color=BASE, alpha=0.5, linewidths=0, rasterized=True)
        ax.scatter(g["lon"], g["lat"], s=2.5, color=ACCENT, alpha=0.45, linewidths=0, rasterized=True)
        key = []
        for i, h in enumerate(H, 1):
            c = cells.row(h["cell"], named=True)
            near = K.getrow(h["cell"])
            near = near.indices[near.data >= 0.75]
            t_ = g.filter(pl.col("cell").is_in(near.tolist())).group_by("owner_city", "owner_state").len().sort(
                "len", descending=True).head(1)
            name = f"{t_['owner_city'][0].title()}, {t_['owner_state'][0]}" if t_.height else ""
            r_lat = 200 / 69.0
            th = np.linspace(0, 2 * np.pi, 120)
            ax.plot(c["lon"] + r_lat / np.cos(np.radians(c["lat"])) * np.cos(th), c["lat"] + r_lat * np.sin(th),
                    color="#222222", linewidth=0.9)
            ax.text(c["lon"], c["lat"] + r_lat + 0.3, str(i), ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#222222")
            key.append(f"{i}  {name}: {100*h['share_nearby']:.0f}% of these filers nearby, "
                       f"{100*h['class_share_nearby']:.0f}% of all filers")
        if key:
            ax.text(0.01, 0.01, "\n".join(key), transform=ax.transAxes, fontsize=6.5, color="#222222",
                    va="bottom", ha="left", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        ax.set_title(f"{title}\n{int(n.sum()):,} filers; co-location L = {L:.2f}", fontsize=8.5, loc="left",
                     color="#222222")
        ax.set_xlim(-129, -66); ax.set_ylim(20.5, 51.5)
        ax.set_aspect(1 / np.cos(np.radians(37)))
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_geo_map.pdf", dpi=200); fig.savefig(FIG / "fig_geo_map.png", dpi=150)
    print("wrote", FIG / "fig_geo_map.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
