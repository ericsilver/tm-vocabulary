"""Where do the costs of leading and the benefits of atypicality accrue, and do they persist?

Registrations 2002-2018 (one row per serial), the paper's gate sample. Two
survival outcomes:
  proof    passed the five-year proof of continued use (all registrations)
  renewal  renewed at year ten, among registrations that passed the proof
           (registered 2002-2013, terminal status 800 renewed vs 710/900 dead)
Lead and atypicality both enter as percentiles within class x registration year,
centred, so each coefficient is the survival difference between the top and the
bottom of that class-year's distribution. Class x registration-year fixed
effects, the paper's controls, owner-clustered SEs (the streaming estimator of
combined_model.py).

Estimates: pooled; by Nice class; by the composite class groups; by
registration period; and, for the moderators that change the cost of leading
(counsel, platform, general-purpose-technology vocabulary), at both stages.

Output: paper/results/where_it_accrues.json
        paper/figures/fig_persistence.pdf (+ .png)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import combined_model as cm  # noqa: E402
from class_profiles import NAMES  # noqa: E402

PROC, RES, FIG = cm.PROC, cm.RES, REPO / "paper" / "figures"
CTRL = ["counsel", "itu", "log_len", "log_owner_n", "foreign", "china", "basis_44e", "basis_66a"]
PERIODS = [(2002, 2007), (2008, 2012), (2013, 2018)]
MODS = ["counsel", "platform", "gpt"]


def load() -> pl.DataFrame:
    d = cm.frame()
    classes = sorted(d["cls"].unique().to_list())
    A = pl.concat([pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                                   columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"])
                   .with_columns(pl.lit(c).alias("cls")) for c in classes]).unique(["serial_number", "cls"])
    A = A.with_columns(((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("atyp_raw")) \
         .select("serial_number", "cls", "atyp_raw")
    st = pl.concat([pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "status_code"])
                    for c in classes]).unique("serial_number")
    d = d.join(A, on=["serial_number", "cls"], how="left").join(st, on="serial_number", how="left")
    d = d.with_columns(
        ((pl.col("atyp_raw").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5)
        .fill_null(0.0).alias("atyp"),
        pl.when(pl.col("status_code") == "800").then(100.0)
        .when(pl.col("status_code").is_in(["710", "900"])).then(0.0).otherwise(None).alias("renew"))
    return d


def fit(d: pl.DataFrame, y: str, extra: dict[str, np.ndarray] | None = None) -> dict:
    """y on lead + atyp (+ extra columns) + controls; returns lead, atyp and extra coefficients."""
    d = d.filter(pl.col(y).is_not_null())
    if d.height < 2000:
        return {"n": d.height}
    des = cm.Design(d)
    cols = [d["lead"].to_numpy(), d["atyp"].to_numpy()]
    names = ["lead", "atyp"]
    if extra:
        mask = d["_keep"].to_numpy() if "_keep" in d.columns else None
        for k, v in extra.items():
            vv = v if mask is None else v[mask]
            cols.append(vv); names.append(k)
    for c in CTRL:
        if extra and c in extra:          # a moderator that is also a control enters once
            continue
        cols.append(d[c].cast(pl.Float64).fill_null(0).to_numpy()); names.append(c)
    X = np.column_stack(cols).astype(np.float32)
    r = des.ols(d[y].to_numpy(), X, names)
    out = {"n": d.height, "base": float(d[y].mean())}
    for k in names[:2] + list((extra or {}).keys()):
        if k in r:
            out[k] = list(r[k][:2])
    return out


def stages(d: pl.DataFrame) -> dict:
    proof = fit(d, "surv")
    ren = fit(d.filter((pl.col("surv") == 100) & pl.col("reg_year").is_between(2002, 2013)), "renew")
    return {"proof": proof, "renewal": ren}


def main() -> int:
    d = load()
    cm.log(f"[frame] {d.height:,}; renewal outcome known for "
           f"{d.filter((pl.col('surv') == 100) & pl.col('reg_year').is_between(2002, 2013) & pl.col('renew').is_not_null()).height:,}")
    out = {"pooled": stages(d)}
    cm.log(f"  pooled: {json.dumps(out['pooled'])[:300]}")
    # moderators at both stages
    mods = {}
    for m in MODS:
        res = {}
        for tag, dd, y in (("proof", d, "surv"),
                           ("renewal", d.filter((pl.col("surv") == 100) & pl.col("reg_year").is_between(2002, 2013)
                                                & pl.col("renew").is_not_null()), "renew")):
            v = dd[m].to_numpy().astype(float)
            vc = v - v.mean()
            lead = dd["lead"].to_numpy()
            atyp = dd["atyp"].to_numpy()
            r = fit(dd, y, {f"{m}": vc, f"{m}:lead": vc * lead, f"{m}:atyp": vc * atyp})
            res[tag] = r
        mods[m] = res
        cm.log(f"  moderator {m}: proof x lead {res['proof'][f'{m}:lead'][0]:+.2f}, renewal x lead "
               f"{res['renewal'][f'{m}:lead'][0]:+.2f}")
    out["moderators"] = mods
    # by period
    out["by_period"] = {f"{lo}-{hi}": fit(d.filter(pl.col("reg_year").is_between(lo, hi)), "surv")
                        for lo, hi in PERIODS}
    # by class
    byc = {}
    for c in sorted(d["cls"].unique().to_list()):
        byc[c] = stages(d.filter(pl.col("cls") == c))
        cm.log(f"  class {c}: proof lead {byc[c]['proof'].get('lead', [float('nan')])[0]:+.1f} "
               f"renewal lead {byc[c]['renewal'].get('lead', [float('nan')])[0]:+.1f}")
    out["by_class"] = byc
    # by composite class group
    groups = json.loads((RES / "combined_model.json").read_text())["class_groups"]
    out["by_group"] = {g: stages(d.filter(pl.col("cls").is_in(members))) for g, members in groups.items()}
    # by class group and period (proof)
    out["group_period"] = {g: {f"{lo}-{hi}": fit(d.filter(pl.col("cls").is_in(members)
                                                        & pl.col("reg_year").is_between(lo, hi)), "surv")
                               for lo, hi in PERIODS} for g, members in groups.items()}
    (RES / "where_it_accrues.json").write_text(json.dumps(out, indent=1))

    # ---- figure: per-class proof vs renewal, lead and atypicality
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    grp_of = {c: i for i, (g, ms) in enumerate(groups.items()) for c in ms}
    colors = ["#D55E00", "#8C8C8C", "#0072B2"]          # costs most / middle / costs least
    for ax, key, title in ((axes[0], "lead", "Lead: most leading minus most lagging"),
                           (axes[1], "atyp", "Atypicality: most unusual minus most typical")):
        xs, ys, ns, cs, labs = [], [], [], [], []
        for c, v in byc.items():
            if key in v["proof"] and key in v["renewal"]:
                xs.append(v["proof"][key][0]); ys.append(v["renewal"][key][0])
                ns.append(v["proof"]["n"]); cs.append(colors[grp_of.get(c, 1)]); labs.append(int(c))
        xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
        ax.axhline(0, color="#bbbbbb", linewidth=0.8); ax.axvline(0, color="#bbbbbb", linewidth=0.8)
        ax.scatter(xs, ys, s=8 + 120 * ns / ns.max(), c=cs, alpha=0.85, edgecolors="white", linewidths=0.6)
        for x_, y_, l_ in zip(xs, ys, labs):
            ax.annotate(str(l_), (x_, y_), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color="#333333")
        ax.set_xlabel("Effect on passing the five-year proof (points)", fontsize=8, color="#555555")
        ax.set_ylabel("Effect on renewal at year ten, given the proof (points)", fontsize=8, color="#555555")
        ax.set_title(title, fontsize=9.5, loc="left", color="#222222")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(labelsize=7, colors="#555555")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[i], label=g)
               for i, g in enumerate(groups.keys())]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_persistence.pdf"); fig.savefig(FIG / "fig_persistence.png", dpi=160)
    cm.log(f"[done] peak {cm.peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
