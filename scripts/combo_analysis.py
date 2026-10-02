"""Combinations of themes and survival: breadth (entropy), specific pairs, and how surprising a pairing is.

Sample: the gate sample (registrations 2002-2018, unique serials). Outcome: survival
of the five-year proof of continued use (percentage points). Every estimate uses
class x registration-year fixed effects and owner-clustered standard errors, with
the paper's controls (counsel, intent-to-use, log description length, log owner
filing count, domicile, foreign basis).

1. Breadth. H = entropy of the filing's theme mix (theme_mix_entropy.py). Its
   correlations with description length, lead and atypicality; survival by H decile;
   and H with its interaction with lead in one model.
2. Pairs. A filing combines two themes when its second theme carries at least 0.10
   of its mix; otherwise it is single-theme. For each unordered pair with at least
   3,000 registrations: excess survival over its class-year cell and the lead effect
   inside the pair.
3. Pair surprise. Within each class and filing year, how much more or less often a
   pair appears than its two themes' frequencies predict:
       PMI = log P(a,b) / (P(a) P(b)),   pairs counted over the five prior filing years
   Low PMI = an unusual pairing for that industry at the time. PMI enters the
   survival model with its lead interaction, with H and the controls held.

Output: paper/results/combo_analysis.json
"""
from __future__ import annotations

import json
import math
import re
import ast
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from combined_model import Design  # noqa: E402

CONTROLS = ["has_attorney", "itu", "log_len", "log_owner_n", "dom_us", "dom_cn", "basis_44e", "basis_66a"]
MIN_PAIR = 3000
PRES = 0.10


def labels() -> dict:
    src = (REPO / "scripts" / "themes_t50_page.py").read_text(encoding="utf-8")
    m = re.search(r"LABELS\s*=\s*(\{.*?\n\})", src, re.S) or re.search(r"LABELS\s*=\s*(\[.*?\n\])", src, re.S)
    L = ast.literal_eval(m.group(1))
    return {k: L[k] for k in range(50)}


def pval(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2))


def fit(d: pl.DataFrame, xs: list[str]) -> dict:
    des = Design(d)
    X = np.column_stack([d[c].cast(pl.Float64).fill_null(0).to_numpy() for c in xs])
    r = des.ols(d["surv"].to_numpy(), X, xs)
    return {k: {"b": r[k][0], "se": r[k][1], "p": pval(r[k][0], r[k][1])} for k in xs if k in r}


def main() -> int:
    lab = labels()
    f = pl.read_parquet(PROC / "battery_frame.parquet",
                        columns=["serial_number", "cls", "cell", "owner_key", "failed1", "z", "fy"] + CONTROLS)
    e = pl.read_parquet(PROC / "theme_mix_entropy.parquet")
    f = f.join(e, on=["serial_number", "cls"], how="inner")
    raw = pl.concat([pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                                     columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"])
                     .with_columns(pl.lit(c).alias("cls")) for c in sorted(f["cls"].unique().to_list())]
                    ).unique(["serial_number", "cls"])
    f = f.join(raw, on=["serial_number", "cls"], how="left").with_columns(
        (100 * (1 - pl.col("failed1").cast(pl.Float64))).alias("surv"),
        ((pl.col("z").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5).alias("lead"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("atyp"),
        *[pl.col(c).cast(pl.Float64) for c in ["has_attorney", "itu", "dom_us", "dom_cn", "basis_44e", "basis_66a"]],
    ).with_columns(
        ((pl.col("H") - pl.col("H").mean()) / pl.col("H").std()).alias("Hz"),
        (pl.col("w2") >= PRES).alias("is_pair"),
        pl.min_horizontal("top1", "top2").alias("pa"), pl.max_horizontal("top1", "top2").alias("pb"),
    ).with_columns((pl.col("Hz") * pl.col("lead")).alias("Hz_x_lead"))
    print(f"[frame] {f.height:,} registrations; {f['is_pair'].mean():.1%} combine two themes", flush=True)
    out: dict = {"n": f.height, "share_pair": float(f["is_pair"].mean())}

    # 1. breadth -----------------------------------------------------------
    def within_corr(a, b):
        g = f.select("cls", a, b).drop_nulls().with_columns(
            (pl.col(a) - pl.col(a).mean().over("cls")).alias("_a"),
            (pl.col(b) - pl.col(b).mean().over("cls")).alias("_b"))
        return float(np.corrcoef(g["_a"].to_numpy(), g["_b"].to_numpy())[0, 1])

    out["H_corr_within_class"] = {v: within_corr("H", v) for v in ("log_len", "z", "atyp")}
    out["H_summary"] = {"mean": float(f["H"].mean()), "sd": float(f["H"].std()),
                        "eff_n_median": float(np.exp(f["H"]).median()),
                        "n_present_dist": {int(k): int(v) for k, v in f.group_by("n_pres").len().iter_rows()}}
    print("  H within-class corr:", {k: round(v, 2) for k, v in out["H_corr_within_class"].items()}, flush=True)
    dec = f.with_columns(
        (pl.col("surv") - pl.col("surv").mean().over("cell") + pl.col("surv").mean()).alias("adj"),
        ((pl.col("H").rank("ordinal") - 1) * 10 // pl.len()).cast(pl.Int8).alias("Hd"))
    out["surv_by_H_decile"] = [float(v) for v in dec.group_by("Hd").agg(pl.col("adj").mean()).sort("Hd")["adj"]]
    # same, within description-length terciles (length drives breadth mechanically)
    dec = dec.with_columns(((pl.col("log_len").rank("ordinal") - 1) * 3 // pl.len()).cast(pl.Int8).alias("Lt"))
    out["surv_by_H_decile_within_length_tercile"] = {
        int(t): [float(v) for v in dec.filter(pl.col("Lt") == t).with_columns(
            ((pl.col("H").rank("ordinal") - 1) * 10 // pl.len()).cast(pl.Int8).alias("Hd2")).group_by("Hd2").agg(
            pl.col("adj").mean()).sort("Hd2")["adj"]] for t in range(3)}
    out["model_breadth"] = fit(f, ["lead", "Hz", "Hz_x_lead"] + CONTROLS)
    mb = out["model_breadth"]
    print(f"  H: survival {mb['Hz']['b']:+.2f}/SD (se {mb['Hz']['se']:.2f}); H x lead {mb['Hz_x_lead']['b']:+.2f} "
          f"(se {mb['Hz_x_lead']['se']:.2f}); lead {mb['lead']['b']:+.2f}", flush=True)

    # 2. pairs ---------------------------------------------------------------
    f = f.with_columns((pl.col("surv") - pl.col("surv").mean().over("cell")).alias("excess"))
    pairs = f.filter(pl.col("is_pair")).group_by("pa", "pb").agg(
        pl.len().alias("n"), pl.col("excess").mean().alias("excess"), pl.col("surv").mean().alias("surv"),
        pl.col("has_attorney").mean().alias("counsel_share")).filter(pl.col("n") >= MIN_PAIR).sort("excess")
    single = f.filter(~pl.col("is_pair")).group_by("top1").agg(
        pl.len().alias("n"), pl.col("excess").mean().alias("excess")).filter(pl.col("n") >= MIN_PAIR)
    out["single_vs_pair_excess"] = {"single": float(f.filter(~pl.col("is_pair"))["excess"].mean()),
                                    "pair": float(f.filter(pl.col("is_pair"))["excess"].mean())}
    rows = []
    for r in pairs.iter_rows(named=True):
        rows.append({"a": lab[r["pa"]], "b": lab[r["pb"]], "pa": r["pa"], "pb": r["pb"], "n": r["n"],
                     "excess_pp": r["excess"], "surv": r["surv"], "counsel_share": r["counsel_share"]})
    out["pairs"] = rows
    print(f"  {len(rows)} pairs with >= {MIN_PAIR:,} registrations; single-theme excess "
          f"{out['single_vs_pair_excess']['single']:+.2f}, pair {out['single_vs_pair_excess']['pair']:+.2f}", flush=True)
    # software partners: themes 5 (online software) and 9 (computer services)
    sw = []
    for s_ in (5, 9):
        for r in rows:
            if s_ in (r["pa"], r["pb"]):
                other = r["pb"] if r["pa"] == s_ else r["pa"]
                sw.append({"software": lab[s_], "partner": lab[other], "n": r["n"], "excess_pp": r["excess_pp"]})
    out["software_partners"] = sorted(sw, key=lambda r: r["excess_pp"])

    # 3. pair surprise (PMI) --------------------------------------------------
    # reference counts from the frame itself by class and filing year (registered filings)
    ref = f.select("cls", "fy", "pa", "pb", "is_pair")
    py = ref.filter(pl.col("is_pair")).group_by("cls", "fy", "pa", "pb").len().rename({"len": "nab"})
    ty = pl.concat([ref.filter(pl.col("is_pair")).select("cls", "fy", pl.col("pa").alias("t")),
                    ref.filter(pl.col("is_pair")).select("cls", "fy", pl.col("pb").alias("t"))]).group_by(
        "cls", "fy", "t").len().rename({"len": "nt"})
    tot = ref.filter(pl.col("is_pair")).group_by("cls", "fy").len().rename({"len": "N"})
    # five prior filing years
    def window(df, keys, val):
        parts = [df.with_columns((pl.col("fy") + k).alias("fy")) for k in range(1, 6)]
        return pl.concat(parts).group_by(keys).agg(pl.col(val).sum())
    pyw = window(py, ["cls", "fy", "pa", "pb"], "nab")
    tyw = window(ty, ["cls", "fy", "t"], "nt")
    totw = window(tot, ["cls", "fy"], "N")
    g = f.filter(pl.col("is_pair")).select("serial_number", "cls", "fy", "pa", "pb").join(
        pyw, on=["cls", "fy", "pa", "pb"], how="left").join(
        tyw.rename({"t": "pa", "nt": "na"}), on=["cls", "fy", "pa"], how="left").join(
        tyw.rename({"t": "pb", "nt": "nb"}), on=["cls", "fy", "pb"], how="left").join(
        totw, on=["cls", "fy"], how="left").filter(pl.col("N") >= 500).with_columns(
        (((pl.col("nab").fill_null(0) + 0.5) / pl.col("N"))
         / (((pl.col("na").fill_null(0) + 0.5) / (2 * pl.col("N"))) * ((pl.col("nb").fill_null(0) + 0.5)
                                                                        / (2 * pl.col("N"))))).log().alias("pmi"))
    fp = f.join(g.select("serial_number", "cls", "pmi"), on=["serial_number", "cls"], how="inner").with_columns(
        ((pl.col("pmi") - pl.col("pmi").mean()) / pl.col("pmi").std()).alias("pmiz")).with_columns(
        (pl.col("pmiz") * pl.col("lead")).alias("pmiz_x_lead"))
    out["pmi_corr_within_class"] = {"lead_z": float(np.corrcoef(fp["pmiz"].to_numpy(), fp["z"].to_numpy())[0, 1]),
                                    "H": float(np.corrcoef(fp["pmiz"].to_numpy(), fp["H"].to_numpy())[0, 1])}
    out["model_pmi"] = fit(fp, ["lead", "pmiz", "pmiz_x_lead", "Hz"] + CONTROLS)
    mp = out["model_pmi"]
    out["n_pmi"] = fp.height
    print(f"  PMI (n={fp.height:,}): survival {mp['pmiz']['b']:+.2f}/SD (se {mp['pmiz']['se']:.2f}); "
          f"PMI x lead {mp['pmiz_x_lead']['b']:+.2f} (se {mp['pmiz_x_lead']['se']:.2f})", flush=True)
    dq = fp.with_columns(((pl.col("pmi").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("pq"),
                         (pl.col("surv") - pl.col("surv").mean().over("cell") + pl.col("surv").mean()).alias("adj"))
    out["surv_by_pmi_quintile"] = [float(v) for v in dq.group_by("pq").agg(pl.col("adj").mean()).sort("pq")["adj"]]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "combo_analysis.json").write_text(json.dumps(out, indent=1))
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
