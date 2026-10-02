"""Shared helpers for the teece_* tests (Paper B: who profits when a new offering appears).

Conventions (as in combined_model.py):
  surv  = 100 * (1 - failed1), survival of the first maintenance deadline in points
  lead  = rank(z) within class x registration-year cell / n - 0.5 (most lagging -0.5 .. most leading +0.5)
  controls = has_attorney, itu, log_len, log_owner_n, dom_us, dom_cn, basis_44e, basis_66a
  class x registration-year fixed effects; SEs clustered by owner_key.
"""
from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
sys.path.insert(0, str(REPO / "scripts"))
from combined_model import Design, peak_gb  # noqa: E402,F401

CLASSES = [f"{i:03d}" for i in range(1, 46)]
CONTROLS = ["has_attorney", "itu", "log_len", "log_owner_n", "dom_us", "dom_cn", "basis_44e", "basis_66a"]


def log(m):
    print(m, file=sys.stderr, flush=True)


def free_gb() -> float:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return 99.0


def mem_wait(label: str = "", need: float = 4.0, max_wait: int = 3600):
    """Block until at least `need` GB of physical memory is free (the owner's
    simulations share the machine)."""
    t0 = time.time()
    while True:
        f = free_gb()
        if f >= need:
            log(f"[mem] {label} free {f:.1f} GB, own peak {peak_gb():.2f} GB")
            return
        if time.time() - t0 > max_wait:
            raise SystemExit(f"[mem] gave up waiting for memory at {label} (free {f:.1f} GB)")
        log(f"[mem] {label} only {f:.1f} GB free; waiting")
        time.sleep(60)


def load_frame(extra: list[str] | None = None) -> pl.DataFrame:
    extra = [c for c in (extra or []) if c not in CONTROLS]
    cols = list(dict.fromkeys(["serial_number", "cls", "cell", "reg_year", "fy", "owner_key", "failed1", "z", "q"]
                              + CONTROLS + extra))
    d = pl.read_parquet(PROC / "battery_frame.parquet", columns=cols)
    d = d.with_columns(
        (100 * (1 - pl.col("failed1").cast(pl.Float64))).alias("surv"),
        ((pl.col("z").rank("average").over("cell") - 0.5) / pl.len().over("cell") - 0.5).alias("lead"),
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in CONTROLS])
    return d


def codes(s: pl.Series) -> np.ndarray:
    a = np.asarray(s.cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy(), dtype=np.int64)
    _, inv = np.unique(a, return_inverse=True)
    return inv


def cell_ols(d: pl.DataFrame, y: str, xs: list[str], fe: str = "cell") -> dict:
    """Within-FE OLS, owner-clustered; returns {name: (b, se)} plus n, owners."""
    des = Design(d, fe=fe)
    X = np.column_stack([d[c].cast(pl.Float64).to_numpy() for c in xs]).astype(np.float64)
    r = des.ols(d[y].to_numpy().astype(np.float64), X, xs)
    out = {k: {"b": v[0], "se": v[1]} for k, v in r.items() if not k.startswith("_")}
    out["_n"] = des.n
    out["_owners"] = des.G
    del des, X
    gc.collect()
    return out


def twoway_ols(y: np.ndarray, X: np.ndarray, names: list[str], fe1: np.ndarray, fe2: np.ndarray,
               cluster: np.ndarray, tol: float = 1e-4, max_iter: int = 500) -> dict:
    """OLS absorbing two sets of fixed effects by alternating projections
    (method of alternating projections / FWL). Iterates until every coefficient
    changes by less than `tol` between checks. Clustered covariance with the
    usual G/(G-1) (n-1)/(n-k) factor; FE nested in clusters do not count in k."""
    V = np.column_stack([y, X]).astype(np.float64)
    n1, n2 = int(fe1.max()) + 1, int(fe2.max()) + 1
    c1 = np.maximum(np.bincount(fe1, minlength=n1), 1).astype(np.float64)
    c2 = np.maximum(np.bincount(fe2, minlength=n2), 1).astype(np.float64)

    def sweep(M):
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(fe1, weights=M[:, j], minlength=n1) / c1)[fe1]
            M[:, j] -= (np.bincount(fe2, weights=M[:, j], minlength=n2) / c2)[fe2]

    b_old, it = None, 0
    while it < max_iter:
        for _ in range(5):
            sweep(V); it += 1
        Xt, yt = V[:, 1:], V[:, 0]
        b = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        if b_old is not None and np.max(np.abs(b - b_old)) < tol:
            break
        b_old = b
    Xt, yt = V[:, 1:], V[:, 0]
    e = yt - Xt @ b
    XtX = Xt.T @ Xt
    inv = np.linalg.inv(XtX)
    G = int(cluster.max()) + 1
    S = np.column_stack([np.bincount(cluster, weights=Xt[:, j] * e, minlength=G) for j in range(Xt.shape[1])])
    meat = S.T @ S
    n, k = len(y), Xt.shape[1]
    Vc = (G / (G - 1)) * ((n - 1) / (n - k)) * inv @ meat @ inv
    out = {nm: {"b": float(b[i]), "se": float(math.sqrt(Vc[i, i]))} for i, nm in enumerate(names)}
    out["_n"] = n
    out["_owners"] = G
    out["_iterations"] = it
    del V, Xt, S
    gc.collect()
    return out


def wls(y: np.ndarray, X: np.ndarray, w: np.ndarray, names: list[str]) -> dict:
    """Precision-weighted least squares with HC1 SEs (constant added)."""
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    m = sm.WLS(y, Xc, weights=w).fit(cov_type="HC1")
    out = {"const": {"b": float(m.params[0]), "se": float(m.bse[0])}}
    for i, nm in enumerate(names):
        out[nm] = {"b": float(m.params[i + 1]), "se": float(m.bse[i + 1]), "p": float(m.pvalues[i + 1])}
    out["_n"] = int(len(y))
    out["_r2"] = float(m.rsquared)
    return out


def save(name: str, obj: dict):
    RES.mkdir(parents=True, exist_ok=True)
    p = RES / f"teece_{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"[save] {p}")
