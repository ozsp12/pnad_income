from __future__ import annotations

import numpy as np
import pandas as pd


def geometric_edges(values, base: float = 1.05, xmin: float | None = None, xmax: float | None = None) -> np.ndarray:
    """Build geometric bin edges over the strictly positive support."""
    if base <= 1:
        raise ValueError("base must be greater than 1.")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    positive = arr[arr > 0]
    if positive.size == 0:
        raise ValueError("At least one strictly positive observation is required.")
    lo = float(positive.min() if xmin is None else xmin)
    hi = float(arr.max() if xmax is None else xmax)
    if lo <= 0 or hi < lo:
        raise ValueError("Require 0 < xmin <= xmax.")
    if hi == lo:
        return np.array([lo, np.nextafter(hi, np.inf)])
    n = int(np.ceil(np.log(hi / lo) / np.log(base))) + 1
    edges = np.geomspace(lo, hi, num=max(n, 2))
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    return np.unique(edges)


def compute_ccdf(values, base: float = 1.05, xmin: float | None = None, xmax: float | None = None, scale: str = "percent") -> pd.DataFrame:
    """Compute a CCDF on positive geometric thresholds with the full population denominator."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite observations were supplied.")
    edges = geometric_edges(arr, base=base, xmin=xmin, xmax=xmax)
    left, right = edges[:-1], edges[1:]
    total = arr.size
    ccdf = np.array([(arr >= threshold).sum() / total for threshold in left], dtype=float)
    if scale == "percent":
        ccdf *= 100.0
    elif scale != "probability":
        raise ValueError("scale must be 'percent' or 'probability'.")
    within = arr[(arr >= edges[0]) & (arr <= edges[-1])]
    idx = np.digitize(within, edges, right=False) - 1
    idx = np.clip(idx, 0, len(left) - 1)
    count = np.bincount(idx, minlength=len(left)).astype(int)
    sum_ = np.bincount(idx, weights=within, minlength=len(left))
    mean = np.divide(sum_, count, out=np.full(len(left), np.nan), where=count > 0)
    sum_log = np.bincount(idx, weights=np.log(within), minlength=len(left))
    mean_geom = np.exp(np.divide(sum_log, count, out=np.full(len(left), np.nan), where=count > 0))
    median = np.full(len(left), np.nan)
    std = np.full(len(left), np.nan)
    for i in range(len(left)):
        vals = within[idx == i]
        if vals.size:
            median[i] = np.median(vals)
            std[i] = vals.std(ddof=1) if vals.size > 1 else 0.0
    return pd.DataFrame({"bin": left, "right_bin": right, "geom_center": np.sqrt(left * right), "count": count, "ccdf": ccdf, "mean_arith": mean, "mean_geom": mean_geom, "median": median, "std": std, "population_n": total})


def build_ccdf_by_year(df: pd.DataFrame, value_col: str = "income", year_col: str = "year", base: float = 1.05, scale: str = "percent") -> pd.DataFrame:
    """Compute annual CCDF tables for one income measure."""
    frames = []
    for year, group in df.groupby(year_col, sort=True):
        values = pd.to_numeric(group[value_col], errors="coerce")
        if values.notna().any() and (values > 0).any():
            table = compute_ccdf(values, base=base, scale=scale)
            table.insert(0, year_col, int(year))
            table.insert(1, "measure", value_col)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_income_measures_ccdf(df: pd.DataFrame, measures: tuple[str, ...] = ("income", "income_effective"), year_col: str = "year", base: float = 1.05, scale: str = "percent") -> pd.DataFrame:
    """Stack CCDFs for habitual/effective or nominal/adjusted measures."""
    tables = []
    for measure in measures:
        if measure in df.columns:
            table = build_ccdf_by_year(df, value_col=measure, year_col=year_col, base=base, scale=scale)
            if not table.empty:
                tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
