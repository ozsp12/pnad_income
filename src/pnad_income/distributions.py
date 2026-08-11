"""Distributional statistics and empirical CCDF construction."""

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

    lower = float(positive.min() if xmin is None else xmin)
    upper = float(arr.max() if xmax is None else xmax)
    if lower <= 0 or upper < lower:
        raise ValueError("Require 0 < xmin <= xmax.")
    if upper == lower:
        return np.array([lower, np.nextafter(upper, np.inf)])

    n_edges = int(np.ceil(np.log(upper / lower) / np.log(base))) + 1
    edges = np.geomspace(lower, upper, num=max(n_edges, 2))
    if edges[-1] < upper:
        edges = np.append(edges, upper)
    return np.unique(edges)


def compute_ccdf(
    values,
    base: float = 1.05,
    xmin: float | None = None,
    xmax: float | None = None,
    scale: str = "percent",
) -> pd.DataFrame:
    """Compute the empirical CCDF using the full finite population denominator.

    Geometric thresholds are positive, but finite zero-income observations remain
    in the denominator. This estimates P(X >= x), not P(X >= x | X > 0).
    """
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

    # Bin statistics use only the positive geometric support; this does not alter
    # the population used above to normalize the CCDF.
    within = arr[(arr >= edges[0]) & (arr <= edges[-1])]
    index = np.digitize(within, edges, right=False) - 1
    index = np.clip(index, 0, len(left) - 1)
    count = np.bincount(index, minlength=len(left)).astype(int)
    sum_values = np.bincount(index, weights=within, minlength=len(left))
    mean_arith = np.divide(sum_values, count, out=np.full(len(left), np.nan), where=count > 0)
    sum_log = np.bincount(index, weights=np.log(within), minlength=len(left))
    mean_geom = np.exp(np.divide(sum_log, count, out=np.full(len(left), np.nan), where=count > 0))

    median = np.full(len(left), np.nan)
    std = np.full(len(left), np.nan)
    for i in range(len(left)):
        values_i = within[index == i]
        if values_i.size:
            median[i] = np.median(values_i)
            std[i] = values_i.std(ddof=1) if values_i.size > 1 else 0.0

    return pd.DataFrame({
        "bin": left,
        "right_bin": right,
        "geom_center": np.sqrt(left * right),
        "count": count,
        "ccdf": ccdf,
        "mean_arith": mean_arith,
        "mean_geom": mean_geom,
        "median": median,
        "std": std,
        "population_n": total,
    })


def build_ccdf_by_year(
    df: pd.DataFrame,
    value_col: str = "income",
    year_col: str = "year",
    base: float = 1.05,
    scale: str = "percent",
) -> pd.DataFrame:
    """Compute one CCDF table for each survey year."""
    if value_col not in df.columns or year_col not in df.columns:
        raise KeyError(f"Required columns '{year_col}' and '{value_col}' are not both present.")
    frames = []
    for year, group in df.groupby(year_col, sort=True):
        values = pd.to_numeric(group[value_col], errors="coerce")
        if values.notna().any() and (values > 0).any():
            table = compute_ccdf(values, base=base, scale=scale)
            table.insert(0, year_col, int(year))
            table.insert(1, "measure", value_col)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_income_measures_ccdf(
    df: pd.DataFrame,
    measures: tuple[str, ...] = ("income", "income_effective"),
    year_col: str = "year",
    base: float = 1.05,
    scale: str = "percent",
) -> pd.DataFrame:
    """Stack annual CCDFs for several compatible income measures."""
    tables = []
    for measure in measures:
        if measure in df.columns:
            table = build_ccdf_by_year(df, measure, year_col, base, scale)
            if not table.empty:
                tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
