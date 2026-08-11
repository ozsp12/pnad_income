"""Inequality measures and annual descriptive statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nonnegative_finite(values) -> np.ndarray:
    """Return finite nonnegative observations as a NumPy array."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr) & (arr >= 0)]


def gini(values) -> float:
    """Compute the unweighted Gini coefficient for nonnegative observations."""
    x = _nonnegative_finite(values)
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    weighted_rank_sum = np.sum(np.arange(1, n + 1) * x)
    return float((2 * weighted_rank_sum / (n * np.sum(x))) - ((n + 1) / n))


def lorenz_curve(values) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative population and cumulative income shares."""
    x = _nonnegative_finite(values)
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    population = np.arange(0, x.size + 1) / x.size
    total = x.sum()
    if total == 0:
        return population, population.copy()
    income_share = np.concatenate(([0.0], np.cumsum(x) / total))
    return population, income_share


def summary_statistics(
    df: pd.DataFrame,
    value_columns: tuple[str, ...] = ("income", "income_adj", "income_effective", "income_effective_adj"),
    year_col: str = "year",
) -> pd.DataFrame:
    """Compute annual location, dispersion, support, and Gini statistics."""
    if year_col not in df.columns:
        raise KeyError(f"Column '{year_col}' is absent from the data.")
    rows = []
    for year, group in df.groupby(year_col, sort=True):
        row = {year_col: int(year), "total": len(group)}
        for column in value_columns:
            if column not in group.columns:
                continue
            series = pd.to_numeric(group[column], errors="coerce")
            valid = series[series.notna()]
            positive = valid[valid > 0]
            row.update({
                f"{column}_n": int(valid.size),
                f"{column}_xmin": positive.min() if not positive.empty else np.nan,
                f"{column}_xmax": valid.max() if not valid.empty else np.nan,
                f"{column}_mean": valid.mean(),
                f"{column}_median": valid.median(),
                f"{column}_std": valid.std(),
                f"{column}_gini": gini(valid),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(year_col).reset_index(drop=True)
