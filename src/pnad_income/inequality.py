from __future__ import annotations

import numpy as np
import pandas as pd


def _nonnegative_finite(values) -> np.ndarray:
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
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x))) - (n + 1) / n)


def lorenz_curve(values) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative population and cumulative income shares."""
    x = _nonnegative_finite(values)
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    p = np.arange(0, x.size + 1) / x.size
    total = x.sum()
    if total == 0:
        return p, p.copy()
    l = np.concatenate(([0.0], np.cumsum(x) / total))
    return p, l


def summary_statistics(df: pd.DataFrame, value_columns: tuple[str, ...] = ("income", "income_adj", "income_effective", "income_effective_adj"), year_col: str = "year") -> pd.DataFrame:
    """Compute yearly location, dispersion, support and Gini statistics."""
    rows = []
    for year, group in df.groupby(year_col, sort=True):
        row = {year_col: int(year), "total": len(group)}
        for col in value_columns:
            if col not in group.columns:
                continue
            s = pd.to_numeric(group[col], errors="coerce")
            valid = s[s.notna()]
            positive = valid[valid > 0]
            row.update({
                f"{col}_n": int(valid.size),
                f"{col}_xmin": positive.min() if not positive.empty else np.nan,
                f"{col}_xmax": valid.max() if not valid.empty else np.nan,
                f"{col}_mean": valid.mean(),
                f"{col}_median": valid.median(),
                f"{col}_std": valid.std(),
                f"{col}_gini": gini(valid),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(year_col).reset_index(drop=True)
