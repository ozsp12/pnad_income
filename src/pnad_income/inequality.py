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
    """Return cumulative population and cumulative income shares on [0, 1]."""
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


def top_income_share(values, fraction: float) -> float:
    """Return the share of aggregate income received by the richest fraction.

    The empirical Lorenz curve is linearly interpolated at ``1 - fraction`` so
    the estimator is well-defined even when the requested population boundary
    falls between observations.
    """
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1].")
    population, income_share = lorenz_curve(values)
    if population.size == 0:
        return np.nan
    if np.isclose(income_share[-1], 0.0):
        return 0.0
    bottom_share = np.interp(1.0 - fraction, population, income_share)
    return float(1.0 - bottom_share)


def top_income_shares(
    values,
    fractions: tuple[float, ...] = (0.10, 0.01, 0.001),
) -> dict[float, float]:
    """Return top-income shares for one or more population fractions."""
    return {float(fraction): top_income_share(values, fraction) for fraction in fractions}


def pietra_index(values) -> float:
    """Compute the Pietra (Hoover/Robin Hood) inequality index.

    With Lorenz coordinates normalized to [0, 1], the index is the maximum
    vertical distance between the equality line and the Lorenz curve.
    """
    population, income_share = lorenz_curve(values)
    if population.size == 0:
        return np.nan
    return float(np.max(population - income_share))


def lorenz_k_index(values) -> float:
    """Compute the Lorenz intersection k satisfying L(k) = 1 - k.

    Piecewise-linear interpolation is used directly on the empirical Lorenz
    curve; no SciPy dependency is required.  The returned value is on [0, 1].
    """
    population, income_share = lorenz_curve(values)
    if population.size == 0:
        return np.nan
    difference = income_share + population - 1.0
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-14))
    if exact.size:
        return float(population[exact[0]])
    crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    if crossings.size == 0:
        return np.nan
    i = int(crossings[0])
    x0, x1 = population[i], population[i + 1]
    d0, d1 = difference[i], difference[i + 1]
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


def _lorenz_with_point(
    population: np.ndarray,
    income_share: np.ndarray,
    x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert an interpolated population coordinate into a Lorenz polyline."""
    if np.any(np.isclose(population, x, atol=1e-14)):
        return population, income_share
    y = float(np.interp(x, population, income_share))
    position = int(np.searchsorted(population, x))
    return (
        np.insert(population, position, x),
        np.insert(income_share, position, y),
    )


def zanardi_index(values) -> float:
    """Compute the legacy Z statistic used in the historical PNAD workflow.

    This reproduces the geometry of the original ``pnad.py`` construction in a
    normalized [0, 1] coordinate system.  The Lorenz curve is partitioned at
    the k intersection.  Partial Gini-like quantities are calculated on the two
    subregions and combined as

        Z = k (1-k) (G2-G1) / G.

    The label ``Zanardi`` is retained for reproducibility of the legacy analysis;
    interpretation and external provenance should be documented separately in
    the scientific manuscript.
    """
    population, income_share = lorenz_curve(values)
    G = gini(values)
    if population.size == 0 or not np.isfinite(G) or np.isclose(G, 0.0):
        return 0.0 if np.isclose(G, 0.0, equal_nan=False) else np.nan
    k = lorenz_k_index(values)
    if not np.isfinite(k) or k <= 0 or k >= 1:
        return np.nan

    population, income_share = _lorenz_with_point(population, income_share, k)
    split = int(np.flatnonzero(np.isclose(population, k, atol=1e-14))[0])

    x1 = population[: split + 1]
    y1 = income_share[: split + 1]
    x2 = population[split:]
    y2 = income_share[split:]

    area_lorenz_1 = float(np.trapezoid(y1, x1))
    area_lorenz_2 = float(np.trapezoid(y2, x2))
    area_equality_1 = 0.5 * k**2
    area_equality_2 = 0.5 - area_equality_1

    G1 = (area_equality_1 - area_lorenz_1) / area_equality_1 if area_equality_1 > 0 else np.nan
    G2 = (area_equality_2 - area_lorenz_2) / area_equality_2 if area_equality_2 > 0 else np.nan
    return float(k * (1.0 - k) * (G2 - G1) / G)


def extended_inequality_statistics(values) -> dict[str, float]:
    """Return the complete inequality block used by the PNAD analysis."""
    shares = top_income_shares(values)
    return {
        "gini": gini(values),
        "pietra": pietra_index(values),
        "k": lorenz_k_index(values),
        "zanardi": zanardi_index(values),
        "top_10_share": shares[0.10],
        "top_1_share": shares[0.01],
        "top_0_1_share": shares[0.001],
    }


def summary_statistics(
    df: pd.DataFrame,
    value_columns: tuple[str, ...] = ("income", "income_adj", "income_effective", "income_effective_adj"),
    year_col: str = "year",
) -> pd.DataFrame:
    """Compute annual support, descriptive, and extended inequality statistics."""
    if year_col not in df.columns:
        raise KeyError(f"Column '{year_col}' is absent from the data.")
    rows = []
    for year, group in df.groupby(year_col, sort=True):
        row = {year_col: int(year), "total": len(group)}
        for column in value_columns:
            if column not in group.columns:
                continue
            series = pd.to_numeric(group[column], errors="coerce")
            finite = series[np.isfinite(series)]
            valid = finite[finite >= 0]
            positive = valid[valid > 0]
            metrics = extended_inequality_statistics(valid)
            row.update({
                f"{column}_n": int(valid.size),
                f"{column}_n_missing": int(series.isna().sum()),
                f"{column}_n_zero": int((valid == 0).sum()),
                f"{column}_n_positive": int((valid > 0).sum()),
                f"{column}_zero_fraction": float((valid == 0).mean()) if valid.size else np.nan,
                f"{column}_missing_fraction": float(series.isna().mean()) if len(series) else np.nan,
                f"{column}_sum": valid.sum() if valid.size else np.nan,
                f"{column}_xmin": positive.min() if not positive.empty else np.nan,
                f"{column}_xmax": valid.max() if not valid.empty else np.nan,
                f"{column}_mean": valid.mean(),
                f"{column}_median": valid.median(),
                f"{column}_std": valid.std(),
                f"{column}_gini": metrics["gini"],
                f"{column}_pietra": metrics["pietra"],
                f"{column}_k": metrics["k"],
                f"{column}_zanardi": metrics["zanardi"],
                f"{column}_top_10_share": metrics["top_10_share"],
                f"{column}_top_1_share": metrics["top_1_share"],
                f"{column}_top_0_1_share": metrics["top_0_1_share"],
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(year_col).reset_index(drop=True)
