"""Distribution, inequality, and validation routines for PNAD income analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_REFERENCE_COLUMNS = {"year", "gini", "source"}


def _values(values, *, sort: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    return np.sort(x) if sort else x


def gini(values) -> float:
    """Return the unweighted Gini coefficient for nonnegative observations."""
    x = _values(values, sort=True)
    if x.size == 0:
        return np.nan
    total = x.sum()
    if total == 0:
        return 0.0
    n = x.size
    ranks = np.arange(1, n + 1)
    return float(2 * np.sum(ranks * x) / (n * total) - (n + 1) / n)


def lorenz_curve(values) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative population and income shares on [0, 1]."""
    x = _values(values, sort=True)
    if x.size == 0:
        return np.array([]), np.array([])
    population = np.arange(x.size + 1) / x.size
    total = x.sum()
    if total == 0:
        return population, population.copy()
    income = np.concatenate(([0.0], np.cumsum(x) / total))
    return population, income


def top_income_share(values, fraction: float) -> float:
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1].")
    population, income = lorenz_curve(values)
    if population.size == 0:
        return np.nan
    return float(1 - np.interp(1 - fraction, population, income))


def pietra_index(values) -> float:
    population, income = lorenz_curve(values)
    return np.nan if population.size == 0 else float(np.max(population - income))


def kolkata_index(values) -> float:
    """Return k satisfying L(k) = 1-k by linear interpolation."""
    population, income = lorenz_curve(values)
    if population.size == 0:
        return np.nan
    f = population + income - 1
    exact = np.flatnonzero(np.isclose(f, 0.0, atol=1e-14))
    if exact.size:
        return float(population[exact[0]])
    crossing = np.flatnonzero(f[:-1] * f[1:] < 0)
    if crossing.size == 0:
        return np.nan
    i = int(crossing[0])
    return float(population[i] - f[i] * (population[i + 1] - population[i]) / (f[i + 1] - f[i]))


lorenz_k_index = kolkata_index


def _insert_lorenz_point(
    population: np.ndarray,
    income: np.ndarray,
    x: float,
    y: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(np.isclose(population, x, atol=1e-14)):
        return population, income
    position = int(np.searchsorted(population, x))
    y = float(np.interp(x, population, income)) if y is None else float(y)
    return np.insert(population, position, x), np.insert(income, position, y)


def zanardi_components(values) -> dict[str, float]:
    """Return the geometric Zanardi asymmetry index and its components."""
    x = _values(values, sort=True)
    if x.size == 0:
        return {key: np.nan for key in ("kolkata", "qd", "gini_poor", "gini_rich", "delta_gini", "zanardi")}
    if np.all(x == 0):
        return {
            "kolkata": 0.5,
            "qd": 0.5,
            "gini_poor": 0.0,
            "gini_rich": 0.0,
            "delta_gini": 0.0,
            "zanardi": 0.0,
        }

    G = gini(x)
    p, L = lorenz_curve(x)
    k = kolkata_index(x)
    q = 1 - k
    p, L = _insert_lorenz_point(p, L, k, q)
    split = int(np.argmin(np.abs(p - k)))

    p_left, L_left = p[: split + 1], L[: split + 1]
    p_right, L_right = p[split:], L[split:]
    chord_left = (q / k) * p_left
    chord_right = q + (k / q) * (p_right - k)
    area_scale = k * q / 2

    if area_scale <= 0:
        poor = rich = 0.0
    else:
        poor = float(np.trapezoid(chord_left - L_left, p_left) / area_scale)
        rich = float(np.trapezoid(chord_right - L_right, p_right) / area_scale)
    delta = rich - poor
    z = 0.0 if np.isclose(G, 0.0) else float(2 * area_scale * delta / G)
    return {
        "kolkata": float(k),
        "qd": float(q),
        "gini_poor": poor,
        "gini_rich": rich,
        "delta_gini": delta,
        "zanardi": z,
    }


def zanardi_index(values) -> float:
    return float(zanardi_components(values)["zanardi"])


def legacy_z_statistic(values) -> float:
    """Reproduce the historical ``pnad.py`` Z statistic under an explicit name."""
    population, income = lorenz_curve(values)
    G = gini(values)
    if population.size == 0 or not np.isfinite(G):
        return np.nan
    if np.isclose(G, 0.0):
        return 0.0
    k = kolkata_index(values)
    if not np.isfinite(k) or k <= 0 or k >= 1:
        return np.nan
    population, income = _insert_lorenz_point(population, income, k)
    split = int(np.flatnonzero(np.isclose(population, k, atol=1e-14))[0])
    x1, y1 = population[: split + 1], income[: split + 1]
    x2, y2 = population[split:], income[split:]
    eq1 = 0.5 * k**2
    eq2 = 0.5 - eq1
    G1 = (eq1 - float(np.trapezoid(y1, x1))) / eq1
    G2 = (eq2 - float(np.trapezoid(y2, x2))) / eq2
    return float(k * (1 - k) * (G2 - G1) / G)


def herfindahl_index(values) -> float:
    x = _values(values)
    total = x.sum()
    if x.size == 0 or total <= 0:
        return np.nan
    shares = x / total
    return float(np.sum(shares**2))


def normalized_herfindahl(values) -> float:
    x = _values(values)
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    H = herfindahl_index(x)
    return float((x.size * H - 1) / (x.size - 1)) if np.isfinite(H) else np.nan


def shannon_entropy(values) -> float:
    x = _values(values)
    total = x.sum()
    if x.size == 0 or total <= 0:
        return np.nan
    shares = x / total
    positive = shares[shares > 0]
    return float(-np.sum(positive * np.log(positive)))


def normalized_shannon_inequality(values) -> float:
    x = _values(values)
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    entropy = shannon_entropy(x)
    return float(1 - entropy / np.log(x.size)) if np.isfinite(entropy) else np.nan


def theil_index(values) -> float:
    x = _values(values)
    if x.size == 0:
        return np.nan
    mean = x.mean()
    if mean <= 0:
        return 0.0
    ratio = x / mean
    positive = ratio > 0
    return float(np.sum(ratio[positive] * np.log(ratio[positive])) / x.size)


def atkinson_index(values, epsilon: float = 0.5) -> float:
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    x = _values(values)
    if x.size == 0:
        return np.nan
    mean = x.mean()
    if mean <= 0:
        return 0.0
    if np.isclose(epsilon, 1.0):
        if np.any(x == 0):
            return 1.0
        ede = float(np.exp(np.mean(np.log(x))))
    else:
        power = 1 - epsilon
        if power < 0 and np.any(x == 0):
            return 1.0
        ede = float(np.mean(x**power) ** (1 / power))
    return float(1 - ede / mean)


def inequality_statistics(values, atkinson_epsilon: float = 0.5) -> dict[str, float]:
    """Return the complete inequality block used by the project."""
    z = zanardi_components(values)
    G = gini(values)
    P = pietra_index(values)
    K = z["kolkata"]
    excess = 2 * K - 1 if np.isfinite(K) else np.nan
    return {
        "gini": G,
        "pietra": P,
        "k": K,
        "zanardi": z["zanardi"],
        "legacy_z": legacy_z_statistic(values),
        "top_10_share": top_income_share(values, 0.10),
        "top_1_share": top_income_share(values, 0.01),
        "top_0_1_share": top_income_share(values, 0.001),
        "herfindahl": herfindahl_index(values),
        "herfindahl_normalized": normalized_herfindahl(values),
        "shannon_entropy": shannon_entropy(values),
        "shannon_inequality": normalized_shannon_inequality(values),
        "theil": theil_index(values),
        f"atkinson_{atkinson_epsilon:g}": atkinson_index(values, atkinson_epsilon),
        "gini_poor": z["gini_poor"],
        "gini_rich": z["gini_rich"],
        "delta_gini": z["delta_gini"],
        "kolkata_excess": excess,
        "pietra_over_kolkata_excess": P / excess if np.isfinite(excess) and excess > 0 else np.nan,
        "pietra_over_gini": P / G if np.isfinite(G) and G > 0 else np.nan,
        "kolkata_small_g_prediction": 0.5 + 0.375 * G if np.isfinite(G) else np.nan,
        "pietra_small_g_prediction": 0.75 * G if np.isfinite(G) else np.nan,
    }


extended_inequality_statistics = inequality_statistics


def summary_statistics(
    df: pd.DataFrame,
    value_columns: tuple[str, ...] = ("income", "income_adj", "income_effective", "income_effective_adj"),
    year_col: str = "year",
    atkinson_epsilon: float = 0.5,
) -> pd.DataFrame:
    """Compute annual descriptive and inequality statistics."""
    if year_col not in df.columns:
        raise KeyError(f"Column '{year_col}' is absent from the data.")
    rows = []
    for year, group in df.groupby(year_col, sort=True):
        row: dict[str, object] = {year_col: int(year), "total": len(group)}
        for column in value_columns:
            if column not in group.columns:
                continue
            series = pd.to_numeric(group[column], errors="coerce")
            valid = series[np.isfinite(series) & (series >= 0)]
            positive = valid[valid > 0]
            metrics = inequality_statistics(valid, atkinson_epsilon)
            row.update(
                {
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
                }
            )
            row.update({f"{column}_{name}": value for name, value in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(year_col).reset_index(drop=True)


annual_statistics = summary_statistics


def annual_inequality_indices(
    df: pd.DataFrame,
    value_col: str = "income",
    year_col: str = "year",
    atkinson_epsilon: float = 0.5,
) -> pd.DataFrame:
    """Return one compact inequality row per survey year."""
    if {value_col, year_col}.difference(df.columns):
        raise KeyError(f"Required columns '{year_col}' and '{value_col}' are not both present.")
    rows = []
    for year, group in df.groupby(year_col, sort=True):
        x = _values(pd.to_numeric(group[value_col], errors="coerce"))
        if x.size:
            rows.append({"year": int(year), "n": int(x.size), **inequality_statistics(x, atkinson_epsilon)})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def geometric_thresholds(
    values,
    base: float = 1.05,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Return exact geometric CCDF thresholds ``xmin * base**k`` up to ``xmax``."""
    if base <= 1:
        raise ValueError("base must be greater than 1.")
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    positive = x[x > 0]
    if positive.size == 0:
        raise ValueError("At least one strictly positive observation is required.")
    lower = float(positive.min() if xmin is None else xmin)
    upper = float(x.max() if xmax is None else xmax)
    if lower <= 0 or upper < lower:
        raise ValueError("Require 0 < xmin <= xmax.")
    if np.isclose(upper, lower):
        return np.array([lower], dtype=float)
    n = int(np.floor(np.log(upper / lower) / np.log(base))) + 1
    thresholds = lower * np.power(base, np.arange(max(n, 1), dtype=float))
    tolerance = np.finfo(float).eps * max(1.0, abs(upper)) * 16
    return thresholds[thresholds <= upper + tolerance]


def geometric_edges(
    values,
    base: float = 1.05,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build interval edges whose left edges are the exact geometric CCDF thresholds."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    thresholds = geometric_thresholds(x, base=base, xmin=xmin, xmax=xmax)
    upper = float(x.max() if xmax is None else xmax)
    if thresholds[-1] < upper:
        final_edge = np.nextafter(upper, np.inf)
    else:
        final_edge = np.nextafter(thresholds[-1], np.inf)
    return np.append(thresholds, final_edge)


def compute_ccdf(
    values,
    base: float = 1.05,
    xmin: float | None = None,
    xmax: float | None = None,
    scale: str = "probability",
) -> pd.DataFrame:
    """Compute the empirical CCDF on an exact geometric threshold grid."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite observations were supplied.")
    edges = geometric_edges(x, base=base, xmin=xmin, xmax=xmax)
    left, right = edges[:-1], edges[1:]
    total = x.size

    sorted_x = np.sort(x)
    ccdf = (total - np.searchsorted(sorted_x, left, side="left")) / total
    if scale == "percent":
        ccdf *= 100
    elif scale != "probability":
        raise ValueError("scale must be 'percent' or 'probability'.")

    within = x[(x >= edges[0]) & (x <= edges[-1])]
    index = np.clip(np.digitize(within, edges, right=False) - 1, 0, len(left) - 1)
    count = np.bincount(index, minlength=len(left)).astype(int)
    sums = np.bincount(index, weights=within, minlength=len(left))
    mean = np.divide(sums, count, out=np.full(len(left), np.nan), where=count > 0)
    log_sums = np.bincount(index, weights=np.log(within), minlength=len(left))
    geom = np.exp(np.divide(log_sums, count, out=np.full(len(left), np.nan), where=count > 0))
    median = np.full(len(left), np.nan)
    std = np.full(len(left), np.nan)
    for i in range(len(left)):
        bin_values = within[index == i]
        if bin_values.size:
            median[i] = np.median(bin_values)
            std[i] = bin_values.std(ddof=1) if bin_values.size > 1 else 0.0

    return pd.DataFrame(
        {
            "bin": left,
            "right_bin": right,
            "geom_center": np.sqrt(left * right),
            "count": count,
            "ccdf": ccdf,
            "mean_arith": mean,
            "mean_geom": geom,
            "median": median,
            "std": std,
            "population_n": total,
        }
    )


def build_ccdf_by_year(
    df: pd.DataFrame,
    value_col: str = "income",
    year_col: str = "year",
    base: float = 1.05,
    scale: str = "probability",
) -> pd.DataFrame:
    """Compute one CCDF table for each survey year."""
    if {value_col, year_col}.difference(df.columns):
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
    scale: str = "probability",
) -> pd.DataFrame:
    """Stack annual CCDFs for all requested available income measures."""
    tables = [
        build_ccdf_by_year(df, measure, year_col, base, scale)
        for measure in measures
        if measure in df.columns
    ]
    tables = [table for table in tables if not table.empty]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def load_gini_reference(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_REFERENCE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("Gini reference is missing: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["gini"] = pd.to_numeric(frame["gini"], errors="coerce")
    frame.loc[frame["gini"] > 1, "gini"] /= 100
    if (frame["gini"].notna() & ~frame["gini"].between(0, 1)).any():
        raise ValueError("Gini reference values must lie on [0, 1] or [0, 100].")
    return frame.sort_values(["source", "year"]).reset_index(drop=True)


def combine_gini_references(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [load_gini_reference(path) for path in paths]
    if not frames:
        return pd.DataFrame(columns=["year", "gini", "source"])
    return pd.concat(frames, ignore_index=True).sort_values(["source", "year"]).reset_index(drop=True)


def compare_gini_series(
    summary: pd.DataFrame,
    references: pd.DataFrame,
    *,
    calculated_col: str = "income_gini",
) -> pd.DataFrame:
    missing = {"year", calculated_col}.difference(summary.columns)
    if missing:
        raise KeyError("Summary is missing: " + ", ".join(sorted(missing)))
    missing = REQUIRED_REFERENCE_COLUMNS.difference(references.columns)
    if missing:
        raise KeyError("References are missing: " + ", ".join(sorted(missing)))
    calculated = summary[["year", calculated_col]].rename(columns={calculated_col: "gini_calculated"})
    merged = references.merge(calculated, on="year", how="inner", validate="many_to_one")
    merged["difference"] = merged["gini_calculated"] - merged["gini"]
    merged["absolute_difference"] = merged["difference"].abs()
    return merged.sort_values(["source", "year"]).reset_index(drop=True)


def gini_validation_statistics(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(columns=["source", "n", "mean_difference", "mae", "rmse", "correlation"])
    rows = []
    for source, group in comparison.groupby("source", sort=True):
        calculated = group["gini_calculated"].to_numpy(float)
        reference = group["gini"].to_numpy(float)
        diff = calculated - reference
        finite = np.isfinite(calculated) & np.isfinite(reference)
        calculated, reference, diff = calculated[finite], reference[finite], diff[finite]
        correlation = (
            float(np.corrcoef(calculated, reference)[0, 1])
            if diff.size > 1 and np.std(calculated) > 0 and np.std(reference) > 0
            else np.nan
        )
        rows.append(
            {
                "source": source,
                "n": int(diff.size),
                "mean_difference": float(np.mean(diff)) if diff.size else np.nan,
                "mae": float(np.mean(np.abs(diff))) if diff.size else np.nan,
                "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan,
                "correlation": correlation,
            }
        )
    return pd.DataFrame(rows)
