"""Reusable visualization utilities for PNAD income-distribution analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inequality import lorenz_curve


def plot_gini_evolution(summary: pd.DataFrame, value_col: str = "income"):
    """Plot the annual Gini coefficient for one income measure."""
    column = f"{value_col}_gini"
    if column not in summary.columns:
        raise KeyError(f"Column '{column}' is absent from the summary table.")
    frame = summary[["year", column]].dropna().sort_values("year")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(frame["year"], frame[column], marker="o", markersize=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gini coefficient")
    ax.set_title(f"Annual Gini coefficient: {value_col}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _ccdf_coordinates(frame: pd.DataFrame, transform: str):
    """Return coordinates for linear, log-log, or double-log CCDF display."""
    x = frame["bin"].to_numpy(dtype=float)
    y = frame["ccdf"].to_numpy(dtype=float)
    if transform == "linear":
        return x, y
    if transform == "loglog":
        mask = (x > 0) & (y > 0)
        return x[mask], y[mask]
    if transform == "double_log":
        mask = y > 1.0
        return x[mask], np.log(np.log(y[mask]))
    raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")


def plot_ccdf_selected_years(
    ccdf: pd.DataFrame,
    measure: str,
    years,
    transform: str = "loglog",
):
    """Overlay selected annual CCDFs for one income measure."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for year in years:
        frame = ccdf.loc[(ccdf["year"] == int(year)) & (ccdf["measure"] == measure)]
        if frame.empty:
            continue
        x, y = _ccdf_coordinates(frame, transform)
        if transform == "loglog":
            ax.loglog(x, y, label=str(int(year)))
        else:
            ax.plot(x, y, label=str(int(year)))
    ax.set_xlabel("Income")
    ax.set_ylabel("ln[ln(CCDF)]" if transform == "double_log" else "CCDF")
    ax.set_title(f"{measure}: selected survey years")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Year")
    fig.tight_layout()
    return fig


def plot_measure_comparison(
    ccdf: pd.DataFrame,
    year: int,
    measures: tuple[str, ...],
    transform: str = "loglog",
):
    """Compare multiple income measures within one survey year."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for measure in measures:
        frame = ccdf.loc[(ccdf["year"] == int(year)) & (ccdf["measure"] == measure)]
        if frame.empty:
            continue
        x, y = _ccdf_coordinates(frame, transform)
        if transform == "loglog":
            ax.loglog(x, y, label=measure)
        else:
            ax.plot(x, y, label=measure)
    ax.set_title(str(int(year)))
    ax.set_xlabel("Income")
    ax.set_ylabel("ln[ln(CCDF)]" if transform == "double_log" else "CCDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_lorenz_curve(df: pd.DataFrame, year: int, value_col: str = "income"):
    """Plot the Lorenz curve for one survey year."""
    values = pd.to_numeric(df.loc[df["year"] == int(year), value_col], errors="coerce")
    population, income_share = lorenz_curve(values)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(population, income_share, label=str(int(year)))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Equality")
    ax.set_xlabel("Cumulative population share")
    ax.set_ylabel("Cumulative income share")
    ax.set_title(f"Lorenz curve: {year}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
