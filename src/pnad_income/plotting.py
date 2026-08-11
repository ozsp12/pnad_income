from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inequality import gini, lorenz_curve


def _axes_for_years(years, n_cols=4, width=5.0, height=3.4):
    n_rows = max(1, math.ceil(len(years) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False)
    return fig, axes.flatten()


def plot_histograms(df, value_col="income", year_col="year", bins=100, n_cols=4):
    years = sorted(df[year_col].dropna().unique())
    fig, axes = _axes_for_years(years, n_cols=n_cols)
    for ax, year in zip(axes, years):
        values = pd.to_numeric(df.loc[df[year_col] == year, value_col], errors="coerce").to_numpy()
        values = values[np.isfinite(values) & (values > 0)]
        ax.hist(values, bins=bins, log=True)
        ax.set_title(str(int(year)))
        ax.set_xlabel(value_col)
        ax.set_ylabel("Frequency (log)")
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(years):]:
        ax.remove()
    fig.tight_layout()
    return fig


def plot_ccdf_grid(ccdf, transform="linear", year_col="year", n_cols=4, min_ccdf=None):
    """Plot CCDFs as linear, log-log, or the legacy ln[ln(CCDF)] representation."""
    years = sorted(ccdf[year_col].dropna().unique())
    fig, axes = _axes_for_years(years, n_cols=n_cols)
    for ax, year in zip(axes, years):
        d = ccdf.loc[ccdf[year_col] == year].copy()
        if min_ccdf is not None:
            d = d.loc[d["ccdf"] > min_ccdf]
        x = d["bin"].to_numpy(dtype=float)
        y = d["ccdf"].to_numpy(dtype=float)
        if transform == "linear":
            ax.plot(x, y)
        elif transform == "loglog":
            mask = (x > 0) & (y > 0)
            ax.loglog(x[mask], y[mask])
        elif transform == "double_log":
            mask = y > 1.0
            ax.plot(x[mask], np.log(np.log(y[mask])))
        else:
            raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")
        ax.set_title(str(int(year)))
        ax.set_xlabel("Income")
        ax.set_ylabel({"linear": "CCDF", "loglog": "CCDF", "double_log": "ln[ln(CCDF)]"}[transform])
        ax.grid(True, alpha=0.3)
    for ax in axes[len(years):]:
        ax.remove()
    fig.tight_layout()
    return fig


def plot_measure_comparison(ccdf, year, measures=("income", "income_effective"), transform="loglog"):
    """Compare two income measures for one year on the same axes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for measure in measures:
        d = ccdf.loc[(ccdf["year"] == year) & (ccdf["measure"] == measure)]
        if d.empty:
            continue
        x = d["bin"].to_numpy(dtype=float)
        y = d["ccdf"].to_numpy(dtype=float)
        if transform == "loglog":
            mask = (x > 0) & (y > 0)
            ax.loglog(x[mask], y[mask], label=measure)
        elif transform == "linear":
            ax.plot(x, y, label=measure)
        elif transform == "double_log":
            mask = y > 1.0
            ax.plot(x[mask], np.log(np.log(y[mask])), label=measure)
        else:
            raise ValueError("Unsupported transform.")
    ax.set_title(str(year))
    ax.set_xlabel("Income")
    ax.set_ylabel("CCDF" if transform != "double_log" else "ln[ln(CCDF)]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_lorenz_by_year(df, value_col="income", year_col="year", n_cols=4):
    years = sorted(df[year_col].dropna().unique())
    fig, axes = _axes_for_years(years, n_cols=n_cols, width=4.2, height=3.8)
    for ax, year in zip(axes, years):
        values = pd.to_numeric(df.loc[df[year_col] == year, value_col], errors="coerce")
        p, l = lorenz_curve(values)
        if p.size:
            ax.plot(p, l)
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.text(0.05, 0.88, f"Gini = {gini(values):.3f}", transform=ax.transAxes)
        ax.set_title(str(int(year)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(years):]:
        ax.remove()
    fig.tight_layout()
    return fig
