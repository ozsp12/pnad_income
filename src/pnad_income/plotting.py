"""Scientific visualization utilities for the PNAD income-distribution analysis.

The functions in this module are intentionally independent of Jupyter.  The
project notebook only orchestrates them and displays the resulting figures.
Annual grid functions paginate the 45 survey years so that every available
year is represented without producing an unreadable single canvas.
"""

from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inequality import lorenz_curve


def _available_years(df: pd.DataFrame, year_col: str = "year") -> list[int]:
    """Return sorted finite survey years present in a data frame."""
    if year_col not in df.columns:
        raise KeyError(f"Column '{year_col}' is absent from the data.")
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


def _year_pages(years, max_panels: int = 24) -> list[list[int]]:
    """Split survey years into deterministic pages of small multiples."""
    values = sorted({int(year) for year in years})
    if max_panels < 1:
        raise ValueError("max_panels must be at least 1.")
    return [values[i : i + max_panels] for i in range(0, len(values), max_panels)]


def _grid_axes(n_panels: int, ncols: int = 4, panel_size=(4.0, 3.0)):
    """Create a rectangular small-multiple grid and return flattened axes."""
    if n_panels < 1:
        raise ValueError("n_panels must be at least 1.")
    ncols = min(max(1, ncols), n_panels)
    nrows = ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    return fig, axes.ravel()


def plot_gini_evolution(summary: pd.DataFrame, value_col: str = "income"):
    """Plot the Gini coefficient for every available survey year."""
    column = f"{value_col}_gini"
    if column not in summary.columns:
        raise KeyError(f"Column '{column}' is absent from the summary table.")
    frame = summary[["year", column]].dropna().sort_values("year")
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(frame["year"], frame[column], marker="o", markersize=4, linewidth=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gini coefficient")
    ax.set_title(f"Annual Gini coefficient: {value_col}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histogram_grid(
    df: pd.DataFrame,
    value_col: str = "income",
    years=None,
    bins: int = 60,
    yscale: str = "linear",
    ncols: int = 4,
    max_panels: int = 24,
) -> list:
    """Plot annual income histograms for every requested survey year.

    Parameters
    ----------
    df:
        Long-format analytical panel.
    value_col:
        Income measure to display, for example ``income`` or ``income_adj``.
    years:
        Optional survey-year subset.  By default all years are shown.
    bins:
        Number of equal-width histogram bins within each annual sample.
    yscale:
        ``"linear"`` or ``"log"`` frequency scale.
    ncols, max_panels:
        Grid pagination controls.  With the defaults, the full 45-year series
        is returned as two figures.
    """
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")

    selected_years = _available_years(df) if years is None else sorted({int(y) for y in years})
    grouped = {int(year): group for year, group in df.groupby("year", sort=True)}
    figures = []

    for page in _year_pages(selected_years, max_panels=max_panels):
        fig, axes = _grid_axes(len(page), ncols=ncols)
        for ax, year in zip(axes, page):
            group = grouped.get(year)
            if group is None:
                ax.set_visible(False)
                continue
            values = pd.to_numeric(group[value_col], errors="coerce")
            values = values[np.isfinite(values)]
            if values.empty:
                ax.text(0.5, 0.5, "No finite observations", ha="center", va="center")
            else:
                ax.hist(values, bins=bins)
                ax.set_yscale(yscale)
            ax.set_title(f"PNAD {year}")
            ax.set_xlabel("Income")
            ax.set_ylabel("Frequency" + (" (log)" if yscale == "log" else ""))
            ax.grid(True, alpha=0.25)
        for ax in axes[len(page) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"Annual histograms: {value_col} — {page[0]}–{page[-1]} ({yscale} frequency)",
            y=1.002,
        )
        fig.tight_layout()
        figures.append(fig)
    return figures


def _ccdf_coordinates(frame: pd.DataFrame, transform: str):
    """Return coordinates for linear, log-log, or double-log CCDF display."""
    x = frame["bin"].to_numpy(dtype=float)
    y = frame["ccdf"].to_numpy(dtype=float)
    if transform == "linear":
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]
    if transform == "loglog":
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        return x[mask], y[mask]
    if transform == "double_log":
        # The historical diagnostic is defined on the percentage CCDF scale,
        # hence y > 1 is required for ln(ln(y)) to remain finite.
        mask = np.isfinite(x) & np.isfinite(y) & (y > 1.0)
        return x[mask], np.log(np.log(y[mask]))
    raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")


def plot_ccdf_grid(
    ccdf: pd.DataFrame,
    measure: str = "income",
    years=None,
    transform: str = "linear",
    ncols: int = 4,
    max_panels: int = 24,
) -> list:
    """Plot the annual CCDF for every requested survey year as small multiples."""
    if transform not in {"linear", "loglog", "double_log"}:
        raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")
    frame = ccdf.loc[ccdf["measure"] == measure].copy()
    if frame.empty:
        raise ValueError(f"No CCDF rows are available for measure '{measure}'.")

    selected_years = _available_years(frame) if years is None else sorted({int(y) for y in years})
    grouped = {int(year): group for year, group in frame.groupby("year", sort=True)}
    figures = []

    for page in _year_pages(selected_years, max_panels=max_panels):
        fig, axes = _grid_axes(len(page), ncols=ncols)
        for ax, year in zip(axes, page):
            annual = grouped.get(year)
            if annual is None or annual.empty:
                ax.set_visible(False)
                continue
            x, y = _ccdf_coordinates(annual, transform)
            if transform == "loglog":
                ax.loglog(x, y)
            else:
                ax.plot(x, y)
            ax.set_title(f"PNAD {year}")
            ax.set_xlabel("Income")
            if transform == "double_log":
                ax.set_ylabel("ln[ln(CCDF [%])] ")
            else:
                ax.set_ylabel("CCDF [%]")
            ax.grid(True, alpha=0.3)
        for ax in axes[len(page) :]:
            ax.set_visible(False)
        label = {
            "linear": "linear axes",
            "loglog": "log-log axes",
            "double_log": "legacy ln[ln(CCDF)] transform",
        }[transform]
        fig.suptitle(f"Annual CCDF: {measure} — {page[0]}–{page[-1]} ({label})", y=1.002)
        fig.tight_layout()
        figures.append(fig)
    return figures


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
    ax.set_ylabel("ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]")
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
    ax.set_ylabel("ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_measure_comparison_grid(
    ccdf: pd.DataFrame,
    measures: tuple[str, ...] = ("income", "income_adj"),
    years=None,
    transform: str = "loglog",
    ncols: int = 4,
    max_panels: int = 24,
) -> list:
    """Compare two or more income measures for all survey years."""
    available = ccdf.loc[ccdf["measure"].isin(measures)].copy()
    if available.empty:
        raise ValueError("No requested measures are available in the CCDF table.")
    selected_years = _available_years(available) if years is None else sorted({int(y) for y in years})
    figures = []

    for page in _year_pages(selected_years, max_panels=max_panels):
        fig, axes = _grid_axes(len(page), ncols=ncols)
        for ax, year in zip(axes, page):
            for measure in measures:
                annual = available.loc[
                    (available["year"] == year) & (available["measure"] == measure)
                ]
                if annual.empty:
                    continue
                x, y = _ccdf_coordinates(annual, transform)
                if transform == "loglog":
                    ax.loglog(x, y, label=measure)
                else:
                    ax.plot(x, y, label=measure)
            ax.set_title(f"PNAD {year}")
            ax.set_xlabel("Income")
            ax.set_ylabel("CCDF [%]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="small")
        for ax in axes[len(page) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"Annual measure comparison — {page[0]}–{page[-1]} ({transform})",
            y=1.002,
        )
        fig.tight_layout()
        figures.append(fig)
    return figures


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


def plot_lorenz_grid(
    df: pd.DataFrame,
    value_col: str = "income",
    years=None,
    ncols: int = 4,
    max_panels: int = 24,
) -> list:
    """Plot one Lorenz curve for every requested survey year."""
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    selected_years = _available_years(df) if years is None else sorted({int(y) for y in years})
    grouped = {int(year): group for year, group in df.groupby("year", sort=True)}
    figures = []

    for page in _year_pages(selected_years, max_panels=max_panels):
        fig, axes = _grid_axes(len(page), ncols=ncols, panel_size=(3.6, 3.3))
        for ax, year in zip(axes, page):
            group = grouped.get(year)
            if group is None:
                ax.set_visible(False)
                continue
            values = pd.to_numeric(group[value_col], errors="coerce")
            population, income_share = lorenz_curve(values)
            ax.plot(population, income_share)
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"PNAD {year}")
            ax.set_xlabel("Population share")
            ax.set_ylabel("Income share")
            ax.grid(True, alpha=0.25)
        for ax in axes[len(page) :]:
            ax.set_visible(False)
        fig.suptitle(f"Annual Lorenz curves: {value_col} — {page[0]}–{page[-1]}", y=1.002)
        fig.tight_layout()
        figures.append(fig)
    return figures
