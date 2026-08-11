"""Scientific visualization utilities for PNAD income-distribution analysis.

The plotting API supports two complementary use cases:

1. individual figures selected with ``year=...``;
2. small-multiple figures selected with ``years=[...]`` and an explicit
   ``nrows`` x ``ncols`` layout.

If the requested number of years exceeds the grid capacity, the grid functions
paginate automatically.  Unused axes on the final page are hidden.  Numerical
analysis remains independent of Jupyter; the notebook only orchestrates these
functions and displays the returned Matplotlib figures.
"""

from __future__ import annotations

from math import ceil
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inequality import lorenz_curve


DEFAULT_NCOLS = 4
DEFAULT_MAX_PANELS = 24


def _available_years(df: pd.DataFrame, year_col: str = "year") -> list[int]:
    """Return sorted finite survey years present in a data frame."""
    if year_col not in df.columns:
        raise KeyError(f"Column '{year_col}' is absent from the data.")
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


def _select_years(
    df: pd.DataFrame,
    years: Iterable[int] | None,
    year_col: str = "year",
) -> list[int]:
    """Resolve a requested year subset and reject unavailable survey years."""
    available = _available_years(df, year_col=year_col)
    if years is None:
        return available

    selected = list(dict.fromkeys(int(year) for year in years))
    if not selected:
        raise ValueError("years must contain at least one survey year.")

    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(
            "Requested survey years are absent from the data: "
            + ", ".join(str(year) for year in missing)
        )
    return selected


def _validate_year(df: pd.DataFrame, year: int, year_col: str = "year") -> int:
    """Return an integer year after checking that it is available."""
    selected = _select_years(df, [int(year)], year_col=year_col)
    return selected[0]


def _resolve_grid(
    n_items: int,
    nrows: int | None,
    ncols: int | None,
    max_panels: int | None,
) -> tuple[int, int, int, bool]:
    """Resolve grid shape, page capacity, and whether rows were user-fixed.

    When both ``nrows`` and ``ncols`` are supplied, their product is the page
    capacity and therefore has priority over ``max_panels``.  When one or both
    dimensions are omitted, a compact layout is derived while retaining the
    legacy ``max_panels`` pagination control.
    """
    if n_items < 1:
        raise ValueError("At least one panel is required.")
    if nrows is not None and nrows < 1:
        raise ValueError("nrows must be at least 1.")
    if ncols is not None and ncols < 1:
        raise ValueError("ncols must be at least 1.")
    if max_panels is not None and max_panels < 1:
        raise ValueError("max_panels must be at least 1.")

    fixed_rows = nrows is not None
    if nrows is not None and ncols is not None:
        capacity = nrows * ncols
        return nrows, ncols, capacity, True

    target_capacity = min(n_items, max_panels or DEFAULT_MAX_PANELS)

    if ncols is not None:
        resolved_ncols = ncols
        resolved_nrows = ceil(target_capacity / resolved_ncols)
    elif nrows is not None:
        resolved_nrows = nrows
        resolved_ncols = ceil(target_capacity / resolved_nrows)
    else:
        resolved_ncols = min(DEFAULT_NCOLS, target_capacity)
        resolved_nrows = ceil(target_capacity / resolved_ncols)

    capacity = resolved_nrows * resolved_ncols
    return resolved_nrows, resolved_ncols, capacity, fixed_rows


def _year_pages(years: list[int], capacity: int) -> list[list[int]]:
    """Split selected survey years into deterministic pages."""
    return [years[i : i + capacity] for i in range(0, len(years), capacity)]


def _grid_axes(
    page_size: int,
    nrows: int,
    ncols: int,
    panel_size: tuple[float, float] = (4.0, 3.0),
    fixed_rows: bool = True,
):
    """Create a small-multiple grid and return flattened axes.

    If rows were not explicitly requested, the final page is compacted to the
    minimum number of rows needed for its remaining panels.
    """
    rows = nrows if fixed_rows else max(1, ceil(page_size / ncols))
    cols = ncols if fixed_rows else min(ncols, page_size)
    if not fixed_rows:
        rows = max(1, ceil(page_size / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(panel_size[0] * cols, panel_size[1] * rows),
        squeeze=False,
    )
    return fig, axes.ravel()


def _finite_values(df: pd.DataFrame, value_col: str, year: int) -> pd.Series:
    """Return finite values for one survey year and one analytical measure."""
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    _validate_year(df, year)
    values = pd.to_numeric(df.loc[df["year"] == int(year), value_col], errors="coerce")
    return values[np.isfinite(values)]


def plot_gini_evolution(
    summary: pd.DataFrame,
    value_col: str = "income",
    years: Iterable[int] | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
):
    """Plot the Gini time series for all or selected survey years."""
    column = f"{value_col}_gini"
    if column not in summary.columns:
        raise KeyError(f"Column '{column}' is absent from the summary table.")
    selected_years = _select_years(summary, years)
    frame = summary.loc[summary["year"].isin(selected_years), ["year", column]].dropna()
    frame = frame.sort_values("year")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frame["year"], frame[column], marker="o", markersize=4, linewidth=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gini coefficient")
    ax.set_title(f"Annual Gini coefficient: {value_col}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histogram(
    df: pd.DataFrame,
    year: int,
    value_col: str = "income",
    bins: int = 60,
    yscale: str = "linear",
    figsize: tuple[float, float] = (7.0, 4.5),
):
    """Plot one annual income histogram selected with ``year``."""
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    values = _finite_values(df, value_col=value_col, year=year)

    fig, ax = plt.subplots(figsize=figsize)
    if values.empty:
        ax.text(0.5, 0.5, "No finite observations", ha="center", va="center")
    else:
        ax.hist(values, bins=bins)
        ax.set_yscale(yscale)
    ax.set_title(f"PNAD {int(year)}")
    ax.set_xlabel("Income")
    ax.set_ylabel("Frequency" + (" (log)" if yscale == "log" else ""))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_histogram_grid(
    df: pd.DataFrame,
    value_col: str = "income",
    years: Iterable[int] | None = None,
    bins: int = 60,
    yscale: str = "linear",
    nrows: int | None = None,
    ncols: int | None = None,
    max_panels: int | None = DEFAULT_MAX_PANELS,
    panel_size: tuple[float, float] = (4.0, 3.0),
) -> list:
    """Plot selected annual histograms in configurable subplot grids.

    Examples
    --------
    ``years=[2001, 2010, 2020, 2025], nrows=2, ncols=2`` produces one 2x2
    figure.  Selecting ten years with ``nrows=2, ncols=3`` produces two pages:
    a 2x3 page followed by a second 2x3 page with unused axes hidden.
    """
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")

    selected_years = _select_years(df, years)
    rows, cols, capacity, fixed_rows = _resolve_grid(
        len(selected_years), nrows=nrows, ncols=ncols, max_panels=max_panels
    )
    grouped = {int(year): group for year, group in df.groupby("year", sort=True)}
    figures = []

    for page in _year_pages(selected_years, capacity):
        fig, axes = _grid_axes(
            len(page), rows, cols, panel_size=panel_size, fixed_rows=fixed_rows
        )
        for ax, year in zip(axes, page):
            values = pd.to_numeric(grouped[year][value_col], errors="coerce")
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
        # This legacy diagnostic is defined on the percentage CCDF scale.
        mask = np.isfinite(x) & np.isfinite(y) & (y > 1.0)
        return x[mask], np.log(np.log(y[mask]))
    raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")


def _ccdf_frame(ccdf: pd.DataFrame, year: int, measure: str) -> pd.DataFrame:
    """Return one year/measure CCDF after validating the selection."""
    if "measure" not in ccdf.columns:
        raise KeyError("Column 'measure' is absent from the CCDF table.")
    measure_frame = ccdf.loc[ccdf["measure"] == measure].copy()
    if measure_frame.empty:
        raise ValueError(f"No CCDF rows are available for measure '{measure}'.")
    _validate_year(measure_frame, year)
    return measure_frame.loc[measure_frame["year"] == int(year)].copy()


def plot_ccdf(
    ccdf: pd.DataFrame,
    year: int,
    measure: str = "income",
    transform: str = "linear",
    figsize: tuple[float, float] = (7.0, 4.5),
):
    """Plot one annual CCDF selected with ``year``."""
    frame = _ccdf_frame(ccdf, year=year, measure=measure)
    x, y = _ccdf_coordinates(frame, transform)

    fig, ax = plt.subplots(figsize=figsize)
    if transform == "loglog":
        ax.loglog(x, y)
    else:
        ax.plot(x, y)
    ax.set_title(f"PNAD {int(year)} — {measure}")
    ax.set_xlabel("Income")
    ax.set_ylabel("ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ccdf_grid(
    ccdf: pd.DataFrame,
    measure: str = "income",
    years: Iterable[int] | None = None,
    transform: str = "linear",
    nrows: int | None = None,
    ncols: int | None = None,
    max_panels: int | None = DEFAULT_MAX_PANELS,
    panel_size: tuple[float, float] = (4.0, 3.0),
) -> list:
    """Plot selected annual CCDFs in configurable subplot grids."""
    if transform not in {"linear", "loglog", "double_log"}:
        raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")
    frame = ccdf.loc[ccdf["measure"] == measure].copy()
    if frame.empty:
        raise ValueError(f"No CCDF rows are available for measure '{measure}'.")

    selected_years = _select_years(frame, years)
    rows, cols, capacity, fixed_rows = _resolve_grid(
        len(selected_years), nrows=nrows, ncols=ncols, max_panels=max_panels
    )
    grouped = {int(year): group for year, group in frame.groupby("year", sort=True)}
    figures = []

    for page in _year_pages(selected_years, capacity):
        fig, axes = _grid_axes(
            len(page), rows, cols, panel_size=panel_size, fixed_rows=fixed_rows
        )
        for ax, year in zip(axes, page):
            x, y = _ccdf_coordinates(grouped[year], transform)
            if transform == "loglog":
                ax.loglog(x, y)
            else:
                ax.plot(x, y)
            ax.set_title(f"PNAD {year}")
            ax.set_xlabel("Income")
            ax.set_ylabel("ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]")
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
    years: Iterable[int],
    transform: str = "loglog",
    figsize: tuple[float, float] = (7.5, 5.0),
):
    """Overlay an explicitly selected set of annual CCDFs in one axes."""
    frame = ccdf.loc[ccdf["measure"] == measure].copy()
    if frame.empty:
        raise ValueError(f"No CCDF rows are available for measure '{measure}'.")
    selected_years = _select_years(frame, years)

    fig, ax = plt.subplots(figsize=figsize)
    for year in selected_years:
        annual = frame.loc[frame["year"] == year]
        x, y = _ccdf_coordinates(annual, transform)
        if transform == "loglog":
            ax.loglog(x, y, label=str(year))
        else:
            ax.plot(x, y, label=str(year))
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
    measures: tuple[str, ...] = ("income", "income_adj"),
    transform: str = "loglog",
    figsize: tuple[float, float] = (6.5, 4.5),
):
    """Compare multiple income measures within one selected survey year."""
    available = ccdf.loc[ccdf["measure"].isin(measures)].copy()
    if available.empty:
        raise ValueError("No requested measures are available in the CCDF table.")
    _validate_year(available, year)

    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    for measure in measures:
        annual = available.loc[
            (available["year"] == int(year)) & (available["measure"] == measure)
        ]
        if annual.empty:
            continue
        x, y = _ccdf_coordinates(annual, transform)
        if transform == "loglog":
            ax.loglog(x, y, label=measure)
        else:
            ax.plot(x, y, label=measure)
        plotted += 1
    if plotted == 0:
        raise ValueError(f"No requested measures are available for year {int(year)}.")
    ax.set_title(f"PNAD {int(year)}")
    ax.set_xlabel("Income")
    ax.set_ylabel("ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_measure_comparison_grid(
    ccdf: pd.DataFrame,
    measures: tuple[str, ...] = ("income", "income_adj"),
    years: Iterable[int] | None = None,
    transform: str = "loglog",
    nrows: int | None = None,
    ncols: int | None = None,
    max_panels: int | None = DEFAULT_MAX_PANELS,
    panel_size: tuple[float, float] = (4.0, 3.0),
) -> list:
    """Compare measures for selected years in configurable subplot grids."""
    available = ccdf.loc[ccdf["measure"].isin(measures)].copy()
    if available.empty:
        raise ValueError("No requested measures are available in the CCDF table.")
    selected_years = _select_years(available, years)
    rows, cols, capacity, fixed_rows = _resolve_grid(
        len(selected_years), nrows=nrows, ncols=ncols, max_panels=max_panels
    )
    figures = []

    for page in _year_pages(selected_years, capacity):
        fig, axes = _grid_axes(
            len(page), rows, cols, panel_size=panel_size, fixed_rows=fixed_rows
        )
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
            ax.set_ylabel("ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]")
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


def plot_lorenz_curve(
    df: pd.DataFrame,
    year: int,
    value_col: str = "income",
    figsize: tuple[float, float] = (5.5, 5.0),
):
    """Plot one Lorenz curve selected with ``year``."""
    values = _finite_values(df, value_col=value_col, year=year)
    population, income_share = lorenz_curve(values)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(population, income_share, label=str(int(year)))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Equality")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative population share")
    ax.set_ylabel("Cumulative income share")
    ax.set_title(f"Lorenz curve: {int(year)}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_lorenz_grid(
    df: pd.DataFrame,
    value_col: str = "income",
    years: Iterable[int] | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    max_panels: int | None = DEFAULT_MAX_PANELS,
    panel_size: tuple[float, float] = (3.6, 3.3),
) -> list:
    """Plot selected Lorenz curves in configurable subplot grids."""
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    selected_years = _select_years(df, years)
    rows, cols, capacity, fixed_rows = _resolve_grid(
        len(selected_years), nrows=nrows, ncols=ncols, max_panels=max_panels
    )
    grouped = {int(year): group for year, group in df.groupby("year", sort=True)}
    figures = []

    for page in _year_pages(selected_years, capacity):
        fig, axes = _grid_axes(
            len(page), rows, cols, panel_size=panel_size, fixed_rows=fixed_rows
        )
        for ax, year in zip(axes, page):
            values = pd.to_numeric(grouped[year][value_col], errors="coerce")
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
