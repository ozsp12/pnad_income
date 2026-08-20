"""Scientific figures used by the PNAD income output exporter."""

from __future__ import annotations

from math import ceil
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import inequality_statistics, lorenz_curve

DEFAULT_NCOLS = 4
DEFAULT_MAX_PANELS = 24


def _years(df: pd.DataFrame, years: Iterable[int] | None = None) -> list[int]:
    if "year" not in df.columns:
        raise KeyError("Column 'year' is absent from the data.")
    available = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique())
    if years is None:
        return available
    selected = list(dict.fromkeys(int(year) for year in years))
    missing = sorted(set(selected).difference(available))
    if not selected:
        raise ValueError("years must contain at least one survey year.")
    if missing:
        raise ValueError("Requested survey years are absent from the data: " + ", ".join(map(str, missing)))
    return selected


def _grid(years: list[int], nrows=None, ncols=None, max_panels=DEFAULT_MAX_PANELS):
    target = min(len(years), max_panels or len(years))
    cols = ncols or min(DEFAULT_NCOLS, target)
    rows = nrows or ceil(target / cols)
    capacity = rows * cols
    return rows, cols, [years[i : i + capacity] for i in range(0, len(years), capacity)]


def _axes(page, rows, cols, panel_size=(4.0, 3.0)):
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(panel_size[0] * cols, panel_size[1] * rows),
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[len(page) :]:
        ax.set_visible(False)
    return fig, flat


def _finite(df: pd.DataFrame, year: int, value_col="income", *, positive=False) -> pd.Series:
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    _years(df, [year])
    values = pd.to_numeric(df.loc[df["year"] == int(year), value_col], errors="coerce")
    values = values[np.isfinite(values)]
    return values[values > 0] if positive else values


def _measure_label(measure: str) -> str:
    labels = {
        "income": "Nominal income",
        "income_adj": "Income (2025 USD)",
        "income_effective": "Effective income",
        "income_effective_adj": "Effective income (2025 USD)",
    }
    return labels.get(measure, measure)


def plot_gini_evolution(summary, value_col="income", years=None, figsize=(10, 5)):
    column = f"{value_col}_gini"
    selected = _years(summary, years)
    frame = summary.loc[summary["year"].isin(selected), ["year", column]].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frame["year"], frame[column], marker="o", markersize=4, linewidth=1.2)
    ax.set(xlabel="Year", ylabel="Gini coefficient", title=f"Annual Gini coefficient: {value_col}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_top_income_shares(summary, value_col="income", years=None, figsize=(10, 5.5)):
    columns = {
        "Top 10%": f"{value_col}_top_10_share",
        "Top 1%": f"{value_col}_top_1_share",
        "Top 0.1%": f"{value_col}_top_0_1_share",
    }
    selected = _years(summary, years)
    frame = summary.loc[summary["year"].isin(selected)].sort_values("year")
    fig, ax = plt.subplots(figsize=figsize)
    for label, column in columns.items():
        ax.plot(frame["year"], 100 * frame[column], marker="o", markersize=3, label=label)
    ax.set(xlabel="Year", ylabel="Share of aggregate income [%]", title="Top-income concentration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_extended_inequality_evolution(summary, value_col="income", years=None, figsize=(10, 5.5)):
    columns = {
        "Pietra": f"{value_col}_pietra",
        "Kolkata": f"{value_col}_k",
        "Zanardi": f"{value_col}_zanardi",
    }
    selected = _years(summary, years)
    frame = summary.loc[summary["year"].isin(selected)].sort_values("year")
    fig, ax = plt.subplots(figsize=figsize)
    for label, column in columns.items():
        ax.plot(frame["year"], frame[column], marker="o", markersize=3, label=label)
    ax.set(xlabel="Year", ylabel="Index value", title="Lorenz-based inequality measures")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gini_validation(summary, references, value_col="income", figsize=(10, 5.5)):
    column = f"{value_col}_gini"
    fig, ax = plt.subplots(figsize=figsize)
    frame = summary[["year", column]].dropna().sort_values("year")
    ax.plot(frame["year"], frame[column], marker="o", markersize=3, label="Calculated PNAD")
    for source, group in references.groupby("source", sort=True):
        ax.plot(group["year"], group["gini"], marker="o", markersize=3, label=str(source))
    ax.set(xlabel="Year", ylabel="Gini coefficient", title="Gini validation against external references")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_histogram(df, year, value_col="income", bins=100, yscale="log", figsize=(7, 4.5)):
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    values = _finite(df, year, value_col, positive=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, log=(yscale == "log"))
    ylabel = "Frequency (log scale)" if yscale == "log" else "Frequency"
    ax.set(title=f"PNAD {int(year)}", xlabel="Income", ylabel=ylabel)
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    fig.tight_layout()
    return fig


def plot_histogram_grid(
    df,
    value_col="income",
    years=None,
    bins=100,
    yscale="log",
    nrows=None,
    ncols=None,
    max_panels=DEFAULT_MAX_PANELS,
    panel_size=(4, 3),
):
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    selected = _years(df, years)
    rows, cols, pages = _grid(selected, nrows, ncols, max_panels)
    figures = []
    for page in pages:
        fig, axes = _axes(page, rows, cols, panel_size)
        for ax, year in zip(axes, page):
            values = _finite(df, year, value_col, positive=True)
            ax.hist(values, bins=bins, log=(yscale == "log"))
            ylabel = "Frequency (log scale)" if yscale == "log" else "Frequency"
            ax.set(title=f"PNAD {year}", xlabel="Income", ylabel=ylabel)
            ax.grid(axis="y", alpha=0.5, linestyle="--")
        fig.suptitle(f"Annual histograms: {value_col} — {page[0]}–{page[-1]}", y=1.002)
        fig.tight_layout()
        figures.append(fig)
    return figures


# Keep the empirical survival function on the probability scale throughout plotting.
def _ccdf_probability(frame) -> np.ndarray:
    values = frame["ccdf"].to_numpy(float)
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmax(finite) > 1.0 + 1e-12:
        values = values / 100.0
    return values


def _ccdf_xy(frame, transform):
    x = frame["bin"].to_numpy(float)
    probability = _ccdf_probability(frame)
    if transform == "linear":
        mask = np.isfinite(x) & np.isfinite(probability)
        return x[mask], probability[mask]
    if transform == "loglog":
        mask = np.isfinite(x) & np.isfinite(probability) & (x > 0) & (probability > 0)
        return x[mask], probability[mask]
    if transform in {"gompertz", "double_log"}:
        mask = np.isfinite(x) & np.isfinite(probability) & (probability > 0) & (probability < 1)
        return x[mask], -np.log(-np.log(probability[mask]))
    raise ValueError("transform must be 'linear', 'loglog', 'gompertz', or 'double_log'.")


def _ccdf_ylabel(transform: str) -> str:
    return "-ln[-ln(S(x))]" if transform in {"gompertz", "double_log"} else "S(x)"


def _plot_ccdf_line(ax, frame, transform, label=None):
    x, y = _ccdf_xy(frame, transform)
    if transform == "loglog":
        ax.loglog(x, y, label=label)
    else:
        ax.plot(x, y, label=label)


def plot_ccdf(ccdf, year, measure="income", transform="loglog", figsize=(7, 4.5)):
    frame = ccdf.loc[(ccdf["year"] == int(year)) & (ccdf["measure"] == measure)]
    if frame.empty:
        raise ValueError(f"No CCDF rows are available for {measure} in {year}.")
    fig, ax = plt.subplots(figsize=figsize)
    _plot_ccdf_line(ax, frame, transform)
    ax.set(title=f"PNAD {int(year)} — {_measure_label(measure)}", xlabel="Income", ylabel=_ccdf_ylabel(transform))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ccdf_grid(
    ccdf,
    measure="income",
    years=None,
    transform="loglog",
    nrows=None,
    ncols=None,
    max_panels=DEFAULT_MAX_PANELS,
    panel_size=(4, 3),
):
    frame = ccdf.loc[ccdf["measure"] == measure]
    if frame.empty:
        raise ValueError(f"No CCDF rows are available for measure '{measure}'.")
    selected = _years(frame, years)
    rows, cols, pages = _grid(selected, nrows, ncols, max_panels)
    figures = []
    for page in pages:
        fig, axes = _axes(page, rows, cols, panel_size)
        for ax, year in zip(axes, page):
            _plot_ccdf_line(ax, frame.loc[frame["year"] == year], transform)
            ax.set(title=f"PNAD {year}", xlabel="Income", ylabel=_ccdf_ylabel(transform))
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Annual CCDF: {_measure_label(measure)} — {page[0]}–{page[-1]} ({transform})", y=1.002)
        fig.tight_layout()
        figures.append(fig)
    return figures


def plot_ccdf_selected_years(ccdf, measure, years, transform="loglog", figsize=(7.5, 5)):
    frame = ccdf.loc[ccdf["measure"] == measure]
    selected = _years(frame, years)
    fig, ax = plt.subplots(figsize=figsize)
    for year in selected:
        _plot_ccdf_line(ax, frame.loc[frame["year"] == year], transform, str(year))
    ax.set(xlabel="Income", ylabel=_ccdf_ylabel(transform), title=f"{_measure_label(measure)}: selected survey years")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Year")
    fig.tight_layout()
    return fig


def plot_measure_comparison(
    ccdf,
    year,
    measures=("income", "income_adj"),
    transform="loglog",
    figsize=(6.5, 4.5),
):
    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    for measure in measures:
        frame = ccdf.loc[(ccdf["year"] == int(year)) & (ccdf["measure"] == measure)]
        if frame.empty:
            continue
        _plot_ccdf_line(ax, frame, transform, _measure_label(measure))
        plotted += 1
    if not plotted:
        raise ValueError(f"No requested measures are available for year {year}.")
    ax.set(title=f"PNAD {int(year)}", xlabel="Income", ylabel=_ccdf_ylabel(transform))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_measure_comparison_grid(
    ccdf,
    measures=("income", "income_adj"),
    years=None,
    transform="loglog",
    nrows=None,
    ncols=None,
    max_panels=DEFAULT_MAX_PANELS,
    panel_size=(4, 3),
):
    available = ccdf.loc[ccdf["measure"].isin(measures)]
    selected = _years(available, years)
    rows, cols, pages = _grid(selected, nrows, ncols, max_panels)
    figures = []
    for page in pages:
        fig, axes = _axes(page, rows, cols, panel_size)
        for ax, year in zip(axes, page):
            for measure in measures:
                frame = available.loc[(available["year"] == year) & (available["measure"] == measure)]
                if not frame.empty:
                    _plot_ccdf_line(ax, frame, transform, _measure_label(measure))
            ax.set(title=f"PNAD {year}", xlabel="Income", ylabel=_ccdf_ylabel(transform))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="small")
        fig.suptitle(f"Annual measure comparison — {page[0]}–{page[-1]}", y=1.002)
        fig.tight_layout()
        figures.append(fig)
    return figures


def _draw_lorenz(ax, values, year, annotate=False):
    population, income = lorenz_curve(values)
    ax.plot(population, income)
    ax.plot([0, 1], [0, 1], linestyle="--")
    if annotate:
        metrics = inequality_statistics(values)
        k = metrics["k"]
        if np.isfinite(k):
            ax.plot([k], [1 - k], marker="o")
        ax.text(
            0.04,
            0.96,
            f"G={metrics['gini']:.3f}\nP={metrics['pietra']:.3f}\nk={metrics['k']:.3f}\nZ={metrics['zanardi']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize="small",
        )
    ax.set(xlim=(0, 1), ylim=(0, 1), title=f"PNAD {year}", xlabel="Population share", ylabel="Income share")
    ax.grid(True, alpha=0.25)


def plot_lorenz_curve(df, year, value_col="income", figsize=(5.5, 5), annotate=False):
    fig, ax = plt.subplots(figsize=figsize)
    _draw_lorenz(ax, _finite(df, year, value_col), int(year), annotate)
    fig.tight_layout()
    return fig


def plot_lorenz_grid(
    df,
    value_col="income",
    years=None,
    nrows=None,
    ncols=None,
    max_panels=DEFAULT_MAX_PANELS,
    panel_size=(3.6, 3.3),
    annotate=False,
):
    selected = _years(df, years)
    rows, cols, pages = _grid(selected, nrows, ncols, max_panels)
    figures = []
    for page in pages:
        fig, axes = _axes(page, rows, cols, panel_size)
        for ax, year in zip(axes, page):
            _draw_lorenz(ax, _finite(df, year, value_col), year, annotate)
        fig.suptitle(f"Annual Lorenz curves: {value_col} — {page[0]}–{page[-1]}", y=1.002)
        fig.tight_layout()
        figures.append(fig)
    return figures


def _transition(ax):
    ax.axvline(2015.5, linestyle="--", linewidth=1, alpha=0.55)


def plot_primary_indices(indices, figsize=(11, 5.8)):
    fig, ax = plt.subplots(figsize=figsize)
    for column, label, marker in (("gini", "Gini", "o"), ("pietra", "Pietra", "s"), ("k", "Kolkata", "^")):
        ax.plot(indices["year"], indices[column], marker=marker, markersize=3.5, linewidth=1.3, label=label)
    ax.set(xlabel="Year", ylabel="Index value", title="Long-run inequality indices", ylim=(0, 1))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    _transition(ax)
    fig.tight_layout()
    return fig


def plot_zanardi(indices, figsize=(11, 5.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, linewidth=1, alpha=0.6)
    ax.plot(indices["year"], indices["zanardi"], marker="o", markersize=3.8, linewidth=1.4)
    ax.set(xlabel="Year", ylabel="Zanardi index", title="Lorenz-curve asymmetry")
    ax.grid(True, alpha=0.25)
    _transition(ax)
    fig.tight_layout()
    return fig


def plot_information_indices(indices, figsize=(11, 5.8)):
    atkinson = next((column for column in indices.columns if column.startswith("atkinson_")), None)
    if atkinson is None:
        raise KeyError("No Atkinson column is available.")
    fig, ax = plt.subplots(figsize=figsize)
    for column, label, marker in (
        ("theil", "Theil T", "o"),
        (atkinson, f"Atkinson ({atkinson.split('_', 1)[1]})", "s"),
        ("shannon_inequality", "Normalized Shannon deficit", "^"),
    ):
        ax.plot(indices["year"], indices[column], marker=marker, markersize=3.3, linewidth=1.3, label=label)
    ax.set(xlabel="Year", ylabel="Index value", title="Information and welfare inequality measures")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    _transition(ax)
    fig.tight_layout()
    return fig


def plot_kolkata_pietra_relationships(indices, figsize=(12.4, 5.2)):
    ggrid = np.linspace(0, max(0.9, float(indices["gini"].max()) * 1.04), 250)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].scatter(indices["gini"], indices["pietra"], s=28, alpha=0.8)
    axes[0].plot(ggrid, 0.75 * ggrid, linestyle="--")
    axes[0].set(xlabel="Gini", ylabel="Pietra", title="Pietra-Gini relation")
    axes[1].scatter(indices["gini"], indices["k"], s=28, alpha=0.8)
    axes[1].plot(ggrid, 0.5 + 0.375 * ggrid, linestyle="--")
    axes[1].set(xlabel="Gini", ylabel="Kolkata", title="Kolkata-Gini relation")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pietra_kolkata_bound(indices, figsize=(10, 5.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(1, linestyle="--", linewidth=1.1)
    ax.plot(indices["year"], indices["pietra_over_kolkata_excess"], marker="o", markersize=3.6, linewidth=1.3)
    ax.set(xlabel="Year", ylabel="P/(2K-1)", title="Pietra-Kolkata bound")
    ax.grid(True, alpha=0.25)
    _transition(ax)
    fig.tight_layout()
    return fig


def plot_gini_zanardi(indices, figsize=(7.2, 5.8)):
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(indices["gini"], indices["zanardi"], c=indices["year"], s=40)
    ax.axhline(0, linewidth=1, alpha=0.55)
    ax.set(xlabel="Gini coefficient", ylabel="Zanardi index", title="Concentration and Lorenz asymmetry")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax).set_label("Year")
    fig.tight_layout()
    return fig
