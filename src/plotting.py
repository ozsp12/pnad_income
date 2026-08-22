"""Scientific figures used by the PNAD income output exporter."""

from __future__ import annotations

from math import ceil
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import gompertz_transform, inequality_statistics, lorenz_curve

DEFAULT_NCOLS = 4
DEFAULT_INEQUALITY_NCOLS = 3
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


def _annual_line_with_gaps(ax, frame, value_col, *, years, label, **plot_kwargs):
    """Plot an annual series on a complete year grid so missing years break the line."""
    series = (
        frame.assign(year=pd.to_numeric(frame["year"], errors="coerce"))
        .dropna(subset=["year"])
        .assign(year=lambda data: data["year"].astype(int))
        .drop_duplicates("year", keep="last")
        .set_index("year")[value_col]
    )
    values = pd.to_numeric(series, errors="coerce").reindex(years)
    return ax.plot(years, values, label=label, **plot_kwargs)[0]


def plot_gini_validation(summary, references, value_col="income", figsize=(10, 5.5)):
    column = f"{value_col}_gini"
    required_summary = {"year", column}
    required_references = {"year", "gini", "source"}
    if missing := required_summary.difference(summary.columns):
        raise KeyError("Summary is missing: " + ", ".join(sorted(missing)))
    if missing := required_references.difference(references.columns):
        raise KeyError("References are missing: " + ", ".join(sorted(missing)))

    available_years = pd.concat(
        [pd.to_numeric(summary["year"], errors="coerce"), pd.to_numeric(references["year"], errors="coerce")],
        ignore_index=True,
    ).dropna()
    if available_years.empty:
        raise ValueError("No annual Gini observations are available.")
    full_years = np.arange(int(available_years.min()), int(available_years.max()) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    _annual_line_with_gaps(
        ax,
        summary[["year", column]],
        column,
        years=full_years,
        label="PNAD (calculated)",
        marker="o",
        markersize=3,
    )
    for source, group in references.groupby("source", sort=True):
        _annual_line_with_gaps(
            ax,
            group[["year", "gini"]],
            "gini",
            years=full_years,
            label=str(source),
            marker="o",
            markersize=3,
        )
    ax.set(xlabel="Year", ylabel="Gini coefficient", title="Gini validation against external references")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_income_group_totals(group_totals, value_col="income", figsize=(12, 6)):
    """Plot non-normalized annual aggregate income split across p80/p99/p100."""
    required = {"year", "p80", "p99", "p100", "total"}
    if missing := required.difference(group_totals.columns):
        raise KeyError("Income-group totals are missing: " + ", ".join(sorted(missing)))

    frame = group_totals.sort_values("year")
    years = pd.to_numeric(frame["year"], errors="raise").astype(int).to_numpy()
    bottom = np.zeros(len(frame), dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    groups = (
        ("p80", "Bottom 80% (p80)", "#4C78A8"),
        ("p99", "Next 19% (p99)", "#F2CF5B"),
        ("p100", "Top 1% (p100)", "#E45756"),
    )
    for column, label, color in groups:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        ax.bar(years, values, bottom=bottom, width=0.8, label=label, color=color)
        bottom += np.nan_to_num(values, nan=0.0)

    ylabel = (
        "Sum of income across records (2025 USD)"
        if value_col == "income_adj"
        else f"Sum of {_measure_label(value_col).lower()} across records"
    )
    ax.set(
        xlabel="Year",
        ylabel=ylabel,
        title="Annual income total across records by population group",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    return fig


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := required.difference(frame.columns):
        raise KeyError(f"{label} is missing: " + ", ".join(sorted(missing)))


def plot_distribution_regime_fits(
    fits: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    years: Iterable[int] | None = None,
    max_years_per_page: int = 6,
    panel_size: tuple[float, float] = (6.0, 3.2),
):
    """Plot annual dual-panel fits using only persisted analytical datasets."""
    fit_columns = {
        "year",
        "cutoff_normalized",
        "cutoff_income_adj",
        "pareto_alpha",
        "fit_status",
    }
    curve_columns = {
        "year",
        "income_normalized",
        "empirical_ccdf_percent",
        "gompertz_transform",
        "regime",
        "gompertz_fitted_transform",
        "pareto_fitted_ccdf_percent",
    }
    _require_columns(fits, fit_columns, "Regime fits")
    _require_columns(curves, curve_columns, "Regime curves")
    if max_years_per_page < 1:
        raise ValueError("max_years_per_page must be at least 1.")

    selected = _years(fits, years)
    pages = [selected[i : i + max_years_per_page] for i in range(0, len(selected), max_years_per_page)]
    figures = []
    for page in pages:
        fig, axes = plt.subplots(
            len(page),
            2,
            figsize=(panel_size[0] * 2, panel_size[1] * len(page)),
            squeeze=False,
        )
        for row, year in enumerate(page):
            left, right = axes[row]
            fit = fits.loc[fits["year"] == year].iloc[0]
            annual = curves.loc[curves["year"] == year].sort_values("income_normalized")
            cutoff = pd.to_numeric(pd.Series([fit["cutoff_normalized"]]), errors="coerce").iloc[0]
            valid_fit = str(fit["fit_status"]) != "no_valid_fit"
            if not valid_fit or annual.empty or not np.isfinite(cutoff):
                for ax, title in ((left, "Gompertz body"), (right, "Pareto tail")):
                    ax.text(0.5, 0.5, f"No valid fit\n{fit['fit_status']}", ha="center", va="center")
                    ax.set(title=f"PNAD {year} — {title}")
                    ax.set_axis_off()
                continue

            body = annual.loc[annual["regime"] == "gompertz_body"].dropna(
                subset=["income_normalized", "gompertz_transform"]
            )
            tail = annual.loc[annual["regime"] == "pareto_tail"].dropna(
                subset=["income_normalized", "empirical_ccdf_percent"]
            )
            left.scatter(
                body["income_normalized"],
                body["gompertz_transform"],
                s=12,
                color="#4C78A8",
                alpha=0.75,
                label="Empirical",
            )
            left.plot(
                body["income_normalized"],
                body["gompertz_fitted_transform"],
                color="#E17C05",
                linewidth=1.8,
                label="Gompertz LS",
            )
            left.axvline(cutoff, color="#3F3F3F", linestyle="--", linewidth=1.1, label="Cutoff")
            left.set(
                title=f"PNAD {year} — Gompertz body",
                xlabel="Normalized income x = income / annual mean",
                ylabel="ln[ln F(x)]  (F in %)",
            )
            left.grid(True, alpha=0.22)
            left.legend(frameon=False, fontsize="small")

            right.loglog(
                tail["income_normalized"],
                tail["empirical_ccdf_percent"],
                marker="o",
                markersize=2.8,
                linestyle="none",
                color="#4C78A8",
                alpha=0.75,
                label="Empirical tail",
            )
            right.loglog(
                tail["income_normalized"],
                tail["pareto_fitted_ccdf_percent"],
                color="#E17C05",
                linewidth=1.8,
                label="Pareto LS",
            )
            right.axvline(cutoff, color="#3F3F3F", linestyle="--", linewidth=1.1, label="Cutoff")
            right.set(
                title=f"PNAD {year} — Pareto tail (CCDF alpha={fit['pareto_alpha']:.2f})",
                xlabel="Normalized income x (log scale)",
                ylabel="F(x), percent (log scale)",
            )
            right.grid(True, which="both", alpha=0.22)
            right.legend(frameon=False, fontsize="small")
            if fit["fit_status"] != "ok_interior":
                for ax in (left, right):
                    ax.text(
                        0.99,
                        0.03,
                        str(fit["fit_status"]).replace("_", " "),
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize="x-small",
                        color="#555555",
                    )
            left.text(
                0.01,
                0.03,
                f"cutoff = {cutoff:.2f} mean incomes\n({fit['cutoff_income_adj']:.0f} in 2025 USD)",
                transform=left.transAxes,
                ha="left",
                va="bottom",
                fontsize="x-small",
                color="#555555",
            )
        fig.suptitle(f"Annual Gompertz-Pareto regime fits — {page[0]}–{page[-1]}", y=1.001)
        fig.tight_layout()
        figures.append(fig)
    return figures


def _plot_regime_history(
    fits: pd.DataFrame,
    value_col: str,
    *,
    ylabel: str,
    title: str,
    figsize: tuple[float, float] = (11, 5.5),
):
    _require_columns(fits, {"year", value_col, "fit_status"}, "Regime fits")
    frame = fits.sort_values("year").copy()
    valid = frame["fit_status"].astype(str).ne("no_valid_fit")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").where(valid)
    full_years = np.arange(int(frame["year"].min()), int(frame["year"].max()) + 1)
    fig, ax = plt.subplots(figsize=figsize)
    _annual_line_with_gaps(
        ax,
        frame[["year", value_col]],
        value_col,
        years=full_years,
        label=title,
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        color="#4C78A8",
    )
    ax.set(xlabel="Year", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    _transition(ax)
    fig.tight_layout()
    return fig


def plot_gompertz_parameter_history(fits: pd.DataFrame, figsize=(11, 5.5)):
    """Plot the annual Gompertz slope from the persisted fit summary."""
    return _plot_regime_history(
        fits,
        "gompertz_B",
        ylabel="Gompertz slope B (normalized-income scale)",
        title="Normalized Gompertz body parameter B",
        figsize=figsize,
    )


def plot_pareto_alpha_history(fits: pd.DataFrame, figsize=(11, 5.5)):
    """Plot the annual Pareto CCDF exponent from the persisted fit summary."""
    return _plot_regime_history(
        fits,
        "pareto_alpha",
        ylabel="Pareto CCDF exponent alpha",
        title="Pareto upper-tail CCDF exponent",
        figsize=figsize,
    )


def plot_distribution_cutoff_history(fits: pd.DataFrame, figsize=(11, 5.5)):
    """Plot the annual Gompertz-Pareto transition income."""
    return _plot_regime_history(
        fits,
        "cutoff_normalized",
        ylabel="Cutoff / annual mean income",
        title="Normalized Gompertz-Pareto distribution cutoff",
        figsize=figsize,
    )


def plot_regime_r2_history(fits: pd.DataFrame, figsize=(11, 5.5)):
    """Plot annual Gompertz and Pareto R-squared values with their means."""
    columns = {"year", "gompertz_r2", "pareto_r2", "fit_status"}
    _require_columns(fits, columns, "Regime fits")
    frame = fits.sort_values("year").copy()
    valid = frame["fit_status"].astype(str).ne("no_valid_fit")
    for column in ("gompertz_r2", "pareto_r2"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").where(valid)
    if frame[["gompertz_r2", "pareto_r2"]].notna().sum().eq(0).any():
        raise ValueError("Valid Gompertz and Pareto R-squared values are required.")

    full_years = np.arange(int(frame["year"].min()), int(frame["year"].max()) + 1)
    series = (
        ("gompertz_r2", "Gompertz R²", "#4C78A8"),
        ("pareto_r2", "Pareto R²", "#F28E2B"),
    )
    fig, ax = plt.subplots(figsize=figsize)
    for column, label, color in series:
        mean = float(frame[column].mean())
        _annual_line_with_gaps(
            ax,
            frame[["year", column]],
            column,
            years=full_years,
            label=label,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=color,
        )
        ax.axhline(
            mean,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"{label} mean ({mean:.3f})",
        )

    finite_r2 = frame[["gompertz_r2", "pareto_r2"]].to_numpy(float)
    finite_r2 = finite_r2[np.isfinite(finite_r2)]
    minimum_r2 = float(finite_r2.min())
    lower_limit = 0.0 if minimum_r2 >= 0 else minimum_r2 - 0.05 * (1.0 - minimum_r2)
    ax.set(
        xlabel="Year",
        ylabel="Coefficient of determination (R²)",
        title="Gompertz and Pareto regression R²",
        ylim=(lower_limit, 1),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
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
        return x[mask], gompertz_transform(probability[mask])
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
    ncols=DEFAULT_INEQUALITY_NCOLS,
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
