"""Scientific visualization utilities for PNAD income-distribution analysis.

The plotting API supports individual figures selected with ``year=...`` and
small-multiple figures selected with ``years=[...]`` plus explicit ``nrows`` x
``ncols`` layouts.  Grid functions paginate automatically when necessary.
"""

from __future__ import annotations

from math import ceil
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inequality import extended_inequality_statistics, lorenz_curve

DEFAULT_NCOLS = 4
DEFAULT_MAX_PANELS = 24


def _available_years(df: pd.DataFrame, year_col: str = "year") -> list[int]:
    if year_col not in df.columns:
        raise KeyError(f"Column '{year_col}' is absent from the data.")
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


def _select_years(df: pd.DataFrame, years: Iterable[int] | None, year_col: str = "year") -> list[int]:
    available = _available_years(df, year_col=year_col)
    if years is None:
        return available
    selected = list(dict.fromkeys(int(year) for year in years))
    if not selected:
        raise ValueError("years must contain at least one survey year.")
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError("Requested survey years are absent from the data: " + ", ".join(map(str, missing)))
    return selected


def _validate_year(df: pd.DataFrame, year: int, year_col: str = "year") -> int:
    return _select_years(df, [int(year)], year_col=year_col)[0]


def _resolve_grid(n_items: int, nrows: int | None, ncols: int | None, max_panels: int | None) -> tuple[int, int, int, bool]:
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
        return nrows, ncols, nrows * ncols, True
    target = min(n_items, max_panels or DEFAULT_MAX_PANELS)
    if ncols is not None:
        rows, cols = ceil(target / ncols), ncols
    elif nrows is not None:
        rows, cols = nrows, ceil(target / nrows)
    else:
        cols = min(DEFAULT_NCOLS, target)
        rows = ceil(target / cols)
    return rows, cols, rows * cols, fixed_rows


def _year_pages(years: list[int], capacity: int) -> list[list[int]]:
    return [years[i:i + capacity] for i in range(0, len(years), capacity)]


def _grid_axes(page_size: int, nrows: int, ncols: int, panel_size=(4.0, 3.0), fixed_rows=True):
    rows = nrows if fixed_rows else max(1, ceil(page_size / ncols))
    cols = ncols if fixed_rows else min(ncols, page_size)
    if not fixed_rows:
        rows = max(1, ceil(page_size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(panel_size[0] * cols, panel_size[1] * rows), squeeze=False)
    return fig, axes.ravel()


def _finite_values(df: pd.DataFrame, value_col: str, year: int) -> pd.Series:
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    _validate_year(df, year)
    values = pd.to_numeric(df.loc[df["year"] == int(year), value_col], errors="coerce")
    return values[np.isfinite(values)]


def plot_gini_evolution(summary, value_col="income", years=None, figsize=(10.0, 5.0)):
    column = f"{value_col}_gini"
    if column not in summary.columns:
        raise KeyError(f"Column '{column}' is absent from the summary table.")
    selected = _select_years(summary, years)
    frame = summary.loc[summary["year"].isin(selected), ["year", column]].dropna().sort_values("year")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frame["year"], frame[column], marker="o", markersize=4, linewidth=1.2)
    ax.set(xlabel="Year", ylabel="Gini coefficient", title=f"Annual Gini coefficient: {value_col}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_top_income_shares(summary, value_col="income", years=None, figsize=(10.0, 5.5)):
    """Plot annual income shares held by the richest 10%, 1%, and 0.1%."""
    columns = {
        "Top 10%": f"{value_col}_top_10_share",
        "Top 1%": f"{value_col}_top_1_share",
        "Top 0.1%": f"{value_col}_top_0_1_share",
    }
    missing = set(columns.values()).difference(summary.columns)
    if missing:
        raise KeyError("Summary is missing: " + ", ".join(sorted(missing)))
    selected = _select_years(summary, years)
    frame = summary.loc[summary["year"].isin(selected)].sort_values("year")
    fig, ax = plt.subplots(figsize=figsize)
    for label, column in columns.items():
        ax.plot(frame["year"], 100 * frame[column], marker="o", markersize=3, label=label)
    ax.set(xlabel="Year", ylabel="Share of aggregate income [%]", title="Top-income concentration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_extended_inequality_evolution(summary, value_col="income", years=None, figsize=(10.0, 5.5)):
    """Plot the longitudinal Pietra, k, and legacy Z statistics."""
    columns = {
        "Pietra": f"{value_col}_pietra",
        "k": f"{value_col}_k",
        "Z": f"{value_col}_zanardi",
    }
    missing = set(columns.values()).difference(summary.columns)
    if missing:
        raise KeyError("Summary is missing: " + ", ".join(sorted(missing)))
    selected = _select_years(summary, years)
    frame = summary.loc[summary["year"].isin(selected)].sort_values("year")
    fig, ax = plt.subplots(figsize=figsize)
    for label, column in columns.items():
        ax.plot(frame["year"], frame[column], marker="o", markersize=3, label=label)
    ax.set(xlabel="Year", ylabel="Index value", title="Extended Lorenz-curve inequality measures")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gini_validation(summary, references, value_col="income", figsize=(10.0, 5.5)):
    """Plot calculated PNAD Gini alongside documented external reference series."""
    column = f"{value_col}_gini"
    if column not in summary.columns:
        raise KeyError(f"Column '{column}' is absent from the summary table.")
    required = {"year", "gini", "source"}
    missing = required.difference(references.columns)
    if missing:
        raise KeyError("References are missing: " + ", ".join(sorted(missing)))
    fig, ax = plt.subplots(figsize=figsize)
    frame = summary[["year", column]].dropna().sort_values("year")
    ax.plot(frame["year"], frame[column], marker="o", markersize=3, label="Calculated PNAD")
    for source, group in references.groupby("source", sort=True):
        group = group.sort_values("year")
        ax.plot(group["year"], group["gini"], marker="o", markersize=3, label=str(source))
    ax.set(xlabel="Year", ylabel="Gini coefficient", title="Gini validation against external references")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_histogram(df, year, value_col="income", bins=60, yscale="linear", figsize=(7.0, 4.5)):
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    values = _finite_values(df, value_col, year)
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


def plot_histogram_grid(df, value_col="income", years=None, bins=60, yscale="linear", nrows=None, ncols=None, max_panels=DEFAULT_MAX_PANELS, panel_size=(4.0, 3.0)):
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' is absent from the data.")
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    selected = _select_years(df, years)
    rows, cols, capacity, fixed = _resolve_grid(len(selected), nrows, ncols, max_panels)
    grouped = {int(y): g for y, g in df.groupby("year", sort=True)}
    figures = []
    for page in _year_pages(selected, capacity):
        fig, axes = _grid_axes(len(page), rows, cols, panel_size, fixed)
        for ax, year in zip(axes, page):
            values = pd.to_numeric(grouped[year][value_col], errors="coerce")
            values = values[np.isfinite(values)]
            if values.empty:
                ax.text(0.5, 0.5, "No finite observations", ha="center", va="center")
            else:
                ax.hist(values, bins=bins)
                ax.set_yscale(yscale)
            ax.set(title=f"PNAD {year}", xlabel="Income", ylabel="Frequency" + (" (log)" if yscale == "log" else ""))
            ax.grid(True, alpha=0.25)
        for ax in axes[len(page):]: ax.set_visible(False)
        fig.suptitle(f"Annual histograms: {value_col} — {page[0]}–{page[-1]} ({yscale} frequency)", y=1.002)
        fig.tight_layout(); figures.append(fig)
    return figures


def _ccdf_coordinates(frame, transform):
    x = frame["bin"].to_numpy(dtype=float); y = frame["ccdf"].to_numpy(dtype=float)
    if transform == "linear":
        mask = np.isfinite(x) & np.isfinite(y); return x[mask], y[mask]
    if transform == "loglog":
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0); return x[mask], y[mask]
    if transform == "double_log":
        mask = np.isfinite(x) & np.isfinite(y) & (y > 1.0); return x[mask], np.log(np.log(y[mask]))
    raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")


def _ccdf_frame(ccdf, year, measure):
    if "measure" not in ccdf.columns: raise KeyError("Column 'measure' is absent from the CCDF table.")
    frame = ccdf.loc[ccdf["measure"] == measure].copy()
    if frame.empty: raise ValueError(f"No CCDF rows are available for measure '{measure}'.")
    _validate_year(frame, year)
    return frame.loc[frame["year"] == int(year)].copy()


def plot_ccdf(ccdf, year, measure="income", transform="linear", figsize=(7.0, 4.5)):
    frame = _ccdf_frame(ccdf, year, measure); x, y = _ccdf_coordinates(frame, transform)
    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(x, y) if transform == "loglog" else ax.plot(x, y)
    ax.set(title=f"PNAD {int(year)} — {measure}", xlabel="Income", ylabel="ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]")
    ax.grid(True, alpha=0.3); fig.tight_layout(); return fig


def plot_ccdf_grid(ccdf, measure="income", years=None, transform="linear", nrows=None, ncols=None, max_panels=DEFAULT_MAX_PANELS, panel_size=(4.0, 3.0)):
    if transform not in {"linear", "loglog", "double_log"}: raise ValueError("transform must be 'linear', 'loglog', or 'double_log'.")
    frame = ccdf.loc[ccdf["measure"] == measure].copy()
    if frame.empty: raise ValueError(f"No CCDF rows are available for measure '{measure}'.")
    selected = _select_years(frame, years); rows, cols, capacity, fixed = _resolve_grid(len(selected), nrows, ncols, max_panels)
    grouped = {int(y): g for y, g in frame.groupby("year", sort=True)}; figures=[]
    for page in _year_pages(selected, capacity):
        fig, axes = _grid_axes(len(page), rows, cols, panel_size, fixed)
        for ax, year in zip(axes, page):
            x, y = _ccdf_coordinates(grouped[year], transform)
            ax.loglog(x, y) if transform == "loglog" else ax.plot(x, y)
            ax.set(title=f"PNAD {year}", xlabel="Income", ylabel="ln[ln(CCDF [%])]" if transform == "double_log" else "CCDF [%]"); ax.grid(True, alpha=0.3)
        for ax in axes[len(page):]: ax.set_visible(False)
        label={"linear":"linear axes","loglog":"log-log axes","double_log":"legacy ln[ln(CCDF)] transform"}[transform]
        fig.suptitle(f"Annual CCDF: {measure} — {page[0]}–{page[-1]} ({label})", y=1.002); fig.tight_layout(); figures.append(fig)
    return figures


def plot_ccdf_selected_years(ccdf, measure, years, transform="loglog", figsize=(7.5,5.0)):
    frame=ccdf.loc[ccdf["measure"]==measure].copy(); selected=_select_years(frame,years); fig,ax=plt.subplots(figsize=figsize)
    for year in selected:
        x,y=_ccdf_coordinates(frame.loc[frame["year"]==year],transform); ax.loglog(x,y,label=str(year)) if transform=="loglog" else ax.plot(x,y,label=str(year))
    ax.set(xlabel="Income",ylabel="ln[ln(CCDF [%])]" if transform=="double_log" else "CCDF [%]",title=f"{measure}: selected survey years"); ax.grid(True,alpha=0.3); ax.legend(title="Year"); fig.tight_layout(); return fig


def plot_measure_comparison(ccdf, year, measures=("income","income_adj"), transform="loglog", figsize=(6.5,4.5)):
    available=ccdf.loc[ccdf["measure"].isin(measures)].copy(); _validate_year(available,year); fig,ax=plt.subplots(figsize=figsize); plotted=0
    for measure in measures:
        annual=available.loc[(available["year"]==int(year))&(available["measure"]==measure)]
        if annual.empty: continue
        x,y=_ccdf_coordinates(annual,transform); ax.loglog(x,y,label=measure) if transform=="loglog" else ax.plot(x,y,label=measure); plotted+=1
    if not plotted: raise ValueError(f"No requested measures are available for year {int(year)}.")
    ax.set(title=f"PNAD {int(year)}",xlabel="Income",ylabel="ln[ln(CCDF [%])]" if transform=="double_log" else "CCDF [%]"); ax.grid(True,alpha=0.3); ax.legend(); fig.tight_layout(); return fig


def plot_measure_comparison_grid(ccdf, measures=("income","income_adj"), years=None, transform="loglog", nrows=None, ncols=None, max_panels=DEFAULT_MAX_PANELS, panel_size=(4.0,3.0)):
    available=ccdf.loc[ccdf["measure"].isin(measures)].copy(); selected=_select_years(available,years); rows,cols,capacity,fixed=_resolve_grid(len(selected),nrows,ncols,max_panels); figures=[]
    for page in _year_pages(selected,capacity):
        fig,axes=_grid_axes(len(page),rows,cols,panel_size,fixed)
        for ax,year in zip(axes,page):
            for measure in measures:
                annual=available.loc[(available["year"]==year)&(available["measure"]==measure)]
                if annual.empty: continue
                x,y=_ccdf_coordinates(annual,transform); ax.loglog(x,y,label=measure) if transform=="loglog" else ax.plot(x,y,label=measure)
            ax.set(title=f"PNAD {year}",xlabel="Income",ylabel="ln[ln(CCDF [%])]" if transform=="double_log" else "CCDF [%]"); ax.grid(True,alpha=0.3); ax.legend(fontsize="small")
        for ax in axes[len(page):]: ax.set_visible(False)
        fig.suptitle(f"Annual measure comparison — {page[0]}–{page[-1]} ({transform})",y=1.002); fig.tight_layout(); figures.append(fig)
    return figures


def _draw_extended_lorenz(ax, values, year):
    population,income_share=lorenz_curve(values); metrics=extended_inequality_statistics(values)
    ax.plot(population,income_share); ax.plot([0,1],[0,1],linestyle="--")
    k=metrics["k"]
    if np.isfinite(k):
        lk=float(np.interp(k,population,income_share)); ax.plot([k],[lk],marker="o"); ax.plot([k,k],[0,lk],linestyle=":")
    text=f"G={metrics['gini']:.3f}\nP={metrics['pietra']:.3f}\nk={metrics['k']:.3f}\nZ={metrics['zanardi']:.3f}"
    ax.text(0.04,0.96,text,transform=ax.transAxes,va="top",ha="left",fontsize="small")
    ax.set(xlim=(0,1),ylim=(0,1),title=f"PNAD {year}",xlabel="Population share",ylabel="Income share"); ax.grid(True,alpha=0.25)


def plot_lorenz_curve(df, year, value_col="income", figsize=(5.5,5.0), annotate=False):
    values=_finite_values(df,value_col,year); fig,ax=plt.subplots(figsize=figsize)
    if annotate: _draw_extended_lorenz(ax,values,int(year))
    else:
        population,income_share=lorenz_curve(values); ax.plot(population,income_share,label=str(int(year))); ax.plot([0,1],[0,1],linestyle="--",label="Equality"); ax.set(xlim=(0,1),ylim=(0,1),xlabel="Cumulative population share",ylabel="Cumulative income share",title=f"Lorenz curve: {int(year)}"); ax.grid(True,alpha=0.3); ax.legend()
    fig.tight_layout(); return fig


def plot_lorenz_grid(df, value_col="income", years=None, nrows=None, ncols=None, max_panels=DEFAULT_MAX_PANELS, panel_size=(3.6,3.3), annotate=False):
    if value_col not in df.columns: raise KeyError(f"Column '{value_col}' is absent from the data.")
    selected=_select_years(df,years); rows,cols,capacity,fixed=_resolve_grid(len(selected),nrows,ncols,max_panels); grouped={int(y):g for y,g in df.groupby("year",sort=True)}; figures=[]
    for page in _year_pages(selected,capacity):
        fig,axes=_grid_axes(len(page),rows,cols,panel_size,fixed)
        for ax,year in zip(axes,page):
            values=pd.to_numeric(grouped[year][value_col],errors="coerce")
            if annotate: _draw_extended_lorenz(ax,values,year)
            else:
                population,income_share=lorenz_curve(values); ax.plot(population,income_share); ax.plot([0,1],[0,1],linestyle="--"); ax.set(xlim=(0,1),ylim=(0,1),title=f"PNAD {year}",xlabel="Population share",ylabel="Income share"); ax.grid(True,alpha=0.25)
        for ax in axes[len(page):]: ax.set_visible(False)
        title="Annotated annual Lorenz curves" if annotate else "Annual Lorenz curves"
        fig.suptitle(f"{title}: {value_col} — {page[0]}–{page[-1]}",y=1.002); fig.tight_layout(); figures.append(fig)
    return figures
