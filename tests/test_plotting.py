"""Tests for selectable-year and configurable-grid plotting interfaces."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pnad_income.distributions import build_ccdf_by_year
from pnad_income.plotting import (
    plot_ccdf,
    plot_ccdf_grid,
    plot_histogram,
    plot_histogram_grid,
    plot_lorenz_curve,
    plot_lorenz_grid,
)


def _panel():
    return pd.DataFrame({
        "year": [2001, 2001, 2005, 2005, 2010, 2010, 2020, 2020, 2025, 2025],
        "income": [0.0, 10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0, 50.0],
    })


def test_individual_histogram_selects_one_year():
    fig = plot_histogram(_panel(), year=2025)
    assert len(fig.axes) == 1
    assert "2025" in fig.axes[0].get_title()
    plt.close(fig)


def test_individual_ccdf_selects_one_year():
    ccdf = build_ccdf_by_year(_panel(), value_col="income")
    fig = plot_ccdf(ccdf, year=2020, measure="income", transform="loglog")
    assert len(fig.axes) == 1
    assert "2020" in fig.axes[0].get_title()
    plt.close(fig)


def test_individual_lorenz_selects_one_year():
    fig = plot_lorenz_curve(_panel(), year=2010)
    assert len(fig.axes) == 1
    assert "2010" in fig.axes[0].get_title()
    plt.close(fig)


def test_histogram_grid_respects_explicit_rows_and_columns():
    figures = plot_histogram_grid(
        _panel(),
        years=[2001, 2005, 2010, 2020],
        nrows=2,
        ncols=2,
    )
    assert len(figures) == 1
    assert len(figures[0].axes) == 4
    plt.close(figures[0])


def test_grid_paginates_when_selection_exceeds_capacity():
    figures = plot_lorenz_grid(
        _panel(),
        years=[2001, 2005, 2010, 2020, 2025],
        nrows=1,
        ncols=2,
    )
    assert len(figures) == 3
    assert all(len(fig.axes) == 2 for fig in figures)
    for fig in figures:
        plt.close(fig)


def test_ccdf_grid_respects_selected_years_and_shape():
    ccdf = build_ccdf_by_year(_panel(), value_col="income")
    figures = plot_ccdf_grid(
        ccdf,
        measure="income",
        years=[2005, 2020, 2025],
        transform="linear",
        nrows=1,
        ncols=3,
    )
    assert len(figures) == 1
    titles = [ax.get_title() for ax in figures[0].axes]
    assert titles == ["PNAD 2005", "PNAD 2020", "PNAD 2025"]
    plt.close(figures[0])
