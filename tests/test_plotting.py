"""Tests for selectable-year, configurable-grid, and inequality plots."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import build_ccdf_by_year, summary_statistics
from plotting import (
    plot_ccdf,
    plot_ccdf_grid,
    plot_extended_inequality_evolution,
    plot_gini_validation,
    plot_histogram,
    plot_histogram_grid,
    plot_lorenz_curve,
    plot_lorenz_grid,
    plot_top_income_shares,
)


def _panel():
    return pd.DataFrame(
        {
            "year": [2001, 2001, 2005, 2005, 2010, 2010, 2020, 2020, 2025, 2025],
            "income": [0.0, 10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0, 50.0],
        }
    )


def test_individual_histogram_selects_one_year_and_uses_log_frequency():
    fig = plot_histogram(_panel(), year=2025)
    assert len(fig.axes) == 1
    assert "2025" in fig.axes[0].get_title()
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)


def test_individual_ccdf_selects_one_year():
    ccdf = build_ccdf_by_year(_panel(), value_col="income")
    fig = plot_ccdf(ccdf, year=2020, measure="income", transform="loglog")
    assert len(fig.axes) == 1
    assert "2020" in fig.axes[0].get_title()
    plt.close(fig)


def test_gompertz_transform_retains_tail_below_one_percent():
    ccdf = pd.DataFrame(
        {
            "year": [2025] * 4,
            "measure": ["income"] * 4,
            "bin": [1.0, 2.0, 3.0, 4.0],
            "ccdf": [0.5, 0.01, 0.001, 0.0001],
        }
    )
    fig = plot_ccdf(ccdf, year=2025, measure="income", transform="gompertz")
    y = fig.axes[0].lines[0].get_ydata()
    assert len(y) == 4
    assert np.isfinite(y).all()
    assert fig.axes[0].get_ylabel() == "-ln[-ln(S(x))]"
    assert np.all(np.diff(y) < 0)
    plt.close(fig)


def test_individual_lorenz_selects_one_year():
    fig = plot_lorenz_curve(_panel(), year=2010)
    assert len(fig.axes) == 1
    assert "2010" in fig.axes[0].get_title()
    plt.close(fig)


def test_annotated_lorenz_contains_extended_metrics():
    fig = plot_lorenz_curve(_panel(), year=2025, annotate=True)
    text = "\n".join(t.get_text() for t in fig.axes[0].texts)
    assert "G=" in text and "P=" in text and "k=" in text and "Z=" in text
    plt.close(fig)


def test_histogram_grid_respects_explicit_rows_and_columns():
    figures = plot_histogram_grid(_panel(), years=[2001, 2005, 2010, 2020], nrows=2, ncols=2)
    assert len(figures) == 1
    assert len(figures[0].axes) == 4
    assert all(ax.get_yscale() == "log" for ax in figures[0].axes)
    plt.close(figures[0])


def test_grid_paginates_when_selection_exceeds_capacity():
    figures = plot_lorenz_grid(_panel(), years=[2001, 2005, 2010, 2020, 2025], nrows=1, ncols=2)
    assert len(figures) == 3
    assert all(len(fig.axes) == 2 for fig in figures)
    [plt.close(fig) for fig in figures]


def test_ccdf_grid_respects_selected_years_and_shape():
    ccdf = build_ccdf_by_year(_panel(), value_col="income")
    figures = plot_ccdf_grid(
        ccdf,
        measure="income",
        years=[2005, 2020, 2025],
        transform="loglog",
        nrows=1,
        ncols=3,
    )
    assert len(figures) == 1
    assert [ax.get_title() for ax in figures[0].axes] == ["PNAD 2005", "PNAD 2020", "PNAD 2025"]
    plt.close(figures[0])


def test_extended_evolution_plots_render():
    summary = summary_statistics(_panel())
    figures = [plot_top_income_shares(summary), plot_extended_inequality_evolution(summary)]
    assert all(len(fig.axes) == 1 for fig in figures)
    [plt.close(fig) for fig in figures]


def test_gini_validation_plot_renders_external_source():
    summary = summary_statistics(_panel())
    references = pd.DataFrame(
        {"year": [2001, 2005], "gini": [0.4, 0.5], "source": ["Reference", "Reference"]}
    )
    fig = plot_gini_validation(summary, references)
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "Calculated PNAD" in labels and "Reference" in labels
    plt.close(fig)


def test_loglog_ccdf_remains_on_probability_scale():
    ccdf = pd.DataFrame(
        {
            "year": [2025, 2025],
            "measure": ["income", "income"],
            "bin": [1.0, 2.0],
            "ccdf": [1.0, 0.01],
        }
    )
    fig = plot_ccdf(ccdf, year=2025, measure="income", transform="loglog")
    y = fig.axes[0].lines[0].get_ydata()
    assert np.allclose(y, [1.0, 0.01])
    assert fig.axes[0].get_ylabel() == "S(x)"
    plt.close(fig)
