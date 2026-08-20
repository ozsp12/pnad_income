"""Tests for exploratory descriptive statistics and sentinel diagnostics."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from descriptive import DescriptiveStatistics


def _frame():
    return pd.DataFrame(
        {
            "year": [2020] * 8 + [2021] * 6,
            "income": [1, 2, 3, 4, 5, 999999, 999999, 999999, 2, 3, 4, 5, 6, 7],
        }
    )


def test_value_frequency_detects_repeated_extreme_value(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "year": [2020, 2021],
            "missing_income_code": [999999, None],
            "available": [True, True],
            "divide_by_household_size": [False, False],
        }
    ).to_csv(metadata, index=False)
    eda = DescriptiveStatistics(_frame(), metadata_path=metadata)
    frequencies = eda.value_frequencies(top_n=10)
    row = frequencies.loc[(frequencies["year"] == 2020) & (frequencies["value"] == 999999)].iloc[0]
    assert int(row["count"]) == 3


def test_candidate_cutoff_is_diagnostic_not_filtering(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "year": [2020, 2021],
            "missing_income_code": [999999, None],
            "available": [True, True],
            "divide_by_household_size": [False, False],
        }
    ).to_csv(metadata, index=False)
    original = _frame()
    eda = DescriptiveStatistics(original, metadata_path=metadata)
    diagnostics = eda.outlier_diagnostics()
    row = diagnostics.loc[diagnostics["year"] == 2020].iloc[0]
    assert row["suggested_cutoff"] == 999999
    assert (original["income"] == 999999).sum() == 3


def test_eda_figures_render():
    eda = DescriptiveStatistics(_frame())
    hist = eda.histogram_pages(max_panels=4, ncols=2)
    box = eda.boxplot_pages(max_panels=4, ncols=2)
    overview = eda.outlier_overview_figure()
    assert hist and box and len(overview.axes) == 1
    [plt.close(fig) for fig in hist + box + [overview]]
