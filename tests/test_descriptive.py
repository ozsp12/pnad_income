"""Tests for exploratory diagnostics and deterministic trusted-layer construction."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from descriptive import DescriptiveStatistics, IncomeDataCleaner


def _frame():
    return pd.DataFrame(
        {
            "year": [2020] * 9 + [2021] * 6,
            "income": [1, 2, 3, 4, 5, 10_000, 999_999, 999_999, 999_999, 2, 3, 4, 5, 6, 7],
        }
    )


def _metadata(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "year": [2020, 2021],
            "missing_income_code": [999_999, None],
            "available": [True, True],
            "divide_by_household_size": [False, False],
        }
    ).to_csv(path, index=False)
    return path


def test_value_frequency_detects_repeated_sentinel(tmp_path):
    eda = DescriptiveStatistics(_frame(), metadata_path=_metadata(tmp_path))
    frequencies = eda.value_frequencies(top_n=10)
    row = frequencies.loc[(frequencies["year"] == 2020) & (frequencies["value"] == 999_999)].iloc[0]
    assert int(row["count"]) == 3
    sentinels = eda.metadata_sentinel_occurrences()
    assert int(sentinels.loc[sentinels["year"] == 2020, "count"].iloc[0]) == 3


def test_cleaning_flags_are_mutually_exclusive_and_prioritize_sentinel(tmp_path):
    cleaner = IncomeDataCleaner(_frame(), metadata_path=_metadata(tmp_path), method="log_mad", threshold=3.0)
    flagged = cleaner.flagged_frame()
    sentinel = flagged["income"] == 999_999
    assert (flagged.loc[sentinel, "flag_metadata_sentinel"] == 1).all()
    assert (flagged.loc[sentinel, "flag_statistical_outlier"] == 0).all()
    assert not ((flagged["flag_metadata_sentinel"] == 1) & (flagged["flag_statistical_outlier"] == 1)).any()
    assert flagged.loc[flagged["income"] == 10_000, "flag_statistical_outlier"].iloc[0] == 1


def test_sentinels_do_not_contaminate_statistical_threshold(tmp_path):
    metadata = _metadata(tmp_path)
    full = IncomeDataCleaner(_frame(), metadata_path=metadata, method="log_mad", threshold=3.0).thresholds()
    without_sentinel = _frame().loc[_frame()["income"] != 999_999].reset_index(drop=True)
    clean = IncomeDataCleaner(without_sentinel, metadata_path=metadata, method="log_mad", threshold=3.0).thresholds()
    full_cutoff = full.loc[full["year"] == 2020, "statistical_cutoff"].iloc[0]
    clean_cutoff = clean.loc[clean["year"] == 2020, "statistical_cutoff"].iloc[0]
    assert full_cutoff == clean_cutoff


def test_trusted_materialization_removes_flagged_rows(tmp_path):
    cleaner = IncomeDataCleaner(_frame(), metadata_path=_metadata(tmp_path), method="log_mad", threshold=3.0)
    trusted = cleaner.trusted_frame()
    assert 999_999 not in trusted["income"].tolist()
    assert 10_000 not in trusted["income"].tolist()
    assert "flag_metadata_sentinel" not in trusted.columns
    assert "flag_statistical_outlier" not in trusted.columns
    paths = cleaner.materialize_trusted(tmp_path / "trusted")
    assert len(paths) == 2
    assert all(path.name.startswith("pnad_trusted_") for path in paths)
    audit = cleaner.cleaning_audit()
    row = audit.loc[audit["year"] == 2020].iloc[0]
    assert int(row["n_metadata_sentinel"]) == 3
    assert int(row["n_statistical_outlier"]) == 1
    assert int(row["n_trusted"]) == 5


def test_eda_figures_render():
    eda = DescriptiveStatistics(_frame())
    hist = eda.histogram_pages(max_panels=4, ncols=2)
    box = eda.boxplot_pages(max_panels=4, ncols=2)
    overview = eda.outlier_overview_figure()
    comparison = eda.compare_upper_tail_figure(eda)
    assert hist and box and len(overview.axes) == 1 and len(comparison.axes) == 1
    [plt.close(fig) for fig in hist + box + [overview, comparison]]
