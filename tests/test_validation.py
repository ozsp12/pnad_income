import numpy as np
import pandas as pd

from analysis import compare_gini_series, gini_validation_statistics, load_gini_reference


def test_load_gini_reference_normalizes_project_wide_format(tmp_path):
    path = tmp_path / "references.csv"
    pd.DataFrame(
        {
            "ano": [2023, 2024],
            "ipea": [0.517, 0.504],
            "banco_mundial": [51.5, np.nan],
        }
    ).to_csv(path, index=False)
    references = load_gini_reference(path)
    assert set(references["source"]) == {"IPEA", "Banco Mundial"}
    assert references.loc[references["source"] == "Banco Mundial", "gini"].dropna().iloc[0] == 0.515
    assert len(references) == 4


def test_compare_gini_series_aligns_by_year_and_source():
    summary = pd.DataFrame({"year": [2000, 2001], "income_gini": [0.55, 0.53]})
    references = pd.DataFrame({"year": [2000, 2001], "gini": [0.56, 0.52], "source": ["Reference", "Reference"]})
    comparison = compare_gini_series(summary, references)
    assert len(comparison) == 2
    assert np.allclose(comparison["difference"], [-0.01, 0.01])


def test_gini_validation_statistics_returns_error_metrics():
    comparison = pd.DataFrame({
        "source": ["Reference", "Reference"],
        "gini_calculated": [0.55, 0.53],
        "gini": [0.56, 0.52],
        "difference": [-0.01, 0.01],
        "absolute_difference": [0.01, 0.01],
    })
    stats = gini_validation_statistics(comparison)
    assert stats.loc[0, "n"] == 2
    assert np.isclose(stats.loc[0, "mae"], 0.01)
    assert np.isclose(stats.loc[0, "rmse"], 0.01)
