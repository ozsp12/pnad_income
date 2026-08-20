import numpy as np
import pandas as pd

from pnad_income.analysis import compare_gini_series, gini_validation_statistics


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
