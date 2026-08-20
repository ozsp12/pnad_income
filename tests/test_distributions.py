import numpy as np

from pnad_income.analysis import compute_ccdf, geometric_thresholds


def test_ccdf_keeps_zero_income_in_denominator():
    table = compute_ccdf([0, 1, 2], xmin=1, xmax=2, base=2, scale="probability")
    assert np.isclose(table.loc[0, "ccdf"], 2 / 3)
    assert table.loc[0, "population_n"] == 3


def test_percent_scale():
    table = compute_ccdf([0, 1, 2], xmin=1, xmax=2, base=2, scale="percent")
    assert np.isclose(table.loc[0, "ccdf"], 200 / 3)


def test_default_scale_is_probability():
    table = compute_ccdf([0, 1, 2], xmin=1, xmax=2, base=2)
    assert np.isclose(table.loc[0, "ccdf"], 2 / 3)


def test_geometric_thresholds_use_exact_multiplier():
    thresholds = geometric_thresholds([1, 100], base=1.05)
    assert np.allclose(thresholds[1:] / thresholds[:-1], 1.05)
    assert thresholds[-1] <= 100


def test_ccdf_matches_original_naive_threshold_counting():
    values = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    thresholds = geometric_thresholds(values, base=1.05)
    expected = np.array([(values >= threshold).mean() for threshold in thresholds])
    actual = compute_ccdf(values, base=1.05, scale="probability")
    np.testing.assert_allclose(actual["bin"].to_numpy(), thresholds)
    np.testing.assert_allclose(actual["ccdf"].to_numpy(), expected)
