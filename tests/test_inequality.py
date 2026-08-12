import numpy as np

from pnad_income.inequality import (
    extended_inequality_statistics,
    gini,
    lorenz_curve,
    lorenz_k_index,
    pietra_index,
    top_income_share,
    zanardi_index,
)


def test_gini_equal_distribution_is_zero():
    assert np.isclose(gini([1, 1, 1, 1]), 0.0)


def test_lorenz_endpoints():
    p, l = lorenz_curve([0, 1, 2, 3])
    assert np.allclose([p[0], p[-1]], [0, 1])
    assert np.allclose([l[0], l[-1]], [0, 1])


def test_equal_distribution_extended_indices():
    values = [1, 1, 1, 1]
    assert np.isclose(pietra_index(values), 0.0)
    assert np.isclose(lorenz_k_index(values), 0.5)
    assert np.isclose(zanardi_index(values), 0.0)
    assert np.isclose(top_income_share(values, 0.25), 0.25)


def test_top_income_share_concentrated_distribution():
    values = [0, 0, 0, 100]
    assert np.isclose(top_income_share(values, 0.25), 1.0)
    assert pietra_index(values) > 0.7


def test_extended_statistics_contains_required_metrics():
    metrics = extended_inequality_statistics([0, 1, 2, 7])
    assert {"gini", "pietra", "k", "zanardi", "top_10_share", "top_1_share", "top_0_1_share"} <= set(metrics)
    assert 0 <= metrics["gini"] <= 1
    assert 0 <= metrics["pietra"] <= 1
    assert 0 <= metrics["k"] <= 1
