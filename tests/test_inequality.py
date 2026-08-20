import numpy as np
import pandas as pd

from analysis import (
    annual_income_group_totals,
    extended_inequality_statistics,
    gini,
    income_group_totals,
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


def test_income_group_totals_are_absolute_and_close_to_observed_total():
    groups = income_group_totals(np.arange(1.0, 101.0))
    assert np.isclose(groups["p80"], np.arange(1.0, 81.0).sum())
    assert np.isclose(groups["p99"], np.arange(81.0, 100.0).sum())
    assert np.isclose(groups["p100"], 100.0)
    assert np.isclose(groups["p80"] + groups["p99"] + groups["p100"], groups["total"])


def test_annual_income_group_totals_keep_yearly_absolute_scale():
    frame = pd.DataFrame(
        {"year": [2020] * 100 + [2021] * 100, "income": [*range(1, 101), *range(2, 202, 2)]}
    )
    annual = annual_income_group_totals(frame)
    assert annual["year"].tolist() == [2020, 2021]
    assert np.allclose(annual[["p80", "p99", "p100"]].sum(axis=1), annual["total"])
    assert np.isclose(annual.loc[1, "total"], 2 * annual.loc[0, "total"])


def test_extended_statistics_contains_required_metrics():
    metrics = extended_inequality_statistics([0, 1, 2, 7])
    assert {"gini", "pietra", "k", "zanardi", "legacy_z", "top_10_share", "top_1_share", "top_0_1_share"} <= set(metrics)
    assert 0 <= metrics["gini"] <= 1
    assert 0 <= metrics["pietra"] <= 1
    assert 0 <= metrics["k"] <= 1
