import numpy as np

from pnad_income.inequality import gini, lorenz_curve


def test_gini_equal_distribution_is_zero():
    assert np.isclose(gini([1, 1, 1, 1]), 0.0)


def test_lorenz_endpoints():
    p, l = lorenz_curve([0, 1, 2, 3])
    assert np.allclose([p[0], p[-1]], [0, 1])
    assert np.allclose([l[0], l[-1]], [0, 1])
