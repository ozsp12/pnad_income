import numpy as np

from pnad_income.distributions import compute_ccdf


def test_ccdf_keeps_zero_income_in_denominator():
    table = compute_ccdf([0, 1, 2], xmin=1, xmax=2, base=2, scale="probability")
    assert np.isclose(table.loc[0, "ccdf"], 2 / 3)
    assert table.loc[0, "population_n"] == 3


def test_percent_scale():
    table = compute_ccdf([0, 1, 2], xmin=1, xmax=2, base=2, scale="percent")
    assert np.isclose(table.loc[0, "ccdf"], 200 / 3)
