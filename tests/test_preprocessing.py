import numpy as np
import pandas as pd

from data import adjust_income_to_2025, standardize_income_frame


def test_household_size_standardization():
    raw = pd.DataFrame({"income_raw": [100, 200], "household_size": [2, 4]})
    spec = pd.Series({"year": 2001, "missing_income_code": 999999999999, "divide_by_household_size": True})
    out = standardize_income_frame(raw, spec)
    assert np.allclose(out["income"], [50, 50])


def test_standardization_removes_declared_raw_sentinel():
    raw = pd.DataFrame({"income_raw": [100, 999999, 200]})
    spec = pd.Series({"year": 2001, "missing_income_code": 999999, "divide_by_household_size": False})
    out = standardize_income_frame(raw, spec)
    assert out["income"].tolist() == [100, 200]


def test_adjustment_uses_metadata_columns():
    df = pd.DataFrame({"income": [100.0], "exchange": [2.0], "inflation_to_2025": [3.0]})
    out = adjust_income_to_2025(df)
    assert np.isclose(out.loc[0, "income_adj"], 150.0)
