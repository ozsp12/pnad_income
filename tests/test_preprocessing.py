import numpy as np
import pandas as pd

from pnad_income.data import (
    LEGACY_MANUAL_OUTLIER_CUTS,
    adjust_income_to_2025,
    apply_manual_outlier_cuts,
    standardize_income_frame,
)


def test_household_size_standardization():
    raw = pd.DataFrame({"income_raw": [100, 200], "household_size": [2, 4]})
    spec = pd.Series({"year": 2001, "missing_income_code": 999999999999, "divide_by_household_size": True})
    out = standardize_income_frame(raw, spec)
    assert np.allclose(out["income"], [50, 50])


def test_adjustment_uses_metadata_columns():
    df = pd.DataFrame({"income": [100.0], "exchange": [2.0], "inflation_to_2025": [3.0]})
    out = adjust_income_to_2025(df)
    assert np.isclose(out.loc[0, "income_adj"], 150.0)


def test_manual_outlier_cuts_are_disabled_by_default():
    threshold = LEGACY_MANUAL_OUTLIER_CUTS[1976]
    df = pd.DataFrame({"year": [1976, 1976], "income": [1.0, threshold * 2]})
    out = apply_manual_outlier_cuts(df)
    assert len(out) == 2


def test_manual_outlier_cuts_remove_only_values_above_threshold_when_enabled():
    threshold = LEGACY_MANUAL_OUTLIER_CUTS[1976]
    df = pd.DataFrame({"year": [1976, 1976, 1981], "income": [threshold, threshold + 1, threshold * 10]})
    out = apply_manual_outlier_cuts(df, enabled=True)
    assert len(out) == 2
    assert (out["year"] == 1981).sum() == 1
    assert np.isclose(out.loc[out["year"] == 1976, "income"].iloc[0], threshold)
