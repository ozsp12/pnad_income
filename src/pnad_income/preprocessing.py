from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def standardize_income_frame(raw: pd.DataFrame, spec: pd.Series) -> pd.DataFrame:
    """Clean one raw annual frame and produce standardized income columns.

    ``income`` is the longitudinal income measure. For years where the original
    code supplied household size separately, income is divided by household size.
    For PNAD Continua 2016-2025, ``income_effective`` preserves VD4020 in addition
    to the habitual measure VD4019 stored in ``income``.
    """
    df = raw.copy()
    missing = float(spec["missing_income_code"]) if pd.notna(spec["missing_income_code"]) else None
    df["income_raw"] = _clean_numeric(df["income_raw"])
    if missing is not None:
        df = df.loc[df["income_raw"] != missing].copy()
    if bool(spec["divide_by_household_size"]):
        df["household_size"] = _clean_numeric(df["household_size"])
        df = df.loc[df["household_size"] > 0].copy()
        df["income"] = df["income_raw"] / df["household_size"]
    else:
        df["income"] = df["income_raw"]
    if "income_effective_raw" in df.columns:
        df["income_effective_raw"] = _clean_numeric(df["income_effective_raw"])
        if missing is not None:
            df.loc[df["income_effective_raw"] == missing, "income_effective_raw"] = np.nan
        df["income_effective"] = df["income_effective_raw"]
    keep = ["income"]
    if "income_effective" in df.columns:
        keep.append("income_effective")
    out = df[keep].copy()
    out.insert(0, "year", int(spec["year"]))
    return out.reset_index(drop=True)


def adjust_income_to_2025(df: pd.DataFrame, income_columns: tuple[str, ...] = ("income", "income_effective")) -> pd.DataFrame:
    """Create 2025-adjusted income columns using metadata exchange and inflation factors."""
    out = df.copy()
    exchange = pd.to_numeric(out["exchange"], errors="coerce")
    inflation = pd.to_numeric(out["inflation_to_2025"], errors="coerce")
    for col in income_columns:
        if col in out.columns:
            out[f"{col}_adj"] = (pd.to_numeric(out[col], errors="coerce") / exchange) * inflation
    return out
