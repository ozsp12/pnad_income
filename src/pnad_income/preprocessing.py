"""Cleaning and monetary-standardization functions for PNAD income data."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


# Legacy upper-income thresholds recovered from the historical ``pnad.py``
# workflow.  They are intentionally disabled by default and are retained only
# for explicit sensitivity analyses.
LEGACY_MANUAL_OUTLIER_CUTS: dict[int, float] = {
    1976: 4_975_956.36618934,
    1977: 396_446.78499665,
    1978: 257_615.93691339,
    1979: 151_364.81112462,
}


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce a survey column to numeric values while preserving missingness."""
    return pd.to_numeric(series, errors="coerce")


def standardize_income_frame(raw: pd.DataFrame, spec: pd.Series) -> pd.DataFrame:
    """Convert one raw annual frame to the common analytical schema."""
    df = raw.copy()
    missing_code = (
        float(spec["missing_income_code"])
        if pd.notna(spec["missing_income_code"])
        else None
    )

    df["income_raw"] = _clean_numeric(df["income_raw"])
    if missing_code is not None:
        df = df.loc[df["income_raw"] != missing_code].copy()

    if bool(spec["divide_by_household_size"]):
        if "household_size" not in df.columns:
            raise KeyError("household_size is required for this survey year.")
        df["household_size"] = _clean_numeric(df["household_size"])
        df = df.loc[df["household_size"] > 0].copy()
        df["income"] = df["income_raw"] / df["household_size"]
    else:
        df["income"] = df["income_raw"]

    # Effective income remains a separate measure and is never substituted for income.
    if "income_effective_raw" in df.columns:
        df["income_effective_raw"] = _clean_numeric(df["income_effective_raw"])
        if missing_code is not None:
            df.loc[df["income_effective_raw"] == missing_code, "income_effective_raw"] = np.nan
        df["income_effective"] = df["income_effective_raw"]

    keep = ["income"]
    if "income_effective" in df.columns:
        keep.append("income_effective")
    out = df[keep].copy()
    out.insert(0, "year", int(spec["year"]))
    return out.reset_index(drop=True)


def apply_manual_outlier_cuts(
    df: pd.DataFrame,
    enabled: bool = False,
    cuts: Mapping[int, float] | None = None,
    *,
    year_col: str = "year",
    value_col: str = "income",
) -> pd.DataFrame:
    """Optionally apply the legacy 1976--1979 upper-income cuts.

    Parameters
    ----------
    df:
        Analytical panel containing survey year and the pre-adjustment income
        measure.
    enabled:
        If ``False`` (default), return an unchanged copy.  This keeps the
        canonical analysis untrimmed.  If ``True``, observations above each
        configured year-specific threshold are removed.
    cuts:
        Optional replacement mapping ``{year: upper_income_threshold}``.  When
        omitted, :data:`LEGACY_MANUAL_OUTLIER_CUTS` is used.
    year_col, value_col:
        Column names used to identify the survey year and income measure.

    Notes
    -----
    The thresholds originate in the legacy ``pnad.py`` workflow.  They are not
    inferred statistically by this function and should therefore be interpreted
    as a reproducibility/sensitivity option rather than a default cleaning rule.
    """
    out = df.copy()
    if not enabled:
        return out
    missing = {year_col, value_col}.difference(out.columns)
    if missing:
        raise KeyError("Manual outlier filtering requires: " + ", ".join(sorted(missing)))

    active = LEGACY_MANUAL_OUTLIER_CUTS if cuts is None else dict(cuts)
    normalized: dict[int, float] = {}
    for year, threshold in active.items():
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Outlier thresholds must be finite positive values.")
        normalized[int(year)] = threshold

    years = pd.to_numeric(out[year_col], errors="coerce")
    values = pd.to_numeric(out[value_col], errors="coerce")
    keep = np.ones(len(out), dtype=bool)
    for year, threshold in normalized.items():
        keep &= ~((years == year) & (values > threshold)).to_numpy()
    return out.loc[keep].reset_index(drop=True)


def adjust_income_to_2025(
    df: pd.DataFrame,
    income_columns: tuple[str, ...] = ("income", "income_effective"),
) -> pd.DataFrame:
    """Create 2025-reference income columns from year-level monetary factors."""
    required = {"exchange", "inflation_to_2025"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError("Monetary adjustment requires: " + ", ".join(sorted(missing)))

    out = df.copy()
    exchange = pd.to_numeric(out["exchange"], errors="coerce").where(lambda x: x > 0)
    inflation = pd.to_numeric(out["inflation_to_2025"], errors="coerce")

    for column in income_columns:
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            out[f"{column}_adj"] = (values / exchange) * inflation
    return out
