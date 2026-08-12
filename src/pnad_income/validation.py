"""External validation utilities for annual PNAD inequality estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_REFERENCE_COLUMNS = {"year", "gini", "source"}


def load_gini_reference(path: str | Path) -> pd.DataFrame:
    """Load a documented external Gini reference table.

    The table must contain ``year``, ``gini`` and ``source``.  Optional columns
    such as ``indicator``, ``url``, ``access_date`` and ``notes`` are preserved.
    Gini values may be supplied either on [0, 1] or [0, 100]; values above one
    are converted to the [0, 1] scale.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_REFERENCE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("Gini reference is missing: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["gini"] = pd.to_numeric(frame["gini"], errors="coerce")
    frame.loc[frame["gini"] > 1.0, "gini"] = frame.loc[frame["gini"] > 1.0, "gini"] / 100.0
    invalid = frame["gini"].notna() & ~frame["gini"].between(0.0, 1.0)
    if invalid.any():
        raise ValueError("Gini reference values must lie on [0, 1] or [0, 100].")
    return frame.sort_values(["source", "year"]).reset_index(drop=True)


def combine_gini_references(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate multiple documented external Gini tables."""
    frames = [load_gini_reference(path) for path in paths]
    if not frames:
        return pd.DataFrame(columns=["year", "gini", "source"])
    return pd.concat(frames, ignore_index=True).sort_values(["source", "year"]).reset_index(drop=True)


def compare_gini_series(
    summary: pd.DataFrame,
    references: pd.DataFrame,
    *,
    calculated_col: str = "income_gini",
) -> pd.DataFrame:
    """Align calculated PNAD Gini values with external annual references."""
    required_summary = {"year", calculated_col}
    missing_summary = required_summary.difference(summary.columns)
    if missing_summary:
        raise KeyError("Summary is missing: " + ", ".join(sorted(missing_summary)))
    missing_reference = REQUIRED_REFERENCE_COLUMNS.difference(references.columns)
    if missing_reference:
        raise KeyError("References are missing: " + ", ".join(sorted(missing_reference)))

    calculated = summary[["year", calculated_col]].rename(columns={calculated_col: "gini_calculated"})
    merged = references.merge(calculated, on="year", how="inner", validate="many_to_one")
    merged["difference"] = merged["gini_calculated"] - merged["gini"]
    merged["absolute_difference"] = merged["difference"].abs()
    return merged.sort_values(["source", "year"]).reset_index(drop=True)


def gini_validation_statistics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize agreement between calculated and external Gini series."""
    if comparison.empty:
        return pd.DataFrame(columns=["source", "n", "mean_difference", "mae", "rmse", "correlation"])
    rows = []
    for source, group in comparison.groupby("source", sort=True):
        diff = group["difference"].to_numpy(dtype=float)
        calculated = group["gini_calculated"].to_numpy(dtype=float)
        reference = group["gini"].to_numpy(dtype=float)
        finite = np.isfinite(diff) & np.isfinite(calculated) & np.isfinite(reference)
        diff = diff[finite]
        calculated = calculated[finite]
        reference = reference[finite]
        correlation = (
            float(np.corrcoef(calculated, reference)[0, 1])
            if diff.size > 1 and np.std(calculated) > 0 and np.std(reference) > 0
            else np.nan
        )
        rows.append({
            "source": source,
            "n": int(diff.size),
            "mean_difference": float(np.mean(diff)) if diff.size else np.nan,
            "mae": float(np.mean(np.abs(diff))) if diff.size else np.nan,
            "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan,
            "correlation": correlation,
        })
    return pd.DataFrame(rows)
