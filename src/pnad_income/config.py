"""Configuration and metadata access for the PNAD income project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_METADATA_PATH = REPOSITORY_ROOT / "metadata" / "pnad_metadata.csv"


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Load and normalize the canonical year-level metadata table."""
    metadata_path = Path(path) if path is not None else DEFAULT_METADATA_PATH
    df = pd.read_csv(metadata_path)

    numeric_columns = [
        "income_start",
        "income_width",
        "household_size_start",
        "household_size_width",
        "effective_income_start",
        "effective_income_width",
        "missing_income_code",
        "exchange",
        "price_index",
        "inflation_to_2025",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("available", "divide_by_household_size"):
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .astype("boolean")
        )

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    return df.sort_values("year").reset_index(drop=True)


def metadata_for_year(year: int, metadata: pd.DataFrame | None = None) -> pd.Series:
    """Return the unique metadata row associated with one survey year."""
    md = load_metadata() if metadata is None else metadata
    rows = md.loc[md["year"] == int(year)]
    if rows.empty:
        raise KeyError(f"Year {year} is absent from metadata.")
    if len(rows) > 1:
        raise ValueError(f"Metadata contains duplicate rows for year {year}.")
    return rows.iloc[0]
