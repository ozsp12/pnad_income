from __future__ import annotations

from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_METADATA_PATH = REPOSITORY_ROOT / "config" / "pnad_metadata.csv"


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Load the project metadata table and normalize nullable numeric columns."""
    metadata_path = Path(path) if path is not None else DEFAULT_METADATA_PATH
    df = pd.read_csv(metadata_path)
    numeric = [
        "income_start", "income_width", "household_size_start",
        "household_size_width", "effective_income_start",
        "effective_income_width", "missing_income_code", "exchange",
        "price_index", "inflation_to_2025",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["available"] = df["available"].astype(bool)
    df["divide_by_household_size"] = df["divide_by_household_size"].astype(bool)
    return df.sort_values("year").reset_index(drop=True)


def metadata_for_year(year: int, metadata: pd.DataFrame | None = None) -> pd.Series:
    """Return one metadata row for a survey year."""
    md = load_metadata() if metadata is None else metadata
    rows = md.loc[md["year"] == int(year)]
    if rows.empty:
        raise KeyError(f"Year {year} is absent from metadata.")
    return rows.iloc[0]
