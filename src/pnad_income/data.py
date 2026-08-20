"""Data access, metadata, validation, and preprocessing for the PNAD project."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_METADATA_PATH = REPOSITORY_ROOT / "metadata" / "pnad_metadata.csv"

SUPPORTED_FILE_SUFFIXES = {".parquet", ".csv", ".feather", ".pkl", ".pickle", ".xlsx", ".xls"}
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")
COLUMN_ALIASES = {
    "ano": "year",
    "renda": "income",
    "renda_efetiva": "income_effective",
    "renda_efet": "income_effective",
}

LEGACY_MANUAL_OUTLIER_CUTS: dict[int, float] = {
    1976: 4_975_956.36618934,
    1977: 396_446.78499665,
    1978: 257_615.93691339,
    1979: 151_364.81112462,
}


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical year-level metadata table."""
    df = pd.read_csv(Path(path) if path is not None else DEFAULT_METADATA_PATH)
    numeric = [
        "income_start", "income_width", "household_size_start", "household_size_width",
        "effective_income_start", "effective_income_width", "missing_income_code",
        "exchange", "price_index", "inflation_to_2025",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("available", "divide_by_household_size"):
        df[column] = (
            df[column].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False}).astype("boolean")
        )
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    return df.sort_values("year").reset_index(drop=True)


def metadata_for_year(year: int, metadata: pd.DataFrame | None = None) -> pd.Series:
    """Return the unique metadata row for one survey year."""
    md = load_metadata() if metadata is None else metadata
    rows = md.loc[md["year"] == int(year)]
    if rows.empty:
        raise KeyError(f"Year {year} is absent from metadata.")
    if len(rows) > 1:
        raise ValueError(f"Metadata contains duplicate rows for year {year}.")
    return rows.iloc[0]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for alias, canonical in COLUMN_ALIASES.items():
        if alias not in out.columns:
            continue
        if canonical in out.columns:
            equal = out[alias].eq(out[canonical]) | (out[alias].isna() & out[canonical].isna())
            if not bool(equal.all()):
                raise ValueError(f"Conflicting columns '{alias}' and '{canonical}' are both present.")
            out = out.drop(columns=[alias])
        else:
            out = out.rename(columns={alias: canonical})
    return out


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    readers = {
        ".parquet": pd.read_parquet,
        ".csv": pd.read_csv,
        ".feather": pd.read_feather,
        ".pkl": pd.read_pickle,
        ".pickle": pd.read_pickle,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    if suffix not in readers:
        raise ValueError(f"Unsupported database format '{suffix}'.")
    return readers[suffix](path)


def _year_from_filename(path: Path) -> int:
    match = YEAR_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not infer a year from filename: {path.name}")
    return int(match.group(0))


def load_database(path: str | Path) -> pd.DataFrame:
    """Load one table or a directory of annual refined Parquet files."""
    database_path = Path(path).expanduser().resolve()
    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    if database_path.is_file():
        if database_path.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
            raise ValueError(f"Unsupported database format '{database_path.suffix}'.")
        return _normalize_columns(_read_table(database_path))

    files = sorted(database_path.glob("pnad_refined_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pnad_refined_*.parquet files were found in {database_path}.")

    frames = []
    for file in files:
        frame = _normalize_columns(pd.read_parquet(file))
        year = _year_from_filename(file)
        if "year" not in frame.columns:
            frame.insert(0, "year", year)
        elif not (pd.to_numeric(frame["year"], errors="coerce") == year).all():
            raise ValueError(f"Year values in {file.name} disagree with its filename.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_database(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the minimal analytical schema."""
    out = _normalize_columns(df)
    missing = {"year", "income"}.difference(out.columns)
    if missing:
        raise ValueError("Database is missing required columns: " + ", ".join(sorted(missing)))

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["income"] = pd.to_numeric(out["income"], errors="coerce")
    if out["year"].isna().any():
        raise ValueError("Column 'year' contains non-numeric or missing values.")

    for column in (
        "income_effective", "income_adj", "income_effective_adj",
        "exchange", "price_index", "inflation_to_2025",
    ):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["year"] = out["year"].astype(int)
    return out.sort_values("year").reset_index(drop=True)


def attach_monetary_metadata(
    df: pd.DataFrame,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Attach missing year-level monetary fields."""
    out = df.copy()
    columns = ["currency", "exchange", "price_index", "inflation_to_2025"]
    missing = [column for column in columns if column not in out.columns]
    if not missing:
        return out
    metadata = load_metadata(metadata_path)[["year", *missing]]
    return out.merge(metadata, on="year", how="left", validate="many_to_one")


def standardize_income_frame(raw: pd.DataFrame, spec: pd.Series) -> pd.DataFrame:
    """Convert one raw annual frame to the common analytical schema."""
    df = raw.copy()
    missing_code = float(spec["missing_income_code"]) if pd.notna(spec["missing_income_code"]) else None
    df["income_raw"] = pd.to_numeric(df["income_raw"], errors="coerce")
    if missing_code is not None:
        df = df.loc[df["income_raw"] != missing_code].copy()

    if bool(spec["divide_by_household_size"]):
        if "household_size" not in df.columns:
            raise KeyError("household_size is required for this survey year.")
        df["household_size"] = pd.to_numeric(df["household_size"], errors="coerce")
        df = df.loc[df["household_size"] > 0].copy()
        df["income"] = df["income_raw"] / df["household_size"]
    else:
        df["income"] = df["income_raw"]

    if "income_effective_raw" in df.columns:
        effective = pd.to_numeric(df["income_effective_raw"], errors="coerce")
        if missing_code is not None:
            effective = effective.mask(effective == missing_code)
        df["income_effective"] = effective

    keep = ["income"] + (["income_effective"] if "income_effective" in df.columns else [])
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
    """Optionally reproduce the historical 1976--1979 upper-income cuts."""
    out = df.copy()
    if not enabled:
        return out
    missing = {year_col, value_col}.difference(out.columns)
    if missing:
        raise KeyError("Manual outlier filtering requires: " + ", ".join(sorted(missing)))

    active = LEGACY_MANUAL_OUTLIER_CUTS if cuts is None else dict(cuts)
    years = pd.to_numeric(out[year_col], errors="coerce")
    values = pd.to_numeric(out[value_col], errors="coerce")
    keep = np.ones(len(out), dtype=bool)
    for year, threshold in active.items():
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Outlier thresholds must be finite positive values.")
        keep &= ~((years == int(year)) & (values > threshold)).to_numpy()
    return out.loc[keep].reset_index(drop=True)


def adjust_income_to_2025(
    df: pd.DataFrame,
    income_columns: tuple[str, ...] = ("income", "income_effective"),
) -> pd.DataFrame:
    """Create 2025-reference income columns from year-level monetary factors."""
    missing = {"exchange", "inflation_to_2025"}.difference(df.columns)
    if missing:
        raise KeyError("Monetary adjustment requires: " + ", ".join(sorted(missing)))
    out = df.copy()
    exchange = pd.to_numeric(out["exchange"], errors="coerce").where(lambda x: x > 0)
    inflation = pd.to_numeric(out["inflation_to_2025"], errors="coerce")
    for column in income_columns:
        if column in out.columns:
            out[f"{column}_adj"] = pd.to_numeric(out[column], errors="coerce") / exchange * inflation
    return out


def prepare_panel(
    database_path: str | Path,
    *,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    start_year: int | None = None,
    end_year: int | None = None,
    apply_outlier_cuts: bool = False,
) -> pd.DataFrame:
    """Load, validate, filter, and monetarily harmonize the analytical panel."""
    panel = attach_monetary_metadata(validate_database(load_database(database_path)), metadata_path)
    if start_year is not None:
        panel = panel.loc[panel["year"] >= int(start_year)]
    if end_year is not None:
        panel = panel.loc[panel["year"] <= int(end_year)]
    if panel.empty:
        raise ValueError("No observations remain after applying the year filter.")
    panel = apply_manual_outlier_cuts(panel, enabled=apply_outlier_cuts)
    if panel.empty:
        raise ValueError("No observations remain after optional outlier filtering.")
    return adjust_income_to_2025(panel).reset_index(drop=True)


def _colspec(start: float, width: float) -> tuple[int, int]:
    return int(start), int(start + width)


def _raw_paths(raw_root: str | Path, pattern: str) -> list[Path]:
    paths = sorted(Path(raw_root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No raw file matched {Path(raw_root) / pattern}")
    return paths


def read_raw_year(
    year: int,
    raw_root: str | Path,
    metadata: pd.DataFrame | None = None,
    include_effective: bool = True,
) -> pd.DataFrame:
    """Read one raw fixed-width PNAD year using project metadata."""
    md = load_metadata() if metadata is None else metadata
    spec = metadata_for_year(year, md)
    if not bool(spec["available"]):
        raise ValueError(f"PNAD data are marked unavailable for {year}.")

    colspecs = [_colspec(spec["income_start"], spec["income_width"])]
    names = ["income_raw"]
    if pd.notna(spec["household_size_start"]):
        colspecs.append(_colspec(spec["household_size_start"], spec["household_size_width"]))
        names.append("household_size")
    if include_effective and pd.notna(spec["effective_income_start"]):
        colspecs.append(_colspec(spec["effective_income_start"], spec["effective_income_width"]))
        names.append("income_effective_raw")

    frames = [pd.read_fwf(path, colspecs=colspecs, names=names) for path in _raw_paths(raw_root, spec["raw_file_pattern"])]
    return standardize_income_frame(pd.concat(frames, ignore_index=True), spec)


def write_refined_year(df: pd.DataFrame, output_root: str | Path, year: int) -> Path:
    """Write one standardized annual dataset as Parquet."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"pnad_refined_{int(year)}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    return path


def build_refined_datasets(
    raw_root: str | Path,
    output_root: str | Path,
    metadata_path: str | Path | None = None,
    years: list[int] | None = None,
    include_effective: bool = True,
) -> list[Path]:
    """Build requested annual refined datasets from raw fixed-width files."""
    metadata = load_metadata(metadata_path)
    selected = years or metadata.loc[metadata["available"], "year"].astype(int).tolist()
    return [
        write_refined_year(
            read_raw_year(year, raw_root, metadata, include_effective=include_effective),
            output_root,
            year,
        )
        for year in selected
    ]
