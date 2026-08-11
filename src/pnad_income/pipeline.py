"""High-level orchestration for the complete PNAD income analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .config import DEFAULT_METADATA_PATH, load_metadata
from .distributions import compare_income_measures_ccdf
from .inequality import summary_statistics
from .preprocessing import adjust_income_to_2025

SUPPORTED_FILE_SUFFIXES = {".parquet", ".csv", ".feather", ".pkl", ".pickle", ".xlsx", ".xls"}
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration required to execute the analytical pipeline."""

    database_path: str | Path
    metadata_path: str | Path = DEFAULT_METADATA_PATH
    ccdf_base: float = 1.05
    start_year: int | None = None
    end_year: int | None = None


@dataclass
class PipelineResults:
    """Structured outputs returned by :func:`run_pipeline`."""

    panel: pd.DataFrame
    summary: pd.DataFrame
    ccdf_nominal_adjusted: pd.DataFrame
    ccdf_habitual_effective: pd.DataFrame

    @property
    def years(self) -> list[int]:
        """Return sorted survey years represented in the panel."""
        return sorted(self.panel["year"].dropna().astype(int).unique().tolist())


def _read_table(path: Path) -> pd.DataFrame:
    """Read one supported tabular file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported database format '{suffix}'.")


def _year_from_filename(path: Path) -> int:
    """Extract the survey year from a filename such as pnad_refined_2025.parquet."""
    match = YEAR_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not infer a year from filename: {path.name}")
    return int(match.group(0))


def load_database(path: str | Path) -> pd.DataFrame:
    """Load either one database file or a directory of annual Parquet files.

    For directory input, files matching `pnad_refined_*.parquet` are concatenated.
    If an annual file does not contain `year`, the value is inferred from its
    filename before concatenation.
    """
    database_path = Path(path).expanduser().resolve()
    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    if database_path.is_dir():
        files = sorted(database_path.glob("pnad_refined_*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No pnad_refined_*.parquet files were found in {database_path}."
            )
        frames = []
        for file in files:
            frame = pd.read_parquet(file).copy()
            inferred_year = _year_from_filename(file)
            if "year" not in frame.columns:
                frame.insert(0, "year", inferred_year)
            elif not (pd.to_numeric(frame["year"], errors="coerce") == inferred_year).all():
                raise ValueError(f"Year values in {file.name} disagree with its filename.")
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    if database_path.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(
            f"Unsupported database format '{database_path.suffix}'. "
            f"Expected one of {sorted(SUPPORTED_FILE_SUFFIXES)}."
        )
    return _read_table(database_path)


def validate_database(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the minimal schema required by the analysis."""
    required = {"year", "income"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Database is missing required columns: " + ", ".join(sorted(missing)))

    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["income"] = pd.to_numeric(out["income"], errors="coerce")
    if out["year"].isna().any():
        raise ValueError("Column 'year' contains non-numeric or missing values.")

    optional_numeric = [
        "income_effective", "income_adj", "income_effective_adj",
        "exchange", "price_index", "inflation_to_2025",
    ]
    for column in optional_numeric:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["year"] = out["year"].astype(int)
    return out.sort_values("year").reset_index(drop=True)


def attach_monetary_metadata(
    df: pd.DataFrame,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Attach missing year-level monetary variables from canonical metadata."""
    out = df.copy()
    metadata = load_metadata(metadata_path)
    candidates = ["currency", "exchange", "price_index", "inflation_to_2025"]
    missing_columns = [column for column in candidates if column not in out.columns]
    if not missing_columns:
        return out
    lookup = metadata[["year", *missing_columns]].copy()
    return out.merge(lookup, on="year", how="left", validate="many_to_one")


def prepare_panel(config: PipelineConfig) -> pd.DataFrame:
    """Load, validate, filter, and monetarily standardize the analytical panel."""
    panel = validate_database(load_database(config.database_path))
    panel = attach_monetary_metadata(panel, config.metadata_path)

    if config.start_year is not None:
        panel = panel.loc[panel["year"] >= int(config.start_year)]
    if config.end_year is not None:
        panel = panel.loc[panel["year"] <= int(config.end_year)]
    if panel.empty:
        raise ValueError("No observations remain after applying the year filter.")

    # Recompute adjusted variables so stale precomputed values cannot survive silently.
    return adjust_income_to_2025(panel).reset_index(drop=True)


def run_pipeline(config: PipelineConfig) -> PipelineResults:
    """Execute the complete analytical pipeline and return structured outputs."""
    panel = prepare_panel(config)
    summary = summary_statistics(panel)

    nominal_measures = tuple(c for c in ("income", "income_adj") if c in panel.columns)
    ccdf_nominal_adjusted = compare_income_measures_ccdf(
        panel, measures=nominal_measures, base=config.ccdf_base, scale="percent"
    )

    if "income_effective" in panel.columns and panel["income_effective"].notna().any():
        effective_panel = panel.loc[panel["income_effective"].notna()].copy()
        ccdf_habitual_effective = compare_income_measures_ccdf(
            effective_panel,
            measures=("income", "income_effective"),
            base=config.ccdf_base,
            scale="percent",
        )
    else:
        ccdf_habitual_effective = pd.DataFrame()

    return PipelineResults(
        panel=panel,
        summary=summary,
        ccdf_nominal_adjusted=ccdf_nominal_adjusted,
        ccdf_habitual_effective=ccdf_habitual_effective,
    )


def pipeline_overview(results: PipelineResults) -> pd.DataFrame:
    """Return compact coverage and output diagnostics for an executed pipeline."""
    years = results.years
    panel = results.panel
    effective_years = (
        panel.loc[panel["income_effective"].notna(), "year"].nunique()
        if "income_effective" in panel.columns else 0
    )
    return pd.DataFrame({
        "metric": [
            "observations", "first_year", "last_year", "number_of_years",
            "years_with_effective_income", "ccdf_rows_nominal_adjusted",
            "ccdf_rows_habitual_effective",
        ],
        "value": [
            len(panel), min(years), max(years), len(years), int(effective_years),
            len(results.ccdf_nominal_adjusted), len(results.ccdf_habitual_effective),
        ],
    })
