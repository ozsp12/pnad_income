from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_metadata, metadata_for_year
from .preprocessing import standardize_income_frame


def _colspec(start: float, width: float) -> tuple[int, int]:
    return int(start), int(start + width)


def _raw_paths(raw_root: str | Path, pattern: str) -> list[Path]:
    root = Path(raw_root)
    paths = sorted(root.glob(pattern))
    if not paths:
        candidate = root / pattern
        raise FileNotFoundError(f"No raw file matched {candidate}")
    return paths


def read_raw_year(year: int, raw_root: str | Path, metadata: pd.DataFrame | None = None, include_effective: bool = True) -> pd.DataFrame:
    """Read one PNAD year from fixed-width raw files using project metadata."""
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
    raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return standardize_income_frame(raw, spec)


def write_refined_year(df: pd.DataFrame, output_root: str | Path, year: int) -> Path:
    """Write one standardized annual dataset in Parquet format."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"pnad_refined_{year}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    return path


def build_refined_datasets(raw_root: str | Path, output_root: str | Path, metadata_path: str | Path | None = None, years: list[int] | None = None, include_effective: bool = True) -> list[Path]:
    """Build all requested annual refined datasets from raw fixed-width files."""
    md = load_metadata(metadata_path)
    if years is None:
        years = md.loc[md["available"], "year"].astype(int).tolist()
    written = []
    for year in years:
        df = read_raw_year(year, raw_root, md, include_effective=include_effective)
        written.append(write_refined_year(df, output_root, year))
    return written


def load_refined_panel(refined_root: str | Path, metadata_path: str | Path | None = None, start_year: int = 1976, end_year: int = 2025, adjust: bool = True) -> pd.DataFrame:
    """Concatenate available annual Parquet files and attach monetary metadata."""
    from .preprocessing import adjust_income_to_2025
    md = load_metadata(metadata_path)
    root = Path(refined_root)
    frames = []
    for year in range(start_year, end_year + 1):
        path = root / f"pnad_refined_{year}.parquet"
        if path.exists():
            frame = pd.read_parquet(path).copy()
            if "year" not in frame.columns:
                frame["year"] = year
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No refined Parquet files were found in the requested interval.")
    panel = pd.concat(frames, ignore_index=True)
    monetary = md[["year", "currency", "exchange", "price_index", "inflation_to_2025"]]
    panel = panel.merge(monetary, on="year", how="left", validate="many_to_one")
    if adjust:
        panel = adjust_income_to_2025(panel)
    return panel.sort_values("year").reset_index(drop=True)
