"""Command-line entry point for trusted PNAD income analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import (
    DEFAULT_METADATA_PATH,
    DEFAULT_REFINED_PATH,
    DEFAULT_TRUSTED_PATH,
    load_database,
    validate_database,
)
from descriptive import IncomeDataCleaner
from outputs import export_analysis_outputs
from pipeline import PipelineConfig, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the trusted PNAD layer, run the scientific analysis, and persist reproducible outputs."
    )
    parser.add_argument("--refined", default=str(DEFAULT_REFINED_PATH), help="Directory containing annual refined files.")
    parser.add_argument("--trusted", default=str(DEFAULT_TRUSTED_PATH), help="Directory where trusted annual files are written.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="Canonical PNAD metadata CSV.")
    parser.add_argument("--output", default="outputs", help="Output directory for figures, tables, and manifest.")
    parser.add_argument("--start-year", type=int, default=1976)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--ccdf-base", type=float, default=1.05)
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument(
        "--outlier-method",
        choices=sorted(IncomeDataCleaner.METHODS),
        default="log_mad",
        help="Deterministic annual upper-tail rule used after metadata sentinels are excluded.",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=6.0,
        help="Positive multiplier applied to the selected dispersion estimator.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    refined_path = Path(args.refined)
    trusted_path = Path(args.trusted)
    metadata_path = Path(args.metadata)

    refined = validate_database(load_database(refined_path))
    cleaner = IncomeDataCleaner(
        refined,
        metadata_path=metadata_path,
        method=args.outlier_method,
        threshold=args.outlier_threshold,
    )
    flagged_refined = cleaner.flagged_frame()
    cleaning_thresholds = cleaner.thresholds()
    cleaning_audit = cleaner.cleaning_audit()
    trusted_files = cleaner.materialize_trusted(trusted_path)

    config = PipelineConfig(
        database_path=trusted_path,
        metadata_path=metadata_path,
        start_year=args.start_year,
        end_year=args.end_year,
        ccdf_base=args.ccdf_base,
    )
    results = run_pipeline(config)
    manifest = export_analysis_outputs(
        results,
        output_root=Path(args.output),
        refined_panel=flagged_refined,
        cleaning_thresholds=cleaning_thresholds,
        cleaning_audit=cleaning_audit,
        histogram_bins=args.histogram_bins,
    )
    print(
        f"Materialized {len(trusted_files)} trusted annual files, processed {len(results.years)} survey years, "
        f"and wrote {len(manifest)} output records."
    )


if __name__ == "__main__":
    main()
