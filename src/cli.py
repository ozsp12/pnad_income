"""Command-line entry point for the PNAD income analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from outputs import export_analysis_outputs
from pipeline import PipelineConfig, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PNAD income analysis and persist reproducible outputs.")
    parser.add_argument("--database", default="dados_refined", help="Directory containing annual refined files.")
    parser.add_argument("--output", default="outputs", help="Output directory for figures, tables, and manifest.")
    parser.add_argument("--start-year", type=int, default=1976)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--ccdf-base", type=float, default=1.05)
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument("--manual-outlier-cuts", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = PipelineConfig(
        database_path=Path(args.database),
        start_year=args.start_year,
        end_year=args.end_year,
        ccdf_base=args.ccdf_base,
        apply_manual_outlier_cuts=args.manual_outlier_cuts,
    )
    results = run_pipeline(config)
    manifest = export_analysis_outputs(
        results,
        output_root=Path(args.output),
        histogram_bins=args.histogram_bins,
    )
    print(f"Processed {len(results.years)} survey years and wrote {len(manifest)} output records.")


if __name__ == "__main__":
    main()
