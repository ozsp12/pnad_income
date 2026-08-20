"""Persistence of reproducible PNAD tables, figures, and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .analysis import annual_inequality_indices, compare_gini_series, gini_validation_statistics
from .pipeline import PipelineResults, pipeline_overview
from .plotting import (
    plot_ccdf,
    plot_ccdf_grid,
    plot_extended_inequality_evolution,
    plot_gini_evolution,
    plot_gini_validation,
    plot_gini_zanardi,
    plot_histogram,
    plot_histogram_grid,
    plot_information_indices,
    plot_kolkata_pietra_relationships,
    plot_lorenz_curve,
    plot_lorenz_grid,
    plot_measure_comparison,
    plot_measure_comparison_grid,
    plot_pietra_kolkata_bound,
    plot_primary_indices,
    plot_top_income_shares,
    plot_zanardi,
)


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    reports: Path


def prepare_output_paths(output_root: str | Path) -> OutputPaths:
    root = Path(output_root).expanduser().resolve()
    paths = OutputPaths(root, root / "figures", root / "tables", root / "reports")
    for path in (paths.root, paths.figures, paths.tables, paths.reports):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_diagnostics(results: PipelineResults) -> pd.DataFrame:
    columns = [
        column
        for column in ("income", "income_adj", "income_effective", "income_effective_adj")
        if column in results.panel.columns
    ]
    return pd.DataFrame(
        {
            "column": columns,
            "non_missing": [int(results.panel[c].notna().sum()) for c in columns],
            "missing": [int(results.panel[c].isna().sum()) for c in columns],
            "minimum": [results.panel[c].min() for c in columns],
            "maximum": [results.panel[c].max() for c in columns],
        }
    )


def save_table(frame: pd.DataFrame, path: str | Path, *, index=False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writers = {
        ".csv": lambda: frame.to_csv(path, index=index),
        ".parquet": lambda: frame.to_parquet(path, index=index),
        ".xlsx": lambda: frame.to_excel(path, index=index),
        ".xls": lambda: frame.to_excel(path, index=index),
    }
    if path.suffix.lower() not in writers:
        raise ValueError(f"Unsupported table output format: {path.suffix}")
    writers[path.suffix.lower()]()
    return path


def save_figure(figure, path: str | Path, *, dpi=200, close=True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
    return path


def _rows(paths: Iterable[Path], category: str) -> list[dict[str, object]]:
    return [
        {
            "category": category,
            "filename": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _save_pages(figures, directory: Path, stem: str, dpi: int) -> list[Path]:
    return [
        save_figure(figure, directory / f"{stem}_page_{i:02d}.png", dpi=dpi)
        for i, figure in enumerate(figures, 1)
    ]


def export_analysis_outputs(
    results: PipelineResults,
    output_root: str | Path = "outputs",
    *,
    selected_year: int | None = None,
    selected_years: Iterable[int] | None = None,
    grid_nrows: int = 2,
    grid_ncols: int = 3,
    complete_nrows: int = 6,
    complete_ncols: int = 4,
    histogram_bins: int = 60,
    dpi: int = 200,
    gini_references: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Persist the complete analysis from one :class:`PipelineResults` object."""
    paths = prepare_output_paths(output_root)
    manifest: list[dict[str, object]] = []
    indices = annual_inequality_indices(results.panel)

    tables = {
        "pipeline_overview.csv": pipeline_overview(results),
        "annual_summary.csv": results.summary,
        "annual_inequality_indices.csv": indices,
        "ccdf_nominal_adjusted.parquet": results.ccdf_nominal_adjusted,
        "data_quality_diagnostics.csv": build_diagnostics(results),
    }
    if not results.ccdf_habitual_effective.empty:
        tables["ccdf_habitual_effective.parquet"] = results.ccdf_habitual_effective

    if gini_references is not None and not gini_references.empty:
        comparison = compare_gini_series(results.summary, gini_references)
        tables.update(
            {
                "gini_external_references.csv": gini_references,
                "gini_external_comparison.csv": comparison,
                "gini_external_validation_statistics.csv": gini_validation_statistics(comparison),
            }
        )

    saved_tables = [save_table(frame, paths.tables / name) for name, frame in tables.items()]
    manifest.extend(_rows(saved_tables, "table"))

    scalar_figures = {
        "gini_income_all_years.png": plot_gini_evolution(results.summary),
        "top_income_shares_all_years.png": plot_top_income_shares(results.summary),
        "extended_inequality_pietra_k_z_all_years.png": plot_extended_inequality_evolution(results.summary),
        "inequality_indices_all_years.png": plot_primary_indices(indices),
        "zanardi_index_all_years.png": plot_zanardi(indices),
        "information_indices_all_years.png": plot_information_indices(indices),
        "gini_pietra_kolkata_relations.png": plot_kolkata_pietra_relationships(indices),
        "pietra_kolkata_bound_all_years.png": plot_pietra_kolkata_bound(indices),
        "gini_zanardi_phase.png": plot_gini_zanardi(indices),
    }
    if gini_references is not None and not gini_references.empty:
        scalar_figures["gini_external_validation.png"] = plot_gini_validation(results.summary, gini_references)

    saved_figures = [
        save_figure(figure, paths.figures / name, dpi=dpi)
        for name, figure in scalar_figures.items()
    ]
    manifest.extend(_rows(saved_figures, "figure"))

    years = results.years
    ccdf = results.ccdf_nominal_adjusted
    page_specs = [
        (
            "histogram_income_linear",
            plot_histogram_grid(results.panel, years=years, bins=histogram_bins, yscale="linear", nrows=complete_nrows, ncols=complete_ncols),
        ),
        (
            "histogram_income_log_frequency",
            plot_histogram_grid(results.panel, years=years, bins=histogram_bins, yscale="log", nrows=complete_nrows, ncols=complete_ncols),
        ),
        (
            "ccdf_income_linear",
            plot_ccdf_grid(ccdf, measure="income", years=years, transform="linear", nrows=complete_nrows, ncols=complete_ncols),
        ),
        (
            "ccdf_income_loglog",
            plot_ccdf_grid(ccdf, measure="income", years=years, transform="loglog", nrows=complete_nrows, ncols=complete_ncols),
        ),
        (
            "ccdf_income_double_log_legacy",
            plot_ccdf_grid(ccdf, measure="income", years=years, transform="double_log", nrows=complete_nrows, ncols=complete_ncols),
        ),
        (
            "lorenz_income",
            plot_lorenz_grid(results.panel, years=years, nrows=complete_nrows, ncols=complete_ncols),
        ),
        (
            "lorenz_income_annotated_g_p_k_z",
            plot_lorenz_grid(results.panel, years=years, nrows=complete_nrows, ncols=complete_ncols, annotate=True),
        ),
        (
            "ccdf_nominal_vs_adjusted_loglog",
            plot_measure_comparison_grid(ccdf, years=years, transform="loglog", nrows=complete_nrows, ncols=complete_ncols),
        ),
    ]
    for stem, figures in page_specs:
        manifest.extend(_rows(_save_pages(figures, paths.figures, stem, dpi), "figure"))

    if selected_year is not None:
        year = int(selected_year)
        individual = {
            f"histogram_income_{year}_linear.png": plot_histogram(results.panel, year, bins=histogram_bins),
            f"histogram_income_{year}_log_frequency.png": plot_histogram(results.panel, year, bins=histogram_bins, yscale="log"),
            f"ccdf_income_{year}_linear.png": plot_ccdf(ccdf, year, transform="linear"),
            f"ccdf_income_{year}_loglog.png": plot_ccdf(ccdf, year, transform="loglog"),
            f"ccdf_income_{year}_double_log_legacy.png": plot_ccdf(ccdf, year, transform="double_log"),
            f"lorenz_income_{year}.png": plot_lorenz_curve(results.panel, year),
            f"lorenz_income_{year}_annotated_g_p_k_z.png": plot_lorenz_curve(results.panel, year, annotate=True),
            f"ccdf_nominal_vs_adjusted_{year}_loglog.png": plot_measure_comparison(ccdf, year),
        }
        paths_saved = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in individual.items()]
        manifest.extend(_rows(paths_saved, "figure"))

    if selected_years is not None:
        selected = [int(year) for year in selected_years]
        selected_specs = [
            (
                "selected_histogram_income_log_frequency",
                plot_histogram_grid(results.panel, years=selected, bins=histogram_bins, yscale="log", nrows=grid_nrows, ncols=grid_ncols),
            ),
            (
                "selected_ccdf_income_loglog",
                plot_ccdf_grid(ccdf, measure="income", years=selected, transform="loglog", nrows=grid_nrows, ncols=grid_ncols),
            ),
            (
                "selected_lorenz_income",
                plot_lorenz_grid(results.panel, years=selected, nrows=grid_nrows, ncols=grid_ncols),
            ),
            (
                "selected_lorenz_income_annotated_g_p_k_z",
                plot_lorenz_grid(results.panel, years=selected, nrows=grid_nrows, ncols=grid_ncols, annotate=True),
            ),
        ]
        for stem, figures in selected_specs:
            manifest.extend(_rows(_save_pages(figures, paths.figures, stem, dpi), "figure"))

    manifest_frame = pd.DataFrame(manifest)
    manifest_path = save_table(manifest_frame, paths.root / "manifest.csv")
    return pd.concat(
        [
            manifest_frame,
            pd.DataFrame(
                [
                    {
                        "category": "manifest",
                        "filename": manifest_path.name,
                        "path": str(manifest_path),
                        "size_bytes": manifest_path.stat().st_size,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
