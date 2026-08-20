"""Persistence of reproducible PNAD tables, figures, and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import annual_inequality_indices, compare_gini_series, gini_validation_statistics
from descriptive import DescriptiveStatistics
from pipeline import PipelineResults, pipeline_overview
from plotting import (
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
    """Flat output tree; analytical hierarchy is encoded in filename prefixes."""

    root: Path
    figures: Path
    tables: Path


def prepare_output_paths(output_root: str | Path) -> OutputPaths:
    root = Path(output_root).expanduser().resolve()
    figures = root / "figures"
    tables = root / "tables"
    for directory in (root, figures, tables):
        directory.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root=root, figures=figures, tables=tables)


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


def _shared_positive_ylim(*frames: pd.DataFrame) -> tuple[float, float] | None:
    positives = []
    for frame in frames:
        if "income" not in frame.columns:
            continue
        values = pd.to_numeric(frame["income"], errors="coerce")
        values = values[np.isfinite(values) & (values > 0)]
        if not values.empty:
            positives.append(values)
    if not positives:
        return None
    merged = pd.concat(positives, ignore_index=True)
    lower = max(float(merged.min()) * 0.8, np.finfo(float).tiny)
    upper = float(merged.max()) * 1.2
    return lower, upper if upper > lower else lower * 10


def export_analysis_outputs(
    results: PipelineResults,
    output_root: str | Path = "outputs",
    *,
    refined_panel: pd.DataFrame | None = None,
    cleaning_thresholds: pd.DataFrame | None = None,
    cleaning_audit: pd.DataFrame | None = None,
    selected_year: int | None = None,
    selected_years: Iterable[int] | None = None,
    grid_nrows: int = 2,
    grid_ncols: int = 3,
    complete_nrows: int = 6,
    complete_ncols: int = 4,
    histogram_bins: int = 100,
    dpi: int = 200,
    gini_references: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Persist trusted scientific outputs plus refined-versus-trusted EDA diagnostics."""
    paths = prepare_output_paths(output_root)
    manifest: list[dict[str, object]] = []
    indices = annual_inequality_indices(results.panel)
    trusted = results.panel.copy()
    refined = refined_panel.copy() if refined_panel is not None else trusted.copy()
    if results.years:
        refined = refined.loc[refined["year"].isin(results.years)].reset_index(drop=True)

    eda_refined = DescriptiveStatistics(refined)
    eda_trusted = DescriptiveStatistics(trusted)

    tables: dict[str, pd.DataFrame] = {
        "eda_refined_descriptive_statistics.csv": eda_refined.annual_summary(),
        "eda_trusted_descriptive_statistics.csv": eda_trusted.annual_summary(),
        "eda_refined_value_frequencies.csv": eda_refined.value_frequencies(),
        "eda_trusted_value_frequencies.csv": eda_trusted.value_frequencies(),
        "eda_refined_metadata_sentinel_occurrences.csv": eda_refined.metadata_sentinel_occurrences(),
        "eda_refined_outlier_diagnostics.csv": eda_refined.outlier_diagnostics(),
        "eda_trusted_outlier_diagnostics.csv": eda_trusted.outlier_diagnostics(),
        "eda_trusted_data_quality_diagnostics.csv": build_diagnostics(results),
        "eda_cleaning_thresholds.csv": cleaning_thresholds if cleaning_thresholds is not None else pd.DataFrame(),
        "eda_cleaning_audit.csv": cleaning_audit if cleaning_audit is not None else pd.DataFrame(),
        "paper_pipeline_overview.csv": pipeline_overview(results),
        "paper_annual_summary.csv": results.summary,
        "paper_annual_inequality_indices.csv": indices,
        "paper_ccdf_income_nominal_adjusted.parquet": results.ccdf_nominal_adjusted,
    }
    if not results.ccdf_habitual_effective.empty:
        tables["paper_ccdf_income_habitual_effective.parquet"] = results.ccdf_habitual_effective

    if gini_references is not None and not gini_references.empty:
        comparison = compare_gini_series(results.summary, gini_references)
        tables.update(
            {
                "paper_gini_external_references.csv": gini_references,
                "paper_gini_external_comparison.csv": comparison,
                "paper_gini_external_validation_statistics.csv": gini_validation_statistics(comparison),
            }
        )

    saved_tables = [save_table(frame, paths.tables / name) for name, frame in tables.items()]
    manifest.extend(_rows(saved_tables, "table"))

    scalar_figures = {
        "paper_inequality_gini_all_years.png": plot_gini_evolution(results.summary),
        "paper_inequality_top_income_shares_all_years.png": plot_top_income_shares(results.summary),
        "paper_inequality_extended_pietra_k_z_all_years.png": plot_extended_inequality_evolution(results.summary),
        "paper_inequality_indices_all_years.png": plot_primary_indices(indices),
        "paper_inequality_zanardi_all_years.png": plot_zanardi(indices),
        "paper_inequality_information_all_years.png": plot_information_indices(indices),
        "paper_inequality_gini_pietra_kolkata_relations.png": plot_kolkata_pietra_relationships(indices),
        "paper_inequality_pietra_kolkata_bound_all_years.png": plot_pietra_kolkata_bound(indices),
        "paper_inequality_gini_zanardi_phase.png": plot_gini_zanardi(indices),
    }
    if gini_references is not None and not gini_references.empty:
        scalar_figures["paper_gini_external_validation.png"] = plot_gini_validation(results.summary, gini_references)
    saved_figures = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in scalar_figures.items()]
    manifest.extend(_rows(saved_figures, "figure"))

    hist_limits = eda_refined.income_limits_by_year()
    box_limits = eda_refined.positive_limits_by_year()
    max_panels = complete_nrows * complete_ncols
    eda_page_specs = [
        (
            "eda_refined_histogram_income",
            eda_refined.histogram_pages(
                bins=histogram_bins,
                max_panels=max_panels,
                ncols=complete_ncols,
                x_limits_by_year=hist_limits,
            ),
        ),
        (
            "eda_trusted_histogram_income",
            eda_trusted.histogram_pages(
                bins=histogram_bins,
                max_panels=max_panels,
                ncols=complete_ncols,
                x_limits_by_year=hist_limits,
            ),
        ),
        (
            "eda_refined_boxplot_income",
            eda_refined.boxplot_pages(max_panels=max_panels, ncols=complete_ncols, y_limits_by_year=box_limits),
        ),
        (
            "eda_trusted_boxplot_income",
            eda_trusted.boxplot_pages(max_panels=max_panels, ncols=complete_ncols, y_limits_by_year=box_limits),
        ),
    ]
    for stem, figures in eda_page_specs:
        manifest.extend(_rows(_save_pages(figures, paths.figures, stem, dpi), "figure"))

    shared_ylim = _shared_positive_ylim(refined, trusted)
    eda_scalar = {
        "eda_refined_outlier_income_upper_tail_all_years.png": eda_refined.outlier_overview_figure(ylim=shared_ylim),
        "eda_trusted_outlier_income_upper_tail_all_years.png": eda_trusted.outlier_overview_figure(ylim=shared_ylim),
        "eda_compare_outlier_income_upper_tail_refined_trusted.png": eda_refined.compare_upper_tail_figure(eda_trusted),
    }
    saved_eda_scalar = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in eda_scalar.items()]
    manifest.extend(_rows(saved_eda_scalar, "figure"))

    years = results.years
    ccdf = results.ccdf_nominal_adjusted
    paper_page_specs = [
        (
            "paper_ccdf_income_loglog",
            plot_ccdf_grid(
                ccdf,
                measure="income",
                years=years,
                transform="loglog",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
        ),
        (
            "paper_ccdf_income_gompertz",
            plot_ccdf_grid(
                ccdf,
                measure="income",
                years=years,
                transform="gompertz",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
        ),
        (
            "paper_lorenz_income",
            plot_lorenz_grid(results.panel, years=years, nrows=complete_nrows, ncols=complete_ncols),
        ),
        (
            "paper_lorenz_income_annotated_g_p_k_z",
            plot_lorenz_grid(
                results.panel,
                years=years,
                nrows=complete_nrows,
                ncols=complete_ncols,
                annotate=True,
            ),
        ),
        (
            "paper_ccdf_income_nominal_vs_adjusted_loglog",
            plot_measure_comparison_grid(
                ccdf,
                years=years,
                transform="loglog",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
        ),
    ]
    for stem, figures in paper_page_specs:
        manifest.extend(_rows(_save_pages(figures, paths.figures, stem, dpi), "figure"))

    if selected_year is not None:
        year = int(selected_year)
        individual = {
            f"eda_trusted_histogram_income_{year}.png": plot_histogram(
                results.panel,
                year,
                bins=histogram_bins,
                yscale="log",
            ),
            f"paper_ccdf_income_{year}_loglog.png": plot_ccdf(ccdf, year, transform="loglog"),
            f"paper_ccdf_income_{year}_gompertz.png": plot_ccdf(ccdf, year, transform="gompertz"),
            f"paper_lorenz_income_{year}.png": plot_lorenz_curve(results.panel, year),
            f"paper_lorenz_income_{year}_annotated_g_p_k_z.png": plot_lorenz_curve(results.panel, year, annotate=True),
            f"paper_ccdf_income_nominal_vs_adjusted_{year}_loglog.png": plot_measure_comparison(ccdf, year),
        }
        paths_saved = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in individual.items()]
        manifest.extend(_rows(paths_saved, "figure"))

    if selected_years is not None:
        selected = [int(year) for year in selected_years]
        selected_specs = [
            (
                "eda_trusted_selected_histogram_income",
                plot_histogram_grid(
                    results.panel,
                    years=selected,
                    bins=histogram_bins,
                    yscale="log",
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                ),
            ),
            (
                "paper_selected_ccdf_income_loglog",
                plot_ccdf_grid(
                    ccdf,
                    measure="income",
                    years=selected,
                    transform="loglog",
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                ),
            ),
            (
                "paper_selected_ccdf_income_gompertz",
                plot_ccdf_grid(
                    ccdf,
                    measure="income",
                    years=selected,
                    transform="gompertz",
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                ),
            ),
            (
                "paper_selected_lorenz_income",
                plot_lorenz_grid(results.panel, years=selected, nrows=grid_nrows, ncols=grid_ncols),
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
