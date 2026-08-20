"""Persistence of reproducible analysis products under the project output tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .advanced_inequality import annual_inequality_indices
from .advanced_plotting import (
    plot_gini_zanardi,
    plot_information_indices,
    plot_kolkata_pietra_relationships,
    plot_pietra_kolkata_bound,
    plot_primary_indices,
    plot_zanardi,
)
from .pipeline import PipelineResults, pipeline_overview
from .plotting import (
    plot_ccdf,
    plot_ccdf_grid,
    plot_extended_inequality_evolution,
    plot_gini_evolution,
    plot_gini_validation,
    plot_histogram,
    plot_histogram_grid,
    plot_lorenz_curve,
    plot_lorenz_grid,
    plot_measure_comparison,
    plot_measure_comparison_grid,
    plot_top_income_shares,
)
from .validation import compare_gini_series, gini_validation_statistics


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    reports: Path


def prepare_output_paths(output_root: str | Path) -> OutputPaths:
    root = Path(output_root).expanduser().resolve()
    figures = root / "figures"
    tables = root / "tables"
    reports = root / "reports"
    for path in (root, figures, tables, reports):
        path.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root, figures, tables, reports)


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


def save_table(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(destination, index=index)
    elif suffix == ".parquet":
        frame.to_parquet(destination, index=index)
    elif suffix in {".xlsx", ".xls"}:
        frame.to_excel(destination, index=index)
    else:
        raise ValueError(f"Unsupported table output format: {suffix}")
    return destination


def save_figure(figure, path: str | Path, *, dpi: int = 200, close: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
    return destination


def save_figure_pages(
    figures,
    output_dir: str | Path,
    stem: str,
    *,
    dpi: int = 200,
    close: bool = True,
) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for page_number, figure in enumerate(figures, start=1):
        saved.append(
            save_figure(
                figure,
                directory / f"{stem}_page_{page_number:02d}.png",
                dpi=dpi,
                close=close,
            )
        )
    return saved


def _manifest_rows(paths: Iterable[Path], category: str) -> list[dict[str, object]]:
    return [
        {
            "category": category,
            "filename": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
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
    """Persist the complete tabular and graphical analysis from one pipeline result."""
    paths = prepare_output_paths(output_root)
    manifest: list[dict[str, object]] = []

    overview = pipeline_overview(results)
    diagnostics = build_diagnostics(results)
    advanced_indices = annual_inequality_indices(results.panel, value_col="income", atkinson_epsilon=0.5)

    table_outputs = [
        save_table(overview, paths.tables / "pipeline_overview.csv"),
        save_table(results.summary, paths.tables / "annual_summary.csv"),
        save_table(advanced_indices, paths.tables / "annual_inequality_indices.csv"),
        save_table(results.ccdf_nominal_adjusted, paths.tables / "ccdf_nominal_adjusted.parquet"),
        save_table(diagnostics, paths.tables / "data_quality_diagnostics.csv"),
    ]
    if not results.ccdf_habitual_effective.empty:
        table_outputs.append(
            save_table(
                results.ccdf_habitual_effective,
                paths.tables / "ccdf_habitual_effective.parquet",
            )
        )
    manifest.extend(_manifest_rows(table_outputs, "table"))

    figures = [
        (plot_gini_evolution(results.summary), "gini_income_all_years.png"),
        (plot_top_income_shares(results.summary), "top_income_shares_all_years.png"),
        (
            plot_extended_inequality_evolution(results.summary),
            "extended_inequality_pietra_k_z_all_years.png",
        ),
        (plot_primary_indices(advanced_indices), "inequality_indices_all_years.png"),
        (plot_zanardi(advanced_indices), "zanardi_index_all_years.png"),
        (plot_information_indices(advanced_indices), "information_indices_all_years.png"),
        (
            plot_kolkata_pietra_relationships(advanced_indices),
            "gini_pietra_kolkata_relations.png",
        ),
        (
            plot_pietra_kolkata_bound(advanced_indices),
            "pietra_kolkata_bound_all_years.png",
        ),
        (plot_gini_zanardi(advanced_indices), "gini_zanardi_phase.png"),
    ]

    if gini_references is not None and not gini_references.empty:
        comparison = compare_gini_series(results.summary, gini_references)
        validation_stats = gini_validation_statistics(comparison)
        validation_tables = [
            save_table(gini_references, paths.tables / "gini_external_references.csv"),
            save_table(comparison, paths.tables / "gini_external_comparison.csv"),
            save_table(
                validation_stats,
                paths.tables / "gini_external_validation_statistics.csv",
            ),
        ]
        manifest.extend(_manifest_rows(validation_tables, "table"))
        figures.append(
            (
                plot_gini_validation(results.summary, gini_references),
                "gini_external_validation.png",
            )
        )

    for figure, filename in figures:
        manifest.extend(
            _manifest_rows(
                [save_figure(figure, paths.figures / filename, dpi=dpi, close=True)],
                "figure",
            )
        )

    years = results.years
    page_groups = [
        (
            plot_histogram_grid(
                results.panel,
                years=years,
                bins=histogram_bins,
                yscale="linear",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
            "histogram_income_linear",
        ),
        (
            plot_histogram_grid(
                results.panel,
                years=years,
                bins=histogram_bins,
                yscale="log",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
            "histogram_income_log_frequency",
        ),
        (
            plot_ccdf_grid(
                results.ccdf_nominal_adjusted,
                measure="income",
                years=years,
                transform="linear",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
            "ccdf_income_linear",
        ),
        (
            plot_ccdf_grid(
                results.ccdf_nominal_adjusted,
                measure="income",
                years=years,
                transform="loglog",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
            "ccdf_income_loglog",
        ),
        (
            plot_ccdf_grid(
                results.ccdf_nominal_adjusted,
                measure="income",
                years=years,
                transform="double_log",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
            "ccdf_income_double_log_legacy",
        ),
        (
            plot_lorenz_grid(
                results.panel,
                years=years,
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
            "lorenz_income",
        ),
        (
            plot_lorenz_grid(
                results.panel,
                years=years,
                nrows=complete_nrows,
                ncols=complete_ncols,
                annotate=True,
            ),
            "lorenz_income_annotated_g_p_k_z",
        ),
        (
            plot_measure_comparison_grid(
                results.ccdf_nominal_adjusted,
                years=years,
                transform="loglog",
                nrows=complete_nrows,
                ncols=complete_ncols,
            ),
            "ccdf_nominal_vs_adjusted_loglog",
        ),
    ]
    for group, stem in page_groups:
        manifest.extend(
            _manifest_rows(
                save_figure_pages(group, paths.figures, stem, dpi=dpi),
                "figure",
            )
        )

    if selected_year is not None:
        year = int(selected_year)
        individual = [
            (
                plot_histogram(results.panel, year, bins=histogram_bins, yscale="linear"),
                f"histogram_income_{year}_linear.png",
            ),
            (
                plot_histogram(results.panel, year, bins=histogram_bins, yscale="log"),
                f"histogram_income_{year}_log_frequency.png",
            ),
            (
                plot_ccdf(results.ccdf_nominal_adjusted, year, transform="linear"),
                f"ccdf_income_{year}_linear.png",
            ),
            (
                plot_ccdf(results.ccdf_nominal_adjusted, year, transform="loglog"),
                f"ccdf_income_{year}_loglog.png",
            ),
            (
                plot_ccdf(results.ccdf_nominal_adjusted, year, transform="double_log"),
                f"ccdf_income_{year}_double_log_legacy.png",
            ),
            (plot_lorenz_curve(results.panel, year), f"lorenz_income_{year}.png"),
            (
                plot_lorenz_curve(results.panel, year, annotate=True),
                f"lorenz_income_{year}_annotated_g_p_k_z.png",
            ),
            (
                plot_measure_comparison(results.ccdf_nominal_adjusted, year),
                f"ccdf_nominal_vs_adjusted_{year}_loglog.png",
            ),
        ]
        for figure, filename in individual:
            manifest.extend(
                _manifest_rows(
                    [save_figure(figure, paths.figures / filename, dpi=dpi, close=True)],
                    "figure",
                )
            )

    if selected_years is not None:
        selected = [int(year) for year in selected_years]
        selected_groups = [
            (
                plot_histogram_grid(
                    results.panel,
                    years=selected,
                    bins=histogram_bins,
                    yscale="log",
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                ),
                "selected_histogram_income_log_frequency",
            ),
            (
                plot_ccdf_grid(
                    results.ccdf_nominal_adjusted,
                    measure="income",
                    years=selected,
                    transform="loglog",
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                ),
                "selected_ccdf_income_loglog",
            ),
            (
                plot_lorenz_grid(
                    results.panel,
                    years=selected,
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                ),
                "selected_lorenz_income",
            ),
            (
                plot_lorenz_grid(
                    results.panel,
                    years=selected,
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                    annotate=True,
                ),
                "selected_lorenz_income_annotated_g_p_k_z",
            ),
        ]
        for group, stem in selected_groups:
            manifest.extend(
                _manifest_rows(
                    save_figure_pages(group, paths.figures, stem, dpi=dpi),
                    "figure",
                )
            )

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
