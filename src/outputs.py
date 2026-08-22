"""Persistence of reproducible PNAD tables, figures, and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import (
    annual_income_group_totals,
    annual_inequality_indices,
    compare_gini_series,
    gini_validation_statistics,
)
from descriptive import DescriptiveStatistics
from pipeline import PipelineResults, pipeline_overview
from plotting import (
    plot_ccdf,
    plot_ccdf_grid,
    plot_distribution_cutoff_history,
    plot_distribution_regime_fits,
    plot_extended_inequality_evolution,
    plot_gini_evolution,
    plot_gini_validation,
    plot_gini_zanardi,
    plot_gompertz_parameter_history,
    plot_histogram,
    plot_histogram_grid,
    plot_income_group_totals,
    plot_information_indices,
    plot_kolkata_pietra_relationships,
    plot_lorenz_curve,
    plot_lorenz_grid,
    plot_measure_comparison,
    plot_measure_comparison_grid,
    plot_pareto_alpha_history,
    plot_pietra_kolkata_bound,
    plot_primary_indices,
    plot_regime_r2_history,
    plot_top_income_shares,
    plot_zanardi,
)


MANIFEST_COLUMNS = [
    "category",
    "stage",
    "data_layer",
    "filename",
    "path",
    "size_bytes",
    "size_human",
    "format",
    "description",
    "url",
    "sha256",
    "commit_sha",
    "generated_at",
]


ARTIFACT_DESCRIPTIONS = {
    "eda_refined_descriptive_statistics.csv": (
        "Annual descriptive statistics for the refined layer before statistical cleaning, used as the baseline for evaluating the trusted-layer transformation."
    ),
    "eda_trusted_descriptive_statistics.csv": (
        "Annual descriptive statistics for the trusted layer after deterministic data-quality treatment and removal of flagged records."
    ),
    "eda_refined_value_frequencies.csv": (
        "Exact-value income frequencies in the refined layer, supporting inspection of repeated values, discrete structures, and concentration before cleaning."
    ),
    "eda_trusted_value_frequencies.csv": (
        "Exact-value income frequencies in the trusted layer, allowing comparison of the observed value structure after cleaning."
    ),
    "eda_refined_metadata_sentinel_occurrences.csv": (
        "Annual counts of metadata-defined missing-value sentinel codes identified in the refined layer."
    ),
    "eda_refined_outlier_diagnostics.csv": (
        "Annual upper-tail and potential-outlier diagnostics for the refined income data before construction of the trusted layer."
    ),
    "eda_trusted_outlier_diagnostics.csv": (
        "Annual upper-tail diagnostics for trusted income data after deterministic cleaning."
    ),
    "eda_trusted_data_quality_diagnostics.csv": (
        "Compact trusted-layer quality summary covering valid observations, missingness, and numerical support of analytical variables."
    ),
    "eda_cleaning_thresholds.csv": (
        "Year-specific statistical thresholds estimated by the configured outlier rule when constructing the trusted layer."
    ),
    "eda_cleaning_audit.csv": (
        "Annual cleaning audit recording initial sample size, sentinel and outlier removals, final trusted sample size, and removal rates."
    ),
    "paper_pipeline_overview.csv": (
        "Compact overview of the scientific pipeline execution, including temporal coverage, observation counts, and availability of major analytical outputs."
    ),
    "paper_annual_summary.csv": (
        "Consolidated annual summary of descriptive statistics and core analytical measures used to characterize the longitudinal income series."
    ),
    "paper_annual_inequality_indices.csv": (
        "Annual inequality and concentration indices calculated from trusted data, including Gini, Pietra, Kolkata, Zanardi, Theil, Atkinson, top-income, Shannon, and Herfindahl measures."
    ),
    "paper_annual_income_groups_p80_p99_p100_2025_usd.csv": (
        "Absolute annual aggregate income in 2025 USD split into the bottom 80 percent, next 19 percent, and top 1 percent of trusted income records."
    ),
    "paper_ccdf_income_nominal_adjusted.parquet": (
        "Annual empirical CCDF data for nominal and monetarily adjusted income, stored in Parquet format for reproducible distributional analysis and plotting."
    ),
    "paper_ccdf_income_habitual_effective.parquet": (
        "Annual empirical CCDF data comparing habitual and effective income measures when both are available in the analytical data."
    ),
    "paper_distribution_regime_fits.csv": (
        "Annual normalized Gompertz-body and Pareto-tail least-squares profiles from positive trusted adjusted income, including common-scale joint SSE, cutoff identification, sensitivity, continuity, and fit diagnostics."
    ),
    "paper_distribution_regime_curves.parquet": (
        "Empirical and fitted annual CCDF curve points that fully reproduce the Gompertz-Pareto regime figures without microdata access."
    ),
    "paper_gini_external_references.csv": (
        "Documented external Gini reference series supplied for validation of the internally calculated PNAD inequality trajectory."
    ),
    "paper_gini_external_comparison.csv": (
        "Year-aligned comparison between internally calculated Gini coefficients and documented external reference series."
    ),
    "paper_gini_external_validation_statistics.csv": (
        "Validation statistics summarizing agreement between calculated and external Gini series, including error and association measures."
    ),
    "paper_inequality_gini_all_years.png": (
        "Temporal evolution of the Gini coefficient across all available PNAD and PNAD Contínua survey years."
    ),
    "paper_inequality_top_income_shares_all_years.png": (
        "Temporal evolution of income shares held by the upper tail of the distribution, including the top 10%, 1%, and 0.1%."
    ),
    "paper_income_groups_p80_p99_p100_absolute_2025_usd_all_years.png": (
        "Non-normalized stacked bars of annual aggregate income in 2025 USD for the bottom 80 percent, next 19 percent, and top 1 percent of trusted records."
    ),
    "paper_inequality_extended_pietra_k_z_all_years.png": (
        "Joint temporal evolution of the Pietra index, Kolkata k-index, and Zanardi Z statistic across the harmonized series."
    ),
    "paper_inequality_indices_all_years.png": (
        "Longitudinal comparison of the principal inequality indices calculated from the trusted annual income distributions."
    ),
    "paper_inequality_zanardi_all_years.png": (
        "Temporal evolution of the Zanardi Z statistic across all available survey years."
    ),
    "paper_inequality_information_all_years.png": (
        "Temporal evolution of information- and concentration-based inequality measures, including Shannon-derived and Herfindahl quantities."
    ),
    "paper_inequality_gini_pietra_kolkata_relations.png": (
        "Empirical relationships among the Gini, Pietra, and Kolkata inequality indices across annual PNAD income distributions."
    ),
    "paper_inequality_pietra_kolkata_bound_all_years.png": (
        "Annual comparison of Pietra and Kolkata indices against the analytical relationship or bound examined by the project."
    ),
    "paper_inequality_gini_zanardi_phase.png": (
        "Phase-space representation of the relationship between Gini inequality and the Zanardi Z statistic across survey years."
    ),
    "paper_gini_external_validation.png": (
        "Graphical comparison of the internally calculated Gini series with documented external reference series used for technical validation."
    ),
    "paper_gompertz_parameter_B_all_years.png": (
        "Annual positive Gompertz body slope B from fixed-intercept least squares on the normalized-income scale."
    ),
    "paper_pareto_alpha_all_years.png": (
        "Annual least-squares Pareto CCDF exponent alpha for the selected upper-income tail."
    ),
    "paper_distribution_cutoff_all_years.png": (
        "Annual normalized-income cutoff separating the profiled Gompertz body and Pareto tail regimes."
    ),
    "paper_distribution_regime_r2_all_years.png": (
        "Annual Gompertz-body and Pareto-tail regression R-squared values with model-specific mean reference lines."
    ),
    "eda_refined_outlier_income_upper_tail_all_years.png": (
        "Upper-tail income diagnostic for all refined survey years before trusted-layer cleaning."
    ),
    "eda_trusted_outlier_income_upper_tail_all_years.png": (
        "Upper-tail income diagnostic for all trusted survey years after deterministic cleaning."
    ),
    "eda_compare_outlier_income_upper_tail_refined_trusted.png": (
        "Direct comparison of refined and trusted upper tails, showing the empirical effect of the outlier-treatment rule."
    ),
}


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


def _human_size(size_bytes: int) -> str:
    """Return a compact human-readable representation of a byte count."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _sha256(path: Path) -> str:
    """Compute the SHA-256 digest of one persisted artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_commit_sha() -> str:
    """Resolve the source commit from GitHub Actions or the local Git checkout."""
    github_sha = os.getenv("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _generated_at() -> str:
    """Return one UTC ISO-8601 timestamp for the export run."""
    explicit = os.getenv("PNAD_GENERATED_AT", "").strip()
    if explicit:
        return explicit
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _artifact_stage(filename: str) -> str:
    if filename.startswith("eda_"):
        return "eda"
    if filename.startswith("paper_"):
        return "paper"
    return "metadata"


def _artifact_data_layer(filename: str) -> str:
    if filename.startswith("eda_refined_"):
        return "refined"
    if filename.startswith("eda_trusted_"):
        return "trusted"
    if filename.startswith("eda_compare_"):
        return "refined+trusted"
    if filename.startswith("eda_cleaning_"):
        return "refined_to_trusted"
    if filename.startswith("paper_gini_external_"):
        return "trusted+external"
    if filename.startswith("paper_"):
        return "trusted"
    return "derived"


def _describe_artifact(filename: str) -> str:
    """Return a concise scientific description for every generated artifact."""
    if filename in ARTIFACT_DESCRIPTIONS:
        return ARTIFACT_DESCRIPTIONS[filename]

    page_match = re.search(r"_page_(\d{2})\.png$", filename)
    page_text = f" Page {int(page_match.group(1))} of the paginated figure set." if page_match else ""

    stems = {
        "eda_refined_histogram_income": "Annual income histograms for the refined layer before statistical cleaning.",
        "eda_trusted_histogram_income": "Annual income histograms for the trusted layer after deterministic cleaning.",
        "eda_refined_boxplot_income": "Annual refined-layer income boxplots used to inspect dispersion and extreme values before cleaning.",
        "eda_trusted_boxplot_income": "Annual trusted-layer income boxplots used to inspect dispersion after cleaning.",
        "paper_ccdf_income_loglog": "Annual empirical income CCDFs in log-log coordinates for inspection of distributional shape and the upper tail.",
        "paper_ccdf_income_gompertz": "Annual Gompertz-transformed income CCDFs used as a complementary diagnostic of distributional form.",
        "paper_lorenz_income_annotated_g_p_k_z": "Annual Lorenz curves annotated with Gini G, Pietra P, Kolkata k, and Zanardi Z inequality measures.",
        "paper_ccdf_income_nominal_vs_adjusted_loglog": "Annual log-log comparison of nominal and monetarily adjusted income CCDFs.",
        "paper_distribution_regime_fit": "Annual dual-panel Gompertz-body and Pareto-tail profile fits reconstructed exclusively from persisted derived datasets.",
        "eda_trusted_selected_histogram_income": "Selected-year trusted-layer income histograms using the user-configured subplot grid.",
        "paper_selected_ccdf_income_loglog": "Selected-year empirical income CCDFs in log-log coordinates using the user-configured subplot grid.",
        "paper_selected_ccdf_income_gompertz": "Selected-year Gompertz-transformed income CCDFs using the user-configured subplot grid.",
        "paper_selected_lorenz_income_annotated_g_p_k_z": "Selected-year Lorenz curves annotated with Gini, Pietra, Kolkata, and Zanardi measures using the user-configured subplot grid.",
    }
    for stem, description in stems.items():
        if filename.startswith(stem):
            return description + page_text

    year_match = re.search(r"_(19\d{2}|20\d{2})(?:_|\.)", filename)
    year_text = f" for survey year {year_match.group(1)}" if year_match else ""
    if filename.startswith("eda_trusted_histogram_income_"):
        return f"Trusted-layer income histogram{year_text}, generated for detailed inspection of one selected year."
    if filename.startswith("paper_ccdf_income_") and filename.endswith("_loglog.png"):
        return f"Empirical trusted-income CCDF in log-log coordinates{year_text}."
    if filename.startswith("paper_ccdf_income_") and filename.endswith("_gompertz.png"):
        return f"Gompertz-transformed trusted-income CCDF{year_text}."
    if filename.startswith("paper_lorenz_income_") and "annotated_g_p_k_z" in filename:
        return f"Lorenz curve{year_text}, annotated with Gini, Pietra, Kolkata, and Zanardi measures."
    if filename.startswith("paper_ccdf_income_nominal_vs_adjusted_"):
        return f"Log-log comparison of nominal and adjusted income CCDFs{year_text}."

    return "Generated PNAD analytical artifact produced by the reproducible output pipeline."


def _canonical_output_path(path: Path, output_root: Path) -> str:
    """Return a repository-relative canonical path independent of runner location."""
    relative = path.resolve().relative_to(output_root.resolve())
    return (Path("outputs") / relative).as_posix()


def _artifact_url(canonical_path: str) -> str:
    repository = os.getenv("GITHUB_REPOSITORY", "ozsp12/pnad_income").strip() or "ozsp12/pnad_income"
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    branch = os.getenv("PNAD_MANIFEST_BRANCH", "main").strip() or "main"
    return f"{server}/{repository}/blob/{branch}/{canonical_path}"


def _rows(
    paths: Iterable[Path],
    category: str,
    *,
    output_root: Path,
    commit_sha: str,
    generated_at: str,
) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        canonical_path = _canonical_output_path(path, output_root)
        size_bytes = path.stat().st_size
        rows.append(
            {
                "category": category,
                "stage": _artifact_stage(path.name),
                "data_layer": _artifact_data_layer(path.name),
                "filename": path.name,
                "path": canonical_path,
                "size_bytes": size_bytes,
                "size_human": _human_size(size_bytes),
                "format": path.suffix.lower().lstrip("."),
                "description": _describe_artifact(path.name),
                "url": _artifact_url(canonical_path),
                "sha256": _sha256(path),
                "commit_sha": commit_sha,
                "generated_at": generated_at,
            }
        )
    return rows


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
    inequality_ncols: int = 3,
    histogram_bins: int = 100,
    dpi: int = 200,
    gini_references: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Persist trusted scientific outputs plus refined-versus-trusted EDA diagnostics."""
    paths = prepare_output_paths(output_root)
    manifest: list[dict[str, object]] = []
    commit_sha = _current_commit_sha()
    generated_at = _generated_at()

    def add_manifest_rows(saved_paths: Iterable[Path], category: str) -> None:
        manifest.extend(
            _rows(
                saved_paths,
                category,
                output_root=paths.root,
                commit_sha=commit_sha,
                generated_at=generated_at,
            )
        )

    indices = annual_inequality_indices(results.panel)
    income_groups = annual_income_group_totals(results.panel, value_col="income_adj")
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
        "paper_annual_income_groups_p80_p99_p100_2025_usd.csv": income_groups,
        "paper_ccdf_income_nominal_adjusted.parquet": results.ccdf_nominal_adjusted,
        "paper_distribution_regime_fits.csv": results.regime_fits,
        "paper_distribution_regime_curves.parquet": results.regime_curves,
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
    add_manifest_rows(saved_tables, "table")

    # The persisted datasets are the public interface to every regime figure.
    # Reloading them here prevents plotting from depending on the trusted panel
    # or on an in-memory estimator result that researchers cannot reproduce.
    persisted_regime_fits = pd.read_csv(paths.tables / "paper_distribution_regime_fits.csv")
    persisted_regime_curves = pd.read_parquet(paths.tables / "paper_distribution_regime_curves.parquet")

    scalar_figures = {
        "paper_inequality_gini_all_years.png": plot_gini_evolution(results.summary),
        "paper_inequality_top_income_shares_all_years.png": plot_top_income_shares(results.summary),
        "paper_income_groups_p80_p99_p100_absolute_2025_usd_all_years.png": plot_income_group_totals(
            income_groups,
            value_col="income_adj",
        ),
        "paper_inequality_extended_pietra_k_z_all_years.png": plot_extended_inequality_evolution(results.summary),
        "paper_inequality_indices_all_years.png": plot_primary_indices(indices),
        "paper_inequality_zanardi_all_years.png": plot_zanardi(indices),
        "paper_inequality_information_all_years.png": plot_information_indices(indices),
        "paper_inequality_gini_pietra_kolkata_relations.png": plot_kolkata_pietra_relationships(indices),
        "paper_inequality_pietra_kolkata_bound_all_years.png": plot_pietra_kolkata_bound(indices),
        "paper_inequality_gini_zanardi_phase.png": plot_gini_zanardi(indices),
        "paper_gompertz_parameter_B_all_years.png": plot_gompertz_parameter_history(persisted_regime_fits),
        "paper_pareto_alpha_all_years.png": plot_pareto_alpha_history(persisted_regime_fits),
        "paper_distribution_cutoff_all_years.png": plot_distribution_cutoff_history(persisted_regime_fits),
        "paper_distribution_regime_r2_all_years.png": plot_regime_r2_history(persisted_regime_fits),
    }
    if gini_references is not None and not gini_references.empty:
        scalar_figures["paper_gini_external_validation.png"] = plot_gini_validation(results.summary, gini_references)
    saved_figures = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in scalar_figures.items()]
    add_manifest_rows(saved_figures, "figure")

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
        add_manifest_rows(_save_pages(figures, paths.figures, stem, dpi), "figure")

    shared_ylim = _shared_positive_ylim(refined, trusted)
    eda_scalar = {
        "eda_refined_outlier_income_upper_tail_all_years.png": eda_refined.outlier_overview_figure(ylim=shared_ylim),
        "eda_trusted_outlier_income_upper_tail_all_years.png": eda_trusted.outlier_overview_figure(ylim=shared_ylim),
        "eda_compare_outlier_income_upper_tail_refined_trusted.png": eda_refined.compare_upper_tail_figure(eda_trusted),
    }
    saved_eda_scalar = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in eda_scalar.items()]
    add_manifest_rows(saved_eda_scalar, "figure")

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
            "paper_lorenz_income_annotated_g_p_k_z",
            plot_lorenz_grid(
                results.panel,
                years=years,
                nrows=complete_nrows,
                ncols=inequality_ncols,
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
        (
            "paper_distribution_regime_fit",
            plot_distribution_regime_fits(
                persisted_regime_fits,
                persisted_regime_curves,
                years=years,
                max_years_per_page=6,
            ),
        ),
    ]
    for stem, figures in paper_page_specs:
        add_manifest_rows(_save_pages(figures, paths.figures, stem, dpi), "figure")

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
            f"paper_lorenz_income_{year}_annotated_g_p_k_z.png": plot_lorenz_curve(results.panel, year, annotate=True),
            f"paper_ccdf_income_nominal_vs_adjusted_{year}_loglog.png": plot_measure_comparison(ccdf, year),
        }
        paths_saved = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in individual.items()]
        add_manifest_rows(paths_saved, "figure")

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
                "paper_selected_lorenz_income_annotated_g_p_k_z",
                plot_lorenz_grid(
                    results.panel,
                    years=selected,
                    nrows=grid_nrows,
                    ncols=grid_ncols,
                    annotate=True,
                ),
            ),
        ]
        for stem, figures in selected_specs:
            add_manifest_rows(_save_pages(figures, paths.figures, stem, dpi), "figure")

    manifest_frame = pd.DataFrame(manifest, columns=MANIFEST_COLUMNS)
    save_table(manifest_frame, paths.root / "manifest.csv")
    return manifest_frame
