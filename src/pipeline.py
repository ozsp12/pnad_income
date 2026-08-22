"""Thin orchestration layer for the trusted PNAD income analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis import compare_income_measures_ccdf, summary_statistics
from data import DEFAULT_METADATA_PATH, DEFAULT_TRUSTED_PATH, prepare_panel as prepare_data_panel
from regime_analysis import RegimeFitConfig, fit_distribution_regimes


@dataclass(frozen=True)
class PipelineConfig:
    """Parameters required to execute the scientific analysis on trusted data."""

    database_path: str | Path = DEFAULT_TRUSTED_PATH
    metadata_path: str | Path = DEFAULT_METADATA_PATH
    ccdf_base: float = 1.05
    start_year: int | None = None
    end_year: int | None = None
    regime_min_body_observations: int = 100
    regime_min_tail_observations: int = 100
    regime_min_tail_fraction: float = 0.005
    regime_cutoff_quantile_min: float = 0.20
    regime_cutoff_quantile_max: float = 0.995
    regime_selection_criterion: str = "log_likelihood"
    regime_gompertz_intercept_mode: str = "fixed"


@dataclass
class PipelineResults:
    """Core analytical products returned by :func:`run_pipeline`."""

    panel: pd.DataFrame
    summary: pd.DataFrame
    ccdf: pd.DataFrame
    regime_fits: pd.DataFrame
    regime_curves: pd.DataFrame
    data_layer: str = "trusted"

    @property
    def years(self) -> list[int]:
        return sorted(self.panel["year"].dropna().astype(int).unique().tolist())

    @property
    def ccdf_nominal_adjusted(self) -> pd.DataFrame:
        """Compatibility view for nominal and 2025-adjusted income."""
        if self.ccdf.empty:
            return self.ccdf.copy()
        return self.ccdf.loc[self.ccdf["measure"].isin(["income", "income_adj"])].reset_index(drop=True)

    @property
    def ccdf_habitual_effective(self) -> pd.DataFrame:
        """Compatibility view for habitual versus effective income."""
        if "income_effective" not in self.panel.columns or self.ccdf.empty:
            return pd.DataFrame()
        years = self.panel.loc[self.panel["income_effective"].notna(), "year"].unique()
        return self.ccdf.loc[
            self.ccdf["year"].isin(years)
            & self.ccdf["measure"].isin(["income", "income_effective"])
        ].reset_index(drop=True)


def prepare_panel(config: PipelineConfig) -> pd.DataFrame:
    """Prepare the trusted harmonized panel defined by ``config``."""
    return prepare_data_panel(
        config.database_path,
        metadata_path=config.metadata_path,
        start_year=config.start_year,
        end_year=config.end_year,
    )


def run_pipeline(config: PipelineConfig) -> PipelineResults:
    """Load trusted data and compute annual statistics and distribution tables."""
    panel = prepare_panel(config)
    summary = summary_statistics(panel)
    measures = tuple(
        column
        for column in ("income", "income_adj", "income_effective", "income_effective_adj")
        if column in panel.columns and panel[column].notna().any()
    )
    ccdf = compare_income_measures_ccdf(panel, measures=measures, base=config.ccdf_base)
    regime_config = RegimeFitConfig(
        ccdf_base=config.ccdf_base,
        min_body_observations=config.regime_min_body_observations,
        min_tail_observations=config.regime_min_tail_observations,
        min_tail_fraction=config.regime_min_tail_fraction,
        cutoff_quantile_min=config.regime_cutoff_quantile_min,
        cutoff_quantile_max=config.regime_cutoff_quantile_max,
        selection_criterion=config.regime_selection_criterion,
        gompertz_intercept_mode=config.regime_gompertz_intercept_mode,
    )
    regime_fits, regime_curves = fit_distribution_regimes(
        panel,
        value_col="income_adj",
        config=regime_config,
    )
    return PipelineResults(
        panel=panel,
        summary=summary,
        ccdf=ccdf,
        regime_fits=regime_fits,
        regime_curves=regime_curves,
        data_layer="trusted",
    )


def pipeline_overview(results: PipelineResults) -> pd.DataFrame:
    """Return compact coverage diagnostics for an executed pipeline."""
    years = results.years
    effective_years = (
        results.panel.loc[results.panel["income_effective"].notna(), "year"].nunique()
        if "income_effective" in results.panel.columns
        else 0
    )
    return pd.DataFrame(
        {
            "metric": [
                "data_layer",
                "observations",
                "first_year",
                "last_year",
                "number_of_years",
                "years_with_effective_income",
                "ccdf_rows",
                "years_with_valid_regime_fit",
                "regime_curve_rows",
            ],
            "value": [
                results.data_layer,
                len(results.panel),
                min(years),
                max(years),
                len(years),
                int(effective_years),
                len(results.ccdf),
                int(results.regime_fits["fit_status"].astype(str).ne("no_valid_fit").sum()),
                len(results.regime_curves),
            ],
        }
    )
