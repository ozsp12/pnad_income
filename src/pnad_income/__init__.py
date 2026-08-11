"""Public interface for the PNAD income-distribution research package."""

from .config import load_metadata
from .distributions import build_ccdf_by_year, compute_ccdf
from .inequality import gini, lorenz_curve, summary_statistics
from .pipeline import PipelineConfig, PipelineResults, pipeline_overview, run_pipeline
from .preprocessing import adjust_income_to_2025, standardize_income_frame

__all__ = [
    "PipelineConfig",
    "PipelineResults",
    "run_pipeline",
    "pipeline_overview",
    "load_metadata",
    "compute_ccdf",
    "build_ccdf_by_year",
    "gini",
    "lorenz_curve",
    "summary_statistics",
    "adjust_income_to_2025",
    "standardize_income_frame",
]
