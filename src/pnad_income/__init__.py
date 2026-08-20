"""Core interface for the PNAD income analysis package."""

from .analysis import compute_ccdf, gini, lorenz_curve
from .data import load_metadata
from .pipeline import PipelineConfig, PipelineResults, pipeline_overview, run_pipeline

__all__ = [
    "PipelineConfig",
    "PipelineResults",
    "run_pipeline",
    "pipeline_overview",
    "load_metadata",
    "compute_ccdf",
    "gini",
    "lorenz_curve",
]
