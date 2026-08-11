"""Tools for the PNAD income-distribution research project."""

from .config import load_metadata
from .distributions import build_ccdf_by_year, compute_ccdf
from .inequality import gini, lorenz_curve, summary_statistics
from .preprocessing import adjust_income_to_2025, standardize_income_frame

__all__ = [
    "load_metadata",
    "compute_ccdf",
    "build_ccdf_by_year",
    "gini",
    "lorenz_curve",
    "summary_statistics",
    "adjust_income_to_2025",
    "standardize_income_frame",
]
