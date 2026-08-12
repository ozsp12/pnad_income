"""Publication-oriented figures for longitudinal inequality measures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _frame(indices: pd.DataFrame) -> pd.DataFrame:
    if "year" not in indices.columns:
        raise KeyError("Column 'year' is required.")
    return indices.sort_values("year").copy()


def _survey_transition(ax) -> None:
    ax.axvline(2015.5, linestyle="--", linewidth=1.0, alpha=0.55)
    ymin, ymax = ax.get_ylim()
    ax.text(2015.7, ymax - 0.05 * (ymax - ymin), "PNAD Continua", fontsize=9, va="top")


def plot_primary_indices(indices: pd.DataFrame, figsize=(11.0, 5.8)):
    """Plot Gini, Pietra, and Kolkata indices for every available year."""
    frame = _frame(indices)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frame["year"], frame["gini"], marker="o", markersize=3.5, linewidth=1.4, label="Gini")
    ax.plot(frame["year"], frame["pietra"], marker="s", markersize=3.2, linewidth=1.3, label="Pietra")
    ax.plot(frame["year"], frame["kolkata"], marker="^", markersize=3.4, linewidth=1.3, label="Kolkata")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index value")
    ax.set_title("Long-run evolution of Lorenz-based inequality indices")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    _survey_transition(ax)
    fig.tight_layout()
    return fig


def plot_zanardi(indices: pd.DataFrame, figsize=(11.0, 5.4)):
    """Plot the Zanardi asymmetry index for every available year."""
    frame = _frame(indices)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.plot(frame["year"], frame["zanardi"], marker="o", markersize=3.8, linewidth=1.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Zanardi index")
    ax.set_title("Lorenz-curve asymmetry measured by the Zanardi index")
    ax.grid(True, alpha=0.25)
    _survey_transition(ax)
    fig.tight_layout()
    return fig


def plot_information_indices(indices: pd.DataFrame, figsize=(11.0, 5.8)):
    """Plot Theil, Atkinson, and normalized Shannon inequality measures."""
    frame = _frame(indices)
    atkinson_cols = [c for c in frame.columns if c.startswith("atkinson_")]
    if not atkinson_cols:
        raise KeyError("No Atkinson column is available.")
    acol = atkinson_cols[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(frame["year"], frame["theil"], marker="o", markersize=3.4, linewidth=1.3, label="Theil T")
    ax.plot(frame["year"], frame[acol], marker="s", markersize=3.2, linewidth=1.3, label=f"Atkinson ($\\epsilon={acol.split('_',1)[1]}$)")
    ax.plot(frame["year"], frame["shannon_inequality"], marker="^", markersize=3.3, linewidth=1.3, label="Normalized Shannon deficit")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index value")
    ax.set_title("Information-theoretic and welfare-sensitive inequality measures")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    _survey_transition(ax)
    fig.tight_layout()
    return fig


def plot_kolkata_pietra_relationships(indices: pd.DataFrame, figsize=(12.4, 5.2)):
    """Compare empirical Gini-Pietra-Kolkata relations with small-G baselines."""
    frame = _frame(indices)
    ggrid = np.linspace(0.0, max(0.9, float(frame["gini"].max()) * 1.04), 250)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.scatter(frame["gini"], frame["pietra"], s=28, alpha=0.8, label="PNAD years")
    ax.plot(ggrid, 0.75 * ggrid, linestyle="--", linewidth=1.3, label=r"$P=3G/4$")
    ax.set_xlabel("Gini coefficient, $G$")
    ax.set_ylabel("Pietra index, $P$")
    ax.set_title("Pietra-Gini relation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.scatter(frame["gini"], frame["kolkata"], s=28, alpha=0.8, label="PNAD years")
    ax.plot(ggrid, 0.5 + 0.375 * ggrid, linestyle="--", linewidth=1.3, label=r"$K=1/2+3G/8$")
    ax.set_xlabel("Gini coefficient, $G$")
    ax.set_ylabel("Kolkata index, $K$")
    ax.set_title("Kolkata-Gini relation")
    ax.set_ylim(0.5, 0.9)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_pietra_kolkata_bound(indices: pd.DataFrame, figsize=(10.0, 5.4)):
    """Plot the empirical ratio testing the rigorous bound P >= 2K-1."""
    frame = _frame(indices)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(1.0, linestyle="--", linewidth=1.1, label=r"Bound: $P/(2K-1)=1$")
    ax.plot(frame["year"], frame["pietra_over_kolkata_excess"], marker="o", markersize=3.6, linewidth=1.3, label="PNAD")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$P/(2K-1)$")
    ax.set_title("Geometric relation between Pietra and Kolkata indices")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _survey_transition(ax)
    fig.tight_layout()
    return fig


def plot_gini_zanardi(indices: pd.DataFrame, figsize=(7.2, 5.8)):
    """Show how Lorenz asymmetry varies at comparable Gini levels."""
    frame = _frame(indices)
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(frame["gini"], frame["zanardi"], c=frame["year"], s=40)
    ax.axhline(0.0, linewidth=1.0, alpha=0.55)
    ax.set_xlabel("Gini coefficient")
    ax.set_ylabel("Zanardi index")
    ax.set_title("Concentration and Lorenz asymmetry")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Year")
    fig.tight_layout()
    return fig
