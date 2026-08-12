"""Compute and persist longitudinal inequality indices used by the manuscript."""

from pathlib import Path

import matplotlib.pyplot as plt

from pnad_income.advanced_inequality import annual_inequality_indices
from pnad_income.advanced_plotting import (
    plot_gini_zanardi,
    plot_information_indices,
    plot_kolkata_pietra_relationships,
    plot_pietra_kolkata_bound,
    plot_primary_indices,
    plot_zanardi,
)
from pnad_income.pipeline import PipelineConfig, run_pipeline


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

results = run_pipeline(PipelineConfig(database_path=ROOT / "dados_refined", start_year=1976, end_year=2025))
indices = annual_inequality_indices(results.panel, value_col="income", atkinson_epsilon=0.5)
indices.to_csv(TABLES / "annual_inequality_indices.csv", index=False)

figures = {
    "inequality_indices_all_years.png": plot_primary_indices(indices),
    "zanardi_index_all_years.png": plot_zanardi(indices),
    "information_indices_all_years.png": plot_information_indices(indices),
    "gini_pietra_kolkata_relations.png": plot_kolkata_pietra_relationships(indices),
    "pietra_kolkata_bound_all_years.png": plot_pietra_kolkata_bound(indices),
    "gini_zanardi_phase.png": plot_gini_zanardi(indices),
}
for filename, fig in figures.items():
    fig.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

print(indices.to_string(index=False))
print(f"Saved {len(indices)} annual rows and {len(figures)} publication figures.")
