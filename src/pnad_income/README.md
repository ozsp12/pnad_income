# `pnad_income` source package

This directory contains the computational core of the PNAD income project. The package is intentionally small: data preparation, statistical analysis, visualization, pipeline orchestration, and output persistence are separated without creating one module per minor responsibility.

## Structure

```text
src/pnad_income/
├── __init__.py   # small public API
├── data.py       # metadata, loading, validation, preprocessing, raw/refined data I/O
├── analysis.py   # distributions, inequality measures, annual statistics, Gini validation
├── plotting.py   # all scientific figures
├── pipeline.py   # PipelineConfig, PipelineResults, and run_pipeline
├── outputs.py    # tables, figures, reports, and manifest persistence
└── README.md     # this file
```

Generated packaging metadata such as `src/pnad_income.egg-info/` is not source code and should not be versioned.

## Files

### `__init__.py`

Defines the small public interface of the package. It exposes the pipeline and a few commonly used analytical functions without re-exporting every internal helper.

### `data.py`

Contains all operations required before statistical analysis:

- loading `metadata/pnad_metadata.csv`;
- mapping stored Portuguese fields such as `ano` and `renda` to the internal schema;
- reading refined Parquet files or supported tabular files;
- validating the analytical schema;
- attaching monetary metadata;
- converting income to the 2025 reference;
- optional reproduction of the historical 1976–1979 outlier cuts;
- reading raw fixed-width PNAD files and writing refined annual Parquet files.

This is the only module that should know about file formats, metadata paths, field aliases, or preprocessing rules.

### `analysis.py`

Contains the numerical/statistical analysis:

- empirical CCDF and geometric bins;
- Gini coefficient and Lorenz curve;
- Pietra and Kolkata indices;
- Zanardi asymmetry index;
- explicitly named `legacy_z_statistic` for the historical `pnad.py` construction;
- top-income shares;
- Theil, Atkinson, Shannon, and Herfindahl measures;
- annual descriptive and inequality tables;
- loading and comparing documented external Gini reference series.

Each scientific quantity has one implementation only. In particular, the historical Z statistic is kept separate from the canonical Zanardi implementation.

### `plotting.py`

Contains all visualization functions used by the notebook and exporter:

- histograms;
- CCDF plots;
- Lorenz curves;
- nominal-versus-adjusted comparisons;
- Gini, top-income, Pietra, Kolkata, Zanardi, Theil, Atkinson, and Shannon evolution plots;
- external Gini validation plots.

Individual-year and multi-year/grid figures share the same internal helpers.

### `pipeline.py`

Provides the high-level execution interface:

```python
config = PipelineConfig(database_path="dados_refined", start_year=1976, end_year=2025)
results = run_pipeline(config)
```

`run_pipeline` delegates data preparation to `data.py` and statistical calculations to `analysis.py`. It returns a `PipelineResults` object containing:

- `panel`: harmonized observation-level data;
- `summary`: annual descriptive and inequality statistics;
- `ccdf`: one long-format CCDF table for all available income measures.

Compatibility properties expose nominal/adjusted and habitual/effective CCDF views when needed.

### `outputs.py`

Persists reproducible analysis products under `outputs/`:

- CSV/Parquet tables;
- publication figures;
- diagnostics;
- external-validation products when references are available;
- `manifest.csv` describing generated artifacts.

It does not implement scientific calculations; it only coordinates analysis results and plotting functions for persistence.

## Dependency flow

```text
metadata/ + dados_refined/
          │
          ▼
       data.py
          │
          ▼
      analysis.py
          │
          ▼
      pipeline.py
       │       │
       ▼       ▼
 plotting.py  outputs.py
       │       │
       └── notebook/workflow
```

The notebook should remain the scientific presentation layer. Numerical definitions belong in `analysis.py`, data rules in `data.py`, and generated artifacts in `outputs.py`.
