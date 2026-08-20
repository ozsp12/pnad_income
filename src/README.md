# Source code

The `src` directory contains the computational implementation used to prepare the harmonized PNAD panel, diagnose data quality, estimate distributional and inequality statistics, generate figures, and persist reproducible outputs. The modules are intentionally kept at one level so that the analytical flow can be inspected without navigating an additional package hierarchy.

| File | Responsibility |
|---|---|
| [`data.py`](data.py) | Data access, metadata validation, survey-specific missing-value handling, optional legacy cuts, and monetary harmonization. |
| [`descriptive.py`](descriptive.py) | `DescriptiveStatistics`: annual descriptive statistics, exact-value frequencies, sentinel candidates, outlier diagnostics, histograms, boxplots, and candidate cutoff inspection. |
| [`analysis.py`](analysis.py) | Empirical CCDF construction, Lorenz curves, inequality indices, and external Gini validation. |
| [`plotting.py`](plotting.py) | Scientific plots for CCDF, Lorenz, concentration, and inequality analyses. |
| [`pipeline.py`](pipeline.py) | `PipelineConfig` and `PipelineResults`, plus orchestration of the analytical pipeline. |
| [`outputs.py`](outputs.py) | `OutputPaths` and persistence of EDA and paper-oriented tables, figures, and the output manifest. |
| [`cli.py`](cli.py) | Command-line entry point exposed as `pnad-income`. |

The EDA layer does not delete extreme observations automatically. Sentinel codes declared in metadata and additional repeated upper-tail values are reported as candidates for inspection; exclusions remain an explicit preprocessing decision.
