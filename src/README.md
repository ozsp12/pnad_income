# Source code

The `src` directory contains the computational implementation for data access, trusted-layer construction, exploratory diagnostics, distributional analysis, plotting, pipeline orchestration, and artifact persistence. The modules remain at one level so the analytical flow can be inspected without an additional package hierarchy.

| File | Responsibility |
|---|---|
| [`data.py`](data.py) | Data-layer paths, metadata loading, annual refined/trusted file access, schema validation, raw-to-refined preparation, and monetary harmonization. |
| [`descriptive.py`](descriptive.py) | `DescriptiveStatistics` provides annual EDA, exact-value frequencies, metadata-sentinel counts, histograms, boxplots, and upper-tail diagnostics. `IncomeDataCleaner` applies the deterministic refined-to-trusted cleaning rule, creates mutually exclusive quality flags, records annual thresholds, and materializes trusted Parquet files. |
| [`analysis.py`](analysis.py) | Empirical CCDF construction, Lorenz curves, inequality indices, and external Gini validation. |
| [`plotting.py`](plotting.py) | Scientific plots for CCDF, Lorenz, concentration, and inequality analyses. |
| [`pipeline.py`](pipeline.py) | `PipelineConfig` and `PipelineResults`, plus orchestration of scientific analyses using the trusted data layer. |
| [`outputs.py`](outputs.py) | Flat `figures/` and `tables/` persistence using `eda_` and `paper_` filename prefixes, including refined-versus-trusted diagnostics and the manifest. |
| [`cli.py`](cli.py) | Command-line entry point exposed as `pnad-income`; it rebuilds trusted data before invoking the scientific pipeline. |

The cleaning order is explicit. Metadata sentinels are flagged first and excluded from the sample used to estimate statistical thresholds. A record with `flag_metadata_sentinel = 1` must therefore have `flag_statistical_outlier = 0`. Statistical upper-tail flags are evaluated only for the remaining observations. The default method is `log_mad`: the annual rule is estimated on `log1p(income)` using a scaled median absolute deviation and a configurable multiplier. Alternative deterministic methods `mad` and `mean_std` are available for sensitivity checks. Repeated exact values remain an EDA diagnostic rather than an additional record-level flag.
