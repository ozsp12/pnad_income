# Analytical Database Schema

The repository stores the analytical database as annual Parquet files under `dados_refined/`.

## File convention

```text
dados_refined/
├── pnad_refined_1976.parquet
├── pnad_refined_1977.parquet
├── ...
└── pnad_refined_2025.parquet
```

Only survey years with available observations need to be present. When a Parquet file does not contain a `year` column, the loader extracts the four-digit year from its filename and inserts it explicitly before concatenating the panel.

## Required analytical column

| Column | Type | Definition |
|---|---|---|
| `income` | numeric | Longitudinal income measure used in the annual distribution analysis. |

A `year` column is recommended. For files following the naming convention above, it can be inferred from the filename.

## Optional analytical columns

| Column | Type | Definition |
|---|---|---|
| `income_effective` | numeric | Effective-income measure when supplied by the survey processing stage. |
| `income_adj` | numeric | Adjusted income; recomputed by the pipeline when monetary factors are available. |
| `income_effective_adj` | numeric | Adjusted effective income. |
| `exchange` | numeric | Year-specific exchange factor. |
| `price_index` | numeric | Year-specific price index retained for validation. |
| `inflation_to_2025` | numeric | Multiplicative factor to the 2025 reference. |

Missing monetary columns are attached from `config/pnad_metadata.csv` by year.

## Missing values and zero income

Survey sentinel codes must be converted to standard missing values during preprocessing. Zero income is valid data and must not be converted to missing. Zero observations remain in the population used to normalize the CCDF.

## Alternative database location

`PipelineConfig.database_path` may point either to the `dados_refined/` directory or to a single Parquet, CSV, Feather, Pickle, or Excel file. The environment variable `PNAD_DATABASE_PATH` can override the default notebook path.
