# Brazilian PNAD Income Dataset, 1976–2025

This repository contains a harmonized longitudinal dataset of Brazilian household per-capita income derived from PNAD and PNAD Contínua, together with the code used to analyze income distributions and inequality from 1976 to 2025.

The analytical database contains 45 survey years. No record is available for 1980, 1991, 1994, 2000, or 2010.

## Repository structure

```text
pnad_income/
├── dados_refined/              # annual analytical Parquet files
├── metadata/                   # year-level metadata and validation references
├── notebooks/                  # single scientific analysis notebook
├── src/pnad_income/            # reusable implementation
├── tests/                      # automated tests
├── docs/                       # methodology and schema documentation
└── outputs/                    # generated tables, figures, and reports
```

The annual records in `dados_refined/` use the published schema:

| Field | Meaning |
|---|---|
| `ano` | survey year |
| `renda` | refined household per-capita income |

The package maps these fields to `year` and `income` in memory. The files themselves are not rewritten during analysis.

## Metadata and monetary harmonization

[`metadata/pnad_metadata.csv`](metadata/pnad_metadata.csv) records the year-specific survey variable, fixed-width location where applicable, missing-value convention, currency, exchange factor, inflation factor, and source location.

Income is expressed relative to the 2025 reference through

$$
y_{2025}=\frac{y}{E}I_{2025},
$$

where $y$ is the survey-year income, $E$ is the corresponding exchange factor, and $I_{2025}$ is the inflation adjustment factor.

External annual Gini series used for validation belong under [`metadata/gini_references/`](metadata/gini_references/). Reference files must include documented provenance rather than hard-coded numerical arrays.

## Analysis

[`notebooks/pnad_income_pipeline.ipynb`](notebooks/pnad_income_pipeline.ipynb) is the single analysis notebook. Numerical implementation remains in [`src/pnad_income/`](src/pnad_income/); the notebook only configures, executes, displays, and exports the analysis.

The pipeline produces annual descriptive statistics, Gini, Pietra, Kolkata and Zanardi measures, top-income shares, Theil and Atkinson indices, normalized Shannon and Herfindahl measures, histograms, CCDFs, Lorenz curves, nominal-versus-adjusted comparisons, diagnostics, and optional external Gini validation.

All persistent products are generated through `pnad_income.outputs.export_analysis_outputs`. There is no second export script or duplicated analysis path.

## Reproduction

From the repository root:

```bash
python -m pip install -e ".[dev,notebooks]"
pytest -q
cd notebooks
jupyter nbconvert \
  --to notebook \
  --execute pnad_income_pipeline.ipynb \
  --output ../outputs/reports/pnad_income_pipeline_executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

For interactive work, install the package first and then open the notebook with Jupyter. The editable installation is required because the project uses the standard `src/` package layout.

## Distributional definitions

For nonnegative income $X$, the empirical complementary cumulative distribution function is

$$
\widehat{\overline F}(x)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(X_i\ge x).
$$

Finite zero-income observations remain in the denominator. For ordered nonnegative observations $x_{(1)}\le\cdots\le x_{(N)}$, the unweighted Gini estimator is

$$
G=\frac{2\sum_{i=1}^{N} i x_{(i)}}{N\sum_{i=1}^{N}x_{(i)}}-\frac{N+1}{N}.
$$

The current analysis is record-weighted. Population inference from PNAD requires the appropriate survey weights and design information.

## Documentation

[`docs/methodology.md`](docs/methodology.md) contains the analytical definitions and transformations. [`docs/database_schema.md`](docs/database_schema.md) specifies the released data-record interface.

## Author

**Dr. Osvaldo L. Santos-Pereira** — [Academic webpage](https://ozsp12.github.io/) · [Lattes](http://lattes.cnpq.br/6730251976463283) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [Google Scholar](https://scholar.google.com/citations?user=HIZp0X8AAAAJ&hl=en) · [ResearchGate](https://www.researchgate.net/profile/Osvaldo-Santos-Pereira) · [GitHub](https://github.com/ozsp12) · [LinkedIn](https://www.linkedin.com/in/ozsp12)
