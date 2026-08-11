# Brazilian PNAD Income Dataset, 1976–2025

This repository contains a harmonized longitudinal dataset of Brazilian household per-capita income derived from the **Pesquisa Nacional por Amostra de Domicílios (PNAD)** and **PNAD Contínua**, together with the metadata and analytical code used to characterize the resulting income distributions from 1976 to 2025.

The repository accompanies a data-descriptor manuscript under preparation for *Scientific Data*. Its primary purpose is scientific documentation and reproducibility: to make explicit how income information collected under changing survey instruments, variable definitions, file layouts, currencies, and historical monetary regimes was transformed into a consistent annual series suitable for distributional and inequality analysis.

## Scientific scope

The database covers **45 survey years** between 1976 and 2025. No annual record is included for 1980, 1991, 1994, 2000, or 2010, corresponding to years without observations in the assembled PNAD series. The historical PNAD and PNAD Contínua files are heterogeneous: the location and definition of the relevant income field change across years, and in several early surveys household income must be combined with the number of household residents to obtain a per-capita measure.

The harmonization procedure therefore treats each survey year as a documented measurement instance rather than assuming a fixed column schema through time. The year-specific variable map, fixed-width positions, missing-value conventions, monetary units, exchange factors, price indices, and source locations are recorded in [`config/pnad_metadata.csv`](config/pnad_metadata.csv).

## Data records

The analytical records are stored in [`dados_refined/`](dados_refined/) as one Parquet file for each available survey year:

```text
pnad_refined_1976.parquet
pnad_refined_1977.parquet
...
pnad_refined_2025.parquet
```

Every current annual file has the same two-field schema:

| Field | Meaning |
|---|---|
| `ano` | Survey year |
| `renda` | Refined household per-capita income measure for that survey year |

The files preserve these Portuguese field names as part of the data record. The analytical package maps them in memory to `year` and `income`; the Parquet records themselves are not rewritten during analysis.

The current release contains a single longitudinal income measure. In particular, the refined 2016–2025 files do **not** contain a second effective-income field such as `VD4020`. Analyses requiring a distinct habitual-versus-effective income comparison therefore require an extended data release and are not inferred from the present records.

## Construction of the longitudinal income series

For each year, the PNAD variable used to represent household per-capita income is identified in the metadata. Where the source survey already contains the required per-capita income variable, that quantity is retained after cleaning. Where household income and household size are stored separately, per-capita income is constructed as

\[
y_i=\frac{Y_i}{n_i},
\]

where \(Y_i\) is household income and \(n_i>0\) is the corresponding number of household residents.

Historical survey-specific sentinel values used for missing income observations are removed before analysis. Household-size observations that cannot support a valid per-capita calculation are excluded from the corresponding transformation. Zero income, when present as a valid observation rather than a sentinel code, is retained in the analytical population.

## Monetary harmonization

Brazil underwent multiple currency regimes during the period covered by the database. To permit intertemporal comparison, the analytical pipeline attaches the year-specific monetary information stored in the metadata and constructs an income measure expressed relative to a common 2025 reference. The implemented transformation is

\[
y_{2025}=\frac{y}{E}I_{2025},
\]

where \(y\) denotes the survey-year income value, \(E\) is the corresponding exchange factor, and \(I_{2025}\) is the inflation adjustment factor to the 2025 reference.

Nominal survey-year income and the harmonized measure are retained as distinct analytical quantities. This distinction is important because the monetary transformation changes the scale used for intertemporal comparison but does not change the ordering of observations within a given year.

## Distributional characterization

The dataset is evaluated using complementary approaches that describe different aspects of the empirical distribution. Annual histograms provide a direct inspection of frequency structure and extreme values. The complementary cumulative distribution function is evaluated as

\[
\widehat{\overline F}(x)
=
\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(X_i\ge x),
\]

with geometrically spaced positive thresholds. Finite zero-income observations remain in the denominator \(N\); consequently, the estimator represents the unconditional probability \(P(X\ge x)\), rather than the conditional quantity \(P(X\ge x\mid X>0)\).

Income concentration is summarized through Lorenz curves and the Gini coefficient. For ordered nonnegative observations \(x_{(1)}\le\cdots\le x_{(N)}\), the implemented unweighted estimator is

\[
G=
\frac{2\sum_{i=1}^{N} i x_{(i)}}{N\sum_{i=1}^{N}x_{(i)}}
-
\frac{N+1}{N}.
\]

The combination of histograms, CCDFs, Lorenz curves, annual moments, and Gini coefficients provides both a quality-control layer and a compact description of the evolution of the income distribution across survey waves.

## Technical validation

Validation is performed at three levels. First, the annual records are checked for schema consistency, year identity, numeric income values, and correspondence between filenames and survey years. Second, the harmonized panel is checked for temporal coverage and the availability of the monetary metadata required for each survey wave. Third, the resulting distributions are examined through annual summary statistics, CCDFs, histograms, Lorenz curves, and Gini coefficients.

The current repository contains **45 annual Parquet files**, all sharing the schema `renda, ano`. Automated tests cover database loading, schema normalization, preprocessing, CCDF construction, and inequality calculations. The complete analytical workflow is additionally executed from a clean kernel through the single research notebook described below.

## Reproducible analysis

[`notebooks/pnad_income_pipeline.ipynb`](notebooks/pnad_income_pipeline.ipynb) is the sole analytical notebook. It is intended as a readable scientific record of the analysis rather than as the location of the implementation. The notebook sequentially loads and validates the annual records, attaches monetary metadata, constructs the 2025-reference income measure, presents annual descriptive statistics, evaluates inequality, constructs empirical CCDFs, and reports final data-quality diagnostics.

The numerical implementation is encapsulated in [`src/pnad_income/`](src/pnad_income/). This separation ensures that quantities shown in the notebook are generated by documented, testable functions rather than by hidden notebook state or duplicated year-specific code.

A minimal local reproduction is:

```bash
pip install -e ".[notebooks]"
cd notebooks
jupyter nbconvert --to notebook --execute pnad_income_pipeline.ipynb \
  --output pnad_income_pipeline_executed.ipynb
```

The notebook reads `../dados_refined` by default.

## Repository contents

[`dados_refined/`](dados_refined/) contains the annual analytical records. [`config/pnad_metadata.csv`](config/pnad_metadata.csv) contains the longitudinal variable map and monetary metadata. [`docs/methodology.md`](docs/methodology.md) records the mathematical and processing definitions, while [`docs/database_schema.md`](docs/database_schema.md) specifies the data-record interface. [`src/pnad_income/`](src/pnad_income/) contains the reusable analytical implementation, and [`tests/`](tests/) contains reproducibility checks for the core transformations and estimators.

## Intended use

The dataset is intended for longitudinal studies of Brazilian income distributions, inequality, distributional tails, monetary harmonization, and related statistical or econophysics analyses. Because survey instruments and variable definitions change over the five-decade interval, analyses should retain the year-specific metadata and should not interpret the series as if it originated from a single invariant survey design.

When using the data for additional research, the annual refined records should be treated as the primary analytical data product and `config/pnad_metadata.csv` as the corresponding provenance and harmonization record.

## Author

**Dr. Osvaldo L. Santos-Pereira** — [Academic webpage](https://ozsp12.github.io/) · [Lattes](http://lattes.cnpq.br/6730251976463283) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [Google Scholar](https://scholar.google.com/citations?user=HIZp0X8AAAAJ&hl=en) · [ResearchGate](https://www.researchgate.net/profile/Osvaldo-Santos-Pereira) · [GitHub](https://github.com/ozsp12) · [LinkedIn](https://www.linkedin.com/in/ozsp12)

## Manuscript and citation

A formal citation should use the associated data-descriptor article once its bibliographic record or DOI is available. No DOI is assigned in this repository at present.
