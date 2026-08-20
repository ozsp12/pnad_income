# PNAD Income Distribution in Brazil, 1976–2025

This repository contains a harmonized longitudinal dataset and a reproducible computational pipeline for the study of Brazilian income distributions from 1976 to 2025. The project combines annual PNAD and PNAD Contínua microdata with year-specific extraction metadata, deterministic data-quality treatment, monetary harmonization, empirical distribution analysis, and inequality measures. The analytical series contains 45 survey years; 1980, 1991, 1994, 2000, and 2010 are absent because no compatible PNAD observation is available for those years.

## Data and sources

The microdata are obtained from the Instituto Brasileiro de Geografia e Estatística (IBGE). Annual PNAD files through 2015 are drawn from the [PNAD annual microdata archive](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_anual/microdados/), while observations from 2016 onward use [PNAD Contínua quarterly microdata](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/). The exact survey field, fixed-width position when applicable, missing-value code, original currency, exchange factor, price index, inflation factor, and source URL for each year are recorded in [`data/metadata/pnad_metadata.csv`](data/metadata/pnad_metadata.csv).

The repository uses a four-layer data layout. [`data/raw/`](data/raw/) is reserved for original source files and is not intended for repository storage. [`data/refined/`](data/refined/) contains structurally harmonized annual Parquet files. [`data/trusted/`](data/trusted/) is rebuilt deterministically from the refined layer after data-quality treatment and is the only data layer used by scientific analyses. [`data/metadata/`](data/metadata/) contains the annual extraction, sentinel, currency, exchange-rate, inflation, and provenance metadata required to reproduce both preparation and cleaning.

Historical exchange-rate information comes from the Banco Central do Brasil through its [historical exchange-rate service](https://www.bcb.gov.br/estabilidadefinanceira/historicocotacoes) and [open exchange-rate dataset](https://dadosabertos.bcb.gov.br/dataset/taxas-de-cambio-todos-os-boletins-diarios). U.S. inflation adjustment uses the [CPIAUCSL series](https://fred.stlouisfed.org/series/CPIAUCSL) from BLS/FRED.

## Trusted-layer construction

The refined-to-trusted transition is implemented in [`src/descriptive.py`](src/descriptive.py) by `IncomeDataCleaner`. Metadata sentinels are identified first. Sentinel observations are excluded from the sample used to estimate statistical thresholds, and the two record-level quality flags are mutually exclusive by construction: a row with `flag_metadata_sentinel = 1` must have `flag_statistical_outlier = 0`. Repeated exact values remain part of the exploratory frequency analysis and are not promoted to a third record-level flag.

The default statistical rule is an annual robust upper-tail cutoff on transformed income. For non-sentinel observations, the cleaner evaluates

$$
z_i=\log(1+y_i),\qquad c=\operatorname{median}(z),\qquad s=1.4826\,\operatorname{median}|z-c|,
$$

and defines the upper cutoff as

$$
y_{\max}=\exp(c+k s)-1,
$$

with default $k=6$. The multiplier and method are explicit CLI parameters; `mad` and `mean_std` are available as deterministic alternatives for sensitivity checks. The annual thresholds, numbers of sentinels and statistical outliers, removal rates, and final trusted sample sizes are written to EDA audit tables. This keeps the rule in code, the realized parameters in outputs, and the trusted datasets reproducible without maintaining a separate spreadsheet of manual cutoffs.

The trusted annual files are materialized as `pnad_trusted_<year>.parquet`. The scientific pipeline reads this layer exclusively; CCDF, Lorenz, inequality, and paper-oriented outputs therefore cannot bypass the cleaning step. Refined-versus-trusted EDA products are retained to show the effect of the transformation before interpretation of the scientific results.

## Monetary harmonization

For cross-year monetary comparison, nominal local-currency income is converted using the stored year-level exchange factor and then adjusted to the September 2025 U.S. CPI level,

$$
y^{(2025)}_{it}=\frac{y_{it}}{E_t}\frac{P_{2025}}{P_t}.
$$

This produces a project-level series in 2025 U.S. dollars and is not the official IBGE real-income deflation procedure. Within a given year the adjustment is a positive multiplicative rescaling, so rank order, Lorenz geometry, and scale-invariant inequality measures are unchanged; nominal and adjusted CCDFs differ only through the horizontal monetary scale.

## Distributional and inequality analysis

The empirical complementary cumulative distribution function is

$$
\widehat{\overline F}(x)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(x_i\ge x),
$$

and is evaluated on exact geometric thresholds with default ratio $b=1.05$. The CCDF is retained as a probability $S(x)\in[0,1]$ in both computation and plotting. Pareto-type behavior is examined in log-log coordinates. Gompertz-type behavior is examined with

$$
-\ln\{-\ln[S(x)]\},\qquad 0<S(x)<1,
$$

which preserves the decreasing visual orientation while retaining the full tail below one percent. The pipeline also computes Lorenz curves and inequality measures including Gini, Pietra, Kolkata, Zanardi, Theil, Atkinson, top-income shares, Shannon-based measures, and Herfindahl concentration. Current estimators are record-weighted; population inference requires the appropriate PNAD survey weights and design information.

## Outputs and auditability

Generated products use a shallow structure. [`outputs/figures/`](outputs/figures/) and [`outputs/tables/`](outputs/tables/) contain all artifacts, while the filename prefix carries the analytical role. Files beginning with `eda_` document refined-versus-trusted diagnostics, cleaning thresholds, histograms, boxplots, upper-tail behavior, frequencies, and audit counts. Files beginning with `paper_` are produced only from trusted data and contain the distributional and inequality results intended for scientific interpretation. Corresponding refined and trusted diagnostic figures use common bins and axis limits whenever possible so visual differences represent data treatment rather than automatic rescaling.

Examples include `eda_refined_histogram_income_page_01.png`, `eda_trusted_boxplot_income_page_01.png`, `eda_compare_outlier_income_upper_tail_refined_trusted.png`, `paper_ccdf_income_gompertz_page_01.png`, and `paper_annual_inequality_indices.csv`. [`outputs/manifest.csv`](outputs/manifest.csv) records the generated artifacts.

## Repository structure

| Path | Contents |
|---|---|
| [`data/raw/`](data/raw/) | Local landing area for original source files; large raw inputs are not committed. |
| [`data/refined/`](data/refined/) | Harmonized annual Parquet files before trusted-layer cleaning. |
| [`data/trusted/`](data/trusted/) | Deterministically cleaned annual Parquet files used by all scientific analyses. |
| [`data/metadata/`](data/metadata/) | Annual survey mapping, sentinel codes, currencies, monetary factors, and source URLs. |
| [`src/`](src/) | Data access, EDA and cleaning classes, scientific analysis, plotting, pipeline orchestration, and output persistence. See [`src/README.md`](src/README.md). |
| [`tests/`](tests/) | Automated validation of data treatment, flag exclusivity, trusted materialization, CCDF construction, inequality measures, plotting, and pipeline behavior. |
| [`outputs/`](outputs/) | Flat generated figure and table directories plus the manifest. |
| [`.github/workflows/`](.github/workflows/) | Automated tests, trusted-layer rebuilding, output audit, artifact upload, and persistence of generated data and results. |

## Reproducibility

Install the project in editable mode with `pip install -e '.[dev]'`. Running `pnad-income --refined data/refined --trusted data/trusted --metadata data/metadata/pnad_metadata.csv --output outputs` rebuilds the trusted layer, executes the 1976–2025 analysis, and writes all audit and scientific artifacts. Pull requests run tests and a complete temporary pipeline. After changes reach `main`, GitHub Actions rebuilds `data/trusted/` and `outputs/`, audits the expected files, uploads the outputs as an artifact, and commits regenerated trusted datasets and analysis products when they change.

## Author

**Osvaldo L. Santos-Pereira**  
[Academic webpage](https://ozsp12.github.io/) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [GitHub](https://github.com/ozsp12)
