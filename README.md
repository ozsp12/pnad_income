# PNAD Income Distribution in Brazil, 1976–2025

This repository contains a harmonized longitudinal dataset and a reproducible computational pipeline for the study of Brazilian income distributions from 1976 to 2025. The project combines annual PNAD and PNAD Contínua microdata with year-specific extraction metadata, monetary harmonization, exploratory data-quality diagnostics, empirical distribution analysis, and inequality measures. The analytical series contains 45 survey years; 1980, 1991, 1994, 2000, and 2010 are absent because no compatible PNAD observation is available for those years.

## Data and sources

The microdata are obtained from the Instituto Brasileiro de Geografia e Estatística (IBGE). Annual PNAD files through 2015 are drawn from the [PNAD annual microdata archive](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_anual/microdados/), while observations from 2016 onward use [PNAD Contínua quarterly microdata](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/). The exact survey field, fixed-width position when applicable, missing-value code, original currency, exchange factor, price index, inflation factor, and source URL for each year are recorded in [`metadata/pnad_metadata.csv`](metadata/pnad_metadata.csv). The harmonized annual files used by the pipeline are stored in [`dados_refined/`](dados_refined/).

Historical exchange-rate information comes from the Banco Central do Brasil through its [historical exchange-rate service](https://www.bcb.gov.br/estabilidadefinanceira/historicocotacoes) and [open exchange-rate dataset](https://dadosabertos.bcb.gov.br/dataset/taxas-de-cambio-todos-os-boletins-diarios). U.S. inflation adjustment uses the [CPIAUCSL series](https://fred.stlouisfed.org/series/CPIAUCSL) from BLS/FRED.

## Data treatment and monetary harmonization

The income concept is selected separately for each survey year and mapped to a common analytical field. Survey-specific missing-value sentinels are defined in metadata and removed when refined datasets are built from raw files. Because the analytical pipeline also accepts already refined Parquet files, the exploratory layer independently checks whether declared sentinels or suspicious repeated upper-tail values remain in the data. Finite zero-income observations remain in the empirical population and in the denominator of the complementary cumulative distribution function, while logarithmic analyses use strictly positive thresholds. When the metadata specify a household quantity, income is divided by a positive household-size field before analysis.

For cross-year monetary comparison, nominal local-currency income is converted using the stored year-level exchange factor and then adjusted to the September 2025 U.S. CPI level,

$$
y^{(2025)}_{it}=\frac{y_{it}}{E_t}\frac{P_{2025}}{P_t}.
$$

This produces a project-level series in 2025 U.S. dollars and is not the official IBGE real-income deflation procedure. Within a given year the adjustment is a positive multiplicative rescaling, so rank order, Lorenz geometry, and scale-invariant inequality measures are unchanged; nominal and adjusted CCDFs differ only through the horizontal monetary scale.

## Exploratory analysis and data quality

Exploratory diagnostics are generated before interpreting distributional results. The [`DescriptiveStatistics`](src/descriptive.py) class computes annual sample size, location and dispersion statistics, high quantiles, Tukey upper limits, maxima, exact-value frequencies, known metadata sentinels, repeated values in the upper tail, and candidate cutoffs. Histograms, logarithmic-scale boxplots, and an annual upper-tail diagnostic are produced alongside tables that record the evidence supporting each candidate. Extreme observations are not removed automatically: a large value may represent a valid high income, whereas a survey sentinel is usually repeated and often appears as a conspicuous code such as a sequence of identical digits. The diagnostics therefore separate detection from the preprocessing decision.

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

## Repository structure

The repository separates source data, metadata, implementation, validation, and generated results. Exploratory outputs are kept apart from figures and tables intended for scientific interpretation.

| Path | Contents |
|---|---|
| [`dados_refined/`](dados_refined/) | Harmonized annual Parquet files used by the analytical pipeline. |
| [`metadata/`](metadata/) | Year-level survey mapping, missing-value codes, currencies, exchange factors, inflation factors, and source URLs. |
| [`src/`](src/) | Source code for data preparation, EDA, distributional analysis, plotting, pipeline orchestration, and output persistence. See [`src/README.md`](src/README.md). |
| [`tests/`](tests/) | Automated tests for preprocessing, CCDF construction, inequality measures, EDA, plotting, outputs, and pipeline behavior. |
| [`outputs/`](outputs/) | Generated manifest, tables, and figures. EDA and paper-oriented products are stored separately. |
| [`.github/workflows/`](.github/workflows/) | Automated validation, pipeline execution, output audit, and persistence of generated results. |

## Reproducibility

The project can be installed in editable mode with `pip install -e '.[dev]'`. Running `pnad-income --database dados_refined --output outputs` executes the harmonized pipeline with the default 1976–2025 coverage and writes a manifest of generated artifacts. The GitHub Actions workflow runs the test suite on pull requests and, after changes reach `main`, rebuilds the complete output tree, audits required products, uploads the results as an artifact, and commits regenerated outputs when they change.

## Author

**Osvaldo L. Santos-Pereira**  
[Academic webpage](https://ozsp12.github.io/) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [GitHub](https://github.com/ozsp12)
