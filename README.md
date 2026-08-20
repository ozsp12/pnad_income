# PNAD Income Distribution in Brazil, 1976–2025

This repository contains a harmonized longitudinal dataset and a reproducible computational pipeline for the study of Brazilian income distributions from 1976 to 2025. The project combines annual PNAD and PNAD Contínua microdata with year-specific extraction metadata, deterministic data-quality treatment, monetary harmonization, empirical distribution analysis, and inequality measures. The analytical series contains 45 survey years; 1980, 1991, 1994, 2000, and 2010 are absent because no compatible PNAD observation is available for those years.

# Repository structure

```text
pnad_income/
├── data/
│   ├── raw/          # Local landing area for original survey files
│   ├── refined/      # Harmonized annual Parquet files
│   ├── trusted/      # Cleaned annual files used by scientific analyses
│   └── metadata/     # Survey mapping, sentinel, currency, and provenance metadata
│
├── src/
│   ├── analysis.py
│   ├── cli.py
│   ├── data.py
│   ├── descriptive.py
│   ├── outputs.py
│   ├── pipeline.py
│   └── plotting.py
│
├── tests/            # Automated validation of data treatment and analyses
│
├── outputs/
│   ├── figures/      # Flat eda_* and paper_* figures
│   ├── tables/       # Flat eda_* and paper_* tables
│   └── manifest.csv
│
├── .github/
│   └── workflows/    # CI, trusted-layer rebuilding, and output persistence
│
├── pyproject.toml
└── README.md
```

| Path | Purpose |
|---|---|
| [`data/`](data/) | Raw landing area, refined annual datasets, trusted analytical datasets, and metadata. |
| [`src/`](src/) | Data access, cleaning, EDA, scientific analysis, plotting, orchestration, and persistence. Technical methodology is documented in [`src/README.md`](src/README.md). |
| [`tests/`](tests/) | Automated tests for preprocessing, trusted-layer construction, CCDFs, inequality measures, plotting, and pipeline behavior. |
| [`outputs/`](outputs/) | Generated figures, tables, and manifest using flat `eda_` and `paper_` prefixes. |
| [`.github/workflows/`](.github/workflows/) | Automated validation, trusted-data rebuilding, output audit, and persistence. |

# Data and sources

| Source | Coverage / series | Role in the project | Reference |
|---|---|---|---|
| Instituto Brasileiro de Geografia e Estatística (IBGE) | Annual PNAD through 2015 | Historical household microdata used to construct annual income files. | [PNAD annual microdata archive](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_anual/microdados/) |
| Instituto Brasileiro de Geografia e Estatística (IBGE) | PNAD Contínua from 2016 onward | Recent household microdata used to extend the longitudinal series. | [PNAD Contínua quarterly microdata](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/) |
| Banco Central do Brasil (BCB) | Historical and open exchange-rate data | Conversion factors used in cross-year monetary harmonization. | [Historical exchange rates](https://www.bcb.gov.br/estabilidadefinanceira/historicocotacoes) · [Open exchange-rate dataset](https://dadosabertos.bcb.gov.br/dataset/taxas-de-cambio-todos-os-boletins-diarios) |
| BLS / FRED | CPIAUCSL | U.S. CPI series used for the project-level adjustment to September 2025 dollars. | [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) |
| Project metadata | 1976–2025 | Survey fields, fixed-width positions when applicable, missing-value codes, currencies, exchange factors, inflation factors, and source URLs. | [`data/metadata/pnad_metadata.csv`](data/metadata/pnad_metadata.csv) |

The four data layers have distinct roles. [`data/raw/`](data/raw/) is reserved for original source files and is not intended for repository storage. [`data/refined/`](data/refined/) contains structurally harmonized annual Parquet files. [`data/trusted/`](data/trusted/) is rebuilt deterministically from the refined layer after data-quality treatment and is the only layer used by scientific analyses. [`data/metadata/`](data/metadata/) contains the annual extraction and provenance information required to reproduce preparation and cleaning.

# Methodology

The root README intentionally keeps the methodological description concise. The deterministic refined-to-trusted cleaning rule, sentinel precedence and flag exclusivity, statistical outlier methods, monetary harmonization, CCDF construction, Lorenz analysis, and inequality measures are documented in [`src/README.md`](src/README.md), alongside the responsibilities of each source module.

# Outputs and auditability

Generated products use a shallow structure. [`outputs/figures/`](outputs/figures/) and [`outputs/tables/`](outputs/tables/) contain all artifacts, while filename prefixes carry the analytical role. Files beginning with `eda_` document refined-versus-trusted diagnostics, cleaning thresholds, histograms, boxplots, upper-tail behavior, frequencies, and audit counts. Files beginning with `paper_` are produced only from trusted data and contain the distributional and inequality results intended for scientific interpretation.

Examples include `eda_refined_histogram_income_page_01.png`, `eda_trusted_boxplot_income_page_01.png`, `eda_compare_outlier_income_upper_tail_refined_trusted.png`, `paper_ccdf_income_gompertz_page_01.png`, and `paper_annual_inequality_indices.csv`. [`outputs/manifest.csv`](outputs/manifest.csv) records the generated artifacts.

# Reproducibility

Install the project in editable mode with `pip install -e '.[dev]'`. Running `pnad-income --refined data/refined --trusted data/trusted --metadata data/metadata/pnad_metadata.csv --output outputs` rebuilds the trusted layer, executes the 1976–2025 analysis, and writes all audit and scientific artifacts. Pull requests run tests and a complete temporary pipeline. After changes reach `main`, GitHub Actions rebuilds `data/trusted/` and `outputs/`, audits the expected files, uploads the outputs as an artifact, and commits regenerated trusted datasets and analysis products when they change.

# Author

**Osvaldo L. Santos-Pereira**  
[Academic webpage](https://ozsp12.github.io/) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [GitHub](https://github.com/ozsp12)
