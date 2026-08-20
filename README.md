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
| [`data/raw/`](data/raw/) | Local landing area for original survey files; raw inputs are not intended for repository storage. |
| [`data/refined/`](data/refined/) | Structurally harmonized annual Parquet files before deterministic data-quality treatment. |
| [`data/trusted/`](data/trusted/) | Deterministically rebuilt from `refined` after cleaning; this is the only data layer used by scientific analyses. |
| [`data/metadata/`](data/metadata/) | Annual extraction metadata, sentinel codes, currencies, monetary factors, and provenance required to reproduce preparation and cleaning. |
| [`src/`](src/) | Data access, cleaning, EDA, distributional analysis, plotting, orchestration, and persistence. The refined-to-trusted cleaning rule, sentinel precedence and flag exclusivity, outlier methods, monetary harmonization, CCDF construction, Lorenz analysis, and inequality measures are documented in [`src/README.md`](src/README.md). |
| [`tests/`](tests/) | Automated tests for preprocessing, trusted-layer construction, CCDFs, inequality measures, plotting, and pipeline behavior. |
| [`outputs/`](outputs/) | Generated figures, tables, and manifest using flat `eda_` and `paper_` prefixes. |
| [`.github/workflows/`](.github/workflows/) | Automated validation, trusted-data rebuilding, output audit, and persistence. |

# Data and sources

The longitudinal series combines IBGE household microdata with exchange-rate and inflation series used for monetary harmonization, together with year-specific project metadata that records extraction fields, missing-value codes, currencies, monetary factors, and provenance.

| Source | Coverage / use | Reference |
|---|---|---|
| IBGE — PNAD annual | Historical household microdata through 2015. | [PNAD annual microdata](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_anual/microdados/) |
| IBGE — PNAD Contínua | Household microdata used to extend the series from 2016 onward. | [PNAD Contínua quarterly microdata](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/) |
| Banco Central do Brasil | Historical foreign-exchange information used in cross-year monetary harmonization. | [Historical exchange rates](https://www.bcb.gov.br/estabilidadefinanceira/historicocotacoes) |
| Banco Central do Brasil | Open exchange-rate data used to reproduce recent conversion factors. | [Open exchange-rate dataset](https://dadosabertos.bcb.gov.br/dataset/taxas-de-cambio-todos-os-boletins-diarios) |
| BLS / FRED | CPIAUCSL used for the project-level adjustment to September 2025 dollars. | [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) |
| Project metadata | Annual extraction fields, fixed-width positions when applicable, missing-value codes, currencies, exchange factors, inflation factors, and source URLs. | [`pnad_metadata.csv`](data/metadata/pnad_metadata.csv) |

# Outputs and auditability

Generated products use a shallow structure under [`outputs/figures/`](outputs/figures/) and [`outputs/tables/`](outputs/tables/). The analytical role is encoded directly in each filename: artifacts beginning with `eda_` document diagnostics for the `refined → trusted` transformation, while artifacts beginning with `paper_` contain scientific outputs computed only from trusted data.

| Location | Purpose | Representative artifacts |
|---|---|---|
| [`outputs/figures/`](outputs/figures/) | EDA histograms, boxplots, upper-tail diagnostics, refined-versus-trusted comparisons, and scientific distributional or inequality figures. | `eda_refined_histogram_income_page_01.png`, `eda_trusted_boxplot_income_page_01.png`, `eda_compare_outlier_income_upper_tail_refined_trusted.png`, `paper_ccdf_income_gompertz_page_01.png`, `paper_inequality_gini_all_years.png` |
| [`outputs/tables/`](outputs/tables/) | Cleaning diagnostics, thresholds, frequencies, descriptive statistics, and scientific summary, CCDF, and inequality tables. | `eda_cleaning_audit.csv`, `eda_cleaning_thresholds.csv`, `eda_refined_descriptive_statistics.csv`, `paper_annual_summary.csv`, `paper_annual_inequality_indices.csv`, `paper_ccdf_income_nominal_adjusted.parquet` |
| [`outputs/manifest.csv`](outputs/manifest.csv) | Inventory of generated analysis artifacts. | Complete output manifest. |

# Reproducibility

Install the project in editable mode with `pip install -e '.[dev]'`. Running `pnad-income --refined data/refined --trusted data/trusted --metadata data/metadata/pnad_metadata.csv --output outputs` rebuilds the trusted layer, executes the 1976–2025 analysis, and writes all audit and scientific artifacts. Pull requests run tests and a complete temporary pipeline. After changes reach `main`, GitHub Actions rebuilds `data/trusted/` and `outputs/`, audits the expected files, uploads the outputs as an artifact, and commits regenerated trusted datasets and analysis products when they change.

# Author

**Osvaldo L. Santos-Pereira**  
[Academic webpage](https://ozsp12.github.io/) · [Lattes](http://lattes.cnpq.br/6730251976463283) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [Google Scholar](https://scholar.google.com/citations?user=HIZp0X8AAAAJ&hl=en) · [ResearchGate](https://www.researchgate.net/profile/Osvaldo-Santos-Pereira) · [GitHub](https://github.com/ozsp12)
