# PNAD Income Distribution in Brazil, 1976–2025

This repository provides a harmonized longitudinal dataset and reproducible analysis pipeline for the study of Brazilian household per-capita income distributions from 1976 to 2025. The project is intended for academic analysis of income distributions, long-run inequality, and distributional regimes.

## Data sources

The microdata are obtained from the **Instituto Brasileiro de Geografia e Estatística (IBGE)**:

- **PNAD — Pesquisa Nacional por Amostra de Domicílios**, annual microdata used through 2015: [IBGE PNAD microdata](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_anual/microdados/).
- **PNAD Contínua — Pesquisa Nacional por Amostra de Domicílios Contínua**, used from 2016 onward: [IBGE PNAD Contínua microdata](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/).

The exact survey, income variable, fixed-width position where applicable, missing-value code, original currency, monetary factors, and year-specific source URL are recorded in [`metadata/pnad_metadata.csv`](metadata/pnad_metadata.csv).

The analytical series contains **45 survey years**. The years **1980, 1991, 1994, 2000, and 2010** are absent from the harmonized database.

## Analytical criteria

The primary quantity is harmonized **household per-capita income**. Year-specific PNAD variables are mapped to a common analytical field while preserving the annual source files in `dados_refined/`.

The current analysis follows these criteria:

- invalid and survey-specific missing-income codes are excluded according to the annual metadata;
- finite zero-income observations are retained when they are part of the empirical distribution, but logarithmic analyses use strictly positive support;
- monetary comparison across years uses the year-level exchange and inflation factors stored in the metadata, with

  $$
  y_{\mathrm{USD},2025}=\frac{y_t}{E_t}\,I_{t\rightarrow2025};
  $$

- empirical CCDFs are evaluated on a geometric income grid and used to study the distributional tail;
- Pareto-type behavior is examined through the CCDF in log-log coordinates, while Gompertz-type behavior is examined through the corresponding double-log survival transformation;
- inequality measures are computed from the harmonized annual distributions, including Gini, Pietra, Kolkata, Zanardi, Theil, Atkinson, top-income shares, and related concentration measures;
- the present estimators are **record-weighted**. Population inference requires the appropriate PNAD sampling weights and survey-design information.

The external provenance and construction of the monetary series used for exchange-rate and inflation harmonization will be specified explicitly in the technical documentation.

## Repository structure

```text
pnad_income/
├── dados_refined/       # annual harmonized Parquet files
├── metadata/            # survey definitions and year-level monetary metadata
├── src/pnad_income/     # analytical implementation
├── tests/               # automated validation
├── outputs/             # generated figures, tables, and manifest
└── README.md
```

## Technical documentation

A separate technical document will provide the complete methodological specification, including data provenance, annual variable mapping, monetary harmonization, CCDF construction, distributional hypotheses, inequality estimators, validation procedures, and reproducibility details. This README is intentionally limited to the scientific scope, source selection, and principal analytical criteria.

## Author

**Osvaldo L. Santos-Pereira**  
[Academic webpage](https://ozsp12.github.io/) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [GitHub](https://github.com/ozsp12)
