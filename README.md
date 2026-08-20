# PNAD Income Distribution in Brazil, 1976–2025

This repository contains a harmonized longitudinal dataset and reproducible computational pipeline for the study of Brazilian income distributions, long-run inequality, and distributional regimes from 1976 to 2025.

## Sources

The microdata are obtained from the **Instituto Brasileiro de Geografia e Estatística (IBGE)**:

- **PNAD — Pesquisa Nacional por Amostra de Domicílios**, annual microdata through 2015: https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_anual/microdados/
- **PNAD Contínua**, used from 2016 onward: https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/

The year-specific survey field, fixed-width position when applicable, missing-value code, original currency, exchange factor, price index, inflation factor, and exact IBGE source URL are stored in [`metadata/pnad_metadata.csv`](metadata/pnad_metadata.csv).

Monetary harmonization uses historical exchange-rate information from the **Banco Central do Brasil** and the U.S. Consumer Price Index series **CPIAUCSL** from **BLS/FRED**:

- Banco Central do Brasil, historical exchange rates: https://www.bcb.gov.br/estabilidadefinanceira/historicocotacoes
- Banco Central do Brasil, open exchange-rate data: https://dadosabertos.bcb.gov.br/dataset/taxas-de-cambio-todos-os-boletins-diarios
- FRED/BLS, CPIAUCSL: https://fred.stlouisfed.org/series/CPIAUCSL

The analytical series contains **45 survey years**. The years **1980, 1991, 1994, 2000, and 2010** are absent.

## Analytical criteria

The project uses the income concept specified for each survey year and maps it to a common analytical field. Survey-specific missing-value sentinels are removed before analysis. When the metadata indicate that a household quantity must be converted to a per-capita quantity, the recorded income is divided by the corresponding positive household-size field.

Finite zero-income observations remain part of the empirical distribution and of the CCDF denominator. Logarithmic distributional analyses use strictly positive income thresholds.

For cross-year monetary comparison, nominal local-currency income is first converted to U.S. dollars using the stored year-level exchange factor and then adjusted to the September 2025 U.S. CPI level:

$$
y^{(2025)}_{it}=\frac{y_{it}}{E_t}\frac{P_{2025}}{P_t}.
$$

This is a project-level harmonization in **2025 U.S. dollars**, not the official IBGE real-income deflation procedure.

The empirical complementary cumulative distribution function is

$$
\widehat{\overline F}(x)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(x_i\ge x),
$$

and is evaluated on geometric thresholds with default ratio $b=1.05$. Pareto-type behavior is examined in log-log coordinates. Gompertz-type behavior is examined through the survival transformation

$$
\ln\{-\ln[\widehat{\overline F}(x)]\},
$$

with the CCDF expressed as a probability on $(0,1)$.

The pipeline also computes Lorenz curves and inequality measures including Gini, Pietra, Kolkata, Zanardi, Theil, Atkinson, top-income shares, Shannon-based measures, and Herfindahl concentration. The present estimators are **record-weighted**; population inference requires the appropriate PNAD sampling weights and survey-design information.

## Repository structure

```text
pnad_income/
├── dados_refined/       # annual harmonized Parquet files
├── metadata/            # annual extraction and monetary metadata
├── src/pnad_income/     # computational implementation
├── tests/               # automated validation
├── outputs/             # generated figures, tables, and manifest
└── README.md
```

## Technical documentation

A separate technical document will contain the complete data provenance, annual variable mapping, monetary harmonization, CCDF construction, Gompertz–Pareto methodology, inequality estimators, validation procedures, and reproducibility specification. It will be linked here when released.

## Author

**Osvaldo L. Santos-Pereira**  
[Academic webpage](https://ozsp12.github.io/) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [GitHub](https://github.com/ozsp12)
