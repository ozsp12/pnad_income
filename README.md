# PNAD Income Distribution Project

A reproducible research project for the longitudinal analysis of Brazilian income distributions using PNAD and PNAD Continua microdata from 1976 to 2025. The repository consolidates an undergraduate research codebase originally developed by Beatriz and its subsequent refactoring into a metadata-driven analytical pipeline.

The primary objective is to study the evolution of income distributions and inequality through annual income statistics, complementary cumulative distribution functions (CCDFs), geometric income bins, Gini coefficients, Lorenz curves, and monetary adjustment to a common 2025 reference. For 2016–2025 the pipeline preserves both `VD4019` and `VD4020`, allowing the habitual/effective income comparison that existed in the original work.

## Repository structure

```text
pnad_income/
├── config/                     # Survey-variable and monetary metadata
├── data/                       # Local raw/processed data (not versioned)
├── docs/                       # Methodology and migration notes
├── notebooks/                  # Thin, reproducible analytical notebooks
├── outputs/                    # Reproducible figures/tables/reports
├── src/pnad_income/            # Reusable Python package
├── tests/                      # Regression/unit tests
└── archive/                    # Historical working material / provenance
```

## Analytical workflow

1. `notebooks/00_metadata.ipynb` inspects the single project metadata table.
2. `notebooks/01_build_refined_data.ipynb` reads fixed-width microdata and produces standardized annual Parquet files.
3. `notebooks/02_descriptive_inequality.ipynb` computes descriptive statistics, Gini coefficients, histograms, and Lorenz curves.
4. `notebooks/03_income_distributions.ipynb` computes ordinary, log-log, and double-log CCDF representations and compares nominal with adjusted distributions.
5. `notebooks/04_habitual_vs_effective.ipynb` compares `VD4019` and `VD4020` for 2016–2025.

Core calculations live under `src/pnad_income/`; notebooks are intentionally thin and should not contain independent copies of constants or analytical functions.

## CCDF definition

For income `X`, the project evaluates the complementary cumulative distribution

```text
CCDF(x) = P(X >= x).
```

Geometric thresholds start at the smallest strictly positive income. Zero-income observations nevertheless remain in the normalization population, avoiding the unintended conditional distribution `P(X >= x | X > 0)` that appeared in an intermediate refactor.

## Data and provenance

Raw PNAD microdata are not committed to this repository. `config/pnad_metadata.csv` records the expected fixed-width positions, historical missing-value codes, local file patterns, currency/adjustment metadata, and the corresponding IBGE source location for each year. Years without PNAD observations in the inherited time series remain explicitly represented in the metadata.

## Installation

```bash
git clone https://github.com/ozsp12/pnad_income.git
cd pnad_income
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,notebooks]"
pytest
```

Place the original fixed-width microdata under `data/raw/` with the file structure described in `config/pnad_metadata.csv`, then execute the notebooks in numerical order.

## Main changes relative to the inherited notebooks

- one metadata source instead of duplicated year-by-year constants;
- reusable functions instead of `globals()` and manually repeated blocks;
- restored `VD4020` and habitual/effective income analysis;
- restored linear and log-log CCDF analyses;
- restored nominal-versus-adjusted CCDF comparisons;
- corrected CCDF normalization with zero incomes retained in the denominator;
- reproducible tests for distribution and inequality functions.

See [`docs/migration_notes.md`](docs/migration_notes.md) for the reconciliation of the two codebases and [`docs/methodology.md`](docs/methodology.md) for mathematical and processing definitions.

## Research provenance

**Supervisor / principal researcher:** Dr. Osvaldo L. Santos-Pereira  
**Undergraduate research student:** Beatriz (Bia)

Dr. Osvaldo L. Santos-Pereira — [Academic webpage](https://ozsp12.github.io/) · [Lattes](http://lattes.cnpq.br/6730251976463283) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [Google Scholar](https://scholar.google.com/citations?user=HIZp0X8AAAAJ&hl=en) · [ResearchGate](https://www.researchgate.net/profile/Osvaldo-Santos-Pereira) · [GitHub](https://github.com/ozsp12) · [LinkedIn](https://www.linkedin.com/in/ozsp12) · [Substack](https://substack.com/@olsp1982)

## License

No software license has been assigned yet. Add an explicit license before treating the repository as reusable third-party software.
