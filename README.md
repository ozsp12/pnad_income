# PNAD Income Analysis

Reproducible research software for longitudinal analysis of Brazilian income distributions using PNAD and PNAD Contínua data from 1976 to 2025.

The repository is organized as a Python project rather than as a collection of exploratory notebooks. All analytical logic is implemented in documented modules under `src/pnad_income/`. A single notebook, `notebooks/pnad_income_pipeline.ipynb`, executes the complete pipeline and explains each analytical step and its outputs.

## Scientific scope

The project provides:

- annual descriptive statistics for income distributions;
- monetary standardization to a common 2025 reference;
- Gini coefficients and Lorenz curves;
- complementary cumulative distribution functions (CCDFs);
- linear, log-log, and `ln[ln(CCDF)]` representations;
- nominal versus adjusted income comparisons;
- habitual versus effective income comparisons when both measures are available.

## Repository structure

```text
pnad_income/
├── config/
│   └── pnad_metadata.csv
├── dados_refined/
│   └── pnad_refined_YYYY.parquet
├── docs/
│   ├── database_schema.md
│   └── methodology.md
├── notebooks/
│   └── pnad_income_pipeline.ipynb
├── outputs/
│   └── README.md
├── src/
│   └── pnad_income/
│       ├── __init__.py
│       ├── config.py
│       ├── distributions.py
│       ├── inequality.py
│       ├── io.py
│       ├── pipeline.py
│       ├── plotting.py
│       └── preprocessing.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

`dados_refined/` is the analytical database. It contains one Parquet file per available survey year. The pipeline reads the directory directly and infers the year from `pnad_refined_YYYY.parquet` if a file does not already contain a `year` column.

## Installation

```bash
git clone https://github.com/ozsp12/pnad_income.git
cd pnad_income
python -m venv .venv
```

Activate the environment and install the project:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -e ".[dev,notebooks]"
pytest
```

## Execute the analysis

Open and execute:

```text
notebooks/pnad_income_pipeline.ipynb
```

By default the notebook reads `../dados_refined`. A different database path can be supplied with the environment variable `PNAD_DATABASE_PATH`.

The notebook performs the following sequence through the encapsulated package:

1. locate and load the annual Parquet database;
2. validate the analytical schema and temporal coverage;
3. attach year-level monetary metadata;
4. construct adjusted income measures;
5. compute annual descriptive and inequality statistics;
6. compute nominal and adjusted CCDFs;
7. compare habitual and effective income when `income_effective` is present;
8. generate figures and final reproducibility diagnostics.

The notebook contains no independent implementation of these calculations. A clean kernel restart therefore executes the same code paths that are covered by the test suite.

## CCDF convention

For a nonnegative income variable \(X\), the empirical complementary cumulative distribution is

\[
\widehat{\overline F}(x)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(X_i\geq x).
\]

Geometric thresholds begin at the smallest strictly positive observation, while finite zero-income observations remain in the denominator \(N\). The estimator is therefore the unconditional CCDF rather than \(P(X\geq x\mid X>0)\).

## Monetary adjustment

The project uses

\[
Y_{2025}=\frac{Y}{E}I,
\]

where \(Y\) is observed income, \(E\) is the year-specific exchange factor, and \(I\) is the factor that brings the value to the 2025 reference. Year-level constants are centralized in `config/pnad_metadata.csv`.

See [`docs/methodology.md`](docs/methodology.md) for the mathematical definitions and [`docs/database_schema.md`](docs/database_schema.md) for the database interface.

## Tests

```bash
pytest
```

The tests evaluate preprocessing, CCDF construction, inequality measures, database loading, schema validation, and high-level pipeline orchestration independently of Jupyter.

## Author

**Dr. Osvaldo L. Santos-Pereira** — [Academic webpage](https://ozsp12.github.io/) · [Lattes](http://lattes.cnpq.br/6730251976463283) · [ORCID](https://orcid.org/0000-0003-2231-517X) · [Google Scholar](https://scholar.google.com/citations?user=HIZp0X8AAAAJ&hl=en) · [ResearchGate](https://www.researchgate.net/profile/Osvaldo-Santos-Pereira) · [GitHub](https://github.com/ozsp12) · [LinkedIn](https://www.linkedin.com/in/ozsp12)

## License

No software license has been assigned yet.
