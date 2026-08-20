# Source code

The `src` directory contains the computational implementation for data access, trusted-layer construction, exploratory diagnostics, distributional analysis, plotting, pipeline orchestration, and artifact persistence. The modules remain at one level so the analytical flow can be inspected without an additional package hierarchy.

| File | Responsibility |
|---|---|
| [`data.py`](data.py) | Data-layer paths, metadata loading, annual refined/trusted file access, schema validation, raw-to-refined preparation, and monetary harmonization. |
| [`descriptive.py`](descriptive.py) | `DescriptiveStatistics` provides annual EDA, exact-value frequencies, metadata-sentinel counts, histograms, boxplots, and upper-tail diagnostics. `IncomeDataCleaner` applies the deterministic refined-to-trusted cleaning rule, creates mutually exclusive quality flags, records annual thresholds, and materializes trusted Parquet files. |
| [`analysis.py`](analysis.py) | Empirical CCDF construction, Lorenz curves, inequality indices, and external Gini validation. |
| [`plotting.py`](plotting.py) | Scientific plots for CCDF, Lorenz, concentration, and inequality analyses. |
| [`pipeline.py`](pipeline.py) | `PipelineConfig` and `PipelineResults`, plus orchestration of scientific analyses using the trusted data layer. |
| [`outputs.py`](outputs.py) | Flat `figures/` and `tables/` persistence using `eda_` and `paper_` filename prefixes, including refined-versus-trusted diagnostics and the manifest. |
| [`cli.py`](cli.py) | Command-line entry point exposed as `pnad-income`; it rebuilds trusted data before invoking the scientific pipeline. |

## Trusted-layer construction

The refined-to-trusted transition is implemented by `IncomeDataCleaner` in [`descriptive.py`](descriptive.py). Metadata sentinels are identified first and excluded from the sample used to estimate statistical thresholds. The two record-level quality flags are mutually exclusive by construction: whenever `flag_metadata_sentinel = 1`, the corresponding `flag_statistical_outlier` value is forced to `0`. Statistical upper-tail flags are evaluated only for the remaining observations. Repeated exact values remain part of the exploratory frequency analysis and are not promoted to a third record-level flag.

The default rule is estimated independently for each survey year on positive, non-sentinel income. First,

$$
z_i = \log(1+y_i).
$$

Let $c$ be the median of the transformed values and let $s$ be 1.4826 times their median absolute deviation. The upper cutoff is then

$$
y_{\max}=\exp(c+ks)-1,
$$

with default $k=6$. The method and multiplier are explicit CLI parameters. `mad` and `mean_std` are available as deterministic alternatives for sensitivity checks. Annual thresholds, sentinel counts, statistical-outlier counts, removal rates, and trusted sample sizes are written to EDA audit tables.

The trusted annual files are materialized as `pnad_trusted_<year>.parquet`. The scientific pipeline reads this layer exclusively, so CCDF, Lorenz, inequality, and paper-oriented outputs cannot bypass the cleaning stage. Refined-versus-trusted EDA products are retained to document the effect of the transformation before scientific interpretation.

## Monetary harmonization

For cross-year monetary comparison, nominal local-currency income is converted using the stored year-level exchange factor and adjusted to the September 2025 U.S. CPI level:

$$
y^{(2025)}_{it}=\frac{y_{it}}{E_t}\frac{P_{2025}}{P_t}.
$$

Here, $E_t$ is the local-currency-units-per-U.S.-dollar exchange factor for year $t$, and $P_t$ is the September CPI level used by the project. This produces a project-level series in 2025 U.S. dollars and is not the official IBGE real-income deflation procedure. Within a given year the adjustment is a positive multiplicative rescaling, so rank order, Lorenz geometry, and scale-invariant inequality measures are unchanged. Nominal and adjusted CCDFs therefore differ through the horizontal monetary scale rather than through within-year ordering.

## Distributional and inequality analysis

For a sample of size $N$, the empirical complementary cumulative distribution function is evaluated as

$$
S(x)=\frac{1}{N}\sum_{i=1}^{N} I(y_i\ge x),
$$

where $I(\cdot)$ is the indicator function. The implementation uses exact geometric thresholds with default ratio $b=1.05$. The CCDF is retained as a probability $S(x)$ between 0 and 1 in computation and plotting. Pareto-type behavior is examined in log-log coordinates.

The Gompertz diagnostic uses

$$
-\ln[-\ln S(x)], \qquad 0<S(x)<1,
$$

which preserves the decreasing visual orientation while retaining the full tail below one percent. The pipeline also computes Lorenz curves and inequality measures including Gini, Pietra, Kolkata, Zanardi, Theil, Atkinson, top-income shares, Shannon-based measures, and Herfindahl concentration.

Current estimators are record-weighted. Population inference requires the appropriate PNAD survey weights and design information; the present implementation should therefore be interpreted as analysis of the harmonized records rather than a survey-design-corrected population estimator.

## Analytical flow

```text
data/refined/
      │
      ▼
DescriptiveStatistics
      │
      ├── refined EDA and quality diagnostics
      │
      ▼
IncomeDataCleaner
      │
      ├── sentinel precedence
      ├── annual dynamic thresholds
      ├── cleaning audit
      │
      ▼
data/trusted/
      │
      ▼
scientific pipeline
      │
      ├── CCDF
      ├── Lorenz curves
      ├── inequality indices
      └── paper_* outputs
```
