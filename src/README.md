# Source code

The `src` directory contains the computational implementation for data access, trusted-layer construction, exploratory diagnostics, distributional analysis, plotting, pipeline orchestration, and artifact persistence. The modules remain at one level so the analytical flow can be inspected without an additional package hierarchy.

| File | Responsibility |
|---|---|
| [`data.py`](data.py) | Data-layer paths, metadata loading, annual refined/trusted file access, schema validation, raw-to-refined preparation, and monetary harmonization. |
| [`descriptive.py`](descriptive.py) | `DescriptiveStatistics` provides annual EDA, exact-value frequencies, metadata-sentinel counts, histograms, boxplots, and upper-tail diagnostics. `IncomeDataCleaner` applies the deterministic refined-to-trusted cleaning rule, creates mutually exclusive quality flags, records annual thresholds, and materializes trusted Parquet files. |
| [`analysis.py`](analysis.py) | Empirical CCDF construction, Lorenz curves, inequality indices, absolute p80/p99/p100 income totals, and external Gini validation. |
| [`regime_analysis.py`](regime_analysis.py) | Annual profile search for the Gompertz-body/Pareto-tail transition, parameter estimation, fit diagnostics, and construction of the two derived datasets used by the regime figures. |
| [`plotting.py`](plotting.py) | Scientific plots for CCDF, annotated Lorenz grids, absolute income-group totals, concentration, and inequality analyses. |
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

Annual aggregate income is also decomposed without percentage normalization. `p80` is the income received by the bottom 80% of records, `p99` is the income received by the next 19% (from the 80th through the 99th percentile), and `p100` is the income received by the top 1%. The boundaries are evaluated from the empirical Lorenz curve, and the three absolute components are constrained to sum exactly to the observed annual income total. The table and stacked bars use the harmonized `income_adj` measure in 2025 USD, preserving changes in both scale and composition while avoiding nominal-currency discontinuities across years.

The annotated Lorenz grid reports Gini, Pietra, Kolkata, and Zanardi values in each annual panel. Complete grids use three columns by default; the CLI option `--plot-columns` and the plotting function's `ncols` argument can change that layout. The output pipeline generates only the annotated Lorenz version.

The calculated PNAD Gini series is plotted together with the IPEA and World Bank reference series stored in [`../data/metadata/series_ipea_banco_mundial.csv`](../data/metadata/series_ipea_banco_mundial.csv). Every series is reindexed to the complete annual calendar before plotting, so absent observations remain `NaN` and create true gaps rather than interpolated connecting segments.

## Gompertz-Pareto regime analysis

The annual fit uses positive, finite trusted `income_adj`, normalized as $x_i=\texttt{income\_adj}_i/\overline{\texttt{income\_adj}}$, and the percent CCDF $F(x)=100P(X\ge x)$. The body model is

$$
G(x)=\exp[\exp(A-Bx)],\qquad A=\ln[\ln(100)].
$$

For each candidate cutoff, $B>0$ is estimated by least squares in $\ln[\ln F(x)]=A-Bx$ with fixed $A$. The tail model is $P(x)=\beta x^{-\alpha}$; $\alpha>0$ is estimated by least squares in $\ln F$ with continuity imposed through $\beta=G(x_t)x_t^\alpha$.

The selected cutoff minimizes `joint_sse` on one response scale: squared residuals in $\ln F$ from $\ln G(x)=\exp(A-Bx)$ in the body plus squared residuals from $\ln P(x)=\ln\beta-\alpha\ln x$ in the tail. The search retains the existing observation, curve-point, quantile, boundary, flat-profile, weak-identification, and p20/p40/p60 sensitivity checks. Gompertz and Pareto R², RMSE, and SSE are diagnostics in their respective fitting spaces.

`paper_distribution_regime_fits.csv` stores one compact LS result per year. `paper_distribution_regime_curves.parquet` stores the empirical and fitted branches needed by every regime figure. `outputs.py` reloads these two assets before plotting; regime figures do not access or re-estimate microdata. Fits are record-weighted rather than survey-design-corrected.

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
      ├── Gompertz-Pareto regime fits
      └── paper_* outputs
```
