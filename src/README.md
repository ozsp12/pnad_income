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

The annual regime analysis uses only positive, finite `income_adj` observations prepared from the trusted layer. The adjusted 2025-USD scale makes the cutoff and Gompertz slope comparable across years; the Pareto exponent and empirical cutoff quantile are invariant to a positive within-year rescaling. No refined records enter this analysis.

For each year, the empirical CCDF uses the same project definition and geometric threshold grid documented above. Candidate cutoffs are thresholds on that empirical grid with default empirical quantiles from 0.80 through 0.99. A candidate is admissible only when the body and tail each contain at least 100 observations, the tail contains at least 1% of the positive sample, both transformed fits contain enough curve points, and all fitted parameters are finite. These defaults are shared across years and are configurable through `RegimeFitConfig` and the corresponding CLI options.

For body points below a candidate cutoff, the estimator fits

$$
-\ln[-\ln S(x)] = A + Bx.
$$

For observations $x_i\ge x_c$, the continuous Pareto density exponent is estimated by

$$
\widehat{\alpha}=1+\frac{n_{tail}}{\sum_i \ln(x_i/x_c)}.
$$

Under this density convention, the conditional CCDF is proportional to $x^{1-\widehat{\alpha}}$, so `pareto_ccdf_slope = 1 - pareto_alpha_mle`. Its unconditional intercept is anchored by the empirical tail fraction at the cutoff.

Cutoff selection does not combine unrelated statistics. Both fitted regimes are mapped back to `log(S(x))`, giving a common response scale. Gaussian-residual AIC and BIC are calculated for the body, tail, and joint segmented fit; the default profile criterion is the minimum joint BIC with three fitted parameters (`A`, `B`, and `alpha`). The Pareto Kolmogorov-Smirnov distance is computed afterward for the selected tail and is reported only as an independent goodness-of-fit diagnostic. `fit_status` records controlled failures rather than substituting invented estimates. A valid minimum on an admissible-search boundary is retained but labeled `ok_boundary_lower` or `ok_boundary_upper`; researchers should treat that status as a sensitivity warning rather than evidence of a precisely located interior transition.

### Derived-data interface for regime figures

The estimator writes two public analytical assets:

- `paper_distribution_regime_fits.csv` contains one row per survey year, the selected cutoff, sample allocation, Gompertz and Pareto parameters, transformed-coordinate diagnostics, common-scale AIC/BIC values, Pareto KS distance, candidate count, criterion, and status.
- `paper_distribution_regime_curves.parquet` contains the empirical CCDF grid, plotting transforms, regime label, cutoff, and both selected fitted curves.

`outputs.py` persists these datasets and reloads them before creating any regime figure. The plotting functions accept only the reloaded frames: they neither read `data/trusted/` nor `data/refined/`, access `PipelineResults.panel`, reconstruct a CCDF, rerun MLE, or search for a cutoff. The asset-to-figure column contract is:

| Figure family | Fit columns | Curve columns |
|---|---|---|
| Annual dual-panel regime fits | `year`, `cutoff`, `pareto_alpha_mle`, `fit_status` | `year`, `income`, `empirical_ccdf`, `gompertz_transform`, `regime`, `gompertz_fitted_transform`, `pareto_fitted_ccdf` |
| Gompertz B history | `year`, `gompertz_B`, `fit_status` | None |
| Pareto alpha history | `year`, `pareto_alpha_mle`, `fit_status` | None |
| Cutoff history | `year`, `cutoff`, `fit_status` | None |

The strict flow is therefore:

```text
data/trusted/
      │
      ▼
regime_analysis.py  (one estimation pass)
      │
      ├── paper_distribution_regime_fits.csv
      └── paper_distribution_regime_curves.parquet
                       │
                       ▼
                  plotting.py
                       │
                       ▼
                     PNGs
```

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
      ├── Gompertz-Pareto regime fits
      └── paper_* outputs
```
