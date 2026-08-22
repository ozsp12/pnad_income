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

The annual regime analysis uses only positive, finite `income_adj` observations from the trusted layer. Each year is normalized before estimation,

$$
x_i=\frac{\texttt{income\_adj}_i}{\overline{\texttt{income\_adj}}},
$$

so the structural parameters and transition can be compared on a currency-free scale. The asset also retains the annual mean and maps the selected cutoff back to adjusted 2025 USD.

The empirical complementary cumulative distribution is expressed in percent,

$$
F(x)=100\,\frac{\#(X\ge x)}{N}.
$$

The Gompertz body follows

$$
G(x)=\exp\{\exp(A-Bx)\},\qquad \ln[\ln G(x)]=A-Bx,
$$

with $B>0$. The boundary condition $G(0)=100$ fixes

$$
A_0=\ln[\ln(100)]=1.5271796258\ldots.
$$

`gompertz_intercept_mode="fixed"` is therefore the default and scientifically recommended mode. A free-$A$ mode is implemented for comparison with unconstrained linear diagnostics, and the fixed-mode asset always reports the corresponding free-$A$ diagnostic. Because free $A$ does not enforce $F(0)=100$, that alternative is explicitly labeled `free_A_unnormalized_quasi_likelihood` and must not be interpreted as a normalized probability model.

For $x\ge x_t$, the Pareto CCDF is

$$
P(x)=\beta x^{-\alpha},\qquad
\widehat{\alpha}=\frac{n_{tail}}{\sum_i\ln(x_i/x_t)}.
$$

Thus `pareto_alpha` is the CCDF exponent and the density exponent is stored separately as `pareto_density_exponent = pareto_alpha + 1`. Because the model is fitted to income normalized by its annual mean, admissible candidates must satisfy $\alpha>1$ so the Pareto component has a finite first moment. Continuity is imposed rather than estimated independently:

$$
\beta=G(x_t)x_t^{\alpha}.
$$

`continuity_error` records the numerical difference between the two fitted CCDF branches at $x_t$ and should be zero up to floating-point precision.

In fixed-$A$ mode, each candidate cutoff is scored with the proper individual-observation likelihood of the continuous piecewise density. The body density is $B\exp(A-Bx)G(x)/100$ and the tail density is $\alpha\beta x^{-(\alpha+1)}/100$. Their masses sum to one because $A=A_0$ and continuity is enforced. $B$ is profiled as a positive parameter, $\alpha$ uses the Pareto MLE above, and the default selection maximizes joint log-likelihood; joint AIC and BIC are also persisted. No residual vectors from different transformed coordinates are concatenated. Pareto KS is an independent post-fit diagnostic.

The default cutoff search spans empirical cumulative quantiles p20 through p99.5, with at least 100 observations in each regime and at least 0.5% of observations in the tail. The broad interval is intentional: a boundary solution is evidence against a precisely located interior transition, not a successful interior fit. `fit_status` is one of `ok_interior`, `boundary_lower`, `boundary_upper`, `flat_profile`, `weakly_identified`, or `no_valid_fit`; `failure_reason` gives details for invalid fits. The asset includes the second-best cutoff, its log-likelihood difference, a discrete 95% likelihood-ratio profile interval, and restricted-search sensitivities with lower bounds p20, p40, and p60. Wide profiles or material cutoff sensitivity are labeled `weakly_identified`.

These conventions follow the primary Brazilian-income studies [Moura Jr. and Ribeiro (2009)](https://arxiv.org/abs/0812.2664) and [Figueira, Moura Jr. and Ribeiro (2011)](https://arxiv.org/abs/1010.1994). Direct numerical agreement is not assumed: this repository uses its own trusted-data treatment, adjusted-income definition, geometric CCDF grid, positive-record restriction, and record weighting.

### Derived-data interface for regime figures

The estimator writes two public analytical assets:

- `paper_distribution_regime_fits.csv` contains one row per survey year with the annual normalization, normalized and 2025-USD cutoffs, sample allocation, both Gompertz intercept diagnostics, positive $B$, Pareto CCDF and density exponents, continuity, R²/RMSE/KS, likelihood/AIC/BIC, profile-identification diagnostics, sensitivities, and status.
- `paper_distribution_regime_curves.parquet` contains the normalized and 2025-USD axes, empirical percent CCDF, correct Gompertz and log-log transforms, regime label, both cutoffs, and fitted branches.

`outputs.py` persists these datasets and reloads them before creating any regime figure. The plotting functions accept only the reloaded frames: they neither read `data/trusted/` nor `data/refined/`, access `PipelineResults.panel`, reconstruct a CCDF, rerun MLE, or search for a cutoff. The asset-to-figure column contract is:

| Figure family | Fit columns | Curve columns |
|---|---|---|
| Annual dual-panel regime fits | `year`, `cutoff_normalized`, `cutoff_income_adj`, `pareto_alpha`, `fit_status` | `year`, `income_normalized`, `empirical_ccdf_percent`, `gompertz_transform`, `regime`, `gompertz_fitted_transform`, `pareto_fitted_ccdf_percent` |
| Gompertz B history | `year`, `gompertz_B`, `fit_status` | None |
| Pareto alpha history | `year`, `pareto_alpha`, `fit_status` | None |
| Cutoff history | `year`, `cutoff_normalized`, `fit_status` | None |
| Gompertz and Pareto R² history | `year`, `gompertz_r2`, `pareto_r2`, `fit_status` | None |

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

The dual-panel figures use the persisted normalized axis: $x$ versus $\ln[\ln F(x)]$ for the Gompertz body and log $x$ versus log percent CCDF for the Pareto tail. Current estimators are record-weighted. Population inference requires the appropriate PNAD survey weights and design information; the present implementation should therefore be interpreted as analysis of harmonized records rather than a survey-design-corrected population estimator.

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
