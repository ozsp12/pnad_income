# Methodology

## Longitudinal income variable

The project follows the variable map encoded in `config/pnad_metadata.csv`. Some historical PNAD files provide household income and household size separately; for those years the standardized longitudinal variable is computed as household income divided by household size. Later PNAD files already provide the income measure used by the original analysis.

For PNAD Continua from 2016 through 2025, the project retains both `VD4019` (the measure used as the longitudinal `income` column in the inherited workflow) and `VD4020` as `income_effective`. This restores the comparison present in Beatriz's original notebooks but omitted from the first refactor.

## Historical monetary adjustment

The inherited transformation is retained:

```text
income_adj = (income / exchange) * inflation_to_2025
```

The adjustment factors now live only in `config/pnad_metadata.csv`; analytical notebooks do not carry private copies of the constants. This prevents the metadata/analysis divergence observed in the previous notebooks.

## CCDF

For a nonnegative income variable `X`, the complementary cumulative distribution function is evaluated at positive geometric thresholds `x` as

```text
CCDF(x) = P(X >= x).
```

The threshold grid starts at the smallest strictly positive observation, but zero-income observations remain in the normalization denominator. This distinction is essential: removing zero values before normalization would estimate the conditional quantity `P(X >= x | X > 0)` rather than the intended unconditional CCDF.

The project provides ordinary CCDF, log-log CCDF, and the legacy `ln[ln(CCDF)]` representation used in the inherited notebooks.

## Inequality

Annual descriptive analysis includes mean, median, standard deviation, support, Gini coefficient, and Lorenz curves for nominal and adjusted measures where available.
