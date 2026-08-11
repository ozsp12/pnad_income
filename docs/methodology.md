# Methodology

## Analytical unit and income measures

The analytical database is stored in long format, with one observation per row and one annual file per available survey year. The canonical longitudinal variable is `income`. When PNAD Contínua supplies an additional effective-income measure, it is stored separately as `income_effective`; the two measures are never silently substituted for one another.

Historical files that require construction of a per-capita measure are processed as household income divided by a strictly positive household-size variable. Survey-specific missing-value codes are removed before the transformation.

## Monetary standardization

Income is transformed to a common 2025 reference through

```text
income_adj = (income / exchange) * inflation_to_2025
```

and, when applicable,

```text
income_effective_adj = (income_effective / exchange) * inflation_to_2025
```

The factors `exchange`, `price_index`, and `inflation_to_2025` are attached by year from `config/pnad_metadata.csv` whenever they are absent from the analytical files. Centralizing these constants prevents different analytical modules from using inconsistent values.

## Complementary cumulative distribution function

For a nonnegative income variable \(X\),

\[
\overline F(x)=P(X\geq x).
\]

For a sample \(X_1,\ldots,X_N\), the empirical estimator is

\[
\widehat{\overline F}(x)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(X_i\geq x).
\]

Evaluation thresholds are geometrically spaced over the strictly positive support. The threshold grid begins at the smallest positive income, but the denominator contains every finite observation, including zeros. Removing zero-income observations before normalization would instead estimate the conditional quantity

\[
P(X\geq x\mid X>0).
\]

The package supports the ordinary CCDF, log-log representation, and the transformed diagnostic `ln[ln(CCDF)]`.

## Geometric income bins

If \(x_{\min}>0\) and \(x_{\max}\) denote the positive support limits and \(b>1\) is a multiplicative spacing factor, geometric thresholds have the form

\[
x_k=x_{\min}b^k,
\]

up to the observed upper support. For each interval the implementation records count, arithmetic mean, geometric mean, median, sample standard deviation, and the CCDF value at the left threshold.

## Descriptive statistics

For each year and available income measure, the project reports the number of finite observations, minimum positive value, maximum value, arithmetic mean, median, sample standard deviation, and Gini coefficient. Nominal and adjusted quantities remain distinct columns.

## Gini coefficient and Lorenz curve

For ordered nonnegative observations

\[
0\leq x_{(1)}\leq\cdots\leq x_{(N)},
\]

the implemented unweighted Gini coefficient is

\[
G=\frac{2\sum_{i=1}^{N}i x_{(i)}}{N\sum_{i=1}^{N}x_{(i)}}-\frac{N+1}{N}.
\]

The Lorenz curve is constructed from cumulative population shares and cumulative income shares. If all observations are zero, the implementation returns \(G=0\).

## Reproducibility

The notebook is an execution and reporting interface, not a second implementation. Numerical routines live in `src/pnad_income/`, are independently testable, and can be reused from scripts or other research workflows. This separation eliminates hidden notebook state and duplicated constants.
