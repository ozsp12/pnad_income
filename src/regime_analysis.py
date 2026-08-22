"""Annual Gompertz-body/Pareto-tail estimation on normalized trusted income.

The default estimator follows the normalized Gompertz-Pareto specification in
Moura Jr. and Ribeiro (2009) and Figueira, Moura Jr. and Ribeiro (2011).  It
profiles the transition on individual observations with the theoretically
normalized Gompertz intercept fixed at ``ln(ln(100))``.  A free-intercept mode
is retained as an explicit diagnostic approximation, not as the recommended
probability model, because a free intercept does not enforce ``F(0) = 100``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis import compute_ccdf


GOMPERTZ_A_THEORETICAL = float(np.log(np.log(100.0)))
PROFILE_LR_95 = 3.841458820694124
FIT_STATUSES = {
    "ok_interior",
    "boundary_lower",
    "boundary_upper",
    "flat_profile",
    "weakly_identified",
    "no_valid_fit",
}

SUMMARY_COLUMNS = [
    "year",
    "value_measure",
    "normalization_mean",
    "cutoff_normalized",
    "cutoff_income_adj",
    "cutoff_quantile",
    "n_total",
    "n_body",
    "n_tail",
    "body_fraction",
    "tail_fraction",
    "gompertz_intercept_mode",
    "gompertz_A",
    "gompertz_A_theoretical",
    "gompertz_A_free_diagnostic",
    "gompertz_A_free_deviation",
    "gompertz_B",
    "gompertz_B_free_diagnostic",
    "gompertz_r2",
    "gompertz_rmse",
    "pareto_alpha",
    "pareto_density_exponent",
    "pareto_beta",
    "pareto_r2",
    "pareto_rmse",
    "pareto_ks",
    "continuity_error",
    "joint_log_likelihood",
    "joint_aic",
    "joint_bic",
    "likelihood_type",
    "candidate_count",
    "second_best_cutoff_normalized",
    "second_best_log_likelihood",
    "log_likelihood_difference",
    "profile_ci_lower",
    "profile_ci_upper",
    "profile_ci_width",
    "sensitivity_cutoff_p20",
    "sensitivity_cutoff_p40",
    "sensitivity_cutoff_p60",
    "sensitivity_spread",
    "sensitivity_max_relative_change",
    "selection_criterion",
    "fit_status",
    "failure_reason",
]

CURVE_COLUMNS = [
    "year",
    "income_normalized",
    "income_adj_2025_usd",
    "empirical_ccdf_percent",
    "gompertz_transform",
    "log_income_normalized",
    "log_empirical_ccdf_percent",
    "regime",
    "cutoff_normalized",
    "cutoff_income_adj",
    "gompertz_fitted_transform",
    "gompertz_fitted_ccdf_percent",
    "pareto_fitted_ccdf_percent",
    "pareto_fitted_log_ccdf",
]


@dataclass(frozen=True)
class RegimeFitConfig:
    """Shared settings for annual normalized Gompertz-Pareto profiles."""

    ccdf_base: float = 1.05
    min_body_observations: int = 100
    min_tail_observations: int = 100
    min_tail_fraction: float = 0.005
    cutoff_quantile_min: float = 0.20
    cutoff_quantile_max: float = 0.995
    selection_criterion: str = "log_likelihood"
    gompertz_intercept_mode: str = "fixed"
    sensitivity_lower_bounds: tuple[float, ...] = (0.20, 0.40, 0.60)
    flat_profile_loglik_tolerance: float = 1e-6
    weak_profile_width_fraction: float = 0.50
    weak_sensitivity_relative_change: float = 0.25
    min_curve_points: int = 5

    def __post_init__(self) -> None:
        if self.ccdf_base <= 1:
            raise ValueError("ccdf_base must be greater than 1.")
        if min(self.min_body_observations, self.min_tail_observations) < 2:
            raise ValueError("Minimum regime observation counts must be at least 2.")
        if not 0 < self.min_tail_fraction < 1:
            raise ValueError("min_tail_fraction must lie strictly between 0 and 1.")
        if not 0 < self.cutoff_quantile_min < self.cutoff_quantile_max < 1:
            raise ValueError("Cutoff quantiles must satisfy 0 < min < max < 1.")
        if self.selection_criterion not in {"log_likelihood", "aic", "bic"}:
            raise ValueError("selection_criterion must be log_likelihood, aic, or bic.")
        if self.gompertz_intercept_mode not in {"fixed", "free"}:
            raise ValueError("gompertz_intercept_mode must be fixed or free.")
        if any(
            not self.cutoff_quantile_min <= q < self.cutoff_quantile_max
            for q in self.sensitivity_lower_bounds
        ):
            raise ValueError("Sensitivity lower bounds must lie inside the cutoff search interval.")
        if self.flat_profile_loglik_tolerance < 0:
            raise ValueError("flat_profile_loglik_tolerance must be nonnegative.")
        if not 0 < self.weak_profile_width_fraction <= 1:
            raise ValueError("weak_profile_width_fraction must lie in (0, 1].")
        if self.weak_sensitivity_relative_change < 0:
            raise ValueError("weak_sensitivity_relative_change must be nonnegative.")
        if self.min_curve_points < 3:
            raise ValueError("min_curve_points must be at least 3.")


def _positive_finite(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return np.sort(array[np.isfinite(array) & (array > 0)])


def estimate_pareto_mle(values, cutoff: float) -> float:
    """Estimate the Pareto CCDF exponent ``alpha`` above ``cutoff``.

    The fitted survival law is ``P(x) = beta * x**(-alpha)``.  Its density
    exponent is therefore ``alpha + 1`` and is reported separately.
    """

    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and strictly positive.")
    tail = _positive_finite(values)
    tail = tail[tail >= cutoff]
    if tail.size == 0:
        return np.nan
    denominator = float(np.log(tail / cutoff).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(tail.size / denominator)


def _fit_metrics(observed: np.ndarray, fitted: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(observed) & np.isfinite(fitted)
    observed = observed[finite]
    fitted = fitted[finite]
    if observed.size < 2:
        return np.nan, np.nan
    residual = observed - fitted
    rss = float(np.sum(residual**2))
    tss = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else (1.0 if np.isclose(rss, 0.0) else np.nan)
    return float(r2), float(np.sqrt(np.mean(residual**2)))


def _free_gompertz_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return np.nan, np.nan
    design = np.column_stack([np.ones(x.size), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(-slope)


def _gompertz_percent(x: np.ndarray | float, A: float, B: float) -> np.ndarray:
    return np.exp(np.exp(A - B * np.asarray(x, dtype=float)))


def _fixed_log_likelihood(
    unique_values: np.ndarray,
    counts: np.ndarray,
    split: int,
    cutoff: float,
    B: float,
    alpha: float,
    *,
    A: float = GOMPERTZ_A_THEORETICAL,
) -> float:
    if not np.isfinite(B) or B <= 0 or not np.isfinite(alpha) or alpha <= 1:
        return -np.inf
    body_x = unique_values[:split]
    body_n = counts[:split]
    tail_x = unique_values[split:]
    tail_n = counts[split:]
    body_log_density = np.log(B) + A - B * body_x + np.exp(A - B * body_x) - np.log(100.0)
    log_beta = float(np.exp(A - B * cutoff) + alpha * np.log(cutoff))
    tail_log_density = np.log(alpha) + log_beta - (alpha + 1.0) * np.log(tail_x) - np.log(100.0)
    return float(np.dot(body_n, body_log_density) + np.dot(tail_n, tail_log_density))


def _profile_B(
    unique_values: np.ndarray,
    counts: np.ndarray,
    split: int,
    cutoff: float,
    alpha: float,
    initial: float,
) -> tuple[float, float]:
    """Maximize the fixed-intercept microdata likelihood over positive ``B``."""

    body_x = unique_values[:split]
    body_n = counts[:split]
    n_body = float(body_n.sum())
    n_tail = float(counts[split:].sum())
    sum_body_x = float(np.dot(body_n, body_x))

    def score(B: float) -> float:
        exponential = np.exp(GOMPERTZ_A_THEORETICAL - B * body_x)
        return float(
            n_body / B
            - sum_body_x
            - np.dot(body_n, body_x * exponential)
            - n_tail * cutoff * np.exp(GOMPERTZ_A_THEORETICAL - B * cutoff)
        )

    lower, upper = 1e-6, 20.0
    low_score, high_score = score(lower), score(upper)
    if not np.isfinite(low_score) or not np.isfinite(high_score) or low_score <= 0 or high_score >= 0:
        candidates = np.array([np.clip(initial, lower, upper), lower, upper], dtype=float)
    else:
        lo, hi = lower, upper
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if score(mid) > 0:
                lo = mid
            else:
                hi = mid
        candidates = np.array([0.5 * (lo + hi), np.clip(initial, lower, upper)], dtype=float)
    likelihoods = np.array(
        [_fixed_log_likelihood(unique_values, counts, split, cutoff, B, alpha) for B in candidates]
    )
    best_index = int(np.nanargmax(likelihoods))
    return float(candidates[best_index]), float(likelihoods[best_index])


def _pareto_ks(sorted_tail: np.ndarray, cutoff: float, alpha: float) -> float:
    if sorted_tail.size == 0 or not np.isfinite(alpha) or alpha <= 1:
        return np.nan
    model_cdf = 1.0 - np.power(sorted_tail / cutoff, -alpha)
    n = sorted_tail.size
    upper = np.arange(1, n + 1, dtype=float) / n
    lower = np.arange(0, n, dtype=float) / n
    return float(max(np.max(upper - model_cdf), np.max(model_cdf - lower)))


def _empty_summary(
    year: int,
    n_total: int,
    value_measure: str,
    selection_criterion: str,
    intercept_mode: str,
    failure_reason: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "year": int(year),
            "value_measure": value_measure,
            "n_total": int(n_total),
            "gompertz_intercept_mode": intercept_mode,
            "gompertz_A_theoretical": GOMPERTZ_A_THEORETICAL,
            "candidate_count": 0,
            "selection_criterion": selection_criterion,
            "fit_status": "no_valid_fit",
            "failure_reason": failure_reason,
        }
    )
    return row


def _empirical_curve(values: np.ndarray, normalization_mean: float, base: float) -> pd.DataFrame:
    curve = compute_ccdf(values, base=base, scale="percent")[["bin", "ccdf"]].rename(
        columns={"bin": "income_normalized", "ccdf": "empirical_ccdf_percent"}
    )
    income = curve["income_normalized"].to_numpy(float)
    percent = curve["empirical_ccdf_percent"].to_numpy(float)
    transform = np.full(percent.shape, np.nan, dtype=float)
    valid_transform = (percent > 1.0) & (percent <= 100.0)
    transform[valid_transform] = np.log(np.log(percent[valid_transform]))
    curve["income_adj_2025_usd"] = income * normalization_mean
    curve["gompertz_transform"] = transform
    curve["log_income_normalized"] = np.log(income)
    curve["log_empirical_ccdf_percent"] = np.log(percent)
    return curve


def _profile_candidate(
    normalized_values: np.ndarray,
    unique_values: np.ndarray,
    counts: np.ndarray,
    cumulative_counts: np.ndarray,
    cumulative_log_sums: np.ndarray,
    curve: pd.DataFrame,
    cutoff: float,
    settings: RegimeFitConfig,
) -> dict[str, float] | None:
    n_total = int(normalized_values.size)
    unique_split = int(np.searchsorted(unique_values, cutoff, side="left"))
    n_body = int(cumulative_counts[unique_split])
    n_tail = n_total - n_body
    if n_body < settings.min_body_observations or n_tail < settings.min_tail_observations:
        return None
    tail_fraction = n_tail / n_total
    if tail_fraction < settings.min_tail_fraction:
        return None

    body_curve = curve.loc[
        (curve["income_normalized"] < cutoff) & curve["gompertz_transform"].notna()
    ]
    tail_curve = curve.loc[curve["income_normalized"] >= cutoff]
    if min(len(body_curve), len(tail_curve)) < settings.min_curve_points:
        return None

    body_x = body_curve["income_normalized"].to_numpy(float)
    body_y = body_curve["gompertz_transform"].to_numpy(float)
    free_A, free_B = _free_gompertz_fit(body_x, body_y)
    if not np.isfinite(free_B) or free_B <= 0:
        return None

    tail_log_sum = float(cumulative_log_sums[-1] - cumulative_log_sums[unique_split])
    denominator = tail_log_sum - n_tail * np.log(cutoff)
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    alpha = float(n_tail / denominator)
    # Normalizing income by its annual mean requires the fitted Pareto tail to
    # have a finite first moment.  For a CCDF exponent this means alpha > 1.
    if not np.isfinite(alpha) or alpha <= 1:
        return None

    if settings.gompertz_intercept_mode == "fixed":
        A = GOMPERTZ_A_THEORETICAL
        fixed_initial = float(np.dot(body_x, A - body_y) / np.dot(body_x, body_x))
        if not np.isfinite(fixed_initial) or fixed_initial <= 0:
            fixed_initial = free_B
        B, joint_log_likelihood = _profile_B(
            unique_values,
            counts,
            unique_split,
            cutoff,
            alpha,
            fixed_initial,
        )
        likelihood_type = "normalized_piecewise_microdata"
        parameter_count = 3
    else:
        A, B = free_A, free_B
        joint_log_likelihood = _fixed_log_likelihood(
            unique_values,
            counts,
            unique_split,
            cutoff,
            B,
            alpha,
            A=A,
        )
        likelihood_type = "free_A_unnormalized_quasi_likelihood"
        parameter_count = 4
    if not np.isfinite([A, B, joint_log_likelihood]).all() or B <= 0:
        return None

    gompertz_fitted_transform = A - B * body_x
    gompertz_r2, gompertz_rmse = _fit_metrics(body_y, gompertz_fitted_transform)
    gompertz_at_cutoff = float(_gompertz_percent(cutoff, A, B))
    beta = float(gompertz_at_cutoff * cutoff**alpha)
    tail_x = tail_curve["income_normalized"].to_numpy(float)
    tail_log_empirical = tail_curve["log_empirical_ccdf_percent"].to_numpy(float)
    tail_log_fitted = np.log(beta) - alpha * np.log(tail_x)
    pareto_r2, pareto_rmse = _fit_metrics(tail_log_empirical, tail_log_fitted)
    pareto_at_cutoff = float(beta * cutoff ** (-alpha))
    continuity_error = float(abs(gompertz_at_cutoff - pareto_at_cutoff))
    joint_aic = float(2 * parameter_count - 2 * joint_log_likelihood)
    joint_bic = float(parameter_count * np.log(n_total) - 2 * joint_log_likelihood)

    raw_split = int(np.searchsorted(normalized_values, cutoff, side="left"))
    return {
        "cutoff_normalized": float(cutoff),
        "cutoff_quantile": float(n_body / n_total),
        "n_total": n_total,
        "n_body": n_body,
        "n_tail": n_tail,
        "body_fraction": float(n_body / n_total),
        "tail_fraction": float(tail_fraction),
        "gompertz_A": float(A),
        "gompertz_A_free_diagnostic": free_A,
        "gompertz_A_free_deviation": float(free_A - GOMPERTZ_A_THEORETICAL),
        "gompertz_B": float(B),
        "gompertz_B_free_diagnostic": free_B,
        "gompertz_r2": gompertz_r2,
        "gompertz_rmse": gompertz_rmse,
        "pareto_alpha": alpha,
        "pareto_density_exponent": float(alpha + 1.0),
        "pareto_beta": beta,
        "pareto_r2": pareto_r2,
        "pareto_rmse": pareto_rmse,
        "pareto_ks": _pareto_ks(normalized_values[raw_split:], cutoff, alpha),
        "continuity_error": continuity_error,
        "joint_log_likelihood": joint_log_likelihood,
        "joint_aic": joint_aic,
        "joint_bic": joint_bic,
        "likelihood_type": likelihood_type,
    }


def _best_profile(profiles: list[dict[str, float]], criterion: str) -> dict[str, float]:
    if criterion == "log_likelihood":
        return max(profiles, key=lambda row: (row["joint_log_likelihood"], -row["cutoff_normalized"]))
    key = f"joint_{criterion}"
    return min(profiles, key=lambda row: (row[key], row["cutoff_normalized"]))


def _profile_diagnostics(
    profiles: list[dict[str, float]],
    best: dict[str, float],
    settings: RegimeFitConfig,
) -> dict[str, object]:
    ranked = sorted(profiles, key=lambda row: row["joint_log_likelihood"], reverse=True)
    second = ranked[1] if len(ranked) > 1 else None
    best_ll = float(best["joint_log_likelihood"])
    eligible_ci = [
        row for row in profiles if 2.0 * (best_ll - float(row["joint_log_likelihood"])) <= PROFILE_LR_95
    ]
    ci_lower = min(row["cutoff_normalized"] for row in eligible_ci)
    ci_upper = max(row["cutoff_normalized"] for row in eligible_ci)
    profile_min = min(row["cutoff_normalized"] for row in profiles)
    profile_max = max(row["cutoff_normalized"] for row in profiles)
    search_width = max(profile_max - profile_min, np.finfo(float).eps)
    profile_ll_span = max(row["joint_log_likelihood"] for row in profiles) - min(
        row["joint_log_likelihood"] for row in profiles
    )

    sensitivities: dict[float, float] = {}
    for lower in settings.sensitivity_lower_bounds:
        restricted = [row for row in profiles if row["cutoff_quantile"] >= lower]
        sensitivities[lower] = (
            float(_best_profile(restricted, settings.selection_criterion)["cutoff_normalized"])
            if restricted
            else np.nan
        )
    sensitivity_values = np.array(list(sensitivities.values()), dtype=float)
    finite_sensitivity = sensitivity_values[np.isfinite(sensitivity_values)]
    sensitivity_spread = (
        float(finite_sensitivity.max() - finite_sensitivity.min()) if finite_sensitivity.size else np.nan
    )
    relative_changes = np.abs(finite_sensitivity - best["cutoff_normalized"]) / best["cutoff_normalized"]
    max_relative_change = float(relative_changes.max()) if relative_changes.size else np.nan

    best_cutoff = float(best["cutoff_normalized"])
    if profile_ll_span <= settings.flat_profile_loglik_tolerance:
        status = "flat_profile"
    elif np.isclose(best_cutoff, profile_min):
        status = "boundary_lower"
    elif np.isclose(best_cutoff, profile_max):
        status = "boundary_upper"
    elif (
        (ci_upper - ci_lower) / search_width >= settings.weak_profile_width_fraction
        or (np.isfinite(max_relative_change) and max_relative_change > settings.weak_sensitivity_relative_change)
    ):
        status = "weakly_identified"
    else:
        status = "ok_interior"

    lookup = {round(key * 100): value for key, value in sensitivities.items()}
    return {
        "candidate_count": len(profiles),
        "second_best_cutoff_normalized": (
            float(second["cutoff_normalized"]) if second is not None else np.nan
        ),
        "second_best_log_likelihood": (
            float(second["joint_log_likelihood"]) if second is not None else np.nan
        ),
        "log_likelihood_difference": (
            float(best_ll - second["joint_log_likelihood"]) if second is not None else np.nan
        ),
        "profile_ci_lower": float(ci_lower),
        "profile_ci_upper": float(ci_upper),
        "profile_ci_width": float(ci_upper - ci_lower),
        "sensitivity_cutoff_p20": float(lookup.get(20, np.nan)),
        "sensitivity_cutoff_p40": float(lookup.get(40, np.nan)),
        "sensitivity_cutoff_p60": float(lookup.get(60, np.nan)),
        "sensitivity_spread": sensitivity_spread,
        "sensitivity_max_relative_change": max_relative_change,
        "fit_status": status,
    }


def fit_year_distribution_regime(
    values,
    year: int,
    *,
    value_measure: str = "income_adj",
    config: RegimeFitConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit one annual normalized Gompertz-Pareto transition profile."""

    settings = config or RegimeFitConfig()
    adjusted_values = _positive_finite(values)
    minimum = settings.min_body_observations + settings.min_tail_observations
    if adjusted_values.size < minimum:
        return (
            _empty_summary(
                year,
                adjusted_values.size,
                value_measure,
                settings.selection_criterion,
                settings.gompertz_intercept_mode,
                "insufficient_positive_observations",
            ),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )
    normalization_mean = float(adjusted_values.mean())
    if not np.isfinite(normalization_mean) or normalization_mean <= 0:
        return (
            _empty_summary(
                year,
                adjusted_values.size,
                value_measure,
                settings.selection_criterion,
                settings.gompertz_intercept_mode,
                "invalid_normalization_mean",
            ),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )
    x = adjusted_values / normalization_mean
    unique_values, counts = np.unique(x, return_counts=True)
    counts = counts.astype(float)
    cumulative_counts = np.concatenate([[0.0], np.cumsum(counts)])
    cumulative_log_sums = np.concatenate([[0.0], np.cumsum(counts * np.log(unique_values))])
    curve = _empirical_curve(x, normalization_mean, settings.ccdf_base)
    thresholds = curve["income_normalized"].to_numpy(float)
    split_indices = np.searchsorted(x, thresholds, side="left")
    quantiles = split_indices / x.size
    n_tail = x.size - split_indices
    eligible = (
        (split_indices >= settings.min_body_observations)
        & (n_tail >= settings.min_tail_observations)
        & (n_tail / x.size >= settings.min_tail_fraction)
        & (quantiles >= settings.cutoff_quantile_min)
        & (quantiles <= settings.cutoff_quantile_max)
    )
    candidates = thresholds[eligible]
    profiles = [
        candidate
        for cutoff in candidates
        if (
            candidate := _profile_candidate(
                x,
                unique_values,
                counts,
                cumulative_counts,
                cumulative_log_sums,
                curve,
                float(cutoff),
                settings,
            )
        )
        is not None
    ]
    if not profiles:
        return (
            _empty_summary(
                year,
                x.size,
                value_measure,
                settings.selection_criterion,
                settings.gompertz_intercept_mode,
                "no_admissible_valid_candidate",
            ),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )

    best = _best_profile(profiles, settings.selection_criterion).copy()
    best.update(_profile_diagnostics(profiles, best, settings))
    cutoff = float(best["cutoff_normalized"])
    cutoff_income = cutoff * normalization_mean
    best.update(
        {
            "year": int(year),
            "value_measure": value_measure,
            "normalization_mean": normalization_mean,
            "cutoff_income_adj": cutoff_income,
            "gompertz_intercept_mode": settings.gompertz_intercept_mode,
            "gompertz_A_theoretical": GOMPERTZ_A_THEORETICAL,
            "selection_criterion": settings.selection_criterion,
            "failure_reason": "",
        }
    )

    output_curve = curve.copy()
    output_curve.insert(0, "year", int(year))
    output_curve["regime"] = np.where(
        output_curve["income_normalized"] < cutoff, "gompertz_body", "pareto_tail"
    )
    output_curve["cutoff_normalized"] = cutoff
    output_curve["cutoff_income_adj"] = cutoff_income
    body = output_curve["regime"] == "gompertz_body"
    tail = ~body
    fitted_transform = best["gompertz_A"] - best["gompertz_B"] * output_curve["income_normalized"]
    output_curve["gompertz_fitted_transform"] = np.where(body, fitted_transform, np.nan)
    output_curve["gompertz_fitted_ccdf_percent"] = np.where(
        body,
        _gompertz_percent(
            output_curve["income_normalized"].to_numpy(float),
            best["gompertz_A"],
            best["gompertz_B"],
        ),
        np.nan,
    )
    pareto_log = np.log(best["pareto_beta"]) - best["pareto_alpha"] * output_curve[
        "log_income_normalized"
    ]
    output_curve["pareto_fitted_log_ccdf"] = np.where(tail, pareto_log, np.nan)
    output_curve["pareto_fitted_ccdf_percent"] = np.where(tail, np.exp(pareto_log), np.nan)
    output_curve = output_curve[CURVE_COLUMNS]
    return {column: best.get(column, np.nan) for column in SUMMARY_COLUMNS}, output_curve


def fit_distribution_regimes(
    df: pd.DataFrame,
    *,
    value_col: str = "income_adj",
    year_col: str = "year",
    config: RegimeFitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one annual fit row and persisted plotting curves for each year."""

    missing = {year_col, value_col}.difference(df.columns)
    if missing:
        raise KeyError("Regime analysis is missing: " + ", ".join(sorted(missing)))
    fits: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    for year, group in df.groupby(year_col, sort=True):
        fit, curve = fit_year_distribution_regime(
            group[value_col],
            int(year),
            value_measure=value_col,
            config=config,
        )
        fits.append(fit)
        if not curve.empty:
            curves.append(curve)
    fit_frame = pd.DataFrame(fits, columns=SUMMARY_COLUMNS).sort_values("year").reset_index(drop=True)
    curve_frame = (
        pd.concat(curves, ignore_index=True)
        .sort_values(["year", "income_normalized"])
        .reset_index(drop=True)
        if curves
        else pd.DataFrame(columns=CURVE_COLUMNS)
    )
    return fit_frame, curve_frame
