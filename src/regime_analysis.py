"""Annual Gompertz-body/Pareto-tail least-squares fits on trusted income.

Positive finite adjusted income is normalized by its annual mean. For every
candidate transition, the Gompertz slope and continuity-constrained Pareto
exponent are estimated by least squares. The transition minimizes the two
regimes' combined squared error on the common ``log(F)`` scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis import compute_ccdf


GOMPERTZ_A_THEORETICAL = float(np.log(np.log(100.0)))
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
    "gompertz_A",
    "gompertz_B",
    "gompertz_r2",
    "gompertz_rmse",
    "gompertz_sse",
    "pareto_alpha",
    "pareto_beta",
    "pareto_r2",
    "pareto_rmse",
    "pareto_sse",
    "joint_sse",
    "continuity_error",
    "candidate_count",
    "second_best_cutoff_normalized",
    "second_best_joint_sse",
    "joint_sse_gap",
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
    """Shared settings for annual normalized profile least-squares fits."""

    ccdf_base: float = 1.05
    min_body_observations: int = 100
    min_tail_observations: int = 100
    min_tail_fraction: float = 0.005
    cutoff_quantile_min: float = 0.20
    cutoff_quantile_max: float = 0.995
    sensitivity_lower_bounds: tuple[float, ...] = (0.20, 0.40, 0.60)
    flat_profile_relative_tolerance: float = 1e-6
    near_optimal_relative_tolerance: float = 0.05
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
        if any(
            not self.cutoff_quantile_min <= q < self.cutoff_quantile_max
            for q in self.sensitivity_lower_bounds
        ):
            raise ValueError("Sensitivity lower bounds must lie inside the cutoff search interval.")
        if self.flat_profile_relative_tolerance < 0:
            raise ValueError("flat_profile_relative_tolerance must be nonnegative.")
        if self.near_optimal_relative_tolerance < 0:
            raise ValueError("near_optimal_relative_tolerance must be nonnegative.")
        if not 0 < self.weak_profile_width_fraction <= 1:
            raise ValueError("weak_profile_width_fraction must lie in (0, 1].")
        if self.weak_sensitivity_relative_change < 0:
            raise ValueError("weak_sensitivity_relative_change must be nonnegative.")
        if self.min_curve_points < 3:
            raise ValueError("min_curve_points must be at least 3.")


def _positive_finite(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return np.sort(array[np.isfinite(array) & (array > 0)])


def estimate_gompertz_ls(
    income_normalized,
    gompertz_transform,
    *,
    intercept: float = GOMPERTZ_A_THEORETICAL,
) -> float:
    """Estimate positive ``B`` in ``log(log(F)) = A - B*x`` with fixed ``A``."""

    x = np.asarray(income_normalized, dtype=float)
    y = np.asarray(gompertz_transform, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    denominator = float(np.dot(x, x))
    if x.size < 2 or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    estimate = float(np.dot(x, intercept - y) / denominator)
    return estimate if np.isfinite(estimate) and estimate > 0 else np.nan


def estimate_pareto_ls(
    income_normalized,
    empirical_ccdf_percent,
    cutoff: float,
    gompertz_ccdf_at_cutoff: float,
) -> float:
    """Estimate the Pareto CCDF exponent with continuity fixed at ``cutoff``."""

    x = np.asarray(income_normalized, dtype=float)
    F = np.asarray(empirical_ccdf_percent, dtype=float)
    cutoff = float(cutoff)
    G_cutoff = float(gompertz_ccdf_at_cutoff)
    finite = np.isfinite(x) & np.isfinite(F) & (x >= cutoff) & (F > 0)
    x = x[finite]
    F = F[finite]
    if x.size < 2 or not np.isfinite(cutoff) or cutoff <= 0:
        return np.nan
    if not np.isfinite(G_cutoff) or G_cutoff <= 0:
        return np.nan
    z = np.log(x / cutoff)
    w = np.log(G_cutoff) - np.log(F)
    denominator = float(np.dot(z, z))
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    estimate = float(np.dot(z, w) / denominator)
    return estimate if np.isfinite(estimate) and estimate > 0 else np.nan


def _fit_metrics(observed: np.ndarray, fitted: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(observed) & np.isfinite(fitted)
    observed = observed[finite]
    fitted = fitted[finite]
    if observed.size < 2:
        return np.nan, np.nan, np.nan
    residual = observed - fitted
    sse = float(np.sum(residual**2))
    tss = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1 - sse / tss if tss > 0 else (1.0 if np.isclose(sse, 0.0) else np.nan)
    return float(r2), float(np.sqrt(sse / observed.size)), sse


def _gompertz_percent(x: np.ndarray | float, B: float) -> np.ndarray:
    return np.exp(np.exp(GOMPERTZ_A_THEORETICAL - B * np.asarray(x, dtype=float)))


def _empty_summary(
    year: int,
    n_total: int,
    value_measure: str,
    failure_reason: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "year": int(year),
            "value_measure": value_measure,
            "n_total": int(n_total),
            "gompertz_A": GOMPERTZ_A_THEORETICAL,
            "candidate_count": 0,
            "selection_criterion": "joint_sse",
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
    curve: pd.DataFrame,
    cutoff: float,
    settings: RegimeFitConfig,
) -> dict[str, float] | None:
    n_total = int(normalized_values.size)
    raw_split = int(np.searchsorted(normalized_values, cutoff, side="left"))
    n_body = raw_split
    n_tail = n_total - n_body
    if n_body < settings.min_body_observations or n_tail < settings.min_tail_observations:
        return None
    tail_fraction = n_tail / n_total
    if tail_fraction < settings.min_tail_fraction:
        return None

    body_common = curve.loc[curve["income_normalized"] < cutoff]
    body_fit = body_common.loc[body_common["gompertz_transform"].notna()]
    tail_fit = curve.loc[curve["income_normalized"] >= cutoff]
    if min(len(body_fit), len(tail_fit)) < settings.min_curve_points:
        return None

    body_x = body_fit["income_normalized"].to_numpy(float)
    body_y = body_fit["gompertz_transform"].to_numpy(float)
    B = estimate_gompertz_ls(body_x, body_y)
    if not np.isfinite(B) or B <= 0:
        return None

    gompertz_at_cutoff = float(_gompertz_percent(cutoff, B))
    tail_x = tail_fit["income_normalized"].to_numpy(float)
    tail_F = tail_fit["empirical_ccdf_percent"].to_numpy(float)
    alpha = estimate_pareto_ls(tail_x, tail_F, cutoff, gompertz_at_cutoff)
    if not np.isfinite(alpha) or alpha <= 0:
        return None

    fitted_body_transform = GOMPERTZ_A_THEORETICAL - B * body_x
    gompertz_r2, gompertz_rmse, gompertz_sse = _fit_metrics(
        body_y, fitted_body_transform
    )
    log_gompertz_at_cutoff = float(np.log(gompertz_at_cutoff))
    fitted_tail_log = log_gompertz_at_cutoff - alpha * np.log(tail_x / cutoff)
    tail_log_F = np.log(tail_F)
    pareto_r2, pareto_rmse, pareto_sse = _fit_metrics(tail_log_F, fitted_tail_log)
    if not np.isfinite(
        [gompertz_r2, gompertz_rmse, gompertz_sse, pareto_r2, pareto_rmse, pareto_sse]
    ).all():
        return None

    common_body_x = body_common["income_normalized"].to_numpy(float)
    common_body_log_F = body_common["log_empirical_ccdf_percent"].to_numpy(float)
    fitted_body_log_F = np.exp(GOMPERTZ_A_THEORETICAL - B * common_body_x)
    body_common_sse = float(np.sum((common_body_log_F - fitted_body_log_F) ** 2))
    tail_common_sse = float(np.sum((tail_log_F - fitted_tail_log) ** 2))
    joint_sse = body_common_sse + tail_common_sse
    beta = float(gompertz_at_cutoff * cutoff**alpha)
    pareto_at_cutoff = float(beta * cutoff ** (-alpha))

    if not np.isfinite([joint_sse, beta, pareto_at_cutoff]).all() or joint_sse < 0:
        return None
    return {
        "cutoff_normalized": float(cutoff),
        "cutoff_quantile": float(n_body / n_total),
        "n_total": n_total,
        "n_body": n_body,
        "n_tail": n_tail,
        "body_fraction": float(n_body / n_total),
        "tail_fraction": float(tail_fraction),
        "gompertz_A": GOMPERTZ_A_THEORETICAL,
        "gompertz_B": float(B),
        "gompertz_r2": gompertz_r2,
        "gompertz_rmse": gompertz_rmse,
        "gompertz_sse": gompertz_sse,
        "pareto_alpha": float(alpha),
        "pareto_beta": beta,
        "pareto_r2": pareto_r2,
        "pareto_rmse": pareto_rmse,
        "pareto_sse": pareto_sse,
        "joint_sse": float(joint_sse),
        "continuity_error": float(abs(gompertz_at_cutoff - pareto_at_cutoff)),
    }


def _best_profile(profiles: list[dict[str, float]]) -> dict[str, float]:
    return min(profiles, key=lambda row: (row["joint_sse"], row["cutoff_normalized"]))


def _profile_diagnostics(
    profiles: list[dict[str, float]],
    best: dict[str, float],
    settings: RegimeFitConfig,
) -> dict[str, object]:
    ranked = sorted(profiles, key=lambda row: (row["joint_sse"], row["cutoff_normalized"]))
    second = ranked[1] if len(ranked) > 1 else None
    best_sse = float(best["joint_sse"])
    sse_values = np.asarray([row["joint_sse"] for row in profiles], dtype=float)
    profile_span = float(sse_values.max() - sse_values.min())
    profile_scale = max(abs(best_sse), np.finfo(float).eps)

    near_limit = best_sse + settings.near_optimal_relative_tolerance * profile_scale
    near_optimal = [row for row in profiles if row["joint_sse"] <= near_limit]
    near_lower = min(row["cutoff_normalized"] for row in near_optimal)
    near_upper = max(row["cutoff_normalized"] for row in near_optimal)
    profile_min = min(row["cutoff_normalized"] for row in profiles)
    profile_max = max(row["cutoff_normalized"] for row in profiles)
    search_width = max(profile_max - profile_min, np.finfo(float).eps)

    sensitivities: dict[float, float] = {}
    for lower in settings.sensitivity_lower_bounds:
        restricted = [row for row in profiles if row["cutoff_quantile"] >= lower]
        sensitivities[lower] = (
            float(_best_profile(restricted)["cutoff_normalized"]) if restricted else np.nan
        )
    sensitivity_values = np.asarray(list(sensitivities.values()), dtype=float)
    finite_sensitivity = sensitivity_values[np.isfinite(sensitivity_values)]
    sensitivity_spread = (
        float(finite_sensitivity.max() - finite_sensitivity.min()) if finite_sensitivity.size else np.nan
    )
    relative_changes = (
        np.abs(finite_sensitivity - best["cutoff_normalized"]) / best["cutoff_normalized"]
    )
    max_relative_change = float(relative_changes.max()) if relative_changes.size else np.nan

    best_cutoff = float(best["cutoff_normalized"])
    if profile_span <= settings.flat_profile_relative_tolerance * profile_scale:
        status = "flat_profile"
    elif np.isclose(best_cutoff, profile_min):
        status = "boundary_lower"
    elif np.isclose(best_cutoff, profile_max):
        status = "boundary_upper"
    elif (
        (near_upper - near_lower) / search_width >= settings.weak_profile_width_fraction
        or (
            np.isfinite(max_relative_change)
            and max_relative_change > settings.weak_sensitivity_relative_change
        )
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
        "second_best_joint_sse": float(second["joint_sse"]) if second is not None else np.nan,
        "joint_sse_gap": float(second["joint_sse"] - best_sse) if second is not None else np.nan,
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
    """Fit one annual normalized Gompertz-Pareto profile by least squares."""

    settings = config or RegimeFitConfig()
    adjusted_values = _positive_finite(values)
    minimum = settings.min_body_observations + settings.min_tail_observations
    if adjusted_values.size < minimum:
        return (
            _empty_summary(
                year,
                adjusted_values.size,
                value_measure,
                "insufficient_positive_observations",
            ),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )
    normalization_mean = float(adjusted_values.mean())
    if not np.isfinite(normalization_mean) or normalization_mean <= 0:
        return (
            _empty_summary(year, adjusted_values.size, value_measure, "invalid_normalization_mean"),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )

    x = adjusted_values / normalization_mean
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
    profiles = [
        candidate
        for cutoff in thresholds[eligible]
        if (candidate := _profile_candidate(x, curve, float(cutoff), settings)) is not None
    ]
    if not profiles:
        return (
            _empty_summary(year, x.size, value_measure, "no_admissible_valid_candidate"),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )

    best = _best_profile(profiles).copy()
    best.update(_profile_diagnostics(profiles, best, settings))
    cutoff = float(best["cutoff_normalized"])
    cutoff_income = cutoff * normalization_mean
    best.update(
        {
            "year": int(year),
            "value_measure": value_measure,
            "normalization_mean": normalization_mean,
            "cutoff_income_adj": cutoff_income,
            "selection_criterion": "joint_sse",
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
    fitted_transform = GOMPERTZ_A_THEORETICAL - best["gompertz_B"] * output_curve[
        "income_normalized"
    ]
    output_curve["gompertz_fitted_transform"] = np.where(body, fitted_transform, np.nan)
    output_curve["gompertz_fitted_ccdf_percent"] = np.where(
        body,
        _gompertz_percent(
            output_curve["income_normalized"].to_numpy(float), best["gompertz_B"]
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
