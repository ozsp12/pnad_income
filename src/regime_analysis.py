"""Profile estimation of Gompertz-body and Pareto-tail income regimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis import compute_ccdf, gompertz_transform


SUMMARY_COLUMNS = [
    "year",
    "value_measure",
    "cutoff",
    "cutoff_quantile",
    "n_total",
    "n_body",
    "n_tail",
    "tail_fraction",
    "gompertz_A",
    "gompertz_B",
    "gompertz_r2",
    "gompertz_rmse",
    "gompertz_aic",
    "gompertz_bic",
    "pareto_alpha_mle",
    "pareto_ccdf_slope",
    "pareto_intercept",
    "pareto_r2",
    "pareto_rmse",
    "pareto_aic",
    "pareto_bic",
    "pareto_ks",
    "joint_aic",
    "joint_bic",
    "candidate_count",
    "selection_criterion",
    "fit_status",
]

CURVE_COLUMNS = [
    "year",
    "income",
    "empirical_ccdf",
    "gompertz_transform",
    "log_income",
    "log_empirical_ccdf",
    "regime",
    "cutoff",
    "gompertz_fitted_transform",
    "gompertz_fitted_ccdf",
    "pareto_fitted_ccdf",
    "pareto_fitted_log_ccdf",
]


@dataclass(frozen=True)
class RegimeFitConfig:
    """Shared profile-search settings applied independently to every year."""

    ccdf_base: float = 1.05
    min_body_observations: int = 100
    min_tail_observations: int = 100
    min_tail_fraction: float = 0.01
    cutoff_quantile_min: float = 0.80
    cutoff_quantile_max: float = 0.99
    selection_criterion: str = "bic"

    def __post_init__(self) -> None:
        if self.ccdf_base <= 1:
            raise ValueError("ccdf_base must be greater than 1.")
        if self.min_body_observations < 3 or self.min_tail_observations < 3:
            raise ValueError("Both minimum-observation settings must be at least 3.")
        if not 0 < self.min_tail_fraction < 1:
            raise ValueError("min_tail_fraction must lie in (0, 1).")
        if not 0 < self.cutoff_quantile_min < self.cutoff_quantile_max < 1:
            raise ValueError("Cutoff quantiles must satisfy 0 < minimum < maximum < 1.")
        if self.selection_criterion not in {"aic", "bic"}:
            raise ValueError("selection_criterion must be 'aic' or 'bic'.")


def _positive_finite(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return np.sort(x[np.isfinite(x) & (x > 0)])


def estimate_pareto_mle(values, cutoff: float) -> float:
    """Estimate the continuous Pareto density exponent above ``cutoff``.

    The density convention is ``p(x) proportional to x**(-alpha)``. Its
    conditional CCDF therefore has slope ``1 - alpha`` in log-log space.
    """
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and strictly positive.")
    x = _positive_finite(values)
    tail = x[x >= cutoff]
    if tail.size == 0:
        return np.nan
    denominator = float(np.log(tail / cutoff).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(1 + tail.size / denominator)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    design = np.column_stack([np.ones(x.size), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    return float(intercept), float(slope), fitted


def _fit_metrics(observed: np.ndarray, fitted: np.ndarray) -> tuple[float, float]:
    residual = observed - fitted
    rss = float(np.sum(residual**2))
    tss = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else (1.0 if np.isclose(rss, 0.0) else np.nan)
    rmse = float(np.sqrt(np.mean(residual**2)))
    return float(r2), rmse


def _information_criteria(residual: np.ndarray, parameters: int) -> tuple[float, float]:
    residual = residual[np.isfinite(residual)]
    n = residual.size
    if n <= parameters:
        return np.nan, np.nan
    rss = max(float(np.sum(residual**2)), np.finfo(float).tiny)
    deviance = n * np.log(rss / n)
    return float(deviance + 2 * parameters), float(deviance + parameters * np.log(n))


def _gompertz_log_ccdf(transform: np.ndarray | pd.Series) -> np.ndarray:
    """Evaluate ``log(S) = -exp(-transform)`` without overflow."""
    exponent = np.clip(-np.asarray(transform, dtype=float), -700.0, 700.0)
    return -np.exp(exponent)


def _pareto_ks(sorted_tail: np.ndarray, cutoff: float, alpha: float) -> float:
    n = sorted_tail.size
    if n == 0 or not np.isfinite(alpha) or alpha <= 1:
        return np.nan
    model_cdf = 1 - np.power(sorted_tail / cutoff, -(alpha - 1))
    upper = np.arange(1, n + 1, dtype=float) / n
    lower = np.arange(0, n, dtype=float) / n
    return float(max(np.max(upper - model_cdf), np.max(model_cdf - lower)))


def _empty_summary(
    year: int,
    n_total: int,
    value_measure: str,
    selection_criterion: str,
    status: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "year": int(year),
            "value_measure": value_measure,
            "n_total": int(n_total),
            "candidate_count": 0,
            "selection_criterion": selection_criterion,
            "fit_status": status,
        }
    )
    return row


def _empirical_curve(values: np.ndarray, base: float) -> pd.DataFrame:
    curve = compute_ccdf(values, base=base, scale="probability")[["bin", "ccdf"]].rename(
        columns={"bin": "income", "ccdf": "empirical_ccdf"}
    )
    probability = curve["empirical_ccdf"].to_numpy(float)
    income = curve["income"].to_numpy(float)
    curve["gompertz_transform"] = gompertz_transform(probability)
    curve["log_income"] = np.log(income)
    curve["log_empirical_ccdf"] = np.log(probability)
    return curve


def _profile_candidate(
    sorted_values: np.ndarray,
    curve: pd.DataFrame,
    cutoff: float,
) -> dict[str, float] | None:
    n_total = sorted_values.size
    split = int(np.searchsorted(sorted_values, cutoff, side="left"))
    n_body = split
    n_tail = n_total - split
    tail_fraction = n_tail / n_total

    body = curve.loc[
        (curve["income"] < cutoff) & curve["gompertz_transform"].notna()
    ].copy()
    tail = curve.loc[(curve["income"] >= cutoff) & (curve["empirical_ccdf"] > 0)].copy()
    if len(body) < 3 or len(tail) < 3:
        return None

    body_x = body["income"].to_numpy(float)
    body_y = body["gompertz_transform"].to_numpy(float)
    gompertz_A, gompertz_B, gompertz_fitted_transform = _linear_fit(body_x, body_y)
    if not np.isfinite(gompertz_B) or gompertz_B >= 0:
        return None

    sorted_tail = sorted_values[split:]
    pareto_alpha = estimate_pareto_mle(sorted_tail, cutoff)
    if not np.isfinite(pareto_alpha) or pareto_alpha <= 1:
        return None
    pareto_slope = 1 - pareto_alpha
    pareto_intercept = np.log(tail_fraction) - pareto_slope * np.log(cutoff)

    body_log_empirical = body["log_empirical_ccdf"].to_numpy(float)
    body_log_fitted = _gompertz_log_ccdf(gompertz_fitted_transform)
    tail_log_income = tail["log_income"].to_numpy(float)
    tail_log_empirical = tail["log_empirical_ccdf"].to_numpy(float)
    tail_log_fitted = pareto_intercept + pareto_slope * tail_log_income

    gompertz_r2, gompertz_rmse = _fit_metrics(body_y, gompertz_fitted_transform)
    pareto_r2, pareto_rmse = _fit_metrics(tail_log_empirical, tail_log_fitted)
    gompertz_residual = body_log_empirical - body_log_fitted
    pareto_residual = tail_log_empirical - tail_log_fitted
    joint_residual = np.concatenate([gompertz_residual, pareto_residual])
    gompertz_aic, gompertz_bic = _information_criteria(gompertz_residual, 2)
    pareto_aic, pareto_bic = _information_criteria(pareto_residual, 1)
    joint_aic, joint_bic = _information_criteria(joint_residual, 3)
    if not np.isfinite([joint_aic, joint_bic]).all():
        return None

    return {
        "cutoff": float(cutoff),
        "cutoff_quantile": float(n_body / n_total),
        "n_total": int(n_total),
        "n_body": int(n_body),
        "n_tail": int(n_tail),
        "tail_fraction": float(tail_fraction),
        "gompertz_A": gompertz_A,
        "gompertz_B": gompertz_B,
        "gompertz_r2": gompertz_r2,
        "gompertz_rmse": gompertz_rmse,
        "gompertz_aic": gompertz_aic,
        "gompertz_bic": gompertz_bic,
        "pareto_alpha_mle": pareto_alpha,
        "pareto_ccdf_slope": pareto_slope,
        "pareto_intercept": float(pareto_intercept),
        "pareto_r2": pareto_r2,
        "pareto_rmse": pareto_rmse,
        "pareto_aic": pareto_aic,
        "pareto_bic": pareto_bic,
        "joint_aic": joint_aic,
        "joint_bic": joint_bic,
    }


def fit_year_distribution_regime(
    values,
    year: int,
    *,
    value_measure: str = "income",
    config: RegimeFitConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit one annual Gompertz-Pareto breakpoint by profile search."""
    settings = config or RegimeFitConfig()
    x = _positive_finite(values)
    minimum = settings.min_body_observations + settings.min_tail_observations
    if x.size < minimum:
        return (
            _empty_summary(
                year,
                x.size,
                value_measure,
                settings.selection_criterion,
                "insufficient_positive_observations",
            ),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )

    curve = _empirical_curve(x, settings.ccdf_base)
    thresholds = curve["income"].to_numpy(float)
    split_indices = np.searchsorted(x, thresholds, side="left")
    n_tail = x.size - split_indices
    quantiles = split_indices / x.size
    eligible = (
        (split_indices >= settings.min_body_observations)
        & (n_tail >= settings.min_tail_observations)
        & (n_tail / x.size >= settings.min_tail_fraction)
        & (quantiles >= settings.cutoff_quantile_min)
        & (quantiles <= settings.cutoff_quantile_max)
    )
    candidates = thresholds[eligible]
    if candidates.size == 0:
        return (
            _empty_summary(
                year,
                x.size,
                value_measure,
                settings.selection_criterion,
                "no_admissible_cutoff",
            ),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )

    profiles = [candidate for cutoff in candidates if (candidate := _profile_candidate(x, curve, cutoff))]
    if not profiles:
        return (
            _empty_summary(
                year,
                x.size,
                value_measure,
                settings.selection_criterion,
                "no_valid_profile_fit",
            ),
            pd.DataFrame(columns=CURVE_COLUMNS),
        )

    criterion = f"joint_{settings.selection_criterion}"
    best = min(profiles, key=lambda row: (row[criterion], row["cutoff"]))
    split = int(np.searchsorted(x, best["cutoff"], side="left"))
    best["pareto_ks"] = _pareto_ks(x[split:], best["cutoff"], best["pareto_alpha_mle"])
    profiled_cutoffs = np.array([row["cutoff"] for row in profiles], dtype=float)
    if len(profiles) == 1:
        fit_status = "ok_single_candidate"
    elif np.isclose(best["cutoff"], profiled_cutoffs.min()):
        fit_status = "ok_boundary_lower"
    elif np.isclose(best["cutoff"], profiled_cutoffs.max()):
        fit_status = "ok_boundary_upper"
    else:
        fit_status = "ok"
    best.update(
        {
            "year": int(year),
            "value_measure": value_measure,
            "candidate_count": len(profiles),
            "selection_criterion": settings.selection_criterion,
            "fit_status": fit_status,
        }
    )

    cutoff = float(best["cutoff"])
    output_curve = curve.copy()
    output_curve.insert(0, "year", int(year))
    output_curve["regime"] = np.where(output_curve["income"] < cutoff, "gompertz_body", "pareto_tail")
    output_curve["cutoff"] = cutoff

    body = output_curve["regime"] == "gompertz_body"
    tail = ~body
    fitted_transform = best["gompertz_A"] + best["gompertz_B"] * output_curve["income"]
    output_curve["gompertz_fitted_transform"] = np.where(body, fitted_transform, np.nan)
    output_curve["gompertz_fitted_ccdf"] = np.where(
        body,
        np.exp(_gompertz_log_ccdf(fitted_transform)),
        np.nan,
    )
    pareto_log = best["pareto_intercept"] + best["pareto_ccdf_slope"] * output_curve["log_income"]
    output_curve["pareto_fitted_log_ccdf"] = np.where(tail, pareto_log, np.nan)
    output_curve["pareto_fitted_ccdf"] = np.where(tail, np.exp(pareto_log), np.nan)
    output_curve = output_curve[CURVE_COLUMNS]
    return {column: best.get(column, np.nan) for column in SUMMARY_COLUMNS}, output_curve


def fit_distribution_regimes(
    df: pd.DataFrame,
    *,
    value_col: str = "income",
    year_col: str = "year",
    config: RegimeFitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one annual fit row and the selected fitted curves for every year."""
    missing = {year_col, value_col}.difference(df.columns)
    if missing:
        raise KeyError("Regime analysis is missing: " + ", ".join(sorted(missing)))
    fits: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    for year, group in df.groupby(year_col, sort=True):
        fit, curve = fit_year_distribution_regime(
            pd.to_numeric(group[value_col], errors="coerce"),
            int(year),
            value_measure=value_col,
            config=config,
        )
        fits.append(fit)
        if not curve.empty:
            curves.append(curve)
    fit_frame = pd.DataFrame(fits, columns=SUMMARY_COLUMNS).sort_values("year").reset_index(drop=True)
    curve_frame = (
        pd.concat(curves, ignore_index=True).sort_values(["year", "income"]).reset_index(drop=True)
        if curves
        else pd.DataFrame(columns=CURVE_COLUMNS)
    )
    return fit_frame, curve_frame
