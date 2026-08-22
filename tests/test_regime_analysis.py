"""Deterministic tests for the normalized Gompertz-Pareto analysis."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import plotting as plotting_module
import regime_analysis
from plotting import (
    plot_distribution_cutoff_history,
    plot_distribution_regime_fits,
    plot_gompertz_parameter_history,
    plot_pareto_alpha_history,
    plot_regime_r2_history,
)
from regime_analysis import (
    FIT_STATUSES,
    GOMPERTZ_A_THEORETICAL,
    RegimeFitConfig,
    estimate_pareto_mle,
    fit_distribution_regimes,
    fit_year_distribution_regime,
)


def _piecewise_sample(n=12_000, cutoff=7.0, B=0.4, alpha=2.0):
    """Build exact Gompertz/Pareto quantiles with a roughly 1%–2% tail."""
    survival = (np.arange(n, 0, -1, dtype=float) - 0.5) / n
    tail_fraction = np.exp(np.exp(GOMPERTZ_A_THEORETICAL - B * cutoff)) / 100.0
    values = np.empty(n, dtype=float)
    body = survival >= tail_fraction
    values[body] = (
        GOMPERTZ_A_THEORETICAL - np.log(np.log(100.0 * survival[body]))
    ) / B
    values[~body] = cutoff * np.power(survival[~body] / tail_fraction, -1.0 / alpha)
    return values


def _config(**overrides):
    settings = {
        "ccdf_base": 1.03,
        "min_body_observations": 50,
        "min_tail_observations": 50,
        "min_tail_fraction": 0.005,
        "cutoff_quantile_min": 0.20,
        "cutoff_quantile_max": 0.995,
    }
    settings.update(overrides)
    return RegimeFitConfig(**settings)


def test_pareto_mle_recovers_known_ccdf_exponent_and_density_convention():
    rng = np.random.default_rng(7391)
    alpha = 2.2
    cutoff = 10.0
    values = cutoff * (1 - rng.random(50_000)) ** (-1 / alpha)
    estimate = estimate_pareto_mle(values, cutoff)
    assert abs(estimate - alpha) < 0.04


def test_profile_recovers_small_tail_break_and_correct_parameter_conventions():
    fit, curves = fit_year_distribution_regime(_piecewise_sample(), 2025, config=_config())
    assert fit["fit_status"] == "ok_interior"
    assert 6.0 <= fit["cutoff_normalized"] <= 9.0
    assert 0.30 <= fit["gompertz_B"] <= 0.42
    assert abs(fit["gompertz_A"] - GOMPERTZ_A_THEORETICAL) < 1e-12
    assert abs(fit["pareto_alpha"] - 2.0) < 0.15
    assert np.isclose(fit["pareto_density_exponent"], fit["pareto_alpha"] + 1)
    assert 0.005 <= fit["tail_fraction"] <= 0.05
    assert fit["continuity_error"] < 1e-10
    assert fit["likelihood_type"] == "normalized_piecewise_microdata"
    assert not curves.empty
    transformed = curves.dropna(subset=["gompertz_transform"])
    assert curves["empirical_ccdf_percent"].between(0, 100, inclusive="right").all()
    assert np.allclose(
        transformed["gompertz_transform"],
        np.log(np.log(transformed["empirical_ccdf_percent"])),
    )
    assert np.isclose(fit["cutoff_income_adj"], fit["cutoff_normalized"] * fit["normalization_mean"])


def test_fixed_and_free_A_modes_are_explicit_and_free_mode_is_labeled_approximate():
    values = _piecewise_sample()
    fixed, _ = fit_year_distribution_regime(values, 2025, config=_config())
    free, _ = fit_year_distribution_regime(
        values,
        2025,
        config=_config(gompertz_intercept_mode="free"),
    )
    assert fixed["gompertz_intercept_mode"] == "fixed"
    assert free["gompertz_intercept_mode"] == "free"
    assert abs(free["gompertz_A"] - GOMPERTZ_A_THEORETICAL) < 0.03
    assert free["likelihood_type"] == "free_A_unnormalized_quasi_likelihood"


def test_scale_order_and_invalid_value_invariance():
    clean = _piecewise_sample()
    dirty_scaled = np.concatenate([clean[::-1] * 123.0, [np.nan, np.inf, -5.0, 0.0]])
    first, _ = fit_year_distribution_regime(clean, 2025, config=_config())
    second, _ = fit_year_distribution_regime(dirty_scaled, 2025, config=_config())
    for column in ("cutoff_normalized", "gompertz_A", "gompertz_B", "pareto_alpha"):
        assert np.isclose(first[column], second[column])
    assert np.isclose(second["normalization_mean"], 123 * first["normalization_mean"])
    assert np.isclose(second["cutoff_income_adj"], 123 * first["cutoff_income_adj"])
    assert second["n_total"] == clean.size


def test_failure_boundary_and_flat_profile_statuses_are_not_reported_as_ok():
    failure, curves = fit_year_distribution_regime(
        np.arange(1.0, 61.0),
        2025,
        config=RegimeFitConfig(min_body_observations=40, min_tail_observations=40),
    )
    assert failure["fit_status"] == "no_valid_fit"
    assert failure["failure_reason"] == "insufficient_positive_observations"
    assert curves.empty

    boundary, _ = fit_year_distribution_regime(
        _piecewise_sample(),
        2025,
        config=_config(cutoff_quantile_min=0.20, cutoff_quantile_max=0.90, sensitivity_lower_bounds=(0.20, 0.40, 0.60)),
    )
    assert boundary["fit_status"] in {"boundary_lower", "boundary_upper"}

    flat, _ = fit_year_distribution_regime(
        _piecewise_sample(),
        2025,
        config=_config(flat_profile_loglik_tolerance=np.inf),
    )
    assert flat["fit_status"] == "flat_profile"

    profiles = [
        {"cutoff_normalized": 1.0, "cutoff_quantile": 0.20, "joint_log_likelihood": 0.0},
        {"cutoff_normalized": 2.0, "cutoff_quantile": 0.40, "joint_log_likelihood": 1.0},
        {"cutoff_normalized": 3.0, "cutoff_quantile": 0.60, "joint_log_likelihood": 0.9},
        {"cutoff_normalized": 4.0, "cutoff_quantile": 0.80, "joint_log_likelihood": 0.8},
    ]
    weak = regime_analysis._profile_diagnostics(profiles, profiles[1], _config())
    assert weak["fit_status"] == "weakly_identified"
    assert {
        failure["fit_status"],
        boundary["fit_status"],
        flat["fit_status"],
        weak["fit_status"],
    }.issubset(FIT_STATUSES)


def test_annual_assets_have_one_row_per_year_and_complete_diagnostics():
    sample = _piecewise_sample()
    frame = pd.DataFrame(
        {
            "year": np.repeat([2024, 2025], sample.size),
            "income_adj": np.concatenate([sample, sample * 1.1]),
        }
    )
    fits, curves = fit_distribution_regimes(frame, config=_config())
    assert fits["year"].tolist() == [2024, 2025]
    assert fits["year"].is_unique
    valid = fits.loc[fits["fit_status"] != "no_valid_fit"]
    assert (valid["n_body"] + valid["n_tail"] == valid["n_total"]).all()
    assert np.allclose(valid["body_fraction"] + valid["tail_fraction"], 1)
    assert valid["pareto_alpha"].gt(1).all()
    assert np.allclose(valid["pareto_density_exponent"], valid["pareto_alpha"] + 1)
    assert valid["continuity_error"].lt(1e-10).all()
    assert valid[["sensitivity_cutoff_p20", "sensitivity_cutoff_p40", "sensitivity_cutoff_p60"]].notna().all().all()
    metrics = [
        "gompertz_A",
        "gompertz_B",
        "gompertz_r2",
        "gompertz_rmse",
        "pareto_alpha",
        "pareto_r2",
        "pareto_rmse",
        "pareto_ks",
        "joint_log_likelihood",
        "joint_aic",
        "joint_bic",
    ]
    assert np.isfinite(valid[metrics].to_numpy(float)).all()
    assert set(curves["year"]) == {2024, 2025}


def test_regime_r2_history_has_two_series_and_two_mean_lines():
    fits = pd.DataFrame(
        {
            "year": [2022, 2023, 2024],
            "gompertz_r2": [0.60, 0.75, 0.90],
            "pareto_r2": [0.70, 0.80, 0.90],
            "fit_status": ["ok_interior", "boundary_lower", "weakly_identified"],
        }
    )
    figure = plot_regime_r2_history(fits)
    lines = figure.axes[0].lines
    assert len(lines) == 4
    means = [line for line in lines if line.get_linestyle() == "--"]
    assert len(means) == 2
    assert np.allclose(means[0].get_ydata(), 0.75)
    assert np.allclose(means[1].get_ydata(), 0.80)
    assert figure.axes[0].get_ylim() == (0.0, 1.0)
    plt.close(figure)


def test_persisted_assets_reproduce_all_plots_without_microdata(tmp_path, monkeypatch):
    original_microdata = pd.DataFrame({"year": 2025, "income_adj": _piecewise_sample()})
    fits, curves = fit_distribution_regimes(original_microdata, config=_config())
    fits_path = tmp_path / "paper_distribution_regime_fits.csv"
    curves_path = tmp_path / "paper_distribution_regime_curves.parquet"
    fits.to_csv(fits_path, index=False)
    curves.to_parquet(curves_path, index=False)

    del original_microdata, fits, curves
    monkeypatch.setattr(
        data,
        "load_database",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("microdata access")),
    )
    monkeypatch.setattr(
        regime_analysis,
        "fit_distribution_regimes",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected re-estimation")),
    )
    restored_fits = pd.read_csv(fits_path)
    restored_curves = pd.read_parquet(curves_path)
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected file read")),
    )
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected file read")),
    )
    assert not hasattr(plotting_module, "load_database")
    assert not hasattr(plotting_module, "fit_distribution_regimes")

    figures = plot_distribution_regime_fits(restored_fits, restored_curves)
    figures.extend(
        [
            plot_gompertz_parameter_history(restored_fits),
            plot_pareto_alpha_history(restored_fits),
            plot_distribution_cutoff_history(restored_fits),
            plot_regime_r2_history(restored_fits),
        ]
    )
    for index, figure in enumerate(figures):
        path = tmp_path / f"figure_{index}.png"
        figure.savefig(path)
        assert path.exists() and path.stat().st_size > 0
        plt.close(figure)
