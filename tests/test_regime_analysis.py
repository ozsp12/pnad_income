"""Deterministic tests for normalized Gompertz-Pareto least squares."""

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
    estimate_gompertz_ls,
    estimate_pareto_ls,
    fit_distribution_regimes,
    fit_year_distribution_regime,
)


def _piecewise_sample(n=12_000, cutoff=7.0, B=0.4, alpha=2.0):
    """Build exact Gompertz/Pareto quantiles with a roughly 1%-2% tail."""
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


def _diagnostic_profiles(sse_values):
    return [
        {
            "cutoff_normalized": float(index + 1),
            "cutoff_quantile": 0.20 * (index + 1),
            "joint_sse": float(sse),
        }
        for index, sse in enumerate(sse_values)
    ]


def test_gompertz_fixed_intercept_ls_recovers_known_B():
    x = np.linspace(0.05, 5.0, 300)
    expected = 0.43
    y = GOMPERTZ_A_THEORETICAL - expected * x
    assert np.isclose(estimate_gompertz_ls(x, y), expected, atol=1e-12)


def test_continuity_constrained_pareto_ls_recovers_known_alpha():
    cutoff = 6.5
    G_cutoff = 1.7
    expected = 2.25
    x = cutoff * np.geomspace(1.0, 20.0, 250)
    F = G_cutoff * (x / cutoff) ** (-expected)
    assert np.isclose(
        estimate_pareto_ls(x, F, cutoff, G_cutoff), expected, atol=1e-12
    )


def test_profile_recovers_break_and_enforces_continuity():
    fit, curves = fit_year_distribution_regime(_piecewise_sample(), 2025, config=_config())
    assert fit["fit_status"] in FIT_STATUSES - {"no_valid_fit"}
    assert 5.5 <= fit["cutoff_normalized"] <= 9.5
    assert 0.30 <= fit["gompertz_B"] <= 0.45
    assert np.isclose(fit["gompertz_A"], GOMPERTZ_A_THEORETICAL)
    assert abs(fit["pareto_alpha"] - 2.0) < 0.25
    assert fit["continuity_error"] < 1e-10
    assert fit["selection_criterion"] == "joint_sse"
    assert not curves.empty
    transformed = curves.dropna(subset=["gompertz_transform"])
    assert curves["empirical_ccdf_percent"].between(0, 100, inclusive="right").all()
    assert np.allclose(
        transformed["gompertz_transform"],
        np.log(np.log(transformed["empirical_ccdf_percent"])),
    )
    assert np.isclose(
        fit["cutoff_income_adj"], fit["cutoff_normalized"] * fit["normalization_mean"]
    )

    body = curves.loc[curves["regime"].eq("gompertz_body")]
    body_transformed = body.dropna(subset=["gompertz_transform"])
    tail = curves.loc[curves["regime"].eq("pareto_tail")]
    expected_gompertz_sse = np.square(
        body_transformed["gompertz_transform"]
        - body_transformed["gompertz_fitted_transform"]
    ).sum()
    expected_pareto_sse = np.square(
        tail["log_empirical_ccdf_percent"] - tail["pareto_fitted_log_ccdf"]
    ).sum()
    expected_joint_sse = (
        np.square(
            body["log_empirical_ccdf_percent"]
            - np.log(body["gompertz_fitted_ccdf_percent"])
        ).sum()
        + expected_pareto_sse
    )
    assert np.isclose(fit["gompertz_sse"], expected_gompertz_sse)
    assert np.isclose(fit["pareto_sse"], expected_pareto_sse)
    assert np.isclose(fit["joint_sse"], expected_joint_sse)

    figures = plot_distribution_regime_fits(pd.DataFrame([fit]), curves)
    labels = [line.get_label() for axis in figures[0].axes for line in axis.lines]
    assert "Gompertz LS" in labels
    assert "Pareto LS" in labels
    plt.close(figures[0])


def test_scale_order_and_invalid_value_invariance():
    clean = _piecewise_sample()
    dirty_scaled = np.concatenate([clean[::-1] * 123.0, [np.nan, np.inf, -5.0, 0.0]])
    first, _ = fit_year_distribution_regime(clean, 2025, config=_config())
    second, _ = fit_year_distribution_regime(dirty_scaled, 2025, config=_config())
    for column in (
        "cutoff_normalized",
        "gompertz_A",
        "gompertz_B",
        "pareto_alpha",
        "joint_sse",
    ):
        assert np.isclose(first[column], second[column])
    assert np.isclose(second["normalization_mean"], 123 * first["normalization_mean"])
    assert np.isclose(second["cutoff_income_adj"], 123 * first["cutoff_income_adj"])
    assert second["n_total"] == clean.size


def test_failure_and_all_profile_identification_statuses():
    failure, curves = fit_year_distribution_regime(
        np.arange(1.0, 61.0),
        2025,
        config=RegimeFitConfig(min_body_observations=40, min_tail_observations=40),
    )
    assert failure["fit_status"] == "no_valid_fit"
    assert failure["failure_reason"] == "insufficient_positive_observations"
    assert curves.empty

    lower_profiles = _diagnostic_profiles([1.0, 2.0, 3.0, 4.0])
    lower = regime_analysis._profile_diagnostics(
        lower_profiles, regime_analysis._best_profile(lower_profiles), _config()
    )
    upper_profiles = _diagnostic_profiles([4.0, 3.0, 2.0, 1.0])
    upper = regime_analysis._profile_diagnostics(
        upper_profiles, regime_analysis._best_profile(upper_profiles), _config()
    )
    flat_profiles = _diagnostic_profiles([1.0, 1.0, 1.0, 1.0])
    flat = regime_analysis._profile_diagnostics(
        flat_profiles, regime_analysis._best_profile(flat_profiles), _config()
    )
    weak_profiles = _diagnostic_profiles([1.04, 1.0, 1.04, 1.04])
    weak = regime_analysis._profile_diagnostics(
        weak_profiles, regime_analysis._best_profile(weak_profiles), _config()
    )

    assert lower["fit_status"] == "boundary_lower"
    assert upper["fit_status"] == "boundary_upper"
    assert flat["fit_status"] == "flat_profile"
    assert weak["fit_status"] == "weakly_identified"
    assert {
        failure["fit_status"],
        lower["fit_status"],
        upper["fit_status"],
        flat["fit_status"],
        weak["fit_status"],
    }.issubset(FIT_STATUSES)


def test_annual_assets_have_one_row_per_year_and_complete_ls_diagnostics():
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
    assert valid["gompertz_B"].gt(0).all()
    assert valid["pareto_alpha"].gt(0).all()
    assert valid["continuity_error"].lt(1e-10).all()
    assert valid[["gompertz_sse", "pareto_sse", "joint_sse"]].ge(0).all().all()
    assert valid[
        ["sensitivity_cutoff_p20", "sensitivity_cutoff_p40", "sensitivity_cutoff_p60"]
    ].notna().all().all()
    metrics = [
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
    ]
    assert np.isfinite(valid[metrics].to_numpy(float)).all()
    forbidden = {
        "joint_log_likelihood",
        "likelihood_type",
        "joint_aic",
        "joint_bic",
        "second_best_log_likelihood",
        "log_likelihood_difference",
        "profile_ci_lower",
        "profile_ci_upper",
        "pareto_density_exponent",
        "pareto_ks",
    }
    assert forbidden.isdisjoint(fits.columns)
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


def test_regime_r2_history_does_not_clip_negative_valid_values():
    fits = pd.DataFrame(
        {
            "year": [2024, 2025],
            "gompertz_r2": [0.90, 0.95],
            "pareto_r2": [-0.25, 0.80],
            "fit_status": ["ok_interior", "ok_interior"],
        }
    )
    figure = plot_regime_r2_history(fits)
    assert figure.axes[0].get_ylim()[0] < -0.25
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
