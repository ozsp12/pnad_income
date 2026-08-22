"""Deterministic tests for the Gompertz-Pareto regime analysis."""

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
)
from regime_analysis import (
    RegimeFitConfig,
    estimate_pareto_mle,
    fit_distribution_regimes,
    fit_year_distribution_regime,
)


def _piecewise_sample(n=4000, cutoff=100.0, alpha=3.0):
    """Build a deterministic Gompertz body and Pareto tail with 80/20 mass."""
    n_tail = n // 5
    n_body = n - n_tail
    survival = np.linspace(0.999, n_tail / n + 1 / n, n_body)
    transform_at_cutoff = -np.log(-np.log(n_tail / n))
    gompertz_B = -0.075
    gompertz_A = transform_at_cutoff - gompertz_B * cutoff
    body = (-np.log(-np.log(survival)) - gompertz_A) / gompertz_B
    probabilities = (np.arange(n_tail, dtype=float) + 0.5) / n_tail
    tail = cutoff * np.power(1 - probabilities, -1 / (alpha - 1))
    return np.concatenate([body, tail])


def _config():
    return RegimeFitConfig(
        ccdf_base=1.02,
        min_body_observations=50,
        min_tail_observations=50,
        min_tail_fraction=0.02,
        cutoff_quantile_min=0.65,
        cutoff_quantile_max=0.95,
    )


def test_pareto_mle_recovers_known_density_exponent():
    rng = np.random.default_rng(7391)
    alpha = 3.2
    cutoff = 10.0
    values = cutoff * (1 - rng.random(50_000)) ** (-1 / (alpha - 1))
    assert abs(estimate_pareto_mle(values, cutoff) - alpha) < 0.06


def test_profile_search_recovers_known_breakpoint_within_tolerance():
    fit, curves = fit_year_distribution_regime(_piecewise_sample(), 2025, config=_config())
    assert fit["fit_status"].startswith("ok")
    assert 0.80 * 100 <= fit["cutoff"] <= 1.25 * 100
    assert abs(fit["pareto_alpha_mle"] - 3.0) < 0.4
    assert not curves.empty


def test_fit_is_invariant_to_record_order_and_ignores_nonpositive_nonfinite_values():
    clean = _piecewise_sample()
    dirty = np.concatenate([clean[::-1], [np.nan, np.inf, -5.0, 0.0]])
    first, _ = fit_year_distribution_regime(clean, 2025, config=_config())
    second, _ = fit_year_distribution_regime(dirty, 2025, config=_config())
    for column in ("cutoff", "gompertz_A", "gompertz_B", "pareto_alpha_mle", "joint_bic"):
        assert np.isclose(first[column], second[column])
    assert second["n_total"] == clean.size


def test_insufficient_tail_fails_with_informative_status():
    config = RegimeFitConfig(min_body_observations=40, min_tail_observations=40)
    fit, curves = fit_year_distribution_regime(np.arange(1.0, 61.0), 2025, config=config)
    assert fit["fit_status"] == "insufficient_positive_observations"
    assert np.isnan(fit["cutoff"])
    assert curves.empty


def test_annual_summary_has_one_consistent_row_per_year():
    sample = _piecewise_sample()
    frame = pd.DataFrame(
        {
            "year": np.repeat([2024, 2025], sample.size),
            "income": np.concatenate([sample, sample * 1.1]),
        }
    )
    fits, curves = fit_distribution_regimes(frame, config=_config())
    assert fits["year"].tolist() == [2024, 2025]
    assert fits["year"].is_unique
    valid = fits.loc[fits["fit_status"].str.startswith("ok")]
    assert (valid["n_body"] + valid["n_tail"] == valid["n_total"]).all()
    assert valid["tail_fraction"].between(0, 1, inclusive="neither").all()
    assert np.allclose(valid["pareto_ccdf_slope"], 1 - valid["pareto_alpha_mle"])
    metric_columns = [
        "gompertz_A",
        "gompertz_B",
        "gompertz_r2",
        "gompertz_rmse",
        "pareto_alpha_mle",
        "pareto_ks",
        "joint_aic",
        "joint_bic",
    ]
    assert np.isfinite(valid[metric_columns].to_numpy(float)).all()
    assert set(curves["year"]) == {2024, 2025}


def test_persisted_assets_reproduce_all_plots_without_microdata(tmp_path, monkeypatch):
    original_microdata = pd.DataFrame({"year": 2025, "income": _piecewise_sample()})
    fits, curves = fit_distribution_regimes(original_microdata, config=_config())
    fits_path = tmp_path / "paper_distribution_regime_fits.csv"
    curves_path = tmp_path / "paper_distribution_regime_curves.parquet"
    fits.to_csv(fits_path, index=False)
    curves.to_parquet(curves_path, index=False)

    del original_microdata, fits, curves
    monkeypatch.setattr(data, "load_database", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("microdata access")))
    monkeypatch.setattr(
        regime_analysis,
        "fit_distribution_regimes",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected re-estimation")),
    )
    restored_fits = pd.read_csv(fits_path)
    restored_curves = pd.read_parquet(curves_path)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected file read")))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected file read")))
    assert not hasattr(plotting_module, "load_database")
    assert not hasattr(plotting_module, "fit_distribution_regimes")

    figures = plot_distribution_regime_fits(restored_fits, restored_curves)
    figures.extend(
        [
            plot_gompertz_parameter_history(restored_fits),
            plot_pareto_alpha_history(restored_fits),
            plot_distribution_cutoff_history(restored_fits),
        ]
    )
    for index, figure in enumerate(figures):
        path = tmp_path / f"figure_{index}.png"
        figure.savefig(path)
        assert path.exists() and path.stat().st_size > 0
        plt.close(figure)
