"""Pipeline tests using compact synthetic analytical databases."""

from pathlib import Path

import pandas as pd

from pnad_income.pipeline import PipelineConfig, load_database, pipeline_overview, run_pipeline, validate_database


def test_validate_database_requires_income():
    frame = pd.DataFrame({"year": [2020, 2021]})
    try:
        validate_database(frame)
    except ValueError as exc:
        assert "income" in str(exc)
    else:
        raise AssertionError("validate_database should reject a missing income column")


def test_validate_database_normalizes_portuguese_fields():
    frame = pd.DataFrame({"ano": [2020, 2021], "renda": [100.0, 200.0]})
    validated = validate_database(frame)
    assert validated.columns.tolist() == ["year", "income"]
    assert validated["year"].tolist() == [2020, 2021]
    assert validated["income"].tolist() == [100.0, 200.0]


def test_directory_loader_infers_year(tmp_path: Path):
    pd.DataFrame({"income": [0.0, 100.0]}).to_parquet(tmp_path / "pnad_refined_2020.parquet")
    pd.DataFrame({"income": [50.0, 200.0]}).to_parquet(tmp_path / "pnad_refined_2021.parquet")
    panel = load_database(tmp_path)
    assert sorted(panel["year"].unique().tolist()) == [2020, 2021]
    assert len(panel) == 4


def test_directory_loader_reads_published_portuguese_schema(tmp_path: Path):
    pd.DataFrame({"renda": [0.0, 100.0], "ano": [2020, 2020]}).to_parquet(
        tmp_path / "pnad_refined_2020.parquet"
    )
    pd.DataFrame({"renda": [50.0, 200.0], "ano": [2021, 2021]}).to_parquet(
        tmp_path / "pnad_refined_2021.parquet"
    )
    panel = load_database(tmp_path)
    assert panel.columns.tolist() == ["income", "year"] or panel.columns.tolist() == ["year", "income"]
    assert sorted(panel["year"].unique().tolist()) == [2020, 2021]
    assert panel["income"].sum() == 350.0


def test_pipeline_runs_from_annual_parquets(tmp_path: Path):
    pd.DataFrame({
        "income": [0.0, 100.0],
        "income_effective": [0.0, 110.0],
        "exchange": [1.0, 1.0],
        "price_index": [100.0, 100.0],
        "inflation_to_2025": [2.0, 2.0],
    }).to_parquet(tmp_path / "pnad_refined_2020.parquet")
    pd.DataFrame({
        "income": [50.0, 200.0],
        "income_effective": [55.0, 210.0],
        "exchange": [1.0, 1.0],
        "price_index": [100.0, 100.0],
        "inflation_to_2025": [1.5, 1.5],
    }).to_parquet(tmp_path / "pnad_refined_2021.parquet")

    results = run_pipeline(PipelineConfig(database_path=tmp_path, start_year=2020, end_year=2021))
    assert results.years == [2020, 2021]
    assert "income_adj" in results.panel.columns
    assert "income_effective_adj" in results.panel.columns
    assert not results.ccdf_nominal_adjusted.empty
    assert not results.ccdf_habitual_effective.empty

    overview = pipeline_overview(results).set_index("metric")["value"]
    assert int(overview["observations"]) == 4
    assert int(overview["number_of_years"]) == 2
