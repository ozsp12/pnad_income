"""Tests for deterministic persistence of scientific analysis products."""

import hashlib

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from outputs import MANIFEST_COLUMNS, _rows, prepare_output_paths, save_figure, save_table
from plotting import plot_histogram


def test_prepare_output_paths_creates_flat_tree(tmp_path):
    paths = prepare_output_paths(tmp_path / "outputs")
    assert paths.root.is_dir()
    assert paths.figures.is_dir()
    assert paths.tables.is_dir()
    assert set(paths.__dict__) == {"root", "figures", "tables"}


def test_save_table_writes_under_outputs(tmp_path):
    paths = prepare_output_paths(tmp_path / "outputs")
    frame = pd.DataFrame({"year": [2020, 2021], "value": [1.0, 2.0]})
    saved = save_table(frame, paths.tables / "eda_example.csv")
    assert saved.exists()
    restored = pd.read_csv(saved)
    assert restored.equals(frame)


def test_save_figure_writes_under_outputs(tmp_path):
    paths = prepare_output_paths(tmp_path / "outputs")
    panel = pd.DataFrame({"year": [2025, 2025], "income": [1.0, 2.0]})
    fig = plot_histogram(panel, year=2025)
    saved = save_figure(fig, paths.figures / "eda_trusted_histogram_2025.png", close=True)
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_manifest_rows_are_portable_auditable_and_described(tmp_path, monkeypatch):
    paths = prepare_output_paths(tmp_path / "outputs")
    frame = pd.DataFrame({"year": [2020, 2021], "gini": [0.51, 0.49]})
    saved = save_table(frame, paths.tables / "paper_annual_inequality_indices.csv")

    monkeypatch.setenv("GITHUB_REPOSITORY", "ozsp12/pnad_income")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("PNAD_MANIFEST_BRANCH", "main")

    rows = _rows(
        [saved],
        "table",
        output_root=paths.root,
        commit_sha="abc123",
        generated_at="2026-08-20T16:30:00Z",
    )
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == MANIFEST_COLUMNS
    assert row["category"] == "table"
    assert row["stage"] == "paper"
    assert row["data_layer"] == "trusted"
    assert row["path"] == "outputs/tables/paper_annual_inequality_indices.csv"
    assert row["size_bytes"] == saved.stat().st_size
    assert row["size_human"]
    assert row["format"] == "csv"
    assert "inequality" in row["description"].lower()
    assert row["url"] == (
        "https://github.com/ozsp12/pnad_income/blob/main/"
        "outputs/tables/paper_annual_inequality_indices.csv"
    )
    assert row["sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert row["commit_sha"] == "abc123"
    assert row["generated_at"] == "2026-08-20T16:30:00Z"


def test_manifest_data_layer_classification(tmp_path):
    paths = prepare_output_paths(tmp_path / "outputs")
    files = [
        save_table(pd.DataFrame({"x": [1]}), paths.tables / "eda_refined_descriptive_statistics.csv"),
        save_table(pd.DataFrame({"x": [1]}), paths.tables / "eda_trusted_descriptive_statistics.csv"),
        save_table(pd.DataFrame({"x": [1]}), paths.tables / "eda_cleaning_audit.csv"),
        save_table(pd.DataFrame({"x": [1]}), paths.tables / "paper_gini_external_comparison.csv"),
    ]
    rows = _rows(
        files,
        "table",
        output_root=paths.root,
        commit_sha="deadbeef",
        generated_at="2026-08-20T16:30:00Z",
    )
    assert [row["data_layer"] for row in rows] == [
        "refined",
        "trusted",
        "refined_to_trusted",
        "trusted+external",
    ]
    assert all(row["description"] for row in rows)
