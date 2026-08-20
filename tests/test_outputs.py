"""Tests for deterministic persistence of scientific analysis products."""

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from outputs import prepare_output_paths, save_figure, save_table
from plotting import plot_histogram


def test_prepare_output_paths_creates_standard_tree(tmp_path):
    paths = prepare_output_paths(tmp_path / "outputs")
    assert paths.root.is_dir()
    assert paths.figures.is_dir()
    assert paths.tables.is_dir()
    assert paths.figures_eda.is_dir()
    assert paths.figures_paper.is_dir()
    assert paths.eda_histograms.is_dir()
    assert paths.paper_ccdf.is_dir()
    assert paths.tables_eda.is_dir()
    assert paths.tables_paper.is_dir()
    assert not hasattr(paths, "reports")


def test_save_table_writes_under_outputs(tmp_path):
    paths = prepare_output_paths(tmp_path / "outputs")
    frame = pd.DataFrame({"year": [2020, 2021], "value": [1.0, 2.0]})
    saved = save_table(frame, paths.tables / "example.csv")
    assert saved.exists()
    restored = pd.read_csv(saved)
    assert restored.equals(frame)


def test_save_figure_writes_under_outputs(tmp_path):
    paths = prepare_output_paths(tmp_path / "outputs")
    panel = pd.DataFrame({"year": [2025, 2025], "income": [1.0, 2.0]})
    fig = plot_histogram(panel, year=2025)
    saved = save_figure(fig, paths.figures / "histogram_2025.png", close=True)
    assert saved.exists()
    assert saved.stat().st_size > 0
