from __future__ import annotations

from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "pnad_income"


def replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def move_sources() -> None:
    if not PACKAGE.exists():
        return
    for name in ("analysis.py", "data.py", "outputs.py", "pipeline.py", "plotting.py"):
        shutil.move(str(PACKAGE / name), str(SRC / name))
    shutil.move(str(PACKAGE / "__main__.py"), str(SRC / "cli.py"))
    shutil.rmtree(PACKAGE)

    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"from \.([a-zA-Z_][a-zA-Z0-9_]*) import", r"from \1 import", text)
        path.write_text(text, encoding="utf-8")

    replace_text(
        SRC / "data.py",
        "REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]",
        "REPOSITORY_ROOT = PACKAGE_ROOT.parent",
    )


def patch_plotting() -> None:
    path = SRC / "plotting.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '        return x[mask], 100.0 * probability[mask]\n    if transform == "loglog":\n        mask = np.isfinite(x) & np.isfinite(probability) & (x > 0) & (probability > 0)\n        return x[mask], 100.0 * probability[mask]\n',
        '        return x[mask], probability[mask]\n    if transform == "loglog":\n        mask = np.isfinite(x) & np.isfinite(probability) & (x > 0) & (probability > 0)\n        return x[mask], probability[mask]\n',
    )
    text = text.replace(
        '        return x[mask], np.log(-np.log(probability[mask]))',
        '        return x[mask], -np.log(-np.log(probability[mask]))',
    )
    text = text.replace(
        '    return "ln[-ln(S(x))]" if transform in {"gompertz", "double_log"} else "CCDF [%]"',
        '    return "-ln[-ln(S(x))]" if transform in {"gompertz", "double_log"} else "S(x)"',
    )
    text = text.replace(
        'def _ccdf_probability(frame) -> np.ndarray:\n',
        '# Keep the empirical survival function on the probability scale throughout plotting.\ndef _ccdf_probability(frame) -> np.ndarray:\n',
    )
    path.write_text(text, encoding="utf-8")


def patch_outputs() -> None:
    path = SRC / "outputs.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'from analysis import annual_inequality_indices, compare_gini_series, gini_validation_statistics\n',
        'from analysis import annual_inequality_indices, compare_gini_series, gini_validation_statistics\nfrom descriptive import DescriptiveStatistics\n',
    )
    old = '''@dataclass(frozen=True)\nclass OutputPaths:\n    root: Path\n    figures: Path\n    tables: Path\n\n\ndef prepare_output_paths(output_root: str | Path) -> OutputPaths:\n    root = Path(output_root).expanduser().resolve()\n    paths = OutputPaths(root, root / "figures", root / "tables")\n    for path in (paths.root, paths.figures, paths.tables):\n        path.mkdir(parents=True, exist_ok=True)\n    return paths\n'''
    new = '''@dataclass(frozen=True)\nclass OutputPaths:\n    """Canonical output tree separating exploratory diagnostics from paper results."""\n\n    root: Path\n    figures: Path\n    tables: Path\n    figures_eda: Path\n    figures_paper: Path\n    eda_histograms: Path\n    eda_boxplots: Path\n    eda_outliers: Path\n    paper_ccdf: Path\n    paper_lorenz: Path\n    paper_inequality: Path\n    tables_eda: Path\n    tables_paper: Path\n\n\ndef prepare_output_paths(output_root: str | Path) -> OutputPaths:\n    root = Path(output_root).expanduser().resolve()\n    figures = root / "figures"\n    tables = root / "tables"\n    figures_eda = figures / "eda"\n    figures_paper = figures / "paper"\n    paths = OutputPaths(\n        root=root,\n        figures=figures,\n        tables=tables,\n        figures_eda=figures_eda,\n        figures_paper=figures_paper,\n        eda_histograms=figures_eda / "histograms",\n        eda_boxplots=figures_eda / "boxplots",\n        eda_outliers=figures_eda / "outliers",\n        paper_ccdf=figures_paper / "ccdf",\n        paper_lorenz=figures_paper / "lorenz",\n        paper_inequality=figures_paper / "inequality",\n        tables_eda=tables / "eda",\n        tables_paper=tables / "paper",\n    )\n    for directory in paths.__dict__.values():\n        Path(directory).mkdir(parents=True, exist_ok=True)\n    return paths\n'''
    if old not in text:
        raise RuntimeError("OutputPaths block not found")
    text = text.replace(old, new)

    text = text.replace(
        '    indices = annual_inequality_indices(results.panel)\n\n    tables = {',
        '    indices = annual_inequality_indices(results.panel)\n    eda = DescriptiveStatistics(results.panel)\n\n    tables = {',
    )
    text = text.replace(
        '        "data_quality_diagnostics.csv": build_diagnostics(results),\n    }',
        '        "data_quality_diagnostics.csv": build_diagnostics(results),\n        "descriptive_statistics.csv": eda.annual_summary(),\n        "value_frequencies.csv": eda.value_frequencies(),\n        "sentinel_candidates.csv": eda.sentinel_candidates(),\n        "outlier_diagnostics.csv": eda.outlier_diagnostics(),\n    }',
    )
    text = text.replace(
        '    saved_tables = [save_table(frame, paths.tables / name) for name, frame in tables.items()]\n    manifest.extend(_rows(saved_tables, "table"))',
        '''    eda_table_names = {\n        "data_quality_diagnostics.csv",\n        "descriptive_statistics.csv",\n        "value_frequencies.csv",\n        "sentinel_candidates.csv",\n        "outlier_diagnostics.csv",\n    }\n    saved_tables = []\n    for name, frame in tables.items():\n        directory = paths.tables_eda if name in eda_table_names else paths.tables_paper\n        saved_tables.append(save_table(frame, directory / name))\n    manifest.extend(_rows(saved_tables, "table"))''',
    )
    text = text.replace(
        '        save_figure(figure, paths.figures / name, dpi=dpi)\n',
        '        save_figure(figure, paths.paper_inequality / name, dpi=dpi)\n',
    )
    text = text.replace(
        '            plot_histogram_grid(\n                results.panel,\n                years=years,\n                bins=histogram_bins,\n                yscale="log",\n                nrows=complete_nrows,\n                ncols=complete_ncols,\n            ),',
        '            eda.histogram_pages(bins=histogram_bins, max_panels=complete_nrows * complete_ncols, ncols=complete_ncols),',
    )
    text = text.replace(
        '    for stem, figures in page_specs:\n        manifest.extend(_rows(_save_pages(figures, paths.figures, stem, dpi), "figure"))',
        '''    for stem, figures in page_specs:\n        if stem.startswith("histogram"):\n            directory = paths.eda_histograms\n        elif stem.startswith("ccdf"):\n            directory = paths.paper_ccdf\n        else:\n            directory = paths.paper_lorenz\n        manifest.extend(_rows(_save_pages(figures, directory, stem, dpi), "figure"))\n\n    manifest.extend(_rows(_save_pages(eda.boxplot_pages(max_panels=complete_nrows * complete_ncols, ncols=complete_ncols), paths.eda_boxplots, "boxplot_income", dpi), "figure"))\n    manifest.extend(_rows([save_figure(eda.outlier_overview_figure(), paths.eda_outliers / "upper_tail_diagnostics_all_years.png", dpi=dpi)], "figure"))''',
    )
    text = text.replace(
        '        paths_saved = [save_figure(fig, paths.figures / name, dpi=dpi) for name, fig in individual.items()]',
        '''        paths_saved = []\n        for name, fig in individual.items():\n            if name.startswith("histogram"):\n                directory = paths.eda_histograms\n            elif name.startswith("ccdf"):\n                directory = paths.paper_ccdf\n            else:\n                directory = paths.paper_lorenz\n            paths_saved.append(save_figure(fig, directory / name, dpi=dpi))''',
    )
    text = text.replace(
        '        for stem, figures in selected_specs:\n            manifest.extend(_rows(_save_pages(figures, paths.figures, stem, dpi), "figure"))',
        '''        for stem, figures in selected_specs:\n            if stem.startswith("selected_histogram"):\n                directory = paths.eda_histograms\n            elif stem.startswith("selected_ccdf"):\n                directory = paths.paper_ccdf\n            else:\n                directory = paths.paper_lorenz\n            manifest.extend(_rows(_save_pages(figures, directory, stem, dpi), "figure"))''',
    )
    path.write_text(text, encoding="utf-8")


def patch_cli() -> None:
    path = SRC / "cli.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("from .outputs", "from outputs").replace("from .pipeline", "from pipeline")
    path.write_text(text, encoding="utf-8")


def write_pyproject() -> None:
    content = '''[build-system]\nrequires = ["setuptools>=69", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "pnad-income"\nversion = "0.2.0"\ndescription = "Reproducible analysis of Brazilian PNAD income distributions, 1976-2025"\nrequires-python = ">=3.10"\ndependencies = [\n    "numpy>=1.26",\n    "pandas>=2.1",\n    "matplotlib>=3.8",\n    "pyarrow>=14",\n    "openpyxl>=3.1",\n]\n\n[project.optional-dependencies]\ndev = ["pytest>=8.0"]\n\n[project.scripts]\npnad-income = "cli:main"\n\n[tool.setuptools]\npackage-dir = {"" = "src"}\npy-modules = ["analysis", "cli", "data", "descriptive", "outputs", "pipeline", "plotting"]\n\n[tool.pytest.ini_options]\npythonpath = ["src"]\ntestpaths = ["tests"]\n'''
    (ROOT / "pyproject.toml").write_text(content, encoding="utf-8")


def patch_tests() -> None:
    for path in (ROOT / "tests").glob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("from pnad_income.", "from ")
        text = text.replace("import pnad_income.", "import ")
        path.write_text(text, encoding="utf-8")

    plotting_test = ROOT / "tests" / "test_plotting.py"
    text = plotting_test.read_text(encoding="utf-8")
    text = text.replace(
        '    assert "ln[-ln" in fig.axes[0].get_ylabel()\n',
        '    assert fig.axes[0].get_ylabel() == "-ln[-ln(S(x))]"\n    assert np.all(np.diff(y) < 0)\n',
    )
    insert = '''\n\ndef test_loglog_ccdf_remains_on_probability_scale():\n    ccdf = pd.DataFrame(\n        {\n            "year": [2025, 2025],\n            "measure": ["income", "income"],\n            "bin": [1.0, 2.0],\n            "ccdf": [1.0, 0.01],\n        }\n    )\n    fig = plot_ccdf(ccdf, year=2025, measure="income", transform="loglog")\n    y = fig.axes[0].lines[0].get_ydata()\n    assert np.allclose(y, [1.0, 0.01])\n    assert fig.axes[0].get_ylabel() == "S(x)"\n    plt.close(fig)\n'''
    text += insert
    plotting_test.write_text(text, encoding="utf-8")

    outputs_test = ROOT / "tests" / "test_outputs.py"
    text = outputs_test.read_text(encoding="utf-8")
    text = text.replace(
        '    assert paths.tables.is_dir()\n    assert not hasattr(paths, "reports")',
        '    assert paths.tables.is_dir()\n    assert paths.figures_eda.is_dir()\n    assert paths.figures_paper.is_dir()\n    assert paths.eda_histograms.is_dir()\n    assert paths.paper_ccdf.is_dir()\n    assert paths.tables_eda.is_dir()\n    assert paths.tables_paper.is_dir()\n    assert not hasattr(paths, "reports")',
    )
    outputs_test.write_text(text, encoding="utf-8")

    descriptive_test = '''"""Tests for exploratory descriptive statistics and sentinel diagnostics."""\n\nimport matplotlib\nmatplotlib.use("Agg")\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nfrom descriptive import DescriptiveStatistics\n\n\ndef _frame():\n    return pd.DataFrame(\n        {\n            "year": [2020] * 8 + [2021] * 6,\n            "income": [1, 2, 3, 4, 5, 999999, 999999, 999999, 2, 3, 4, 5, 6, 7],\n        }\n    )\n\n\ndef test_value_frequency_detects_repeated_extreme_value(tmp_path):\n    metadata = tmp_path / "metadata.csv"\n    pd.DataFrame(\n        {\n            "year": [2020, 2021],\n            "missing_income_code": [999999, None],\n            "available": [True, True],\n            "divide_by_household_size": [False, False],\n        }\n    ).to_csv(metadata, index=False)\n    eda = DescriptiveStatistics(_frame(), metadata_path=metadata)\n    frequencies = eda.value_frequencies(top_n=10)\n    row = frequencies.loc[(frequencies["year"] == 2020) & (frequencies["value"] == 999999)].iloc[0]\n    assert int(row["count"]) == 3\n\n\ndef test_candidate_cutoff_is_diagnostic_not_filtering(tmp_path):\n    metadata = tmp_path / "metadata.csv"\n    pd.DataFrame(\n        {\n            "year": [2020, 2021],\n            "missing_income_code": [999999, None],\n            "available": [True, True],\n            "divide_by_household_size": [False, False],\n        }\n    ).to_csv(metadata, index=False)\n    original = _frame()\n    eda = DescriptiveStatistics(original, metadata_path=metadata)\n    diagnostics = eda.outlier_diagnostics()\n    row = diagnostics.loc[diagnostics["year"] == 2020].iloc[0]\n    assert row["suggested_cutoff"] == 999999\n    assert (original["income"] == 999999).sum() == 3\n\n\ndef test_eda_figures_render():\n    eda = DescriptiveStatistics(_frame())\n    hist = eda.histogram_pages(max_panels=4, ncols=2)\n    box = eda.boxplot_pages(max_panels=4, ncols=2)\n    overview = eda.outlier_overview_figure()\n    assert hist and box and len(overview.axes) == 1\n    [plt.close(fig) for fig in hist + box + [overview]]\n'''
    (ROOT / "tests" / "test_descriptive.py").write_text(descriptive_test, encoding="utf-8")


def patch_workflow() -> None:
    path = ROOT / ".github" / "workflows" / "run-pipeline-and-persist-outputs.yml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '          python -m pnad_income \\\n',
        '          pnad-income \\\n',
    )
    text = text.replace(
        "              Path('outputs/tables/pipeline_overview.csv'),\n              Path('outputs/tables/annual_summary.csv'),\n              Path('outputs/tables/annual_inequality_indices.csv'),\n              Path('outputs/tables/ccdf_nominal_adjusted.parquet'),\n              Path('outputs/tables/data_quality_diagnostics.csv'),\n              Path('outputs/figures/gini_income_all_years.png'),\n              Path('outputs/figures/inequality_indices_all_years.png'),\n              Path('outputs/figures/zanardi_index_all_years.png'),\n              Path('outputs/figures/histogram_income_log_frequency_page_01.png'),\n              Path('outputs/figures/ccdf_income_loglog_page_01.png'),\n              Path('outputs/figures/ccdf_income_gompertz_page_01.png'),",
        "              Path('outputs/tables/paper/pipeline_overview.csv'),\n              Path('outputs/tables/paper/annual_summary.csv'),\n              Path('outputs/tables/paper/annual_inequality_indices.csv'),\n              Path('outputs/tables/paper/ccdf_nominal_adjusted.parquet'),\n              Path('outputs/tables/eda/data_quality_diagnostics.csv'),\n              Path('outputs/tables/eda/descriptive_statistics.csv'),\n              Path('outputs/tables/eda/value_frequencies.csv'),\n              Path('outputs/tables/eda/sentinel_candidates.csv'),\n              Path('outputs/tables/eda/outlier_diagnostics.csv'),\n              Path('outputs/figures/paper/inequality/gini_income_all_years.png'),\n              Path('outputs/figures/paper/inequality/inequality_indices_all_years.png'),\n              Path('outputs/figures/paper/inequality/zanardi_index_all_years.png'),\n              Path('outputs/figures/eda/histograms/histogram_income_log_frequency_page_01.png'),\n              Path('outputs/figures/eda/boxplots/boxplot_income_page_01.png'),\n              Path('outputs/figures/eda/outliers/upper_tail_diagnostics_all_years.png'),\n              Path('outputs/figures/paper/ccdf/ccdf_income_loglog_page_01.png'),\n              Path('outputs/figures/paper/ccdf/ccdf_income_gompertz_page_01.png'),",
    )
    text = text.replace(
        "          figures = list(Path('outputs/figures').glob('*.png'))",
        "          figures = list(Path('outputs/figures').rglob('*.png'))",
    )
    text = text.replace(
        "              *Path('outputs/figures').glob('*_linear*.png'),\n              *Path('outputs/figures').glob('*double_log_legacy*.png'),",
        "              *Path('outputs/figures').rglob('*_linear*.png'),\n              *Path('outputs/figures').rglob('*double_log_legacy*.png'),",
    )
    path.write_text(text, encoding="utf-8")


def add_comments() -> None:
    analysis = SRC / "analysis.py"
    text = analysis.read_text(encoding="utf-8")
    text = text.replace(
        '    sorted_x = np.sort(x)\n    ccdf = (total - np.searchsorted(sorted_x, left, side="left")) / total',
        '    # Searchsorted reproduces the empirical count #(X >= x) exactly without repeated full-array scans.\n    sorted_x = np.sort(x)\n    ccdf = (total - np.searchsorted(sorted_x, left, side="left")) / total',
    )
    analysis.write_text(text, encoding="utf-8")

    data = SRC / "data.py"
    text = data.read_text(encoding="utf-8")
    text = text.replace(
        '    if missing_code is not None:\n        df = df.loc[df["income_raw"] != missing_code].copy()',
        '    # Survey-specific sentinel codes are metadata, not legitimate income observations.\n    if missing_code is not None:\n        df = df.loc[df["income_raw"] != missing_code].copy()',
    )
    text = text.replace(
        '            out[f"{column}_adj"] = pd.to_numeric(out[column], errors="coerce") / exchange * inflation',
        '            # Within each year this is a positive scale transformation, preserving income ranks.\n            out[f"{column}_adj"] = pd.to_numeric(out[column], errors="coerce") / exchange * inflation',
    )
    data.write_text(text, encoding="utf-8")


def main() -> None:
    move_sources()
    patch_plotting()
    patch_outputs()
    patch_cli()
    write_pyproject()
    patch_tests()
    patch_workflow()
    add_comments()

    # The temporary migration machinery removes itself before the workflow commits the refactor.
    (ROOT / "scripts" / "refactor_loop_engineering.py").unlink(missing_ok=True)
    (ROOT / ".github" / "workflows" / "refactor-loop-engineering.yml").unlink(missing_ok=True)
    try:
        (ROOT / "scripts").rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
