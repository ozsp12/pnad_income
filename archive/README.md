# Source archive inventory

The refactor was reconstructed from two source bundles supplied in the research workflow. The original files were inspected in full, but large notebook outputs and generated binary artifacts are not dependencies of the new pipeline and are therefore not duplicated into the active implementation.

## Beatriz source bundle

- `Projeto_renda.ipynb`
- `Projeto_renda-Copy1.ipynb`
- `codigo_beatriz_arrumado_v01.ipynb`
- `grade_comparativaloglog_1976-1999.png`
- `gradecomparativa_duploln_1976-1999png.png`
- `gradecomparativa_duploln_2001_2025.png`
- Jupyter checkpoint (redundant editor state; excluded)

## Osvaldo refactor bundle

- `00_cria_metadata.ipynb`
- `01_gera_datasets.ipynb`
- `02_validacao_codigos_beatriz.ipynb`
- `03_analise_codigos_melhorados.ipynb`
- `df_stats_all.xlsx`

The active code under `src/` and `notebooks/` supersedes these working files. `docs/migration_notes.md` records the analyses restored and bugs not propagated into the new implementation.
