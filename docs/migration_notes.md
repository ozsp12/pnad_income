# Migration notes

This repository was reconstructed from two working archives: the original notebooks supplied by Beatriz and the subsequent refactor by Osvaldo L. Santos-Pereira. The objective was to preserve analytical intent while removing notebook duplication and execution-state dependence.

## Analyses restored

The refactored project restores four components that were present in Beatriz's notebooks but absent from the latest analytical notebook: `VD4020` for 2016-2025; habitual-versus-effective income comparisons; ordinary/log-log CCDF views; and nominal-versus-adjusted distribution comparisons.

## Corrections made during migration

1. The prior `compute_ccdf` filtered observations below `xmin` before defining the denominator. The new implementation keeps all finite observations, including zero incomes, in the denominator while using positive values only to define geometric thresholds.
2. Monetary constants are no longer duplicated between metadata and analysis code. The metadata values for 2006 (`2.1686`) and 2011 (`1.7598`) follow the values independently present in Beatriz's analytical notebook and Osvaldo's later analytical notebook; they correct `2.1687` and `1.7498` in the earlier metadata notebook.
3. The execution-order error in which `df_stats_all` was exported before being constructed disappears because statistics are now explicit function calls.
4. The validation-notebook typo that computed adjusted `xmin` from the unadjusted income series is not carried forward.
5. The `binslog` versus `bins_log` state-dependent naming inconsistency is eliminated by DataFrame-returning functions.
6. Original notebook errors such as the 2014/2024 bin-name mix-up are retained only in the archive and are not propagated into the package.

## What remains archival

The original notebooks are retained separately for provenance. Jupyter checkpoint files are deliberately excluded because they duplicate notebook state rather than constitute an independent analytical artifact. Historical PNG and spreadsheet outputs are generated artifacts and are not dependencies of the new pipeline.
