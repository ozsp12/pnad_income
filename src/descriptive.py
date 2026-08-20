"""Exploratory diagnostics and deterministic trusted-layer construction for PNAD income data."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import DEFAULT_METADATA_PATH, DEFAULT_TRUSTED_PATH


class DescriptiveStatistics:
    """Compute annual EDA products without modifying the supplied analytical sample."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        value_col: str = "income",
        year_col: str = "year",
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        if {year_col, value_col}.difference(frame.columns):
            raise KeyError(f"Required columns '{year_col}' and '{value_col}' are not both present.")
        self.frame = frame.copy()
        self.value_col = value_col
        self.year_col = year_col
        self.metadata_path = Path(metadata_path)

    def _series(self, group: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(group[self.value_col], errors="coerce")
        return values[np.isfinite(values)]

    def _metadata(self) -> pd.DataFrame:
        """Read the metadata fields required for sentinel diagnostics."""
        if not self.metadata_path.exists():
            return pd.DataFrame(columns=["year", "missing_income_code"])
        frame = pd.read_csv(
            self.metadata_path,
            usecols=lambda column: column in {"year", "missing_income_code"},
        )
        if "year" not in frame.columns:
            return pd.DataFrame(columns=["year", "missing_income_code"])
        if "missing_income_code" not in frame.columns:
            frame["missing_income_code"] = np.nan
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame["missing_income_code"] = pd.to_numeric(frame["missing_income_code"], errors="coerce")
        frame = frame.dropna(subset=["year"]).copy()
        frame["year"] = frame["year"].astype(int)
        return frame[["year", "missing_income_code"]].drop_duplicates("year").reset_index(drop=True)

    def annual_summary(self) -> pd.DataFrame:
        """Return annual descriptive statistics and upper-tail reference levels."""
        rows: list[dict[str, float | int]] = []
        for year, group in self.frame.groupby(self.year_col, sort=True):
            x = self._series(group)
            if x.empty:
                continue
            q = x.quantile([0.25, 0.50, 0.75, 0.95, 0.99, 0.999])
            iqr = q.loc[0.75] - q.loc[0.25]
            rows.append(
                {
                    "year": int(year),
                    "n": int(x.size),
                    "n_missing": int(pd.to_numeric(group[self.value_col], errors="coerce").isna().sum()),
                    "n_zero": int((x == 0).sum()),
                    "minimum": float(x.min()),
                    "q25": float(q.loc[0.25]),
                    "median": float(q.loc[0.50]),
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
                    "q75": float(q.loc[0.75]),
                    "p95": float(q.loc[0.95]),
                    "p99": float(q.loc[0.99]),
                    "p99_9": float(q.loc[0.999]),
                    "iqr_upper": float(q.loc[0.75] + 1.5 * iqr),
                    "maximum": float(x.max()),
                }
            )
        return pd.DataFrame(rows)

    def value_frequencies(self, top_n: int = 50) -> pd.DataFrame:
        """Return the most frequent exact income values within each survey year."""
        rows: list[pd.DataFrame] = []
        for year, group in self.frame.groupby(self.year_col, sort=True):
            x = self._series(group)
            counts = x.value_counts(dropna=False).rename_axis("value").reset_index(name="count")
            counts["frequency"] = counts["count"] / max(len(x), 1)
            counts = counts.sort_values(["count", "value"], ascending=[False, False]).head(int(top_n)).copy()
            counts.insert(0, "year", int(year))
            counts.insert(1, "rank", np.arange(1, len(counts) + 1))
            rows.append(counts)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["year", "rank", "value", "count", "frequency"]
        )

    def metadata_sentinel_occurrences(self) -> pd.DataFrame:
        """Count exact metadata-defined sentinel values observed in each annual income series."""
        metadata = self._metadata().set_index("year")
        rows: list[dict[str, object]] = []
        for year, group in self.frame.groupby(self.year_col, sort=True):
            year = int(year)
            x = self._series(group)
            sentinel = np.nan
            if not metadata.empty and year in metadata.index:
                sentinel = pd.to_numeric(
                    pd.Series([metadata.loc[year, "missing_income_code"]]), errors="coerce"
                ).iloc[0]
            count = 0
            if pd.notna(sentinel) and not x.empty:
                count = int(np.isclose(x.to_numpy(float), float(sentinel), rtol=0.0, atol=1e-12).sum())
            rows.append(
                {
                    "year": year,
                    "missing_income_code": float(sentinel) if pd.notna(sentinel) else np.nan,
                    "count": count,
                    "frequency": count / max(len(x), 1),
                }
            )
        return pd.DataFrame(rows)

    def outlier_diagnostics(self) -> pd.DataFrame:
        """Return annual tail diagnostics without imposing an exclusion rule."""
        summary = self.annual_summary()
        sentinel = self.metadata_sentinel_occurrences().rename(
            columns={"count": "metadata_sentinel_count", "frequency": "metadata_sentinel_frequency"}
        )
        if summary.empty:
            return summary
        return summary.merge(sentinel, on="year", how="left", validate="one_to_one")

    def income_limits_by_year(self) -> dict[int, tuple[float, float]]:
        """Return linear income limits that can be shared across before/after histograms."""
        limits: dict[int, tuple[float, float]] = {}
        for year, group in self.frame.groupby(self.year_col, sort=True):
            x = self._series(group)
            if x.empty:
                continue
            upper = float(x.max())
            limits[int(year)] = (0.0, upper if upper > 0 else 1.0)
        return limits

    def positive_limits_by_year(self) -> dict[int, tuple[float, float]]:
        """Return positive limits suitable for shared logarithmic boxplot axes."""
        limits: dict[int, tuple[float, float]] = {}
        for year, group in self.frame.groupby(self.year_col, sort=True):
            x = self._series(group)
            x = x[x > 0]
            if x.empty:
                continue
            lower = max(float(x.min()) * 0.8, np.finfo(float).tiny)
            upper = float(x.max()) * 1.2
            limits[int(year)] = (lower, upper if upper > lower else lower * 10)
        return limits

    def histogram_pages(
        self,
        bins: int = 100,
        max_panels: int = 24,
        ncols: int = 4,
        x_limits_by_year: dict[int, tuple[float, float]] | None = None,
    ) -> list:
        """Create annual histograms with logarithmic frequency and linear income axes."""
        years = sorted(pd.to_numeric(self.frame[self.year_col], errors="coerce").dropna().astype(int).unique())
        pages = []
        for start in range(0, len(years), max_panels):
            page = years[start : start + max_panels]
            rows = ceil(len(page) / ncols)
            fig, axes = plt.subplots(rows, ncols, figsize=(4 * ncols, 3 * rows), squeeze=False)
            flat = axes.ravel()
            for ax, year in zip(flat, page):
                values = self._series(self.frame.loc[self.frame[self.year_col] == year])
                values = values[values > 0]
                limits = (x_limits_by_year or {}).get(year)
                ax.hist(values, bins=bins, range=limits, log=True)
                if limits is not None:
                    ax.set_xlim(*limits)
                ax.set(title=f"PNAD {year}", xlabel="Income", ylabel="Frequency (log scale)")
                ax.grid(axis="y", alpha=0.5, linestyle="--")
            for ax in flat[len(page) :]:
                ax.set_visible(False)
            fig.suptitle(f"Annual income histograms — {page[0]}–{page[-1]}", y=1.002)
            fig.tight_layout()
            pages.append(fig)
        return pages

    def boxplot_pages(
        self,
        max_panels: int = 24,
        ncols: int = 4,
        y_limits_by_year: dict[int, tuple[float, float]] | None = None,
    ) -> list:
        """Create annual boxplots on a logarithmic income axis."""
        years = sorted(pd.to_numeric(self.frame[self.year_col], errors="coerce").dropna().astype(int).unique())
        pages = []
        for start in range(0, len(years), max_panels):
            page = years[start : start + max_panels]
            rows = ceil(len(page) / ncols)
            fig, axes = plt.subplots(rows, ncols, figsize=(4 * ncols, 3 * rows), squeeze=False)
            flat = axes.ravel()
            for ax, year in zip(flat, page):
                values = self._series(self.frame.loc[self.frame[self.year_col] == year])
                values = values[values > 0]
                ax.boxplot(values.to_numpy(), orientation="vertical", showfliers=True)
                ax.set_yscale("log")
                limits = (y_limits_by_year or {}).get(year)
                if limits is not None:
                    ax.set_ylim(*limits)
                ax.set(title=f"PNAD {year}", ylabel="Income (log scale)")
                ax.set_xticks([])
                ax.grid(axis="y", alpha=0.3)
            for ax in flat[len(page) :]:
                ax.set_visible(False)
            fig.suptitle(f"Annual income boxplots — {page[0]}–{page[-1]}", y=1.002)
            fig.tight_layout()
            pages.append(fig)
        return pages

    def outlier_overview_figure(self, *, ylim: tuple[float, float] | None = None):
        """Plot annual p99, p99.9 and maxima to reveal upper-tail discontinuities."""
        summary = self.annual_summary()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for column, label in (("p99", "p99"), ("p99_9", "p99.9"), ("maximum", "maximum")):
            ax.plot(summary["year"], summary[column], marker="o", markersize=3, label=label)
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set(xlabel="Year", ylabel="Income (log scale)", title="Upper-tail diagnostic levels by survey year")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def compare_upper_tail_figure(self, other: "DescriptiveStatistics", *, labels=("refined", "trusted")):
        """Compare high quantiles and maxima between two data layers on common axes."""
        left = self.annual_summary()
        right = other.annual_summary()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for frame, layer in ((left, labels[0]), (right, labels[1])):
            ax.plot(frame["year"], frame["p99_9"], marker="o", markersize=3, label=f"{layer} p99.9")
            ax.plot(frame["year"], frame["maximum"], marker="o", markersize=3, label=f"{layer} maximum")
        ax.set_yscale("log")
        ax.set(xlabel="Year", ylabel="Income (log scale)", title="Upper-tail comparison: refined vs trusted")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig


class IncomeDataCleaner:
    """Build the trusted layer using deterministic, mutually exclusive quality flags.

    Metadata sentinels are identified first and are excluded from threshold estimation.
    Statistical outliers are then evaluated only among non-sentinel observations. The
    default rule uses a robust upper threshold on ``log1p(income)``.
    """

    METHODS = {"log_mad", "mad", "mean_std"}

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        value_col: str = "income",
        year_col: str = "year",
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        method: str = "log_mad",
        threshold: float = 6.0,
    ) -> None:
        if {year_col, value_col}.difference(frame.columns):
            raise KeyError(f"Required columns '{year_col}' and '{value_col}' are not both present.")
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {sorted(self.METHODS)}")
        if not np.isfinite(threshold) or float(threshold) <= 0:
            raise ValueError("threshold must be a finite positive number.")
        self.frame = frame.copy().reset_index(drop=True)
        self.value_col = value_col
        self.year_col = year_col
        self.metadata_path = Path(metadata_path)
        self.method = method
        self.threshold = float(threshold)
        self._flagged_cache: pd.DataFrame | None = None
        self._threshold_cache: pd.DataFrame | None = None

    def _metadata_map(self) -> dict[int, float]:
        metadata = DescriptiveStatistics(
            self.frame,
            value_col=self.value_col,
            year_col=self.year_col,
            metadata_path=self.metadata_path,
        )._metadata()
        if metadata.empty:
            return {}
        return {
            int(row.year): float(row.missing_income_code)
            for row in metadata.itertuples(index=False)
            if pd.notna(row.missing_income_code)
        }

    def _compute_threshold(self, values: np.ndarray) -> tuple[float, float, float, str]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if self.method == "log_mad":
            values = values[values >= 0]
            if values.size == 0:
                return np.nan, np.nan, np.nan, "log1p"
            transformed = np.log1p(values)
            center = float(np.median(transformed))
            dispersion = float(1.4826 * np.median(np.abs(transformed - center)))
            if dispersion <= 0 and transformed.size > 1:
                dispersion = float(np.std(transformed, ddof=1))
            cutoff = float(np.expm1(center + self.threshold * dispersion)) if dispersion > 0 else float(values.max())
            return center, dispersion, cutoff, "log1p"

        if values.size == 0:
            return np.nan, np.nan, np.nan, "income"
        if self.method == "mad":
            center = float(np.median(values))
            dispersion = float(1.4826 * np.median(np.abs(values - center)))
            if dispersion <= 0 and values.size > 1:
                dispersion = float(np.std(values, ddof=1))
        else:
            center = float(np.mean(values))
            dispersion = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        cutoff = float(center + self.threshold * dispersion) if dispersion > 0 else float(values.max())
        return center, dispersion, cutoff, "income"

    def flagged_frame(self) -> pd.DataFrame:
        """Return refined data with mutually exclusive sentinel and outlier flags."""
        if self._flagged_cache is not None:
            return self._flagged_cache.copy()

        out = self.frame.copy()
        years = pd.to_numeric(out[self.year_col], errors="coerce")
        values = pd.to_numeric(out[self.value_col], errors="coerce")
        out["flag_metadata_sentinel"] = np.zeros(len(out), dtype=np.int8)
        out["flag_statistical_outlier"] = np.zeros(len(out), dtype=np.int8)
        sentinel_map = self._metadata_map()
        threshold_rows: list[dict[str, object]] = []

        for year in sorted(years.dropna().astype(int).unique()):
            idx = years.index[years == year]
            year_values = values.loc[idx]
            sentinel = sentinel_map.get(int(year), np.nan)
            sentinel_mask = pd.Series(False, index=idx)
            if pd.notna(sentinel):
                sentinel_mask = pd.Series(
                    np.isclose(year_values.to_numpy(float), float(sentinel), rtol=0.0, atol=1e-12, equal_nan=False),
                    index=idx,
                )
                out.loc[idx[sentinel_mask.to_numpy()], "flag_metadata_sentinel"] = 1

            valid_mask = np.isfinite(year_values.to_numpy(float)) & ~sentinel_mask.to_numpy()
            estimation = year_values.to_numpy(float)[valid_mask]
            center, dispersion, cutoff, scale = self._compute_threshold(estimation)

            outlier_mask = np.zeros(len(idx), dtype=bool)
            if np.isfinite(cutoff):
                outlier_mask = (
                    np.isfinite(year_values.to_numpy(float))
                    & ~sentinel_mask.to_numpy()
                    & (year_values.to_numpy(float) > cutoff)
                )
                out.loc[idx[outlier_mask], "flag_statistical_outlier"] = 1

            n_sentinel = int(sentinel_mask.sum())
            n_outlier = int(outlier_mask.sum())
            threshold_rows.append(
                {
                    "year": int(year),
                    "method": self.method,
                    "parameter": self.threshold,
                    "scale": scale,
                    "missing_income_code": float(sentinel) if pd.notna(sentinel) else np.nan,
                    "n_refined": int(len(idx)),
                    "n_metadata_sentinel": n_sentinel,
                    "n_valid_for_estimation": int(valid_mask.sum()),
                    "center": center,
                    "dispersion": dispersion,
                    "statistical_cutoff": cutoff,
                    "n_statistical_outlier": n_outlier,
                    "n_trusted": int(len(idx) - n_sentinel - n_outlier),
                }
            )

        intersection = (out["flag_metadata_sentinel"] == 1) & (out["flag_statistical_outlier"] == 1)
        if bool(intersection.any()):
            raise AssertionError("Sentinel and statistical-outlier flags must be mutually exclusive.")

        self._flagged_cache = out
        self._threshold_cache = pd.DataFrame(threshold_rows)
        return out.copy()

    def thresholds(self) -> pd.DataFrame:
        """Return the deterministic annual parameters used by the cleaning rule."""
        if self._threshold_cache is None:
            self.flagged_frame()
        return self._threshold_cache.copy()

    def cleaning_audit(self) -> pd.DataFrame:
        """Return annual record counts before and after trusted-layer construction."""
        audit = self.thresholds().copy()
        audit["n_removed"] = audit["n_metadata_sentinel"] + audit["n_statistical_outlier"]
        audit["removal_rate"] = audit["n_removed"] / audit["n_refined"].where(audit["n_refined"] > 0, np.nan)
        return audit[
            [
                "year",
                "n_refined",
                "n_metadata_sentinel",
                "n_statistical_outlier",
                "n_removed",
                "n_trusted",
                "removal_rate",
                "method",
                "parameter",
                "statistical_cutoff",
            ]
        ]

    def trusted_frame(self, *, drop_flags: bool = True) -> pd.DataFrame:
        """Return observations accepted by both quality rules."""
        flagged = self.flagged_frame()
        keep = (flagged["flag_metadata_sentinel"] == 0) & (flagged["flag_statistical_outlier"] == 0)
        trusted = flagged.loc[keep].reset_index(drop=True)
        if drop_flags:
            trusted = trusted.drop(columns=["flag_metadata_sentinel", "flag_statistical_outlier"])
        return trusted

    def materialize_trusted(self, output_root: str | Path = DEFAULT_TRUSTED_PATH) -> list[Path]:
        """Persist one trusted Parquet file per year, replacing stale generated files."""
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        for stale in root.glob("pnad_trusted_*.parquet"):
            stale.unlink()
        trusted = self.trusted_frame(drop_flags=True)
        paths: list[Path] = []
        for year, group in trusted.groupby(self.year_col, sort=True):
            path = root / f"pnad_trusted_{int(year)}.parquet"
            group.to_parquet(path, index=False, compression="snappy")
            paths.append(path)
        return paths
