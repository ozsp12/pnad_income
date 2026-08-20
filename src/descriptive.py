"""Exploratory descriptive statistics and data-quality diagnostics for PNAD income data."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import DEFAULT_METADATA_PATH, load_metadata


class DescriptiveStatistics:
    """Generate annual EDA tables and figures without altering the analytical sample.

    Extreme values are treated as diagnostic evidence rather than automatic errors. Known
    survey sentinels come from metadata; additional candidates are flagged from repeated
    upper-tail values and robust distributional thresholds.
    """

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
        if not self.metadata_path.exists():
            return pd.DataFrame(columns=["year", "missing_income_code"])
        return load_metadata(self.metadata_path)[["year", "missing_income_code"]]

    def annual_summary(self) -> pd.DataFrame:
        """Return robust annual descriptive statistics and upper-tail reference levels."""
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

    def sentinel_candidates(self, min_repetitions: int = 2) -> pd.DataFrame:
        """Flag known sentinels and repeated upper-tail values as candidates for inspection."""
        summary = self.annual_summary().set_index("year")
        metadata_frame = self._metadata()
        metadata = metadata_frame.set_index("year") if not metadata_frame.empty else pd.DataFrame()
        rows: list[dict[str, object]] = []

        for year, group in self.frame.groupby(self.year_col, sort=True):
            year = int(year)
            x = self._series(group)
            if x.empty:
                continue
            p99 = float(summary.loc[year, "p99"])
            counts = x.value_counts()
            sentinel = np.nan
            if not metadata.empty and year in metadata.index:
                sentinel = pd.to_numeric(
                    pd.Series([metadata.loc[year, "missing_income_code"]]), errors="coerce"
                ).iloc[0]

            candidates = counts[(counts >= int(min_repetitions)) & (counts.index >= p99)]
            candidate_values = {float(value) for value in candidates.index}
            if pd.notna(sentinel) and bool((x == float(sentinel)).any()):
                candidate_values.add(float(sentinel))

            for value in sorted(candidate_values):
                count = int(counts.get(value, 0))
                is_metadata = bool(pd.notna(sentinel) and np.isclose(value, float(sentinel)))
                rows.append(
                    {
                        "year": year,
                        "value": float(value),
                        "count": count,
                        "frequency": count / len(x),
                        "p99": p99,
                        "is_metadata_sentinel": is_metadata,
                        "is_repeated_upper_tail": bool(count >= int(min_repetitions) and value >= p99),
                        "reason": "metadata sentinel present" if is_metadata else "repeated value in upper 1% tail",
                    }
                )

        columns = [
            "year",
            "value",
            "count",
            "frequency",
            "p99",
            "is_metadata_sentinel",
            "is_repeated_upper_tail",
            "reason",
        ]
        return pd.DataFrame(rows, columns=columns).sort_values(["year", "value"]).reset_index(drop=True)

    def outlier_diagnostics(self) -> pd.DataFrame:
        """Summarize robust thresholds and a non-destructive candidate cutoff for each year."""
        summary = self.annual_summary()
        metadata = self._metadata()
        candidates = self.sentinel_candidates()
        if not metadata.empty:
            summary = summary.merge(metadata, on="year", how="left", validate="one_to_one")
        else:
            summary["missing_income_code"] = np.nan

        rows: list[dict[str, object]] = []
        for _, row in summary.iterrows():
            year = int(row["year"])
            year_candidates = candidates.loc[candidates["year"] == year] if not candidates.empty else candidates
            metadata_present = year_candidates.loc[year_candidates["is_metadata_sentinel"]]
            repeated = year_candidates.loc[~year_candidates["is_metadata_sentinel"]]
            if not metadata_present.empty:
                cutoff = float(metadata_present["value"].min())
                reason = "metadata sentinel observed"
            elif not repeated.empty:
                cutoff = float(repeated["value"].min())
                reason = "repeated upper-tail value; inspect before exclusion"
            else:
                cutoff = np.nan
                reason = "no automatic cutoff suggested"
            rows.append(
                {
                    **row.to_dict(),
                    "metadata_sentinel_count": int(metadata_present["count"].sum()) if not metadata_present.empty else 0,
                    "repeated_upper_tail_candidates": int(len(repeated)),
                    "suggested_cutoff": cutoff,
                    "cutoff_reason": reason,
                }
            )
        return pd.DataFrame(rows)

    def histogram_pages(self, bins: int = 100, max_panels: int = 24, ncols: int = 4) -> list:
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
                ax.hist(values, bins=bins, log=True)
                ax.set(title=f"PNAD {year}", xlabel="Income", ylabel="Frequency (log scale)")
                ax.grid(axis="y", alpha=0.5, linestyle="--")
            for ax in flat[len(page) :]:
                ax.set_visible(False)
            fig.suptitle(f"Annual income histograms — {page[0]}–{page[-1]}", y=1.002)
            fig.tight_layout()
            pages.append(fig)
        return pages

    def boxplot_pages(self, max_panels: int = 24, ncols: int = 4) -> list:
        """Create annual boxplots on a logarithmic income axis to expose upper-tail anomalies."""
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
                ax.boxplot(values.to_numpy(), vert=True, showfliers=True)
                ax.set_yscale("log")
                ax.set(title=f"PNAD {year}", ylabel="Income (log scale)")
                ax.set_xticks([])
                ax.grid(axis="y", alpha=0.3)
            for ax in flat[len(page) :]:
                ax.set_visible(False)
            fig.suptitle(f"Annual income boxplots — {page[0]}–{page[-1]}", y=1.002)
            fig.tight_layout()
            pages.append(fig)
        return pages

    def outlier_overview_figure(self):
        """Plot annual p99, p99.9 and maxima to reveal discontinuities in the upper tail."""
        summary = self.annual_summary()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for column, label in (("p99", "p99"), ("p99_9", "p99.9"), ("maximum", "maximum")):
            ax.plot(summary["year"], summary[column], marker="o", markersize=3, label=label)
        ax.set_yscale("log")
        ax.set(xlabel="Year", ylabel="Income (log scale)", title="Upper-tail diagnostic levels by survey year")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig
