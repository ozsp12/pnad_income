# Analytical Data Records

The harmonized analytical database is stored as annual Parquet records under `dados_refined/`. There are 45 files covering the available survey years from 1976 through 2025.

## File convention

```text
dados_refined/
├── pnad_refined_1976.parquet
├── pnad_refined_1977.parquet
├── ...
└── pnad_refined_2025.parquet
```

No file is present for 1980, 1991, 1994, 2000, or 2010.

## Stored schema

Every current Parquet file contains exactly two fields:

| Field | Type | Scientific meaning |
|---|---|---|
| `ano` | integer | Survey year represented by the record. |
| `renda` | numeric | Refined household per-capita income measure used for the longitudinal distribution analysis. |

The value of `ano` is checked against the four-digit year encoded in the filename. A disagreement is treated as a data-integrity error.

The stored Portuguese names are part of the data record. The analytical implementation maps `ano` to `year` and `renda` to `income` only in memory. This normalization is deliberately separated from the Parquet files so that analysis does not silently modify the published records.

## Monetary variables

Exchange factors, historical currency labels, price-index values, and inflation-to-2025 factors are not duplicated in every observation-level Parquet file. They are year-level metadata stored in `metadata/pnad_metadata.csv` and joined to the panel during analysis. The adjusted variable generated in memory is therefore a derived analytical quantity rather than an additional field in the released annual record.

## Additional income measures

The current 45-file release contains only the longitudinal `renda` measure. The 2016–2025 records do not presently include a separate effective-income series such as `VD4020`. Code support for an additional measure does not imply that such a field is part of this data release.

## Missing values and zero income

Survey-specific sentinel values are handled during construction of the refined records. Valid zero-income observations are retained. In the empirical CCDF they remain in the normalization population even though geometric evaluation thresholds begin on the strictly positive support.
