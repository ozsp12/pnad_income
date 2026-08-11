# Data

Raw IBGE microdata are intentionally not versioned in Git. Place locally available fixed-width files under `data/raw/` using the file names recorded in `config/pnad_metadata.csv`. The processing notebook writes standardized annual Parquet files to `data/processed/`.

The repository contains code and metadata, not redistributed PNAD microdata. Source URLs are retained in the metadata table for provenance.
