# Data Layout

Layer 1 uses this directory layout locally and mirrors it in Google Drive for
Colab runs.

```text
data/
  raw/          original source datasets
  processed/    cleaned tweet-level and window-level intermediate CSVs
  scored/       FinBERT-scored baseline tweet datasets
  mutated/      scored mutation datasets grouped by mutation family
  reports/      EDA summaries and attack target window tables
  deprecated/   legacy prototype datasets kept for reproducibility
```

Generated CSV and parquet files under `processed/`, `scored/`, `mutated/`, and
`reports/` are ignored by git. Keep large Colab outputs in Google Drive and
commit only code, notebooks, and small documentation.
