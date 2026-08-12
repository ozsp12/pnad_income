# External Gini validation data

This directory is the documented interface for external annual Gini series used to validate the Gini coefficients calculated from the refined PNAD records.

Reference files must be CSV tables containing at least:

- `year`: calendar/survey year;
- `gini`: Gini coefficient, accepted either on `[0,1]` or `[0,100]`;
- `source`: source label, for example `IPEA` or `World Bank`.

Recommended provenance fields are `indicator`, `url`, `access_date`, and `notes`.

The historical `pnad.py` script contained hard-coded IPEA and World Bank arrays without sufficient source metadata to establish their exact indicator definitions and provenance. Those numerical arrays are therefore **not silently copied into the scientific pipeline**. Once documented source tables are added here, `pnad_income.validation.load_gini_reference`, `compare_gini_series`, and the validation plotting functions consume them directly.
