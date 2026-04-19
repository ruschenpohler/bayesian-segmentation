# Implementation Log

## 2026-04-19: Scaffold

### What changed
Initialised uv project with Python 3.11, created repo structure per plan
(data/, notebooks/, src/bcs/, tests/, scripts/, configs/), added .gitignore,
pyproject.toml with all dependencies, configs/project_standards.yaml,
pushed initial commit to GitHub.

### Why
Establishes clean baseline before any data or code is added.

### Open decisions
None

## 2026-04-19: Data-cleaning

### What changed
Implemented `src/bcs/data.py` with `load_raw`, `drop_missing_customers`,
`drop_cancellations`, `drop_negative_quantity`, `drop_negative_price`.
Added `tests/test_data.py` with cancellation and quantity tests.
Both tests pass.

### Why
Data cleaning functions kept in `src/bcs/` makes them testable and importable from notebooks.

### Open decisions
None

## 2026-04-19: Feasibility checks

### What changed
Created `notebooks/01_eda_and_checks.ipynb` with all four feasibility checks:
1. Country segment distribution
2. Customer ID missingness
3. Churn base rate and class balance
4. PyMC sampling speed (timing model)

Also added `scripts/download_data.py` for reproducible data acquisition,
installed Jupyter kernel `bayesian-segmentation`, and configured pytest
`pythonpath = ["src"]` in pyproject.toml.

### Why
Phase 1 is a strict gate: We need to confirm dataset supports the intended model
before any feature engineering or modelling begins.

### Open decisions
None

## 2026-04-19: Feasibility results

### What changed
Executed all four feasibility checks. All passed:
1. Country distribution: 17 countries in 10-500 range. Restricting to top 12
   segments (>=15 customers) retains 97.7% of identified customers.
2. Customer ID missingness: 22.8% overall, not concentrated in key segments.
   Will drop rows with missing Customer ID.
3. Churn base rate: 50.9% with 90-day window, 5,942 identified customers.
4. PyMC sampling: 35s for timing model. NUTS is tractable.

Decision: grouping variable is Country, dropping countries with <15 customers.

### Why
Documents the go/no-go outcomes so future readers understand the design choices.

### Open decisions
None
## 2026-04-19: Feature engineering

### What changed
Implemented `src/bcs/features.py` with `build_customer_panel` and `log1p_scale`.
Added `tests/test_features.py` (2 tests, both pass).
Created `notebooks/02_feature_engineering.ipynb` to construct the customer panel
with churn flags, CLV, and RFM features. Panel will be saved as parquet.

### Why
Feature engineering transforms raw transactions into customer-level observations
ready for modelling. Keeping logic in `src/bcs/` keeps it testable.

### Open decisions
None
