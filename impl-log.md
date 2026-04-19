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
- Will execute feasibility checks shortly to confirm go/no-go
