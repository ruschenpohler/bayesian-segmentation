# Implementation Plan: `bayesian-segmentation`

**Project:** Bayesian hierarchical customer segmentation with partial pooling  
**Dataset:** UCI Online Retail II  
**Author:** Julius Rüschenpöhler  
**Target audience:** DKB (lending/CLV), Hilti (B2B retention)  
**Environment:** WSL2, Python via uv, Jupyter notebooks  
**Agent note:** This document is self-contained. Execute sequentially. Each phase ends with explicit go/no-go criteria. When a no-go is triggered, follow the prescribed fallback before proceeding. Do not proceed to the next phase without confirming the prior phase's exit criteria.

---

## A. Governance and Workflow Rules (Read Before Executing Anything)

These rules apply to every step in this plan. They are not optional.

### A.1 Commit discipline

**Commit after every contiguous, coherent unit of work.** A coherent unit is a single logical task — one file created, one function implemented, one check completed. Do not bundle heterogeneous changes (e.g., data cleaning + feature engineering + a test) into a single commit. If a step in this plan involves multiple files but they form one logical unit (e.g., `data.py` + its test), commit them together. If they are conceptually distinct, commit separately.

**Commit message format:**
```
<scope>: <what was done>

[optional: one-line note on why if non-obvious]
```

Examples:
```
scaffold: initialise uv project and repo structure
data: implement load_raw and drop helpers in data.py
tests: add test_data.py with cancellation and quantity checks
feasibility: complete check 1 — country segment distribution
models: implement partial pooling with non-centred parameterisation
results: add shrinkage plot to notebook 04
```

**Do not wait to be prompted to commit.** After completing each discrete task, stage and commit immediately:
```bash
git add <specific files>   # never git add . blindly
git commit -m "<message>"
git push origin main
```

Always `git add` specific files rather than `.` to avoid accidentally staging data files, notebook checkpoints, or compiled artifacts.

### A.2 impl-log.md

Create `impl-log.md` at the repo root on first commit. It is append-only — never edit past entries. Add an entry after every commit using the schema below.

**Schema:**
```markdown
## YYYY-MM-DD — <component> — <status>

### What changed
<What was implemented or changed. Be concrete: file names, function names, decisions made.>

### Why
<Why this was done this way. Note any alternatives considered and rejected.>

### Open decisions
<Anything unresolved that a future agent or Julius should be aware of.>
```

`<status>` must be one of: `complete`, `in-progress`, `blocked`.

**Example first entry:**
```markdown
## 2026-04-19 — scaffold — complete

### What changed
Initialised uv project with Python 3.11, created repo structure per plan
(data/, notebooks/, src/bcs/, tests/, scripts/), added .gitignore,
pushed initial commit to GitHub.

### Why
Establishes clean baseline before any data or code is added.

### Open decisions
None.
```

**Component names to use consistently:**
`scaffold`, `github`, `data-acquisition`, `feasibility`, `data-cleaning`, `features`, `tests`, `models-no-pool`, `models-full-pool`, `models-partial-pool`, `results`, `readme`

---

## B. GitHub Setup

Do this before any code is written. The repo should exist on GitHub from the first commit.

### B.1 Create remote repo

On GitHub (browser):
1. New repository → name: `bayesian-segmentation`
2. Visibility: Public (this is a portfolio piece)
3. Do NOT initialise with README, .gitignore, or licence — the local repo will push these
4. Copy the SSH remote URL: `git@github.com:<username>/bayesian-segmentation.git`

### B.2 Connect local to remote

```bash
# In WSL2, from ~/bayesian-segmentation
git remote add origin git@github.com:<username>/bayesian-segmentation.git
git branch -M main
```

Confirm SSH key is available in WSL2:
```bash
ssh -T git@github.com
# Expected: "Hi <username>! You've successfully authenticated..."
```

If SSH key is not set up in WSL2 (separate keychain from Windows):
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Paste this into GitHub → Settings → SSH Keys → New SSH Key
```

### B.3 First push

The first push happens at the end of Section 0 (environment + scaffold), after `.gitignore`, `pyproject.toml`, repo structure, `impl-log.md`, and `configs/project_standards.yaml` are in place:

```bash
git add pyproject.toml .gitignore impl-log.md \
        src/bcs/__init__.py src/__init__.py data/.gitkeep
git commit -m "scaffold: initialise project structure and GitHub remote"
git push -u origin main
```

After this, every subsequent commit ends with `git push origin main`.

---

## 0. Environment Setup

### 0.1 WSL2 baseline

```bash
# Confirm WSL2 is Ubuntu 22.04+
lsb_release -a

# Install uv if not present
curl -Lsf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version  # confirm
```

### 0.2 Repo initialisation

```bash
mkdir ~/bayesian-segmentation && cd ~/bayesian-segmentation
git init
uv init --python 3.11
```

### 0.3 Dependencies

Add to `pyproject.toml` under `[project.dependencies]`:

```toml
[project]
name = "bayesian-segmentation"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pymc>=5.10",
    "pytensor>=2.18",
    "numpy>=1.26",
    "pandas>=2.1",
    "openpyxl>=3.1",       # for reading .xlsx
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "scikit-learn>=1.4",
    "jupyter>=1.0",
    "ipykernel>=6.29",
    "pytest>=8.0",
    "arviz>=0.18",         # posterior visualisation, pairs with PyMC
]
```

```bash
uv sync
uv run jupyter notebook  # confirm notebook launches in browser
```

**Known WSL2 issue:** If browser does not open automatically, copy the token URL from terminal output and paste into Windows browser manually. This is cosmetic — not an error.

**Known PyMC issue on first import:** pytensor may compile its C extensions on first `import pymc`. This takes 2–5 minutes and is normal. If it fails with a compiler error, run `sudo apt-get install gcc g++ build-essential` and retry.

### 0.4 Repo structure

```bash
mkdir -p data notebooks src/bcs tests scripts
touch data/.gitkeep
touch src/bcs/__init__.py
touch src/__init__.py
```

Final structure:
```
bayesian-segmentation/
├── README.md
├── pyproject.toml
├── .gitignore
├── impl-log.md
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda_and_checks.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_models.ipynb
│   └── 04_results.ipynb
├── src/
│   └── bcs/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       └── models.py
├── tests/
│   ├── test_data.py
│   └── test_features.py
└── scripts/
    └── download_data.py
```

### 0.5 .gitignore

Data files are excluded here — this is the sole mechanism for keeping large files out of the repo. No pre-commit hook is needed.

```
data/*.xlsx
data/*.csv
data/*.zip
__pycache__/
*.pyc
.venv/
.pytest_cache/
notebooks/.ipynb_checkpoints/
```

---

## 1. Data Acquisition

### 1.1 Download

The dataset is UCI Online Retail II. It is available directly from the UCI ML Repository.

```python
# scripts/download_data.py
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DEST = Path("data")

def download():
    zip_path = DEST / "online_retail_ii.zip"
    print("Downloading...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DEST)
    print("Done. Files:", list(DEST.iterdir()))

if __name__ == "__main__":
    download()
```

```bash
uv run python scripts/download_data.py
ls data/  # expect: online_retail_ii.xlsx or similar
```

**Known issue:** UCI occasionally changes file naming. If the extracted file is not named `online_retail_II.xlsx`, note the actual filename and update all downstream references accordingly.

---

## 2. Phase 1 — Feasibility Checks (Notebook 01)

**Purpose:** This phase is a strict gate. Its sole function is to determine whether the dataset supports the intended model. Do not begin feature engineering until all checks pass or fallbacks are confirmed.

Open `notebooks/01_eda_and_checks.ipynb`.

### 2.1 Load data

```python
import pandas as pd

df = pd.read_excel("../data/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df2 = pd.read_excel("../data/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = pd.concat([df, df2], ignore_index=True)

print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
```

Expected columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.

**If columns differ:** adjust all downstream column references before continuing.

### 2.2 CHECK 1 — Country segment distribution

**This is the most important feasibility check.**

```python
seg = (
    df.groupby("Country")["Customer ID"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"Customer ID": "n_customers"})
)
print(seg.to_string())
print(f"\nTotal countries: {len(seg)}")
print(f"Countries with >50 customers: {(seg.n_customers > 50).sum()}")
print(f"Countries with 10-500 customers: {seg.n_customers.between(10,500).sum()}")
```

**Interpretation and go/no-go:**

- **Go (ideal):** UK dominates (expected, ~90% of transactions) but at least 8–10 countries have 50–500 unique customers. This gives a meaningful middle tier for shrinkage demonstration.
- **Go (acceptable):** Fewer than 8 countries in the middle tier — proceed but restrict analysis to top 15 countries by customer count, dropping very small ones.
- **No-go:** Fewer than 5 countries with >30 customers outside UK. **Fallback:** Switch grouping variable from Country to product division. Construct product divisions by first character of StockCode (letters A–Z become categories). Re-run this check on the product grouping before continuing.

### 2.3 CHECK 2 — Customer ID missingness

```python
overall_missing = df["Customer ID"].isna().mean()
print(f"Overall Customer ID missing: {overall_missing:.1%}")

missing_by_country = (
    df.groupby("Country")["Customer ID"]
    .apply(lambda x: x.isna().mean())
    .sort_values(ascending=False)
)
print(missing_by_country)
```

**Interpretation and go/no-go:**

- **Go:** Overall missingness <35% and not concentrated in the top 10 countries by transaction volume. Proceed by dropping rows with missing Customer ID (document the share dropped).
- **Marginal (35–50% overall):** Proceed but flag in README that analysis is restricted to identified customers. Check whether missingness correlates with country or time — if systematic, note this as a limitation (DGP of missingness is likely MNAR: guest checkouts are a distinct customer type, not missing at random).
- **No-go (>50% or concentrated in key segments):** Switch outcome from customer-level churn to invoice-level repeat purchase rate, which does not require Customer ID tracking. The hierarchical model still applies; the estimand changes.

**Research note on missingness DGP:** Missing Customer ID almost certainly reflects guest checkouts — a structurally different purchase mode, not a random omission. This is Missing Not At Random (MNAR): the probability of missingness is related to the unobserved customer type (one-time buyers are more likely to check out as guests). Dropping these rows introduces selection bias toward repeat customers. Document this explicitly in the notebook as a limitation. Do not attempt to impute Customer ID — it is not recoverable.

### 2.4 CHECK 3 — Churn base rate and class balance

Construct a preliminary churn flag (no purchase in final 90 days of the dataset window) and check base rate.

```python
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
max_date = df["InvoiceDate"].max()
cutoff = max_date - pd.Timedelta(days=90)

customer_last = (
    df.dropna(subset=["Customer ID"])
    .groupby("Customer ID")["InvoiceDate"]
    .max()
    .reset_index()
    .rename(columns={"InvoiceDate": "last_purchase"})
)
customer_last["churned"] = (customer_last["last_purchase"] < cutoff).astype(int)

print(f"Churn base rate: {customer_last.churned.mean():.1%}")
print(f"Total identified customers: {len(customer_last)}")
```

**Interpretation:**

- **Acceptable range:** 20–70% churn rate. Outside this range the classification problem is heavily imbalanced and calibration becomes the dominant concern rather than the hierarchical structure.
- **If base rate <15% or >85%:** Adjust the churn window. Try 60 days or 120 days and re-check. Document the chosen window and justify it. The window choice is a modelling decision with substantive implications — flag it.

### 2.5 CHECK 4 — PyMC sampling speed

Run a minimal timed model before committing to the full pipeline.

```python
import pymc as pm
import numpy as np
import time

# Use first 5000 identified customers
sample_df = customer_last.sample(5000, random_state=42)
outcome = sample_df["churned"].values
# Dummy segment: binary, 2 groups
idx = (sample_df.index % 2).values

with pm.Model() as timing_model:
    mu = pm.Normal("mu", 0, 1)
    tau = pm.HalfNormal("tau", 1)
    beta = pm.Normal("beta", mu, tau, shape=2)
    p = pm.math.sigmoid(beta[idx])
    y = pm.Bernoulli("y", p=p, observed=outcome)
    
    t0 = time.time()
    # IMPORTANT: cores=1 always on WSL2/Windows to avoid multiprocessing issues
    trace = pm.sample(200, tune=200, cores=1, progressbar=True, random_seed=42)
    t1 = time.time()

print(f"Elapsed: {t1-t0:.0f}s for 200 samples + 200 tune, 5k obs, 2 segments")
```

**Interpretation:**

- **<120s:** Full model tractable with NUTS. Proceed as planned.
- **120–300s:** Tractable but slow. Plan for overnight runs or reduce to top 10 countries only (~60–70% of identified customers). Use `pm.sample(1000, tune=1000, cores=1)` for final model.
- **>300s:** Switch to ADVI (variational inference) as primary fitting method. Replace `pm.sample(...)` with `pm.fit(10000, method="advi")` throughout. The point estimates are approximate but sufficient for a showcase. Document the choice.

### 2.6 Phase 1 exit criteria

All four checks must have a confirmed go or confirmed fallback before proceeding. Write a single markdown cell at the bottom of notebook 01 summarising:

1. Grouping variable chosen (Country or product division)
2. Missingness rate and handling decision
3. Churn window chosen and base rate
4. Sampling strategy (NUTS or ADVI)

---

## 3. Phase 2 — Feature Engineering (Notebook 02 + `src/bcs/`)

All logic in this phase should be implemented as functions in `src/bcs/data.py` and `src/bcs/features.py`, then imported into the notebook. This keeps the notebook narrative clean and the logic testable.

### 3.1 `src/bcs/data.py`

```python
import pandas as pd
from pathlib import Path

def load_raw(path: str | Path) -> pd.DataFrame:
    """Load and concatenate both sheets of Online Retail II."""
    path = Path(path)
    df1 = pd.read_excel(path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(path, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

def drop_missing_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Drop rows with missing Customer ID. Returns cleaned df and drop rate."""
    rate = df["Customer ID"].isna().mean()
    return df.dropna(subset=["Customer ID"]).copy(), rate

def drop_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancelled invoices (Invoice starting with C)."""
    return df[~df["Invoice"].astype(str).str.startswith("C")].copy()

def drop_negative_quantity(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Quantity"] > 0].copy()

def drop_negative_price(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Price"] > 0].copy()
```

**Research note:** Cancellations (Invoice starting with "C") are a meaningful signal — high cancellation rate per customer may predict churn. Do not drop them silently before examining their distribution. Examine first, then decide whether to drop or retain as a feature.

### 3.2 `src/bcs/features.py`

```python
import pandas as pd
import numpy as np

def build_customer_panel(
    df: pd.DataFrame,
    churn_window_days: int = 90,
    segment_col: str = "Country",
    min_segment_size: int = 30,
) -> pd.DataFrame:
    """
    Construct customer-level panel with:
    - churn flag
    - CLV estimate (total revenue over observation window)
    - RFM features
    - segment identifier (integer encoded)
    
    Parameters
    ----------
    df : cleaned transaction dataframe (no missing Customer ID, no cancellations)
    churn_window_days : days before max_date defining churn
    segment_col : grouping variable ("Country" or alternative)
    min_segment_size : segments below this threshold are grouped into "Other"
    """
    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=churn_window_days)
    
    df = df.copy()
    df["revenue"] = df["Quantity"] * df["Price"]
    
    panel = df.groupby("Customer ID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        n_invoices=("Invoice", "nunique"),
        n_transactions=("Invoice", "count"),
        total_revenue=("revenue", "sum"),
        segment=(segment_col, "first"),  # take first observed value
    ).reset_index()
    
    # Churn flag
    panel["churned"] = (panel["last_purchase"] < cutoff).astype(int)
    
    # CLV: total discounted revenue (simple: undiscounted sum as baseline)
    panel["clv"] = panel["total_revenue"]
    
    # Recency (days since last purchase)
    panel["recency_days"] = (max_date - panel["last_purchase"]).dt.days
    
    # Tenure (days between first and last purchase)
    panel["tenure_days"] = (panel["last_purchase"] - panel["first_purchase"]).dt.days
    
    # Frequency (invoices per tenure month, floored at 1 day)
    panel["tenure_months"] = (panel["tenure_days"] / 30).clip(lower=1/30)
    panel["purchase_freq"] = panel["n_invoices"] / panel["tenure_months"]
    
    # Segment encoding: collapse rare segments into "Other"
    seg_counts = panel["segment"].value_counts()
    rare = seg_counts[seg_counts < min_segment_size].index
    panel["segment"] = panel["segment"].where(~panel["segment"].isin(rare), "Other")
    
    # Integer encode segments
    seg_labels = panel["segment"].astype("category")
    panel["segment_idx"] = seg_labels.cat.codes
    panel["segment_name"] = seg_labels
    n_segments = panel["segment_idx"].nunique()
    
    return panel, n_segments

def log1p_scale(series: pd.Series) -> pd.Series:
    """Log1p transform then standardise. Robust to zeros."""
    transformed = np.log1p(series)
    return (transformed - transformed.mean()) / transformed.std()
```

**Research notes for agent:**

- **Distribution of CLV:** CLV in retail is almost always heavily right-skewed (log-normal or Pareto-tailed). Check `panel["clv"].describe()` and `panel["clv"].hist(bins=50)`. If skew is extreme (skewness >5), use `log1p_scale` for any CLV-as-covariate usage. For CLV as outcome in a regression model, consider modelling log(CLV) rather than CLV directly.
- **Extent and shape of heterogeneity:** Plot churn rate by segment as a first pass. If all segments have churn rates within 5 percentage points of each other, true heterogeneity is low and partial pooling will add little over full pooling — the model will still run but the shrinkage plot will be unimpressive. In this case, consider using CLV decile as a moderating variable to generate more visible heterogeneity.
- **Segment "Other":** The collapsed rare-segment group is methodologically problematic for hierarchical modelling — it aggregates genuinely different countries into a single pseudo-segment. In the final model, either exclude "Other" from the hierarchical structure (give it its own fixed intercept) or drop it from the analysis and document the restriction.

### 3.3 Tests

```python
# tests/test_data.py
import pandas as pd
import pytest
from src.bcs.data import drop_cancellations, drop_negative_quantity

def make_fake_df():
    return pd.DataFrame({
        "Invoice": ["123", "C456", "789"],
        "Quantity": [2, -1, 0],
        "Price": [1.0, 2.0, 3.0],
        "Customer ID": [1.0, 2.0, 3.0],
    })

def test_drop_cancellations():
    df = make_fake_df()
    result = drop_cancellations(df)
    assert "C456" not in result["Invoice"].values
    assert len(result) == 2

def test_drop_negative_quantity():
    df = make_fake_df()
    result = drop_negative_quantity(df)
    assert (result["Quantity"] > 0).all()
```

```python
# tests/test_features.py
import pandas as pd
import numpy as np
from src.bcs.features import log1p_scale

def test_log1p_scale_zero_mean():
    s = pd.Series([1.0, 10.0, 100.0, 1000.0])
    result = log1p_scale(s)
    assert abs(result.mean()) < 1e-10

def test_log1p_scale_handles_zeros():
    s = pd.Series([0.0, 1.0, 2.0])
    result = log1p_scale(s)
    assert result.isna().sum() == 0
```

```bash
uv run pytest tests/ -v
```

All tests must pass before proceeding to modelling.

---

## 4. Phase 3 — Models (Notebook 03 + `src/bcs/models.py`)

This is the core of the project. Three models are fitted to the same data and compared directly.

### 4.1 Model specifications

#### Model A: No pooling

Each segment gets an independent intercept estimated only from its own data. Small segments have high variance; no information is shared.

$$\text{logit}(p_i) = \alpha_{j[i]}$$
$$\alpha_j \sim \mathcal{N}(0, 10) \quad \text{(weakly informative, independent)}$$

#### Model B: Full pooling

Single global intercept. Segment membership is ignored entirely.

$$\text{logit}(p_i) = \alpha$$
$$\alpha \sim \mathcal{N}(0, 10)$$

#### Model C: Partial pooling (the target model)

Segment intercepts are drawn from a shared population distribution. The hyperparameters $\mu$ and $\tau$ are estimated from data.

$$\text{logit}(p_i) = \alpha_{j[i]}$$
$$\alpha_j \sim \mathcal{N}(\mu, \tau)$$
$$\mu \sim \mathcal{N}(0, 1)$$
$$\tau \sim \text{HalfNormal}(1)$$

The prior on $\tau$ is the key modelling choice: HalfNormal(1) on the log-odds scale is weakly informative — it allows substantial between-segment heterogeneity but regularises toward zero. If the data shows very strong heterogeneity (churn rates ranging from 10% to 90% across segments), widen to HalfNormal(2).

### 4.2 `src/bcs/models.py`

```python
import pymc as pm
import numpy as np
import arviz as az

def fit_no_pooling(outcome: np.ndarray, segment_idx: np.ndarray, n_segments: int):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0, 10, shape=n_segments)
        p = pm.math.sigmoid(alpha[segment_idx])
        y = pm.Bernoulli("y", p=p, observed=outcome)
        # cores=1 required on WSL2/Windows
        trace = pm.sample(1000, tune=1000, cores=1,
                          random_seed=42, progressbar=True)
    return model, trace

def fit_full_pooling(outcome: np.ndarray):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0, 10)
        p = pm.math.sigmoid(alpha)
        y = pm.Bernoulli("y", p=p, observed=outcome)
        trace = pm.sample(1000, tune=1000, cores=1,
                          random_seed=42, progressbar=True)
    return model, trace

def fit_partial_pooling(outcome: np.ndarray, segment_idx: np.ndarray, n_segments: int):
    with pm.Model() as model:
        # Hyperpriors
        mu = pm.Normal("mu", 0, 1)
        tau = pm.HalfNormal("tau", 1)
        # Non-centred parameterisation — critical for sampling efficiency
        alpha_offset = pm.Normal("alpha_offset", 0, 1, shape=n_segments)
        alpha = pm.Deterministic("alpha", mu + tau * alpha_offset)
        p = pm.math.sigmoid(alpha[segment_idx])
        y = pm.Bernoulli("y", p=p, observed=outcome)
        trace = pm.sample(1000, tune=1000, cores=1,
                          random_seed=42, progressbar=True)
    return model, trace
```

**Critical implementation note — non-centred parameterisation:**
The partial pooling model uses `alpha_offset` (standard normal) multiplied by `tau` rather than sampling `alpha` directly from `Normal(mu, tau)`. This is the non-centred parameterisation and is essential for sampling efficiency when `tau` is small (low heterogeneity case). With the centred parameterisation, NUTS develops funnel geometry in the posterior that causes divergences and poor mixing. Always use non-centred for hierarchical models in PyMC.

**Divergence check — run after every model:**
```python
divergences = trace.sample_stats["diverging"].sum().item()
print(f"Divergences: {divergences}")
# Acceptable: 0-5. If >10, the model has geometry problems.
# Fix: increase target_accept to 0.9 in pm.sample(..., target_accept=0.9)
# If still diverging: check prior scale, consider wider tau prior
```

### 4.3 ADVI fallback

If NUTS is too slow (>300s in timing check), replace `pm.sample(...)` with:

```python
with model:
    approx = pm.fit(30000, method="advi", random_seed=42)
    trace = approx.sample(2000)
```

ADVI assumes a mean-field approximation (independent posterior factors) which underestimates posterior correlations. For the showcase purpose this is acceptable — note it in the README.

---

## 5. Phase 4 — Results and Visualisation (Notebook 04)

### 5.1 The shrinkage plot (primary deliverable)

This is the money visualisation. It must show, for each segment:
- No-pooling estimate (with wide CI for small segments)
- Partial-pooling estimate (shrunk toward global mean)
- The global mean (full-pooling estimate)
- Sample size on the x-axis

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit  # sigmoid

fig, ax = plt.subplots(figsize=(10, 6))

# Extract posterior means and 94% HDI for each model
# (arviz.summary returns these)
import arviz as az

# Partial pooling: posterior mean of alpha per segment
pp_summary = az.summary(trace_partial, var_names=["alpha"], hdi_prob=0.94)
pp_means = expit(pp_summary["mean"].values)
pp_lo = expit(pp_summary["hdi_3%"].values)
pp_hi = expit(pp_summary["hdi_97%"].values)

# No pooling: posterior mean of alpha per segment  
np_summary = az.summary(trace_no_pool, var_names=["alpha"], hdi_prob=0.94)
np_means = expit(np_summary["mean"].values)
np_lo = expit(np_summary["hdi_3%"].values)
np_hi = expit(np_summary["hdi_97%"].values)

# Full pooling: single global estimate
fp_summary = az.summary(trace_full, var_names=["alpha"], hdi_prob=0.94)
fp_mean = expit(fp_summary["mean"].values[0])

# Segment sample sizes (x-axis)
seg_sizes = panel.groupby("segment_idx").size().sort_index().values
seg_names = panel.groupby("segment_idx")["segment_name"].first().values

# Plot
ax.hlines(fp_mean, 0, seg_sizes.max(), colors="grey",
          linestyles="dashed", label="Full pooling (global mean)")

for i, (n, name) in enumerate(zip(seg_sizes, seg_names)):
    ax.plot(n, np_means[i], "o", color="steelblue", alpha=0.7)
    ax.plot(n, pp_means[i], "o", color="firebrick", alpha=0.7)
    ax.plot([n, n], [np_lo[i], np_hi[i]], color="steelblue", alpha=0.3, linewidth=1)
    ax.plot([n, n], [pp_lo[i], pp_hi[i]], color="firebrick", alpha=0.3, linewidth=1)
    ax.annotate(name, (n, pp_means[i]), fontsize=7, alpha=0.7)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='steelblue', label='No pooling'),
    Line2D([0], [0], marker='o', color='firebrick', label='Partial pooling'),
    Line2D([0], [0], color='grey', linestyle='dashed', label='Full pooling'),
]
ax.legend(handles=legend_elements)
ax.set_xlabel("Segment sample size (n customers)")
ax.set_ylabel("Estimated churn probability")
ax.set_title("Shrinkage: partial pooling vs. no pooling vs. full pooling\nby customer segment")
plt.tight_layout()
plt.savefig("../figures/shrinkage_plot.png", dpi=150)
```

**What to look for:** The firebrick (partial pooling) points should lie between the steelblue (no pooling) points and the grey dashed line, with the degree of shrinkage increasing as segment size decreases. If partial pooling estimates are identical to no pooling, $\hat{\tau}$ is large (strong heterogeneity, little pooling). If they collapse to the global mean, $\hat{\tau}$ is near zero (little heterogeneity, near full pooling). Both are valid findings — document which occurred and why.

### 5.2 Tau posterior (key research quantity)

```python
az.plot_posterior(trace_partial, var_names=["tau"], hdi_prob=0.94)
```

$\tau$ is the between-segment standard deviation on the log-odds scale. Interpret as follows:
- $\tau < 0.5$: modest heterogeneity — partial pooling reduces to near full pooling
- $\tau \approx 1$: meaningful heterogeneity — segments differ by ~1 SD on log-odds (~18pp difference in churn probability at the centre of the logistic)
- $\tau > 2$: strong heterogeneity — segments are nearly independent, partial pooling adds little over no pooling

Report the posterior median and 94% HDI for $\tau$ in the README.

### 5.3 Model comparison via LOO-CV

```python
loo_partial = az.loo(trace_partial, model_partial)
loo_no_pool = az.loo(trace_no_pool, model_no_pool)
loo_full = az.loo(trace_full, model_full)

az.compare({
    "partial_pooling": trace_partial,
    "no_pooling": trace_no_pool,
    "full_pooling": trace_full,
}, ic="loo")
```

LOO-CV [Leave-One-Out Cross-Validation] via PSIS [Pareto-Smoothed Importance Sampling] is the standard Bayesian model comparison metric. Higher ELPD [Expected Log Predictive Density] is better. Partial pooling should outperform both extremes, especially on small segments. If it does not outperform no pooling, the segments are genuinely independent and the hierarchical structure is not supported by the data — report this honestly.

---

## 6. README

The README is a first-class deliverable. It is what a DS lead reads before opening a single notebook.

Structure:
1. **One-paragraph problem statement** — why partial pooling, why it matters for lending/CLV
2. **The key result** — one sentence summarising $\hat{\tau}$ and what it implies
3. **Shrinkage plot** — embedded image, immediately visible
4. **Method** — three-paragraph non-technical explanation of no/full/partial pooling
5. **Repo structure** — brief
6. **Reproduce** — exact commands from clone to plot

---

## 7. Known Issues and Failure Modes Reference

| Issue | Symptom | Fix |
|---|---|---|
| pytensor compile error | `ImportError` on `import pymc` | `sudo apt-get install gcc g++ build-essential` |
| NUTS hangs | No progress after 5min | Kill, set `cores=1`, retry |
| Many divergences (>20) | Warning in sample output | Add `target_accept=0.9`; if persists, use non-centred parameterisation (already in plan) |
| Segment heterogeneity too low | Shrinkage plot flat | Switch to CLV decile as segment, or add RFM covariate to model |
| Segment heterogeneity too high | Partial pooling = no pooling | Report $\tau$ posterior, still a valid finding |
| CLV extreme skew | Histogram has very long tail | Use log1p_scale; consider Gamma regression instead of Gaussian for CLV model extension |
| Missing Customer ID >50% | Longitudinal panel unreliable | Switch to invoice-level model as described in Check 2 fallback |
| ADVI underestimates uncertainty | CI too narrow vs. NUTS | Note in README, acceptable for showcase |

---

## 8. Extensions (if time allows)

These are not required for the showcase but strengthen the repo:

1. **Add RFM covariates to the partial pooling model** — extend the linear predictor to include recency, frequency, monetary value as customer-level predictors alongside the segment random effect. This is a random-intercept model with fixed slopes.
2. **CLV as outcome** — replace churn (binary) with log(CLV) (continuous) using a Normal likelihood. The hierarchical structure is identical; the interpretation shifts from retention to revenue.
3. **Posterior predictive check** — `pm.sample_posterior_predictive(trace, model)` then compare predicted vs. observed churn rates per segment. This is the Bayesian equivalent of calibration plotting and belongs in the results notebook.

---

*End of implementation plan. Agent should return to Julius at the end of Phase 1 (feasibility checks) with the four check results before proceeding to Phase 2.*
