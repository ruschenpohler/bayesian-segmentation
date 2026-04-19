import json

cells = []

# Cell 1: Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Phase 1 — Feasibility Checks\n",
        "\n",
        "**Purpose:** Determine whether the dataset supports the intended hierarchical model.\n",
        "All four checks must pass or have confirmed fallbacks before proceeding."
    ]
})

# Cell 2: Load data
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "\n",
        "df1 = pd.read_excel(\"../data/online_retail_II.xlsx\", sheet_name=\"Year 2009-2010\")\n",
        "df2 = pd.read_excel(\"../data/online_retail_II.xlsx\", sheet_name=\"Year 2010-2011\")\n",
        "df = pd.concat([df1, df2], ignore_index=True)\n",
        "\n",
        "print(df.shape)\n",
        "print(df.columns.tolist())\n",
        "print(df.dtypes)\n",
        "print(df.head())"
    ]
})

# Cell 3: CHECK 1 markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## CHECK 1 — Country segment distribution\n",
        "\n",
        "**This is the most important feasibility check.**\n",
        "\n",
        "- **Go (ideal):** UK dominates but at least 8–10 countries have 50–500 unique customers.\n",
        "- **Go (acceptable):** Fewer than 8 countries in middle tier — restrict to top 15.\n",
        "- **No-go:** Fewer than 5 countries with >30 customers outside UK. Fallback: switch to product division grouping."
    ]
})

# Cell 4: CHECK 1 code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "seg = (\n",
        "    df.groupby(\"Country\")[\"Customer ID\"]\n",
        "    .nunique()\n",
        "    .sort_values(ascending=False)\n",
        "    .reset_index()\n",
        "    .rename(columns={\"Customer ID\": \"n_customers\"})\n",
        ")\n",
        "print(seg.to_string())\n",
        "print(f\"\\nTotal countries: {len(seg)}\")\n",
        "print(f\"Countries with >50 customers: {(seg.n_customers > 50).sum()}\")\n",
        "print(f\"Countries with 10-500 customers: {seg.n_customers.between(10,500).sum()}\")"
    ]
})

# Cell 5: CHECK 2 markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## CHECK 2 — Customer ID missingness\n",
        "\n",
        "- **Go:** Overall missingness <35%. Drop rows with missing Customer ID.\n",
        "- **Marginal (35–50%):** Proceed but flag in README. Check if missingness correlates with country/time.\n",
        "- **No-go (>50%):** Switch to invoice-level repeat purchase rate.\n",
        "\n",
        "**Research note:** Missing Customer ID likely reflects guest checkouts (MNAR). Do not impute."
    ]
})

# Cell 6: CHECK 2 code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "overall_missing = df[\"Customer ID\"].isna().mean()\n",
        "print(f\"Overall Customer ID missing: {overall_missing:.1%}\")\n",
        "\n",
        "missing_by_country = (\n",
        "    df.groupby(\"Country\")[\"Customer ID\"]\n",
        "    .apply(lambda x: x.isna().mean())\n",
        "    .sort_values(ascending=False)\n",
        ")\n",
        "print(missing_by_country)"
    ]
})

# Cell 7: CHECK 3 markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## CHECK 3 — Churn base rate and class balance\n",
        "\n",
        "- **Acceptable range:** 20–70% churn rate.\n",
        "- If outside this range, adjust the churn window (try 60 or 120 days) and re-check."
    ]
})

# Cell 8: CHECK 3 code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "df[\"InvoiceDate\"] = pd.to_datetime(df[\"InvoiceDate\"])\n",
        "max_date = df[\"InvoiceDate\"].max()\n",
        "cutoff = max_date - pd.Timedelta(days=90)\n",
        "\n",
        "customer_last = (\n",
        "    df.dropna(subset=[\"Customer ID\"])\n",
        "    .groupby(\"Customer ID\")[\"InvoiceDate\"]\n",
        "    .max()\n",
        "    .reset_index()\n",
        "    .rename(columns={\"InvoiceDate\": \"last_purchase\"})\n",
        ")\n",
        "customer_last[\"churned\"] = (customer_last[\"last_purchase\"] < cutoff).astype(int)\n",
        "\n",
        "print(f\"Churn base rate: {customer_last.churned.mean():.1%}\")\n",
        "print(f\"Total identified customers: {len(customer_last)}\")"
    ]
})

# Cell 9: CHECK 4 markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## CHECK 4 — PyMC sampling speed\n",
        "\n",
        "- **<120s:** Full model tractable with NUTS.\n",
        "- **120–300s:** Tractable but slow. Use top 10 countries only.\n",
        "- **>300s:** Switch to ADVI (variational inference)."
    ]
})

# Cell 10: CHECK 4 code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pymc as pm\n",
        "import numpy as np\n",
        "import time\n",
        "\n",
        "sample_df = customer_last.sample(5000, random_state=42)\n",
        "outcome = sample_df[\"churned\"].values\n",
        "idx = (sample_df.index % 2).values\n",
        "\n",
        "with pm.Model() as timing_model:\n",
        "    mu = pm.Normal(\"mu\", 0, 1)\n",
        "    tau = pm.HalfNormal(\"tau\", 1)\n",
        "    beta = pm.Normal(\"beta\", mu, tau, shape=2)\n",
        "    p = pm.math.sigmoid(beta[idx])\n",
        "    y = pm.Bernoulli(\"y\", p=p, observed=outcome)\n",
        "    \n",
        "    t0 = time.time()\n",
        "    trace = pm.sample(200, tune=200, cores=1, progressbar=True, random_seed=42)\n",
        "    t1 = time.time()\n",
        "\n",
        "print(f\"Elapsed: {t1-t0:.0f}s for 200 samples + 200 tune, 5k obs, 2 segments\")"
    ]
})

# Cell 11: Summary
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Phase 1 Exit Criteria Summary\n",
        "\n",
        "Fill in after running all checks:\n",
        "\n",
        "1. **Grouping variable:** Country / Product division\n",
        "2. **Missingness rate:** __% — Handling: drop / flag / switch to invoice-level\n",
        "3. **Churn window:** __ days — Base rate: __%\n",
        "4. **Sampling strategy:** NUTS / ADVI"
    ]
})

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

with open("/home/rueschenpoehler/bayesian-segmentation/notebooks/01_eda_and_checks.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook created.")
