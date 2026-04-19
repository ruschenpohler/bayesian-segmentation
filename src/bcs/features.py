import pandas as pd
import numpy as np


def build_customer_panel(
    df: pd.DataFrame,
    churn_window_days: int = 90,
    segment_col: str = "Country",
    min_segment_size: int = 15,
    drop_other: bool = True,
) -> tuple[pd.DataFrame, int, int]:
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
    drop_other : if True, drop the "Other" segment entirely (recommended)
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
        segment=(segment_col, "first"),
    ).reset_index()

    # Churn flag
    panel["churned"] = (panel["last_purchase"] < cutoff).astype(int)

    # CLV: total revenue (simple: undiscounted sum as baseline)
    panel["clv"] = panel["total_revenue"]

    # Recency (days since last purchase)
    panel["recency_days"] = (max_date - panel["last_purchase"]).dt.days

    # Tenure (days between first and last purchase)
    panel["tenure_days"] = (panel["last_purchase"] - panel["first_purchase"]).dt.days

    # Frequency (invoices per tenure month, floored at 1 day)
    panel["tenure_months"] = (panel["tenure_days"] / 30).clip(lower=1 / 30)
    panel["purchase_freq"] = panel["n_invoices"] / panel["tenure_months"]

    # Segment encoding: collapse rare segments into "Other"
    seg_counts = panel["segment"].value_counts()
    rare = seg_counts[seg_counts < min_segment_size].index
    n_other = panel["segment"].isin(rare).sum()
    panel["segment"] = panel["segment"].where(~panel["segment"].isin(rare), "Other")

    # Drop "Other" segment entirely.
    # The "Other" group aggregates genuinely different countries (Channel Islands,
    # Norway, Austria, Denmark, Cyprus, Japan, USA, etc.) into a single pseudo-segment.
    # This is methodologically problematic for hierarchical modelling — these countries
    # have nothing in common except being small, so the "Other" estimate sits near the
    # global mean anyway, making partial pooling look redundant. Dropping 138 customers
    # (2.3% of identified customers) gives a cleaner interpretation.
    if drop_other:
        panel = panel[panel["segment"] != "Other"].copy()

    # Integer encode segments
    seg_labels = panel["segment"].astype("category")
    panel["segment_idx"] = seg_labels.cat.codes
    panel["segment_name"] = seg_labels
    n_segments = panel["segment_idx"].nunique()

    return panel, n_segments, n_other


def log1p_scale(series: pd.Series) -> pd.Series:
    """Log1p transform then standardise. Robust to zeros."""
    transformed = np.log1p(series)
    return (transformed - transformed.mean()) / transformed.std()
