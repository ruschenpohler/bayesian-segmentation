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
