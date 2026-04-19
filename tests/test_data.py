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
