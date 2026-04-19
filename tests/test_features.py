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
