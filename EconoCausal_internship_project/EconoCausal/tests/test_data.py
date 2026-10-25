import pandas as pd
import pytest

from src.data import generate_synthetic_data


def test_generator_returns_expected_columns_and_rows():
    data = generate_synthetic_data(n_samples=250, random_state=7)
    assert len(data) == 250
    assert {"price", "units_sold", "revenue", "seasonality",
            "marketing_spend", "competitor_price", "baseline_demand"} <= set(data.columns)


def test_generator_is_reproducible():
    first = generate_synthetic_data(n_samples=150, random_state=12)
    second = generate_synthetic_data(n_samples=150, random_state=12)
    pd.testing.assert_frame_equal(first, second)


def test_generator_rejects_too_few_rows():
    with pytest.raises(ValueError):
        generate_synthetic_data(n_samples=20)
