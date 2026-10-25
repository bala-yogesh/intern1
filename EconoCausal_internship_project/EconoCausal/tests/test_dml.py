import pytest

from src.data import generate_synthetic_data
from src.dml import estimate_price_effect


def test_dml_returns_expected_result_fields():
    data = generate_synthetic_data(n_samples=600, random_state=4)
    result = estimate_price_effect(data, n_splits=3)
    assert "effect_per_price_unit" in result
    assert "standard_error_approx" in result
    assert result["n_samples"] == 600
    assert result["n_splits"] == 3
    assert result["standard_error_approx"] >= 0


def test_dml_rejects_missing_columns():
    data = generate_synthetic_data(n_samples=200).drop(columns=["price"])
    with pytest.raises(ValueError, match="Missing required columns"):
        estimate_price_effect(data)


def test_dml_rejects_invalid_split_count():
    data = generate_synthetic_data(n_samples=200)
    with pytest.raises(ValueError, match="n_splits"):
        estimate_price_effect(data, n_splits=1)
