"""Simple descriptive metrics for the pricing demo."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_data(data: pd.DataFrame) -> dict[str, float]:
    """Return basic descriptive statistics for price, demand, and revenue."""
    required = {"price", "units_sold", "revenue"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return {
        "average_price": float(data["price"].mean()),
        "average_units_sold": float(data["units_sold"].mean()),
        "average_revenue": float(data["revenue"].mean()),
        "price_demand_correlation": float(data["price"].corr(data["units_sold"])),
    }


def candidate_price_table(data: pd.DataFrame, estimated_effect: float, prices=None) -> pd.DataFrame:
    """Illustrate demand/revenue across candidate prices using a linear effect.

    This is an illustrative scenario calculation, not a validated optimizer.
    """
    if prices is None:
        prices = np.linspace(data["price"].quantile(0.1), data["price"].quantile(0.9), 9)
    reference_price = float(data["price"].mean())
    reference_demand = float(data["units_sold"].mean())
    predicted_demand = np.maximum(
        0, reference_demand + estimated_effect * (np.asarray(prices) - reference_price)
    )
    return pd.DataFrame({
        "candidate_price": np.asarray(prices, dtype=float),
        "illustrative_demand": predicted_demand,
        "illustrative_revenue": np.asarray(prices, dtype=float) * predicted_demand,
    })
