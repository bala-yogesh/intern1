"""Synthetic retail data generator for the EconoCausal demo."""
from __future__ import annotations

import numpy as np
import pandas as pd


CONTROL_COLUMNS = [
    "seasonality",
    "marketing_spend",
    "competitor_price",
    "baseline_demand",
]


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate reproducible synthetic data with price confounding.

    The true price effect is deliberately set to -2.0 units per price unit.
    This is simulated data for learning only, not a real business dataset.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 for a useful demonstration.")

    rng = np.random.default_rng(random_state)
    seasonality = rng.uniform(0, 1, n_samples)
    marketing_spend = rng.uniform(0, 100, n_samples)
    competitor_price = rng.uniform(15, 45, n_samples)
    baseline_demand = rng.normal(60, 12, n_samples)

    # Price depends partly on observed business conditions, creating confounding.
    price = (
        22
        + 0.12 * competitor_price
        + 0.035 * marketing_spend
        - 2.0 * seasonality
        - 0.035 * baseline_demand
        + rng.normal(0, 2.5, n_samples)
    )
    price = np.clip(price, 8, 60)

    # True simulated causal effect: +1 price unit -> about -2 units sold,
    # conditional on the controls and before random noise.
    units_sold = (
        115
        - 2.0 * price
        + 0.22 * marketing_spend
        + 14 * seasonality
        + 0.55 * baseline_demand
        + 0.45 * competitor_price
        + 5 * seasonality * (marketing_spend / 100)
        + rng.normal(0, 7, n_samples)
    )
    units_sold = np.clip(units_sold, 0, None)
    revenue = price * units_sold

    return pd.DataFrame({
        "price": price,
        "units_sold": units_sold,
        "revenue": revenue,
        "seasonality": seasonality,
        "marketing_spend": marketing_spend,
        "competitor_price": competitor_price,
        "baseline_demand": baseline_demand,
    })
