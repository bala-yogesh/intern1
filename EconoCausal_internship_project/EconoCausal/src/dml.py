"""A small cross-fitted Double Machine Learning estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


def estimate_price_effect(
    data: pd.DataFrame,
    treatment: str = "price",
    outcome: str = "units_sold",
    controls: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate a partially linear treatment effect with cross-fitting.

    Returns the coefficient from regressing cross-fitted outcome residuals
    on treatment residuals. This educational implementation is not a
    substitute for a full causal analysis or a validated production library.
    """
    if controls is None:
        controls = [
            "seasonality",
            "marketing_spend",
            "competitor_price",
            "baseline_demand",
        ]
    required = [treatment, outcome, *controls]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Input data contains missing values in required columns.")
    if len(data) < max(100, n_splits * 10):
        raise ValueError("Not enough rows for the requested cross-fitting setup.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    X = data[controls].to_numpy(dtype=float)
    D = data[treatment].to_numpy(dtype=float)
    Y = data[outcome].to_numpy(dtype=float)

    y_hat = np.zeros(len(data), dtype=float)
    d_hat = np.zeros(len(data), dtype=float)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in cv.split(X):
        y_model = RandomForestRegressor(
            n_estimators=120, min_samples_leaf=8, random_state=random_state, n_jobs=-1
        )
        d_model = RandomForestRegressor(
            n_estimators=120, min_samples_leaf=8, random_state=random_state + 1, n_jobs=-1
        )
        y_model.fit(X[train_idx], Y[train_idx])
        d_model.fit(X[train_idx], D[train_idx])
        y_hat[test_idx] = y_model.predict(X[test_idx])
        d_hat[test_idx] = d_model.predict(X[test_idx])

    y_resid = Y - y_hat
    d_resid = D - d_hat
    denominator = float(np.dot(d_resid, d_resid))
    if denominator <= 1e-12:
        raise ValueError("Price residual variation is too small to estimate an effect.")

    theta = float(np.dot(d_resid, y_resid) / denominator)
    score_resid = y_resid - theta * d_resid
    standard_error = float(
        np.sqrt(np.mean((d_resid * score_resid) ** 2))
        / max(np.sqrt(np.mean(d_resid ** 2)), 1e-12)
        / np.sqrt(len(data))
    )
    return {
        "effect_per_price_unit": theta,
        "standard_error_approx": standard_error,
        "n_samples": int(len(data)),
        "n_splits": int(n_splits),
    }
