"""
Baseline models: Nearest Neighbors, Linear Regression, optional XGBoost.
Used for comparison with LLM predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _ensure_numpy(X: Any) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    return np.array(X)


def nearest_neighbors_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_row: np.ndarray,
    k: int = 5,
) -> float:
    """Single prediction: mean of k nearest neighbors by L2 distance."""
    train_X = _ensure_numpy(train_X)
    train_y = _ensure_numpy(train_y)
    test_row = _ensure_numpy(test_row).reshape(1, -1)
    n_train = train_X.shape[0]
    k_use = min(k, max(1, n_train))
    dists = np.linalg.norm(train_X - test_row, axis=1)
    kth = min(k_use - 1, n_train - 1)
    idx = np.argpartition(dists, kth)[:k_use]
    return float(np.mean(train_y[idx]))


def linear_regression_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_row: np.ndarray,
) -> float:
    """Single prediction: OLS fit on train, predict test_row."""
    train_X = _ensure_numpy(train_X)
    train_y = _ensure_numpy(train_y)
    test_row = _ensure_numpy(test_row).reshape(1, -1)
    # Add intercept
    ones = np.ones((train_X.shape[0], 1))
    X = np.hstack([ones, train_X])
    try:
        beta = np.linalg.lstsq(X, train_y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float(np.mean(train_y))
    test_1 = np.hstack([np.ones((1, 1)), test_row])
    return float(np.dot(test_1, beta).item())


def xgboost_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_row: np.ndarray,
    **kwargs: Any,
) -> float:
    """Optional XGBoost baseline. Returns 0.0 if xgboost not installed."""
    try:
        import xgboost as xgb
    except ImportError:
        return 0.0
    train_X = _ensure_numpy(train_X)
    train_y = _ensure_numpy(train_y)
    test_row = _ensure_numpy(test_row).reshape(1, -1)
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dtest = xgb.DMatrix(test_row)
    params = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror", **kwargs}
    model = xgb.train(params, dtrain, num_boost_round=50)
    return float(model.predict(dtest).item())


def build_feature_vector(record: dict[str, Any], numeric_keys: list[str] | None = None) -> np.ndarray:
    """Build a numeric feature vector from vehicle record for baselines."""
    if numeric_keys is None:
        numeric_keys = ["year", "mileage"]
    vals = []
    for k in numeric_keys:
        v = record.get(k) or record.get("features", {}).get(k)
        if v is None:
            v = 0
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(0.0)
    return np.array(vals, dtype=float)
