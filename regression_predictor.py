"""
Clean wrapper for trained regression models (Random Forest, XGBoost).

Loads saved pkl files at import time. compare.py and app.py import from here,
never from regression_baseline.py directly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
RF_PATH = DATA_DIR / "random_forest_model.pkl"
XGB_PATH = DATA_DIR / "xgboost_model.pkl"
ENCODERS_PATH = DATA_DIR / "label_encoders.pkl"
MEDIANS_PATH = DATA_DIR / "training_medians.pkl"

NUMERIC_FEATURES = ["year", "odometer", "condition", "mmr"]
CATEGORICAL_FEATURES = ["make", "model", "body", "transmission", "color", "interior"]

_rf_bundle: dict[str, Any] | None = None
_xgb_bundle: dict[str, Any] | None = None
_encoders: dict | None = None
_medians: dict[str, float] | None = None
_loaded = False


class RegressionModelsNotTrained(RuntimeError):
    pass


def _load() -> None:
    global _rf_bundle, _xgb_bundle, _encoders, _medians, _loaded
    if _loaded:
        return
    for p, name in [(RF_PATH, "Random Forest"), (XGB_PATH, "XGBoost"),
                     (ENCODERS_PATH, "Label encoders"), (MEDIANS_PATH, "Training medians")]:
        if not p.exists():
            raise RegressionModelsNotTrained(
                f"Regression models not trained ({name} missing: {p}). "
                f"Run: python3 regression_baseline.py"
            )
    with open(RF_PATH, "rb") as f:
        _rf_bundle = pickle.load(f)
    with open(XGB_PATH, "rb") as f:
        _xgb_bundle = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        _encoders = pickle.load(f)
    with open(MEDIANS_PATH, "rb") as f:
        _medians = pickle.load(f)
    _loaded = True


def models_available() -> bool:
    """Check if trained model files exist without raising."""
    return all(p.exists() for p in [RF_PATH, XGB_PATH, ENCODERS_PATH, MEDIANS_PATH])


def _encode_value(col: str, value: str) -> float:
    """Encode a categorical value, falling back to most common class if unseen."""
    assert _encoders is not None and _medians is not None
    le = _encoders.get(col)
    if le is None:
        return _medians.get(col, 0.0)
    value = str(value)
    if value in le.classes_:
        return float(le.transform([value])[0])
    return float(le.transform([le.classes_[0]])[0])


def _build_feature_vector(
    year: float, odometer: float, condition: float, mmr: float,
    make: str, model: str, body: str, transmission: str,
    color: str, interior: str,
) -> np.ndarray:
    """Build feature vector matching training column order."""
    _load()
    assert _medians is not None
    feature_cols = _rf_bundle["feature_cols"] if _rf_bundle else NUMERIC_FEATURES + CATEGORICAL_FEATURES

    values: dict[str, float] = {}
    values["year"] = float(year) if year else _medians.get("year", 2018)
    values["odometer"] = float(odometer) if odometer else _medians.get("odometer", 50000)
    values["condition"] = float(condition) if condition else _medians.get("condition", 3.0)
    values["mmr"] = float(mmr) if mmr else _medians.get("mmr", 15000)

    values["make"] = _encode_value("make", make)
    values["model"] = _encode_value("model", model)
    values["body"] = _encode_value("body", body)
    values["transmission"] = _encode_value("transmission", transmission)
    values["color"] = _encode_value("color", color)
    values["interior"] = _encode_value("interior", interior)

    vec = [values.get(c, _medians.get(c, 0.0)) for c in feature_cols]
    return np.array([vec], dtype=float)


def predict_rf(
    year: float = 0, odometer: float = 0, condition: float = 0, mmr: float = 0,
    make: str = "", model: str = "", body: str = "", transmission: str = "",
    color: str = "", interior: str = "",
) -> float:
    """Predict price using Random Forest. Returns float."""
    _load()
    assert _rf_bundle is not None
    X = _build_feature_vector(year, odometer, condition, mmr, make, model, body, transmission, color, interior)
    return float(_rf_bundle["model"].predict(X)[0])


def predict_xgb(
    year: float = 0, odometer: float = 0, condition: float = 0, mmr: float = 0,
    make: str = "", model: str = "", body: str = "", transmission: str = "",
    color: str = "", interior: str = "",
) -> float:
    """Predict price using XGBoost. Returns float."""
    _load()
    assert _xgb_bundle is not None
    X = _build_feature_vector(year, odometer, condition, mmr, make, model, body, transmission, color, interior)
    return float(_xgb_bundle["model"].predict(X)[0])
