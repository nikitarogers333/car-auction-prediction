#!/usr/bin/env python3
"""
Train Random Forest and XGBoost regression models on the Kaggle car_prices.csv dataset.

Usage:
    1. Place car_prices.csv in the data/ folder
    2. Run: python3 regression_baseline.py
    3. Models and encoders are saved to data/*.pkl
    4. compare.py and app.py import from regression_predictor.py (not this file)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent / "data"
KAGGLE_CSV = DATA_DIR / "car_prices.csv"

NUMERIC_FEATURES = ["year", "odometer", "condition", "mmr"]
CATEGORICAL_FEATURES = ["make", "model", "body", "transmission", "color", "interior"]
DROP_COLUMNS = ["vin", "state", "seller", "saledate", "trim"]
TARGET = "sellingprice"

RF_PATH = DATA_DIR / "random_forest_model.pkl"
XGB_PATH = DATA_DIR / "xgboost_model.pkl"
ENCODERS_PATH = DATA_DIR / "label_encoders.pkl"
MEDIANS_PATH = DATA_DIR / "training_medians.pkl"
METRICS_JSON_PATH = DATA_DIR / "training_metrics.json"


TRAIN_SAMPLE_SIZE = 5000


def load_and_clean(path: Path = KAGGLE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].copy()
    df = df.sample(n=TRAIN_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
    return df


def encode_and_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, LabelEncoder], dict[str, float]]:
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols].values.astype(float)
    y = df[TARGET].values.astype(float)

    medians: dict[str, float] = {}
    for col in feature_cols:
        medians[col] = float(df[col].median())

    return X, y, encoders, medians


def train_and_save() -> None:
    print(f"Loading {KAGGLE_CSV}...")
    df = load_and_clean()
    print(f"  {len(df)} rows after cleaning + sampling {TRAIN_SAMPLE_SIZE}")

    X, y, encoders, medians = encode_and_split(df)
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    print(f"  Features: {feature_cols}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # Pipeline C: Random Forest
    print("\nTraining Random Forest (5-fold CV)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_mae_scores = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error")
    rf_r2_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    print(f"  CV MAE:  ${-rf_mae_scores.mean():,.0f} (+/- ${rf_mae_scores.std():,.0f})")
    print(f"  CV R^2:  {rf_r2_scores.mean():.4f} (+/- {rf_r2_scores.std():.4f})")
    rf.fit(X, y)

    # Pipeline D: XGBoost
    print("\nTraining XGBoost (5-fold CV)...")
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        return
    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    xgb_mae_scores = cross_val_score(xgb, X, y, cv=5, scoring="neg_mean_absolute_error")
    xgb_r2_scores = cross_val_score(xgb, X, y, cv=5, scoring="r2")
    print(f"  CV MAE:  ${-xgb_mae_scores.mean():,.0f} (+/- ${xgb_mae_scores.std():,.0f})")
    print(f"  CV R^2:  {xgb_r2_scores.mean():.4f} (+/- {xgb_r2_scores.std():.4f})")
    xgb.fit(X, y)

    # Save
    with open(RF_PATH, "wb") as f:
        pickle.dump({"model": rf, "feature_cols": feature_cols}, f)
    with open(XGB_PATH, "wb") as f:
        pickle.dump({"model": xgb, "feature_cols": feature_cols}, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    with open(MEDIANS_PATH, "wb") as f:
        pickle.dump(medians, f)

    rf_mae = float(-rf_mae_scores.mean())
    rf_r2 = float(rf_r2_scores.mean())
    xgb_mae = float(-xgb_mae_scores.mean())
    xgb_r2 = float(xgb_r2_scores.mean())
    metrics = {"rf_mae": rf_mae, "rf_r2": rf_r2, "xgb_mae": xgb_mae, "xgb_r2": xgb_r2}
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    print(f"\nSaved:")
    print(f"  {RF_PATH}")
    print(f"  {XGB_PATH}")
    print(f"  {ENCODERS_PATH}")
    print(f"  {MEDIANS_PATH}")
    print("\nDone. You can now run compare.py with all four pipelines.")


if __name__ == "__main__":
    train_and_save()
