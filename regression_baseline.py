#!/usr/bin/env python3
"""
Train Random Forest and XGBoost regression models on any car dataset.

Supports CSV and Excel. Auto-detects columns via fuzzy matching.
Required: price/target column. Optional: year, mileage, make, model, etc.
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

from config import REGRESSION_DATA_DIR

DATA_DIR = REGRESSION_DATA_DIR
KAGGLE_CSV = DATA_DIR / "car_prices.csv"

# Internal names -> possible external column names (case-insensitive)
COLUMN_ALIASES: dict[str, list[str]] = {
    # Required
    "target": [
        "sellingprice", "selling_price", "Selling_Price",
        "price", "Price", "sale_price", "Sale_Price",
        "final_price", "auction_price", "target",
    ],
    # Optional numeric
    "odometer": ["odometer", "Kms_Driven", "mileage", "Mileage", "miles", "km", "kilometers"],
    "year": ["year", "Year", "model_year", "yr"],
    "condition": ["condition", "Condition", "grade"],
    "mmr": ["mmr", "MMR", "market_value", "wholesale_price"],
    # Optional categorical
    "make": ["make", "Make", "brand", "Brand", "manufacturer"],
    "model": ["model", "Model", "car_model"],
    "transmission": ["transmission", "Transmission", "gearbox"],
    "fuel_type": ["fuel_type", "Fuel_Type", "fuel", "Fuel"],
    "body": ["body", "Body", "body_type", "body_style", "type"],
    "color": ["color", "Color"],
    "interior": ["interior", "Interior"],
    "drive": ["drive", "drivetrain", "Drive"],
}

RF_PATH = DATA_DIR / "random_forest_model.pkl"
XGB_PATH = DATA_DIR / "xgboost_model.pkl"
ENCODERS_PATH = DATA_DIR / "label_encoders.pkl"
MEDIANS_PATH = DATA_DIR / "training_medians.pkl"
METRICS_JSON_PATH = DATA_DIR / "training_metrics.json"

TRAIN_SAMPLE_SIZE = 5000


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Find first column that matches any alias (case-insensitive)."""
    cols_lower = {c.strip().lower(): c for c in df.columns}
    for a in aliases:
        key = a.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
        # Also try exact match with column as-is
        for orig in df.columns:
            if orig.strip().lower() == key:
                return orig
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Map internal feature names to actual column names in the dataframe.
    Returns dict: internal_name -> actual_column_name.
    """
    mapping: dict[str, str] = {}
    for internal, aliases in COLUMN_ALIASES.items():
        found = _find_column(df, aliases)
        if found:
            mapping[internal] = found
    return mapping


def load_file(path: Path) -> pd.DataFrame:
    """Load CSV or Excel based on extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        df = pd.read_excel(path, engine=engine)
    else:
        df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_and_clean(
    path: Path,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Load, clean, and rename columns using the detected mapping.
    Returns dataframe with internal column names.
    """
    df = load_file(path)
    target_col = column_mapping.get("target")
    if not target_col:
        raise ValueError(
            "Could not find a price column. Please rename your price column to 'price' and re-upload."
        )

    # Rename to internal names
    rename = {v: k for k, v in column_mapping.items()}
    df = df.rename(columns=rename)

    # Drop rows with null or zero target
    df = df.dropna(subset=["target"])
    df = df[df["target"] > 0].copy()

    # Sample
    if len(df) > TRAIN_SAMPLE_SIZE:
        df = df.sample(n=TRAIN_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    # Get list of feature columns (everything except target)
    numeric_internal = ["year", "odometer", "condition", "mmr"]
    categorical_internal = ["make", "model", "body", "transmission", "color", "interior", "fuel_type", "drive"]

    for col in numeric_internal:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_internal:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    return df


def encode_and_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, LabelEncoder], dict[str, float]]:
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df = df.copy()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(float)

    medians: dict[str, float] = {}
    for col in feature_cols:
        medians[col] = float(df[col].median())

    return X, y, encoders, medians


def train_and_save(path: Path | None = None) -> None:
    path = path or KAGGLE_CSV
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"Loading {path}...")
    df_raw = load_file(path)
    print(f"  Loaded {len(df_raw)} rows, columns: {list(df_raw.columns)}")

    column_mapping = detect_columns(df_raw)
    target_col = column_mapping.get("target")
    if not target_col:
        raise ValueError(
            "Could not find a price column. Please rename your price column to 'price' and re-upload."
        )

    # Build feature lists
    numeric_internal = ["year", "odometer", "condition", "mmr"]
    categorical_internal = ["make", "model", "body", "transmission", "color", "interior", "fuel_type", "drive"]

    detected_numeric = [c for c in numeric_internal if c in column_mapping]
    detected_categorical = [c for c in categorical_internal if c in column_mapping]
    feature_cols = detected_numeric + detected_categorical

    missing_numeric = [c for c in numeric_internal if c not in column_mapping]
    missing_categorical = [c for c in categorical_internal if c not in column_mapping]

    print(f"\nUsing these features: {feature_cols}")
    if missing_numeric or missing_categorical:
        print(f"Missing but optional: {missing_numeric + missing_categorical}")

    if len(feature_cols) < 1:
        raise ValueError(
            "Need at least one feature column (year, mileage, make, model, etc.). "
            "Your file has a price column but no detectable features."
        )

    df = load_and_clean(path, column_mapping)
    print(f"  {len(df)} rows after cleaning + sampling {TRAIN_SAMPLE_SIZE}")

    X, y, encoders, medians = encode_and_split(
        df, feature_cols, detected_numeric, detected_categorical
    )
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
        raise RuntimeError("xgboost not installed. Run: pip install xgboost")
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
    metrics = {
        "rf_mae": rf_mae,
        "rf_r2": rf_r2,
        "xgb_mae": xgb_mae,
        "xgb_r2": xgb_r2,
        "features_used": feature_cols,
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    print(f"\nSaved:")
    print(f"  {RF_PATH}")
    print(f"  {XGB_PATH}")
    print(f"  {ENCODERS_PATH}")
    print(f"  {MEDIANS_PATH}")
    print(f"\nDone. Features used: {feature_cols}")


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else KAGGLE_CSV
    train_and_save(path)
