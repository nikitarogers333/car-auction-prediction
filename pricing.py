"""
Deterministic pricing: compute final price from LLM-extracted features.

IMPORTANT: This formula is DELIBERATELY DIFFERENT from the ground truth
formula in data/generate_eval_dataset.py. The ground truth uses exponential
depreciation, quadratic mileage, polynomial condition, and interaction terms.
This formula uses linear depreciation, mileage buckets, and additive adjustments.
If they were identical, Pipeline B (E5) would win trivially.

Supports two modes:
  - Continuous (E5): all scoring fields are 0-1 floats, formula interpolates smoothly.
  - Categorical (A' backward compat): string categories map to fixed values.
  Auto-detected based on whether market_demand is numeric or string.
"""

from __future__ import annotations

from typing import Any

# Base MSRP grid: market (rows) x trim (cols)
# Anchors: 0.0, 1/3, 2/3, 1.0
_BASE_PRICE_GRID = [
    [28000, 32000, 38000, 45000],  # budget      (market=0.0)
    [32000, 38000, 46000, 55000],  # mainstream   (market≈0.33)
    [38000, 45000, 55000, 70000],  # luxury       (market≈0.67)
    [42000, 50000, 60000, 82000],  # performance  (market=1.0)
]
_GRID_ANCHORS = [0.0, 1 / 3, 2 / 3, 1.0]

# Categorical lookups (kept for A' backward compatibility)
_BASE_PRICES: dict[str, dict[str, float]] = {
    "budget":      {"base": 28000, "mid": 32000, "premium": 38000, "performance": 45000},
    "mainstream":  {"base": 32000, "mid": 38000, "premium": 46000, "performance": 55000},
    "luxury":      {"base": 38000, "mid": 45000, "premium": 55000, "performance": 70000},
    "performance": {"base": 42000, "mid": 50000, "premium": 60000, "performance": 82000},
}
_DEMAND_MULTIPLIER = {"low": 0.88, "medium": 1.00, "high": 1.12}
_DEPRECIATION_PER_YEAR = {"slow": 0.06, "normal": 0.10, "fast": 0.15}
_MILEAGE_DISCOUNT = {"low": 0, "average": -1500, "high": -4000, "very_high": -7500}


def _lerp_1d(score: float, anchors: list[float], values: list[float]) -> float:
    """Piecewise linear interpolation along anchored control points."""
    score = max(anchors[0], min(anchors[-1], score))
    for i in range(len(anchors) - 1):
        if score <= anchors[i + 1]:
            t = (score - anchors[i]) / (anchors[i + 1] - anchors[i]) if anchors[i + 1] != anchors[i] else 0.0
            return values[i] + t * (values[i + 1] - values[i])
    return values[-1]


def _interpolate_base_price(market_score: float, trim_score: float) -> float:
    """Bilinear interpolation of base price from continuous market and trim scores."""
    row_vals = []
    for row in _BASE_PRICE_GRID:
        row_vals.append(_lerp_1d(trim_score, _GRID_ANCHORS, row))
    return _lerp_1d(market_score, _GRID_ANCHORS, row_vals)


def compute_price_from_features(
    features: dict[str, Any],
    year: int,
    mileage: int,
) -> float:
    """
    Pure deterministic computation: no randomness, no LLM calls.
    Auto-detects continuous (numeric) vs categorical (string) mode.
    Returns predicted price >= 2000.
    """
    demand_raw = features.get("market_demand", 0.5)

    if isinstance(demand_raw, (int, float)):
        return _compute_continuous(features, year, mileage)
    return _compute_categorical(features, year, mileage)


def _snap(v: float, step: float = 0.05) -> float:
    """Snap to nearest grid point. Eliminates micro-variation between runs
    while keeping 21 distinct levels (vs 3-4 for old categories)."""
    return round(v / step) * step


def _compute_continuous(features: dict[str, Any], year: int, mileage: int) -> float:
    """E5 path: all scoring fields are 0-1 floats, snapped to 0.05 grid,
    formula interpolates smoothly between grid points."""
    demand = _snap(max(0.0, min(1.0, float(features.get("market_demand", 0.5)))))
    trim = _snap(max(0.0, min(1.0, float(features.get("trim_tier", 0.5)))))
    market = _snap(max(0.0, min(1.0, float(features.get("comparable_market", 0.5)))))
    dep = _snap(max(0.0, min(1.0, float(features.get("depreciation_rate", 0.5)))))
    mi = _snap(max(0.0, min(1.0, float(features.get("mileage_assessment", 0.5)))))
    condition_score = round(float(features.get("condition_score", 5.0)) * 2) / 2  # snap to 0.5

    base = _interpolate_base_price(market, trim)

    age = max(0, 2025 - year)
    annual_dep = 0.06 + dep * (0.15 - 0.06)
    depreciation = base * annual_dep * age

    mileage_adj = -7500.0 * mi

    demand_mult = 0.88 + demand * (1.12 - 0.88)

    condition_adj = (condition_score - 5.0) * 1200

    price = (base - depreciation + mileage_adj + condition_adj) * demand_mult
    return max(2000.0, round(price, 2))


def _compute_categorical(features: dict[str, Any], year: int, mileage: int) -> float:
    """A' backward-compatible path: string categories map to fixed values."""
    trim = str(features.get("trim_tier", "mid")).lower()
    market = str(features.get("comparable_market", "mainstream")).lower()
    demand = str(features.get("market_demand", "medium")).lower()
    dep_rate = str(features.get("depreciation_rate", "normal")).lower()
    mileage_assess = str(features.get("mileage_assessment", "average")).lower()
    condition_score = float(features.get("condition_score", 5.0))

    base = _BASE_PRICES.get(market, _BASE_PRICES["mainstream"]).get(trim, 38000)

    age = max(0, 2025 - year)
    annual_dep = _DEPRECIATION_PER_YEAR.get(dep_rate, 0.10)
    depreciation = base * annual_dep * age

    mileage_adj = _MILEAGE_DISCOUNT.get(mileage_assess, -1500)

    demand_mult = _DEMAND_MULTIPLIER.get(demand, 1.00)

    condition_adj = (condition_score - 5.0) * 1200

    price = (base - depreciation + mileage_adj + condition_adj) * demand_mult
    return max(2000.0, round(price, 2))
