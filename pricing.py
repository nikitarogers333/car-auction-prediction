"""
Deterministic pricing: compute final price from LLM-extracted features.

IMPORTANT: This formula is DELIBERATELY DIFFERENT from the ground truth
formula in data/generate_eval_dataset.py. The ground truth uses exponential
depreciation, quadratic mileage, polynomial condition, and interaction terms.
This formula uses linear depreciation, mileage buckets, and additive adjustments.
If they were identical, Pipeline B (E5) would win trivially.
"""

from __future__ import annotations

from typing import Any

# Base MSRP lookup by comparable_market x trim_tier (rough BMW figures)
_BASE_PRICES: dict[str, dict[str, float]] = {
    "budget":      {"base": 28000, "mid": 32000, "premium": 38000, "performance": 45000},
    "mainstream":  {"base": 32000, "mid": 38000, "premium": 46000, "performance": 55000},
    "luxury":      {"base": 38000, "mid": 45000, "premium": 55000, "performance": 70000},
    "performance": {"base": 42000, "mid": 50000, "premium": 60000, "performance": 82000},
}

_DEMAND_MULTIPLIER = {"low": 0.88, "medium": 1.00, "high": 1.12}

_DEPRECIATION_PER_YEAR = {"slow": 0.06, "normal": 0.10, "fast": 0.15}

_MILEAGE_DISCOUNT = {"low": 0, "average": -1500, "high": -4000, "very_high": -7500}


def compute_price_from_features(
    features: dict[str, Any],
    year: int,
    mileage: int,
) -> float:
    """
    Pure deterministic computation: no randomness, no LLM calls.
    Returns predicted price >= 2000.
    """
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
