"""
Strict JSON schemas for the deterministic car auction price prediction pipeline.
All agent outputs must conform to these schemas.
"""

from __future__ import annotations

from typing import Any, Literal

# -----------------------------------------------------------------------------
# PredictionAgent output (THE canonical format)
# -----------------------------------------------------------------------------

PREDICTION_METHODS = ("llm_internal", "nearest_neighbors", "regression", "feature_extraction")
SUBGROUP_EXAMPLES = ("M3", "340i", "328i", "M5", "generic")  # extensible via subgroup_map

PredictionOutputSchema = {
    "type": "object",
    "required": ["predicted_price", "confidence", "method", "subgroup_detected", "notes"],
    "properties": {
        "predicted_price": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "method": {"type": "string", "enum": list(PREDICTION_METHODS)},
        "subgroup_detected": {"type": "string"},
        "notes": {"type": "string", "maxLength": 200},
    },
    "additionalProperties": False,
}


def validate_prediction_output(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate PredictionAgent output. Returns (valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "output must be a JSON object"
    required = {"predicted_price", "confidence", "method", "subgroup_detected", "notes"}
    if set(data.keys()) != required and set(data.keys()) - required:
        extra = set(data.keys()) - required
        if extra:
            return False, f"unknown keys: {extra}"
    for key in required:
        if key not in data:
            return False, f"missing key: {key}"
    if not isinstance(data["predicted_price"], (int, float)) or data["predicted_price"] < 0:
        return False, "predicted_price must be a non-negative number"
    if not isinstance(data["confidence"], (int, float)) or not (0 <= data["confidence"] <= 1):
        return False, "confidence must be a number in [0, 1]"
    if data["method"] not in PREDICTION_METHODS:
        return False, f"method must be one of {PREDICTION_METHODS}"
    if not isinstance(data["subgroup_detected"], str):
        return False, "subgroup_detected must be a string"
    if not isinstance(data["notes"], str):
        return False, "notes must be a string"
    if len(data["notes"]) > 200:
        return False, "notes must be max 200 chars"
    return True, ""


# -----------------------------------------------------------------------------
# E5 feature-extraction output (LLM outputs features, code computes price)
# -----------------------------------------------------------------------------

TRIM_TIERS = ("base", "mid", "premium", "performance")
DEMAND_LEVELS = ("low", "medium", "high")
DEPRECIATION_RATES = ("slow", "normal", "fast")
MILEAGE_ASSESSMENTS = ("low", "average", "high", "very_high")
MARKET_SEGMENTS = ("budget", "mainstream", "luxury", "performance")

FeatureOutputSchema = {
    "type": "object",
    "required": [
        "condition_score", "market_demand", "trim_tier",
        "depreciation_rate", "mileage_assessment", "comparable_market", "notes",
    ],
    "properties": {
        "condition_score": {"type": "number", "minimum": 1, "maximum": 10},
        "market_demand": {"type": "string", "enum": list(DEMAND_LEVELS)},
        "trim_tier": {"type": "string", "enum": list(TRIM_TIERS)},
        "depreciation_rate": {"type": "string", "enum": list(DEPRECIATION_RATES)},
        "mileage_assessment": {"type": "string", "enum": list(MILEAGE_ASSESSMENTS)},
        "comparable_market": {"type": "string", "enum": list(MARKET_SEGMENTS)},
        "notes": {"type": "string", "maxLength": 100},
    },
    "additionalProperties": False,
}


def validate_feature_output(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate E5 feature-extraction output. Returns (valid, error_message)."""
    if not isinstance(data, dict):
        return False, "output must be a JSON object"
    required = {"condition_score", "market_demand", "trim_tier", "depreciation_rate",
                "mileage_assessment", "comparable_market", "notes"}
    for key in required:
        if key not in data:
            return False, f"missing key: {key}"
    extra = set(data.keys()) - required
    if extra:
        return False, f"unknown keys: {extra}"
    cs = data["condition_score"]
    if not isinstance(cs, (int, float)) or not (1 <= cs <= 10):
        return False, "condition_score must be a number in [1, 10]"
    if data["market_demand"] not in DEMAND_LEVELS:
        return False, f"market_demand must be one of {DEMAND_LEVELS}"
    if data["trim_tier"] not in TRIM_TIERS:
        return False, f"trim_tier must be one of {TRIM_TIERS}"
    if data["depreciation_rate"] not in DEPRECIATION_RATES:
        return False, f"depreciation_rate must be one of {DEPRECIATION_RATES}"
    if data["mileage_assessment"] not in MILEAGE_ASSESSMENTS:
        return False, f"mileage_assessment must be one of {MILEAGE_ASSESSMENTS}"
    if data["comparable_market"] not in MARKET_SEGMENTS:
        return False, f"comparable_market must be one of {MARKET_SEGMENTS}"
    if not isinstance(data["notes"], str):
        return False, "notes must be a string"
    if len(data["notes"]) > 100:
        return False, "notes must be max 100 chars"
    return True, ""


# -----------------------------------------------------------------------------
# Internal pipeline payload (passed between steps)
# -----------------------------------------------------------------------------

def pipeline_payload_schema() -> dict[str, Any]:
    """Schema for the JSON passed between pipeline steps."""
    return {
        "vehicle_id": str,
        "vin": str | None,
        "make": str,
        "model": str,
        "year": int,
        "mileage": int | None,
        "features": dict[str, Any],
        "subgroup": str,
        "allowed_comparables": list[str],
        "restrictions_applied": list[str],
        "prediction": dict[str, Any] | None,
        "valid": bool,
        "violation_reason": str | None,
    }
