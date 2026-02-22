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
        "market_demand": {"type": "number", "minimum": 0, "maximum": 1},
        "trim_tier": {"type": "number", "minimum": 0, "maximum": 1},
        "depreciation_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "mileage_assessment": {"type": "number", "minimum": 0, "maximum": 1},
        "comparable_market": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string", "maxLength": 100},
    },
    "additionalProperties": False,
}

CONTINUOUS_SCORE_FIELDS = (
    "market_demand", "trim_tier", "depreciation_rate",
    "mileage_assessment", "comparable_market",
)


def validate_feature_output(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate E5 feature-extraction output (continuous 0-1 scores)."""
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
    for field in CONTINUOUS_SCORE_FIELDS:
        v = data[field]
        if not isinstance(v, (int, float)) or not (0 <= v <= 1):
            return False, f"{field} must be a number in [0, 1]"
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
