"""
PreValidationGuard: validates input vehicle record before any agent runs.
Deterministic; no LLM. Outputs strict JSON for next step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# VIN and other sensitive keys that must be stripped before prediction to prevent leakage
SENSITIVE_KEYS = {"vin", "VIN", "target", "price", "sale_price", "actual_price", "ground_truth"}


class PreValidationGuard:
    """Validates and normalizes input; removes VIN/target for prediction path."""

    def __init__(self, allow_vin_in_context: bool = False) -> None:
        self.allow_vin_in_context = allow_vin_in_context

    def run(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        """
        Input: raw vehicle record (may contain vin, make, model, year, mileage, etc.)
        Output: strict JSON payload for FeatureExtractionAgent.
        """
        out: dict[str, Any] = {
            "vehicle_id": "",
            "vin": None,
            "make": "",
            "model": "",
            "year": 0,
            "mileage": None,
            "features": {},
            "subgroup": "",
            "allowed_comparables": [],
            "restrictions_applied": [],
            "prediction": None,
            "valid": True,
            "violation_reason": None,
        }
        # Vehicle ID: require some identifier
        out["vehicle_id"] = str(
            raw_input.get("vehicle_id")
            or raw_input.get("id")
            or raw_input.get("VIN")
            or raw_input.get("vin")
            or "unknown"
        ).strip() or "unknown"

        vin_raw = raw_input.get("vin") or raw_input.get("VIN")
        if self.allow_vin_in_context:
            out["vin"] = str(vin_raw).strip() if vin_raw else None
        else:
            out["vin"] = None  # strip for leakage prevention

        out["make"] = str(raw_input.get("make", "")).strip()
        out["model"] = str(raw_input.get("model", "")).strip()
        try:
            out["year"] = int(raw_input.get("year", 0))
        except (TypeError, ValueError):
            out["year"] = 0
        try:
            out["mileage"] = int(raw_input.get("mileage", 0)) if raw_input.get("mileage") is not None else None
        except (TypeError, ValueError):
            out["mileage"] = None

        # Build features from raw_input, excluding sensitive keys
        features: dict[str, Any] = {}
        for k, v in raw_input.items():
            if k in SENSITIVE_KEYS:
                continue
            if k.lower() in {"vin", "target", "price", "sale_price", "actual_price", "ground_truth"}:
                continue
            if v is not None and v != "":
                features[k] = v
        out["features"] = features

        # Basic validity
        if not out["make"] or not out["model"]:
            out["valid"] = False
            out["violation_reason"] = "missing make or model"
        return out
