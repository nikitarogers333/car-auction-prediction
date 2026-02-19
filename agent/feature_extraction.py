"""
FeatureExtractionAgent: extracts structured features from vehicle record.
In mock mode: passes through; in LLM mode could use LLM to normalize (we keep deterministic).
"""

from __future__ import annotations

from typing import Any


class FeatureExtractionAgent:
    """Deterministic feature extraction. Outputs same schema as input payload."""

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Enrich features if needed; pass through payload structure."""
        out = dict(payload)
        features = dict(out.get("features") or {})
        # Normalize common keys
        if "year" not in features and out.get("year"):
            features["year"] = out["year"]
        if "make" not in features and out.get("make"):
            features["make"] = out["make"]
        if "model" not in features and out.get("model"):
            features["model"] = out["model"]
        if out.get("mileage") is not None and "mileage" not in features:
            features["mileage"] = out["mileage"]
        out["features"] = features
        return out
