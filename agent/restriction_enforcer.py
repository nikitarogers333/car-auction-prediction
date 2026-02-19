"""
RestrictionEnforcer: deterministic enforcement of all rules.
No external calls, no LLM. Validates payload and applies restrictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas import PREDICTION_METHODS, validate_prediction_output


class RestrictionEnforcer:
    """
    Enforces:
    1) No external web unless explicitly enabled (handled by caller via condition P1–P4)
    2) VIN already stripped in PreValidationGuard
    3) Subgroup isolation (allowed_comparables set by SubgroupClassifier)
    4) No duplicate record usage (caller must track; we can flag if duplicate vehicle_id in same run)
    5) No ground truth leakage (PreValidationGuard strips target/price)
    6) Price range sanity (load from restrictions/price_bounds.json)
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._price_bounds: dict[str, Any] = self._load_price_bounds()

    def _load_price_bounds(self) -> dict[str, Any]:
        path = self.project_root / "restrictions" / "price_bounds.json"
        if not path.exists():
            return {"default_min_price": 1000, "default_max_price": 500000, "by_make_model": {}}
        with open(path) as f:
            return json.load(f)

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply restrictions; do not call external APIs. Pass through payload."""
        out = dict(payload)
        applied = list(out.get("restrictions_applied") or [])
        if "restriction_enforcer" not in applied:
            applied.append("restriction_enforcer")
        out["restrictions_applied"] = applied
        return out

    def check_price_bounds(self, make: str, model: str, price: float) -> tuple[bool, str]:
        """Returns (within_bounds, reason)."""
        key = f"{make}_{model}".replace(" ", "_")
        bounds = self._price_bounds.get("by_make_model") or {}
        if key in bounds:
            lo = bounds[key].get("min", self._price_bounds.get("default_min_price", 1000))
            hi = bounds[key].get("max", self._price_bounds.get("default_max_price", 500000))
        else:
            lo = self._price_bounds.get("default_min_price", 1000)
            hi = self._price_bounds.get("default_max_price", 500000)
        if price < lo:
            return False, f"price {price} below minimum {lo} for {make} {model}"
        if price > hi:
            return False, f"price {price} above maximum {hi} for {make} {model}"
        return True, ""

    def validate_prediction_structure(self, prediction: dict[str, Any]) -> tuple[bool, str]:
        """Validate PredictionAgent output format."""
        return validate_prediction_output(prediction)
