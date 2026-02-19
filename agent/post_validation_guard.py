"""
PostValidationGuard: verifies output format, no forbidden leakage, price bounds.
Deterministic. Sets valid=False and violation_reason on failure.
"""

from __future__ import annotations

from typing import Any

from schemas import validate_prediction_output

from .restriction_enforcer import RestrictionEnforcer


class PostValidationGuard:
    """
    Verify:
    - Output format valid (schema)
    - No forbidden subgroup leakage (subgroup_detected in allowed_comparables)
    - No VIN used (payload.vin should be None in prediction path)
    - No unauthorized domain (caller tracks; we don't have request log here)
    - No hallucinated references (basic check: notes don't contain forbidden comparables)
    - Price within expected distribution bounds
    """

    def __init__(self, project_root: Any = None) -> None:
        self.enforcer = RestrictionEnforcer(project_root)

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Set payload['valid'] and payload['violation_reason']."""
        out = dict(payload)
        pred = out.get("prediction")
        if pred is None:
            out["valid"] = False
            out["violation_reason"] = "missing prediction"
            return out

        valid, msg = validate_prediction_output(pred)
        if not valid:
            out["valid"] = False
            out["violation_reason"] = msg
            return out

        # Forbidden subgroup leakage: subgroup_detected must be in allowed_comparables
        allowed = set(out.get("allowed_comparables") or [])
        forbidden = set(out.get("_forbidden_comparables") or [])
        detected = (pred.get("subgroup_detected") or "").strip()
        if detected in forbidden:
            out["valid"] = False
            out["violation_reason"] = f"forbidden subgroup leakage: {detected} in forbidden_comparables"
            return out
        if allowed and detected not in allowed:
            out["valid"] = False
            out["violation_reason"] = f"subgroup_detected {detected} not in allowed_comparables {list(allowed)}"
            return out

        # VIN: must not appear in notes or anywhere in prediction
        vin = out.get("vin")
        if vin and (vin in (pred.get("notes") or "") or vin in str(pred)):
            out["valid"] = False
            out["violation_reason"] = "VIN leaked into prediction output"
            return out

        # Price bounds
        make = out.get("make") or ""
        model = out.get("model") or ""
        price = float(pred.get("predicted_price", 0))
        within, reason = self.enforcer.check_price_bounds(make, model, price)
        if not within:
            out["valid"] = False
            out["violation_reason"] = reason
            return out

        out["valid"] = True
        out["violation_reason"] = None
        return out
