"""
SubgroupClassifier: deterministic classification of vehicle subgroup (e.g. M3 vs 340i).
Uses restrictions/subgroup_map.json. No LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SUBGROUP = "generic"
DEFAULT_ALLOWED = ["generic"]
DEFAULT_FORBIDDEN: list[str] = []


class SubgroupClassifier:
    """Maps make/model to subgroup and allowed/forbidden comparables."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._map: dict[str, Any] = self._load_map()

    def _load_map(self) -> dict[str, Any]:
        path = self.project_root / "restrictions" / "subgroup_map.json"
        if not path.exists():
            return {"rules": [], "default_subgroup": DEFAULT_SUBGROUP, "default_allowed": DEFAULT_ALLOWED, "default_forbidden": DEFAULT_FORBIDDEN}
        with open(path) as f:
            return json.load(f)

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Set subgroup, allowed_comparables, and implicitly forbidden from map."""
        out = dict(payload)
        make = (out.get("make") or "").strip()
        model = (out.get("model") or "").strip()
        rules = self._map.get("rules") or []
        subgroup = self._map.get("default_subgroup", DEFAULT_SUBGROUP)
        allowed = list(self._map.get("default_allowed", DEFAULT_ALLOWED))
        forbidden = list(self._map.get("default_forbidden", DEFAULT_FORBIDDEN))

        for rule in rules:
            if (rule.get("make") or "").strip() != make:
                continue
            pattern = (rule.get("model_pattern") or "").strip()
            if pattern == "default":
                subgroup = rule.get("subgroup", subgroup)
                allowed = list(rule.get("allowed_comparables", allowed))
                forbidden = list(rule.get("forbidden_comparables", forbidden))
                break
            if pattern.upper() in model.upper() or model.upper() in pattern.upper():
                subgroup = rule.get("subgroup", subgroup)
                allowed = list(rule.get("allowed_comparables", allowed))
                forbidden = list(rule.get("forbidden_comparables", forbidden))
                break

        out["subgroup"] = subgroup
        out["allowed_comparables"] = allowed
        out["restrictions_applied"] = list(out.get("restrictions_applied") or [])
        if "subgroup_classified" not in out["restrictions_applied"]:
            out["restrictions_applied"].append("subgroup_classified")
        # Store forbidden for RestrictionEnforcer / PostValidationGuard
        out["_forbidden_comparables"] = forbidden
        return out
