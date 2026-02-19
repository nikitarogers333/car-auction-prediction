"""
PredictionAgent: produces strict JSON prediction.
Supports: mock LLM, real OpenAI (when key present), and fallback to baseline (nearest_neighbors/regression).
Determinism: temperature=0, top_p=1, fixed system prompt, seed when supported.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from schemas import PREDICTION_METHODS

# Fixed system prompt for determinism
SYSTEM_PROMPT = """You are a car auction price prediction system. You must respond with valid JSON only.
Output format (no other text):
{"predicted_price": <number>, "confidence": <0-1>, "method": "llm_internal", "subgroup_detected": "<subgroup>", "notes": "<max 200 chars>"}
Rules: Use only internal knowledge. Do not use external sources or web. predicted_price must be a number. notes must be under 200 characters."""

USER_PROMPT_TEMPLATE = """Predict auction price for:
Make: {make}
Model: {model}
Year: {year}
Mileage: {mileage}
Subgroup (use this): {subgroup}
Allowed comparables: {allowed_comparables}
Respond with exactly one JSON object."""


class PredictionAgent:
    """Outputs strict JSON matching PredictionOutput schema."""

    def __init__(
        self,
        use_mock: bool | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        model_name: str = "gpt-4o-mini",
        seed: int | None = 42,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.seed = seed
        if use_mock is not None:
            self._use_mock = use_mock
        else:
            self._use_mock = not bool(os.environ.get("OPENAI_API_KEY"))

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce prediction and set payload['prediction']."""
        out = dict(payload)
        if self._use_mock:
            pred = self._mock_predict(out)
        else:
            pred = self._llm_predict(out)
        out["prediction"] = pred
        return out

    def _mock_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Deterministic mock: formula from make/model/year/mileage."""
        make = (payload.get("make") or "").strip() or "Unknown"
        model = (payload.get("model") or "").strip() or "Unknown"
        year = int(payload.get("year") or 2020)
        mileage = int(payload.get("mileage") or 50000)
        subgroup = (payload.get("subgroup") or "generic").strip() or "generic"
        # Simple deterministic formula (no randomness)
        base = 25000
        year_factor = (year - 2015) * 800
        mileage_factor = -0.15 * (mileage - 50000)
        if "M3" in subgroup or "M3" in model:
            base = 55000
        elif "M5" in subgroup or "M5" in model:
            base = 75000
        elif "340i" in subgroup or "340i" in model:
            base = 35000
        price = max(5000, min(200000, base + year_factor + mileage_factor))
        return {
            "predicted_price": round(price, 2),
            "confidence": 0.85,
            "method": "llm_internal",
            "subgroup_detected": subgroup,
            "notes": f"Mock prediction for {make} {model} {year}",
        }

    def _llm_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call OpenAI with strict params; parse JSON from response."""
        try:
            import openai
        except ImportError:
            return self._mock_predict(payload)

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_prompt = USER_PROMPT_TEMPLATE.format(
            make=payload.get("make", ""),
            model=payload.get("model", ""),
            year=payload.get("year", 0),
            mileage=payload.get("mileage") or "N/A",
            subgroup=payload.get("subgroup", "generic"),
            allowed_comparables=payload.get("allowed_comparables", []),
        )
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return {
                "predicted_price": 0,
                "confidence": 0,
                "method": "llm_internal",
                "subgroup_detected": payload.get("subgroup", "generic"),
                "notes": f"LLM error: {str(e)[:180]}",
            }

        parsed = self._extract_json(text)
        if parsed is None:
            return {
                "predicted_price": 0,
                "confidence": 0,
                "method": "llm_internal",
                "subgroup_detected": payload.get("subgroup", "generic"),
                "notes": f"Invalid JSON: {text[:180]}",
            }
        # Ensure required keys and types
        return {
            "predicted_price": float(parsed.get("predicted_price", 0)),
            "confidence": float(parsed.get("confidence", 0)),
            "method": str(parsed.get("method", "llm_internal")) if parsed.get("method") in PREDICTION_METHODS else "llm_internal",
            "subgroup_detected": str(parsed.get("subgroup_detected", payload.get("subgroup", "generic"))),
            "notes": (str(parsed.get("notes", "")) or "")[:200],
        }

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract first JSON object from text."""
        # Try raw parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find {...}
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
