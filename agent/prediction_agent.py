"""
PredictionAgent: produces strict JSON prediction.
Supports: mock LLM, OpenAI, Claude. Determinism: temperature=0, top_p=1, fixed system prompt, seed when supported.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from schemas import (
    DEMAND_LEVELS,
    DEPRECIATION_RATES,
    MARKET_SEGMENTS,
    MILEAGE_ASSESSMENTS,
    PREDICTION_METHODS,
    TRIM_TIERS,
    validate_feature_output,
)

# Fixed system prompt for price prediction (E0-E4)
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

# System prompt for E5 feature extraction (LLM outputs features, not price)
FEATURE_SYSTEM_PROMPT = """You are a car auction assessment system. You do NOT predict a price. Instead, you assess the vehicle and output structured features as JSON.

Output format (no other text):
{{"condition_score": <1-10>, "market_demand": "<low|medium|high>", "trim_tier": "<base|mid|premium|performance>", "depreciation_rate": "<slow|normal|fast>", "mileage_assessment": "<low|average|high|very_high>", "comparable_market": "<budget|mainstream|luxury|performance>", "notes": "<max 100 chars>"}}

Field definitions:
- condition_score: 1 (salvage) to 10 (showroom), based on year, mileage, and typical condition for this model.
- market_demand: current market demand for this specific model/year.
- trim_tier: where this model sits in its manufacturer lineup.
- depreciation_rate: how fast this model depreciates relative to peers.
- mileage_assessment: how this mileage compares to typical for the age.
- comparable_market: the market segment buyers compare this car to.
- notes: brief assessment rationale (max 100 chars).

Allowed values:
- trim_tier: {trim_tiers}
- market_demand: {demand_levels}
- depreciation_rate: {dep_rates}
- mileage_assessment: {mileage_assess}
- comparable_market: {market_segments}

Rules: Use only internal knowledge. Do NOT output a price. Output only the JSON object."""

FEATURE_USER_PROMPT_TEMPLATE = """Assess this vehicle (do NOT predict a price, only output feature values):
Make: {make}
Model: {model}
Year: {year}
Mileage: {mileage}
Subgroup: {subgroup}
Respond with exactly one JSON object containing the features."""

RETRY_APPEND = "\n\nPrevious attempt was rejected: {error}. Output corrected JSON only."


class PredictionAgent:
    """Outputs strict JSON matching PredictionOutput schema. Supports provider='openai' or 'claude'."""

    def __init__(
        self,
        use_mock: bool | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        model_name: str = "gpt-4o-mini",
        seed: int | None = 42,
        provider: str = "openai",
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.seed = seed
        self._provider = (provider or "openai").lower()
        if use_mock is not None:
            self._use_mock = use_mock
        else:
            has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
            has_claude = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
            self._use_mock = not (has_openai or has_claude)

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce prediction and set payload['prediction']."""
        out = dict(payload)
        if self._use_mock:
            pred = self._mock_predict(out)
        elif self._provider == "claude":
            pred = self._llm_predict_claude(out)
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

    def _build_user_prompt(self, payload: dict[str, Any]) -> str:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            make=payload.get("make", ""),
            model=payload.get("model", ""),
            year=payload.get("year", 0),
            mileage=payload.get("mileage") or "N/A",
            subgroup=payload.get("subgroup", "generic"),
            allowed_comparables=payload.get("allowed_comparables", []),
        )
        retry_err = payload.get("_validation_error_from_previous_attempt")
        if retry_err:
            user_prompt += RETRY_APPEND.format(error=retry_err)
        return user_prompt

    def _llm_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call OpenAI with strict params; parse JSON from response."""
        try:
            import openai
        except ImportError:
            return self._mock_predict(payload)

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_prompt = self._build_user_prompt(payload)
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

        return self._parse_llm_response(text, payload)

    def _llm_predict_claude(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call Anthropic Claude with system + user message; parse JSON from response."""
        try:
            from anthropic import Anthropic
        except ImportError:
            return self._mock_predict(payload)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return self._mock_predict(payload)

        client = Anthropic(api_key=api_key)
        user_prompt = self._build_user_prompt(payload)
        try:
            resp = client.messages.create(
                model=self.model_name,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.temperature,
            )
            text = ""
            for block in (resp.content or []):
                if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                    text += block.text
            text = text.strip()
        except Exception as e:
            return {
                "predicted_price": 0,
                "confidence": 0,
                "method": "llm_internal",
                "subgroup_detected": payload.get("subgroup", "generic"),
                "notes": f"Claude error: {str(e)[:180]}",
            }
        if not text:
            return {
                "predicted_price": 0,
                "confidence": 0,
                "method": "llm_internal",
                "subgroup_detected": payload.get("subgroup", "generic"),
                "notes": "Claude returned empty response",
            }
        return self._parse_llm_response(text, payload)

    def _parse_llm_response(self, text: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON from LLM text and return normalized prediction dict."""
        parsed = self._extract_json(text)
        if parsed is None:
            return {
                "predicted_price": 0,
                "confidence": 0,
                "method": "llm_internal",
                "subgroup_detected": payload.get("subgroup", "generic"),
                "notes": f"Invalid JSON: {text[:180]}",
            }
        return {
            "predicted_price": float(parsed.get("predicted_price", 0)),
            "confidence": float(parsed.get("confidence", 0)),
            "method": str(parsed.get("method", "llm_internal")) if parsed.get("method") in PREDICTION_METHODS else "llm_internal",
            "subgroup_detected": str(parsed.get("subgroup_detected", payload.get("subgroup", "generic"))),
            "notes": (str(parsed.get("notes", "")) or "")[:200],
        }

    # ----- E5: Feature extraction mode -----

    def run_feature_extraction(self, payload: dict[str, Any]) -> dict[str, Any]:
        """E5 mode: extract features (no price). Sets payload['extracted_features']."""
        out = dict(payload)
        if self._use_mock:
            features = self._mock_features(out)
        elif self._provider == "claude":
            features = self._llm_extract_features_claude(out)
        else:
            features = self._llm_extract_features_openai(out)
        out["extracted_features"] = features
        return out

    def _mock_features(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = (payload.get("model") or "").strip()
        year = int(payload.get("year") or 2020)
        mileage = int(payload.get("mileage") or 50000)
        age = max(0, 2025 - year)
        tier = "performance" if model in ("M3", "M4", "M5") else "premium" if model in ("X5", "X3", "340i") else "mid"
        demand = "high" if model in ("M3", "M4", "M5") else "medium"
        dep = "slow" if tier == "performance" else "normal"
        mi = "low" if mileage < 20000 else "average" if mileage < 50000 else "high" if mileage < 80000 else "very_high"
        mkt = "performance" if tier == "performance" else "luxury" if tier == "premium" else "mainstream"
        cond = max(1.0, min(10.0, 8.0 - age * 0.5 - mileage / 30000))
        return {
            "condition_score": round(cond, 1),
            "market_demand": demand,
            "trim_tier": tier,
            "depreciation_rate": dep,
            "mileage_assessment": mi,
            "comparable_market": mkt,
            "notes": f"Mock features for {payload.get('make', '')} {model} {year}",
        }

    def _build_feature_system_prompt(self) -> str:
        return FEATURE_SYSTEM_PROMPT.format(
            trim_tiers=", ".join(TRIM_TIERS),
            demand_levels=", ".join(DEMAND_LEVELS),
            dep_rates=", ".join(DEPRECIATION_RATES),
            mileage_assess=", ".join(MILEAGE_ASSESSMENTS),
            market_segments=", ".join(MARKET_SEGMENTS),
        )

    def _build_feature_user_prompt(self, payload: dict[str, Any]) -> str:
        prompt = FEATURE_USER_PROMPT_TEMPLATE.format(
            make=payload.get("make", ""),
            model=payload.get("model", ""),
            year=payload.get("year", 0),
            mileage=payload.get("mileage") or "N/A",
            subgroup=payload.get("subgroup", "generic"),
        )
        retry_err = payload.get("_validation_error_from_previous_attempt")
        if retry_err:
            prompt += RETRY_APPEND.format(error=retry_err)
        return prompt

    def _llm_extract_features_openai(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import openai
        except ImportError:
            return self._mock_features(payload)
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        system = self._build_feature_system_prompt()
        user = self._build_feature_user_prompt(payload)
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return self._feature_error_fallback(f"OpenAI error: {e}")
        return self._parse_feature_response(text)

    def _llm_extract_features_claude(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            from anthropic import Anthropic
        except ImportError:
            return self._mock_features(payload)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return self._mock_features(payload)
        client = Anthropic(api_key=api_key)
        system = self._build_feature_system_prompt()
        user = self._build_feature_user_prompt(payload)
        try:
            resp = client.messages.create(
                model=self.model_name,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=self.temperature,
            )
            text = ""
            for block in (resp.content or []):
                if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                    text += block.text
            text = text.strip()
        except Exception as e:
            return self._feature_error_fallback(f"Claude error: {e}")
        if not text:
            return self._feature_error_fallback("Claude returned empty response")
        return self._parse_feature_response(text)

    def _parse_feature_response(self, text: str) -> dict[str, Any]:
        parsed = self._extract_json(text)
        if parsed is None:
            return self._feature_error_fallback(f"Invalid JSON: {text[:120]}")
        valid, err = validate_feature_output(parsed)
        if not valid:
            parsed["_validation_error"] = err
        return parsed

    @staticmethod
    def _feature_error_fallback(reason: str) -> dict[str, Any]:
        return {
            "condition_score": 5.0,
            "market_demand": "medium",
            "trim_tier": "mid",
            "depreciation_rate": "normal",
            "mileage_assessment": "average",
            "comparable_market": "mainstream",
            "notes": reason[:100],
            "_validation_error": reason[:200],
        }

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract first JSON object from text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
