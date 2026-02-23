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
    PREDICTION_METHODS,
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

# System prompt for E5 feature extraction (continuous 0-1 scores, not categories)
FEATURE_SYSTEM_PROMPT = """You are a car auction assessment system. You do NOT predict a price. Instead, you assess the vehicle and output structured features as JSON.

Output format (no other text):
{"condition_score": <integer 1-10>, "market_demand": <0.0-1.0 in 0.1 steps>, "trim_tier": <0.0-1.0 in 0.1 steps>, "depreciation_rate": <0.0-1.0 in 0.1 steps>, "mileage_assessment": <0.0-1.0 in 0.1 steps>, "comparable_market": <0.0-1.0 in 0.1 steps>, "notes": "<max 100 chars>"}

Field definitions:
- condition_score: integer 1 (salvage) to 10 (showroom), based on year, mileage, and typical condition for this model.
- market_demand: 0.0 to 1.0 in 0.1 increments. 0.0 = no demand, 0.3 = below average, 0.5 = average, 0.7 = strong, 1.0 = exceptional.
- trim_tier: 0.0 to 1.0 in 0.1 increments. 0.0 = base/entry, 0.3 = standard, 0.5 = mid-range, 0.7 = premium, 1.0 = flagship.
- depreciation_rate: 0.0 to 1.0 in 0.1 increments. 0.0 = holds value, 0.3 = slow depreciation, 0.5 = average, 0.7 = fast, 1.0 = rapid.
- mileage_assessment: 0.0 to 1.0 in 0.1 increments. 0.0 = very low for age, 0.5 = typical, 1.0 = extremely high.
- comparable_market: 0.0 to 1.0 in 0.1 increments. 0.0 = budget/economy, 0.5 = mainstream, 1.0 = performance/enthusiast.
- notes: brief assessment rationale (max 100 chars).

IMPORTANT: All 0-1 scores must use exactly one decimal place in 0.1 increments (0.0, 0.1, 0.2, ..., 0.9, 1.0). condition_score must be an integer.

Rules: Use only internal knowledge. Do NOT output a price. Output only the JSON object."""

FEATURE_USER_PROMPT_TEMPLATE = """Assess this vehicle (do NOT predict a price, only output feature values):
Make: {make}
Model: {model}
Year: {year}
Mileage: {mileage}
Subgroup: {subgroup}
Respond with exactly one JSON object containing the features."""

RETRY_APPEND = "\n\nPrevious attempt was rejected: {error}. Output corrected JSON only."

# A-prime: free-form assessment prompt (NO schema, NO JSON, NO enforcement)
FREEFORM_SYSTEM_PROMPT = """You are a car auction assessment expert. Write a natural paragraph describing this vehicle's condition and market positioning. Do NOT use JSON or any structured format.

Cover these aspects in your own words:
- The vehicle's overall physical condition (rate it on a 1 to 10 scale, where 1 is salvage and 10 is showroom)
- How strong current market demand is for this specific model
- Where this model sits in its manufacturer's lineup
- How the depreciation rate compares to similar vehicles
- Whether the mileage is typical, below, or above average for a vehicle of this age
- What market segment buyers would compare this car to

Write naturally in plain English. Use only your internal knowledge. Do not use bullet points, tables, or any structured format."""

FREEFORM_USER_PROMPT_TEMPLATE = """Describe the condition and market positioning of this vehicle in plain English:
Make: {make}
Model: {model}
Year: {year}
Mileage: {mileage}
Write a natural assessment paragraph."""

FREEFORM_RETRY_APPEND = "\n\nYour previous response could not be interpreted. Please make sure to clearly mention: a condition rating out of 10, demand level, where the model sits in the lineup, depreciation speed, mileage assessment, and market segment."


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
        """Deterministic mock returning values in 0.1 increments (matching prompt)."""
        model = (payload.get("model") or "").strip()
        year = int(payload.get("year") or 2020)
        mileage = int(payload.get("mileage") or 50000)
        age = max(0, 2025 - year)

        if model in ("M3", "M4", "M5"):
            trim, demand, dep, mkt = 0.9, 0.8, 0.2, 0.9
        elif model in ("X5", "X3", "340i"):
            trim, demand, dep, mkt = 0.7, 0.6, 0.5, 0.7
        else:
            trim, demand, dep, mkt = 0.4, 0.5, 0.5, 0.4

        typical_for_age = max(12000, age * 12000)
        mi = round(max(0.0, min(1.0, mileage / (typical_for_age * 2))) * 10) / 10
        cond = round(max(1.0, min(10.0, 8.0 - age * 0.5 - mileage / 30000)))

        return {
            "condition_score": int(cond),
            "market_demand": demand,
            "trim_tier": trim,
            "depreciation_rate": dep,
            "mileage_assessment": mi,
            "comparable_market": mkt,
            "notes": f"Mock features for {payload.get('make', '')} {model} {year}",
        }

    def _build_feature_system_prompt(self) -> str:
        return FEATURE_SYSTEM_PROMPT

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
            "condition_score": 5,
            "market_demand": 0.5,
            "trim_tier": 0.5,
            "depreciation_rate": 0.5,
            "mileage_assessment": 0.5,
            "comparable_market": 0.5,
            "notes": reason[:100],
            "_validation_error": reason[:200],
        }

    # ----- A-prime: Free-form extraction (no schema enforcement) -----

    def run_freeform_extraction(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A-prime mode: LLM describes car in prose, parser extracts features."""
        out = dict(payload)
        if self._use_mock:
            text = self._mock_freeform_text(out)
        elif self._provider == "claude":
            text = self._llm_freeform_claude(out)
        else:
            text = self._llm_freeform_openai(out)
        features, success = self._parse_freeform_to_features(text)
        features["_parsing_succeeded"] = success
        out["extracted_features"] = features
        out["_freeform_raw_text"] = text
        return out

    def _mock_freeform_text(self, payload: dict[str, Any]) -> str:
        """Generate realistic prose that the parser must interpret."""
        model = (payload.get("model") or "").strip()
        year = int(payload.get("year") or 2020)
        mileage = int(payload.get("mileage") or 50000)
        make = payload.get("make") or "BMW"
        age = max(0, 2025 - year)

        tier = "performance" if model in ("M3", "M4", "M5") else "premium" if model in ("X5", "X3", "340i") else "mid"
        demand = "high" if model in ("M3", "M4", "M5") else "medium"
        dep = "slow" if tier == "performance" else "normal"
        mi = "low" if mileage < 20000 else "average" if mileage < 50000 else "high" if mileage < 80000 else "very high"
        mkt = "performance" if tier == "performance" else "luxury" if tier == "premium" else "mainstream"
        cond = max(1.0, min(10.0, 8.0 - age * 0.5 - mileage / 30000))

        demand_prose = {"low": "fairly limited", "medium": "moderate and steady", "high": "strong and robust"}
        dep_prose = {"slow": "holds its value well and depreciates slowly", "normal": "depreciates at a normal rate", "fast": "depreciates rather quickly"}
        tier_prose = {"base": "entry-level in the lineup", "mid": "mid-range in the lineup", "premium": "upper premium end of the lineup", "performance": "performance tier of the lineup"}
        mkt_prose = {"budget": "budget market segment", "mainstream": "mainstream market segment", "luxury": "luxury market segment", "performance": "performance and enthusiast market segment"}
        mi_prose = {"low": "low relative to its age", "average": "about average for its age", "high": "above average for its age", "very high": "very high for its age"}

        return (
            f"This {year} {make} {model} is in reasonable shape overall. "
            f"I would rate its condition around {cond:.0f} out of 10. "
            f"Market demand for this model is currently {demand_prose.get(demand, demand)}. "
            f"In {make}'s range, this model sits at the {tier_prose.get(tier, tier)}. "
            f"It {dep_prose.get(dep, 'depreciates normally')} compared to peers. "
            f"With {mileage:,} miles, the mileage is {mi_prose.get(mi, mi)} for a {age}-year-old vehicle. "
            f"Buyers shopping for this car typically compare it to vehicles in the {mkt_prose.get(mkt, mkt)}."
        )

    def _build_freeform_user_prompt(self, payload: dict[str, Any]) -> str:
        prompt = FREEFORM_USER_PROMPT_TEMPLATE.format(
            make=payload.get("make", ""),
            model=payload.get("model", ""),
            year=payload.get("year", 0),
            mileage=payload.get("mileage") or "N/A",
        )
        retry_err = payload.get("_validation_error_from_previous_attempt")
        if retry_err:
            prompt += FREEFORM_RETRY_APPEND
        return prompt

    def _llm_freeform_openai(self, payload: dict[str, Any]) -> str:
        try:
            import openai
        except ImportError:
            return self._mock_freeform_text(payload)
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user = self._build_freeform_user_prompt(payload)
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": FREEFORM_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Error: {e}"

    def _llm_freeform_claude(self, payload: dict[str, Any]) -> str:
        try:
            from anthropic import Anthropic
        except ImportError:
            return self._mock_freeform_text(payload)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return self._mock_freeform_text(payload)
        client = Anthropic(api_key=api_key)
        user = self._build_freeform_user_prompt(payload)
        try:
            resp = client.messages.create(
                model=self.model_name,
                max_tokens=512,
                system=FREEFORM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                temperature=self.temperature,
            )
            text = ""
            for block in (resp.content or []):
                if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                    text += block.text
            return text.strip()
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _parse_freeform_to_features(text: str) -> tuple[dict[str, Any], bool]:
        """Parse free-form prose into feature dict for pricing formula.
        Returns (features_dict, parsing_succeeded).
        """
        lower = text.lower()
        features: dict[str, Any] = {}
        found_count = 0

        cond = None
        for pat in [
            r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10',
            r'condition[^.]{0,40}?(\d+(?:\.\d+)?)',
            r'(?:rate|score|grade)[^.]{0,30}?(\d+(?:\.\d+)?)',
        ]:
            m = re.search(pat, lower)
            if m:
                val = float(m.group(1))
                if 1 <= val <= 10:
                    cond = val
                    break
        features["condition_score"] = cond if cond is not None else 5.0
        if cond is not None:
            found_count += 1

        val, ok = PredictionAgent._match_near(
            lower, ["demand"],
            {"low": "low", "weak": "low", "limited": "low",
             "moderate": "medium", "medium": "medium", "average": "medium", "steady": "medium",
             "high": "high", "strong": "high", "robust": "high", "significant": "high"},
            "medium")
        features["market_demand"] = val
        found_count += ok

        val, ok = PredictionAgent._match_near(
            lower, ["tier", "trim", "lineup", "range", "positioning"],
            {"base": "base", "entry": "base", "entry-level": "base", "basic": "base",
             "mid": "mid", "middle": "mid", "mid-range": "mid", "midrange": "mid",
             "premium": "premium", "upper": "premium", "upmarket": "premium",
             "performance": "performance", "top-tier": "performance", "flagship": "performance",
             "sport": "performance", "high-performance": "performance"},
            "mid")
        features["trim_tier"] = val
        found_count += ok

        val, ok = PredictionAgent._match_near(
            lower, ["depreciat", "value retention", "resale", "holds"],
            {"slow": "slow", "slowly": "slow", "gradual": "slow", "well": "slow", "retain": "slow",
             "normal": "normal", "average": "normal", "moderate": "normal", "typical": "normal",
             "fast": "fast", "rapid": "fast", "steep": "fast", "quickly": "fast"},
            "normal")
        features["depreciation_rate"] = val
        found_count += ok

        val, ok = PredictionAgent._match_near(
            lower, ["mileage", "miles", "odometer"],
            {"very high": "very_high", "extremely high": "very_high", "excessive": "very_high",
             "high": "high", "above average": "high", "elevated": "high",
             "low": "low", "below": "low", "minimal": "low", "light": "low",
             "average": "average", "typical": "average", "moderate": "average", "normal": "average"},
            "average")
        features["mileage_assessment"] = val
        found_count += ok

        val, ok = PredictionAgent._match_near(
            lower, ["market", "segment", "compet", "compar", "category", "class", "shopping"],
            {"budget": "budget", "economy": "budget", "affordable": "budget",
             "mainstream": "mainstream", "mid-market": "mainstream", "mass market": "mainstream",
             "luxury": "luxury", "upscale": "luxury",
             "performance": "performance", "enthusiast": "performance", "sport": "performance"},
            "mainstream")
        features["comparable_market"] = val
        found_count += ok

        features["notes"] = "Parsed from free-form LLM text"
        success = found_count >= 3

        return features, success

    @staticmethod
    def _match_near(text: str, context_words: list[str], value_map: dict[str, str], default: str) -> tuple[str, bool]:
        """Find a value from value_map in sentences containing context words.
        Returns (matched_value, was_found)."""
        sentences = re.split(r'[.!?\n]', text)
        relevant = [s for s in sentences if any(cw in s for cw in context_words)]
        search = " ".join(relevant) if relevant else text
        for key in sorted(value_map.keys(), key=len, reverse=True):
            if key in search:
                return value_map[key], True
        return default, False

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
