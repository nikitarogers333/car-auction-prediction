"""
Deterministic agent pipeline: strict flow with JSON between steps.
Start → PreValidationGuard → FeatureExtraction → SubgroupClassifier → RestrictionEnforcer
→ [PrePredictionHook] → PredictionAgent → [PostPredictionHook] → PostValidationGuard
→ (Scoring/Logging done by caller)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent import (
    FeatureExtractionAgent,
    PostValidationGuard,
    PreValidationGuard,
    PredictionAgent,
    RestrictionEnforcer,
    SubgroupClassifier,
)
from config import (
    COMPARISON_TEMPERATURE,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    ENFORCEMENT_LEVELS,
    MAX_VALIDATION_RETRIES,
    PROVIDERS,
)
from hooks import HookDecision, PostPredictionHook, PrePredictionHook, StopHook
from pricing import compute_price_from_features
from schemas import validate_feature_output, validate_prediction_output


def run_pipeline(
    raw_input: dict[str, Any],
    condition_id: str = "P1",
    use_mock_llm: bool | None = None,
    project_root: Path | None = None,
    pre_hook: PrePredictionHook | None = None,
    post_hook: PostPredictionHook | None = None,
    stop_hook: StopHook | None = None,
    provider_id: str = "openai",
    enforcement_level: str = "E2",
    temperature_override: float | None = None,
) -> dict[str, Any]:
    """
    Single run through the pipeline. Returns final payload.
    enforcement_level E5: LLM extracts features, code computes price.
    temperature_override: if set, overrides config temperature (used by compare.py).
    """
    root = project_root or Path(__file__).resolve().parent
    level = (enforcement_level or "E2").upper()
    if level not in ENFORCEMENT_LEVELS:
        level = "E2"

    pre_guard = PreValidationGuard(allow_vin_in_context=False)
    feat_agent = FeatureExtractionAgent()
    subgroup = SubgroupClassifier(root)
    enforcer = RestrictionEnforcer(root)
    provider = (provider_id or "openai").lower()
    if provider not in PROVIDERS:
        provider = "openai"
    model_name = DEFAULT_CLAUDE_MODEL if provider == "claude" else DEFAULT_MODEL
    temp = temperature_override if temperature_override is not None else DEFAULT_TEMPERATURE
    pred_agent = PredictionAgent(
        use_mock=use_mock_llm,
        temperature=temp,
        top_p=DEFAULT_TOP_P,
        model_name=model_name,
        seed=DEFAULT_SEED if temp == 0.0 else None,
        provider=provider,
    )
    post_guard = PostValidationGuard(root)

    payload = pre_guard.run(raw_input)
    if not payload.get("valid", True):
        return payload
    payload = feat_agent.run(payload)
    payload = subgroup.run(payload)
    payload = enforcer.run(payload)

    if pre_hook:
        decision, modified = pre_hook.run(payload)
        if decision == HookDecision.DENY:
            payload["valid"] = False
            payload["violation_reason"] = "PrePredictionHook denied"
            return payload
        if decision == HookDecision.MODIFY and modified is not None:
            payload = modified

    # --- E5 path: feature extraction + deterministic pricing ---
    if level == "E5":
        return _run_e5(payload, pred_agent, stop_hook)

    # --- E0-E4 path: LLM predicts price directly ---
    retry_count = 0
    max_retries = MAX_VALIDATION_RETRIES if level in ("E3", "E4") else 0

    while True:
        payload.pop("_validation_error_from_previous_attempt", None)
        payload = pred_agent.run(payload)

        if post_hook:
            decision, modified = post_hook.run(payload)
            if decision == HookDecision.DENY:
                payload["valid"] = False
                payload["violation_reason"] = "PostPredictionHook denied"
                payload["retry_count"] = retry_count
                return payload
            if decision == HookDecision.MODIFY and modified is not None:
                payload = modified

        if level == "E0":
            pred = payload.get("prediction") or {}
            payload["valid"] = (
                isinstance(pred.get("predicted_price"), (int, float))
                and float(pred.get("predicted_price", 0)) >= 0
            )
            payload["violation_reason"] = None if payload["valid"] else "E0: no enforcement (parseable check only)"
            payload["retry_count"] = retry_count
            if stop_hook:
                decision, _ = stop_hook.run(payload)
                if decision == HookDecision.DENY:
                    payload["valid"] = False
                    payload["violation_reason"] = payload.get("violation_reason") or "StopHook denied"
            return payload

        if level == "E1":
            pred = payload.get("prediction")
            if not pred:
                payload["valid"] = False
                payload["violation_reason"] = "missing prediction"
            else:
                valid, msg = validate_prediction_output(pred)
                payload["valid"] = valid
                payload["violation_reason"] = None if valid else msg
            payload["retry_count"] = retry_count
            if stop_hook:
                decision, _ = stop_hook.run(payload)
                if decision == HookDecision.DENY:
                    payload["valid"] = False
                    payload["violation_reason"] = payload.get("violation_reason") or "StopHook denied"
            return payload

        payload = post_guard.run(payload)

        if payload.get("valid", False):
            payload["retry_count"] = retry_count
            if stop_hook:
                decision, _ = stop_hook.run(payload)
                if decision == HookDecision.DENY:
                    payload["valid"] = False
                    payload["violation_reason"] = payload.get("violation_reason") or "StopHook denied"
            return payload

        if retry_count >= max_retries:
            payload["retry_count"] = retry_count
            if stop_hook:
                decision, _ = stop_hook.run(payload)
                if decision == HookDecision.DENY:
                    payload["violation_reason"] = payload.get("violation_reason") or "StopHook denied"
            return payload

        retry_count += 1
        payload["_validation_error_from_previous_attempt"] = payload.get("violation_reason") or "validation failed"

    return payload


def _run_e5(
    payload: dict[str, Any],
    pred_agent: PredictionAgent,
    stop_hook: StopHook | None,
) -> dict[str, Any]:
    """E5: LLM extracts features -> validate -> deterministic pricing."""
    retry_count = 0
    max_retries = MAX_VALIDATION_RETRIES

    while True:
        payload.pop("_validation_error_from_previous_attempt", None)
        payload = pred_agent.run_feature_extraction(payload)
        features = payload.get("extracted_features") or {}

        feat_err = features.pop("_validation_error", None)
        if feat_err is None:
            valid, feat_err = validate_feature_output(features)
        else:
            valid = False

        if valid:
            year = int(payload.get("year") or 2020)
            mileage = int(payload.get("mileage") or 50000)
            price = compute_price_from_features(features, year, mileage)
            payload["prediction"] = {
                "predicted_price": price,
                "confidence": 0.90,
                "method": "feature_extraction",
                "subgroup_detected": payload.get("subgroup", "generic"),
                "notes": f"E5 deterministic from features",
            }
            payload["valid"] = True
            payload["violation_reason"] = None
            payload["retry_count"] = retry_count
            if stop_hook:
                decision, _ = stop_hook.run(payload)
                if decision == HookDecision.DENY:
                    payload["valid"] = False
                    payload["violation_reason"] = "StopHook denied"
            return payload

        if retry_count >= max_retries:
            payload["prediction"] = {
                "predicted_price": 0,
                "confidence": 0,
                "method": "feature_extraction",
                "subgroup_detected": payload.get("subgroup", "generic"),
                "notes": f"E5 feature validation failed after {retry_count} retries: {str(feat_err)[:120]}",
            }
            payload["valid"] = False
            payload["violation_reason"] = f"E5 feature validation: {feat_err}"
            payload["retry_count"] = retry_count
            return payload

        retry_count += 1
        payload["_validation_error_from_previous_attempt"] = feat_err or "feature validation failed"


def run_consistency_check(
    raw_input: dict[str, Any],
    n_repeats: int,
    condition_id: str = "P1",
    use_mock_llm: bool | None = None,
    project_root: Path | None = None,
    pre_hook: PrePredictionHook | None = None,
    post_hook: PostPredictionHook | None = None,
    stop_hook: StopHook | None = None,
    provider_id: str = "openai",
    enforcement_level: str = "E2",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run prediction N times; return (list of payloads, variance_stats)."""
    import os
    from scoring import variance_stats
    results: list[dict[str, Any]] = []
    prices: list[float] = []
    for i in range(n_repeats):
        # Force mock if no key so repeats are deterministic
        use_mock = use_mock_llm if use_mock_llm is not None else not bool(os.environ.get("OPENAI_API_KEY"))
        payload = run_pipeline(
            raw_input,
            condition_id=condition_id,
            use_mock_llm=use_mock,
            project_root=project_root,
            pre_hook=pre_hook,
            post_hook=post_hook,
            stop_hook=stop_hook,
            provider_id=provider_id,
            enforcement_level=enforcement_level,
        )
        results.append(payload)
        pred = payload.get("prediction") or {}
        p = pred.get("predicted_price")
        if p is not None and payload.get("valid"):
            prices.append(float(p))
    vs = variance_stats(prices) if prices else {"mean": 0.0, "std": 0.0, "cv": 0.0, "n": 0}
    vs["invalid_rate"] = sum(1 for r in results if not r.get("valid")) / len(results) if results else 0.0
    return results, vs
