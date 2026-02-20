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
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    PROVIDERS,
)
from hooks import HookDecision, PostPredictionHook, PrePredictionHook, StopHook


def run_pipeline(
    raw_input: dict[str, Any],
    condition_id: str = "P1",
    use_mock_llm: bool | None = None,
    project_root: Path | None = None,
    pre_hook: PrePredictionHook | None = None,
    post_hook: PostPredictionHook | None = None,
    stop_hook: StopHook | None = None,
    provider_id: str = "openai",
) -> dict[str, Any]:
    """
    Single run through the pipeline. Returns final payload (with valid, prediction, etc.).
    provider_id: "openai" or "claude" to choose which API to call.
    """
    root = project_root or Path(__file__).resolve().parent
    pre_guard = PreValidationGuard(allow_vin_in_context=False)
    feat_agent = FeatureExtractionAgent()
    subgroup = SubgroupClassifier(root)
    enforcer = RestrictionEnforcer(root)
    provider = (provider_id or "openai").lower()
    if provider not in PROVIDERS:
        provider = "openai"
    model_name = DEFAULT_CLAUDE_MODEL if provider == "claude" else DEFAULT_MODEL
    pred_agent = PredictionAgent(
        use_mock=use_mock_llm,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        model_name=model_name,
        seed=DEFAULT_SEED,
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

    payload = pred_agent.run(payload)

    if post_hook:
        decision, modified = post_hook.run(payload)
        if decision == HookDecision.DENY:
            payload["valid"] = False
            payload["violation_reason"] = "PostPredictionHook denied"
            return payload
        if decision == HookDecision.MODIFY and modified is not None:
            payload = modified

    payload = post_guard.run(payload)

    if stop_hook:
        decision, _ = stop_hook.run(payload)
        if decision == HookDecision.DENY:
            payload["valid"] = False
            payload["violation_reason"] = payload.get("violation_reason") or "StopHook denied"
    return payload


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
        )
        results.append(payload)
        pred = payload.get("prediction") or {}
        p = pred.get("predicted_price")
        if p is not None and payload.get("valid"):
            prices.append(float(p))
    vs = variance_stats(prices) if prices else {"mean": 0.0, "std": 0.0, "cv": 0.0, "n": 0}
    vs["invalid_rate"] = sum(1 for r in results if not r.get("valid")) / len(results) if results else 0.0
    return results, vs
