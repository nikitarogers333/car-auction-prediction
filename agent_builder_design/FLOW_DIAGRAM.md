# Flow Diagram: Deterministic Car Auction Price Prediction

## High-Level Flow

```
Start
  → PreValidationGuard (strip VIN/target, validate make/model)
  → FeatureExtractionAgent (normalize features)
  → SubgroupClassifier (e.g. M3 vs 340i; set allowed_comparables)
  → RestrictionEnforcer (price bounds, rules)
  → [PrePredictionHook] — optional allow/deny/modify
  → PredictionAgent (LLM or mock → strict JSON)
  → [PostPredictionHook] — optional allow/deny/modify
  → PostValidationGuard (schema, subgroup leakage, price bounds)
  → ScoringModule (MAE, MAPE, variance)
  → Logging (audit trail)
  → End
```

## Node Definitions for Agent Builder Canvas

### 1. PreValidationGuard (Classification / Guard node)
- **Input:** Raw vehicle record (make, model, year, mileage, optional VIN/price).
- **Output:** Structured payload with `vehicle_id`, `make`, `model`, `year`, `mileage`, `features`; `vin` and `target` removed.
- **Branch:** If `make` or `model` missing → mark invalid and short-circuit.

### 2. SubgroupClassifier (If/Else + Classification)
- **Input:** Payload with make/model.
- **Logic:** If make=BMW and model contains "M3" → subgroup=M3, allowed=[M3,M4]; if "340i" → subgroup=340i, allowed=[340i,330i,328i]; else generic.
- **Output:** Same payload + `subgroup`, `allowed_comparables`, `_forbidden_comparables`.

### 3. RestrictionEnforcer (Code / Rule node)
- **Input:** Payload.
- **Logic:** Load price_bounds; no external call; append "restriction_enforcer" to restrictions_applied.
- **Output:** Same payload.

### 4. PredictionAgent (LLM / Specialist)
- **Input:** Payload with make, model, year, mileage, subgroup, allowed_comparables.
- **System prompt:** Fixed (internal knowledge only; output strict JSON).
- **Output schema:** `{ predicted_price, confidence, method, subgroup_detected, notes }`.
- **Parameters:** temperature=0, top_p=1, seed=42.

### 5. PostValidationGuard (Jailbreak / Hallucination guard)
- **Input:** Payload with `prediction`.
- **Checks:** JSON schema valid; subgroup_detected in allowed_comparables; no VIN in output; price in bounds.
- **Output:** Payload with `valid` true/false, `violation_reason` if false.

### 6. StopHook (Verification hook)
- **When:** After PostValidationGuard, before marking run complete.
- **Logic:** If tests fail or custom rule fails → deny (block stop).
- **Implementation in Builder:** Use “Stop verification” / post-step check that must pass before ending.

## Branch Conditions

| From Node           | Condition              | To Node              |
|--------------------|------------------------|----------------------|
| PreValidationGuard | valid == false         | Logging (invalid) / End |
| PreValidationGuard | valid == true          | FeatureExtraction    |
| PostValidationGuard| valid == false         | Logging (violation)  |
| PostValidationGuard| valid == true          | ScoringModule → Logging |
| PrePredictionHook  | decision == DENY       | Logging (denied)     |
| StopHook           | decision == DENY       | Do not end; retry or flag |

## Guardrail Logic (Summary)

- **Jailbreak guard:** PostValidationGuard rejects if output format is not strict JSON or if forbidden subgroup appears.
- **Classification agent:** SubgroupClassifier assigns subgroup and allowed comparables from rules.
- **If/Else node:** SubgroupClassifier is implemented as if/else on make + model pattern.
- **Specialist prediction agent:** PredictionAgent is the only LLM step; single fixed system prompt.
- **Hallucination guard:** PostValidationGuard ensures no VIN in output, no out-of-bounds price, no forbidden subgroup.
- **Stop verification hook:** StopHook runs after pipeline; if it returns DENY, run is not considered successful.
