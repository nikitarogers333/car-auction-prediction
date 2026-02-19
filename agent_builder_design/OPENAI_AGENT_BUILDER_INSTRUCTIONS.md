# How to Implement in OpenAI Agent Builder Canvas

## 1. Jailbreak Guard

- Add a **Guard** or **Validation** node after the LLM response.
- Configure it to:
  - Only allow responses that are valid JSON.
  - Reject if the JSON does not contain exactly: `predicted_price`, `confidence`, `method`, `subgroup_detected`, `notes`.
  - Reject if `subgroup_detected` is not in the precomputed `allowed_comparables` for this vehicle.
- On rejection: set `valid: false`, `violation_reason: "jailbreak or invalid format"`, and do not pass to scoring.

## 2. Classification Agent

- Create a **Classification** or **Router** agent that takes `make` and `model`.
- Use **If/Else** logic (or a small lookup table):
  - If make=BMW and model contains "M3" → subgroup = M3, allowed_comparables = [M3, M4].
  - If make=BMW and model contains "340i" → subgroup = 340i, allowed_comparables = [340i, 330i, 328i].
  - Else → subgroup = generic, allowed_comparables = [generic].
- Output: append `subgroup` and `allowed_comparables` to the payload for the next step.

## 3. If/Else Node

- **Condition 1:** make == "BMW" AND model contains "M3" → set subgroup = M3.
- **Condition 2:** make == "BMW" AND model contains "340i" → set subgroup = 340i.
- **Default:** subgroup = generic.
- Each branch sets the same payload keys so the rest of the flow is unchanged.

## 4. Specialist Prediction Agent

- Single **LLM** node used only for price prediction.
- **System prompt:** Fixed. Example: "You are a car auction price prediction system. Respond with valid JSON only: {\"predicted_price\": <number>, \"confidence\": <0-1>, \"method\": \"llm_internal\", \"subgroup_detected\": \"<subgroup>\", \"notes\": \"<max 200 chars>\"}. Use only internal knowledge. Do not use external sources."
- **User prompt:** Template with make, model, year, mileage, subgroup, allowed_comparables.
- **Parameters:** temperature=0, top_p=1, seed=42 (if supported).
- **Output parsing:** Parse JSON from completion; if parsing fails, set valid=false.

## 5. Hallucination Guard

- After the Prediction agent, add a **Validation** step that checks:
  - No occurrence of VIN in the response text or notes.
  - No references to forbidden comparables (e.g. "340i" when subgroup is M3).
  - `predicted_price` is within configured min/max for that make/model.
- If any check fails: set valid=false, violation_reason="hallucination or out-of-bounds".

## 6. Stop Verification Hook

- Configure a **post-completion** or **Stop** hook that runs before the run is marked complete.
- The hook receives the final payload (with prediction and valid flag).
- Logic: If valid==false, or if a custom condition (e.g. "run tests") fails, the hook returns **deny** so that the run is not considered successful.
- In Agent Builder this may be implemented as a "Workflow end condition" or "Verification step" that must return success before the run ends.

## 7. Determinism Checklist

- Set **temperature** = 0 for the prediction LLM.
- Set **top_p** = 1.
- Use a **fixed system prompt** (no dynamic injection of user-controlled content into system prompt).
- Set **seed** if the API supports it.
- Do not allow external tools (web search, browser) unless in an explicit "P2" or "P3" experiment; for P1/P4 disable all external tools.

## 8. Restriction Enforcement (No LLM)

- **VIN leakage:** Remove VIN from the payload before sending anything to the LLM; never include VIN in the user or system prompt in the prediction path.
- **Subgroup isolation:** Computed in the Classification step; the Hallucination guard then enforces that the LLM’s `subgroup_detected` is in `allowed_comparables`.
- **Domain allowlist:** If you enable web search (P2/P3), configure the tool to only allow specific domains (e.g. kbb.com, edmunds.com); block all others in the tool configuration.
