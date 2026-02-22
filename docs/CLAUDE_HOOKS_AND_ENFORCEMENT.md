# Claude Code Hooks and Our Enforcement Model

This doc maps our **Python hook system** and **validation pipeline** to **Claude Code Hooks** and to the “deterministic under constraints” stack (structured output + validation + deterministic settings).

---

## What Claude Code Hooks Are

From [code.claude.com](https://code.claude.com/docs/en/hooks-guide) and [aiorg.dev](https://aiorg.dev/blog/claude-code-hooks):

- **Hooks** run at fixed points in a workflow (e.g. before/after a tool, at session start/stop).
- They can **block** (deny), **allow**, or **modify** what happens next.
- That gives **deterministic control** over an otherwise probabilistic system: we enforce rules in code, not only in prompts.

So: same idea we use—run code at specific steps to allow/deny/modify.

---

## How Our Hooks Map to Claude Code Hooks

| Our hook | When it runs | Claude analogue | Purpose |
|----------|--------------|------------------|---------|
| **PrePredictionHook** | After RestrictionEnforcer, before PredictionAgent | PreToolUse / “before LLM call” | Allow/deny/modify the payload before the model predicts. E.g. enforce “no external data” or inject constraints. |
| **PostPredictionHook** | After PredictionAgent, before PostValidationGuard | PostToolUse / “after LLM call” | Allow/deny/modify the raw prediction. E.g. redact forbidden fields or reject if a rule is broken. |
| **StopHook** | At end of pipeline | Stop / “before run is considered complete” | Final gate: deny so the run is not accepted even if earlier steps passed. |

Our hooks take a **context** (dict) and return:

- **ALLOW** — continue as-is  
- **DENY** — mark run invalid, set `violation_reason`, return  
- **MODIFY** — replace context with a new dict and continue  

That matches the idea of “exit 0 = proceed, exit 2 = block” in Claude; we use an enum instead of process exit.

---

## The Full “Deterministic Under Constraints” Stack (What We Use)

1. **Constrain the output format**  
   The LLM is instructed and validated to return **strict JSON** (predicted_price, confidence, method, subgroup_detected, notes). Our **PostValidationGuard** checks schema and business rules. Invalid format → `valid=False`.

2. **Deterministic decode settings**  
   We use `temperature=0`, fixed `top_p`, and a **seed** where supported (see `config.py`). That reduces sampling variance across runs.

3. **Hard validation gate**  
   After the LLM responds:
   - **JSON schema** + **custom rules** (subgroup in allowlist, price in bounds, no VIN, no forbidden sources) run in **code**.
   - If any check fails → we **reject** (never accept). Optional: in the future we could “regenerate with error feedback”; for now we only reject.

4. **Hook-style enforcement**  
   PrePredictionHook, PostPredictionHook, StopHook run at fixed points. They can deny or modify so that the pipeline never “accepts” an output that violates policy—even if the LLM returned something that would otherwise pass schema.

So: **structured output + deterministic settings + verifier (no accept on failure) + hooks** = our implementation of “deterministic, constrained behavior” for the car auction price use case.

---

**Enforcement ladder (E0–E4):** E0 = accept if parseable; E1 = schema only; E2 = schema + validation gate; E3 = E2 + retry/repair; E4 = E3 + verifier. See `config.ENFORCEMENT_LEVELS` and `pipeline.run_pipeline(..., enforcement_level=...)`. Metrics: valid rate, retries per request, CV.

## Using the Hooks in This Repo

- **Defined in:** `hooks.py` (PrePredictionHook, PostPredictionHook, StopHook, HookDecision).
- **Wired in:** `pipeline.run_pipeline()` and `pipeline.run_consistency_check()` accept optional `pre_hook`, `post_hook`, `stop_hook`. If you pass custom hook instances, they run at the points above.
- **Default:** If you don’t pass hooks, the pipeline uses no-op hooks (always ALLOW). So all enforcement is currently in PreValidationGuard, RestrictionEnforcer, and PostValidationGuard; hooks are the extension point for “Claude-style” deterministic checks.
- **Web app:** The Flask UI has a **Use enforcement hooks** checkbox. When enabled, a PostPredictionHook (confidence floor 0.3) runs: any prediction with confidence &lt; 0.3 is rejected. Compare valid rate and variance with vs without hooks (see `hooks.make_confidence_floor_hook`).

For research, you can implement custom hooks (e.g. “deny if payload mentions a forbidden domain” or “modify to strip external URLs”) and compare runs with vs without hooks, or with different conditions (P1–P4).

---

## References

- Claude Code Hooks: [code.claude.com/docs/en/hooks-guide](https://code.claude.com/docs/en/hooks-guide), [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks).
- Structured outputs: [platform.claude.com/docs/en/build-with-claude/structured-outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
- Verifier + retry pattern: validate in code, reject or regenerate; we use “reject only” in this project.
