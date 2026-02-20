# Research: Deterministic & Constrained LLM Behavior

**Research question:** How can we make LLMs give **deterministic** outputs and **follow the rules** we give them—so we can trust AI to do real, important work?

**Use case in this project:** Predicting car auction price. Same prompt → different prices; “only use these sites” → model still uses others. We use this task to study **enforcement** and **quality checks**.

---

## The Problem (Observation)

- Give the **same prompt** (e.g. “What’s the auction value of this BMW M3?”) → the LLM returns **different values** each time.
- Tell it to **only use certain websites** → it still accesses others.
- So: **non-determinism** and **constraint violation** are real. How do we fix that?

## Research Questions

1. Can we get **consistent** (deterministic) numerical predictions?
2. Can we **enforce** behavioral constraints (no external lookup, or allowlist-only)?
3. How do we **check quality** (format, bounds, rule compliance) so we only accept valid outputs?

## Our Approach (Hypothesis)

We combine several tools that the literature and your professor’s discussion point to:

| Tool | What we do in this project |
|------|----------------------------|
| **Structured outputs** | LLM must return strict JSON (predicted_price, confidence, method, subgroup_detected, notes). Invalid JSON → rejected. |
| **Deterministic decode** | `temperature=0`, `top_p=1`, fixed seed where supported. |
| **Verifier + reject** | After the LLM responds, **deterministic code** (no LLM) checks: schema, subgroup rules, price bounds, no VIN leakage. If fail → run marked invalid, never accepted. |
| **Hook-like enforcement** | We implement **PrePredictionHook**, **PostPredictionHook**, **StopHook** in Python. Same idea as [Claude Code Hooks](https://code.claude.com/docs/en/hooks-guide): run code at fixed points to allow/deny/modify. (See `hooks.py` and [docs/CLAUDE_HOOKS_AND_ENFORCEMENT.md](docs/CLAUDE_HOOKS_AND_ENFORCEMENT.md).) |

So: **constrain format + deterministic settings + hard validation gate + hook-style checks** → we only accept outputs that pass every rule.

## How the Car Price Example Fits

- **Task:** Predict sale price at auction for a list of vehicles.
- **Restrictions we enforce:** No external web unless allowed (P1–P4), no VIN in prompt/output, subgroup isolation (e.g. M3 vs 340i), price within bounds.
- **Quality checks:** JSON schema, subgroup in allowed set, no forbidden leakage, bounds check.
- **Experiments:** Compare conditions (P1: internal only, P2: any source, P3: allowlist, P4: no web) and measure variance, invalid rate, MAE/MAPE.

The **front-end** is designed around the **scientific process**: Observation → Question → Hypothesis → Experiment → Data Collection → Analysis → Conclusion → Repeat.

## References (from your professor and the other LLM)

- **Claude Code Hooks** (deterministic enforcement at runtime): [code.claude.com/docs/en/hooks-guide](https://code.claude.com/docs/en/hooks-guide), [aiorg.dev/blog/claude-code-hooks](https://aiorg.dev/blog/claude-code-hooks).
- **Structured outputs:** OpenAI Structured Outputs / JSON Schema; Claude structured outputs.
- **Verifier + retry:** Validate with code → reject or regenerate. We do “reject only” (no retry in the app) for clarity.
- **Constrained decoding** (optional next step): e.g. [Outlines](https://dottxt-ai.github.io/outlines/), [LMQL](https://github.com/eth-sri/lmql).

## Repo layout (where what lives)

- **`app.py`** — Web app: scientific-process UI, upload, run experiment, view results and analysis.
- **`hooks.py`** — PrePredictionHook, PostPredictionHook, StopHook (hook-style enforcement).
- **`agent/`** — PreValidationGuard, SubgroupClassifier, RestrictionEnforcer, PredictionAgent, PostValidationGuard (verifier layer).
- **`schemas.py`** — Strict JSON schema for the only accepted output format.
- **`restrictions/`** — subgroup_map.json, domain_allowlist.json, price_bounds.json (rules in code).
- **`docs/CLAUDE_HOOKS_AND_ENFORCEMENT.md`** — How our hooks map to Claude Code Hooks and how to think about deterministic, constrained behavior.

This project uses **OpenAI** for the prediction API in the demo; the same **enforcement and evaluation** ideas apply to **Claude** (e.g. Claude Agent SDK + hooks + structured outputs) as your professor suggested.
