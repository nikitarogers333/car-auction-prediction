# Research: Where Should the AI Stop?

**Thesis:** Does moving the LLM out of the final price calculation and into a structured feature extraction role produce more consistent and accurate auction price predictions?

**One-liner:** We tested whether offloading the numerical output to deterministic code improves both reliability and accuracy, even when the LLM component is identical -- and measured the reliability and cost tradeoffs of doing so.

**Use case:** Car auction price prediction. The LLM is a **component** inside a box; anything outside the rules gets rejected, repaired, or replaced.

---

## Hypotheses (stated before running)

These are written down before any experimental data is collected. If any fail, that is an interesting result, not a bad one.

- **H1:** Pipeline B (E5, LLM extracts features, code computes price) will have **lower price CV** than Pipeline A (E3, LLM predicts price directly) across repeated runs of the same vehicle.
- **H2:** Pipeline B will have **lower MAE** against ground truth prices than Pipeline A.
- **H3:** LLM feature assignments (trim_tier, market_demand, etc.) will be **more consistent across repeats** than LLM price outputs. This is the smoking gun -- if the LLM's qualitative judgments are stable but its numbers aren't, that's why E5 wins.
- **H4:** Pipeline A-prime (LLM picks features freely without schema enforcement, same pricing formula) will perform **worse** than Pipeline B, confirming that enforcement placement matters more than the formula itself.

**Temperature:** All comparison runs use **temperature=0.7** (not 0). If temperature=0 gives perfect consistency, the entire variance argument collapses. This is set explicitly in `config.COMPARISON_TEMPERATURE` and documented here.

---

## Why not just filter harder? (The framing shift)

- (1) "Are LLMs deterministic?" and (2) "Do they follow rules when we ask?" are known: **no** and **no**.
- E0 through E3 are filtering systems -- reject or repair outputs post-hoc. High valid rates under E3 tell you almost nothing about the model's underlying behavior; you could achieve the same by retrying forever. CV reduction may be a selection effect from a truncated distribution.
- The real question is **(3) where to place the probabilistic boundary**. E5 moves the boundary: the LLM makes qualitative judgments (which may be more stable), and deterministic code produces the final number.
- The enforcement ladder (E0-E3) is evidence for **why** filtering alone isn't enough. E5 vs E3 is the contribution.

---

## The enforcement toolbox

| Layer | What it does |
|-------|-------------|
| **1. Hard output constraints** | Structured output / JSON schema: response must match strict shape. Enforces correct fields and types. Does not enforce truthfulness. |
| **2. Validation gates** | Code checks: price in bounds, confidence in [0,1], no VIN leakage, subgroup in allowed set. If any rule fails, mark invalid or retry. |
| **3. Retry and repair** | On failure, send error back, ask for corrected JSON, repeat up to N times. If still invalid, fallback. |
| **4. Tool restriction** | "No web" enforced by not providing web tools (P4). "Allowlist" = retrieval only points to allowed docs (P3). |
| **5. Hooks** | Pre/Post/Stop hooks run at fixed points. Same idea as Claude Code Hooks. |
| **6. Boundary placement (E5)** | LLM outputs features only; code computes final price. Strongest form: the LLM can't directly freehand the number. |

---

## Experiment design

### Pipelines

- **Pipeline A (E3):** LLM receives vehicle fields, outputs predicted_price directly, enforcement filters/retries bad outputs.
- **Pipeline B (E5):** LLM receives vehicle fields, outputs structured features only (condition_score, trim_tier, market_demand, depreciation_rate, mileage_assessment, comparable_market), deterministic code computes final price from those features.
- **Pipeline A-prime (ablation):** LLM picks feature values freely (no schema enforcement on features), same pricing formula. Isolates enforcement from formula effect.

### What we measure

**Reliability:**
- Valid rate at each pipeline
- Mean retries per request
- CV of final prices across N=5 repeats of the same vehicle

**Accuracy:**
- MAE (mean absolute error) against ground truth prices
- % of predictions within 10% of actual sale price

**Feature consistency (the smoking gun):**
- Across 5 repeats, how often does the LLM assign the same trim_tier, market_demand, depreciation_rate, mileage_assessment, comparable_market to the same vehicle?
- Reported as % agreement per feature and overall feature stability score.

**Cost:**
- Total tokens per prediction
- Retries per request

### Conditions

- Each vehicle runs through Pipeline A, Pipeline B, and Pipeline A-prime, **5 times each** with **temperature=0.7**.
- We compare OpenAI vs Claude under both pipelines.
- Dataset: 200+ vehicles with ground truth prices (synthetic, with a ground truth formula deliberately different from the pricing formula to avoid trivial wins).

---

## Repo layout

- **`config.py`** -- Enforcement levels (E0-E5), comparison temperature, providers.
- **`pipeline.py`** -- `run_pipeline(..., enforcement_level="E5")` routes to feature extraction + deterministic pricing.
- **`pricing.py`** -- `compute_price_from_features()`: deterministic formula, no LLM.
- **`compare.py`** -- Comparison runner: E3 vs E5, N repeats, outputs metrics table + feature consistency.
- **`agent/prediction_agent.py`** -- E5 mode: feature-extraction prompt, `_llm_extract_features()`.
- **`schemas.py`** -- `validate_feature_output()` for E5 feature schema.
- **`data/generate_eval_dataset.py`** -- Synthetic ground truth generator (different formula from pricing.py).
- **`data/eval_dataset.csv`** -- 200+ vehicles with ground truth prices.
- **`app.py`** -- Web UI: comparison mode (E3 vs E5 side-by-side).
- **`hooks.py`** -- Pre/Post/Stop hooks for optional extra enforcement.
- **`docs/CLAUDE_HOOKS_AND_ENFORCEMENT.md`** -- Mapping to Claude Code Hooks.

This project uses **OpenAI** and **Claude**; the same design applies to either provider.
