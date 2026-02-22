# Experiment Findings: Where Should the AI Stop?

**Nikita Rogers · February 2026**

## The Question

When an AI predicts car auction prices, where should enforcement go — on the output (check and retry) or on the architecture (restructure what the AI does)?

## The Experiment

Five pipelines predict auction prices for the same cars under the same conditions. Each pipeline uses a different enforcement strategy.

| Pipeline | What It Does | Enforcement |
|---|---|---|
| **A: E3** | LLM predicts price directly. Schema + validation + retry on failure. | Check what it said |
| **B: E5** | LLM extracts structured features (JSON schema). Formula computes price. | Control what it can do |
| **A': Ablation** | LLM describes car in free-form prose. Parser extracts features. Same formula. | No enforcement on output |
| **C: Random Forest** | Traditional ML regression on historical data. | N/A (deterministic) |
| **D: XGBoost** | Traditional ML gradient boosting on historical data. | N/A (deterministic) |

**Settings:** GPT-4o-mini, temperature 0.7, 5 repeats per vehicle.

## The Hypotheses

All four hypotheses were written before running the experiment.

| # | Prediction | Result |
|---|---|---|
| **H1** | E5 prices will vary less run-to-run than E3 | **Supported** — E5 CV 3.37% vs E3 CV 3.75% |
| **H2** | E5 will be more accurate than E3 | **Supported** — E5 MAE $2,818 vs E3 MAE $14,401 |
| **H3** | Feature descriptions more consistent than price predictions | **Supported** — 97.3% feature stability vs 3.75% price CV |
| **H4** | E5 (schema) will beat A' (free-form) with the same formula | **Supported** — E5 MAE $2,818 vs A' MAE $8,045 |

## The Results

| Metric | A: E3 | B: E5 | A': Ablation | C: RF | D: XGB |
|---|---|---|---|---|---|
| Valid rate | 100% | 100% | 100% | 100% | 100% |
| Retries | 0 | 0 | 0 | 0 | 0 |
| **Price CV (%)** | 3.75 | 3.37 | **67.71** | 0.00 | 0.00 |
| **MAE ($)** | 14,401 | **2,818** | 8,045 | 22,221 | 17,005 |
| Within 10% of actual | 20% | **60%** | 20% | 0% | 40% |
| Feature stability | — | 97.3% | 85.1% | — | — |

ML models (C, D) had access to the MMR feature (professional wholesale price estimate) that the LLM pipelines did not.

## What It Means

### E5 is 5x more accurate than E3

E5 error: $2,818. E3 error: $14,401. The formula-based pipeline doesn't just beat the direct LLM prediction — it beats both traditional ML models despite those models having professional pricing data the LLM didn't have.

### H4 is the most important finding

E5 MAE $2,818 vs A' MAE $8,045 — nearly 3x worse with the same formula and same cars. A' had a 100% valid rate and zero retries — the parser never failed. It extracted the *wrong* values because the LLM's free-form prose was ambiguous.

Without a schema forcing a decision, the LLM produced "fairly strong" one run and "moderate" the next. The parser mapped those to different feature values, which produced different prices. The 67.71% price CV on A' vs 3.37% on E5 is the proof.

**The schema forces the LLM to commit to a specific interpretation of the car at the reasoning step, not just the formatting step.** That's the core architectural insight: schema enforcement is a precision constraint on reasoning, not just on output format.

### The three approaches to constraining AI

1. **Control what it can say** — Block bad outputs before generation (grammar-constrained decoding, Outlines, Guidance). Not tested here.
2. **Check what it said** — Let the AI generate freely, validate afterward, retry on failure. This is E3. It works but doesn't improve the underlying predictions.
3. **Control what it can do** — Restructure the AI's role so it only handles what it's reliable at. This is E5. It forces structured reasoning, not just structured output.

The experiment shows that approach 3 produces fundamentally better results than approach 2, and H4 proves the improvement comes from the structured reasoning constraint, not from the pricing formula alone.

## The Live Tool

The full experiment runs as a live web application:

- **Explainer:** [web-production-d4ab7.up.railway.app](https://web-production-d4ab7.up.railway.app) — walks through the enforcement ladder and experiment design
- **Pipeline comparison:** [web-production-d4ab7.up.railway.app/compare](https://web-production-d4ab7.up.railway.app/compare) — runs all five pipelines and evaluates hypotheses live

## Source Code

[github.com/nikitarogers333/car-auction-prediction](https://github.com/nikitarogers333/car-auction-prediction)

Key files:
- `pipeline.py` — E3, E5, and A' pipeline implementations
- `compare.py` — comparison runner and hypothesis evaluation
- `pricing.py` — deterministic pricing formula (shared by E5 and A')
- `agent/prediction_agent.py` — LLM prompts, mock logic, and free-form parser
- `app.py` — Flask web app with results visualization

## Design Notes

**Retry asymmetry:** A' retries once on parse failure; E5 retries three times on schema validation failure. This turned out to be irrelevant — A' had a 0% retry rate and 100% valid rate. The parser always succeeded; it just extracted inconsistent values from ambiguous prose.

**Mock vs real LLM:** In mock mode, A' ≈ E5 because the mock text is clear and predictable. The gap only appears with a real LLM producing genuinely ambiguous free-form text. This confirms the finding is about LLM behavior under ambiguity, not about parser quality.

**Temperature 0.7:** Chosen deliberately so variance is real. At temperature 0, consistency is trivial and the experiment tests nothing. At 0.7, the LLM produces meaningfully different outputs across runs, which is what makes the CV comparison valid.
