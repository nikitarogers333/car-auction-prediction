# Experiment Guide: Where Should the AI Stop?

## The core question

Does moving the LLM out of the final price calculation and into a structured feature extraction role produce more consistent and accurate auction price predictions?

We test this by comparing two pipeline architectures on a ground truth dataset.

---

## Hypotheses (stated before running)

- **H1:** Pipeline B (E5) will have lower price CV than Pipeline A (E3)
- **H2:** Pipeline B will have lower MAE against ground truth than Pipeline A
- **H3:** LLM feature assignments will be more consistent across repeats than LLM price outputs
- **H4:** Pipeline A-prime will perform worse than Pipeline B, confirming enforcement placement matters more than the formula

If any fail, that is an interesting result. "We expected E5 to win and it didn't, here's why" is publishable. "We built a thing and it worked" is not.

---

## The two pipelines

**Pipeline A (E3 -- current approach)**
LLM receives vehicle fields -> outputs predicted_price directly -> enforcement filters/retries bad outputs.

**Pipeline B (E5 -- new approach)**
LLM receives vehicle fields -> outputs structured features only (condition_score, trim_tier, market_demand, depreciation_rate, mileage_assessment, comparable_market) -> deterministic code computes final price from those features. The LLM never touches a number.

**Pipeline A-prime (ablation)**
LLM picks feature values freely (no schema enforcement) -> same pricing formula as B. Isolates the enforcement effect from the formula effect.

---

## How to run

### Web app: comparison mode

1. Open the app -> click **E3 vs E5 Comparison** (or go to `/compare`).
2. Choose provider (OpenAI or Claude), number of vehicles, number of repeats.
3. Click **Run E3 vs E5 comparison**.
4. Results show a head-to-head metrics table and hypothesis assessment.

### CLI: full comparison

```bash
# Full run with real LLM (requires API keys)
python3 compare.py --provider openai --n-vehicles 20 --n-repeats 50

# Quick test with mock LLM
python3 compare.py --mock --n-vehicles 10 --n-repeats 3

# Use Claude
python3 compare.py --provider claude --n-vehicles 20 --n-repeats 5
```

### Web app: enforcement ladder (supporting evidence)

The main page (`/`) still supports running individual vehicles through E0-E4 with consistency checks. Use this to generate supporting evidence for why filtering alone (E0-E3) isn't enough.

---

## What the output looks like

### Comparison table

| Metric | E3 (direct) | E5 (features) |
|--------|-------------|----------------|
| Valid rate | X% | X% |
| Mean retries | X | X |
| Price CV (%) | X | X |
| MAE ($) | $X | $X |
| Within 10% of actual (%) | X% | X% |
| Feature stability (E5 only) | --- | X |

### Hypothesis assessment

Each hypothesis is marked SUPPORTED, NOT SUPPORTED, or PARTIAL, with the specific numbers as evidence.

### Results JSON

Full results are saved to `data/comparison_results.json` with per-vehicle metrics, feature consistency detail, and aggregate summary.

---

## Critical experimental details

**Temperature = 0.7** (set in `config.COMPARISON_TEMPERATURE`). This is not 0. If temperature=0, perfect consistency is trivial and the variance argument collapses. Reviewers will ask about this.

**Ground truth formula is deliberately different from the pricing formula.** The eval dataset uses exponential depreciation, quadratic mileage, polynomial condition bonuses, and interaction effects. The E5 pricing formula uses linear depreciation, mileage buckets, and additive adjustments. If they were identical, Pipeline B would win trivially.

**N repeats = 50 (recommended).** Because LLM outputs are non-deterministic, comparisons between pipelines should be based on distributions, not single outputs. Run each vehicle through each pipeline **~50 times** (or more) with the same inputs, then report mean metrics and confidence intervals.

---

## What we will know after running

1. **Does architectural boundary placement matter?** If E5 beats E3 on both MAE and CV, the finding is: offloading numerical output to deterministic code improves both reliability and accuracy, even when the LLM component is identical.

2. **Are LLM qualitative judgments more stable than LLM numerical outputs?** Feature consistency metric tells us this directly. If features are stable but prices aren't, that's the smoking gun for why E5 works.

3. **Does enforcement on features matter?** A-prime vs B isolates this. If Pipeline B beats A-prime, then schema-enforced feature extraction is doing real work beyond just the formula.

4. **Provider comparison.** Running both OpenAI and Claude under both pipelines tells us if the finding generalizes across providers or is provider-specific.

5. **Cost tradeoffs.** Token usage, retry counts, and latency per pipeline tell us if E5 is cheaper or more expensive in practice.

---

## The one-sentence pitch

We tested whether moving the LLM out of the final price calculation and into a structured feature extraction role produces more consistent and accurate auction price predictions -- and measured the reliability and cost tradeoffs of doing so.
