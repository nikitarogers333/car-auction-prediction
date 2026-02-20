# Deterministic, Auditable LLM Car Auction Price Prediction

A **controlled research framework** for numeric car auction price prediction: consistent predictions, strict behavioral restrictions, no unauthorized lookups, variance measurement, full audit trail, deterministic routing, and comparison to statistical baselines.

**Research angle:** This project explores how to make LLMs **deterministic** and **constraint-following** (car price is the example). The web app is structured around the **scientific process** (Observation → Question → Hypothesis → Experiment → Data Collection → Analysis → Conclusion → Repeat). See **[RESEARCH_README.md](RESEARCH_README.md)** for the research questions and how we use structured outputs, validation gates, and **hook-style enforcement** (aligned with [Claude Code Hooks](https://code.claude.com/docs/en/hooks-guide)). Mapping to Claude hooks and the “deterministic under constraints” stack: **[docs/CLAUDE_HOOKS_AND_ENFORCEMENT.md](docs/CLAUDE_HOOKS_AND_ENFORCEMENT.md)**.

## What This System Does

- **Produces** price predictions in a strict JSON format (schema-validated).
- **Enforces** restrictions: no external web unless allowed, VIN stripped, subgroup isolation (e.g. M3 vs 340i), no ground-truth leakage, price bounds.
- **Measures** output variance (N repeats per vehicle: mean, std, coefficient of variation, invalid rate).
- **Logs** every run to `logs/YYYYMMDD_HHMMSS.jsonl` with vehicle_id, condition, prompt, raw/parsed output, valid flag, violation reason, variance stats, model, temperature.
- **Compares** four prompt/access conditions (P1–P4) and LLM vs KNN/regression baselines.
- **Requires** `OPENAI_API_KEY` (real API only, temperature=0, seed=42).
- **Input:** Excel (.xlsx) files with vehicle data (columns: vehicle_id, make, model, year, mileage, price optional).
- **Web app:** Flask app (`app.py`) for easy upload and results viewing (deploy to Railway).

## Directory Layout

```
/data          Input Excel (.xlsx) or JSON; template.xlsx for reference
/eval          Experiment outputs, baselines summary
/agent         Pipeline components (guards, classifier, prediction agent)
/restrictions  subgroup_map.json, domain_allowlist.json, price_bounds.json
/logs          Audit logs (JSONL per run batch)
/agent_builder_design  Flow diagram, schemas, OpenAI Agent Builder instructions
app.py         Flask web app (upload Excel, run predictions, download results)
main.py        CLI entry point
pipeline.py    Deterministic pipeline runner
data_loader.py  Excel/JSON loader
schemas.py     Strict output schema validation
scoring.py     MAE, MAPE, variance, invalid rate
audit_logger.py  Audit trail writer
baselines.py   KNN, linear regression, optional XGBoost
hooks.py       PrePredictionHook, PostPredictionHook, StopHook
config.py      P1–P4 conditions, determinism defaults
requirements.txt  Python dependencies
Procfile       Railway deployment config
```

## Quick links

- **How to use it yourself:** see **HOW_TO_RUN.md** (copy-paste commands, no API key needed).
- **How to share with your professor:** zip the project or use Git; give them **PROFESSOR.md** for run instructions and grading notes.
- **Regression testing:** run `python3 scripts/run_eval.py` (mock) to validate pipeline output on a fixed dataset.

## How to Run

### Prerequisites

- Python 3.11+
- **OPENAI_API_KEY** and/or **ANTHROPIC_API_KEY** (at least one required for real predictions; use both to compare providers)
- Install dependencies: `pip install -r requirements.txt`

### Web App (Recommended for Professors)

1. **Set at least one API key** (use both to compare OpenAI vs Claude):
   ```bash
   export OPENAI_API_KEY="sk-..."    # for OpenAI (GPT)
   export ANTHROPIC_API_KEY="sk-ant-..."   # for Claude
   ```

2. **Run the Flask app:**
   ```bash
   python3 app.py
   ```
   Opens at `http://localhost:5000`

3. **Upload Excel file** (columns: vehicle_id, make, model, year, mileage, price optional)
   - Choose **Provider**: OpenAI (GPT) or Claude (Anthropic) to compare both
   - Download template: visit `/create_template` or use `data/template.xlsx`
   - Optionally enable **Consistency check** (same vehicle, N repeats) or **Use enforcement hooks** (reject if confidence &lt; 0.3)
   - View results table and download results as Excel

### CLI (Command Line)

Put global options **before** the command (`single`, `consistency`, etc.):

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Single run (one vehicle)
python3 main.py --condition P1 --log single

# Consistency (N repeats, variance)
python3 main.py --repeats 5 consistency

# Experiments (P1–P4 comparison)
python3 main.py --repeats 2 --limit 5 experiments
# Output: eval/experiments_YYYYMMDD_HHMMSS.json

# Baselines (KNN + linear regression)
python3 main.py baselines
# Output: eval/baselines_summary.json
```

**Input:** Place Excel file at `data/vehicles.xlsx` or use `--data path/to/file.xlsx`

## Conditions P1–P4

| Condition | External access      | Use case                    |
|-----------|----------------------|-----------------------------|
| P1        | No                   | Internal knowledge only     |
| P2        | Yes, any             | Any source allowed         |
| P3        | Yes, allowlist only  | Allowlist domains only     |
| P4        | No                   | No web enforced            |

Restriction enforcement (VIN stripping, subgroup rules, price bounds) is deterministic code in `RestrictionEnforcer` and `PostValidationGuard`; P1–P4 control whether the **prediction step** is allowed to call external tools (in a full deployment you’d wire this to your LLM/tool config).

## How to Compare Variance

- Run `python main.py consistency --repeats 5` for a vehicle.
- Inspect `variance_stats`: `mean`, `std`, `cv` (coefficient of variation %), `invalid_rate`.
- If `cv > 15` (or your threshold in `config.VARIANCE_CV_THRESHOLD`), flag as unstable.
- Logs in `logs/*.jsonl` include `variance_stats` per run batch.

## How to Enforce Restrictions

- **No external web:** Use condition P1 or P4; in Agent Builder disable web/search tools for the prediction agent.
- **Allowlist only:** Use P3; configure tools to allow only domains in `restrictions/domain_allowlist.json`.
- **VIN leakage:** `PreValidationGuard` removes VIN from the payload before prediction; never send VIN to the LLM.
- **Subgroup isolation:** `SubgroupClassifier` sets `allowed_comparables` from `restrictions/subgroup_map.json`; `PostValidationGuard` rejects if `subgroup_detected` is not in that list or is in `forbidden_comparables`.
- **Price bounds:** `restrictions/price_bounds.json`; `PostValidationGuard` rejects predictions outside min/max for that make/model.

## How to Extend to Other Domains

1. **Data:** Put domain records in `data/` (e.g. same JSON shape: id, features, optional ground_truth).
2. **Subgroups:** Edit `restrictions/subgroup_map.json` with your categories and allowed/forbidden comparables.
3. **Bounds:** Edit `restrictions/price_bounds.json` (or equivalent) for your target variable.
4. **Schema:** In `schemas.py`, adjust `PredictionOutputSchema` and `validate_prediction_output` if you need different fields.
5. **Prompts:** In `agent/prediction_agent.py`, change `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` for your task.
6. **Baselines:** In `baselines.py`, add or change feature keys in `build_feature_vector` and add other models as needed.

## Reproducibility

- **Determinism:** temperature=0, top_p=1, fixed system prompt, seed=42 when supported.
- **Audit:** Every run logged with timestamp, model, temperature, condition, valid, violation_reason.
- **No free-form chaining:** Each step consumes and produces structured JSON; no ad-hoc text between steps.

## Deploy to Railway

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   # Create repo on GitHub, then:
   git remote add origin https://github.com/yourusername/repo-name.git
   git push -u origin main
   ```

2. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app)
   - New Project → Deploy from GitHub repo
   - Add environment variable: `OPENAI_API_KEY` = your key
   - Railway auto-detects `Procfile` and deploys
   - Your app will be live at `https://your-app.railway.app`

3. **Share with professor:** Give them the Railway URL and they can upload Excel files directly.

## Research Questions This Framework Supports

- Why does the LLM give inconsistent outputs? → Use **consistency** runs and variance stats.
- Can we force deterministic compliance? → Use **PostValidationGuard** + **RestrictionEnforcer**; log violations.
- How do restrictions change performance? → Compare **P1 vs P2 vs P3 vs P4** in **experiments**.
- When is LLM inferior to structured models? → Compare **experiments** output to **baselines** (KNN, regression).
- Can we audit reasoning? → Use **logs/** and `parsed_output` / `raw_output` in each record.
- Can we prevent rule violations? → Hooks + guards; log `violation_reason` and invalid rate.
