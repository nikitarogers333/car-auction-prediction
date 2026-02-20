# For Instructors: How to Run and Evaluate This Project

This document explains how to obtain, run, and evaluate the car auction price prediction project (deterministic, auditable LLM pipeline) for grading or review.

---

## Research focus

The project is framed around a **research question**: How can we make LLM predictions **deterministic** (consistent) and **constraint-following**, and how do we **check quality** so we only accept valid outputs? Car auction price is the running example.

- **RESEARCH_README.md** — Research questions, approach (structured output + validation gate + hook-style enforcement), and references (Claude Code Hooks, structured outputs, verifier pattern).
- **Web app** — The Flask app (`app.py`) is structured around the **scientific process**: Observation → Question → Hypothesis → Experiment → Data Collection → Analysis → Conclusion → Repeat. Students can run **batch experiments** (upload file, P1–P4) or a **consistency check** (same vehicle, N repeats) to measure variance (CV, valid rate).
- **docs/CLAUDE_HOOKS_AND_ENFORCEMENT.md** — Maps the project’s PrePredictionHook, PostPredictionHook, StopHook to Claude Code Hooks and the “deterministic under constraints” stack.

---

## What the project is

- A **research-style framework** for car auction price prediction using a strict pipeline: validation → feature extraction → subgroup classification → restrictions → prediction → post-validation → scoring/logging.
- **Deterministic controls**: temperature=0, fixed prompts, optional seed; mock mode requires no API key.
- **Audit trail**: optional JSONL logs for every run.
- **Comparisons**: P1–P4 conditions (internal vs external knowledge) and KNN/regression baselines.

---

## How to get the project

**Option A — Zip (simplest)**  
1. Zip the project folder (include everything except `textbook pdfs/` if you don’t need the PDFs).  
2. Share the zip (email, LMS, etc.).  
3. Student/instructor unzips and runs from that folder.

**Option B — Git**  
1. Initialize a repo in the project folder: `git init`  
2. Add a `.gitignore` (see below) so secrets and large PDFs aren’t committed.  
3. Push to a private GitHub/GitLab repo and share the link.  
4. Clone: `git clone <repo-url>` then `cd <repo-folder>`

Suggested **.gitignore** (create in project root if using git):

```
.env
*.key
logs/*.jsonl
__pycache__/
*.pyc
.DS_Store
textbook pdfs/
```

---

## How to run it (no API key required)

All of the following work **without** an OpenAI API key (mock predictor is deterministic).

1. **Open a terminal** and go to the project directory:
   ```bash
   cd /path/to/Anton-Project
   ```

2. **Single prediction:**
   ```bash
   python3 main.py --mock single
   ```
   Expected: JSON printed to the terminal with `vehicle_id`, `make`, `model`, `subgroup`, `prediction` (with `predicted_price`, `confidence`, `method`, `subgroup_detected`, `notes`), and `valid: true`.

3. **Consistency (variance) run:**
   ```bash
   python3 main.py --mock --repeats 5 consistency
   ```
   Expected: Variance stats (mean, std, cv, invalid_rate) and “Unstable (CV > threshold): False” or True.

4. **P1–P4 experiments:**
   ```bash
   python3 main.py --mock --repeats 2 --limit 4 experiments
   ```
   Expected: Summary printed (MAE, MAPE, invalid_rate per condition) and a new file under `eval/experiments_YYYYMMDD_HHMMSS.json`.

5. **Baselines:**
   ```bash
   python3 main.py baselines
   ```
   Expected: KNN MAE and Linear Regression MAE printed, and `eval/baselines_summary.json` created.

**Requirements:** Python 3.11+ and standard library only for mock mode. For real API: `pip install openai` and `OPENAI_API_KEY` set.

6. **Web app (with API key):**
   ```bash
   pip install -r requirements.txt
   python3 app.py
   ```
   Open http://localhost:5000. Use the **Experiment** section: upload a file, choose condition (P1–P4), optionally check **Consistency check** (first vehicle only, N repeats). Results appear under Data collection and Analysis (valid rate, mean/std/CV of predicted price). Use **Repeat / Refine** to run another experiment.

---

## What to look for when evaluating

- **Structure:**  
  - `main.py` as entry point; `agent/` (guards, classifier, prediction agent, restriction enforcer); `restrictions/` (subgroup_map, price_bounds, domain_allowlist); `data/`, `eval/`, `logs/`.

- **Pipeline:**  
  - Clear flow: PreValidation → FeatureExtraction → SubgroupClassifier → RestrictionEnforcer → PredictionAgent → PostValidationGuard; no free-form chaining.

- **Strict output:**  
  - Prediction must match schema (`predicted_price`, `confidence`, `method`, `subgroup_detected`, `notes`); invalid outputs set `valid: false` and a violation reason.

- **Restrictions:**  
  - VIN/target stripped before prediction; subgroup rules (e.g. M3 vs 340i) in `restrictions/subgroup_map.json`; price bounds in `restrictions/price_bounds.json`; PostValidationGuard enforces them.

- **Reproducibility:**  
  - Mock mode is deterministic; with API, temperature=0 and fixed prompts; audit logging available with `--log`.

- **Documentation:**  
  - README.md (overview, run instructions); HOW_TO_RUN.md (quick copy-paste commands); this PROFESSOR.md (instructor run and evaluation).

- **Design docs:**  
  - `agent_builder_design/` with flow diagram, JSON schemas, and instructions for replicating in an Agent Builder–style tool.

---

## Suggested grading checklist (example)

| Criterion              | Points | Notes |
|------------------------|--------|--------|
| Project runs without errors (mock) | 15 | Run `single`, `consistency`, `experiments`, `baselines` |
| Pipeline structure      | 20 | Guards, classifier, enforcer, prediction, post-validation present and wired |
| Strict JSON + validation| 15 | Schema enforced; invalid runs marked and explained |
| Restrictions enforced  | 15 | VIN stripped; subgroup/price rules used and validated |
| Variance + experiments  | 15 | Consistency and P1–P4 experiments produce metrics |
| Baselines               | 10 | KNN and regression implemented and runnable |
| Logging / audit         | 5  | Optional `--log` and logs/ output |
| Documentation           | 5  | README + run instructions + (optional) PROFESSOR.md |

---

## Contact / attribution

This project was built as a controlled research framework for deterministic, auditable LLM-based car auction price prediction. For questions about running or grading, refer to README.md and HOW_TO_RUN.md in the project root.
