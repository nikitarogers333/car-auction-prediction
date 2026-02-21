"""
Flask web app for car auction price prediction.
Upload Excel file, run pipeline on each row, show results table + download.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
from io import BytesIO

import pandas as pd
from flask import Flask, render_template_string, request, send_file

from compare import load_eval_data, run_comparison
from config import COMPARISON_TEMPERATURE, CONDITIONS, ENFORCEMENT_LEVELS, ENFORCEMENT_DESCRIPTIONS, PROVIDERS
from regression_predictor import models_available as regression_models_available
from data_loader import load_from_file
from hooks import make_confidence_floor_hook
from pipeline import run_consistency_check, run_pipeline
from scoring import variance_stats

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "car_auction_predict"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Each row calls the OpenAI API; limit rows so the request doesn't time out (e.g. 25 × ~6s ≈ 2.5 min)
MAX_ROWS_PER_UPLOAD = 25
# Consistency check: same vehicle, N repeats (keep N small to avoid timeout)
CONSISTENCY_CHECK_REPEATS = 5
MAX_CONSISTENCY_REPEATS = 10


def require_any_api_key() -> None:
    openai = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not openai and not anthropic:
        print(
            "Error: Set at least one of OPENAI_API_KEY or ANTHROPIC_API_KEY (e.g. in Railway env vars).",
            file=sys.stderr,
        )
        sys.exit(1)


require_any_api_key()


def _base_ctx(
    use_hooks_default: bool = True,
    provider_default: str = "openai",
    enforcement_default: str = "E2",
):
    return {
        "template_link": "/create_template",
        "max_rows": MAX_ROWS_PER_UPLOAD,
        "repeats_default": CONSISTENCY_CHECK_REPEATS,
        "max_repeats": MAX_CONSISTENCY_REPEATS,
        "use_hooks_default": use_hooks_default,
        "provider_default": provider_default,
        "enforcement_default": enforcement_default,
        "enforcement_levels": list(ENFORCEMENT_LEVELS),
        "enforcement_descriptions": ENFORCEMENT_DESCRIPTIONS,
    }


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Research: Deterministic & Constrained LLM Predictions</title>
    <style>
        :root { --step-bg: #f8f9fa; --step-border: #dee2e6; --accent: #0d6efd; --valid: #198754; --invalid: #dc3545; }
        body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; line-height: 1.5; color: #212529; }
        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
        h2 { font-size: 1.15rem; margin: 1.25rem 0 0.5rem; color: #495057; }
        .subtitle { color: #6c757d; margin-bottom: 1.5rem; }
        .step { background: var(--step-bg); border: 1px solid var(--step-border); border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
        .step-title { font-weight: 600; color: #0d6efd; margin-bottom: 0.35rem; }
        .upload { border: 2px dashed var(--step-border); padding: 24px; text-align: center; margin: 12px 0; border-radius: 8px; background: #fff; }
        .btn { background: var(--accent); color: #fff; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block; font-size: 0.95rem; }
        .btn:hover { filter: brightness(0.95); }
        .btn-secondary { background: #6c757d; }
        table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9rem; }
        th, td { border: 1px solid var(--step-border); padding: 6px 10px; text-align: left; }
        th { background: #e9ecef; font-weight: 600; }
        .valid { color: var(--valid); }
        .invalid { color: var(--invalid); }
        .error { color: var(--invalid); padding: 10px; background: #f8d7da; border-radius: 6px; margin: 10px 0; }
        .success { color: var(--valid); padding: 10px; background: #d1e7dd; border-radius: 6px; margin: 10px 0; }
        .analysis-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin: 10px 0; }
        .analysis-item { background: #fff; border: 1px solid var(--step-border); border-radius: 6px; padding: 10px; text-align: center; }
        .analysis-item .value { font-weight: 700; font-size: 1.1rem; }
        .analysis-item .label { font-size: 0.8rem; color: #6c757d; }
        .conclusion-note { font-style: italic; color: #495057; }
        #experiment-section { scroll-margin-top: 1rem; }
    </style>
</head>
<body>
    <h1>Research: Deterministic & Constrained LLM Behavior</h1>
    <p class="subtitle">Car auction price prediction as a use case — scientific process</p>
    <p><a href="/compare" class="btn">E3 vs E5 Comparison (main experiment)</a></p>

    <div class="step">
        <div class="step-title">1. Observation</div>
        <p>LLMs aren’t truly deterministic, and “please follow rules” is not enforcement. You don’t make the model obey by willpower — you build a system where the LLM only contributes inside a box, and anything outside the rules gets rejected, repaired, or replaced. The LLM is a component, not the authority.</p>
    </div>
    <div class="step">
        <div class="step-title">2. Question</div>
        <p>How much can <strong>enforcement</strong> reduce invalid outputs and reduce variance in accepted outputs, and at what cost (retries, latency, throughput)? We measure: valid rate, retries per request, variance (CV), and time per accepted prediction.</p>
    </div>
    <div class="step">
        <div class="step-title">3. Hypothesis</div>
        <p>We build an enforcement system where invalid outputs cannot pass. We test an <strong>enforcement ladder</strong>: E0 (no enforcement) → E1 (schema only) → E2 (schema + validation gate) → E3 (E2 + retry/repair). Only outputs that pass the chosen level are accepted.</p>
    </div>

    <div class="step" id="experiment-section">
        <div class="step-title">4. Experiment</div>
        <p>Upload a file with vehicles (Excel, CSV, TSV, or JSON). Choose <strong>Enforcement level</strong> (E0–E4) to test how much enforcement improves valid rate and reduces variance. Max <strong>{{ max_rows }} rows</strong> per upload.</p>
        <div class="upload">
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".xlsx,.xls,.csv,.tsv,.txt,.json,.jsonl" required>
                <br><br>
                <label>Enforcement:
                    <select name="enforcement_level">
                        {% for e in enforcement_levels %}
                        <option value="{{ e }}" {{ 'selected' if enforcement_default == e else '' }}>{{ e }} — {{ enforcement_descriptions.get(e, '') }}</option>
                        {% endfor %}
                    </select>
                </label>
                <label style="margin-left: 12px;">Provider:
                    <select name="provider">
                        <option value="openai" {{ 'selected' if provider_default == 'openai' else '' }}>OpenAI (GPT)</option>
                        <option value="claude" {{ 'selected' if provider_default == 'claude' else '' }}>Claude (Anthropic)</option>
                    </select>
                </label>
                <label style="margin-left: 12px;">Condition (P1–P4): <select name="condition">
                    <option value="P1">P1 — Internal only</option>
                    <option value="P2">P2 — Any source</option>
                    <option value="P3">P3 — Allowlist only</option>
                    <option value="P4">P4 — No web</option>
                </select></label>
                <br><br>
                <label><input type="checkbox" name="consistency_check" value="1"> Consistency check (first vehicle only, N repeats)</label>
                <label style="margin-left: 12px;">Repeats: <input type="number" name="repeats" value="{{ repeats_default }}" min="2" max="{{ max_repeats }}" style="width: 4em;"></label>
                <br><br>
                <label><input type="checkbox" name="use_hooks" value="1" {{ 'checked' if use_hooks_default else '' }}> Use enforcement hooks</label>
                <span style="font-size: 0.85rem; color: #6c757d;"> (PostPredictionHook: reject if confidence &lt; 0.3)</span>
                <br><br>
                <button type="submit" class="btn">Run experiment</button>
            </form>
        </div>
        {% if template_link %}
        <p>
            <a href="{{ template_link }}" class="btn btn-secondary">Download Excel Template</a>
            <a href="/create_template_csv" class="btn btn-secondary">Download CSV Template</a>
        </p>
        {% endif %}
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
    </div>

    {% if results %}
    <div class="step">
        <div class="step-title">5. Data collection</div>
        <div class="success">Processed {{ results|length }} vehicles{{ row_limit_msg|default("", true) }}. {% if row_limit_msg %}Use chunks of {{ max_rows }} rows or fewer for larger files.{% endif %}</div>
        <table>
            <thead>
                <tr>
                    {% if analysis.consistency_check %}<th>Repeat</th>{% endif %}
                    <th>Vehicle ID</th>
                    <th>Make</th>
                    <th>Model</th>
                    <th>Year</th>
                    <th>Mileage</th>
                    <th>Predicted price</th>
                    <th>Conf.</th>
                    <th>Subgroup</th>
                    <th>Valid</th>
                    {% if analysis.mean_retries is defined and analysis.mean_retries > 0 %}<th>Retries</th>{% endif %}
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                {% for r in results %}
                <tr>
                    {% if analysis.consistency_check %}<td>{{ r.repeat }}</td>{% endif %}
                    <td>{{ r.vehicle_id }}</td>
                    <td>{{ r.make }}</td>
                    <td>{{ r.model }}</td>
                    <td>{{ r.year }}</td>
                    <td>{{ r.mileage or 'N/A' }}</td>
                    <td>${{ "%.2f"|format(r.predicted_price) if r.predicted_price else 'N/A' }}</td>
                    <td>{{ "%.2f"|format(r.confidence) if r.confidence else 'N/A' }}</td>
                    <td>{{ r.subgroup_detected or 'N/A' }}</td>
                    <td class="{{ 'valid' if r.valid else 'invalid' }}">{{ 'Yes' if r.valid else 'No' }}</td>
                    {% if analysis.mean_retries is defined and analysis.mean_retries > 0 %}<td>{{ r.retry_count or 0 }}</td>{% endif %}
                    <td>{{ (r.notes or '')[:50] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% if download_link %}
        <p><a href="{{ download_link }}" class="btn">Download results (Excel)</a></p>
        {% endif %}
    </div>
    <div class="step">
        <div class="step-title">6. Analysis</div>
        {% if analysis.consistency_check %}
        <p>Consistency check: same vehicle, multiple repeats. <strong>CV (coefficient of variation)</strong> measures spread of predicted prices — lower = more deterministic.</p>
        {% else %}
        <p>Summary over this run (valid predictions only for price stats):</p>
        {% endif %}
        <div class="analysis-grid">
            <div class="analysis-item"><span class="value">{{ analysis.n_total }}</span><br><span class="label">Rows</span></div>
            <div class="analysis-item"><span class="value">{{ "%.0f"|format(analysis.valid_rate_pct) }}%</span><br><span class="label">Valid rate</span></div>
            <div class="analysis-item"><span class="value">{{ "%.2f"|format(analysis.mean_retries) if analysis.mean_retries is defined else '—' }}</span><br><span class="label">Mean retries</span></div>
            <div class="analysis-item"><span class="value">${{ "%.0f"|format(analysis.mean_price) if analysis.mean_price != none else '—' }}</span><br><span class="label">Mean price</span></div>
            <div class="analysis-item"><span class="value">${{ "%.0f"|format(analysis.std_price) if analysis.std_price != none else '—' }}</span><br><span class="label">Std price</span></div>
            <div class="analysis-item"><span class="value">{{ "%.1f"|format(analysis.cv) if analysis.cv != none else '—' }}%</span><br><span class="label">CV %</span></div>
        </div>
    </div>
    <div class="step">
        <div class="step-title">7. Conclusion</div>
        <p class="conclusion-note">Did valid rate and variance match what you expected for this condition? Use different conditions (P1–P4) or the same file again to test consistency.</p>
        <p><strong>8. Repeat / Refine</strong> — <a href="#experiment-section" class="btn">Run another experiment</a> (e.g. same file, different condition, or new file).</p>
    </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = _base_ctx()
    if request.method == "POST":
        if "file" not in request.files:
            return render_template_string(HTML_TEMPLATE, **ctx, error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template_string(HTML_TEMPLATE, **ctx, error="No file selected")
        allowed = (".xlsx", ".xls", ".csv", ".tsv", ".txt", ".json", ".jsonl")
        if not file.filename or not file.filename.lower().endswith(allowed):
            return render_template_string(HTML_TEMPLATE, **ctx, error=f"File must be one of: {', '.join(allowed)}")
        condition = request.form.get("condition", "P1")
        if condition not in CONDITIONS:
            condition = "P1"
        enforcement_level = (request.form.get("enforcement_level") or "E2").upper()
        if enforcement_level not in ENFORCEMENT_LEVELS:
            enforcement_level = "E2"
        provider_id = (request.form.get("provider") or "openai").lower()
        if provider_id not in PROVIDERS:
            provider_id = "openai"
        if provider_id == "claude" and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return render_template_string(
                HTML_TEMPLATE,
                **_base_ctx(provider_default=provider_id),
                error="ANTHROPIC_API_KEY is required for Claude. Set it in your environment (e.g. Railway).",
            )
        if provider_id == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
            return render_template_string(
                HTML_TEMPLATE,
                **_base_ctx(provider_default=provider_id),
                error="OPENAI_API_KEY is required for OpenAI. Set it in your environment (e.g. Railway).",
            )
        consistency_check = request.form.get("consistency_check") == "1"
        use_hooks = request.form.get("use_hooks") == "1"
        try:
            n_repeats = min(MAX_CONSISTENCY_REPEATS, max(2, int(request.form.get("repeats", CONSISTENCY_CHECK_REPEATS))))
        except (TypeError, ValueError):
            n_repeats = CONSISTENCY_CHECK_REPEATS
        post_hook = make_confidence_floor_hook(0.3) if use_hooks else None
        try:
            suffix = Path(file.filename).suffix or ".csv"
            if suffix.lower() not in (".xlsx", ".xls", ".csv", ".tsv", ".txt", ".json", ".jsonl"):
                suffix = ".csv"
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                tmp_path = Path(tmp.name)
                vehicles = load_from_file(tmp_path)
                tmp_path.unlink()
            if not vehicles:
                return render_template_string(HTML_TEMPLATE, **ctx, error="No vehicles found in file.")
            total_rows = len(vehicles)
            row_limit_msg = ""
            results = []
            if consistency_check:
                vehicle = vehicles[0]
                payloads, vs = run_consistency_check(
                    vehicle,
                    n_repeats=n_repeats,
                    condition_id=condition,
                    use_mock_llm=False,
                    project_root=PROJECT_ROOT,
                    post_hook=post_hook,
                    provider_id=provider_id,
                    enforcement_level=enforcement_level,
                )
                for i, payload in enumerate(payloads):
                    pred = payload.get("prediction") or {}
                    results.append({
                        "repeat": i + 1,
                        "vehicle_id": payload.get("vehicle_id", ""),
                        "make": payload.get("make", ""),
                        "model": payload.get("model", ""),
                        "year": payload.get("year", 0),
                        "mileage": payload.get("mileage"),
                        "predicted_price": pred.get("predicted_price"),
                        "confidence": pred.get("confidence"),
                        "subgroup_detected": pred.get("subgroup_detected"),
                        "valid": payload.get("valid", False),
                        "notes": pred.get("notes", ""),
                        "violation_reason": payload.get("violation_reason"),
                        "retry_count": payload.get("retry_count", 0),
                    })
                retries = [r.get("retry_count", 0) for r in results if isinstance(r.get("retry_count"), (int, float))]
                analysis = {
                    "n_total": len(results),
                    "valid_rate_pct": (1.0 - vs.get("invalid_rate", 0)) * 100.0,
                    "mean_price": vs.get("mean"),
                    "std_price": vs.get("std"),
                    "cv": vs.get("cv"),
                    "consistency_check": True,
                    "mean_retries": sum(retries) / len(retries) if retries else 0,
                }
                row_limit_msg = f" (consistency check: 1 vehicle × {n_repeats} repeats)"
            else:
                if total_rows > MAX_ROWS_PER_UPLOAD:
                    vehicles = vehicles[:MAX_ROWS_PER_UPLOAD]
                    row_limit_msg = f" (first {len(vehicles)} of {total_rows} rows)"
                for vehicle in vehicles:
                    payload = run_pipeline(
                        vehicle,
                        condition_id=condition,
                        use_mock_llm=False,
                        project_root=PROJECT_ROOT,
                        post_hook=post_hook,
                        provider_id=provider_id,
                        enforcement_level=enforcement_level,
                    )
                    pred = payload.get("prediction") or {}
                    results.append({
                        "repeat": None,
                        "vehicle_id": payload.get("vehicle_id", ""),
                        "make": payload.get("make", ""),
                        "model": payload.get("model", ""),
                        "year": payload.get("year", 0),
                        "mileage": payload.get("mileage"),
                        "predicted_price": pred.get("predicted_price"),
                        "confidence": pred.get("confidence"),
                        "subgroup_detected": pred.get("subgroup_detected"),
                        "valid": payload.get("valid", False),
                        "notes": pred.get("notes", ""),
                        "violation_reason": payload.get("violation_reason"),
                        "retry_count": payload.get("retry_count", 0),
                    })
                valid_prices = [r["predicted_price"] for r in results if r.get("valid") and r.get("predicted_price") is not None]
                vs = variance_stats(valid_prices) if valid_prices else {}
                n_total = len(results)
                valid_count = sum(1 for r in results if r.get("valid"))
                retries = [r.get("retry_count", 0) for r in results if isinstance(r.get("retry_count"), (int, float))]
                analysis = {
                    "n_total": n_total,
                    "valid_rate_pct": (valid_count / n_total * 100.0) if n_total else 0.0,
                    "mean_price": vs.get("mean") if vs else None,
                    "std_price": vs.get("std") if vs else None,
                    "cv": vs.get("cv") if vs else None,
                    "consistency_check": False,
                    "mean_retries": sum(retries) / len(retries) if retries else 0,
                }
            download_link = None
            if results:
                out_df = pd.DataFrame(results)
                out_name = f"results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.xlsx"
                out_path = TEMP_DIR / out_name
                out_df.to_excel(out_path, index=False, engine="openpyxl")
                download_link = f"/download/{out_name}"
            ctx["use_hooks_default"] = use_hooks
            ctx["provider_default"] = provider_id
            ctx["enforcement_default"] = enforcement_level
            return render_template_string(
                HTML_TEMPLATE,
                **ctx,
                results=results,
                download_link=download_link,
                row_limit_msg=row_limit_msg,
                analysis=analysis,
            )
        except Exception as e:
            err_msg = str(e)
            if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                err_msg = "Request timed out. Try a smaller file (max 25 rows) or fewer consistency-check repeats."
            return render_template_string(HTML_TEMPLATE, **ctx, error=f"Error processing file: {err_msg}")
    return render_template_string(HTML_TEMPLATE, **ctx)


@app.route("/download/<filename>")
def download(filename: str):
    # Serve generated files from temp dir, and allow legacy data/ files.
    for base in (TEMP_DIR, DATA_DIR):
        path = base / filename
        if path.exists():
            return send_file(str(path), as_attachment=True)
    return "File not found", 404


def _template_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"vehicle_id": "v1", "make": "BMW", "model": "M3", "year": 2020, "mileage": 25000, "price": 52000},
        {"vehicle_id": "v2", "make": "BMW", "model": "340i", "year": 2019, "mileage": 40000, "price": 32000},
        {"vehicle_id": "v3", "make": "BMW", "model": "328i", "year": 2018, "mileage": 55000, "price": 22000},
    ])


@app.route("/create_template")
def create_template():
    df = _template_df()
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="vehicles_template.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/create_template_csv")
def create_template_csv():
    df = _template_df()
    buf = BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="vehicles_template.csv",
        mimetype="text/csv",
    )


COMPARE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Comparison</title>
    <style>
        :root { --accent: #0d6efd; --valid: #198754; --invalid: #dc3545; --border: #dee2e6; --bg: #f8f9fa; }
        body { font-family: system-ui, -apple-system, sans-serif; max-width: 1060px; margin: 0 auto; padding: 24px; line-height: 1.5; color: #212529; }
        h1 { font-size: 1.5rem; }
        h2 { font-size: 1.15rem; color: #495057; margin-top: 1.5rem; }
        .subtitle { color: #6c757d; margin-bottom: 1.5rem; }
        .step { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
        .step-title { font-weight: 600; color: var(--accent); margin-bottom: 0.35rem; }
        .btn { background: var(--accent); color: #fff; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block; font-size: 0.95rem; }
        .btn:hover { filter: brightness(0.95); }
        .btn-secondary { background: #6c757d; }
        table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.85rem; }
        th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; }
        th { background: #e9ecef; font-weight: 600; }
        .col-header-llm { background: #e8f0fe; }
        .col-header-ml { background: #fef3e2; }
        .supported { color: var(--valid); font-weight: 700; }
        .not-supported { color: var(--invalid); font-weight: 700; }
        .partial { color: #fd7e14; font-weight: 700; }
        .error { color: var(--invalid); padding: 10px; background: #f8d7da; border-radius: 6px; margin: 10px 0; }
        .warning { padding: 10px; background: #fff3cd; border-radius: 6px; margin: 10px 0; color: #664d03; }
        .info { padding: 10px; background: #cfe2ff; border-radius: 6px; margin: 10px 0; color: #084298; }
        .metric-row td { text-align: right; font-variant-numeric: tabular-nums; }
        .metric-row td:first-child { text-align: left; }
        .winner { background: #d1e7dd; }
        .hypothesis { margin: 6px 0; padding: 8px 12px; border-radius: 6px; }
        .ml-note { font-size: 0.85rem; color: #664d03; background: #fff3cd; padding: 8px 12px; border-radius: 6px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Pipeline Comparison: Where Should the AI Stop?</h1>
    <p class="subtitle">LLM pipelines (E3, E5) vs traditional ML baselines (Random Forest, XGBoost)</p>
    <p><a href="/" class="btn btn-secondary">Back to main experiment</a></p>

    {% if not regression_available %}
    <div class="warning">Regression models not trained yet. SSH into the server and run: <code>python3 regression_baseline.py</code><br>The comparison will still run E3 vs E5 without the regression columns.</div>
    {% endif %}

    <div class="step">
        <div class="step-title">Hypotheses (stated before running)</div>
        <ul>
            <li><strong>H1:</strong> Pipeline B (E5) will have lower price CV than Pipeline A (E3)</li>
            <li><strong>H2:</strong> Pipeline B will have lower MAE against ground truth</li>
            <li><strong>H3:</strong> LLM feature assignments will be more consistent across repeats than LLM price outputs</li>
            <li><strong>H4:</strong> Pipeline A-prime (free features + formula) will perform worse than Pipeline B</li>
        </ul>
    </div>

    <div class="step">
        <div class="step-title">Run Comparison</div>
        <form method="post">
            <label>Provider:
                <select name="provider">
                    <option value="openai">OpenAI (GPT)</option>
                    <option value="claude">Claude (Anthropic)</option>
                </select>
            </label>
            <label style="margin-left: 12px;">Vehicles:
                <input type="number" name="n_vehicles" value="10" min="1" max="200" style="width: 5em;">
            </label>
            <label style="margin-left: 12px;">Repeats per vehicle:
                <input type="number" name="n_repeats" value="5" min="2" max="10" style="width: 4em;">
            </label>
            <label style="margin-left: 12px;"><input type="checkbox" name="mock" value="1"> Mock LLM (for testing)</label>
            <br><br>
            <button type="submit" class="btn">Run comparison (all pipelines)</button>
            <span style="font-size: 0.85rem; color: #6c757d; margin-left: 8px;">Temperature: {{ temperature }} (set in config)</span>
        </form>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
    </div>

    {% if summary %}
    <div class="step">
        <div class="step-title">Results: Head-to-Head Metrics</div>
        <p>{{ n_vehicles }} vehicles, {{ n_repeats }} repeats each, provider: {{ provider }}, temperature: {{ temperature }}</p>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th class="col-header-llm">A: E3<br><small>LLM predicts price</small></th>
                    <th class="col-header-llm">B: E5<br><small>LLM extracts features</small></th>
                    {% if has_regression %}
                    <th class="col-header-ml">C: Random Forest<br><small>Traditional ML</small></th>
                    <th class="col-header-ml">D: XGBoost<br><small>Traditional ML</small></th>
                    {% endif %}
                </tr>
            </thead>
            <tbody>
                {% for row in metric_rows %}
                <tr class="metric-row">
                    <td>{{ row.label }}</td>
                    <td class="{{ 'winner' if row.e3_wins else '' }}">{{ row.e3_val }}</td>
                    <td class="{{ 'winner' if row.e5_wins else '' }}">{{ row.e5_val }}</td>
                    {% if has_regression %}
                    <td class="{{ 'winner' if row.rf_wins else '' }}">{{ row.rf_val }}</td>
                    <td class="{{ 'winner' if row.xgb_wins else '' }}">{{ row.xgb_val }}</td>
                    {% endif %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% if has_regression %}
        <div class="ml-note">Note: Regression models (C, D) have access to the <strong>mmr</strong> feature (Manheim Market Report wholesale price estimate) that the LLM pipelines do not. This is a very strong predictor and the regression models lean on it heavily.</div>
        {% endif %}
    </div>

    <div class="step">
        <div class="step-title">Hypothesis Assessment</div>
        {% for h in hypotheses %}
        <div class="hypothesis" style="background: {{ '#d1e7dd' if h.status == 'SUPPORTED' else '#f8d7da' if h.status == 'NOT SUPPORTED' else '#fff3cd' }};">
            <strong>{{ h.id }}:</strong> {{ h.text }}
            <span class="{{ 'supported' if h.status == 'SUPPORTED' else 'not-supported' if h.status == 'NOT SUPPORTED' else 'partial' }}">{{ h.status }}</span>
            <br><span style="font-size: 0.85rem; color: #495057;">{{ h.evidence }}</span>
            {% if h.regression_note %}
            <br><span style="font-size: 0.82rem; color: #664d03;">Regression: {{ h.regression_note }}</span>
            {% endif %}
        </div>
        {% endfor %}
        {% if ml_summary_line %}
        <div class="ml-note" style="margin-top: 12px;">{{ ml_summary_line }}</div>
        {% endif %}
    </div>

    {% if elapsed %}
    <p style="font-size: 0.85rem; color: #6c757d;">Elapsed: {{ elapsed }}s</p>
    {% endif %}
    {% endif %}
</body>
</html>
"""


@app.route("/compare", methods=["GET", "POST"])
def compare_view():
    reg_avail = regression_models_available()
    ctx = {"temperature": COMPARISON_TEMPERATURE, "regression_available": reg_avail}
    if request.method == "POST":
        provider = (request.form.get("provider") or "openai").lower()
        try:
            n_vehicles = min(200, max(1, int(request.form.get("n_vehicles", 10))))
        except (TypeError, ValueError):
            n_vehicles = 10
        try:
            n_repeats = min(10, max(2, int(request.form.get("n_repeats", 5))))
        except (TypeError, ValueError):
            n_repeats = 5
        use_mock = request.form.get("mock") == "1"

        if provider == "claude" and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return render_template_string(COMPARE_TEMPLATE, **ctx, error="ANTHROPIC_API_KEY required for Claude.")
        if provider == "openai" and not use_mock and not os.environ.get("OPENAI_API_KEY", "").strip():
            return render_template_string(COMPARE_TEMPLATE, **ctx, error="OPENAI_API_KEY required for OpenAI.")

        try:
            import time as _time
            start = _time.time()
            vehicles = load_eval_data(limit=n_vehicles)
            result = run_comparison(
                vehicles=vehicles,
                n_repeats=n_repeats,
                provider=provider,
                use_mock=use_mock if use_mock else None,
            )
            elapsed = round(_time.time() - start, 1)
            summary = result["summary"]
            e3 = summary.get("E3", {})
            e5 = summary.get("E5", {})
            rf = summary.get("RF", {})
            xgb_s = summary.get("XGB", {})
            has_reg = bool(rf or xgb_s)

            metric_rows = []

            def _add(label, key, fmt=".2f", lower_better=True, na_for_reg=False):
                vals = {}
                for lbl, src in [("e3", e3), ("e5", e5), ("rf", rf), ("xgb", xgb_s)]:
                    vals[lbl] = src.get(key, "N/A") if src else "N/A"

                all_nums = {k: v for k, v in vals.items() if isinstance(v, (int, float))}
                if na_for_reg:
                    compare_nums = {k: v for k, v in all_nums.items() if k in ("e3", "e5")}
                else:
                    compare_nums = all_nums

                if lower_better and compare_nums:
                    best = min(compare_nums.values())
                elif not lower_better and compare_nums:
                    best = max(compare_nums.values())
                else:
                    best = None

                row = {"label": label}
                for lbl in ["e3", "e5", "rf", "xgb"]:
                    v = vals[lbl]
                    if na_for_reg and lbl in ("rf", "xgb"):
                        row[f"{lbl}_val"] = "N/A"
                        row[f"{lbl}_wins"] = False
                    elif isinstance(v, (int, float)):
                        row[f"{lbl}_val"] = f"{v:{fmt}}"
                        row[f"{lbl}_wins"] = best is not None and v == best and len([x for x in compare_nums.values() if x == best]) == 1
                    else:
                        row[f"{lbl}_val"] = str(v)
                        row[f"{lbl}_wins"] = False
                metric_rows.append(row)

            _add("Valid rate", "mean_valid_rate", ".4f", lower_better=False)
            _add("Mean retries", "mean_retries", ".2f", lower_better=True)
            _add("Price CV (%)", "mean_cv", ".2f", lower_better=True)
            _add("MAE ($)", "mean_mae", ".0f", lower_better=True)
            _add("Within 10% of actual (%)", "pct_within_10", ".1f", lower_better=False)

            stab = e5.get("mean_feature_stability")
            if stab is not None:
                metric_rows.append({
                    "label": "Feature stability (E5 only)",
                    "e3_val": "---", "e3_wins": False,
                    "e5_val": f"{stab:.4f}", "e5_wins": False,
                    "rf_val": "---", "rf_wins": False,
                    "xgb_val": "---", "xgb_wins": False,
                })

            metric_rows.append({
                "label": "Avg tokens/request",
                "e3_val": "~200-400", "e3_wins": False,
                "e5_val": "~200-400", "e5_wins": False,
                "rf_val": "N/A", "rf_wins": False,
                "xgb_val": "N/A", "xgb_wins": False,
            })

            h1_ok = e5.get("mean_cv", 999) < e3.get("mean_cv", 999)
            h2_ok = e5.get("mean_mae", 999) < e3.get("mean_mae", 999)
            h3_stab = e5.get("mean_feature_stability", 0)
            h3_ok = "SUPPORTED" if h3_stab > 0.8 else "PARTIAL" if h3_stab > 0.6 else "NOT SUPPORTED"

            h1_reg = "RF CV=0.00%, XGB CV=0.00% (deterministic — trivially zero, not because the model is better)" if has_reg else ""
            h2_reg = ""
            if has_reg:
                rf_mae = rf.get("mean_mae", "N/A")
                xgb_mae = xgb_s.get("mean_mae", "N/A")
                rf_s = f"${rf_mae:,.0f}" if isinstance(rf_mae, (int, float)) else "N/A"
                xgb_str = f"${xgb_mae:,.0f}" if isinstance(xgb_mae, (int, float)) else "N/A"
                h2_reg = f"RF MAE={rf_s}, XGB MAE={xgb_str} (note: regression has access to mmr feature)"

            hypotheses = [
                {"id": "H1", "text": "E5 lower CV than E3",
                 "status": "SUPPORTED" if h1_ok else "NOT SUPPORTED",
                 "evidence": f"E3 CV={e3.get('mean_cv', 'N/A'):.2f}%, E5 CV={e5.get('mean_cv', 'N/A'):.2f}%",
                 "regression_note": h1_reg},
                {"id": "H2", "text": "E5 lower MAE than E3",
                 "status": "SUPPORTED" if h2_ok else "NOT SUPPORTED",
                 "evidence": f"E3 MAE=${e3.get('mean_mae', 0):,.0f}, E5 MAE=${e5.get('mean_mae', 0):,.0f}",
                 "regression_note": h2_reg},
                {"id": "H3", "text": "Features more stable than prices",
                 "status": h3_ok,
                 "evidence": f"Feature stability={h3_stab:.4f}, E3 price CV={e3.get('mean_cv', 0):.2f}%",
                 "regression_note": "N/A for regression — feature stability only applies to LLM pipelines" if has_reg else ""},
                {"id": "H4", "text": "A-prime worse than B (enforcement placement matters)",
                 "status": "REQUIRES ABLATION",
                 "evidence": "Ablation run (Pipeline A-prime) not yet implemented in web UI",
                 "regression_note": "N/A for regression — this hypothesis is about LLM enforcement placement" if has_reg else ""},
            ]

            ml_summary_line = ""
            if has_reg:
                best_reg = min(rf.get("mean_mae", 999999), xgb_s.get("mean_mae", 999999))
                best_llm = min(e3.get("mean_mae", 999999), e5.get("mean_mae", 999999))
                if best_reg < best_llm * 0.8:
                    comp = "better"
                elif best_reg > best_llm * 1.2:
                    comp = "worse"
                else:
                    comp = "comparable"
                ml_summary_line = (
                    f"Traditional ML vs LLM: {comp} on accuracy, with the caveat that regression "
                    f"models have access to the mmr feature (a professional wholesale price estimate) "
                    f"that the LLM pipelines do not."
                )

            return render_template_string(
                COMPARE_TEMPLATE, **ctx,
                summary=summary, metric_rows=metric_rows, hypotheses=hypotheses,
                n_vehicles=n_vehicles, n_repeats=n_repeats,
                provider=provider, elapsed=elapsed,
                has_regression=has_reg, ml_summary_line=ml_summary_line,
                regression_available=reg_avail,
            )
        except Exception as e:
            return render_template_string(COMPARE_TEMPLATE, **ctx, error=f"Error: {e}")

    return render_template_string(COMPARE_TEMPLATE, **ctx)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
