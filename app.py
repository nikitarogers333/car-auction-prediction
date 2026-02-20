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

from config import CONDITIONS, PROVIDERS
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


def _base_ctx(use_hooks_default: bool = True, provider_default: str = "openai"):
    return {
        "template_link": "/create_template",
        "max_rows": MAX_ROWS_PER_UPLOAD,
        "repeats_default": CONSISTENCY_CHECK_REPEATS,
        "max_repeats": MAX_CONSISTENCY_REPEATS,
        "use_hooks_default": use_hooks_default,
        "provider_default": provider_default,
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

    <div class="step">
        <div class="step-title">1. Observation</div>
        <p>Same prompt → different predicted values each time. Asking the model to use only certain sources does not guarantee it will obey. We need ways to get consistent, rule-following behavior.</p>
    </div>
    <div class="step">
        <div class="step-title">2. Question</div>
        <p>How can we make LLM predictions (e.g. auction price) <strong>deterministic</strong> (consistent) and <strong>constrained</strong> (follow our rules), and how do we <strong>check quality</strong> so we only accept valid outputs?</p>
    </div>
    <div class="step">
        <div class="step-title">3. Hypothesis</div>
        <p>We can combine: (1) <strong>structured output</strong> (strict JSON), (2) <strong>deterministic settings</strong> (temperature=0, seed), (3) a <strong>validation gate</strong> (reject invalid outputs in code), and (4) <strong>hook-style checks</strong> (pre/post/stop) to enforce constraints. Only outputs that pass all checks are accepted.</p>
    </div>

    <div class="step" id="experiment-section">
        <div class="step-title">4. Experiment</div>
        <p>Upload a file with vehicles (Excel, CSV, TSV, or JSON). Choose a <strong>condition</strong> (P1–P4) to vary access rules. Each row is one prediction run. Max <strong>{{ max_rows }} rows</strong> per upload (each row calls the API).</p>
        <p>Or run a <strong>consistency check</strong>: same vehicle, multiple repeats — to measure variance (non-determinism).</p>
        <div class="upload">
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".xlsx,.xls,.csv,.tsv,.txt,.json,.jsonl" required>
                <br><br>
                <label>Provider:
                    <select name="provider">
                        <option value="openai" {{ 'selected' if provider_default == 'openai' else '' }}>OpenAI (GPT)</option>
                        <option value="claude" {{ 'selected' if provider_default == 'claude' else '' }}>Claude (Anthropic)</option>
                    </select>
                </label>
                <label style="margin-left: 12px;">Condition:
                    <select name="condition">
                        <option value="P1">P1 — Internal knowledge only</option>
                        <option value="P2">P2 — Any source allowed</option>
                        <option value="P3">P3 — Allowlist domains only</option>
                        <option value="P4">P4 — No web access enforced</option>
                    </select>
                </label>
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
                    })
                analysis = {
                    "n_total": len(results),
                    "valid_rate_pct": (1.0 - vs.get("invalid_rate", 0)) * 100.0,
                    "mean_price": vs.get("mean"),
                    "std_price": vs.get("std"),
                    "cv": vs.get("cv"),
                    "consistency_check": True,
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
                    })
                valid_prices = [r["predicted_price"] for r in results if r.get("valid") and r.get("predicted_price") is not None]
                vs = variance_stats(valid_prices) if valid_prices else {}
                n_total = len(results)
                valid_count = sum(1 for r in results if r.get("valid"))
                analysis = {
                    "n_total": n_total,
                    "valid_rate_pct": (valid_count / n_total * 100.0) if n_total else 0.0,
                    "mean_price": vs.get("mean") if vs else None,
                    "std_price": vs.get("std") if vs else None,
                    "cv": vs.get("cv") if vs else None,
                    "consistency_check": False,
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
