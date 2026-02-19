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

from config import CONDITIONS
from data_loader import load_from_file
from pipeline import run_pipeline

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "car_auction_predict"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Each row calls the OpenAI API; limit rows so the request doesn't time out (e.g. 25 × ~6s ≈ 2.5 min)
MAX_ROWS_PER_UPLOAD = 25


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("Error: OPENAI_API_KEY is required. Set it in Railway environment variables.", file=sys.stderr)
        sys.exit(1)


require_openai_key()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Auction Price Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; }
        h1 { color: #333; }
        .upload { border: 2px dashed #ccc; padding: 30px; text-align: center; margin: 20px 0; border-radius: 8px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { background: #0056b3; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
        .valid { color: green; }
        .invalid { color: red; }
        .error { color: red; padding: 10px; background: #ffe6e6; border-radius: 4px; margin: 10px 0; }
        .success { color: green; padding: 10px; background: #e6ffe6; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Car Auction Price Prediction Pipeline</h1>
    <p>Upload a file: Excel (.xlsx, .xls), CSV (.csv), TSV (.tsv), or JSON (.json, .jsonl). Columns: vehicle_id (or id), make, model, year, mileage, price (optional). Order doesn't matter. <strong>Max {{ max_rows }} rows per upload</strong> (each row calls the API; larger files will time out).</p>
    
    <div class="upload">
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".xlsx,.xls,.csv,.tsv,.txt,.json,.jsonl" required>
            <br><br>
            <label>Condition: 
                <select name="condition">
                    <option value="P1">P1 - Internal knowledge only</option>
                    <option value="P2">P2 - Any source allowed</option>
                    <option value="P3">P3 - Allowlist domains only</option>
                    <option value="P4">P4 - No web access enforced</option>
                </select>
            </label>
            <br><br>
            <button type="submit" class="btn">Run Predictions</button>
        </form>
    </div>
    
    {% if template_link %}
    <p>
        <a href="{{ template_link }}" class="btn">Download Excel Template</a>
        <a href="/create_template_csv" class="btn">Download CSV Template</a>
        — or use your own file (CSV, TSV, JSON) with make, model, year.
    </p>
    {% endif %}
    
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    
    {% if results %}
    <div class="success">Processed {{ results|length }} vehicles{{ row_limit_msg|default("", true) }}. {% if row_limit_msg %}Split large files into chunks of {{ max_rows }} rows or fewer.{% endif %}</div>
    <h2>Results</h2>
    <table>
        <thead>
            <tr>
                <th>Vehicle ID</th>
                <th>Make</th>
                <th>Model</th>
                <th>Year</th>
                <th>Mileage</th>
                <th>Predicted Price</th>
                <th>Confidence</th>
                <th>Subgroup</th>
                <th>Valid</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
            {% for r in results %}
            <tr>
                <td>{{ r.vehicle_id }}</td>
                <td>{{ r.make }}</td>
                <td>{{ r.model }}</td>
                <td>{{ r.year }}</td>
                <td>{{ r.mileage or 'N/A' }}</td>
                <td>${{ "%.2f"|format(r.predicted_price) if r.predicted_price else 'N/A' }}</td>
                <td>{{ "%.2f"|format(r.confidence) if r.confidence else 'N/A' }}</td>
                <td>{{ r.subgroup_detected or 'N/A' }}</td>
                <td class="{{ 'valid' if r.valid else 'invalid' }}">{{ 'Yes' if r.valid else 'No' }}</td>
                <td>{{ r.notes[:50] if r.notes else '' }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% if download_link %}
    <p><a href="{{ download_link }}" class="btn">Download Results as Excel</a></p>
    {% endif %}
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template_string(HTML_TEMPLATE, error="No file uploaded", template_link="/create_template", max_rows=MAX_ROWS_PER_UPLOAD)
        file = request.files["file"]
        if file.filename == "":
            return render_template_string(HTML_TEMPLATE, error="No file selected", template_link="/create_template", max_rows=MAX_ROWS_PER_UPLOAD)
        allowed = (".xlsx", ".xls", ".csv", ".tsv", ".txt", ".json", ".jsonl")
        if not file.filename or not file.filename.lower().endswith(allowed):
            return render_template_string(HTML_TEMPLATE, error=f"File must be one of: {', '.join(allowed)}", template_link="/create_template", max_rows=MAX_ROWS_PER_UPLOAD)
        condition = request.form.get("condition", "P1")
        if condition not in CONDITIONS:
            condition = "P1"
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
                return render_template_string(HTML_TEMPLATE, error="No vehicles found in file.", template_link="/create_template", max_rows=MAX_ROWS_PER_UPLOAD)
            total_rows = len(vehicles)
            row_limit_msg = ""
            if total_rows > MAX_ROWS_PER_UPLOAD:
                vehicles = vehicles[:MAX_ROWS_PER_UPLOAD]
                row_limit_msg = f" (first {len(vehicles)} of {total_rows} rows)"
            results = []
            for vehicle in vehicles:
                payload = run_pipeline(vehicle, condition_id=condition, use_mock_llm=False, project_root=PROJECT_ROOT)
                pred = payload.get("prediction") or {}
                results.append({
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
            download_link = None
            if results:
                out_df = pd.DataFrame(results)
                out_name = f"results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.xlsx"
                out_path = TEMP_DIR / out_name
                out_df.to_excel(out_path, index=False, engine="openpyxl")
                download_link = f"/download/{out_name}"
            return render_template_string(
                HTML_TEMPLATE,
                results=results,
                download_link=download_link,
                template_link="/create_template",
                max_rows=MAX_ROWS_PER_UPLOAD,
                row_limit_msg=row_limit_msg,
            )
        except Exception as e:
            err_msg = str(e)
            if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                err_msg = "Request timed out. Try a smaller file (max 25 rows recommended)."
            return render_template_string(
                HTML_TEMPLATE,
                error=f"Error processing file: {err_msg}",
                template_link="/create_template",
                max_rows=MAX_ROWS_PER_UPLOAD,
            )
    return render_template_string(HTML_TEMPLATE, template_link="/create_template", max_rows=MAX_ROWS_PER_UPLOAD)


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
