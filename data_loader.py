"""
Load vehicle data from Excel (.xlsx) or JSON. Excel is the primary input format.
Expected columns (Excel): vehicle_id (or id), make, model, year, mileage, price (optional).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_from_excel(path: Path) -> list[dict[str, Any]]:
    """Load first sheet from .xlsx; normalize column names to lowercase with underscores."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Excel support requires pandas and openpyxl: pip install pandas openpyxl") from None
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    col_map = {"id": "vehicle_id", "sale_price": "price", "ground_truth": "price", "miles": "mileage"}
    rename = {k: v for k, v in col_map.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)
    if "vehicle_id" not in df.columns and "id" in df.columns:
        df["vehicle_id"] = df["id"].astype(str)
    elif "vehicle_id" not in df.columns:
        df["vehicle_id"] = [f"row_{i+1}" for i in range(len(df))]
    records = df.to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for r in records:
        row = {}
        for k, v in r.items():
            if pd.isna(v):
                continue
            if isinstance(v, (int, float)) and k in ("year", "mileage", "price") and float(v) == int(v):
                row[k] = int(v)
            else:
                row[k] = v
        out.append(row)
    return out


def load_data(path: Path | None = None, data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load from path or data_dir. Prefer Excel (vehicles.xlsx) then JSON (vehicles.json)."""
    base = Path(__file__).resolve().parent
    data_dir = data_dir or base / "data"
    if path is None:
        xlsx = data_dir / "vehicles.xlsx"
        json_path = data_dir / "vehicles.json"
        if xlsx.exists():
            path = xlsx
        elif json_path.exists():
            path = json_path
        else:
            return []
    path = Path(path)
    if not path.exists():
        return []
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return load_from_excel(path)
    if suf == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    if suf == ".json":
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    return []
