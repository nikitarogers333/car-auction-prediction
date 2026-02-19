"""
Load vehicle data from Excel (.xlsx, .xls), CSV (.csv), TSV (.tsv), or JSON (.json, .jsonl).
Column order and extra columns are fine; require at least make, model, year (recommended).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

# Column name aliases (any of these → canonical name)
COLUMN_ALIASES = {
    "id": "vehicle_id",
    "sale_price": "price",
    "ground_truth": "price",
    "miles": "mileage",
    "brand": "make",
    "trim": "model",
}


def _normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Lowercase, strip, replace spaces with underscores; apply aliases."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)
    if "vehicle_id" not in df.columns and "id" in df.columns:
        df["vehicle_id"] = df["id"].astype(str)
    elif "vehicle_id" not in df.columns:
        df["vehicle_id"] = [f"row_{i+1}" for i in range(len(df))]
    return df


def _records_from_df(df: "pd.DataFrame") -> list[dict[str, Any]]:
    """Convert DataFrame to list of dicts, drop NaN, coerce year/mileage/price to int where appropriate."""
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


def load_from_excel(path: Path) -> list[dict[str, Any]]:
    """Load first sheet from .xlsx or .xls."""
    if pd is None:
        raise ImportError("Excel/CSV support requires pandas (and openpyxl for Excel): pip install pandas openpyxl") from None
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = _normalize_columns(df)
    return _records_from_df(df)


def load_from_csv(path: Path, sep: str = ",") -> list[dict[str, Any]]:
    """Load CSV or TSV (sep=',' or '\\t')."""
    if pd is None:
        raise ImportError("CSV support requires pandas: pip install pandas") from None
    df = pd.read_csv(path, sep=sep, encoding="utf-8", encoding_errors="replace")
    df = _normalize_columns(df)
    return _records_from_df(df)


def load_from_file(path: Path) -> list[dict[str, Any]]:
    """
    Load vehicles from a file. Supported formats:
    - .xlsx, .xls  (Excel)
    - .csv         (comma-separated)
    - .tsv, .txt   (tab-separated, if suffix is .tsv or filename contains 'tsv')
    - .json        (array of objects)
    - .jsonl       (one JSON object per line)
    """
    path = Path(path)
    if not path.exists():
        return []
    suf = path.suffix.lower()
    name_lower = path.name.lower()

    if suf in (".xlsx", ".xls"):
        return load_from_excel(path)
    if suf == ".csv":
        return load_from_csv(path, sep=",")
    if suf == ".tsv" or "tsv" in name_lower:
        return load_from_csv(path, sep="\t")
    if suf == ".txt":
        try:
            if pd is None:
                raise ImportError("pandas required for .txt tables")
            # Auto-detect delimiter (tab or comma)
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", encoding_errors="replace")
            df = _normalize_columns(df)
            return _records_from_df(df)
        except Exception:
            return []
    if suf == ".jsonl":
        with open(path, encoding="utf-8", errors="replace") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suf == ".json":
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    return []


def load_data(path: Path | None = None, data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load from path or data_dir. Tries vehicles.xlsx, vehicles.csv, vehicles.json in that order."""
    base = Path(__file__).resolve().parent
    data_dir = data_dir or base / "data"
    if path is None:
        for name in ("vehicles.xlsx", "vehicles.csv", "vehicles.json"):
            candidate = data_dir / name
            if candidate.exists():
                return load_from_file(candidate)
        return []
    return load_from_file(Path(path))
