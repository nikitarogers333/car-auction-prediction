#!/usr/bin/env python3
"""
Eval harness: run pipeline on a fixed dataset (mock mode) and assert schema + valid + bounds.
Use for regression testing. Exit 0 if all pass, 1 otherwise.
  python3 scripts/run_eval.py
  python3 scripts/run_eval.py --data data/vehicles.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import run_pipeline
from schemas import PREDICTION_METHODS, validate_prediction_output


# Default eval set (subset of data/vehicles.json)
DEFAULT_EVAL = [
    {"vehicle_id": "v1", "make": "BMW", "model": "M3", "year": 2020, "mileage": 25000},
    {"vehicle_id": "v2", "make": "BMW", "model": "340i", "year": 2019, "mileage": 40000},
    {"vehicle_id": "v3", "make": "BMW", "model": "328i", "year": 2018, "mileage": 55000},
]


def load_data(path: Path | None) -> list[dict]:
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    return DEFAULT_EVAL


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval harness: mock pipeline run + schema/bounds checks")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "vehicles.json", help="JSON array of vehicles")
    ap.add_argument("--mock", action="store_true", default=True, help="Use mock predictor (default)")
    ap.add_argument("--no-mock", action="store_false", dest="mock", help="Use real API if OPENAI_API_KEY set")
    args = ap.parse_args()

    data = load_data(args.data)
    if not data:
        print("No vehicles to evaluate", file=sys.stderr)
        return 1

    errors: list[str] = []
    for i, vehicle in enumerate(data):
        vid = vehicle.get("vehicle_id", f"row_{i}")
        payload = run_pipeline(
            vehicle,
            condition_id="P1",
            use_mock_llm=args.mock,
            project_root=PROJECT_ROOT,
        )
        pred = payload.get("prediction")
        if not pred:
            errors.append(f"{vid}: missing 'prediction'")
            continue
        ok, msg = validate_prediction_output(pred)
        if not ok:
            errors.append(f"{vid}: schema — {msg}")
            continue
        if not payload.get("valid", True):
            errors.append(f"{vid}: pipeline marked valid=false — {payload.get('violation_reason', 'unknown')}")
            continue
        price = pred.get("predicted_price")
        if price is not None and (price < 0 or price > 1_000_000):
            errors.append(f"{vid}: predicted_price {price} outside sanity bounds [0, 1e6]")
        if pred.get("method") not in PREDICTION_METHODS:
            errors.append(f"{vid}: method not in {PREDICTION_METHODS}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print(f"Eval OK: {len(data)} vehicles passed schema, valid, and bounds checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
