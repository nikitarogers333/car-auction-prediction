#!/usr/bin/env python3
"""
Deterministic, auditable LLM prediction system for car auction price prediction.
Requires OPENAI_API_KEY. Input: Excel (.xlsx) or JSON in data/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_logger import log_run
from baselines import build_feature_vector, linear_regression_predict, nearest_neighbors_predict
from config import CONDITIONS, DEFAULT_N_REPEATS, VARIANCE_CV_THRESHOLD
from data_loader import load_data as load_vehicle_data
from pipeline import run_consistency_check, run_pipeline
from scoring import invalid_rate, mae, mape, variance_stats


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"
LOGS_DIR = PROJECT_ROOT / "logs"


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("Error: OPENAI_API_KEY is required. Set it in your environment or .env file.", file=sys.stderr)
        print("Example: export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        sys.exit(1)


def load_data(path: Path | None) -> list[dict[str, Any]]:
    """Load from Excel (data/vehicles.xlsx) or JSON (data/vehicles.json)."""
    return load_vehicle_data(path=path, data_dir=DATA_DIR)


def run_single(args: argparse.Namespace) -> None:
    """Single vehicle, single run."""
    data = load_data(args.data)
    if not data:
        print("No data. Add data/vehicles.xlsx or data/vehicles.json (or use --data path).")
        return
    record = data[args.index] if args.index < len(data) else data[0]
    payload = run_pipeline(
        record,
        condition_id=args.condition,
        use_mock_llm=False,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(payload, indent=2, default=str))
    if args.log:
        raw_out = json.dumps(payload.get("prediction") or {})
        log_run(
            LOGS_DIR,
            vehicle_id=payload.get("vehicle_id", ""),
            condition=args.condition,
            repeat=0,
            prompt_used="",
            raw_output=raw_out,
            parsed_output=payload.get("prediction") or {},
            valid=payload.get("valid", False),
            violation_reason=payload.get("violation_reason"),
            variance_stats=None,
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.0,
        )
        print("Logged to", LOGS_DIR)


def run_consistency(args: argparse.Namespace) -> None:
    """N repeats per vehicle; variance stats."""
    data = load_data(args.data)
    if not data:
        print("No data")
        return
    n = args.repeats or DEFAULT_N_REPEATS
    record = data[args.index] if args.index < len(data) else data[0]
    results, vs = run_consistency_check(
        record,
        n_repeats=n,
        condition_id=args.condition,
        use_mock_llm=False,
        project_root=PROJECT_ROOT,
    )
    print("Variance stats:", json.dumps(vs, indent=2))
    print("Invalid rate:", vs.get("invalid_rate", 0))
    unstable = (vs.get("cv") or 0) > VARIANCE_CV_THRESHOLD
    print("Unstable (CV > threshold):", unstable)
    if args.log:
        for i, r in enumerate(results):
            log_run(
                LOGS_DIR,
                vehicle_id=r.get("vehicle_id", ""),
                condition=args.condition,
                repeat=i + 1,
                prompt_used="",
                raw_output=json.dumps(r.get("prediction") or {}),
                parsed_output=r.get("prediction") or {},
                valid=r.get("valid", False),
                violation_reason=r.get("violation_reason"),
                variance_stats=vs,
                model_name=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.0,
            )
        print("Logged to", LOGS_DIR)


def run_experiments(args: argparse.Namespace) -> None:
    """P1–P4 conditions; compare MAE, MAPE, variance, invalid rate."""
    data = load_data(args.data)
    if not data:
        print("No data")
        return
    n_repeats = args.repeats or 2
    limit = min(args.limit or len(data), len(data))
    results_by_condition: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}
    all_ground_truth: list[float] = []
    for i in range(limit):
        record = data[i]
        gt = record.get("price") or record.get("sale_price") or record.get("ground_truth")
        if gt is not None:
            all_ground_truth.append(float(gt))
        else:
            all_ground_truth.append(0.0)  # placeholder when no GT
        for cid in CONDITIONS:
            for rep in range(n_repeats):
                payload = run_pipeline(
                    record,
                    condition_id=cid,
                    use_mock_llm=False,
                    project_root=PROJECT_ROOT,
                )
                results_by_condition[cid].append(payload)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for cid in CONDITIONS:
        preds = []
        for p in results_by_condition[cid]:
            pr = p.get("prediction") or {}
            if p.get("valid") and pr.get("predicted_price") is not None:
                preds.append(float(pr["predicted_price"]))
            else:
                preds.append(0.0)
        # For MAE/MAPE we need same length as ground truth: one value per vehicle per condition (use first repeat or mean)
        n_vehicles = limit
        preds_per_vehicle = [preds[i * n_repeats : (i + 1) * n_repeats] for i in range(n_vehicles)]
        pred_means = [sum(x) / len(x) if x else 0 for x in preds_per_vehicle]
        gt = all_ground_truth[:n_vehicles]
        if len(gt) < len(pred_means):
            gt = gt + [0.0] * (len(pred_means) - len(gt))
        summary[cid] = {
            "mae": mae(pred_means, gt),
            "mape": mape(pred_means, gt),
            "invalid_rate": invalid_rate(results_by_condition[cid]),
            "n": len(results_by_condition[cid]),
        }
    out_path = EVAL_DIR / f"experiments_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))
    print("Wrote", out_path)


def run_baselines(args: argparse.Namespace) -> None:
    """Run KNN and regression baselines; optional compare to LLM."""
    data = load_data(args.data)
    if not data:
        print("No data")
        return
    import numpy as np
    numeric_keys = ["year", "mileage"]
    X = np.array([build_feature_vector(r, numeric_keys) for r in data])
    y = np.array([float(r.get("price") or r.get("sale_price") or 0) for r in data])
    if not y.any():
        y = np.array([35000.0] * len(data))  # dummy for demo
    preds_knn = []
    preds_lr = []
    for i in range(len(data)):
        train_idx = [j for j in range(len(data)) if j != i]
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_row = X[i]
        preds_knn.append(nearest_neighbors_predict(train_X, train_y, test_row, k=5))
        preds_lr.append(linear_regression_predict(train_X, train_y, test_row))
    mae_knn = mae(preds_knn, list(y))
    mae_lr = mae(preds_lr, list(y))
    print("Baselines (leave-one-out):")
    print("  KNN MAE:", mae_knn)
    print("  Linear Regression MAE:", mae_lr)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "baselines_summary.json", "w") as f:
        json.dump({"knn_mae": mae_knn, "lr_mae": mae_lr}, f, indent=2)
    print("Wrote", EVAL_DIR / "baselines_summary.json")


def main() -> None:
    require_openai_key()
    parser = argparse.ArgumentParser(description="Car auction price prediction pipeline (Excel or JSON input)")
    parser.add_argument("--data", type=Path, default=None, help="Input Excel (.xlsx) or JSON path (default: data/vehicles.xlsx or data/vehicles.json)")
    parser.add_argument("--condition", choices=list(CONDITIONS), default="P1", help="P1–P4")
    parser.add_argument("--log", action="store_true", help="Write audit log")
    parser.add_argument("--index", type=int, default=0, help="Vehicle index for single/consistency")
    parser.add_argument("--repeats", type=int, default=DEFAULT_N_REPEATS, help="N repeats for consistency")
    parser.add_argument("--limit", type=int, default=None, help="Max vehicles for experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("single", help="Single run one vehicle").set_defaults(func=run_single)
    sub.add_parser("consistency", help="N repeats, variance stats").set_defaults(func=run_consistency)
    sub.add_parser("experiments", help="P1–P4 comparison").set_defaults(func=run_experiments)
    sub.add_parser("baselines", help="KNN + regression baselines").set_defaults(func=run_baselines)
    args = parser.parse_args()
    if getattr(args, "data", None) is None:
        args.data = None
    args.func(args)


if __name__ == "__main__":
    main()
