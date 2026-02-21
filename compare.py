"""
Comparison runner: Pipeline A (E3) vs Pipeline B (E5) vs Pipeline A' (ablation).

Runs every vehicle through each pipeline N times, collects metrics, computes
feature consistency, and outputs a head-to-head summary.

Usage:
    python3 compare.py                          # full run (requires API keys)
    python3 compare.py --mock                   # mock LLM (for testing)
    python3 compare.py --provider claude        # use Claude
    python3 compare.py --n-vehicles 20 --n-repeats 3  # smaller run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import COMPARISON_TEMPERATURE, DEFAULT_N_REPEATS
from pipeline import run_pipeline
from scoring import mae, variance_stats

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "eval_dataset.csv"

# Features whose consistency we track for E5
CATEGORICAL_FEATURES = ["market_demand", "trim_tier", "depreciation_rate",
                        "mileage_assessment", "comparable_market"]
NUMERIC_FEATURES = ["condition_score"]


def load_eval_data(path: Path | None = None, limit: int | None = None) -> list[dict]:
    p = path or EVAL_DATA_PATH
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["year"] = int(row["year"])
            row["mileage"] = int(row["mileage"])
            row["condition"] = float(row["condition"])
            row["price"] = float(row["price"])
            rows.append(row)
    if limit:
        rows = rows[:limit]
    return rows


def _run_vehicle_n_times(
    vehicle: dict,
    enforcement_level: str,
    n_repeats: int,
    provider: str,
    use_mock: bool | None,
) -> list[dict]:
    """Run one vehicle through the pipeline n_repeats times."""
    raw = {
        "vehicle_id": vehicle["vehicle_id"],
        "make": vehicle["make"],
        "model": vehicle["model"],
        "year": vehicle["year"],
        "mileage": vehicle["mileage"],
    }
    results = []
    for _ in range(n_repeats):
        payload = run_pipeline(
            raw,
            condition_id="P1",
            use_mock_llm=use_mock,
            project_root=PROJECT_ROOT,
            provider_id=provider,
            enforcement_level=enforcement_level,
            temperature_override=COMPARISON_TEMPERATURE,
        )
        results.append(payload)
    return results


def _extract_prices(results: list[dict]) -> list[float]:
    prices = []
    for r in results:
        pred = r.get("prediction") or {}
        p = pred.get("predicted_price")
        if p is not None and r.get("valid"):
            prices.append(float(p))
    return prices


def _feature_consistency(results: list[dict]) -> dict[str, Any]:
    """Measure how consistently the LLM assigns features across repeats."""
    if not results:
        return {"overall_stability": 0.0}

    features_per_run = [r.get("extracted_features") or {} for r in results]
    n = len(features_per_run)
    if n < 2:
        return {"overall_stability": 1.0}

    cat_scores = {}
    for feat in CATEGORICAL_FEATURES:
        values = [f.get(feat) for f in features_per_run if f.get(feat) is not None]
        if not values:
            cat_scores[feat] = 0.0
            continue
        most_common_count = Counter(values).most_common(1)[0][1]
        cat_scores[feat] = most_common_count / len(values)

    num_scores = {}
    for feat in NUMERIC_FEATURES:
        values = [f.get(feat) for f in features_per_run
                  if isinstance(f.get(feat), (int, float))]
        if len(values) < 2:
            num_scores[feat] = 1.0
            continue
        vs = variance_stats(values)
        num_scores[feat] = max(0.0, 1.0 - vs["cv"] / 100.0)

    all_scores = list(cat_scores.values()) + list(num_scores.values())
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "categorical": cat_scores,
        "numeric": num_scores,
        "overall_stability": round(overall, 4),
    }


def run_comparison(
    vehicles: list[dict],
    n_repeats: int = 5,
    provider: str = "openai",
    use_mock: bool | None = None,
    run_ablation: bool = True,
) -> dict[str, Any]:
    """
    Run Pipeline A (E3), Pipeline B (E5), and optionally A' on all vehicles.
    Returns structured results with per-vehicle and aggregate metrics.
    """
    pipelines = {"E3": "E3", "E5": "E5"}
    results: dict[str, list[dict]] = {k: [] for k in pipelines}
    vehicle_metrics: dict[str, list[dict]] = {k: [] for k in pipelines}

    total = len(vehicles)
    for idx, v in enumerate(vehicles, 1):
        vid = v["vehicle_id"]
        gt = v["price"]
        print(f"  [{idx}/{total}] {vid} ({v['make']} {v['model']} {v['year']})...", flush=True)

        for label, level in pipelines.items():
            runs = _run_vehicle_n_times(v, level, n_repeats, provider, use_mock)
            results[label].extend(runs)
            prices = _extract_prices(runs)
            vs = variance_stats(prices) if prices else {"mean": 0, "std": 0, "cv": 0, "n": 0}
            valid_rate = sum(1 for r in runs if r.get("valid")) / len(runs)
            retries = [r.get("retry_count", 0) for r in runs]
            mean_retries = sum(retries) / len(retries) if retries else 0

            vm: dict[str, Any] = {
                "vehicle_id": vid,
                "ground_truth": gt,
                "mean_price": vs["mean"],
                "std_price": vs["std"],
                "cv": vs["cv"],
                "valid_rate": valid_rate,
                "mean_retries": mean_retries,
                "n_prices": len(prices),
            }

            if level == "E5":
                fc = _feature_consistency(runs)
                vm["feature_stability"] = fc["overall_stability"]
                vm["feature_detail"] = fc

            if prices:
                vm["mae_single"] = abs(vs["mean"] - gt)
                vm["within_10pct"] = 1 if abs(vs["mean"] - gt) / max(gt, 1) <= 0.10 else 0
            else:
                vm["mae_single"] = None
                vm["within_10pct"] = 0

            vehicle_metrics[label].append(vm)

    summary = _compute_summary(vehicle_metrics, pipelines)
    return {
        "pipelines": list(pipelines.keys()),
        "n_vehicles": total,
        "n_repeats": n_repeats,
        "provider": provider,
        "temperature": COMPARISON_TEMPERATURE,
        "vehicle_metrics": vehicle_metrics,
        "summary": summary,
    }


def _compute_summary(
    vehicle_metrics: dict[str, list[dict]],
    pipelines: dict[str, str],
) -> dict[str, dict[str, Any]]:
    summary = {}
    for label in pipelines:
        vms = vehicle_metrics[label]
        if not vms:
            continue
        n = len(vms)

        valid_rates = [v["valid_rate"] for v in vms]
        cvs = [v["cv"] for v in vms if v["n_prices"] > 0]
        maes = [v["mae_single"] for v in vms if v["mae_single"] is not None]
        within10 = [v["within_10pct"] for v in vms]
        retries = [v["mean_retries"] for v in vms]

        s: dict[str, Any] = {
            "n_vehicles": n,
            "mean_valid_rate": round(sum(valid_rates) / n, 4) if n else 0,
            "mean_cv": round(sum(cvs) / len(cvs), 2) if cvs else 0,
            "mean_mae": round(sum(maes) / len(maes), 2) if maes else 0,
            "pct_within_10": round(sum(within10) / n * 100, 1) if n else 0,
            "mean_retries": round(sum(retries) / n, 2) if n else 0,
        }

        if label == "E5":
            stabs = [v.get("feature_stability", 0) for v in vms]
            s["mean_feature_stability"] = round(sum(stabs) / len(stabs), 4) if stabs else 0

        summary[label] = s
    return summary


def print_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"COMPARISON: Pipeline A (E3) vs Pipeline B (E5)")
    print(f"  Vehicles: {result['n_vehicles']}  |  Repeats: {result['n_repeats']}  "
          f"|  Provider: {result['provider']}  |  Temperature: {result['temperature']}")
    print("=" * 72)

    summary = result["summary"]

    header = f"{'Metric':<30} {'E3 (direct)':>15} {'E5 (features)':>15}"
    print(header)
    print("-" * len(header))

    def _row(label: str, key: str, fmt: str = ".2f"):
        e3_val = summary.get("E3", {}).get(key, "N/A")
        e5_val = summary.get("E5", {}).get(key, "N/A")
        e3_s = f"{e3_val:{fmt}}" if isinstance(e3_val, (int, float)) else str(e3_val)
        e5_s = f"{e5_val:{fmt}}" if isinstance(e5_val, (int, float)) else str(e5_val)
        print(f"{label:<30} {e3_s:>15} {e5_s:>15}")

    _row("Valid rate", "mean_valid_rate", ".4f")
    _row("Mean retries", "mean_retries", ".2f")
    _row("Price CV (%)", "mean_cv", ".2f")
    _row("MAE ($)", "mean_mae", ".0f")
    _row("Within 10% of actual (%)", "pct_within_10", ".1f")

    e5_stab = summary.get("E5", {}).get("mean_feature_stability")
    if e5_stab is not None:
        print(f"{'Feature stability (E5 only)':<30} {'---':>15} {e5_stab:>15.4f}")

    print("=" * 72)

    # H1-H4 assessment
    print("\nHypothesis assessment:")
    e3 = summary.get("E3", {})
    e5 = summary.get("E5", {})

    h1 = e5.get("mean_cv", 999) < e3.get("mean_cv", 999)
    h2 = e5.get("mean_mae", 999) < e3.get("mean_mae", 999)
    h3_stab = e5.get("mean_feature_stability", 0)
    h3_e3cv = e3.get("mean_cv", 0)

    print(f"  H1 (E5 lower CV):             {'SUPPORTED' if h1 else 'NOT SUPPORTED'}  (E3={e3.get('mean_cv', 'N/A'):.2f}%, E5={e5.get('mean_cv', 'N/A'):.2f}%)")
    print(f"  H2 (E5 lower MAE):            {'SUPPORTED' if h2 else 'NOT SUPPORTED'}  (E3=${e3.get('mean_mae', 'N/A'):,.0f}, E5=${e5.get('mean_mae', 'N/A'):,.0f})")
    print(f"  H3 (features more stable):     {'SUPPORTED' if h3_stab > 0.8 else 'PARTIAL' if h3_stab > 0.6 else 'NOT SUPPORTED'}  (stability={h3_stab:.4f}, E3 price CV={h3_e3cv:.2f}%)")
    print(f"  H4 (A' worse than B):          REQUIRES ABLATION RUN")
    print()


def save_results(result: dict[str, Any], path: Path | None = None) -> Path:
    out = path or (PROJECT_ROOT / "data" / "comparison_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="E3 vs E5 comparison experiment")
    parser.add_argument("--data", type=str, default=None, help="Path to eval CSV")
    parser.add_argument("--n-vehicles", type=int, default=None, help="Limit vehicles")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "claude"])
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else None
    vehicles = load_eval_data(data_path, limit=args.n_vehicles)
    print(f"Loaded {len(vehicles)} vehicles from {data_path or EVAL_DATA_PATH}")
    print(f"Running E3 vs E5, {args.n_repeats} repeats each, temperature={COMPARISON_TEMPERATURE}")
    print()

    start = time.time()
    result = run_comparison(
        vehicles=vehicles,
        n_repeats=args.n_repeats,
        provider=args.provider,
        use_mock=args.mock if args.mock else None,
    )
    elapsed = time.time() - start
    result["elapsed_seconds"] = round(elapsed, 1)

    print_summary(result)
    out_path = Path(args.output) if args.output else None
    save_results(result, out_path)
    print(f"Total time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
