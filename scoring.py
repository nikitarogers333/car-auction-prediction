"""
ScoringModule: MAE, MAPE, variance metrics, invalid rate.
"""

from __future__ import annotations

from typing import Any


def mae(predictions: list[float], ground_truth: list[float]) -> float:
    if not predictions or len(predictions) != len(ground_truth):
        return float("nan")
    return sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / len(predictions)


def mape(predictions: list[float], ground_truth: list[float], epsilon: float = 1e-6) -> float:
    if not predictions or len(predictions) != len(ground_truth):
        return float("nan")
    errors = []
    for p, g in zip(predictions, ground_truth):
        if abs(g) < epsilon:
            continue
        errors.append(abs(p - g) / abs(g))
    return (sum(errors) / len(errors)) * 100.0 if errors else float("nan")


def variance_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = var ** 0.5
    cv = (std / mean * 100.0) if mean else 0.0
    return {"mean": mean, "std": std, "cv": cv, "n": n}


def invalid_rate(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    inv = sum(1 for r in results if r.get("valid") is False)
    return inv / len(results)


def scoring_summary(
    predictions: list[float],
    ground_truth: list[float],
    results: list[dict[str, Any]],
    variance_stats_per_vehicle: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mae": mae(predictions, ground_truth),
        "mape": mape(predictions, ground_truth),
        "invalid_rate": invalid_rate(results),
        "n": len(predictions),
    }
    if variance_stats_per_vehicle:
        cvs = [v.get("cv", 0) for v in variance_stats_per_vehicle if isinstance(v.get("cv"), (int, float))]
        out["mean_cv"] = sum(cvs) / len(cvs) if cvs else 0.0
        out["unstable_count"] = sum(1 for v in variance_stats_per_vehicle if (v.get("cv") or 0) > 15.0)
    return out
