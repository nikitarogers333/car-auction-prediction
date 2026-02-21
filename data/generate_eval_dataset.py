#!/usr/bin/env python3
"""
Generate a synthetic eval dataset with ground truth prices.

IMPORTANT: The ground truth formula here is DELIBERATELY DIFFERENT from
pricing.compute_price_from_features(). If they were the same, Pipeline B (E5)
would win trivially by approximating the same function. We use polynomial terms,
interaction effects, and exponential depreciation to make this a real test.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

MODELS = {
    "M3":   {"trim": "performance", "base_msrp": 72000, "depreciation_k": 0.12},
    "M5":   {"trim": "performance", "base_msrp": 105000, "depreciation_k": 0.11},
    "340i": {"trim": "premium",     "base_msrp": 48000,  "depreciation_k": 0.14},
    "330i": {"trim": "mid",         "base_msrp": 42000,  "depreciation_k": 0.15},
    "328i": {"trim": "mid",         "base_msrp": 38000,  "depreciation_k": 0.16},
    "X3":   {"trim": "premium",     "base_msrp": 46000,  "depreciation_k": 0.13},
    "X5":   {"trim": "premium",     "base_msrp": 65000,  "depreciation_k": 0.12},
    "228i": {"trim": "base",        "base_msrp": 37000,  "depreciation_k": 0.17},
    "430i": {"trim": "mid",         "base_msrp": 46000,  "depreciation_k": 0.14},
    "M4":   {"trim": "performance", "base_msrp": 75000,  "depreciation_k": 0.12},
}


def ground_truth_price(
    model_key: str, year: int, mileage: int, condition: float, rng: random.Random
) -> float:
    """
    Ground truth pricing: exponential depreciation + mileage polynomial + interaction + noise.
    This is NOT the same as pricing.compute_price_from_features().
    """
    info = MODELS[model_key]
    age = max(0, 2025 - year)

    # Exponential depreciation (not linear like the pricing formula)
    dep_factor = math.exp(-info["depreciation_k"] * age)
    base = info["base_msrp"] * dep_factor

    # Mileage: quadratic penalty (not the bucket system in pricing.py)
    miles_10k = mileage / 10000.0
    mileage_penalty = 800 * miles_10k + 25 * miles_10k ** 2

    # Condition: polynomial bonus (not the linear multiplier in pricing.py)
    cond_bonus = (condition / 10.0) ** 1.5 * 8000

    # Interaction: high-mileage + old = extra penalty
    interaction = -200 * age * miles_10k if age > 3 and miles_10k > 4 else 0

    raw = base - mileage_penalty + cond_bonus + interaction

    # Gaussian noise (5% of raw)
    noise = rng.gauss(0, abs(raw) * 0.05)
    return max(2000, round(raw + noise, 2))


def generate_dataset(n: int = 200, seed: int = 12345) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    model_keys = list(MODELS.keys())
    for i in range(n):
        model_key = rng.choice(model_keys)
        year = rng.randint(2012, 2024)
        mileage = rng.randint(5000, 120000)
        condition = round(rng.uniform(3.0, 10.0), 1)
        price = ground_truth_price(model_key, year, mileage, condition, rng)
        rows.append({
            "vehicle_id": f"eval_{i+1:04d}",
            "make": "BMW",
            "model": model_key,
            "year": year,
            "mileage": mileage,
            "condition": condition,
            "price": price,
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["vehicle_id", "make", "model", "year", "mileage", "condition", "price"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "eval_dataset.csv"
    rows = generate_dataset(200)
    write_csv(rows, out)
    prices = [r["price"] for r in rows]
    print(f"Generated {len(rows)} vehicles -> {out}")
    print(f"  Price range: ${min(prices):,.0f} – ${max(prices):,.0f}")
    print(f"  Mean: ${sum(prices)/len(prices):,.0f}")
