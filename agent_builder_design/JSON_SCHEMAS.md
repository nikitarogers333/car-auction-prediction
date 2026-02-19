# JSON Schemas for Pipeline

## PredictionAgent Output (Canonical)

```json
{
  "predicted_price": 12345,
  "confidence": 0.83,
  "method": "llm_internal",
  "subgroup_detected": "M3",
  "notes": "max 200 chars"
}
```

- **method** must be one of: `llm_internal` | `nearest_neighbors` | `regression`
- **notes** max 200 characters
- Anything else → INVALID

## Internal Payload (Between Steps)

```json
{
  "vehicle_id": "v1",
  "vin": null,
  "make": "BMW",
  "model": "M3",
  "year": 2020,
  "mileage": 25000,
  "features": { "year": 2020, "make": "BMW", "model": "M3", "mileage": 25000 },
  "subgroup": "M3",
  "allowed_comparables": ["M3", "M4"],
  "restrictions_applied": ["subgroup_classified", "restriction_enforcer"],
  "prediction": { ... },
  "valid": true,
  "violation_reason": null
}
```

## Audit Log Record

```json
{
  "vehicle_id": "...",
  "condition": "P1",
  "repeat": 1,
  "prompt_used": "...",
  "raw_output": "...",
  "parsed_output": { ... },
  "valid": true,
  "violation_reason": null,
  "variance_stats": { "mean": 45000, "std": 1200, "cv": 2.67, "n": 5 },
  "timestamp": "2026-02-17T22:00:00Z",
  "model_name": "gpt-4o-mini",
  "temperature": 0
}
```

## Subgroup Map Rule (restrictions/subgroup_map.json)

```json
{
  "make": "BMW",
  "model_pattern": "M3",
  "subgroup": "M3",
  "allowed_comparables": ["M3", "M4"],
  "forbidden_comparables": ["340i", "328i", "330i", "generic"]
}
```
