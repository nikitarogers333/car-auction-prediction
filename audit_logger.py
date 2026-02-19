"""
Logging: audit trail for every run. Writes to /logs/YYYYMMDD_HHMMSS.jsonl
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_run(
    log_dir: Path,
    vehicle_id: str,
    condition: str,
    repeat: int,
    prompt_used: str,
    raw_output: str,
    parsed_output: dict[str, Any],
    valid: bool,
    violation_reason: str | None,
    variance_stats: dict[str, Any] | None,
    timestamp: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    **extra: Any,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # One file per experiment batch (same timestamp prefix)
    log_file = log_dir / f"{ts}.jsonl"
    record = {
        "vehicle_id": vehicle_id,
        "condition": condition,
        "repeat": repeat,
        "prompt_used": prompt_used[:500] if prompt_used else "",
        "raw_output": raw_output[:2000] if raw_output else "",
        "parsed_output": parsed_output,
        "valid": valid,
        "violation_reason": violation_reason,
        "variance_stats": variance_stats,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "temperature": temperature,
        **extra,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return log_file
