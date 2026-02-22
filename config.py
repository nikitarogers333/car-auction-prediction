"""
Experiment conditions P1–P4 and deterministic controls.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

# Railway: use RAILWAY_VOLUME_MOUNT_PATH if set, else /data if it exists, else project data/
_REG_ROOT = Path(__file__).resolve().parent
_mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
REGRESSION_DATA_DIR = Path(_mount) if _mount else (Path("/data") if Path("/data").exists() else _REG_ROOT / "data")
from typing import Any

# P1: Internal knowledge only — no external tools
# P2: Any source allowed
# P3: Allowlist domains only
# P4: No web access enforced (same as P1 but explicit)

@dataclass
class ExperimentCondition:
    id: str
    allow_external: bool
    allowlist_only: bool
    description: str


CONDITIONS = {
    "P1": ExperimentCondition("P1", allow_external=False, allowlist_only=False, description="Internal knowledge only"),
    "P2": ExperimentCondition("P2", allow_external=True, allowlist_only=False, description="Any source allowed"),
    "P3": ExperimentCondition("P3", allow_external=True, allowlist_only=True, description="Allowlist domains only"),
    "P4": ExperimentCondition("P4", allow_external=False, allowlist_only=False, description="No web access enforced"),
}

# Determinism (for single-run predictions)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 42
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-20241022"

# Comparison experiments: temperature > 0 so variance is real.
# If temperature=0, perfect consistency is trivial and the argument collapses.
COMPARISON_TEMPERATURE = 0.7

# Providers (for UI: OpenAI vs Claude)
PROVIDERS = ("openai", "claude")

# Enforcement ladder: how much we enforce before accepting an output
# E0 = no enforcement (accept if parseable), E1 = schema only, E2 = schema + validation gate,
# E3 = E2 + retry/repair loop, E4 = E3 + verifier, E5 = deterministic pricing from LLM features
ENFORCEMENT_LEVELS = ("E0", "E1", "E2", "E3", "E4", "E5", "APRIME")
ENFORCEMENT_DESCRIPTIONS = {
    "E0": "No enforcement (accept if parseable)",
    "E1": "Schema only",
    "E2": "Schema + validation gate",
    "E3": "Schema + validation + retry/repair",
    "E4": "E3 + verifier model",
    "E5": "LLM extracts features, code computes price",
    "APRIME": "Free-form LLM + parser + same pricing formula (H4 ablation)",
}
MAX_VALIDATION_RETRIES = 3  # for E3, E4

# Consistency testing
DEFAULT_N_REPEATS = 5
VARIANCE_CV_THRESHOLD = 15.0  # flag if CV% > this
