"""
Experiment conditions P1–P4 and deterministic controls.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# Determinism
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 42
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CLAUDE_MODEL = "claude-3-5-haiku-20241022"

# Providers (for UI: OpenAI vs Claude)
PROVIDERS = ("openai", "claude")

# Consistency testing
DEFAULT_N_REPEATS = 5
VARIANCE_CV_THRESHOLD = 15.0  # flag if CV% > this
