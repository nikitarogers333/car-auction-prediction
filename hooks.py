"""
Lightweight hook system: PrePredictionHook, PostPredictionHook, StopHook.
Each receives JSON context and can: Allow, Deny, Modify.
Mimics Claude Code–style deterministic enforcement.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

HookDecision = Enum("HookDecision", "ALLOW DENY MODIFY")


def _default_pre(_ctx: dict[str, Any]) -> tuple[HookDecision, dict[str, Any] | None]:
    return HookDecision.ALLOW, None


def _default_post(_ctx: dict[str, Any]) -> tuple[HookDecision, dict[str, Any] | None]:
    return HookDecision.ALLOW, None


def _default_stop(_ctx: dict[str, Any]) -> tuple[HookDecision, dict[str, Any] | None]:
    return HookDecision.ALLOW, None


class PrePredictionHook:
    """Runs before PredictionAgent. Can allow, deny, or modify payload."""

    def __init__(self, fn: Callable[[dict[str, Any]], tuple[HookDecision, dict[str, Any] | None]] | None = None) -> None:
        self.fn = fn or _default_pre

    def run(self, context: dict[str, Any]) -> tuple[HookDecision, dict[str, Any] | None]:
        return self.fn(context)


class PostPredictionHook:
    """Runs after PredictionAgent, before PostValidationGuard. Can allow, deny, or modify payload."""

    def __init__(self, fn: Callable[[dict[str, Any]], tuple[HookDecision, dict[str, Any] | None]] | None = None) -> None:
        self.fn = fn or _default_post

    def run(self, context: dict[str, Any]) -> tuple[HookDecision, dict[str, Any] | None]:
        return self.fn(context)


class StopHook:
    """Runs at end of pipeline. Can block (deny) to prevent run from being considered complete."""

    def __init__(self, fn: Callable[[dict[str, Any]], tuple[HookDecision, dict[str, Any] | None]] | None = None) -> None:
        self.fn = fn or _default_stop

    def run(self, context: dict[str, Any]) -> tuple[HookDecision, dict[str, Any] | None]:
        return self.fn(context)


def make_confidence_floor_hook(floor: float = 0.3) -> PostPredictionHook:
    """PostPredictionHook that DENYs when prediction confidence is below floor. Use for research: compare valid rate with vs without hooks."""

    def _fn(ctx: dict[str, Any]) -> tuple[HookDecision, dict[str, Any] | None]:
        pred = ctx.get("prediction") or {}
        conf = pred.get("confidence")
        if conf is None:
            return HookDecision.DENY, None
        try:
            if float(conf) < floor:
                return HookDecision.DENY, None
        except (TypeError, ValueError):
            return HookDecision.DENY, None
        return HookDecision.ALLOW, None

    return PostPredictionHook(_fn)


# Re-export for use in pipeline
HookDecision = HookDecision
