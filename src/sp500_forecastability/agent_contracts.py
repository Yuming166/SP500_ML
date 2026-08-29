"""Strict, model-agnostic contracts for stage-two LLM agents.

The contract deliberately stores short, auditable claims instead of hidden
chain-of-thought. Numerical execution remains in the simulator, not in the
language model response.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite


_ACTIONS = {"cash", "long"}
_DECISION_FIELDS = {
    "agent_id",
    "action",
    "target_exposure",
    "horizon_days",
    "confidence",
    "rationale_claims",
    "evidence_tags",
}


@dataclass(frozen=True)
class MarketState:
    """A compact, as-of market state exposed to an LLM agent."""

    timestamp: str
    regime: str
    p_up: float
    interval_width: float
    signals: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.timestamp or not self.regime:
            raise ValueError("timestamp and regime must not be empty")
        if not 0.0 <= self.p_up <= 1.0 or not isfinite(self.p_up):
            raise ValueError("p_up must be finite and in [0, 1]")
        if not isfinite(float(self.interval_width)) or self.interval_width < 0.0:
            raise ValueError("interval_width must be finite and non-negative")
        normalized = {str(key): float(value) for key, value in self.signals.items()}
        if any(not isfinite(value) for value in normalized.values()):
            raise ValueError("signals must contain only finite values")
        object.__setattr__(self, "signals", normalized)

    def to_prompt_payload(self) -> dict[str, object]:
        """Return the only state payload an agent should receive."""

        return {
            "timestamp": self.timestamp,
            "regime": self.regime,
            "p_up": self.p_up,
            "interval_width": self.interval_width,
            "signals": dict(self.signals),
        }


@dataclass(frozen=True)
class AgentDecision:
    """A validated, auditable decision returned by one behavior agent."""

    agent_id: str
    action: str
    target_exposure: float
    horizon_days: int
    confidence: float
    rationale_claims: tuple[str, ...]
    evidence_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if self.action not in _ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ACTIONS)}")
        if not 0.0 <= self.target_exposure <= 1.0 or not isfinite(self.target_exposure):
            raise ValueError("target_exposure must be finite and in [0, 1]")
        if self.action == "cash" and self.target_exposure != 0.0:
            raise ValueError("cash decisions must have zero target_exposure")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be positive")
        if not 0.0 <= self.confidence <= 1.0 or not isfinite(self.confidence):
            raise ValueError("confidence must be finite and in [0, 1]")
        if not self.rationale_claims:
            raise ValueError("at least one rationale claim is required")
        if any(not claim.strip() for claim in self.rationale_claims):
            raise ValueError("rationale claims must not be empty")

    def to_payload(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "action": self.action,
            "target_exposure": self.target_exposure,
            "horizon_days": self.horizon_days,
            "confidence": self.confidence,
            "rationale_claims": list(self.rationale_claims),
            "evidence_tags": list(self.evidence_tags),
        }


def _string_tuple(value: object, field: str, *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if required and not result:
        raise ValueError(f"{field} must not be empty")
    if any(not item for item in result):
        raise ValueError(f"{field} must not contain empty strings")
    return result


def parse_agent_decision(
    payload: Mapping[str, object], expected_agent_id: str | None = None
) -> AgentDecision:
    """Validate a structured LLM response and reject schema drift."""

    unknown = set(payload) - _DECISION_FIELDS
    missing = _DECISION_FIELDS - set(payload)
    if unknown:
        raise ValueError(f"unexpected decision fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing decision fields: {sorted(missing)}")

    agent_id = str(payload["agent_id"])
    if expected_agent_id is not None and agent_id != expected_agent_id:
        raise ValueError("agent_id does not match the expected agent")
    try:
        horizon_days = int(payload["horizon_days"])
        if float(payload["horizon_days"]) != horizon_days:
            raise ValueError("horizon_days must be an integer")
    except (TypeError, ValueError) as error:
        raise ValueError("horizon_days must be an integer") from error

    return AgentDecision(
        agent_id=agent_id,
        action=str(payload["action"]),
        target_exposure=float(payload["target_exposure"]),
        horizon_days=horizon_days,
        confidence=float(payload["confidence"]),
        rationale_claims=_string_tuple(payload["rationale_claims"], "rationale_claims", required=True),
        evidence_tags=_string_tuple(payload["evidence_tags"], "evidence_tags", required=False),
    )

