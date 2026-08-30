"""Strict, provenance-aware contracts for stage-two LLM agents.

Agents only return short, auditable claims and references to an environment-owned
evidence catalog. They never supply source metadata themselves, which lets the
runtime reject future-leaked evidence, unknown references, and schema drift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from types import MappingProxyType

_ACTIONS = {"cash", "long"}
_STANCES = {"supports", "attacks"}
_CLAIM_FIELDS = {"claim_id", "text", "stance", "evidence_ids"}
_DECISION_FIELDS = {
    "agent_id",
    "decision_time",
    "action",
    "target_exposure",
    "horizon_days",
    "confidence",
    "claims",
}


def _parse_timestamp(value: object, field: str) -> datetime:
    """Parse an offset-aware ISO-8601 timestamp as UTC."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _string_tuple(value: object, field: str, *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if required and not result:
        raise ValueError(f"{field} must not be empty")
    if any(not item for item in result):
        raise ValueError(f"{field} must not contain empty strings")
    return result


@dataclass(frozen=True)
class EvidenceItem:
    """Environment-owned evidence with temporal and derivation provenance.

    ``source_id`` identifies the raw source or a deterministic transformation.
    For derived evidence, ``parent_evidence_ids`` must point to its inputs.
    Independence is computed from leaf source IDs, so transformed versions of
    one raw signal cannot become independent evidence.
    """

    evidence_id: str
    source_id: str
    event_time: str
    publication_time: str
    available_at: str
    summary: str
    parent_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("evidence_id", "source_id", "summary"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must not be empty")
        parents = _string_tuple(self.parent_evidence_ids, "parent_evidence_ids", required=False)
        if self.evidence_id in parents:
            raise ValueError("an evidence item cannot be its own parent")
        publication = _parse_timestamp(self.publication_time, "publication_time")
        available = _parse_timestamp(self.available_at, "available_at")
        _parse_timestamp(self.event_time, "event_time")
        if available < publication:
            raise ValueError("available_at must not be earlier than publication_time")
        object.__setattr__(self, "parent_evidence_ids", parents)


@dataclass(frozen=True)
class Claim:
    """A concise claim whose evidence references are externally verifiable."""

    claim_id: str
    text: str
    stance: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ValueError("claim_id must not be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("claim text must not be empty")
        if self.stance not in _STANCES:
            raise ValueError(f"stance must be one of {sorted(_STANCES)}")
        evidence_ids = _string_tuple(self.evidence_ids, "evidence_ids", required=True)
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_payload(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "stance": self.stance,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ProvenanceGraph:
    """Validated evidence catalog and its derivation graph."""

    evidence: Mapping[str, EvidenceItem]

    def __post_init__(self) -> None:
        normalized: dict[str, EvidenceItem] = {}
        for key, item in self.evidence.items():
            if not isinstance(item, EvidenceItem):
                raise TypeError("evidence catalog values must be EvidenceItem instances")
            evidence_id = str(key).strip()
            if not evidence_id or evidence_id != item.evidence_id:
                raise ValueError("evidence catalog keys must match non-empty evidence_id values")
            if evidence_id in normalized:
                raise ValueError(f"duplicate evidence_id: {evidence_id}")
            normalized[evidence_id] = item
        self._validate_parents_and_cycles(normalized)
        object.__setattr__(self, "evidence", MappingProxyType(normalized))

    @classmethod
    def from_items(cls, items: Sequence[EvidenceItem]) -> ProvenanceGraph:
        catalog: dict[str, EvidenceItem] = {}
        for item in items:
            if item.evidence_id in catalog:
                raise ValueError(f"duplicate evidence_id: {item.evidence_id}")
            catalog[item.evidence_id] = item
        return cls(catalog)

    @staticmethod
    def _validate_parents_and_cycles(catalog: Mapping[str, EvidenceItem]) -> None:
        for evidence_id, item in catalog.items():
            unknown = set(item.parent_evidence_ids) - set(catalog)
            if unknown:
                raise ValueError(
                    f"evidence {evidence_id} references unknown parents: {sorted(unknown)}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(evidence_id: str) -> None:
            if evidence_id in visiting:
                raise ValueError(f"cycle detected in evidence provenance at {evidence_id}")
            if evidence_id in visited:
                return
            visiting.add(evidence_id)
            for parent_id in catalog[evidence_id].parent_evidence_ids:
                visit(parent_id)
            visiting.remove(evidence_id)
            visited.add(evidence_id)

        for evidence_id in catalog:
            visit(evidence_id)

    def _require_known(self, evidence_ids: Sequence[str]) -> tuple[str, ...]:
        normalized = _string_tuple(evidence_ids, "evidence_ids", required=True)
        unknown = set(normalized) - set(self.evidence)
        if unknown:
            raise ValueError(f"claims reference evidence outside the catalog: {sorted(unknown)}")
        return normalized

    def _ancestor_ids(self, evidence_id: str) -> frozenset[str]:
        ancestors = {evidence_id}
        for parent_id in self.evidence[evidence_id].parent_evidence_ids:
            ancestors.update(self._ancestor_ids(parent_id))
        return frozenset(ancestors)

    def root_source_ids(self, evidence_id: str) -> frozenset[str]:
        """Return leaf source IDs used to construct one evidence item."""

        self._require_known((evidence_id,))
        item = self.evidence[evidence_id]
        if not item.parent_evidence_ids:
            return frozenset({item.source_id})
        roots: set[str] = set()
        for parent_id in item.parent_evidence_ids:
            roots.update(self.root_source_ids(parent_id))
        return frozenset(roots)

    def are_independent(self, evidence_ids: Sequence[str]) -> bool:
        """Whether evidence items have pairwise disjoint leaf source IDs."""

        normalized = self._require_known(evidence_ids)
        if len(set(normalized)) != len(normalized):
            return False
        seen_sources: set[str] = set()
        for evidence_id in normalized:
            root_sources = self.root_source_ids(evidence_id)
            if seen_sources.intersection(root_sources):
                return False
            seen_sources.update(root_sources)
        return True

    def validate_claims(
        self,
        claims: Sequence[Claim],
        decision_time: str,
        *,
        allowed_evidence_ids: Sequence[str] | None = None,
    ) -> None:
        """Reject unknown, unavailable, or packet-external evidence references."""

        decision_at = _parse_timestamp(decision_time, "decision_time")
        allowed: set[str] | None = None
        if allowed_evidence_ids is not None:
            allowed = set(self._require_known(allowed_evidence_ids))

        for claim in claims:
            references = self._require_known(claim.evidence_ids)
            if allowed is not None:
                outside_packet = set(references) - allowed
                if outside_packet:
                    raise ValueError(
                        "claims reference evidence outside the agent packet: "
                        f"{sorted(outside_packet)}"
                    )
            for evidence_id in references:
                for ancestor_id in self._ancestor_ids(evidence_id):
                    item = self.evidence[ancestor_id]
                    if _parse_timestamp(item.available_at, "available_at") > decision_at:
                        raise ValueError(
                            f"evidence {ancestor_id} was not available at decision_time"
                        )


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
        _parse_timestamp(self.timestamp, "timestamp")
        if not 0.0 <= self.p_up <= 1.0 or not isfinite(self.p_up):
            raise ValueError("p_up must be finite and in [0, 1]")
        if not isfinite(float(self.interval_width)) or self.interval_width < 0.0:
            raise ValueError("interval_width must be finite and non-negative")
        normalized = {str(key): float(value) for key, value in self.signals.items()}
        if any(not isfinite(value) for value in normalized.values()):
            raise ValueError("signals must contain only finite values")
        object.__setattr__(self, "signals", normalized)

    def to_prompt_payload(self) -> dict[str, object]:
        """Return the state payload; evidence remains a separate packet."""

        return {
            "timestamp": self.timestamp,
            "regime": self.regime,
            "p_up": self.p_up,
            "interval_width": self.interval_width,
            "signals": dict(self.signals),
        }


@dataclass(frozen=True)
class AgentDecision:
    """A validated decision returned by one behavior agent."""

    agent_id: str
    decision_time: str
    action: str
    target_exposure: float
    horizon_days: int
    confidence: float
    claims: tuple[Claim, ...]

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        _parse_timestamp(self.decision_time, "decision_time")
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
        if not self.claims:
            raise ValueError("at least one claim is required")
        if any(not isinstance(claim, Claim) for claim in self.claims):
            raise TypeError("claims must contain Claim instances")
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("claim_id values must be unique within a decision")

    def to_payload(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "decision_time": self.decision_time,
            "action": self.action,
            "target_exposure": self.target_exposure,
            "horizon_days": self.horizon_days,
            "confidence": self.confidence,
            "claims": [claim.to_payload() for claim in self.claims],
        }


def _parse_claim(payload: object) -> Claim:
    if not isinstance(payload, Mapping):
        raise TypeError("each claim must be an object")
    unknown = set(payload) - _CLAIM_FIELDS
    missing = _CLAIM_FIELDS - set(payload)
    if unknown:
        raise ValueError(f"unexpected claim fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing claim fields: {sorted(missing)}")
    return Claim(
        claim_id=str(payload["claim_id"]),
        text=str(payload["text"]),
        stance=str(payload["stance"]),
        evidence_ids=_string_tuple(payload["evidence_ids"], "evidence_ids", required=True),
    )


def parse_agent_decision(
    payload: Mapping[str, object],
    expected_agent_id: str | None = None,
    *,
    provenance_graph: ProvenanceGraph,
    allowed_evidence_ids: Sequence[str] | None = None,
) -> AgentDecision:
    """Validate an LLM response against an environment-owned evidence graph."""

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
    claims_value = payload["claims"]
    if not isinstance(claims_value, Sequence) or isinstance(claims_value, (str, bytes)):
        raise TypeError("claims must be a sequence of objects")
    claims = tuple(_parse_claim(claim) for claim in claims_value)
    decision_time = str(payload["decision_time"])
    provenance_graph.validate_claims(
        claims,
        decision_time,
        allowed_evidence_ids=allowed_evidence_ids,
    )
    return AgentDecision(
        agent_id=agent_id,
        decision_time=decision_time,
        action=str(payload["action"]),
        target_exposure=float(payload["target_exposure"]),
        horizon_days=horizon_days,
        confidence=float(payload["confidence"]),
        claims=claims,
    )
