"""A controlled pre-outcome harness for provenance-aware consensus research.

This module is deliberately small and deterministic. It is not a market
simulator and does not claim that explicit source-overlap heuristics solve false
consensus. Its purpose is to validate the evidence contract and establish
paired, outcome-hidden episodes before LLM or financial replay experiments.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import ceil, isfinite
from random import Random
from types import MappingProxyType

from sp500_forecastability.agent_contracts import (
    AgentDecision,
    EvidenceItem,
    ProvenanceGraph,
    parse_agent_decision,
)


class Scenario(str, Enum):
    """Controlled source structures for the initial benchmark harness."""

    INDEPENDENT_CLEAN = "independent_clean"
    SHARED_CLEAN = "shared_clean"
    SHARED_CORRUPTION = "shared_corruption"
    STALE_EVIDENCE = "stale_evidence"
    PARTIAL_CORRUPTION = "partial_corruption"


class EvidenceIntervention(str, Enum):
    """Counterfactual interventions for the deterministic rule-agent oracle."""

    REMOVE = "remove"
    REVERSE = "reverse"


_AGENT_IDS = ("trend_01", "volatility_01", "flow_01")
_DECISION_AT = datetime(2026, 8, 30, 21, 0, tzinfo=timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    text = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _opposite(action: str) -> str:
    return "cash" if action == "long" else "long"


@dataclass(frozen=True)
class PreOutcomeObservation:
    """The only information a detector or router may consume."""

    decision_time: str
    provenance_graph: ProvenanceGraph
    decisions: tuple[AgentDecision, ...]
    agent_evidence_ids: Mapping[str, tuple[str, ...]]
    source_quality: Mapping[str, float]

    def __post_init__(self) -> None:
        normalized = {
            str(agent_id): tuple(str(evidence_id) for evidence_id in evidence_ids)
            for agent_id, evidence_ids in self.agent_evidence_ids.items()
        }
        decision_agents = {decision.agent_id for decision in self.decisions}
        if set(normalized) != decision_agents:
            raise ValueError("agent_evidence_ids must be specified for every decision agent")
        normalized_quality = {str(source_id): float(value) for source_id, value in self.source_quality.items()}
        if any(
            not source_id or not isfinite(value) or not 0.0 <= value <= 1.0
            for source_id, value in normalized_quality.items()
        ):
            raise ValueError("source_quality values must be finite values in [0, 1]")
        root_sources = {
            root_source
            for evidence_id in _referenced_evidence_ids(self.decisions)
            for root_source in self.provenance_graph.root_source_ids(evidence_id)
        }
        missing_quality = root_sources - set(normalized_quality)
        if missing_quality:
            raise ValueError(f"missing source_quality for: {sorted(missing_quality)}")
        object.__setattr__(self, "agent_evidence_ids", MappingProxyType(normalized))
        object.__setattr__(self, "source_quality", MappingProxyType(normalized_quality))


@dataclass(frozen=True)
class SyntheticEpisode:
    """One generated episode; outcome fields are offline evaluation labels."""

    episode_id: str
    scenario: Scenario
    observation: PreOutcomeObservation
    outcome_action: str

    def __post_init__(self) -> None:
        if self.outcome_action not in {"cash", "long"}:
            raise ValueError("outcome_action must be cash or long")

    @property
    def consensus_action(self) -> str:
        votes = Counter(decision.action for decision in self.observation.decisions)
        if votes["long"] == votes["cash"]:
            raise ValueError("the initial benchmark requires a non-tied consensus")
        return "long" if votes["long"] > votes["cash"] else "cash"

    @property
    def agreement(self) -> float:
        return sum(
            decision.action == self.consensus_action for decision in self.observation.decisions
        ) / len(self.observation.decisions)

    @property
    def correlated_consensus(self) -> bool:
        evidence_ids = _referenced_evidence_ids(self.observation.decisions)
        return not self.observation.provenance_graph.are_independent(evidence_ids)

    @property
    def harmful_false_consensus(self) -> bool:
        return self.correlated_consensus and self.consensus_action != self.outcome_action


@dataclass(frozen=True)
class ProvenanceRisk:
    """Transparent pre-outcome risk features for a consensus decision."""

    source_concentration: float
    source_quality_risk: float
    stale_fraction: float
    temporal_violation_fraction: float
    score: float


@dataclass(frozen=True)
class RoutedDecision:
    """Result of a selective consensus router."""

    action: str
    abstained: bool
    consensus_action: str
    risk: ProvenanceRisk


@dataclass(frozen=True)
class RuleAgentInterventionResult:
    """Outcome-free response of a deterministic rule agent to one intervention."""

    agent_id: str
    intervention: EvidenceIntervention
    original_action: str
    counterfactual_action: str | None
    abstained: bool

    @property
    def action_changed(self) -> bool:
        return self.counterfactual_action is not None and self.counterfactual_action != self.original_action


@dataclass(frozen=True)
class FutureLeakageAttempt:
    """A non-routable payload that should be rejected by the agent contract."""

    payload: Mapping[str, object]
    provenance_graph: ProvenanceGraph
    allowed_evidence_ids: tuple[str, ...]

    def rejected_by_contract(self) -> bool:
        try:
            parse_agent_decision(
                self.payload,
                provenance_graph=self.provenance_graph,
                allowed_evidence_ids=self.allowed_evidence_ids,
            )
        except ValueError:
            return True
        return False


def _referenced_evidence_ids(decisions: Sequence[AgentDecision]) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for decision in decisions
        for claim in decision.claims
        for evidence_id in claim.evidence_ids
    )


def _evidence_items(
    scenario: Scenario, outcome_action: str
) -> tuple[
    ProvenanceGraph,
    Mapping[str, tuple[str, ...]],
    Mapping[str, str],
    Mapping[str, float],
]:
    observed_action = outcome_action if scenario in {
        Scenario.INDEPENDENT_CLEAN,
        Scenario.SHARED_CLEAN,
    } else _opposite(outcome_action)
    current_event = _DECISION_AT - timedelta(hours=1)
    current_publication = _DECISION_AT - timedelta(minutes=50)
    current_available = _DECISION_AT - timedelta(minutes=49)
    stale_event = _DECISION_AT - timedelta(days=7, hours=1)
    stale_publication = _DECISION_AT - timedelta(days=7, minutes=50)
    stale_available = _DECISION_AT - timedelta(days=7, minutes=49)

    items: list[EvidenceItem] = []
    agent_evidence: dict[str, tuple[str, ...]] = {}
    agent_actions: dict[str, str] = {}
    source_quality: dict[str, float] = {}
    if scenario is Scenario.INDEPENDENT_CLEAN:
        for index, agent_id in enumerate(_AGENT_IDS):
            root_id = f"source_{index}"
            root_source = f"independent_source_{index}"
            evidence_id = f"{agent_id}_signal"
            items.extend(
                (
                    EvidenceItem(
                        evidence_id=root_id,
                        source_id=root_source,
                        event_time=_isoformat(current_event),
                        publication_time=_isoformat(current_publication),
                        available_at=_isoformat(current_available),
                        summary=f"Independent source {index} indicates {observed_action}.",
                    ),
                    EvidenceItem(
                        evidence_id=evidence_id,
                        source_id=f"{agent_id}_transform",
                        event_time=_isoformat(current_event),
                        publication_time=_isoformat(current_available),
                        available_at=_isoformat(current_available),
                        summary=f"{agent_id} feature indicates {observed_action}.",
                        parent_evidence_ids=(root_id,),
                    ),
                )
            )
            agent_evidence[agent_id] = (evidence_id,)
            agent_actions[agent_id] = observed_action
            source_quality[root_source] = 0.95
    elif scenario is Scenario.PARTIAL_CORRUPTION:
        shared_root_id = "shared_source"
        shared_source = "shared_market_feed"
        corrupted_action = _opposite(outcome_action)
        items.append(
            EvidenceItem(
                evidence_id=shared_root_id,
                source_id=shared_source,
                event_time=_isoformat(current_event),
                publication_time=_isoformat(current_publication),
                available_at=_isoformat(current_available),
                summary=f"Shared source incorrectly indicates {corrupted_action}.",
            )
        )
        for agent_id in _AGENT_IDS[:2]:
            evidence_id = f"{agent_id}_signal"
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_id=f"{agent_id}_transform",
                    event_time=_isoformat(current_event),
                    publication_time=_isoformat(current_available),
                    available_at=_isoformat(current_available),
                    summary=f"{agent_id} feature indicates {corrupted_action}.",
                    parent_evidence_ids=(shared_root_id,),
                )
            )
            agent_evidence[agent_id] = (evidence_id,)
            agent_actions[agent_id] = corrupted_action
        independent_agent = _AGENT_IDS[2]
        independent_root_id = "independent_source"
        independent_root_source = "independent_source_2"
        independent_evidence_id = f"{independent_agent}_signal"
        items.extend(
            (
                EvidenceItem(
                    evidence_id=independent_root_id,
                    source_id=independent_root_source,
                    event_time=_isoformat(current_event),
                    publication_time=_isoformat(current_publication),
                    available_at=_isoformat(current_available),
                    summary=f"Independent source indicates {outcome_action}.",
                ),
                EvidenceItem(
                    evidence_id=independent_evidence_id,
                    source_id=f"{independent_agent}_transform",
                    event_time=_isoformat(current_event),
                    publication_time=_isoformat(current_available),
                    available_at=_isoformat(current_available),
                    summary=f"{independent_agent} feature indicates {outcome_action}.",
                    parent_evidence_ids=(independent_root_id,),
                ),
            )
        )
        agent_evidence[independent_agent] = (independent_evidence_id,)
        agent_actions[independent_agent] = outcome_action
        source_quality[shared_source] = 0.2
        source_quality[independent_root_source] = 0.95
    else:
        root_id = "shared_source"
        is_stale = scenario is Scenario.STALE_EVIDENCE
        event = stale_event if is_stale else current_event
        publication = stale_publication if is_stale else current_publication
        available = stale_available if is_stale else current_available
        items.append(
            EvidenceItem(
                evidence_id=root_id,
                source_id="shared_market_feed" if not is_stale else "market_history_feed",
                event_time=_isoformat(event),
                publication_time=_isoformat(publication),
                available_at=_isoformat(available),
                summary=f"Shared source indicates {observed_action}.",
            )
        )
        source_quality["shared_market_feed" if not is_stale else "market_history_feed"] = (
            0.95 if scenario is Scenario.SHARED_CLEAN else 0.5 if is_stale else 0.2
        )
        for agent_id in _AGENT_IDS:
            evidence_id = f"{agent_id}_signal"
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_id=f"{agent_id}_transform",
                    event_time=_isoformat(event),
                    publication_time=_isoformat(available),
                    available_at=_isoformat(available),
                    summary=f"{agent_id} feature indicates {observed_action}.",
                    parent_evidence_ids=(root_id,),
                )
            )
            agent_evidence[agent_id] = (evidence_id,)
            agent_actions[agent_id] = observed_action
    return (
        ProvenanceGraph.from_items(items),
        MappingProxyType(agent_evidence),
        MappingProxyType(agent_actions),
        MappingProxyType(source_quality),
    )


def generate_episode(scenario: Scenario | str, *, seed: int) -> SyntheticEpisode:
    """Generate one deterministic episode without exposing its outcome to agents."""

    normalized_scenario = Scenario(scenario)
    outcome_action = "long" if Random(seed).random() >= 0.5 else "cash"
    graph, agent_evidence, agent_actions, source_quality = _evidence_items(
        normalized_scenario, outcome_action
    )
    decision_time = _isoformat(_DECISION_AT)
    decisions: list[AgentDecision] = []
    for agent_id in _AGENT_IDS:
        evidence_id = agent_evidence[agent_id][0]
        agent_action = agent_actions[agent_id]
        payload: dict[str, object] = {
            "agent_id": agent_id,
            "decision_time": decision_time,
            "action": agent_action,
            "target_exposure": 1.0 if agent_action == "long" else 0.0,
            "horizon_days": 5,
            "confidence": 0.9,
            "claims": [
                {
                    "claim_id": f"{agent_id}_claim",
                    "text": f"The assigned evidence supports {agent_action}.",
                    "stance": "supports",
                    "evidence_ids": [evidence_id],
                }
            ],
        }
        decisions.append(
            parse_agent_decision(
                payload,
                expected_agent_id=agent_id,
                provenance_graph=graph,
                allowed_evidence_ids=agent_evidence[agent_id],
            )
        )
    observation = PreOutcomeObservation(
        decision_time=decision_time,
        provenance_graph=graph,
        decisions=tuple(decisions),
        agent_evidence_ids=agent_evidence,
        source_quality=source_quality,
    )
    return SyntheticEpisode(
        episode_id=f"{normalized_scenario.value}-{seed}",
        scenario=normalized_scenario,
        observation=observation,
        outcome_action=outcome_action,
    )


def generate_paired_episodes(
    corruption: Scenario | str, *, seed: int
) -> tuple[SyntheticEpisode, SyntheticEpisode]:
    """Return clean and corrupted episodes with the same latent outcome."""

    normalized_corruption = Scenario(corruption)
    if normalized_corruption is Scenario.INDEPENDENT_CLEAN:
        raise ValueError("corruption must differ from independent_clean")
    return (
        generate_episode(Scenario.INDEPENDENT_CLEAN, seed=seed),
        generate_episode(normalized_corruption, seed=seed),
    )


def generate_future_leakage_attempt(*, seed: int) -> FutureLeakageAttempt:
    """Generate an outcome-hidden payload whose evidence becomes available too late."""

    proposed_action = "long" if Random(seed).random() >= 0.5 else "cash"
    decision_time = _isoformat(_DECISION_AT)
    future_publication = _DECISION_AT + timedelta(minutes=1)
    future_available = _DECISION_AT + timedelta(minutes=2)
    graph = ProvenanceGraph.from_items(
        [
            EvidenceItem(
                evidence_id="future_source",
                source_id="future_market_feed",
                event_time=_isoformat(_DECISION_AT),
                publication_time=_isoformat(future_publication),
                available_at=_isoformat(future_available),
                summary=f"Future source indicates {proposed_action}.",
            ),
            EvidenceItem(
                evidence_id="future_signal",
                source_id="future_transform",
                event_time=_isoformat(_DECISION_AT),
                publication_time=_isoformat(future_available),
                available_at=_isoformat(future_available),
                summary=f"Derived future signal indicates {proposed_action}.",
                parent_evidence_ids=("future_source",),
            ),
        ]
    )
    payload: dict[str, object] = {
        "agent_id": "future_01",
        "decision_time": decision_time,
        "action": proposed_action,
        "target_exposure": 1.0 if proposed_action == "long" else 0.0,
        "horizon_days": 5,
        "confidence": 0.99,
        "claims": [
            {
                "claim_id": "future_claim",
                "text": "Use a not-yet-available signal.",
                "stance": "supports",
                "evidence_ids": ["future_signal"],
            }
        ],
    }
    return FutureLeakageAttempt(
        payload=MappingProxyType(payload),
        provenance_graph=graph,
        allowed_evidence_ids=("future_signal",),
    )


def intervene_rule_agent(
    episode: SyntheticEpisode, agent_id: str, intervention: EvidenceIntervention | str
) -> RuleAgentInterventionResult:
    """Apply removal/reversal to a deterministic rule agent without using outcome.

    This is a causal-wiring oracle only. Future LLM experiments must replace this
    deterministic response with paired model calls under the same intervention.
    """

    normalized_intervention = EvidenceIntervention(intervention)
    decision = next(
        (candidate for candidate in episode.observation.decisions if candidate.agent_id == agent_id),
        None,
    )
    if decision is None:
        raise KeyError(f"unknown agent_id: {agent_id}")
    if normalized_intervention is EvidenceIntervention.REMOVE:
        return RuleAgentInterventionResult(
            agent_id=agent_id,
            intervention=normalized_intervention,
            original_action=decision.action,
            counterfactual_action=None,
            abstained=True,
        )
    return RuleAgentInterventionResult(
        agent_id=agent_id,
        intervention=normalized_intervention,
        original_action=decision.action,
        counterfactual_action=_opposite(decision.action),
        abstained=False,
    )


def provenance_risk(
    observation: PreOutcomeObservation, *, stale_after: timedelta = timedelta(days=1)
) -> ProvenanceRisk:
    """Compute an outcome-free source-overlap and temporal risk score."""

    if stale_after <= timedelta(0):
        raise ValueError("stale_after must be positive")
    decision_at = _parse_utc(observation.decision_time)
    graph = observation.provenance_graph
    root_counts: Counter[str] = Counter()
    quality_risk_total = 0.0
    stale_agents = 0
    invalid_agents = 0
    for decision in observation.decisions:
        references = tuple(
            evidence_id for claim in decision.claims for evidence_id in claim.evidence_ids
        )
        roots: set[str] = set()
        is_stale = False
        is_invalid = False
        for evidence_id in references:
            roots.update(graph.root_source_ids(evidence_id))
            item = graph.evidence[evidence_id]
            if decision_at - _parse_utc(item.publication_time) > stale_after:
                is_stale = True
            if _parse_utc(item.available_at) > decision_at:
                is_invalid = True
        root_counts.update(roots)
        quality_risk_total += 1.0 - sum(
            observation.source_quality[root_source] for root_source in roots
        ) / len(roots)
        stale_agents += is_stale
        invalid_agents += is_invalid
    total_agents = len(observation.decisions)
    concentration = max(root_counts.values(), default=0) / total_agents
    source_quality_risk = quality_risk_total / total_agents
    stale_fraction = stale_agents / total_agents
    temporal_fraction = invalid_agents / total_agents
    score = min(
        1.0,
        0.4 * concentration
        + 0.35 * source_quality_risk
        + 0.25 * stale_fraction
        + temporal_fraction,
    )
    return ProvenanceRisk(
        source_concentration=concentration,
        source_quality_risk=source_quality_risk,
        stale_fraction=stale_fraction,
        temporal_violation_fraction=temporal_fraction,
        score=score,
    )


def majority_action(observation: PreOutcomeObservation) -> str:
    """Return the majority action without inspecting the realized outcome."""

    votes = Counter(decision.action for decision in observation.decisions)
    if votes["long"] == votes["cash"]:
        raise ValueError("majority_action requires a non-tied vote")
    return "long" if votes["long"] > votes["cash"] else "cash"


def route_selectively(
    observation: PreOutcomeObservation, *, abstain_threshold: float = 0.45
) -> RoutedDecision:
    """Route majority consensus unless provenance risk requires abstention."""

    if not 0.0 <= abstain_threshold <= 1.0:
        raise ValueError("abstain_threshold must be in [0, 1]")
    consensus = majority_action(observation)
    risk = provenance_risk(observation)
    abstained = risk.score >= abstain_threshold
    return RoutedDecision(
        action="cash" if abstained else consensus,
        abstained=abstained,
        consensus_action=consensus,
        risk=risk,
    )


class ProvenanceVisibility(str, Enum):
    """How much of the true source identity is exposed to the detector."""

    FULL = "full"
    ALIASED = "aliased"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters for the harder synthetic benchmark family.

    ``source_quality_noise`` represents error in an as-of historical quality
    estimate. It is generated before the outcome and is distinct from the
    latent, current-episode source quality retained only for evaluation.
    """

    agent_count: int = 5
    corruption_strength: float = 0.6
    source_quality_noise: float = 0.1
    provenance_visibility: ProvenanceVisibility | str = ProvenanceVisibility.ALIASED
    renamed_transformations: bool = True
    confidence_quality_coupling: float = 1.0
    confidence_noise: float = 0.0

    def __post_init__(self) -> None:
        if self.agent_count < 3 or self.agent_count % 2 == 0:
            raise ValueError("agent_count must be an odd integer of at least three")
        if not 0.0 < self.corruption_strength <= 1.0:
            raise ValueError("corruption_strength must be in (0, 1]")
        if self.source_quality_noise < 0.0 or not isfinite(self.source_quality_noise):
            raise ValueError("source_quality_noise must be finite and non-negative")
        if not 0.0 <= self.confidence_quality_coupling <= 1.0:
            raise ValueError("confidence_quality_coupling must be in [0, 1]")
        if self.confidence_noise < 0.0 or not isfinite(self.confidence_noise):
            raise ValueError("confidence_noise must be finite and non-negative")
        object.__setattr__(
            self, "provenance_visibility", ProvenanceVisibility(self.provenance_visibility)
        )


@dataclass(frozen=True)
class ParameterizedEpisode:
    """A harder episode with separate observed and latent provenance views."""

    episode_id: str
    scenario: Scenario
    observation: PreOutcomeObservation
    outcome_action: str
    corruption_strength: float
    true_root_sources_by_agent: Mapping[str, tuple[str, ...]]
    latent_source_quality: Mapping[str, float]
    agent_historical_performance: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.outcome_action not in {"cash", "long"}:
            raise ValueError("outcome_action must be cash or long")
        if not 0.0 < self.corruption_strength <= 1.0:
            raise ValueError("corruption_strength must be in (0, 1]")
        decision_agents = {decision.agent_id for decision in self.observation.decisions}
        normalized_roots = {
            str(agent_id): tuple(str(source_id) for source_id in source_ids)
            for agent_id, source_ids in self.true_root_sources_by_agent.items()
        }
        if set(normalized_roots) != decision_agents or any(
            not source_ids for source_ids in normalized_roots.values()
        ):
            raise ValueError("true_root_sources_by_agent must cover every decision agent")
        normalized_quality = {
            str(source_id): float(value)
            for source_id, value in self.latent_source_quality.items()
        }
        if any(
            not source_id or not isfinite(value) or not 0.0 <= value <= 1.0
            for source_id, value in normalized_quality.items()
        ):
            raise ValueError("latent_source_quality values must be finite values in [0, 1]")
        true_roots = {source_id for source_ids in normalized_roots.values() for source_id in source_ids}
        if true_roots - set(normalized_quality):
            raise ValueError("latent_source_quality is missing a true source")
        normalized_performance = {
            str(agent_id): float(value)
            for agent_id, value in self.agent_historical_performance.items()
        }
        if set(normalized_performance) != decision_agents or any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in normalized_performance.values()
        ):
            raise ValueError("agent_historical_performance must cover agents with values in [0, 1]")
        object.__setattr__(self, "true_root_sources_by_agent", MappingProxyType(normalized_roots))
        object.__setattr__(self, "latent_source_quality", MappingProxyType(normalized_quality))
        object.__setattr__(self, "agent_historical_performance", MappingProxyType(normalized_performance))

    @property
    def consensus_action(self) -> str:
        return majority_action(self.observation)

    @property
    def agreement(self) -> float:
        return sum(
            decision.action == self.consensus_action for decision in self.observation.decisions
        ) / len(self.observation.decisions)

    @property
    def correlated_consensus(self) -> bool:
        source_counts = Counter(
            source_id
            for source_ids in self.true_root_sources_by_agent.values()
            for source_id in source_ids
        )
        return any(count > 1 for count in source_counts.values())

    @property
    def visible_correlated_consensus(self) -> bool:
        return not self.observation.provenance_graph.are_independent(
            _referenced_evidence_ids(self.observation.decisions)
        )

    @property
    def harmful_false_consensus(self) -> bool:
        return self.correlated_consensus and self.consensus_action != self.outcome_action


@dataclass(frozen=True)
class MechanismHeldOutSplit:
    """Episodes partitioned by corruption mechanism, never by random row."""

    train: tuple[ParameterizedEpisode, ...]
    test: tuple[ParameterizedEpisode, ...]
    train_scenarios: tuple[Scenario, ...]
    held_out_scenarios: tuple[Scenario, ...]
    control_scenarios: tuple[Scenario, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.test:
            raise ValueError("train and test episodes must not be empty")
        if set(self.train_scenarios).intersection(self.held_out_scenarios):
            raise ValueError("train and held-out scenarios must be disjoint")
        if set(self.control_scenarios).intersection(
            set(self.train_scenarios).union(self.held_out_scenarios)
        ):
            raise ValueError("control scenarios must not be held-out mechanisms")
        expected_train = set(self.train_scenarios).union(self.control_scenarios)
        expected_test = set(self.held_out_scenarios).union(self.control_scenarios)
        if {episode.scenario for episode in self.train} - expected_train:
            raise ValueError("train contains an unexpected scenario")
        if {episode.scenario for episode in self.test} - expected_test:
            raise ValueError("test contains an unexpected scenario")


@dataclass(frozen=True)
class SelectiveRoutingSummary:
    """Outcome-time summary of a pre-outcome selective routing policy."""

    total_count: int
    routed_count: int
    abstained_count: int
    routed_error_count: int
    harmful_consensus_count: int
    harmful_consensus_abstained_count: int

    @property
    def coverage(self) -> float:
        return self.routed_count / self.total_count

    @property
    def routed_error_rate(self) -> float:
        return self.routed_error_count / self.routed_count if self.routed_count else 0.0

    @property
    def harmful_consensus_interception_rate(self) -> float:
        if not self.harmful_consensus_count:
            return 0.0
        return self.harmful_consensus_abstained_count / self.harmful_consensus_count


def _parameterized_agent_ids(agent_count: int) -> tuple[str, ...]:
    return tuple(f"agent_{index:02d}" for index in range(agent_count))


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _opaque_identifier(random: Random, prefix: str) -> str:
    return f"{prefix}_{random.randrange(1_000_000_000):09d}"


def _shared_agent_count(config: BenchmarkConfig) -> int:
    return max(1, min(config.agent_count, ceil(config.agent_count * config.corruption_strength)))


def _latent_quality(scenario: Scenario, strength: float, *, shared: bool) -> float:
    if not shared:
        return 0.95
    if scenario is Scenario.SHARED_CLEAN:
        return 0.95
    if scenario is Scenario.STALE_EVIDENCE:
        return max(0.05, 0.7 - 0.35 * strength)
    return max(0.05, 0.95 - 0.9 * strength)


def generate_parameterized_episode(
    scenario: Scenario | str, *, seed: int, config: BenchmarkConfig | None = None
) -> ParameterizedEpisode:
    """Generate a configurable episode with outcome-hidden noisy provenance.

    The returned ``observation`` is the detector input. True root identities and
    latent source quality are retained only on the episode for offline labels.
    """

    normalized_scenario = Scenario(scenario)
    benchmark_config = BenchmarkConfig() if config is None else config
    random = Random(seed)
    outcome_action = "long" if random.random() >= 0.5 else "cash"
    agent_ids = _parameterized_agent_ids(benchmark_config.agent_count)
    if normalized_scenario is Scenario.INDEPENDENT_CLEAN:
        shared_agents: set[str] = set()
    elif normalized_scenario is Scenario.SHARED_CLEAN:
        shared_agents = set(agent_ids)
    else:
        shared_agents = set(agent_ids[:_shared_agent_count(benchmark_config)])

    current_event = _DECISION_AT - timedelta(hours=1)
    current_publication = _DECISION_AT - timedelta(minutes=50)
    current_available = _DECISION_AT - timedelta(minutes=49)
    stale_event = _DECISION_AT - timedelta(days=7, hours=1)
    stale_publication = _DECISION_AT - timedelta(days=7, minutes=50)
    stale_available = _DECISION_AT - timedelta(days=7, minutes=49)
    alias_by_true_source: dict[str, str] = {}
    observed_quality: dict[str, float] = {}
    latent_quality: dict[str, float] = {}
    root_items: dict[str, EvidenceItem] = {}
    derived_items: list[EvidenceItem] = []
    agent_evidence: dict[str, tuple[str, ...]] = {}
    agent_actions: dict[str, str] = {}
    true_roots_by_agent: dict[str, tuple[str, ...]] = {}

    def observed_root_id(true_source: str, agent_id: str) -> str:
        if benchmark_config.provenance_visibility is ProvenanceVisibility.FULL:
            return true_source
        if benchmark_config.provenance_visibility is ProvenanceVisibility.ALIASED:
            return alias_by_true_source.setdefault(true_source, _opaque_identifier(random, "src"))
        return _opaque_identifier(random, f"src_{agent_id}")

    for index, agent_id in enumerate(agent_ids):
        is_shared = agent_id in shared_agents
        true_source = (
            f"true_shared_{normalized_scenario.value}"
            if is_shared
            else f"true_independent_{index}"
        )
        is_corrupted = is_shared and normalized_scenario not in {
            Scenario.INDEPENDENT_CLEAN,
            Scenario.SHARED_CLEAN,
        }
        agent_action = _opposite(outcome_action) if is_corrupted else outcome_action
        is_stale = is_corrupted and normalized_scenario is Scenario.STALE_EVIDENCE
        event = stale_event if is_stale else current_event
        publication = stale_publication if is_stale else current_publication
        available = stale_available if is_stale else current_available
        root_id = observed_root_id(true_source, agent_id)
        source_quality = _latent_quality(
            normalized_scenario, benchmark_config.corruption_strength, shared=is_shared
        )
        latent_quality[true_source] = source_quality
        if root_id not in root_items:
            root_items[root_id] = EvidenceItem(
                evidence_id=root_id,
                source_id=root_id,
                event_time=_isoformat(event),
                publication_time=_isoformat(publication),
                available_at=_isoformat(available),
                summary="A structured source observation is available.",
            )
            observed_quality[root_id] = _clip_probability(
                source_quality + random.gauss(0.0, benchmark_config.source_quality_noise)
            )
        evidence_id = (
            _opaque_identifier(random, "feature")
            if benchmark_config.renamed_transformations
            else f"{agent_id}_feature"
        )
        transform_id = (
            _opaque_identifier(random, "transform")
            if benchmark_config.renamed_transformations
            else f"{agent_id}_transform"
        )
        derived_items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                source_id=transform_id,
                event_time=_isoformat(event),
                publication_time=_isoformat(available),
                available_at=_isoformat(available),
                summary="A derived feature supports the proposed action.",
                parent_evidence_ids=(root_id,),
            )
        )
        agent_evidence[agent_id] = (evidence_id,)
        agent_actions[agent_id] = agent_action
        true_roots_by_agent[agent_id] = (true_source,)

    graph = ProvenanceGraph.from_items([*root_items.values(), *derived_items])
    decision_time = _isoformat(_DECISION_AT)
    decisions: list[AgentDecision] = []
    for agent_id in agent_ids:
        action = agent_actions[agent_id]
        evidence_id = agent_evidence[agent_id][0]
        root_id = graph.evidence[evidence_id].parent_evidence_ids[0]
        quality_confidence = 0.55 + 0.4 * observed_quality[root_id]
        if (
            benchmark_config.confidence_quality_coupling == 1.0
            and benchmark_config.confidence_noise == 0.0
        ):
            confidence = quality_confidence
        else:
            uncalibrated_confidence = _clip_probability(0.78 + random.gauss(0.0, 0.08))
            confidence = _clip_probability(
                benchmark_config.confidence_quality_coupling * quality_confidence
                + (1.0 - benchmark_config.confidence_quality_coupling) * uncalibrated_confidence
                + random.gauss(0.0, benchmark_config.confidence_noise)
            )
        payload: dict[str, object] = {
            "agent_id": agent_id,
            "decision_time": decision_time,
            "action": action,
            "target_exposure": 1.0 if action == "long" else 0.0,
            "horizon_days": 5,
            "confidence": confidence,
            "claims": [
                {
                    "claim_id": f"{agent_id}_claim",
                    "text": "The assigned evidence supports the proposed action.",
                    "stance": "supports",
                    "evidence_ids": [evidence_id],
                }
            ],
        }
        decisions.append(
            parse_agent_decision(
                payload,
                expected_agent_id=agent_id,
                provenance_graph=graph,
                allowed_evidence_ids=agent_evidence[agent_id],
            )
        )
    observation = PreOutcomeObservation(
        decision_time=decision_time,
        provenance_graph=graph,
        decisions=tuple(decisions),
        agent_evidence_ids=agent_evidence,
        source_quality=observed_quality,
    )
    agent_historical_performance = {
        agent_id: _clip_probability(0.80 + random.gauss(0.0, 0.06))
        for agent_id in agent_ids
    }
    return ParameterizedEpisode(
        episode_id=(
            f"parameterized-{normalized_scenario.value}-{benchmark_config.provenance_visibility}-{seed}"
        ),
        scenario=normalized_scenario,
        observation=observation,
        outcome_action=outcome_action,
        corruption_strength=benchmark_config.corruption_strength,
        true_root_sources_by_agent=true_roots_by_agent,
        latent_source_quality=latent_quality,
        agent_historical_performance=agent_historical_performance,
    )


def generate_mechanism_heldout_split(
    *,
    seeds: Sequence[int],
    train_scenarios: Sequence[Scenario | str],
    held_out_scenarios: Sequence[Scenario | str],
    control_scenarios: Sequence[Scenario | str] = (),
    config: BenchmarkConfig | None = None,
) -> MechanismHeldOutSplit:
    """Generate a split that holds out whole corruption mechanisms at test time."""

    if not seeds:
        raise ValueError("seeds must not be empty")
    train = tuple(Scenario(scenario) for scenario in train_scenarios)
    held_out = tuple(Scenario(scenario) for scenario in held_out_scenarios)
    controls = tuple(Scenario(scenario) for scenario in control_scenarios)
    if not train or not held_out:
        raise ValueError("train_scenarios and held_out_scenarios must not be empty")
    if set(train).intersection(held_out):
        raise ValueError("train_scenarios and held_out_scenarios must be disjoint")
    if set(controls).intersection(set(train).union(held_out)):
        raise ValueError("control_scenarios must be disjoint from corruption mechanisms")
    benchmark_config = BenchmarkConfig() if config is None else config
    return MechanismHeldOutSplit(
        train=tuple(
            generate_parameterized_episode(scenario, seed=seed, config=benchmark_config)
            for scenario in (*controls, *train)
            for seed in seeds
        ),
        test=tuple(
            generate_parameterized_episode(scenario, seed=seed, config=benchmark_config)
            for scenario in (*controls, *held_out)
            for seed in seeds
        ),
        train_scenarios=train,
        held_out_scenarios=held_out,
        control_scenarios=controls,
    )


def evaluate_selective_router(
    episodes: Sequence[ParameterizedEpisode], *, abstain_threshold: float = 0.45
) -> SelectiveRoutingSummary:
    """Evaluate routing only after outcomes are revealed, never during routing."""

    if not episodes:
        raise ValueError("episodes must not be empty")
    routed_count = 0
    abstained_count = 0
    routed_error_count = 0
    harmful_count = 0
    harmful_abstained_count = 0
    for episode in episodes:
        routed = route_selectively(
            episode.observation,
            abstain_threshold=abstain_threshold,
        )
        if episode.harmful_false_consensus:
            harmful_count += 1
        if routed.abstained:
            abstained_count += 1
            if episode.harmful_false_consensus:
                harmful_abstained_count += 1
            continue
        routed_count += 1
        routed_error_count += routed.action != episode.outcome_action
    return SelectiveRoutingSummary(
        total_count=len(episodes),
        routed_count=routed_count,
        abstained_count=abstained_count,
        routed_error_count=routed_error_count,
        harmful_consensus_count=harmful_count,
        harmful_consensus_abstained_count=harmful_abstained_count,
    )
