import pytest

from sp500_forecastability.agent_contracts import (
    EvidenceItem,
    MarketState,
    ProvenanceGraph,
    parse_agent_decision,
)

DECISION_TIME = "2026-08-30T21:00:00Z"


def _graph() -> ProvenanceGraph:
    return ProvenanceGraph.from_items(
        [
            EvidenceItem(
                evidence_id="close",
                source_id="sp500_close",
                event_time="2026-08-30T20:00:00Z",
                publication_time="2026-08-30T20:05:00Z",
                available_at="2026-08-30T20:06:00Z",
                summary="S&P 500 close is available.",
            ),
            EvidenceItem(
                evidence_id="trend",
                source_id="ema_20_transform",
                event_time="2026-08-30T20:00:00Z",
                publication_time="2026-08-30T20:06:00Z",
                available_at="2026-08-30T20:06:00Z",
                summary="The 20-day trend is positive.",
                parent_evidence_ids=("close",),
            ),
            EvidenceItem(
                evidence_id="momentum",
                source_id="return_5d_transform",
                event_time="2026-08-30T20:00:00Z",
                publication_time="2026-08-30T20:06:00Z",
                available_at="2026-08-30T20:06:00Z",
                summary="The 5-day momentum is positive.",
                parent_evidence_ids=("close",),
            ),
            EvidenceItem(
                evidence_id="vix",
                source_id="cboe_vix_close",
                event_time="2026-08-30T20:00:00Z",
                publication_time="2026-08-30T20:10:00Z",
                available_at="2026-08-30T20:11:00Z",
                summary="VIX is low.",
            ),
        ]
    )


def _valid_payload() -> dict[str, object]:
    return {
        "agent_id": "trend_01",
        "decision_time": DECISION_TIME,
        "action": "long",
        "target_exposure": 0.75,
        "horizon_days": 5,
        "confidence": 0.8,
        "claims": [
            {
                "claim_id": "trend_support",
                "text": "The low-frequency trend is positive.",
                "stance": "supports",
                "evidence_ids": ["trend"],
            }
        ],
    }


def test_agent_decision_is_strictly_parsed_against_catalog() -> None:
    decision = parse_agent_decision(
        _valid_payload(),
        expected_agent_id="trend_01",
        provenance_graph=_graph(),
        allowed_evidence_ids=("trend",),
    )

    assert decision.action == "long"
    assert decision.target_exposure == 0.75
    assert decision.claims[0].evidence_ids == ("trend",)


def test_agent_decision_rejects_schema_drift() -> None:
    payload = _valid_payload()
    payload["free_form_reasoning"] = "not part of the auditable contract"

    with pytest.raises(ValueError, match="unexpected decision fields"):
        parse_agent_decision(payload, provenance_graph=_graph())


def test_future_evidence_is_rejected() -> None:
    graph = ProvenanceGraph.from_items(
        [
            EvidenceItem(
                evidence_id="future_vix",
                source_id="cboe_vix_close",
                event_time="2026-08-30T20:00:00Z",
                publication_time="2026-08-30T22:00:00Z",
                available_at="2026-08-30T22:01:00Z",
                summary="This VIX close was published after the decision.",
            )
        ]
    )
    payload = _valid_payload()
    payload["claims"] = [
        {
            "claim_id": "future_claim",
            "text": "Use a future VIX observation.",
            "stance": "supports",
            "evidence_ids": ["future_vix"],
        }
    ]

    with pytest.raises(ValueError, match="not available at decision_time"):
        parse_agent_decision(payload, provenance_graph=graph)


def test_derived_features_from_one_raw_source_are_not_independent() -> None:
    graph = _graph()

    assert graph.root_source_ids("trend") == frozenset({"sp500_close"})
    assert not graph.are_independent(("trend", "momentum"))
    assert graph.are_independent(("trend", "vix"))


def test_agent_cannot_reference_evidence_outside_catalog_or_packet() -> None:
    payload = _valid_payload()
    payload["claims"] = [
        {
            "claim_id": "unknown_evidence",
            "text": "Use an invented evidence ID.",
            "stance": "supports",
            "evidence_ids": ["invented"],
        }
    ]
    with pytest.raises(ValueError, match="outside the catalog"):
        parse_agent_decision(payload, provenance_graph=_graph())

    payload = _valid_payload()
    payload["claims"] = [
        {
            "claim_id": "outside_packet",
            "text": "Use evidence that another agent received.",
            "stance": "supports",
            "evidence_ids": ["vix"],
        }
    ]
    with pytest.raises(ValueError, match="outside the agent packet"):
        parse_agent_decision(
            payload,
            provenance_graph=_graph(),
            allowed_evidence_ids=("trend",),
        )


def test_market_state_payload_contains_only_as_of_features() -> None:
    state = MarketState(
        timestamp="2026-08-30T21:00:00Z",
        regime="transition",
        p_up=0.55,
        interval_width=0.04,
        signals={"vix": 1.2},
    )

    assert state.to_prompt_payload()["signals"] == {"vix": 1.2}
