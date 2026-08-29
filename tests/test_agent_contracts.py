from sp500_forecastability.agent_contracts import MarketState, parse_agent_decision


def _valid_payload() -> dict[str, object]:
    return {
        "agent_id": "trend_01",
        "action": "long",
        "target_exposure": 0.75,
        "horizon_days": 5,
        "confidence": 0.8,
        "rationale_claims": ["The low-frequency trend is positive."],
        "evidence_tags": ["low_frequency_trend"],
    }


def test_agent_decision_is_strictly_parsed() -> None:
    decision = parse_agent_decision(_valid_payload(), expected_agent_id="trend_01")

    assert decision.action == "long"
    assert decision.target_exposure == 0.75
    assert decision.rationale_claims == ("The low-frequency trend is positive.",)


def test_agent_decision_rejects_schema_drift() -> None:
    payload = _valid_payload()
    payload["free_form_reasoning"] = "not part of the auditable contract"

    try:
        parse_agent_decision(payload)
    except ValueError as error:
        assert "unexpected decision fields" in str(error)
    else:
        raise AssertionError("schema drift should be rejected")


def test_market_state_payload_contains_only_as_of_features() -> None:
    state = MarketState(
        timestamp="2026-08-30",
        regime="transition",
        p_up=0.55,
        interval_width=0.04,
        signals={"vix": 1.2},
    )

    assert state.to_prompt_payload()["signals"] == {"vix": 1.2}
