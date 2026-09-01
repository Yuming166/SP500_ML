"""Regressions for V12.1's one-item bounded auxiliary amendment."""

from __future__ import annotations

from sp500_forecastability import pilot_llm_v12_1 as v12_1


def test_v12_1_inherits_selection_and_confirmatory_contract() -> None:
    manifest, comps = v12_1.build_inherited_selection(
        v12_1.v12.DEFAULT_DATASET, status="test",
    )
    assert len(comps) == 358
    assert manifest["selection_parent"]["selection_changed"] is False
    assert manifest["primary_endpoint"] == v12_1.v12.PRIMARY_ENDPOINT
    assert manifest["risk_contract"]["weights"] == {
        "D_inert": 0.1, "flip_inertia": 0.3, "frac_shared": 0.6,
    }
    assert manifest["parallel_contract"]["workers"] == 4
    v12_1.validate_manifest_v12_1(
        manifest, v12_1.v12.DEFAULT_DATASET, require_substitutes=False,
    )


def test_v12_1_repairs_exactly_one_frozen_evidence_id() -> None:
    manifest, _ = v12_1.build_inherited_selection(
        v12_1.v12.DEFAULT_DATASET, status="test",
    )
    amendment = manifest["auxiliary_amendment"]
    assert amendment["second_repair_qids"] == [
        "boolq-1592052e5f54e039-e03"
    ]
    assert amendment["maximum_second_repair_calls"] == 1
    assert amendment["overlong_repairs_truncated"] is False
    assert amendment["selection_or_confirmatory_contract_changed"] is False


def test_v12_1_second_repair_prompt_freezes_exact_length() -> None:
    _manifest, comps = v12_1.build_inherited_selection(
        v12_1.v12.DEFAULT_DATASET, status="test",
    )
    item = next(
        item for comp in comps for item in comp.items
        if item.qid == v12_1.FAILED_QID
    )
    prompt = v12_1._second_repair_prompt(item)
    assert len(item.passage.split()) == 13
    assert "exactly 13 whitespace-separated tokens" in prompt
    assert "7 through 19 tokens" in prompt
