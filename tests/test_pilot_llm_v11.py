"""Offline and outcome-leakage regression tests for frozen V11."""

from __future__ import annotations

from functools import lru_cache

import pytest

from sp500_forecastability import pilot_llm_v11 as v11

_BASE_GLOBALS = (
    "PROTOCOL_VERSION", "SALT", "CQID_PROTOCOL_VERSION", "CQID_PREFIX",
    "FORMAL_EXAMPLES", "FORMAL_PER_LABEL", "DEFAULT_DATASET", "DEFAULT_ROOT",
    "BOOTSTRAP_SEED", "BOOTSTRAP_REPLICATES",
)
_REWRITE_GLOBALS = (
    "PROTOCOL_VERSION", "DEFAULT_ROOT", "INITIAL_REWRITE_SEED",
)
_BASE_STATE = {name: getattr(v11.base, name) for name in _BASE_GLOBALS}
_REWRITE_STATE = {
    name: getattr(v11.rewrite_impl, name) for name in _REWRITE_GLOBALS
}


@pytest.fixture(autouse=True)
def _restore_shared_module_state():
    yield
    for name, value in _BASE_STATE.items():
        setattr(v11.base, name, value)
    for name, value in _REWRITE_STATE.items():
        setattr(v11.rewrite_impl, name, value)


@lru_cache(maxsize=1)
def _composites():
    v11.configure_base()
    return tuple(v11.base.build_composite_questions(
        v11.base.load_boolq(v11.DEFAULT_DATASET)
    ))


def test_v11_validation_selection_is_balanced_and_fresh() -> None:
    comps = _composites()
    assert len(comps) == 200
    assert sum(comp.label == "yes" for comp in comps) == 100
    assert sum(comp.label == "no" for comp in comps) == 100
    assert all(comp.cqid.startswith("v11-") for comp in comps)


def test_v11_risk_score_is_invariant_to_outcome_fields() -> None:
    row = {
        "D_inert": 0.6, "flip_inertia": 0.7, "frac_shared": 0.4,
        "correct": 1, "harmful_fc": 0, "label": "yes", "gold_binary": 1,
    }
    before = v11.risk_score_from_preoutcome(row)
    row.update({
        "correct": 0, "harmful_fc": 1, "label": "no", "gold_binary": 0,
    })
    after = v11.risk_score_from_preoutcome(row)
    assert before == after
    assert before == 0.1 * 0.6 + 0.3 * 0.7 + 0.6 * 0.4


def test_v11_manifest_has_one_primary_and_true_40_call_smoke() -> None:
    comps = _composites()
    manifest = v11.build_manifest_v11(
        v11.DEFAULT_DATASET, comps, {}, status="test",
    )
    restored = v11.validate_manifest_v11(
        manifest, v11.DEFAULT_DATASET, require_substitutes=False,
    )
    assert len(restored) == 200
    assert manifest["co_primary_endpoints"] == []
    assert manifest["risk_contract"]["outcome_independent"] is True
    assert manifest["smoke_contract"]["logical_calls"] == 40
    assert "shared_weighted" not in str(manifest["primary_endpoint"])


def test_v11_high_consensus_primary_uses_frozen_score_only() -> None:
    rows = [
        {"R_PI": 0.9, "agreement": 1.0, "consensus_wrong": 1},
        {"R_PI": 0.1, "agreement": 0.8, "consensus_wrong": 0},
        {"R_PI": 1.0, "agreement": 0.6, "consensus_wrong": 1},
    ]
    high = [row for row in rows if row["agreement"] >= v11.HIGH_CONSENSUS_THRESHOLD]
    metric = v11._metric(high, "R_PI", "consensus_wrong")
    assert metric["n"] == 2
    assert metric["auroc"] == 1.0
