from __future__ import annotations

import json

from sp500_forecastability import recovery_v3_13 as v313


def test_v313_development_grid_and_declared_selection_rule() -> None:
    manifest = v313.build_router_manifest()
    assert manifest["development_grid"]["0.13"]["harms"] == 1
    assert manifest["development_grid"]["0.15"] == {
        "accepted_routes": 13,
        "fixes": 13,
        "harms": 0,
        "net_fixes": 13,
    }
    assert manifest["thresholds"]["relation_confidence_margin"] == 0.15


def test_v313_fresh_selection_audit() -> None:
    selection = v313.build_selection()
    audit = v313.audit_selection(selection, rebuild=False)
    assert audit["passed"]
    assert audit["counts"] == {"formal": 80}
    assert audit["labels"] == {"Supported": 40, "Refuted": 40}
    assert audit["annotated_distinct_roots"] == 80
    assert audit["candidate_0_annotated_fraction"] == 0.5
    assert audit["maximum_auxiliary_root_reuse"] <= 7


def test_v313_selection_is_disjoint_from_all_declared_exposure() -> None:
    selection = v313.build_selection()
    audit = v313.audit_selection(selection, rebuild=False)
    assert audit["gates"]["zero_exposed_claim_overlap"]
    assert audit["gates"]["zero_exposed_root_overlap"]


def test_v313_frozen_artifacts_validate_when_present() -> None:
    if not v313.SELECTION_PATH.exists():
        return
    selection = json.loads(v313.SELECTION_PATH.read_text(encoding="utf-8"))
    assert v313.audit_selection(selection)["passed"]
    if v313.ROUTER_MANIFEST.exists():
        v313.validate_router_manifest()
    if v313.PROTOCOL_MANIFEST.exists():
        v313.validate_protocol_manifest()
