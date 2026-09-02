from __future__ import annotations

from sp500_forecastability import recovery_v3_7 as frozen
from sp500_forecastability import recovery_v3_8_3_analysis as analysis


def test_none_ledger_is_sanitized_only_as_empty_ledger() -> None:
    row = {"success": False, "ledger": None, "final_error": "schema"}
    valid = {"success": True, "ledger": {"min_confidence": 0.9}}
    result = analysis._sanitize_ledger_groups({("a", "x"): row, ("b", "y"): valid})
    assert result[("a", "x")] == {**row, "ledger": {}}
    assert result[("b", "y")] is valid


def test_patch_is_scoped_and_restored() -> None:
    original = frozen._select_elar
    with analysis._patched_diagnostic():
        assert frozen._select_elar is analysis._select_elar_none_safe
    assert frozen._select_elar is original


def test_analysis_manifest_is_outcome_blind_and_binds_records() -> None:
    manifest = analysis.build_analysis_manifest()
    assert manifest["primary_elar_changed"] is False
    assert manifest["target_model_calls"] is False
    assert manifest["outcomes_used_to_define_amendment"] is False
    assert set(manifest["formal_records_sha256"]) == {
        "actions",
        "certificates",
        "ledgers",
    }
