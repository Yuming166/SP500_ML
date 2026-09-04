from __future__ import annotations

from sp500_forecastability import detection_v3_16_1_formal as formal
from sp500_forecastability import detection_v3_16_1_selection as selection


def test_fresh_protocol_sizes_and_frozen_weights() -> None:
    assert selection.FRESH_PAIRS == 289
    assert selection.FORMAL_ITEMS == 578
    assert formal.EXPECTED_CALLS == 11_560
    assert formal.EXPECTED_RISK_WEIGHTS == {
        "reverse_inertia": 0.3,
        "remove_inertia": 0.0,
        "substitute_inertia": 0.0,
        "reverse_confident_nonresponse": 0.0,
        "intervention_disagreement": 0.7,
    }


def test_fresh_selection_excludes_parent_pages() -> None:
    payload = selection.build_selection(selection.load_rows())
    audit = selection.audit_selection(payload)
    assert audit["passed"] is True
    assert len(payload["pairs"]) == selection.FRESH_PAIRS
    parent_target_pages = set(payload["parent_selection"]["target_pages"])
    parent_pages = set(payload["parent_selection"]["excluded_pages"])
    target_pages = {row["page"] for row in payload["pairs"]}
    distractor_pages = {row["distractor_page"] for row in payload["pairs"]}
    assert not target_pages & parent_target_pages
    assert not distractor_pages & parent_pages
    assert not target_pages & distractor_pages


def test_formal_score_metrics_preserve_error_ledger() -> None:
    rows = []
    for index in range(20):
        label = "SUPPORTS" if index < 10 else "REFUTES"
        error = int(index in {0, 1, 10, 11})
        rows.append(
            {
                "opaque_id": f"item-{index}",
                "pair_id": f"pair-{index % 5}",
                "gold_label": label,
                "error": error,
                "risk": float(error),
                "agreement_risk": 0.5,
            }
        )
    risk = formal._score_metrics(rows, "risk")
    baseline = formal._score_metrics(rows, "agreement_risk")
    assert risk["errors"] == baseline["errors"] == 4
    assert risk["overall_auroc"] == 1.0
    assert risk["risk_at_80"]["retained_error"] < baseline["risk_at_80"]["retained_error"]
