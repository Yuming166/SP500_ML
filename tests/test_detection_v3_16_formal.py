from __future__ import annotations

import json
from pathlib import Path

import pytest

from sp500_forecastability import detection_v3_16_formal as formal


def test_public_manifest_is_balanced_but_outcome_free() -> None:
    public, ledger = formal.build_manifests()
    audit = formal.audit_public_manifest(public, ledger)
    assert audit["passed"] is True
    assert len(public["items"]) == 500
    assert audit["label_counts"] == {"SUPPORTS": 250, "REFUTES": 250}
    forbidden = {"gold_label", "label", "correct", "error", "consensus_wrong"}
    assert not (forbidden & formal._recursive_keys(public))
    assert all(
        "support" not in item["opaque_item_id"] and "refute" not in item["opaque_item_id"]
        for item in public["items"]
    )


def test_formal_tasks_do_not_read_outcome_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public, _ = formal.build_manifests()
    public_path = tmp_path / "public.json"
    public_path.write_text(json.dumps(public), encoding="utf-8")
    monkeypatch.setattr(formal, "PUBLIC_MANIFEST", public_path)
    monkeypatch.setattr(formal, "OUTCOME_LEDGER", tmp_path / "must-not-exist.json")
    tasks = formal.load_tasks()
    assert len(tasks) == formal.EXPECTED_CALLS == 10_000
    assert all(task.split == "formal" for task in tasks)


def test_formal_item_and_evidence_ids_are_deterministic_and_opaque() -> None:
    item = formal._formal_item_id("pair", "natural-support-id")
    assert item == formal._formal_item_id("pair", "natural-support-id")
    assert "support" not in item
    evidence = formal._formal_evidence_id("Example Page", "row-1")
    assert evidence.startswith("root_") and "::evidence_" in evidence
    assert "Example" not in evidence


def test_risk_metrics_keep_identical_base_errors() -> None:
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
    assert risk["overall_auroc"] == pytest.approx(1.0)
    assert risk["risk_at_80"]["retained_error"] < baseline["risk_at_80"]["retained_error"]
