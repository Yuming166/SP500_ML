import json

import pytest

from sp500_forecastability import recovery_v3_7 as elar


def _example() -> dict:
    return {
        "example_id": "elar-x",
        "claim": "The writer was American.",
        "anchor": {
            "root": "anchor",
            "evidence": [{"evidence_id": "A00", "text": "The book was published."}],
        },
        "candidates": [
            {
                "root": "candidate",
                "evidence": [
                    {
                        "evidence_id": "C000",
                        "text": "The writer was American and lived in Boston.",
                    }
                ],
            },
            {
                "root": "distractor",
                "evidence": [{"evidence_id": "C100", "text": "A fantasy novel."}],
            },
        ],
    }


def _certificate() -> dict:
    return {
        "relation": "supports",
        "atomic_checks": [
            {
                "claim_span": "The writer was American",
                "status": "supported",
                "evidence_ids": ["C000"],
            }
        ],
    }


def _ledger_payload(quote: str = "The writer was American") -> dict:
    return {
        "entries": [
            {
                "atom_index": 0,
                "evidence_id": "C000",
                "evidence_quote": quote,
                "semantic_verdict": "entailed",
                "confidence": 0.9,
                "unsupported_terms": [],
            }
        ],
        "challenge": {
            "status": "none",
            "reason_code": "none",
            "claim_span": "",
            "evidence_id": "",
            "evidence_quote": "",
        },
    }


def test_protocol_freezes_root_disjoint_formal_size_and_auc_gate() -> None:
    assert elar.N_FORMAL == 400
    assert elar.N_PER_LABEL == 200
    assert elar.MAX_ORIENTED_ROLE_AUC == 0.65
    assert elar.LEDGER_MAX_COMPLETION_TOKENS == 768


def test_ledger_accepts_exact_local_quote_and_computes_coverage() -> None:
    ledger = elar.parse_ledger(
        json.dumps(_ledger_payload()),
        _example(),
        "candidate_0",
        _certificate(),
        "no",
    )
    assert ledger["all_expected_verdict"] is True
    assert ledger["challenge"]["status"] == "none"
    assert ledger["min_lexical_coverage"] == 1.0


def test_ledger_rejects_quote_not_present_in_cited_evidence() -> None:
    with pytest.raises(ValueError, match="exact evidence substring"):
        elar.parse_ledger(
            json.dumps(_ledger_payload("The writer was British")),
            _example(),
            "candidate_0",
            _certificate(),
            "no",
        )


def test_ledger_rejects_evidence_id_not_allowed_by_atomic_certificate() -> None:
    payload = _ledger_payload()
    payload["entries"][0]["evidence_id"] = "A00"
    payload["entries"][0]["evidence_quote"] = "The book was published"
    with pytest.raises(ValueError, match="certificate-local"):
        elar.parse_ledger(
            json.dumps(payload),
            _example(),
            "candidate_0",
            _certificate(),
            "no",
        )


def test_ledger_gate_is_fail_closed_and_honors_challenge() -> None:
    parsed = elar.parse_ledger(
        json.dumps(_ledger_payload()),
        _example(),
        "candidate_0",
        _certificate(),
        "no",
    )
    row = {"success": True, "ledger": parsed}
    assert elar._ledger_gate(
        row,
        confidence_threshold=0.8,
        lexical_threshold=0.8,
        unsupported_term_cap=0,
    )
    row["ledger"]["challenge"]["status"] = "found"
    assert not elar._ledger_gate(
        row,
        confidence_threshold=0.8,
        lexical_threshold=0.8,
        unsupported_term_cap=0,
    )
    assert not elar._ledger_gate(
        None,
        confidence_threshold=0.0,
        lexical_threshold=0.0,
        unsupported_term_cap=999,
    )


def test_required_atoms_follow_counter_consensus_direction() -> None:
    certificate = {
        "atomic_checks": [
            {"status": "supported"},
            {"status": "contradicted"},
            {"status": "unresolved"},
        ]
    }
    assert elar._required_atom_indices(certificate, "no") == [0]
    assert elar._required_atom_indices(certificate, "yes") == [1]
