import json
from pathlib import Path

import pytest

from sp500_forecastability import recovery_v2 as recovery


def _example() -> dict[str, object]:
    return {
        "example_id": "x1",
        "split": "train",
        "claim": "The example claim is supported.",
        "label": "SUPPORTS",
        "gold_binary": 1,
        "anchor": {
            "root": "Anchor_Page",
            "retrieval_score": 0.2,
            "title_overlap": 0.1,
            "evidence": [{"evidence_id": "A00", "text": "Anchor evidence."}],
        },
        "candidates": [
            {
                "root": "Candidate_Zero",
                "annotation_role": "held_out_annotated_root",
                "retrieval_score": 0.4,
                "title_overlap": 0.2,
                "evidence": [{"evidence_id": "C000", "text": "Candidate zero."}],
            },
            {
                "root": "Candidate_One",
                "annotation_role": "unannotated_retrieval_candidate",
                "retrieval_score": 0.5,
                "title_overlap": 0.3,
                "evidence": [{"evidence_id": "C100", "text": "Candidate one."}],
            },
        ],
    }


def _baseline_rows() -> list[dict[str, object]]:
    return [
        {
            "decision": {
                "answer": "yes",
                "confidence": 0.8,
                "cited_evidence_ids": ["A00"],
            }
        }
        for _ in range(5)
    ]


def test_frozen_selection_passes_structural_audit() -> None:
    path = Path("results/recovery_v2_2/selection_manifest.json")
    selection = json.loads(path.read_text(encoding="utf-8"))
    report = recovery.audit_selection(selection)
    assert report["passed"] is True
    assert report["split_counts"] == {"train": 689, "dev": 230, "test": 230}
    assert not any(report["cross_split_page_overlap"].values())


def test_candidate_position_does_not_reveal_annotation() -> None:
    selection = recovery.build_selection()
    annotated_at_zero = sum(
        row["candidates"][0]["annotation_role"] == "held_out_annotated_root"
        for row in selection["examples"]
    )
    fraction = annotated_at_zero / len(selection["examples"])
    assert 0.45 <= fraction <= 0.55


def test_prompts_do_not_expose_gold_or_annotation_role() -> None:
    example = _example()
    baseline = recovery.build_baseline_messages(example, 0)
    routed = recovery.build_recovery_messages(example, "candidate_0", "no")
    text = "\n".join(message["content"] for message in [*baseline, *routed])
    assert "SUPPORTS" not in text
    assert "gold_binary" not in text
    assert "held_out_annotated_root" not in text


def test_parser_rejects_out_of_packet_citation() -> None:
    valid = json.dumps({
        "answer": "yes",
        "confidence": 0.8,
        "cited_evidence_ids": ["A00", "C000"],
    })
    assert recovery.parse_decision(valid, ["A00", "C000"])["answer"] == "yes"
    with pytest.raises(ValueError, match="outside action packet"):
        recovery.parse_decision(valid, ["A00"])


def test_parser_normalizes_only_one_frozen_confidence_quote_pattern() -> None:
    malformed = '{"answer":"no","confidence":1.0","cited_evidence_ids":["A00"]}'
    parsed = recovery.parse_decision(malformed, ["A00"])
    assert parsed["confidence"] == 1.0
    assert parsed["parse_mode"] == "normalized_single_confidence_quote"
    with pytest.raises(ValueError):
        recovery.parse_decision(malformed.replace('1.0"', '1.0""'), ["A00"])


def test_feature_schema_excludes_outcome_and_annotation_fields() -> None:
    names, values = recovery._feature_vector(_example(), _baseline_rows(), "candidate_0")
    assert len(names) == len(values)
    assert not any(
        fragment in name
        for name in names
        for fragment in recovery.FORBIDDEN_FEATURE_FRAGMENTS
    )


def test_calibration_margin_is_one_sided_and_nonnegative() -> None:
    scores = recovery.np.asarray([-1.0, -0.2, 0.1, 0.4, 0.8])
    assert recovery._calibration_quantile(scores, 0.8) >= 0.0


def test_relocated_endpoint_is_bounded_without_changing_legacy_client() -> None:
    client = recovery.RecoveryChatClient(
        recovery.RELOCATED_RUNTIME_ENDPOINT,
        recovery.DEFAULT_MODEL,
        Path("results/recovery_v2_2/cache"),
    )
    assert client.endpoint == "http://10.63.0.82:31518/v1/chat/completions"
    with pytest.raises(ValueError, match="not preregistered"):
        recovery.RecoveryChatClient(
            "http://example.invalid/v1/chat/completions",
            recovery.DEFAULT_MODEL,
            Path("results/recovery_v2_2/cache"),
        )
