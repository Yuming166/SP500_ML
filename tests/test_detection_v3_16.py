from __future__ import annotations

from sp500_forecastability import detection_v3_16 as v316


def _row(
    *, unique_id: str, label: str, evidence: str, page: str = "Example page"
) -> dict[str, str]:
    return {
        "unique_id": unique_id,
        "case_id": "case-1",
        "wiki_revision_id": "revision-1",
        "label": label,
        "claim": "The recorded population was exactly one million people.",
        "evidence": evidence,
        "page": page,
        "group": "AN01",
    }


def test_natural_contrastive_pair_passes_symmetric_gate() -> None:
    support = _row(
        unique_id="support",
        label="SUPPORTS",
        evidence=(
            "The official census reported that the recorded population was exactly "
            "one million people in the referenced year according to the national archive."
        ),
    )
    refute = _row(
        unique_id="refute",
        label="REFUTES",
        evidence=(
            "The official census reported that the recorded population was exactly "
            "two million people in the referenced year according to the national archive."
        ),
    )
    pairs = v316.candidate_pairs([support, refute])
    assert len(pairs) == 1
    assert pairs[0].supports_id == "support"
    assert pairs[0].refutes_id == "refute"
    assert pairs[0].page == "Example page"
    assert pairs[0].character_ratio >= v316.CHARACTER_RATIO_MIN
    assert pairs[0].token_jaccard >= v316.TOKEN_JACCARD_MIN


def test_pair_gate_rejects_encoding_damage() -> None:
    support = _row(
        unique_id="support",
        label="SUPPORTS",
        evidence="The official census contained replacement text and enough other words for testing.",
    )
    refute = _row(
        unique_id="refute",
        label="REFUTES",
        evidence="The official census contained replacement \ufffd and enough other words for testing.",
    )
    assert v316.candidate_pairs([support, refute]) == []


def test_one_pair_per_page_is_deterministic() -> None:
    evidence_a = (
        "The official census reported that the recorded population was exactly one million "
        "people in the referenced year according to the national archive."
    )
    evidence_b = evidence_a.replace("one million", "two million")
    rows = [_row(unique_id="s1", label="SUPPORTS", evidence=evidence_a)]
    rows.append(_row(unique_id="r1", label="REFUTES", evidence=evidence_b))
    second = [dict(row) for row in rows]
    for row in second:
        row["unique_id"] += "-second"
        row["case_id"] = "case-2"
    pairs = v316.candidate_pairs([*rows, *second])
    selected = v316.one_pair_per_page(pairs)
    assert len(pairs) == 2
    assert len(selected) == 1
    assert selected == v316.one_pair_per_page(list(reversed(pairs)))


def test_audit_requires_exact_pairwise_label_balance() -> None:
    pair = {
        "split": "smoke",
        "pair_id": "p",
        "page": "target",
        "distractor_page": "distractor",
        "character_ratio": 0.99,
        "token_jaccard": 0.95,
        "distractor_claim_jaccard": 0.0,
        "items": [
            {
                "item_id": "p:support",
                "gold_label": "SUPPORTS",
                "original_id": "s",
                "reverse_id": "r",
            },
            {
                "item_id": "p:refute",
                "gold_label": "REFUTES",
                "original_id": "r",
                "reverse_id": "s",
            },
        ],
    }
    pairs = []
    for split, count in (
        ("smoke", v316.SMOKE_PAIRS),
        ("development", v316.DEVELOPMENT_PAIRS),
        ("formal", v316.FORMAL_PAIRS),
    ):
        for index in range(count):
            row = {
                **pair,
                "split": split,
                "pair_id": f"{split}-{index}",
                "page": f"target-{split}-{index}",
                "distractor_page": f"distractor-{split}-{index}",
                "items": [dict(item) for item in pair["items"]],
            }
            row["items"][0]["item_id"] = f"{split}-{index}:support"
            row["items"][1]["item_id"] = f"{split}-{index}:refute"
            pairs.append(row)
    payload = {
        "pairs": pairs,
        "dataset": {"sha256": v316.file_sha256(v316.DATASET)},
        "archive": {"sha256": v316.EXPECTED_ARCHIVE_SHA256},
        "claim_boundary": {"formal_calls_authorized": False},
    }
    result = v316.audit_selection(payload)
    assert result["passed"] is True
    pairs[0]["items"][0]["gold_label"] = "REFUTES"
    assert v316.audit_selection(payload)["gates"]["labels_exactly_balanced"] is False
