"""Tests for Pilot-LLM V4 protocol structure and metrics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from sp500_forecastability.pilot_llm_v4 import (
    AGENT_PERSONAS,
    CONDITIONS,
    FACTS_PER_QUESTION,
    N_AGENTS,
    N_TOPIC_CLUSTERS,
    PARTITION_TABLE,
    PROTOCOL_VERSION,
    TruthfulQAItem,
    _auroc,
    _question_bootstrap_ci,
    _group_agents,
    _intervention_flip_rates,
    _label_from_three,
    _per_question_risks,
    _per_question_metric_bootstrap,
    _risk_at_coverage,
    agent_packet,
    build_composite_questions,
    build_evidence_view,
    build_manifest,
    build_substitute_manifest,
    first_content_word,
    load_truthfulqa,
    validate_manifest,
)

DATASET = Path(
    "/storage/gaoym/argumentative-llms/Datasets/TruthfulQA/Prompt/data.jsonl"
)


def test_partition_table_is_deterministic_and_complete():
    """PARTITION_TABLE must satisfy: 5 agents, each sees 2-of-3, all 3 IDs covered."""
    assert len(PARTITION_TABLE) == N_AGENTS == 5
    all_ids = {f"E0{i + 1}" for i in range(FACTS_PER_QUESTION)}
    for agent_sees in PARTITION_TABLE:
        assert len(agent_sees) == 2
        assert set(agent_sees) <= all_ids
    # Every evidence ID must appear in at least one agent's packet.
    seen_anywhere = set().union(*PARTITION_TABLE)
    assert seen_anywhere == all_ids


def test_first_content_word_skips_lead_stopwords():
    assert first_content_word("If you dream of doing something, ...") == "dream"
    assert first_content_word("The Aztec Empire existed ...") == "aztec"
    assert first_content_word("All Europeans speak French.") == "europeans"


def test_load_truthfulqa_handles_zero_one_valid():
    items = load_truthfulqa(DATASET)
    assert items, "TruthfulQA dataset must yield at least one item"
    labels = Counter(item.label for item in items)
    assert set(labels) == {True, False}, f"expected both labels, got {labels}"
    # Each TruthfulQA item has a derived qid.
    qids = [item.qid for item in items]
    assert len(set(qids)) == len(qids)


def test_label_from_three_majority():
    a = TruthfulQAItem(qid="a", claim="x", question="", answer="", label=False,
                       cluster=0)
    b = TruthfulQAItem(qid="b", claim="y", question="", answer="", label=True,
                       cluster=0)
    c = TruthfulQAItem(qid="c", claim="z", question="", answer="", label=False,
                       cluster=0)
    assert _label_from_three((a, b, c)) is False
    assert _label_from_three((b, a, c)) is False
    assert _label_from_three((b, c, a)) is False
    # Ties (1-2 majority) - 2 True vs 1 False.
    assert _label_from_three((b, b, a)) is True


def test_substitute_manifest_opposite_valid_same_cluster():
    items = load_truthfulqa(DATASET)
    manifest = build_substitute_manifest(items)
    # Items with substitutes must point to opposite-valid candidates from the
    # same cluster.
    by_qid = {item.qid: item for item in items}
    for qid, entry in manifest.items():
        if not entry["substitute_qid"]:
            continue
        source = by_qid[qid]
        target = by_qid[entry["substitute_qid"]]
        assert source.label != target.label, \
            f"substitute must have opposite valid for {qid}"
        assert source.cluster == target.cluster, \
            f"substitute must share cluster for {qid}"


def test_composite_questions_balanced_and_3_items_each():
    items = load_truthfulqa(DATASET)
    sub_manifest = build_substitute_manifest(items)
    composites = build_composite_questions(items, sub_manifest)
    assert len(composites) == 50
    label_counts = Counter(c.label for c in composites)
    assert label_counts == Counter({False: 25, True: 25})
    for comp in composites:
        assert len(comp.items) == FACTS_PER_QUESTION == 3
        # Gold label must match majority of items' labels.
        majority = Counter(item.label for item in comp.items).most_common(1)[0][0]
        assert comp.label == majority


def test_build_manifest_round_trips_through_validate():
    items = load_truthfulqa(DATASET)
    sub_manifest = build_substitute_manifest(items)
    composites = build_composite_questions(items, sub_manifest)
    manifest = build_manifest(DATASET, composites, sub_manifest)
    out_path = Path("/tmp/v4_manifest_test.json")
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    reloaded = json.loads(out_path.read_text())
    reconstructed = validate_manifest(reloaded, DATASET)
    assert len(reconstructed) == 50
    # Substitute manifest deviations are recorded.
    sample = next(iter(manifest["substitute_manifest"].values()))
    assert "deviation_log" in sample


def test_evidence_view_substitute_replaces_text():
    items = load_truthfulqa(DATASET)
    sub_manifest = build_substitute_manifest(items)
    composites = build_composite_questions(items, sub_manifest)
    comp = composites[0]
    sub_view = build_evidence_view(comp, agent_index=0, condition="substitute",
                                   substitute_manifest=sub_manifest)
    # Every visible item must have non-empty text.
    assert all(text for _, text in sub_view.items)
    # Partition must still be 2 items.
    assert len(sub_view.items) == 2


def test_agent_packet_returns_two_claims_per_agent():
    items = load_truthfulqa(DATASET)
    sub_manifest = build_substitute_manifest(items)
    composites = build_composite_questions(items, sub_manifest)
    comp = composites[0]
    for agent_index in range(N_AGENTS):
        packet = agent_packet(comp, agent_index)
        assert len(packet) == 2
        for claim in packet:
            assert isinstance(claim, str) and claim


def test_auroc_basic():
    assert _auroc([1, 0], [1, 0]) == 1.0
    assert _auroc([0, 1], [1, 0]) == 0.0
    assert _auroc([0.5, 0.5], [1, 0]) == 0.5


def test_risk_at_coverage_basic():
    # _risk_at_coverage keeps the top-coverage fraction by score and reports
    # the fraction of kept items with positive label (i.e. "errors caught at
    # the risk threshold"). Scores [0.9, 0.1], labels [1, 0] -> top 50%
    # keeps (0.9, label=1), so risk = 1.0.
    assert _risk_at_coverage([0.9, 0.1], [1, 0], 0.5) == 1.0
    # If the kept one is a true negative, risk = 0.
    assert _risk_at_coverage([0.9, 0.1], [0, 1], 0.5) == 0.0
    # Coverage 100% keeps everything, so risk = mean label.
    assert _risk_at_coverage([0.9, 0.1], [1, 0], 1.0) == 0.5


def test_per_question_risks_correctness_on_synthetic():
    """Build a synthetic record set where the math is known."""
    from sp500_forecastability.pilot_llm_v4 import CompositeQuestion
    # 1 question, 5 agents, all 4 conditions. All agents are inert (no flips)
    # but stable in confidence.
    records = []
    for agent_index in range(N_AGENTS):
        for condition in CONDITIONS:
            original_answer = "yes" if condition == "original" else "yes"  # inert
            records.append({
                "cqid": "test-q1",
                "label": False,
                "agent_id": AGENT_PERSONAS[agent_index][0],
                "agent_index": agent_index,
                "condition": condition,
                "decision": {
                    "agent_id": AGENT_PERSONAS[agent_index][0],
                    "answer": original_answer,
                    "confidence": 0.9,  # all stable
                    "cited_evidence_ids": ["E01"],
                    "decision": "answer",
                },
                "success": True,
            })
    grouped = {"test-q1": records}
    rows = _per_question_risks(grouped)
    assert len(rows) == 1
    row = rows[0]
    # All 5 agents are inert + stable, so D_inert = D_conf = D_OR = 1.0.
    assert row["D_inert"] == 1.0
    assert row["D_conf"] == 1.0
    assert row["D_OR"] == 1.0
    # Consensus "yes" but label False => correct=0, harmful_fc=1 (agreement=1.0 >= 0.8)
    assert row["correct"] == 0
    assert row["harmful_fc"] == 1


def test_per_question_metric_bootstrap_returns_ci():
    rows = [
        {"D_OR": 0.6, "harmful_fc": 1},
        {"D_OR": 0.4, "harmful_fc": 0},
        {"D_OR": 0.7, "harmful_fc": 1},
        {"D_OR": 0.3, "harmful_fc": 0},
    ] * 5  # 20 rows total
    lo, hi = _per_question_metric_bootstrap(_auroc, rows, "D_OR", "harmful_fc",
                                            n_replicates=200, seed=1)
    # AUROC should be ~1.0 since high D_OR aligns with positive class.
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo <= hi


def test_intervention_flip_rates_count_inert_and_flipping():
    records = []
    for agent_index in range(N_AGENTS):
        for condition in CONDITIONS:
            answer = "yes" if condition == "original" else (
                "no" if condition in ("remove", "reverse") else "yes"
            )
            records.append({
                "cqid": "q",
                "label": False,
                "agent_id": AGENT_PERSONAS[agent_index][0],
                "agent_index": agent_index,
                "condition": condition,
                "decision": {
                    "agent_id": AGENT_PERSONAS[agent_index][0],
                    "answer": answer,
                    "confidence": 0.9,
                    "cited_evidence_ids": [],
                    "decision": "answer",
                },
                "success": True,
            })
    rates = _intervention_flip_rates({"q": records})
    assert rates["remove"] == 1.0
    assert rates["reverse"] == 1.0
    assert rates["substitute"] == 0.0


def test_question_bootstrap_ci_endpoints():
    values = [0.1] * 20 + [0.9] * 20
    lo, hi = _question_bootstrap_ci(values, n_replicates=200, seed=42)
    assert lo <= 0.5 <= hi  # mixed bag, mean near 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])