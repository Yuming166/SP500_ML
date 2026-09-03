from __future__ import annotations

import json

import pytest

from sp500_forecastability import detection_v3_16_calls as calls


def test_parser_enforces_exact_visible_citation() -> None:
    evidence_id = "root_abc::evidence_1"
    content = json.dumps(
        {
            "answer": "SUPPORTS",
            "confidence": 0.8,
            "cited_evidence_ids": [evidence_id],
        }
    )
    parsed = calls.parse_decision(content, [evidence_id])
    assert parsed["answer"] == "SUPPORTS"
    assert parsed["confidence"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="exact visible"):
        calls.parse_decision(content, ["root_other::evidence_2"])


def test_parser_requires_empty_citations_for_remove() -> None:
    content = json.dumps(
        {
            "answer": "REFUTES",
            "confidence": 0.5,
            "cited_evidence_ids": [],
        }
    )
    assert calls.parse_decision(content, [])["answer"] == "REFUTES"
    payload = json.loads(content)
    payload["cited_evidence_ids"] = ["invented"]
    with pytest.raises(ValueError, match="empty citations"):
        calls.parse_decision(json.dumps(payload), [])


def test_substitute_may_correctly_decline_to_cite_unrelated_evidence() -> None:
    content = json.dumps({"answer": "REFUTES", "confidence": 0.9, "cited_evidence_ids": []})
    evidence_id = "root_unrelated::evidence_1"
    parsed = calls.parse_decision(content, [evidence_id], allow_empty_nonempty=True)
    assert parsed["cited_evidence_ids"] == []
    with pytest.raises(ValueError, match="exact visible"):
        calls.parse_decision(content, [evidence_id], allow_empty_nonempty=False)


def test_registered_task_counts_and_natural_swap() -> None:
    smoke = calls.load_tasks("smoke")
    development = calls.load_tasks("development")
    assert len(smoke) == calls.SMOKE_CALLS == 160
    assert len(development) == calls.DEVELOPMENT_CALLS == 1200
    assert len({calls.task_key(task.__dict__) for task in smoke}) == len(smoke)
    originals = {
        (task.item_id, task.agent_index): task for task in smoke if task.condition == "original"
    }
    reverses = {
        (task.item_id, task.agent_index): task for task in smoke if task.condition == "reverse"
    }
    assert originals.keys() == reverses.keys()
    assert all(originals[key].evidence != reverses[key].evidence for key in originals)


def test_development_caller_refuses_formal_partition() -> None:
    with pytest.raises(ValueError, match="refuses"):
        calls.load_tasks("formal")


def test_seed_changes_by_model_and_condition() -> None:
    task = calls.load_tasks("smoke")[0]
    qwen = calls._seed(calls.MODELS["qwen"], task)
    ling = calls._seed(calls.MODELS["ling"], task)
    assert qwen != ling
    changed = calls.Task(**{**task.__dict__, "condition": "reverse"})
    assert qwen != calls._seed(calls.MODELS["qwen"], changed)
