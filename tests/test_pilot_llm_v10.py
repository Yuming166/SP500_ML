"""Offline regression tests for the unexecuted V10.1 BoolQ protocol."""

from __future__ import annotations

import json
from functools import lru_cache
from types import SimpleNamespace

from sp500_forecastability import pilot_llm_v10 as v10


class _RewriteClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    def call(self, messages, *, seed):
        self.messages.append(messages)
        return SimpleNamespace(content=self.response)


@lru_cache(maxsize=1)
def _selected_composites():
    return tuple(v10.build_composite_questions(v10.load_boolq(v10.DEFAULT_DATASET)))


def test_v10_1_text_only_selection_is_balanced_and_leak_free() -> None:
    composites = _selected_composites()

    assert len(composites) == 100
    assert sum(comp.label == "yes" for comp in composites) == 50
    assert sum(comp.label == "no" for comp in composites) == 50
    for comp in composites:
        assert len(comp.items) == 3
        assert len({item.source_root for item in comp.items}) == 1
        assert [item.evidence_index for item in comp.items] == [1, 2, 3]
        assert len({item.question for item in comp.items}) == 1
        assert "Gold answer:" not in comp.question_text
        assert all(item.passage not in comp.question_text for item in comp.items)
        assert 0.08 <= comp.items[0].overlap_mean
        assert 0.10 <= comp.items[0].overlap_max <= 0.60


def test_v10_1_selection_manifest_validates_before_rewrites() -> None:
    composites = _selected_composites()
    manifest = v10.build_manifest(
        v10.DEFAULT_DATASET,
        composites,
        {},
        status="selection_frozen_pre_substitution",
    )

    restored = v10.validate_manifest(
        manifest, v10.DEFAULT_DATASET, require_substitutes=False
    )
    assert [comp.cqid for comp in restored] == [comp.cqid for comp in composites]
    assert manifest["selection"]["source_root"].startswith("sha256")


def test_v10_1_rewrite_uses_question_and_selected_items_only() -> None:
    comp = _selected_composites()[0]
    client = _RewriteClient(comp.items[0].passage)

    manifest, stats = v10.build_substitute_manifest(
        comp.items, client=client, cache_dir=None
    )

    assert stats["n_items"] == 3
    assert stats["n_unusable"] == 0
    assert stats["passed_fail_fast"] is True
    assert set(manifest) == {item.qid for item in comp.items}
    assert comp.items[0].question in client.messages[0][0]["content"]


def test_v10_1_agent_prompt_exposes_only_question_plus_packet() -> None:
    comp = _selected_composites()[0]
    substitutes = {
        item.qid: {
            "substitute_sentence": item.passage,
            "in_length_window": True,
        }
        for item in comp.items
    }
    view = v10.build_evidence_view(comp, 0, "original", substitutes)
    message = v10.build_messages(
        comp,
        view,
        agent_id=v10.AGENT_PERSONAS[0][0],
        persona=v10.AGENT_PERSONAS[0][1],
    )[1]["content"]
    payload = json.loads(message.split("Task payload:\n", 1)[1])

    assert payload["question"] == comp.question_text
    assert "Gold answer:" not in payload["question"]
    assert len(payload["evidence_packet"]) == 2
    assert payload["evidence_packet"][0]["text"] in {
        comp.items[0].passage,
        comp.items[1].passage,
        comp.items[2].passage,
    }


def test_v10_1_audit_freezes_exact_selection_without_replacement(tmp_path) -> None:
    composites = _selected_composites()
    selection = v10.build_manifest(
        v10.DEFAULT_DATASET,
        composites,
        {},
        status="selection_frozen_pre_substitution",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    substitutes = {
        item.qid: {
            "substitute_sentence": item.passage,
            "in_length_window": True,
        }
        for comp in composites
        for item in comp.items
    }
    (cache_dir / "substitute_manifest.json").write_text(
        json.dumps(substitutes), encoding="utf-8"
    )
    run_path = tmp_path / "run.json"

    audit = v10._pre_formal_audit(
        v10.DEFAULT_DATASET,
        selection_path,
        run_path,
        cache_dir=cache_dir,
    )

    assert audit["n_items"] == 300
    assert audit["n_substitute_hits"] == 300
    assert audit["passes_yield_threshold"] is True
    assert json.loads(run_path.read_text(encoding="utf-8"))["status"] == "run_frozen"


def test_v10_1_correctness_encoding_matches_yes_no_labels() -> None:
    yes = _selected_composites()[0]
    no = next(comp for comp in _selected_composites() if comp.label == "no")

    assert yes.gold_binary == (1 if yes.label == "yes" else 0)
    assert no.gold_binary == 0
