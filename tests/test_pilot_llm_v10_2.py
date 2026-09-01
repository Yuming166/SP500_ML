"""Unit tests for V10.2's fixed, once-only rewrite repair contract."""

from __future__ import annotations

from types import SimpleNamespace

from sp500_forecastability import pilot_llm_v10 as v10
from sp500_forecastability import pilot_llm_v10_2 as v10_2


class _TwoStageClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], int]] = []

    def call(self, messages, *, seed):
        self.calls.append((messages, seed))
        if len(self.calls) == 1:
            return SimpleNamespace(content="too short")
        return SimpleNamespace(content="This counterfactual evidence sentence has exactly eight stable tokens.")


def test_v10_2_uses_one_fixed_repair_without_replacing_the_item() -> None:
    item = v10.BoolQItem(
        qid="fixed-e01", question="Is the claim true?",
        passage="This original evidence sentence has exactly eight stable tokens.",
        label="yes", source_root="source-fixed", evidence_index=1,
        overlap_mean=0.2, overlap_max=0.3,
    )
    client = _TwoStageClient()

    manifest, stats = v10_2.build_substitute_manifest_v10_2(
        [item], client=client, cache_dir=None,
    )

    assert list(manifest) == [item.qid]
    assert manifest[item.qid]["generation_mode"] == "length_repair"
    assert manifest[item.qid]["in_length_window"] is True
    assert stats["initial_valid"] == 0
    assert stats["repair_attempted"] == 1
    assert stats["repair_valid"] == 1
    assert stats["passed_fail_fast"] is True
    assert [seed for _messages, seed in client.calls] == [
        v10_2.INITIAL_REWRITE_SEED, v10_2.LENGTH_REPAIR_SEED,
    ]
    assert "Unusable prior candidate: too short" in client.calls[1][0][0]["content"]
