from __future__ import annotations

import json
from pathlib import Path

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_7 as frozen
from sp500_forecastability import recovery_v3_8 as target


def _source_examples() -> list[dict]:
    payload = json.loads(target.SOURCE_SELECTION.read_text(encoding="utf-8"))
    return [row for row in payload["examples"] if row["split"] == "formal"]


def _tag(records: list[dict], *, split: str = "formal") -> list[dict]:
    return target._tag_rows(records, split=split)


def test_protocol_manifest_freezes_source_router_and_target() -> None:
    manifest = target.build_protocol_manifest()
    assert manifest["source"]["router_manifest_sha256"] == base._sha256_path(
        target.SOURCE_ROUTER
    )
    assert manifest["source"]["policy"] == {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    assert manifest["target"]["model"] == "Ling-3.0-tiny"
    assert manifest["target"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert manifest["target"]["runtime"]["vllm"] == "0.28.0"
    assert manifest["target"]["runtime"]["cuda_visible_devices"] == "4"
    assert manifest["target"]["runtime"]["generation_config"] == "vllm"
    assert manifest["server_script_sha256"] == base._sha256_path(target.SERVER_SCRIPT)
    assert manifest["evaluation"]["target_model_fit_or_calibration"] is False
    assert manifest["evaluation"]["bootstrap_replicates"] == 10_000


def test_protocol_manifest_contains_no_target_outcomes() -> None:
    serialized = json.dumps(target.build_protocol_manifest(), sort_keys=True).casefold()
    for forbidden in ("ling_accuracy", "ling_gain", "ling_correct", "target_label"):
        assert forbidden not in serialized


def test_target_client_uses_frozen_request_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "model": target.TARGET_MODEL,
                    "choices": [{"message": {"content": '{"answer":"yes"}'}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                }
            ).encode()

    def fake_urlopen(request, timeout):
        assert timeout == 60.0
        captured.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(target.urllib_request, "urlopen", fake_urlopen)
    client = target.CrossModelChatClient(cache_dir=tmp_path)
    client.max_completion_tokens = 512
    first = client.call([{"role": "user", "content": "test"}], seed=17)
    second = client.call([{"role": "user", "content": "test"}], seed=17)
    assert captured == [
        {
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 512,
            "messages": [{"content": "test", "role": "user"}],
            "model": target.TARGET_MODEL,
            "seed": 17,
            "temperature": 0.0,
        }
    ]
    assert first.cache_hit is False
    assert second.cache_hit is True


def test_target_client_rejects_wrong_response_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "model": "Qwen3.5-4B",
                    "choices": [{"message": {"content": '{"answer":"yes"}'}}],
                }
            ).encode()

    monkeypatch.setattr(target.urllib_request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    client = target.CrossModelChatClient(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="expected 'Ling-3.0-tiny'"):
        client.call([{"role": "user", "content": "test"}], seed=17)


def test_action_validator_accepts_only_complete_target_bundle() -> None:
    example = _source_examples()[0]
    source = [
        row
        for row in base._load_jsonl(
            target.DEFAULT_ROOT.parent
            / "recovery_v3_7_1"
            / "formal"
            / "actions"
            / "records.jsonl"
        )
        if row["example_id"] == example["example_id"]
    ]
    records = _tag(source)
    target._validate_actions([example], records, split="formal")
    with pytest.raises(ValueError, match="action coverage|invalid action bundle"):
        target._validate_actions([example], records[:-1], split="formal")


def test_certificate_validator_preserves_fail_closed_rows() -> None:
    example = _source_examples()[0]
    source = [
        row
        for row in base._load_jsonl(
            target.DEFAULT_ROOT.parent
            / "recovery_v3_7_1"
            / "formal"
            / "certificates"
            / "records.jsonl"
        )
        if row["example_id"] == example["example_id"]
    ]
    records = _tag(source)
    target._validate_certificates([example], records, split="formal")
    records[0]["runtime_model"] = "wrong-model"
    with pytest.raises(ValueError, match="certificate metadata"):
        target._validate_certificates([example], records, split="formal")


def test_ledger_validator_requires_target_protocol_and_model() -> None:
    source_ledgers = base._load_jsonl(
        target.DEFAULT_ROOT.parent
        / "recovery_v3_7_1"
        / "formal"
        / "ledgers"
        / "records.jsonl"
    )
    example_id = source_ledgers[0]["example_id"]
    example = next(row for row in _source_examples() if row["example_id"] == example_id)
    action_rows = [
        row
        for row in base._load_jsonl(
            target.DEFAULT_ROOT.parent
            / "recovery_v3_7_1"
            / "formal"
            / "actions"
            / "records.jsonl"
        )
        if row["example_id"] == example_id
    ]
    certificate_rows = [
        row
        for row in base._load_jsonl(
            target.DEFAULT_ROOT.parent
            / "recovery_v3_7_1"
            / "formal"
            / "certificates"
            / "records.jsonl"
        )
        if row["example_id"] == example_id
    ]
    actions = _tag(action_rows)
    certificates = _tag(certificate_rows)
    candidates = frozen._proof_candidates([example], actions, certificates)
    expected_actions = {action for _example, action, _certificate, _consensus in candidates}
    records = _tag(
        [
            row
            for row in source_ledgers
            if row["example_id"] == example_id and row["action"] in expected_actions
        ]
    )
    target._validate_ledgers(candidates, records, split="formal")
    records[0]["protocol_version"] = frozen.PROTOCOL_VERSION
    with pytest.raises(ValueError, match="ledger metadata"):
        target._validate_ledgers(candidates, records, split="formal")


def test_target_selection_is_model_held_out_and_training_root_disjoint() -> None:
    manifest = target.build_protocol_manifest()
    assert manifest["source"]["model"] != manifest["target"]["model"]
    assert manifest["evaluation"]["zero_claim_and_root_overlap_with_router_training"]
    assert len(_source_examples()) == target.EXPECTED_FORMAL


def test_attempt_stats_do_not_use_outcomes() -> None:
    stats = target._attempt_stats(
        [
            {
                "success": True,
                "first_pass_valid": True,
                "attempts": [{"cache_hit": False}],
                "gold_binary": 1,
            }
        ]
    )
    assert stats == {"rows": 1, "successful": 1, "first_pass_valid": 1, "cache_hits": 0}
