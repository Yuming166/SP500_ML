"""Zero-shot Qwen-to-Ling transfer of the frozen V3.7.1 ELAR router."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_7 as frozen
from sp500_forecastability.pilot_llm_v1 import (
    MAX_COMPLETION_TOKENS,
    MAX_RESPONSE_BYTES,
    ChatResult,
    _canonical_json,
)

PROTOCOL_VERSION = "recovery-v3.8-qwen-to-ling-elar-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v3_8_ling")
PREREGISTRATION = Path("docs/recovery_v3_8_preregistration.md")
SERVER_SCRIPT = Path("scripts/start_ling_v3_8.sh")
SOURCE_SELECTION = Path("results/recovery_v3_7_1/selection_manifest.json")
SOURCE_ROUTER = Path("results/recovery_v3_7_1/router/manifest.json")
SOURCE_SUMMARY = Path("results/recovery_v3_7_1/evaluation/summary.json")
TARGET_MODEL = "Ling-3.0-tiny"
TARGET_ENDPOINT = "http://127.0.0.1:31520/v1/chat/completions"
TARGET_MODEL_DIR = Path("/storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4")
TARGET_CONFIG = TARGET_MODEL_DIR / "config.json"
TARGET_WEIGHT_INDEX = TARGET_MODEL_DIR / "model.safetensors.index.json"
TARGET_SMALL_ARTIFACTS = (
    TARGET_CONFIG,
    TARGET_MODEL_DIR / "configuration.json",
    TARGET_MODEL_DIR / "configuration_bailing_moe_v3.py",
    TARGET_MODEL_DIR / "modeling_bailing_moe_v3.py",
    TARGET_MODEL_DIR / "generation_config.json",
    TARGET_MODEL_DIR / "tokenizer_config.json",
    TARGET_MODEL_DIR / "chat_template.jinja",
    TARGET_WEIGHT_INDEX,
)
VLLM_VERSION = "0.28.0"
TORCH_VERSION = "2.13.0"
TRANSFORMERS_VERSION = "5.16.1"
COMPRESSED_TENSORS_VERSION = "0.17.0"
THINKING_KWARGS = {"enable_thinking": False}
EXPECTED_FORMAL = 400
EXPECTED_PER_LABEL = 200
SMOKE_EXAMPLES = 2


class CrossModelChatClient:
    """Content-addressed Ling client with the frozen observable response contract."""

    def __init__(
        self,
        endpoint: str = TARGET_ENDPOINT,
        model: str = TARGET_MODEL,
        cache_dir: Path = DEFAULT_ROOT / "cache",
        timeout: float = 60.0,
    ) -> None:
        if endpoint != TARGET_ENDPOINT or model != TARGET_MODEL:
            raise ValueError("V3.8 endpoint and model are frozen")
        self.endpoint = endpoint
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_completion_tokens = MAX_COMPLETION_TOKENS
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def call(self, messages: Sequence[Mapping[str, str]], *, seed: int) -> ChatResult:
        request_payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": int(self.max_completion_tokens),
            "seed": seed,
            "chat_template_kwargs": THINKING_KWARGS,
        }
        cache_material = {"endpoint": self.endpoint, "request": request_payload}
        cache_key = frozen.sha256(_canonical_json(cache_material).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ChatResult(
                content=str(cached["content"]),
                model=str(cached["model"]),
                usage=dict(cached["usage"]),
                http_status=int(cached["http_status"]),
                request_bytes=0,
                response_bytes=int(cached["response_bytes"]),
                latency_seconds=0.0,
                cache_hit=True,
                cache_key=cache_key,
            )

        body = _canonical_json(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib_request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        started = time.monotonic()
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
        except urllib_error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except (urllib_error.URLError, TimeoutError) as error:
            raise RuntimeError(f"chat request failed: {error}") from error
        latency = time.monotonic() - started
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise ValueError("chat response exceeded the one-megabyte safety limit")
        try:
            response_payload = json.loads(response_body)
            content = response_payload["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("chat endpoint returned an unexpected schema") from error
        if not isinstance(content, str) or not content.strip():
            raise TypeError("chat response content must be nonempty text")
        response_model = str(response_payload.get("model", ""))
        if response_model != self.model:
            raise ValueError(
                f"chat endpoint returned model {response_model!r}, expected {self.model!r}"
            )
        usage_payload = response_payload.get("usage", {})
        usage = {
            "prompt_tokens": usage_payload.get("prompt_tokens"),
            "completion_tokens": usage_payload.get("completion_tokens"),
            "total_tokens": usage_payload.get("total_tokens"),
        }
        result = ChatResult(
            content=content,
            model=response_model,
            usage=usage,
            http_status=status,
            request_bytes=len(body),
            response_bytes=len(response_body),
            latency_seconds=latency,
            cache_hit=False,
            cache_key=cache_key,
        )
        base._write_json(
            cache_path,
            {
                "content": result.content,
                "model": result.model,
                "usage": result.usage,
                "http_status": result.http_status,
                "response_bytes": result.response_bytes,
            },
        )
        return result


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _source_policy() -> dict[str, Any]:
    return dict(json.loads(SOURCE_ROUTER.read_text(encoding="utf-8"))["policy"])


def _target_artifact_fingerprint() -> dict[str, Any]:
    index = json.loads(TARGET_WEIGHT_INDEX.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    shard_sizes = {}
    for name in shard_names:
        path = TARGET_MODEL_DIR / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty Ling weight shard: {path}")
        shard_sizes[name] = path.stat().st_size
    return {
        "small_file_sha256": {
            path.name: base._sha256_path(path) for path in TARGET_SMALL_ARTIFACTS
        },
        "weight_shard_sizes_bytes": shard_sizes,
    }


def build_protocol_manifest() -> dict[str, Any]:
    frozen.validate_router_manifest(SOURCE_ROUTER, SOURCE_SELECTION)
    selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    audit = frozen.audit_selection(selection)
    if not audit["passed"]:
        raise ValueError("source selection no longer passes its frozen audit")
    formal = [row for row in selection["examples"] if row["split"] == "formal"]
    counts = Counter(str(row["label"]) for row in formal)
    if len(formal) != EXPECTED_FORMAL or set(counts.values()) != {EXPECTED_PER_LABEL}:
        raise ValueError("frozen formal selection size or label balance drifted")
    policy = _source_policy()
    expected_policy = {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    if {key: policy[key] for key in expected_policy} != expected_policy:
        raise ValueError("source ELAR policy thresholds drifted")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_ling_formal_task_call",
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": base._sha256_path(PREREGISTRATION),
        "implementation_path": str(_implementation_path()),
        "implementation_sha256": base._sha256_path(_implementation_path()),
        "server_script_path": str(SERVER_SCRIPT),
        "server_script_sha256": base._sha256_path(SERVER_SCRIPT),
        "source": {
            "model": "Qwen3.5-4B",
            "selection_path": str(SOURCE_SELECTION),
            "selection_sha256": base._sha256_path(SOURCE_SELECTION),
            "router_manifest_path": str(SOURCE_ROUTER),
            "router_manifest_sha256": base._sha256_path(SOURCE_ROUTER),
            "formal_summary_sha256": base._sha256_path(SOURCE_SUMMARY),
            "router_protocol_version": frozen.PROTOCOL_VERSION,
            "policy": expected_policy,
        },
        "target": {
            "model": TARGET_MODEL,
            "endpoint": TARGET_ENDPOINT,
            "model_dir": str(TARGET_MODEL_DIR),
            "artifact_fingerprint": _target_artifact_fingerprint(),
            "runtime": {
                "vllm": VLLM_VERSION,
                "torch": TORCH_VERSION,
                "transformers": TRANSFORMERS_VERSION,
                "compressed_tensors": COMPRESSED_TENSORS_VERSION,
                "gpu": "NVIDIA GeForce RTX 4090",
                "cuda_visible_devices": "4",
                "context_length": 8192,
                "quantization": "compressed-tensors-int4",
                "generation_config": "vllm",
            },
            "temperature": 0.0,
            "chat_template_kwargs": THINKING_KWARGS,
            "schema_repair_attempts": 1,
        },
        "evaluation": {
            "formal_examples": EXPECTED_FORMAL,
            "per_native_label": EXPECTED_PER_LABEL,
            "same_fixed_roots_as_qwen_formal": True,
            "zero_claim_and_root_overlap_with_router_training": True,
            "all_target_model_observables_regenerated": True,
            "target_model_fit_or_calibration": False,
            "bootstrap_seed": frozen.BOOTSTRAP_SEED,
            "bootstrap_replicates": frozen.BOOTSTRAP_REPLICATES,
            "primary_gates": [
                "macro_gain_ci_lower_above_zero",
                "damage_rate_at_most_005",
                "both_label_groups_nonnegative",
                "annotation_supported_repairs_at_least_10",
                "net_fixes_above_keep_and_all_matched_baselines",
            ],
        },
        "feature_boundary": {
            "uses_target_gold_or_action_outcomes_at_inference": False,
            "uses_target_annotation_role_at_inference": False,
            "uses_source_identity_or_retrieval_score_at_inference": False,
            "source_router_and_thresholds_immutable": True,
            "certificate_or_ledger_failure_is_keep": True,
        },
    }


def validate_protocol_manifest(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = build_protocol_manifest()
    if manifest != expected:
        raise ValueError("V3.8 protocol manifest or a frozen dependency drifted")


def prepare(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> dict[str, Any]:
    formal_root = DEFAULT_ROOT / "formal"
    if not path.exists() and any(formal_root.glob("**/records*.jsonl")):
        raise ValueError("cannot freeze V3.8 after target formal records exist")
    expected = build_protocol_manifest()
    if path.exists():
        validate_protocol_manifest(path)
        return expected
    base._write_json(path, expected)
    return expected


def endpoint_model_ids() -> set[str]:
    url = TARGET_ENDPOINT.removesuffix("/chat/completions") + "/models"
    try:
        with urllib_request.urlopen(url, timeout=10.0) as response:
            payload = json.loads(response.read(MAX_RESPONSE_BYTES + 1))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ling model inventory failed: {error}") from error
    ids = {
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, Mapping) and "id" in item
    }
    if TARGET_MODEL not in ids:
        raise ValueError(f"expected {TARGET_MODEL!r} in endpoint inventory: {sorted(ids)}")
    return ids


def _examples(split: str) -> list[dict[str, Any]]:
    selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    return [dict(row) for row in selection["examples"] if row["split"] == split]


def _tag_rows(rows: Sequence[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    tagged = []
    for source in rows:
        row = dict(source)
        row["protocol_version"] = PROTOCOL_VERSION
        row["split"] = split
        row["runtime_endpoint"] = TARGET_ENDPOINT
        row["runtime_model"] = TARGET_MODEL
        tagged.append(row)
    return tagged


def _validate_actions(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> None:
    grouped = base._record_groups(records)
    expected = {str(example["example_id"]): example for example in examples}
    if set(grouped) != set(expected):
        raise ValueError(f"{split} action coverage mismatch")
    for example_id, rows in grouped.items():
        baseline = [row for row in rows if row.get("phase") == "baseline"]
        recovery = [row for row in rows if row.get("phase") == "recovery"]
        if len(baseline) != 5 or len(recovery) != 3:
            raise ValueError(f"invalid action bundle for {example_id}")
        if {row.get("agent_index") for row in baseline} != set(range(5)):
            raise ValueError(f"invalid baseline agents for {example_id}")
        if {row.get("action") for row in recovery} != set(base.RECOVERY_ACTIONS):
            raise ValueError(f"invalid recovery actions for {example_id}")
        if any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("runtime_model") != TARGET_MODEL
            or row.get("split") != split
            or row.get("gold_binary") != expected[example_id]["gold_binary"]
            or not row.get("success")
            or row.get("decision") is None
            for row in rows
        ):
            raise ValueError(f"invalid target action metadata for {example_id}")


def _validate_certificates(
    examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> None:
    grouped = base._record_groups(records)
    expected = {str(example["example_id"]) for example in examples}
    if set(grouped) != expected:
        raise ValueError(f"{split} certificate coverage mismatch")
    for example_id, rows in grouped.items():
        if len(rows) != 2 or {row.get("action") for row in rows} != set(
            v362.CERTIFICATE_ACTIONS
        ):
            raise ValueError(f"invalid certificate bundle for {example_id}")
        for row in rows:
            if (
                row.get("protocol_version") != PROTOCOL_VERSION
                or row.get("runtime_model") != TARGET_MODEL
                or row.get("split") != split
            ):
                raise ValueError("invalid target certificate metadata")
            if row.get("success") and row.get("certificate") is None:
                raise ValueError("successful certificate is empty")
            if not row.get("success") and (
                row.get("certificate") is not None
                or not row.get("final_error")
                or len(row.get("attempts", [])) != 2
            ):
                raise ValueError("invalid fail-closed certificate")


def execute_bundles(
    examples: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = base._load_jsonl(partial_path) if partial_path.exists() else []
    expected_per_example = 2 if kind == "certificates" else 8
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded:
        by_example[str(row["example_id"])].append(row)
    if any(len(rows) != expected_per_example for rows in by_example.values()):
        raise ValueError("partial file contains an incomplete target bundle")
    if kind == "certificates":
        terminal = lambda rows: all(
            row.get("success")
            or (
                row.get("certificate") is None
                and row.get("final_error")
                and len(row.get("attempts", [])) == 2
            )
            for row in rows
        )
        run_one = v362._run_certificate_example
    elif kind == "actions":
        terminal = lambda rows: all(row.get("success") for row in rows)
        run_one = v362._run_action_example
    else:
        raise ValueError("kind must be actions or certificates")
    existing = {key: rows for key, rows in by_example.items() if terminal(rows)}
    records = [row for rows in existing.values() for row in rows]
    allowed = {str(row["example_id"]) for row in examples}
    if set(existing) - allowed:
        raise ValueError("partial file contains examples outside target run")
    pending = [row for row in examples if str(row["example_id"]) not in existing]
    client = CrossModelChatClient(cache_dir=cache_dir)
    if kind == "certificates":
        client.max_completion_tokens = v362.ATOMIC_MAX_COMPLETION_TOKENS
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_one, client, row): row for row in pending}
        for future in as_completed(futures):
            bundle = _tag_rows(future.result(), split=split)
            records.extend(bundle)
            records.sort(
                key=lambda row: (
                    str(row["example_id"]),
                    str(row.get("phase", "certificate")),
                    -1 if row.get("agent_index") is None else int(row["agent_index"]),
                    str(row["action"]),
                )
            )
            base._write_jsonl(partial_path, records)
            print(
                f"[{len(records) // expected_per_example}/{len(examples)}] "
                f"{bundle[0]['example_id']} success="
                f"{all(row['success'] for row in bundle)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if len(records) != len(examples) * expected_per_example:
        raise ValueError("target bundle run incomplete")
    if kind == "certificates":
        _validate_certificates(examples, records, split=split)
    else:
        _validate_actions(examples, records, split=split)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


def _validate_ledgers(
    candidates: Sequence[tuple[Mapping[str, Any], str, Mapping[str, Any], str]],
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> None:
    expected = {
        (str(example["example_id"]), action)
        for example, action, _certificate, _consensus in candidates
    }
    actual = {(str(row["example_id"]), str(row["action"])) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError(f"{split} target ledger coverage mismatch")
    for row in records:
        if (
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("runtime_model") != TARGET_MODEL
            or row.get("split") != split
            or row.get("action") not in v362.CERTIFICATE_ACTIONS
        ):
            raise ValueError("invalid target ledger metadata")
        if row.get("success"):
            if row.get("ledger") is None:
                raise ValueError("successful target ledger is empty")
        elif (
            row.get("ledger") is not None
            or not row.get("final_error")
            or len(row.get("attempts", [])) != 2
        ):
            raise ValueError("invalid fail-closed target ledger row")


def execute_ledgers(
    examples: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    *,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    candidates = frozen._proof_candidates(examples, actions, certificates)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    loaded = base._load_jsonl(partial_path) if partial_path.exists() else []
    existing = {
        (str(row["example_id"]), str(row["action"])): row for row in loaded
    }
    expected_keys = {
        (str(example["example_id"]), action)
        for example, action, _certificate, _consensus in candidates
    }
    if set(existing) - expected_keys:
        raise ValueError("partial ledger file contains candidates outside target run")
    pending = [
        item
        for item in candidates
        if (str(item[0]["example_id"]), item[1]) not in existing
    ]
    records = list(existing.values())
    client = CrossModelChatClient(cache_dir=cache_dir)
    client.max_completion_tokens = frozen.LEDGER_MAX_COMPLETION_TOKENS

    def run_one(
        item: tuple[Mapping[str, Any], str, Mapping[str, Any], str]
    ) -> dict[str, Any]:
        example, action, certificate, consensus = item
        ledger, attempts, final_error = frozen._call_ledger_with_retry(
            client, example, action, certificate, consensus
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "runtime_endpoint": TARGET_ENDPOINT,
            "runtime_model": TARGET_MODEL,
            "example_id": str(example["example_id"]),
            "split": split,
            "action": action,
            "success": ledger is not None,
            "first_pass_valid": ledger is not None and len(attempts) == 1,
            "attempts": attempts,
            "ledger": ledger,
            "final_error": final_error,
        }

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_one, item): item for item in pending}
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            records.sort(key=lambda item: (str(item["example_id"]), str(item["action"])))
            base._write_jsonl(partial_path, records)
            print(
                f"[{len(records)}/{len(candidates)}] "
                f"{row['example_id']}:{row['action']} success={row['success']} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    _validate_ledgers(candidates, records, split=split)
    base._write_jsonl(output_dir / "records.jsonl", records)
    return records


def _attempt_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "successful": sum(bool(row.get("success")) for row in records),
        "first_pass_valid": sum(bool(row.get("first_pass_valid")) for row in records),
        "cache_hits": sum(
            bool(attempt.get("cache_hit"))
            for row in records
            for attempt in row.get("attempts", [])
        ),
    }


def evaluate(output_dir: Path = DEFAULT_ROOT / "evaluation") -> dict[str, Any]:
    validate_protocol_manifest()
    frozen.validate_router_manifest(SOURCE_ROUTER, SOURCE_SELECTION)
    examples = _examples("formal")
    actions_path = DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"
    certificates_path = DEFAULT_ROOT / "formal" / "certificates" / "records.jsonl"
    ledgers_path = DEFAULT_ROOT / "formal" / "ledgers" / "records.jsonl"
    actions = base._load_jsonl(actions_path)
    certificates = base._load_jsonl(certificates_path)
    ledgers = base._load_jsonl(ledgers_path)
    _validate_actions(examples, actions, split="formal")
    _validate_certificates(examples, certificates, split="formal")
    candidates = frozen._proof_candidates(examples, actions, certificates)
    _validate_ledgers(candidates, ledgers, split="formal")
    action_groups = base._record_groups(actions)
    certificate_groups = v362._certificate_groups(certificates)
    ledger_groups = frozen._ledger_groups(ledgers)
    predictions = frozen._prior_predictions(examples, actions, certificates)
    parameters = _source_policy()
    primary = frozen._select_elar(
        examples,
        action_groups,
        certificate_groups,
        ledger_groups,
        predictions,
        confidence_threshold=float(parameters["confidence_threshold"]),
        lexical_threshold=float(parameters["lexical_threshold"]),
        unsupported_term_cap=int(parameters["unsupported_term_cap"]),
    )
    proof_only = frozen._select_elar(
        examples,
        action_groups,
        certificate_groups,
        ledger_groups,
        predictions,
        confidence_threshold=0.0,
        lexical_threshold=0.0,
        unsupported_term_cap=10**9,
        require_ledger=False,
    )
    policies = {
        "elar": primary,
        "atomic_proof_only": proof_only,
        "keep": {str(row["example_id"]): "KEEP" for row in examples},
    }
    root_budget = sum(action != "KEEP" for action in primary.values())
    for name, proposed in v34._comparison_proposals(examples, action_groups).items():
        policies[f"matched_{name}"] = base._truncate_to_budget(
            examples, proposed, root_budget, name=name
        )
        policies[f"unlimited_{name}"] = proposed
    output_dir.mkdir(parents=True, exist_ok=True)
    preoutcome_path = output_dir / "preoutcome_routes.json"
    preoutcome = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest_sha256": base._sha256_path(
            DEFAULT_ROOT / "protocol_manifest.json"
        ),
        "source_router_manifest_sha256": base._sha256_path(SOURCE_ROUTER),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "target_certificate_records_sha256": base._sha256_path(certificates_path),
        "target_ledger_records_sha256": base._sha256_path(ledgers_path),
        "outcomes_accessed_by_route_selection": False,
        "target_model_fit_or_calibration": False,
        "policies": policies,
        "predictions": {
            f"{example_id}:{action}": {"p_fix": values[0], "p_harm": values[1]}
            for (example_id, action), values in predictions.items()
        },
    }
    v34._write_or_validate_preoutcome(preoutcome_path, preoutcome)

    oracle = {}
    for example in examples:
        example_id = str(example["example_id"])
        keep, agreement, outcomes, _baseline = base._outcomes(
            example, action_groups[example_id]
        )
        oracle[example_id] = (
            max(
                ("KEEP", *base.RECOVERY_ACTIONS),
                key=lambda action: keep if action == "KEEP" else outcomes[action],
            )
            if agreement >= base.HIGH_CONSENSUS
            else "KEEP"
        )
    policies["available_action_oracle_diagnostic"] = oracle
    metrics = {
        name: frozen._policy_metrics(examples, action_groups, selected)
        for name, selected in policies.items()
    }
    primary_metrics = metrics["elar"]
    matched_names = [name for name in metrics if name.startswith("matched_")]
    gates = {
        "macro_gain_ci_lower_above_zero": primary_metrics["macro_gain_ci"][0] > 0,
        "damage_rate_at_most_005": (
            primary_metrics["damage_rate_high_consensus_correct"] <= 0.05
        ),
        "both_label_groups_nonnegative": all(
            group["net_gain"] >= 0
            for group in primary_metrics["by_native_label"].values()
        ),
        "annotation_supported_repairs_at_least_10": (
            primary_metrics["annotation_supported_repairs"] >= 10
        ),
        "net_fixes_above_keep_and_all_matched_baselines": primary_metrics["net_fixes"]
        > max(0, *(metrics[name]["net_fixes"] for name in matched_names)),
    }
    routed = [row for row in examples if primary[str(row["example_id"])] != "KEEP"]
    annotated_choices = sum(
        row["candidates"][int(primary[str(row["example_id"])][-1])]["annotation_role"]
        == "held_out_annotated_root"
        for row in routed
    )
    source_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": "zero_shot_qwen_to_ling_fixed_root_formal",
        "protocol_manifest_sha256": base._sha256_path(
            DEFAULT_ROOT / "protocol_manifest.json"
        ),
        "source_router_manifest_sha256": base._sha256_path(SOURCE_ROUTER),
        "target_action_records_sha256": base._sha256_path(actions_path),
        "target_certificate_records_sha256": base._sha256_path(certificates_path),
        "target_ledger_records_sha256": base._sha256_path(ledgers_path),
        "preoutcome_routes_sha256": base._sha256_path(preoutcome_path),
        "n_formal": len(examples),
        "root_budget": root_budget,
        "transport_and_schema": {
            "actions": _attempt_stats(actions),
            "certificates": _attempt_stats(certificates),
            "ledgers": _attempt_stats(ledgers),
        },
        "ledger_validity": {
            "proof_candidates": len(candidates),
            "valid_rows": sum(bool(row.get("success")) for row in ledgers),
            "fail_closed_rows": sum(not bool(row.get("success")) for row in ledgers),
        },
        "policies": metrics,
        "primary_gates": gates,
        "passes": all(gates.values()),
        "verdict": (
            "PASS_ZERO_SHOT_CROSS_MODEL_ELAR_V3_8"
            if all(gates.values())
            else "NO_VERIFIED_CROSS_MODEL_ELAR_TRANSFER"
        ),
        "annotation_role_selection": {
            "routed": len(routed),
            "annotated_root_selected": annotated_choices,
            "accuracy": annotated_choices / max(1, len(routed)),
        },
        "paired_source_qwen": {
            "protocol_version": source_summary["protocol_version"],
            "elar": source_summary["policies"]["elar"],
        },
        "claim_boundary": {
            "source_model": "Qwen3.5-4B",
            "target_model": TARGET_MODEL,
            "target_fit_or_calibration": False,
            "root_disjoint_from_router_training": True,
            "same_roots_as_prior_qwen_formal": True,
            "static_wikipedia_page_roots": True,
            "publisher_independence": False,
            "universal_cross_model_transfer": False,
        },
    }
    base._write_json(output_dir / "summary.json", summary)
    _write_report(summary, output_dir / "report.md")
    return summary


def _write_report(summary: Mapping[str, Any], path: Path) -> None:
    primary = summary["policies"]["elar"]
    keep = summary["policies"]["keep"]
    gates = summary["primary_gates"]
    lines = [
        "# Recovery V3.8 result: zero-shot Qwen-to-Ling ELAR",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        "| metric | KEEP | zero-shot ELAR |",
        "| --- | ---: | ---: |",
        f"| accuracy | {keep['final_accuracy']:.2%} | {primary['final_accuracy']:.2%} |",
        f"| native-label macro gain | 0.00pp | {100 * primary['macro_label_gain']:+.2f}pp |",
        f"| fixes / harms | 0 / 0 | {primary['fixes']} / {primary['harms']} |",
        f"| acquired roots | 0 | {primary['total_added_roots']} |",
        "",
        (
            "Macro-gain 95% CI: "
            f"[{100 * primary['macro_gain_ci'][0]:+.2f}, "
            f"{100 * primary['macro_gain_ci'][1]:+.2f}]pp."
        ),
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gates.items())
    lines.extend(
        [
            "",
            "The router, thresholds, prompts, seeds, and formal roots were frozen before",
            "target formal calls. Ling outcomes were not used for fitting or calibration.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def smoke(workers: int) -> None:
    validate_protocol_manifest()
    endpoint_model_ids()
    examples = _examples("development")[:SMOKE_EXAMPLES]
    actions = execute_bundles(
        examples,
        kind="actions",
        split="smoke",
        output_dir=DEFAULT_ROOT / "smoke" / "actions",
        cache_dir=DEFAULT_ROOT / "cache",
        workers=workers,
    )
    certificates = execute_bundles(
        examples,
        kind="certificates",
        split="smoke",
        output_dir=DEFAULT_ROOT / "smoke" / "certificates",
        cache_dir=DEFAULT_ROOT / "cache",
        workers=workers,
    )
    execute_ledgers(
        examples,
        actions,
        certificates,
        split="smoke",
        output_dir=DEFAULT_ROOT / "smoke" / "ledgers",
        cache_dir=DEFAULT_ROOT / "cache",
        workers=workers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("endpoint-check")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--workers", type=int, default=2)
    for name in ("formal-actions", "formal-certificates", "formal-ledgers"):
        formal_parser = subparsers.add_parser(name)
        formal_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
        return 0
    if args.command == "endpoint-check":
        validate_protocol_manifest()
        print(json.dumps(sorted(endpoint_model_ids())))
        return 0
    if args.command == "smoke":
        smoke(args.workers)
        return 0
    validate_protocol_manifest()
    examples = _examples("formal")
    if args.command in {"formal-actions", "formal-certificates"}:
        kind = args.command.removeprefix("formal-")
        execute_bundles(
            examples,
            kind=kind,
            split="formal",
            output_dir=DEFAULT_ROOT / "formal" / kind,
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        return 0
    if args.command == "formal-ledgers":
        execute_ledgers(
            examples,
            base._load_jsonl(DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"),
            base._load_jsonl(
                DEFAULT_ROOT / "formal" / "certificates" / "records.jsonl"
            ),
            split="formal",
            output_dir=DEFAULT_ROOT / "formal" / "ledgers",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=args.workers,
        )
        return 0
    if args.command == "evaluate":
        summary = evaluate()
        print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
