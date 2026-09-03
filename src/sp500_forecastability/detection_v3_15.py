"""Frozen cross-family replication of the BoolQ V12.1 detection protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from sp500_forecastability import pilot_llm_v10 as base
from sp500_forecastability import pilot_llm_v11 as v11
from sp500_forecastability import pilot_llm_v12 as v12
from sp500_forecastability import pilot_llm_v12_1 as v121
from sp500_forecastability.pilot_llm_v1 import (
    MAX_RESPONSE_BYTES,
    ChatResult,
    _agent_seed,
    _attempt_payload,
    _canonical_json,
    _extract_json_object,
    _write_json,
    _write_jsonl,
    file_sha256,
)

PROTOCOL_VERSION = "detection-v3.15-ling-boolq-v12.1-2026-09-03"
DEFAULT_ROOT = Path("results/detection_v3_15_ling")
PROTOCOL_MANIFEST = DEFAULT_ROOT / "protocol_manifest.json"
PREREGISTRATION = Path("docs/detection_v3_15_preregistration.md")
RUN_SCRIPT = Path("scripts/run_detection_v3_15.sh")
SERVER_SCRIPT = Path("scripts/start_ling_v3_8.sh")
PARENT_MANIFEST = Path("results/pilot_llm_v12_1/manifest.json")
PARENT_RECORDS = Path("results/pilot_llm_v12_1/formal/records.jsonl")
PARENT_SUMMARY = Path("results/pilot_llm_v12_1/formal/summary.json")
PARENT_MANIFEST_SHA256 = "dea715447a2f7e3f32db22fc8a2bc04fa17920611ff90293ccd06bd72717ef68"
PARENT_RECORDS_SHA256 = "1c9a1e4d01b198a5e4e6f7527500c44b9058a772a93afb9844c3ef3b22d7e5a8"
PARENT_SUMMARY_SHA256 = "6a2b90969cac1fc1a8af69bd448a198c627b3fb4ef213ae435bf16e838f4b9e8"
LING_TRANSPORT_MANIFEST = Path("results/recovery_v3_8_3_ling/protocol_manifest.json")
LING_TRANSPORT_MANIFEST_SHA256 = "4821b88d667e15a503b43c9a8253de823656dd14b02d7568e9a734c05ecb0397"

TARGET_MODEL = "Ling-3.0-tiny"
TARGET_ENDPOINT = "http://127.0.0.1:31520/v1/chat/completions"
TARGET_MODEL_DIR = Path("/storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4")
DATASET = v12.DEFAULT_DATASET
EXPECTED_EXAMPLES = 358
EXPECTED_CALLS = 7_160
N_WORKERS = 4
CONDITIONS = tuple(base.CONDITIONS)
RISK_WEIGHTS = {"D_inert": 0.1, "flip_inertia": 0.3, "frac_shared": 0.6}
HIGH_CONSENSUS = 0.8
ROUTER_COVERAGE = 0.8
BOOTSTRAP_SEED = 20_261_503
BOOTSTRAP_REPLICATES = 10_000
SMOKE_EXAMPLES = 4
SMOKE_CALLS = 80
SMOKE_MIN_FIRST_PASS = 76
MAX_COMPLETION_TOKENS = 160
THINKING_KWARGS = {"enable_thinking": False}
FORBIDDEN_PREOUTCOME = {"label", "gold_binary", "correct", "consensus_wrong", "harmful_fc"}


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _validated_parent() -> tuple[dict[str, Any], list[base.CompositeQuestion]]:
    expected = {
        PARENT_MANIFEST: PARENT_MANIFEST_SHA256,
        PARENT_RECORDS: PARENT_RECORDS_SHA256,
        PARENT_SUMMARY: PARENT_SUMMARY_SHA256,
        LING_TRANSPORT_MANIFEST: LING_TRANSPORT_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"frozen V3.15 dependency drifted: {path}")
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    composites = v121.validate_manifest_v12_1(manifest, DATASET, require_substitutes=True)
    if len(composites) != EXPECTED_EXAMPLES:
        raise ValueError("V3.15 must inherit all 358 V12.1 composites")
    return manifest, composites


def _target_fingerprint() -> dict[str, Any]:
    names = (
        "config.json",
        "configuration.json",
        "configuration_bailing_moe_v3.py",
        "modeling_bailing_moe_v3.py",
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    )
    paths = [TARGET_MODEL_DIR / name for name in names]
    if any(not path.is_file() for path in paths):
        raise ValueError("Ling target fingerprint is incomplete")
    index = json.loads((TARGET_MODEL_DIR / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    return {
        "small_file_sha256": {path.name: file_sha256(path) for path in paths},
        "weight_shard_sizes_bytes": {
            name: (TARGET_MODEL_DIR / name).stat().st_size for name in shards
        },
    }


def build_protocol_manifest() -> dict[str, Any]:
    parent, composites = _validated_parent()
    if file_sha256(DATASET) != v12.DATASET_SHA256:
        raise ValueError("BoolQ validation data drifted")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_any_ling_boolq_detection_call",
        "preregistration": {"path": str(PREREGISTRATION), "sha256": file_sha256(PREREGISTRATION)},
        "implementation": {
            "path": str(_implementation_path()),
            "sha256": file_sha256(_implementation_path()),
        },
        "run_script": {"path": str(RUN_SCRIPT), "sha256": file_sha256(RUN_SCRIPT)},
        "server_script": {"path": str(SERVER_SCRIPT), "sha256": file_sha256(SERVER_SCRIPT)},
        "parent": {
            "protocol": str(parent["protocol_version"]),
            "manifest": str(PARENT_MANIFEST),
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "qwen_records_sha256": PARENT_RECORDS_SHA256,
            "qwen_summary_sha256": PARENT_SUMMARY_SHA256,
            "selection_changed": False,
            "substitutes_changed": False,
            "prompts_changed": False,
            "questions": len(composites),
            "logical_calls": EXPECTED_CALLS,
        },
        "dataset": {"path": str(DATASET), "sha256": file_sha256(DATASET)},
        "target": {
            "model": TARGET_MODEL,
            "family": "Ling_BailingMoeV3",
            "endpoint": TARGET_ENDPOINT,
            "model_dir": str(TARGET_MODEL_DIR),
            "artifact_fingerprint": _target_fingerprint(),
            "temperature": 0.0,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "chat_template_kwargs": THINKING_KWARGS,
            "schema_repair_attempts": 1,
            "prior_exposure": {
                "scope": "FEVER recovery transport and actions only; no BoolQ V12.1 outputs",
                "manifest": str(LING_TRANSPORT_MANIFEST),
                "manifest_sha256": LING_TRANSPORT_MANIFEST_SHA256,
            },
        },
        "risk_contract": {
            "score": "0.1*D_inert + 0.3*flip_inertia + 0.6*frac_shared",
            "weights": RISK_WEIGHTS,
            "high_consensus_threshold": HIGH_CONSENSUS,
            "router_coverage": ROUTER_COVERAGE,
            "outcome_independent": True,
            "forbidden_inputs": sorted(FORBIDDEN_PREOUTCOME),
        },
        "analysis": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "aggregate_replication_gates": [
                "high_consensus_count_and_class_gate",
                "aggregate_auroc_ci_lower_above_0_5",
                "risk80_error_reduction_ci_lower_above_zero",
                "final_valid_rate_is_one",
                "first_pass_valid_rate_at_least_0_95",
            ],
            "label_robustness_gates": [
                "both_label_count_gates",
                "macro_label_auroc_ci_lower_above_0_5",
                "worst_label_auroc_at_least_0_5",
            ],
            "report_yes_no_separately": True,
        },
        "conformance": {
            "source": "frozen outcome-blind Ling V3.8.3 FEVER transport failures",
            "closed_before_boolq_calls": True,
            "canonicalizations": [
                "strip_and_casefold_answer_only_if_yes_or_no",
                "evidence_ids_to_cited_evidence_ids_only_if_unambiguous",
                "empty_citation_string_to_empty_list",
                "exact_allowed_citation_string_to_singleton_list",
                "finite_numeric_confidence_in_1_to_100_divided_by_100",
            ],
        },
        "claim_boundary": {
            "cross_model_family": True,
            "same_questions_and_interventions_as_qwen": True,
            "target_fit_prompt_tuning_or_threshold_calibration": False,
            "aggregate_pass_does_not_imply_label_robust_pass": True,
            "cross_dataset_transfer": False,
            "universal_transfer": False,
        },
    }


def freeze_protocol() -> dict[str, Any]:
    if not all(path.is_file() for path in (PREREGISTRATION, RUN_SCRIPT, SERVER_SCRIPT)):
        raise ValueError("V3.15 protocol files are incomplete")
    if not PROTOCOL_MANIFEST.exists() and any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot freeze V3.15 after Ling BoolQ calls")
    expected = build_protocol_manifest()
    if PROTOCOL_MANIFEST.exists():
        actual = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("frozen V3.15 protocol drifted")
    else:
        _write_json(PROTOCOL_MANIFEST, expected)
    return expected


def validate_protocol() -> None:
    if json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8")) != build_protocol_manifest():
        raise ValueError("frozen V3.15 protocol drifted")


class LingBoolQClient:
    def __init__(self, cache_dir: Path, timeout: float = 90.0) -> None:
        self.endpoint = TARGET_ENDPOINT
        self.model = TARGET_MODEL
        self.cache_dir = cache_dir
        self.timeout = timeout
        cache_dir.mkdir(parents=True, exist_ok=True)

    def call(self, messages: Sequence[Mapping[str, str]], *, seed: int) -> ChatResult:
        request_payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "seed": seed,
            "chat_template_kwargs": THINKING_KWARGS,
        }
        material = {"endpoint": self.endpoint, "request": request_payload}
        cache_key = sha256(_canonical_json(material).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            row = json.loads(cache_path.read_text(encoding="utf-8"))
            return ChatResult(
                content=str(row["content"]),
                model=str(row["model"]),
                usage=dict(row["usage"]),
                http_status=int(row["http_status"]),
                request_bytes=0,
                response_bytes=int(row["response_bytes"]),
                latency_seconds=0.0,
                cache_hit=True,
                cache_key=cache_key,
            )
        body = _canonical_json(request_payload).encode()
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib_request.Request(self.endpoint, data=body, headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError) as error:
            raise RuntimeError(f"Ling request failed: {error}") from error
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise ValueError("Ling response exceeded safety limit")
        payload = json.loads(response_body)
        content = payload["choices"][0]["message"]["content"]
        model = str(payload.get("model", ""))
        if model != TARGET_MODEL:
            raise ValueError(f"Ling endpoint returned model {model!r}")
        usage = dict(payload.get("usage", {}))
        result = ChatResult(
            content=str(content),
            model=model,
            usage=usage,
            http_status=status,
            request_bytes=len(body),
            response_bytes=len(response_body),
            latency_seconds=time.monotonic() - started,
            cache_hit=False,
            cache_key=cache_key,
        )
        _write_json(
            cache_path,
            {
                "content": result.content,
                "model": result.model,
                "usage": result.usage,
                "http_status": status,
                "response_bytes": result.response_bytes,
            },
        )
        return result


def parse_ling_decision(
    content: str, *, expected_agent_id: str, allowed_evidence_ids: Sequence[str]
) -> dict[str, Any]:
    payload = _extract_json_object(content)
    modes: list[str] = []
    answer = payload.get("answer")
    if isinstance(answer, str):
        canonical = answer.strip().casefold()
        if canonical in {"yes", "no"} and canonical != answer:
            payload["answer"] = canonical
            modes.append("answer_casefold")
    if "evidence_ids" in payload and "cited_evidence_ids" not in payload:
        payload["cited_evidence_ids"] = payload.pop("evidence_ids")
        modes.append("evidence_ids_alias")
    cited = payload.get("cited_evidence_ids")
    if cited == "":
        payload["cited_evidence_ids"] = []
        modes.append("empty_citation_string")
    elif isinstance(cited, str) and cited in allowed_evidence_ids:
        payload["cited_evidence_ids"] = [cited]
        modes.append("singleton_citation_string")
    confidence = payload.get("confidence")
    if (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 1 < float(confidence) <= 100
    ):
        payload["confidence"] = float(confidence) / 100
        modes.append("confidence_percent")
    decision = base.parse_forced_qa_decision(
        json.dumps(payload),
        expected_agent_id=expected_agent_id,
        allowed_evidence_ids=allowed_evidence_ids,
    )
    if modes:
        decision["parse_mode"] = "v3_15_" + "_and_".join(modes)
    return decision


def run_one_call(
    client: LingBoolQClient,
    composite: base.CompositeQuestion,
    view: base.EvidenceView,
    *,
    agent_index: int,
) -> dict[str, Any]:
    agent_id, persona = base.AGENT_PERSONAS[agent_index]
    attempts: list[dict[str, Any]] = []
    decision = None
    final_error = None
    for attempt_index in range(2):
        attempt: dict[str, Any] | None = None
        try:
            messages = base.build_messages(
                composite,
                view,
                agent_id=agent_id,
                persona=persona,
                repair=attempt_index > 0,
            )
            result = client.call(messages, seed=_agent_seed(agent_index))
            attempt = _attempt_payload(result)
            decision = parse_ling_decision(
                result.content,
                expected_agent_id=agent_id,
                allowed_evidence_ids=view.allowed_evidence_ids,
            )
        except (RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            final_error = f"{type(error).__name__}: {error}"
            if attempt is None:
                attempt = {
                    "cache_hit": False,
                    "cache_key": None,
                    "http_status": None,
                    "request_bytes": None,
                    "response_bytes": None,
                    "latency_seconds": None,
                    "usage": {},
                }
            attempt["parse_error"] = final_error
            attempts.append(attempt)
            continue
        attempts.append(attempt)
        final_error = None
        break
    return {
        "protocol_version": PROTOCOL_VERSION,
        "runtime_model": TARGET_MODEL,
        "runtime_endpoint": TARGET_ENDPOINT,
        "cqid": composite.cqid,
        "agent_id": agent_id,
        "agent_index": agent_index,
        "condition": view.condition,
        "partition": sorted(base.PARTITION_TABLE[agent_index]),
        "success": decision is not None,
        "first_pass_valid": decision is not None and len(attempts) == 1,
        "attempts": attempts,
        "decision": decision,
        "final_error": final_error,
    }


def _record_key(record: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(record["cqid"]), int(record["agent_index"]), str(record["condition"])


def _worker(
    shard_index: int,
    composites: Sequence[base.CompositeQuestion],
    substitutes: Mapping[str, Mapping[str, str]],
    *,
    output_dir: Path,
    cache_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    shard_dir = output_dir / "shards" / f"shard_{shard_index}"
    partial_path = shard_dir / "records.partial.jsonl"
    records = base._load_partial_records(partial_path) if resume else []
    done = {_record_key(row) for row in records}
    allowed = {
        (comp.cqid, agent, condition)
        for comp in composites
        for agent in range(base.N_AGENTS)
        for condition in CONDITIONS
    }
    if len(done) != len(records) or done - allowed:
        raise ValueError(f"V3.15 shard {shard_index} partial is invalid")
    client = LingBoolQClient(cache_dir / f"shard_{shard_index}")
    started = time.monotonic()
    for comp in composites:
        for agent_index in range(base.N_AGENTS):
            for condition in CONDITIONS:
                key = (comp.cqid, agent_index, condition)
                if key in done:
                    continue
                view = base.build_evidence_view(comp, agent_index, condition, substitutes)
                record = run_one_call(client, comp, view, agent_index=agent_index)
                records.append(record)
                done.add(key)
                _write_jsonl(partial_path, sorted(records, key=_record_key))
                if len(done) % 100 == 0 or len(done) == len(allowed):
                    print(
                        f"[v3.15 shard {shard_index} {len(done)}/{len(allowed)}] "
                        f"success={record['success']} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    return sorted(records, key=_record_key)


def execute(
    *, mode: str, output_dir: Path, cache_dir: Path, resume: bool = True
) -> list[dict[str, Any]]:
    validate_protocol()
    parent, composites = _validated_parent()
    full_shards = v12.assign_shards(composites)
    if mode == "smoke":
        shards = [[shard[0]] for shard in full_shards]
        expected = SMOKE_CALLS
    elif mode == "formal":
        _validate_smoke()
        shards = full_shards
        expected = EXPECTED_CALLS
    else:
        raise ValueError("mode must be smoke or formal")
    substitutes = parent["substitute_manifest"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                _worker,
                index,
                shards[index],
                substitutes,
                output_dir=output_dir,
                cache_dir=cache_dir,
                resume=resume,
            ): index
            for index in range(N_WORKERS)
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    records = sorted([row for shard in outputs for row in shard], key=_record_key)
    if len(records) != expected or len({_record_key(row) for row in records}) != expected:
        raise ValueError(f"V3.15 merge integrity failed: {len(records)}/{expected}")
    if any(
        row.get("protocol_version") != PROTOCOL_VERSION
        or row.get("runtime_model") != TARGET_MODEL
        or not row.get("success")
        for row in records
    ):
        raise RuntimeError("V3.15 contains a failed or foreign record")
    _write_jsonl(output_dir / "records.jsonl", records)
    return records


def endpoint_models() -> set[str]:
    url = TARGET_ENDPOINT.removesuffix("/chat/completions") + "/models"
    with urllib_request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read(MAX_RESPONSE_BYTES + 1))
    models = {str(row["id"]) for row in payload.get("data", [])}
    if TARGET_MODEL not in models:
        raise ValueError(f"expected {TARGET_MODEL!r}, got {sorted(models)}")
    return models


def smoke(workers: int = N_WORKERS) -> dict[str, Any]:
    del workers
    endpoint_models()
    records = execute(
        mode="smoke",
        output_dir=DEFAULT_ROOT / "smoke",
        cache_dir=DEFAULT_ROOT / "cache" / "smoke",
        resume=True,
    )
    result = {
        "rows": len(records),
        "successful": sum(bool(row["success"]) for row in records),
        "first_pass_valid": sum(bool(row["first_pass_valid"]) for row in records),
    }
    result["qualified"] = (
        result["rows"] == SMOKE_CALLS
        and result["successful"] == SMOKE_CALLS
        and result["first_pass_valid"] >= SMOKE_MIN_FIRST_PASS
    )
    _write_json(DEFAULT_ROOT / "smoke" / "qualification.json", result)
    if not result["qualified"]:
        raise RuntimeError(f"V3.15 smoke failed: {result}")
    return result


def _validate_smoke() -> None:
    path = DEFAULT_ROOT / "smoke" / "qualification.json"
    if not path.is_file() or not json.loads(path.read_text()).get("qualified"):
        raise ValueError("V3.15 formal execution requires qualified smoke")


def _preoutcome_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = base._group_by_question(records)
    rows: list[dict[str, Any]] = []
    for cqid, recs in sorted(grouped.items()):
        original = [row for row in recs if row["condition"] == "original"]
        if len(original) != base.N_AGENTS:
            raise ValueError(f"incomplete original bundle: {cqid}")
        answers = [str(row["decision"]["answer"]) for row in original]
        consensus, count = Counter(answers).most_common(1)[0]
        agreement = count / base.N_AGENTS
        signals = [base._agent_signal(group) for group in base._group_agents(recs)]
        if len(signals) != base.N_AGENTS or any(not row.get("complete") for row in signals):
            raise ValueError(f"incomplete intervention bundle: {cqid}")
        d_inert = sum(int(row["inert"]) for row in signals) / base.N_AGENTS
        flips = sum(sum(int(value) for value in row["flips"].values()) for row in signals)
        flip_inertia = 1.0 - flips / (base.N_AGENTS * 3)
        shared = 0
        seen: list[set[str]] = []
        for signal in signals:
            citations = set(signal["citations"]["original"])
            shared += int(any(citations & prior for prior in seen))
            seen.append(citations)
        frac_shared = shared / base.N_AGENTS
        risk = (
            RISK_WEIGHTS["D_inert"] * d_inert
            + RISK_WEIGHTS["flip_inertia"] * flip_inertia
            + RISK_WEIGHTS["frac_shared"] * frac_shared
        )
        rows.append(
            {
                "cqid": cqid,
                "consensus": consensus,
                "agreement": agreement,
                "D_inert": d_inert,
                "flip_inertia": flip_inertia,
                "frac_shared": frac_shared,
                "R_PI": risk,
            }
        )
    return rows


def freeze_preoutcome() -> dict[str, Any]:
    validate_protocol()
    records_path = DEFAULT_ROOT / "formal" / "records.jsonl"
    records = base._load_partial_records(records_path)
    if len(records) != EXPECTED_CALLS:
        raise ValueError("V3.15 formal records are incomplete")
    if any(FORBIDDEN_PREOUTCOME & set(row) for row in records):
        raise ValueError("V3.15 target records contain forbidden outcomes")
    rows = _preoutcome_rows(records)
    high = [row for row in rows if float(row["agreement"]) >= HIGH_CONSENSUS]
    keep_n = max(1, round(len(high) * ROUTER_COVERAGE))
    kept = [
        row["cqid"] for row in sorted(high, key=lambda row: (row["R_PI"], row["cqid"]))[:keep_n]
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest_sha256": file_sha256(PROTOCOL_MANIFEST),
        "formal_records_sha256": file_sha256(records_path),
        "outcomes_accessed": False,
        "forbidden_record_fields_absent": True,
        "risk_weights": RISK_WEIGHTS,
        "high_consensus_threshold": HIGH_CONSENSUS,
        "coverage": ROUTER_COVERAGE,
        "rows": rows,
        "high_consensus_cqids": [row["cqid"] for row in high],
        "retained_cqids": kept,
    }
    path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("V3.15 preoutcome snapshot drifted")
    else:
        _write_json(path, payload)
    return payload


def _metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["consensus_wrong"]) for row in rows])
    scores = np.asarray([float(row["R_PI"]) for row in rows])
    if len(set(labels.tolist())) < 2:
        return {"n": len(rows), "positives": int(labels.sum()), "auroc": None, "ci": None}
    observed = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, len(rows), len(rows))
        if len(set(labels[index].tolist())) == 2:
            samples.append(float(roc_auc_score(labels[index], scores[index])))
    return {
        "n": len(rows),
        "positives": int(labels.sum()),
        "negatives": int(len(rows) - labels.sum()),
        "auroc": observed,
        "auroc_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _router(rows: Sequence[Mapping[str, Any]], retained: set[str]) -> dict[str, Any]:
    baseline = float(np.mean([row["consensus_wrong"] for row in rows]))
    kept = [row for row in rows if row["cqid"] in retained]
    routed = float(np.mean([row["consensus_wrong"] for row in kept]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        boot = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        boot_kept = [row for row in boot if row["cqid"] in retained]
        if boot_kept:
            samples.append(
                float(np.mean([row["consensus_wrong"] for row in boot]))
                - float(np.mean([row["consensus_wrong"] for row in boot_kept]))
            )
    return {
        "n": len(rows),
        "retained": len(kept),
        "coverage": len(kept) / len(rows),
        "baseline_error": baseline,
        "retained_error": routed,
        "error_reduction": baseline - routed,
        "error_reduction_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def _label_robustness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = {label: [row for row in rows if row["label"] == label] for label in ("yes", "no")}
    metrics = {label: _metric(group) for label, group in groups.items()}
    points = [metrics[label]["auroc"] for label in ("yes", "no")]
    macro = float(np.mean(points)) if all(value is not None for value in points) else None
    worst = float(min(points)) if all(value is not None for value in points) else None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macro_samples, worst_samples = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        values = []
        for label in ("yes", "no"):
            group = groups[label]
            boot = [group[index] for index in rng.integers(0, len(group), len(group))]
            y = [row["consensus_wrong"] for row in boot]
            if len(set(y)) < 2:
                values = []
                break
            values.append(roc_auc_score(y, [row["R_PI"] for row in boot]))
        if values:
            macro_samples.append(float(np.mean(values)))
            worst_samples.append(float(min(values)))
    return {
        "by_label": metrics,
        "macro_label_auroc": macro,
        "macro_label_auroc_ci": np.quantile(macro_samples, [0.025, 0.975]).tolist(),
        "worst_label_auroc": worst,
        "worst_label_auroc_ci": np.quantile(worst_samples, [0.025, 0.975]).tolist(),
    }


def evaluate() -> dict[str, Any]:
    validate_protocol()
    snapshot_path = DEFAULT_ROOT / "evaluation" / "preoutcome_routes.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot != freeze_preoutcome():
        raise ValueError("V3.15 preoutcome snapshot drifted")
    parent, composites = _validated_parent()
    outcomes = {
        comp.cqid: {"label": comp.label, "gold_binary": comp.gold_binary} for comp in composites
    }
    rows = []
    for source in snapshot["rows"]:
        outcome = outcomes[source["cqid"]]
        row = {**source, **outcome}
        row["consensus_wrong"] = int((row["consensus"] == "yes") != bool(row["gold_binary"]))
        rows.append(row)
    high_ids = set(snapshot["high_consensus_cqids"])
    high = [row for row in rows if row["cqid"] in high_ids]
    primary = _metric(high)
    router = _router(high, set(snapshot["retained_cqids"]))
    robustness = _label_robustness(high)
    records = base._load_partial_records(DEFAULT_ROOT / "formal" / "records.jsonl")
    transport = {
        "records": len(records),
        "valid_rate": sum(bool(row["success"]) for row in records) / len(records),
        "first_pass_valid_rate": sum(bool(row["first_pass_valid"]) for row in records)
        / len(records),
        "parse_modes": dict(
            sorted(
                Counter(str(row["decision"].get("parse_mode", "strict")) for row in records).items()
            )
        ),
    }
    count_gate = len(high) >= 80 and primary["positives"] >= 10 and primary["negatives"] >= 10
    aggregate_gates = {
        "high_consensus_count_and_class_gate": count_gate,
        "aggregate_auroc_ci_lower_above_0_5": bool(
            primary.get("auroc_ci") and primary["auroc_ci"][0] > 0.5
        ),
        "risk80_error_reduction_ci_lower_above_zero": router["error_reduction_ci"][0] > 0,
        "final_valid_rate_is_one": transport["valid_rate"] == 1.0,
        "first_pass_valid_rate_at_least_0_95": transport["first_pass_valid_rate"] >= 0.95,
    }
    by_label = robustness["by_label"]
    label_gates = {
        "both_label_count_gates": all(
            metric["n"] >= 40 and metric["positives"] >= 10 and metric["negatives"] >= 10
            for metric in by_label.values()
        ),
        "macro_label_auroc_ci_lower_above_0_5": robustness["macro_label_auroc_ci"][0] > 0.5,
        "worst_label_auroc_at_least_0_5": robustness["worst_label_auroc"] >= 0.5,
    }
    aggregate_pass = all(aggregate_gates.values())
    label_pass = all(label_gates.values())
    v121.configure_v12_1()
    qwen_records = base._load_partial_records(PARENT_RECORDS)
    qwen_risk = {row["cqid"]: row for row in v11._risk_rows(qwen_records)}
    common = sorted(set(qwen_risk) & {row["cqid"] for row in rows})
    ling_by_id = {row["cqid"]: row for row in rows}
    correlation = float(
        spearmanr(
            [qwen_risk[key]["R_PI"] for key in common],
            [ling_by_id[key]["R_PI"] for key in common],
        ).statistic
    )
    verdict = (
        "PASS_CROSS_FAMILY_LABEL_ROBUST_DETECTION_V3_15"
        if aggregate_pass and label_pass
        else "PASS_CROSS_FAMILY_AGGREGATE_ONLY_DETECTION_V3_15"
        if aggregate_pass
        else "NO_VERIFIED_CROSS_FAMILY_DETECTION_REPLICATION_V3_15"
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "aggregate_replication_pass": aggregate_pass,
        "label_robustness_pass": label_pass,
        "primary": primary,
        "router_at_80": router,
        "label_robustness": robustness,
        "aggregate_gates": aggregate_gates,
        "label_gates": label_gates,
        "transport": transport,
        "cross_model_common_items": {
            "n": len(common),
            "r_pi_spearman_qwen_ling": correlation,
        },
        "hashes": {
            "protocol_manifest": file_sha256(PROTOCOL_MANIFEST),
            "formal_records": file_sha256(DEFAULT_ROOT / "formal" / "records.jsonl"),
            "preoutcome_routes": file_sha256(snapshot_path),
        },
        "claim_boundary": build_protocol_manifest()["claim_boundary"],
        "parent_substitute_modes": dict(
            Counter(
                str(row.get("generation_mode")) for row in parent["substitute_manifest"].values()
            )
        ),
    }
    _write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
    _write_report(summary)
    return summary


def _write_report(summary: Mapping[str, Any]) -> None:
    primary = summary["primary"]
    router = summary["router_at_80"]
    robust = summary["label_robustness"]
    lines = [
        "# Detection V3.15: Ling cross-family BoolQ replication",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        (
            f"- High-consensus AUROC: {primary['auroc']:.3f} "
            f"[{primary['auroc_ci'][0]:.3f}, {primary['auroc_ci'][1]:.3f}]"
        ),
        f"- Risk@80 error: {router['baseline_error']:.3f} -> {router['retained_error']:.3f}",
        (
            f"- Error reduction: {router['error_reduction']:.3f} "
            f"[{router['error_reduction_ci'][0]:.3f}, {router['error_reduction_ci'][1]:.3f}]"
        ),
        (
            f"- Macro-label AUROC: {robust['macro_label_auroc']:.3f} "
            f"[{robust['macro_label_auroc_ci'][0]:.3f}, {robust['macro_label_auroc_ci'][1]:.3f}]"
        ),
        f"- Worst-label AUROC: {robust['worst_label_auroc']:.3f}",
        "",
        "## Label subgroups",
        "",
    ]
    for label, metric in robust["by_label"].items():
        lines.append(
            f"- {label}: AUROC {metric['auroc']:.3f} "
            f"[{metric['auroc_ci'][0]:.3f}, {metric['auroc_ci'][1]:.3f}], "
            f"n={metric['n']}, wrong={metric['positives']}"
        )
    lines.extend(["", "## Frozen gates", ""])
    for group in (summary["aggregate_gates"], summary["label_gates"]):
        lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in group.items())
    lines.extend(
        [
            "",
            (
                "Aggregate and label-robust verdicts are separate. An aggregate-only "
                "pass does not support label-invariant or universal transfer."
            ),
            "",
        ]
    )
    path = DEFAULT_ROOT / "evaluation" / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-protocol")
    sub.add_parser("endpoint-check")
    sub.add_parser("smoke")
    formal = sub.add_parser("formal")
    formal.add_argument("--no-resume", action="store_true")
    sub.add_parser("freeze-preoutcome")
    sub.add_parser("evaluate")
    args = parser.parse_args(argv)
    if args.command == "freeze-protocol":
        print(json.dumps(freeze_protocol(), indent=2))
    elif args.command == "endpoint-check":
        validate_protocol()
        print(json.dumps({"models": sorted(endpoint_models())}, indent=2))
    elif args.command == "smoke":
        print(json.dumps(smoke(), indent=2))
    elif args.command == "formal":
        rows = execute(
            mode="formal",
            output_dir=DEFAULT_ROOT / "formal",
            cache_dir=DEFAULT_ROOT / "cache" / "formal",
            resume=not args.no_resume,
        )
        print(json.dumps({"records": len(rows)}, indent=2))
    elif args.command == "freeze-preoutcome":
        print(json.dumps(freeze_preoutcome(), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
