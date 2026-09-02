"""Schema-constrained zero-shot Qwen-to-Fin-R1 ELAR evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3_analysis as v383_analysis
from sp500_forecastability import recovery_v3_9 as v39
from sp500_forecastability import recovery_v3_9_2 as v392
from sp500_forecastability.pilot_llm_v1 import (
    MAX_COMPLETION_TOKENS,
    MAX_RESPONSE_BYTES,
    ChatResult,
    _canonical_json,
)

PROTOCOL_VERSION = "recovery-v3.10-qwen-to-finr1-guided-elar-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_10_finr1")
PREREGISTRATION = Path("docs/recovery_v3_10_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_10.sh")
V392_MANIFEST = Path("results/recovery_v3_9_2_finr1/protocol_manifest.json")
V392_ABORT = Path("results/recovery_v3_9_2_finr1/ABORTED.md")
V392_PARTIAL = Path(
    "results/recovery_v3_9_2_finr1/formal/actions/records.partial.jsonl"
)
V392_CACHE = Path("results/recovery_v3_9_2_finr1/cache")

ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "cited_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "confidence", "cited_evidence_ids"],
}

CERTIFICATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "atomic_checks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_span": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "unresolved"],
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["claim_span", "status", "evidence_ids"],
            },
        },
        "coverage_complete": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["atomic_checks", "coverage_complete", "confidence"],
}

LEDGER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "atom_index": {"type": "integer", "minimum": 0},
                    "evidence_id": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "semantic_verdict": {
                        "type": "string",
                        "enum": ["entailed", "contradicted", "insufficient"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "unsupported_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "atom_index",
                    "evidence_id",
                    "evidence_quote",
                    "semantic_verdict",
                    "confidence",
                    "unsupported_terms",
                ],
            },
        },
        "challenge": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["none", "found"]},
                "reason_code": {
                    "type": "string",
                    "enum": [
                        "none",
                        "unsupported_attribute",
                        "entity_mismatch",
                        "numeric_mismatch",
                        "negation_mismatch",
                        "relation_mismatch",
                        "insufficient_context",
                    ],
                },
                "claim_span": {"type": "string"},
                "evidence_id": {"type": "string"},
                "evidence_quote": {"type": "string"},
            },
            "required": [
                "status",
                "reason_code",
                "claim_span",
                "evidence_id",
                "evidence_quote",
            ],
        },
    },
    "required": ["entries", "challenge"],
}

SCHEMAS = {
    "action": ACTION_SCHEMA,
    "certificate": CERTIFICATE_SCHEMA,
    "ledger": LEDGER_SCHEMA,
}


def _artifact_kind(max_completion_tokens: int) -> str:
    mapping = {
        MAX_COMPLETION_TOKENS: "action",
        512: "certificate",
        768: "ledger",
    }
    if max_completion_tokens not in mapping:
        raise ValueError("completion-token limit does not identify an artifact kind")
    return mapping[max_completion_tokens]


def _response_format(kind: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"recovery_{kind}",
            "schema": SCHEMAS[kind],
            "strict": True,
        },
    }


class GuidedFinR1ChatClient(v39.FinR1ChatClient):
    """Fin-R1 client whose cache key binds artifact-specific JSON schemas."""

    def call(self, messages: Sequence[Mapping[str, str]], *, seed: int) -> ChatResult:
        kind = _artifact_kind(int(self.max_completion_tokens))
        request_payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": int(self.max_completion_tokens),
            "seed": seed,
            "chat_template_kwargs": v38.THINKING_KWARGS,
            "response_format": _response_format(kind),
        }
        cache_material = {"endpoint": self.endpoint, "request": request_payload}
        cache_key = sha256(_canonical_json(cache_material).encode()).hexdigest()
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
                "artifact_kind": kind,
                "response_format_sha256": sha256(
                    _canonical_json(_response_format(kind)).encode()
                ).hexdigest(),
            },
        )
        return result


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _audit_v392_abort() -> dict[str, Any]:
    records = base._load_jsonl(V392_PARTIAL)
    failures = [row for row in records if not bool(row.get("success"))]
    audit = {
        "rows": len(records),
        "complete_example_bundles": len({str(row["example_id"]) for row in records}),
        "terminal_failures": len(failures),
        "failed_attempts": sum(len(row["attempts"]) for row in failures),
        "terminal_error_counts": dict(
            sorted(Counter(str(row.get("final_error")) for row in failures).items())
        ),
    }
    expected = {
        "rows": 176,
        "complete_example_bundles": 22,
        "terminal_failures": 18,
        "failed_attempts": 36,
        "terminal_error_counts": {
            "ValueError: response must be exactly one JSON object": 18
        },
    }
    if audit != expected:
        raise ValueError("V3.9.2 outcome-blind abort facts drifted")
    cache_artifacts = []
    invalid_json = 0
    for row in failures:
        for attempt in row["attempts"]:
            cache_key = str(attempt["cache_key"])
            cache_path = V392_CACHE / f"{cache_key}.json"
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            try:
                json.loads(str(cached["content"]))
            except json.JSONDecodeError:
                invalid_json += 1
            cache_artifacts.append(
                {"cache_key": cache_key, "sha256": base._sha256_path(cache_path)}
            )
    if invalid_json != 36:
        raise ValueError("V3.9.2 failed responses are not all invalid JSON")
    return {
        "partial_action_records": str(V392_PARTIAL),
        "partial_action_records_sha256": base._sha256_path(V392_PARTIAL),
        "invalid_json_attempts": invalid_json,
        "failed_response_cache_artifacts": cache_artifacts,
        "outcomes_accessed": False,
        **audit,
    }


@contextmanager
def _configured_base() -> Iterator[None]:
    old = {
        "protocol_version": v392.PROTOCOL_VERSION,
        "default_root": v392.DEFAULT_ROOT,
        "preregistration": v392.PREREGISTRATION,
        "run_script": v392.RUN_SCRIPT,
        "validator": v392.validate_protocol_manifest,
        "implementation_path": v392._implementation_path,
        "client": v39.FinR1ChatClient,
    }
    v392.PROTOCOL_VERSION = PROTOCOL_VERSION
    v392.DEFAULT_ROOT = DEFAULT_ROOT
    v392.PREREGISTRATION = PREREGISTRATION
    v392.RUN_SCRIPT = RUN_SCRIPT
    v392.validate_protocol_manifest = validate_protocol_manifest
    v392._implementation_path = _implementation_path
    v39.FinR1ChatClient = GuidedFinR1ChatClient
    try:
        with v392._configured_base():
            yield
    finally:
        v392.PROTOCOL_VERSION = old["protocol_version"]
        v392.DEFAULT_ROOT = old["default_root"]
        v392.PREREGISTRATION = old["preregistration"]
        v392.RUN_SCRIPT = old["run_script"]
        v392.validate_protocol_manifest = old["validator"]
        v392._implementation_path = old["implementation_path"]
        v39.FinR1ChatClient = old["client"]


def _build_protocol_manifest() -> dict[str, Any]:
    manifest = v392._build_protocol_manifest()
    manifest["status"] = "frozen_before_any_v3_10_target_task_call"
    manifest["schema_constrained_interface"] = {
        "api_field": "response_format=json_schema",
        "dispatcher": "max_completion_tokens_identifies_artifact_kind",
        "schemas": SCHEMAS,
        "schemas_sha256": sha256(_canonical_json(SCHEMAS).encode()).hexdigest(),
        "semantic_validators_changed": False,
        "prompts_or_decoding_parameters_changed": False,
        "no_further_schema_or_parser_extension_after_freeze": True,
        "prior_protocol_manifest": str(V392_MANIFEST),
        "prior_protocol_manifest_sha256": base._sha256_path(V392_MANIFEST),
        "prior_abort_record": str(V392_ABORT),
        "prior_abort_record_sha256": base._sha256_path(V392_ABORT),
        "prior_outcome_blind_abort_audit": _audit_v392_abort(),
        "fresh_target_cache": True,
        "reruns_all_400_formal_examples": True,
        "uses_prior_target_responses_at_inference": False,
        "uses_target_correctness_or_action_outcomes": False,
    }
    return manifest


def build_protocol_manifest() -> dict[str, Any]:
    with _configured_base():
        return _build_protocol_manifest()


def validate_protocol_manifest(
    path: Path = DEFAULT_ROOT / "protocol_manifest.json",
) -> None:
    with _configured_base():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != _build_protocol_manifest():
            raise ValueError("V3.10 manifest or a frozen dependency drifted")


def prepare(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> dict[str, Any]:
    with _configured_base():
        task_records_exist = any(
            (DEFAULT_ROOT / "formal").glob("**/records*.jsonl")
        ) or any((DEFAULT_ROOT / "smoke").glob("**/records*.jsonl"))
        if not path.exists() and task_records_exist:
            raise ValueError("cannot freeze V3.10 after target task records exist")
        expected = _build_protocol_manifest()
        if path.exists():
            validate_protocol_manifest(path)
            return expected
        base._write_json(path, expected)
        return expected


def smoke(workers: int) -> None:
    with _configured_base():
        validate_protocol_manifest()
        v38.endpoint_model_ids()
        examples = v38._examples("development")[: v38.SMOKE_EXAMPLES]
        actions = v38.execute_bundles(
            examples,
            kind="actions",
            split="smoke",
            output_dir=DEFAULT_ROOT / "smoke" / "actions",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=workers,
        )
        certificates = v38.execute_bundles(
            examples,
            kind="certificates",
            split="smoke",
            output_dir=DEFAULT_ROOT / "smoke" / "certificates",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=workers,
        )
        v38.execute_ledgers(
            examples,
            actions,
            certificates,
            split="smoke",
            output_dir=DEFAULT_ROOT / "smoke" / "ledgers",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=workers,
        )


def evaluate() -> dict[str, Any]:
    with _configured_base(), v383_analysis._patched_diagnostic():
        summary = v38.evaluate(DEFAULT_ROOT / "evaluation")
        summary["verdict"] = (
            "PASS_SCHEMA_CONSTRAINED_QWEN_TO_FINR1_ELAR_V3_10"
            if summary["passes"]
            else "NO_VERIFIED_QWEN_TO_FINR1_ELAR_TRANSFER_V3_10"
        )
        summary["schema_constrained_interface"] = _build_protocol_manifest()[
            "schema_constrained_interface"
        ]
        summary["replication_context"] = {
            "ling_outcomes_used_for_finr1_fit_or_selection": False,
            "different_organization_checkpoint": True,
            "broad_qwen_lineage_shared": True,
        }
        base._write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
        report_path = DEFAULT_ROOT / "evaluation" / "report.md"
        v38._write_report(summary, report_path)
        report = report_path.read_text(encoding="utf-8").replace(
            "# Recovery V3.8 result: zero-shot Qwen-to-Ling ELAR",
            "# Recovery V3.10 result: schema-constrained Qwen-to-Fin-R1 ELAR",
            1,
        )
        report += (
            "## Structured-output and claim boundary\n\n"
            "All target artifacts used frozen vLLM JSON schemas and then the original "
            "semantic validators. Structured decoding is an interface control, not the "
            "claimed routing contribution. Fin-R1 shares the broad Qwen lineage with "
            "the source model.\n"
        )
        report_path.write_text(report, encoding="utf-8")
        return summary


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
        with _configured_base():
            print(json.dumps(sorted(v38.endpoint_model_ids())))
        return 0
    if args.command == "smoke":
        smoke(args.workers)
        return 0
    validate_protocol_manifest()
    with _configured_base():
        examples = v38._examples("formal")
        if args.command in {"formal-actions", "formal-certificates"}:
            kind = args.command.removeprefix("formal-")
            v38.execute_bundles(
                examples,
                kind=kind,
                split="formal",
                output_dir=DEFAULT_ROOT / "formal" / kind,
                cache_dir=DEFAULT_ROOT / "cache",
                workers=args.workers,
            )
            return 0
        if args.command == "formal-ledgers":
            v38.execute_ledgers(
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
