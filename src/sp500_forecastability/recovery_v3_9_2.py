"""Uniform JSON-envelope conformance for frozen Qwen-to-Fin-R1 ELAR."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_7 as frozen
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3 as v383
from sp500_forecastability import recovery_v3_8_3_analysis as v383_analysis
from sp500_forecastability import recovery_v3_9_1 as v391

PROTOCOL_VERSION = "recovery-v3.9.2-qwen-to-finr1-elar-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_9_2_finr1")
PREREGISTRATION = Path("docs/recovery_v3_9_2_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_9_2.sh")
V391_MANIFEST = Path("results/recovery_v3_9_1_finr1/protocol_manifest.json")
V391_ABORT = Path("results/recovery_v3_9_1_finr1/ABORTED.md")
V391_ACTIONS = Path("results/recovery_v3_9_1_finr1/smoke/actions/records.jsonl")
V391_CERTIFICATES = Path(
    "results/recovery_v3_9_1_finr1/smoke/certificates/records.jsonl"
)
V391_CACHE = Path("results/recovery_v3_9_1_finr1/cache")

_ORIGINAL_CERTIFICATE_PARSER = v362.parse_certificate
_ORIGINAL_LEDGER_PARSER = frozen.parse_ledger
_ORIGINAL_ATTEMPT_STATS = v383._attempt_stats


def _leading_mapping_json(content: str, original_error: Exception) -> str:
    text = content.strip()
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        raise original_error
    trailing = text[end:].strip()
    if (
        not isinstance(payload, Mapping)
        or not trailing
        or "{" in trailing
        or "}" in trailing
    ):
        raise original_error
    return json.dumps(payload)


def parse_certificate(
    content: str,
    claim: str,
    allowed_ids: Sequence[str],
    allowed_new_ids: Sequence[str],
) -> dict[str, Any]:
    try:
        return _ORIGINAL_CERTIFICATE_PARSER(
            content, claim, allowed_ids, allowed_new_ids
        )
    except (TypeError, ValueError) as original_error:
        canonical = _leading_mapping_json(content, original_error)
        certificate = _ORIGINAL_CERTIFICATE_PARSER(
            canonical, claim, allowed_ids, allowed_new_ids
        )
        certificate["transport_parse_mode"] = "leading_json_with_trailing_text"
        return certificate


def parse_ledger(
    content: str,
    example: Mapping[str, Any],
    action: str,
    certificate: Mapping[str, Any],
    consensus: str,
) -> dict[str, Any]:
    try:
        return _ORIGINAL_LEDGER_PARSER(
            content, example, action, certificate, consensus
        )
    except (TypeError, ValueError) as original_error:
        canonical = _leading_mapping_json(content, original_error)
        ledger = _ORIGINAL_LEDGER_PARSER(
            canonical, example, action, certificate, consensus
        )
        ledger["transport_parse_mode"] = "leading_json_with_trailing_text"
        return ledger


def _attempt_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats = _ORIGINAL_ATTEMPT_STATS(records)
    modes = Counter()
    for row in records:
        artifact = row.get("certificate") or row.get("ledger")
        if row.get("success") and isinstance(artifact, Mapping):
            modes[str(artifact.get("transport_parse_mode", "strict"))] += 1
    if modes:
        stats["artifact_parse_modes"] = dict(sorted(modes.items()))
    return stats


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _audit_v391_smoke_abort() -> dict[str, Any]:
    actions = base._load_jsonl(V391_ACTIONS)
    certificates = base._load_jsonl(V391_CERTIFICATES)
    if len(actions) != 16 or sum(bool(row.get("success")) for row in actions) != 16:
        raise ValueError("V3.9.1 smoke action audit drifted")
    failed = [row for row in certificates if not bool(row.get("success"))]
    if len(certificates) != 4 or len(failed) != 2:
        raise ValueError("V3.9.1 smoke certificate audit drifted")
    selection = json.loads(v38.SOURCE_SELECTION.read_text(encoding="utf-8"))
    examples = {str(row["example_id"]): row for row in selection["examples"]}
    artifacts = []
    replayed = 0
    for row in failed:
        example = examples[str(row["example_id"])]
        index = int(str(row["action"])[-1])
        allowed_ids = [
            str(item["evidence_id"])
            for item in (
                *example["anchor"]["evidence"],
                *example["candidates"][index]["evidence"],
            )
        ]
        allowed_new_ids = [
            str(item["evidence_id"])
            for item in example["candidates"][index]["evidence"]
        ]
        for attempt in row["attempts"]:
            cache_key = str(attempt["cache_key"])
            cache_path = V391_CACHE / f"{cache_key}.json"
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            certificate = parse_certificate(
                str(cached["content"]),
                str(example["claim"]),
                allowed_ids,
                allowed_new_ids,
            )
            if certificate.get("transport_parse_mode") != (
                "leading_json_with_trailing_text"
            ):
                raise ValueError("V3.9.1 failed certificate is not envelope-only")
            replayed += 1
            artifacts.append(
                {
                    "cache_key": cache_key,
                    "sha256": base._sha256_path(cache_path),
                }
            )
    if replayed != 4:
        raise ValueError("V3.9.1 failed certificate replay count drifted")
    action_modes = Counter(
        str(row["decision"].get("parse_mode", "strict")) for row in actions
    )
    return {
        "action_records_sha256": base._sha256_path(V391_ACTIONS),
        "certificate_records_sha256": base._sha256_path(V391_CERTIFICATES),
        "actions": len(actions),
        "successful_actions": 16,
        "action_parse_modes": dict(sorted(action_modes.items())),
        "certificates": len(certificates),
        "successful_certificates": 2,
        "failed_certificates": 2,
        "failed_attempts_replayed_with_envelope_only": replayed,
        "failed_response_cache_artifacts": artifacts,
        "formal_target_calls": 0,
        "outcomes_accessed": False,
    }


@contextmanager
def _configured_base() -> Iterator[None]:
    old = {
        "protocol_version": v391.PROTOCOL_VERSION,
        "default_root": v391.DEFAULT_ROOT,
        "preregistration": v391.PREREGISTRATION,
        "run_script": v391.RUN_SCRIPT,
        "validator": v391.validate_protocol_manifest,
        "implementation_path": v391._implementation_path,
        "certificate_parser": v362.parse_certificate,
        "ledger_parser": frozen.parse_ledger,
        "attempt_stats": v383._attempt_stats,
    }
    v391.PROTOCOL_VERSION = PROTOCOL_VERSION
    v391.DEFAULT_ROOT = DEFAULT_ROOT
    v391.PREREGISTRATION = PREREGISTRATION
    v391.RUN_SCRIPT = RUN_SCRIPT
    v391.validate_protocol_manifest = validate_protocol_manifest
    v391._implementation_path = _implementation_path
    v362.parse_certificate = parse_certificate
    frozen.parse_ledger = parse_ledger
    v383._attempt_stats = _attempt_stats
    try:
        with v391._configured_base():
            yield
    finally:
        v391.PROTOCOL_VERSION = old["protocol_version"]
        v391.DEFAULT_ROOT = old["default_root"]
        v391.PREREGISTRATION = old["preregistration"]
        v391.RUN_SCRIPT = old["run_script"]
        v391.validate_protocol_manifest = old["validator"]
        v391._implementation_path = old["implementation_path"]
        v362.parse_certificate = old["certificate_parser"]
        frozen.parse_ledger = old["ledger_parser"]
        v383._attempt_stats = old["attempt_stats"]


def _build_protocol_manifest() -> dict[str, Any]:
    manifest = v391._build_protocol_manifest()
    manifest["status"] = "frozen_before_any_v3_9_2_target_task_call"
    manifest["uniform_envelope_amendment"] = {
        "scope": "same_leading_json_envelope_for_actions_certificates_ledgers",
        "certificate_or_ledger_semantic_validator_changed": False,
        "no_further_transport_extension_after_freeze": True,
        "prior_protocol_manifest": str(V391_MANIFEST),
        "prior_protocol_manifest_sha256": base._sha256_path(V391_MANIFEST),
        "prior_abort_record": str(V391_ABORT),
        "prior_abort_record_sha256": base._sha256_path(V391_ABORT),
        "prior_outcome_blind_smoke_audit": _audit_v391_smoke_abort(),
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
            raise ValueError("V3.9.2 manifest or a frozen dependency drifted")


def prepare(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> dict[str, Any]:
    with _configured_base():
        task_records_exist = any(
            (DEFAULT_ROOT / "formal").glob("**/records*.jsonl")
        ) or any((DEFAULT_ROOT / "smoke").glob("**/records*.jsonl"))
        if not path.exists() and task_records_exist:
            raise ValueError("cannot freeze V3.9.2 after target task records exist")
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
            "PASS_ZERO_SHOT_QWEN_TO_FINR1_ELAR_V3_9_2"
            if summary["passes"]
            else "NO_VERIFIED_QWEN_TO_FINR1_ELAR_TRANSFER_V3_9_2"
        )
        summary["uniform_envelope_amendment"] = _build_protocol_manifest()[
            "uniform_envelope_amendment"
        ]
        summary["replication_context"] = {
            "registered_before_ling_result": True,
            "ling_outcomes_used_for_finr1_fit_or_selection": False,
            "different_organization_checkpoint": True,
            "broad_qwen_lineage_shared": True,
        }
        base._write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
        report_path = DEFAULT_ROOT / "evaluation" / "report.md"
        v38._write_report(summary, report_path)
        report = report_path.read_text(encoding="utf-8").replace(
            "# Recovery V3.8 result: zero-shot Qwen-to-Ling ELAR",
            "# Recovery V3.9.2 result: zero-shot Qwen-to-Fin-R1 ELAR",
            1,
        )
        report += (
            "## Uniform JSON envelope and claim boundary\n\n"
            "The development-qualified leading-JSON envelope applies to actions, "
            "certificates, and ledgers without changing their semantic validators.\n"
            "Fin-R1 is a separately trained SUFE checkpoint that shares the broad "
            "Qwen lineage with the source.\n"
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
