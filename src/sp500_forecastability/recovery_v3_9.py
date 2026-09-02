"""Zero-shot transfer of frozen Qwen ELAR to the SUFE Fin-R1 checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3 as v383
from sp500_forecastability import recovery_v3_8_3_analysis as v383_analysis

PROTOCOL_VERSION = "recovery-v3.9-qwen-to-finr1-elar-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_9_finr1")
PREREGISTRATION = Path("docs/recovery_v3_9_preregistration.md")
SERVER_SCRIPT = Path("scripts/start_finr1_v3_9.sh")
RUN_SCRIPT = Path("scripts/run_recovery_v3_9.sh")
TARGET_MODEL = "Fin-R1"
TARGET_ENDPOINT = "http://127.0.0.1:31520/v1/chat/completions"
TARGET_MODEL_DIR = Path("/storage/lianjh/modelzoos/SUFE-AIFLM-Lab/Fin-R1")
TARGET_CONFIG = TARGET_MODEL_DIR / "config.json"
TARGET_WEIGHT_INDEX = TARGET_MODEL_DIR / "model.safetensors.index.json"
TARGET_SMALL_ARTIFACTS = (
    TARGET_CONFIG,
    TARGET_MODEL_DIR / "generation_config.json",
    TARGET_MODEL_DIR / "tokenizer_config.json",
    TARGET_MODEL_DIR / "special_tokens_map.json",
    TARGET_MODEL_DIR / "added_tokens.json",
    TARGET_MODEL_DIR / "merges.txt",
    TARGET_MODEL_DIR / "vocab.json",
    TARGET_MODEL_DIR / "tokenizer.json",
    TARGET_WEIGHT_INDEX,
)
LING_PREREGISTRATION = Path("docs/recovery_v3_8_preregistration.md")
LING_SUMMARY = Path("results/recovery_v3_8_3_ling/evaluation/summary.json")

_ORIGINAL_BUILD_MANIFEST = v38.build_protocol_manifest


class FinR1ChatClient(v38.CrossModelChatClient):
    """Bind the generic V3.8 target client to the frozen Fin-R1 endpoint."""

    def __init__(
        self,
        cache_dir: Path,
        endpoint: str = TARGET_ENDPOINT,
        model: str = TARGET_MODEL,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            model=model,
            cache_dir=cache_dir,
            timeout=timeout,
        )


def _implementation_path() -> Path:
    return Path(__file__).resolve()


@contextmanager
def _configured_base() -> Iterator[None]:
    old = {
        "protocol_version": v38.PROTOCOL_VERSION,
        "default_root": v38.DEFAULT_ROOT,
        "preregistration": v38.PREREGISTRATION,
        "server_script": v38.SERVER_SCRIPT,
        "target_model": v38.TARGET_MODEL,
        "target_endpoint": v38.TARGET_ENDPOINT,
        "target_model_dir": v38.TARGET_MODEL_DIR,
        "target_config": v38.TARGET_CONFIG,
        "target_weight_index": v38.TARGET_WEIGHT_INDEX,
        "target_small_artifacts": v38.TARGET_SMALL_ARTIFACTS,
        "client": v38.CrossModelChatClient,
        "parser": v362.parse_action_decision,
        "validator": v38.validate_protocol_manifest,
        "attempt_stats": v38._attempt_stats,
        "implementation_path": v38._implementation_path,
    }
    v38.PROTOCOL_VERSION = PROTOCOL_VERSION
    v38.DEFAULT_ROOT = DEFAULT_ROOT
    v38.PREREGISTRATION = PREREGISTRATION
    v38.SERVER_SCRIPT = SERVER_SCRIPT
    v38.TARGET_MODEL = TARGET_MODEL
    v38.TARGET_ENDPOINT = TARGET_ENDPOINT
    v38.TARGET_MODEL_DIR = TARGET_MODEL_DIR
    v38.TARGET_CONFIG = TARGET_CONFIG
    v38.TARGET_WEIGHT_INDEX = TARGET_WEIGHT_INDEX
    v38.TARGET_SMALL_ARTIFACTS = TARGET_SMALL_ARTIFACTS
    v38.CrossModelChatClient = FinR1ChatClient
    v362.parse_action_decision = v383.parse_action_decision
    v38.validate_protocol_manifest = validate_protocol_manifest
    v38._attempt_stats = v383._attempt_stats
    v38._implementation_path = _implementation_path
    try:
        yield
    finally:
        v38.PROTOCOL_VERSION = old["protocol_version"]
        v38.DEFAULT_ROOT = old["default_root"]
        v38.PREREGISTRATION = old["preregistration"]
        v38.SERVER_SCRIPT = old["server_script"]
        v38.TARGET_MODEL = old["target_model"]
        v38.TARGET_ENDPOINT = old["target_endpoint"]
        v38.TARGET_MODEL_DIR = old["target_model_dir"]
        v38.TARGET_CONFIG = old["target_config"]
        v38.TARGET_WEIGHT_INDEX = old["target_weight_index"]
        v38.TARGET_SMALL_ARTIFACTS = old["target_small_artifacts"]
        v38.CrossModelChatClient = old["client"]
        v362.parse_action_decision = old["parser"]
        v38.validate_protocol_manifest = old["validator"]
        v38._attempt_stats = old["attempt_stats"]
        v38._implementation_path = old["implementation_path"]


def _build_protocol_manifest() -> dict[str, Any]:
    manifest = _ORIGINAL_BUILD_MANIFEST()
    manifest["status"] = "frozen_before_any_finr1_formal_task_call"
    manifest["runner_script_path"] = str(RUN_SCRIPT)
    manifest["runner_script_sha256"] = base._sha256_path(RUN_SCRIPT)
    manifest["transport_conformance"] = {
        "source": "Recovery V3.8.3 final closed conformance",
        "implementation_sha256": base._sha256_path(v383._implementation_path()),
        "no_further_parser_extension_after_freeze": True,
        "canonicalizations": [
            "strip_and_casefold_answer_only_if_yes_or_no",
            "evidence_ids_to_cited_evidence_ids_only_if_unambiguous",
            "empty_citation_string_to_empty_list",
            "exact_allowed_citation_string_to_singleton_list",
            "finite_numeric_confidence_in_1_to_100_divided_by_100",
        ],
    }
    manifest["registered_continuation"] = {
        "ling_preregistration_path": str(LING_PREREGISTRATION),
        "ling_preregistration_sha256": base._sha256_path(LING_PREREGISTRATION),
        "ling_summary_path": str(LING_SUMMARY),
        "ling_summary_sha256": base._sha256_path(LING_SUMMARY),
        "ling_outcomes_used_for_finr1_fit_or_selection": False,
    }
    manifest["target"]["architecture"] = "Qwen2ForCausalLM"
    manifest["target"]["checkpoint_dtype"] = "bfloat16"
    manifest["target"]["runtime"]["quantization"] = "none"
    manifest["target"]["runtime"]["compressed_tensors"] = "not_used"
    manifest["target"]["runtime"]["torch"] = "2.13.0+cu130"
    manifest["target"]["runtime"]["transformers"] = "5.16.1"
    manifest["evaluation_diagnostic_fix"] = {
        "scope": "atomic_proof_only_none_ledger_only",
        "primary_elar_changed": False,
        "implementation_sha256": base._sha256_path(
            v383_analysis._implementation_path()
        ),
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
            raise ValueError("V3.9 manifest or a frozen dependency drifted")


def prepare(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> dict[str, Any]:
    with _configured_base():
        formal_root = DEFAULT_ROOT / "formal"
        smoke_root = DEFAULT_ROOT / "smoke"
        task_records_exist = any(formal_root.glob("**/records*.jsonl")) or any(
            smoke_root.glob("**/records*.jsonl")
        )
        if not path.exists() and task_records_exist:
            raise ValueError("cannot freeze V3.9 after target task records exist")
        expected = _build_protocol_manifest()
        if path.exists():
            if task_records_exist:
                validate_protocol_manifest(path)
            else:
                base._write_json(path, expected)
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
            "PASS_ZERO_SHOT_QWEN_TO_FINR1_ELAR_V3_9"
            if summary["passes"]
            else "NO_VERIFIED_QWEN_TO_FINR1_ELAR_TRANSFER_V3_9"
        )
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
            "# Recovery V3.9 result: zero-shot Qwen-to-Fin-R1 ELAR",
            1,
        )
        parse_modes = summary["transport_and_schema"]["actions"].get(
            "decision_parse_modes", {}
        )
        report += "\n".join(
            [
                "## Replication boundary",
                "",
                (
                    "Fin-R1 is a separately trained SUFE checkpoint but shares the "
                    "broad Qwen lineage with the source model."
                ),
                f"Accepted action parse modes: `{json.dumps(parse_modes, sort_keys=True)}`.",
                "Ling outcomes were not used for Fin-R1 fitting or policy selection.",
                "",
            ]
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
