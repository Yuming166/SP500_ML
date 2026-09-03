"""V3.15.2 empty-packet citation conformance after a formal transport abort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sp500_forecastability import detection_v3_15 as base
from sp500_forecastability import detection_v3_15_1 as parent

PROTOCOL_VERSION = "detection-v3.15.2-ling-boolq-v12.1-2026-09-03"
DEFAULT_ROOT = Path("results/detection_v3_15_2_ling")
PROTOCOL_MANIFEST = DEFAULT_ROOT / "protocol_manifest.json"
PREREGISTRATION = Path("docs/detection_v3_15_2_preregistration.md")
RUN_SCRIPT = Path("scripts/run_detection_v3_15_2.sh")
PARENT_PROTOCOL = parent.PROTOCOL_MANIFEST
PARENT_FORMAL_ROOT = parent.DEFAULT_ROOT / "formal" / "shards"

_ORIGINAL_BUILD = base.build_protocol_manifest
_ORIGINAL_VALIDATE = base.validate_protocol
_ORIGINAL_PARSE = parent.parse_ling_decision
_ORIGINAL_IMPLEMENTATION_PATH = base._implementation_path


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def parse_ling_decision(
    content: str, *, expected_agent_id: str, allowed_evidence_ids: Sequence[str]
) -> dict[str, Any]:
    payload = base._extract_json_object(content)
    cleared = not allowed_evidence_ids and bool(
        payload.get("cited_evidence_ids") or payload.get("evidence_ids")
    )
    if not allowed_evidence_ids:
        payload.pop("evidence_ids", None)
        payload["cited_evidence_ids"] = []
    decision = _ORIGINAL_PARSE(
        json.dumps(payload),
        expected_agent_id=expected_agent_id,
        allowed_evidence_ids=allowed_evidence_ids,
    )
    if cleared:
        previous = str(decision.get("parse_mode", "strict"))
        decision["parse_mode"] = (
            "v3_15_2_empty_packet_citations_cleared"
            if previous == "strict"
            else previous + "_and_empty_packet_citations_cleared"
        )
    return decision


def _formal_abort_audit() -> dict[str, Any]:
    paths = sorted(PARENT_FORMAL_ROOT.glob("shard_*/records.partial.jsonl"))
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    terminal = Counter(str(row.get("final_error")) for row in rows if not row.get("success"))
    audit = {
        "rows": len(rows),
        "unique_tuples": len(
            {(row["cqid"], int(row["agent_index"]), row["condition"]) for row in rows}
        ),
        "successful": sum(bool(row.get("success")) for row in rows),
        "first_pass_valid": sum(bool(row.get("first_pass_valid")) for row in rows),
        "terminal_error_counts": dict(sorted(terminal.items())),
        "all_terminal_failures_are_empty_packet_remove": all(
            row.get("condition") == "remove" for row in rows if not row.get("success")
        ),
        "record_fields_contain_no_outcomes": all(
            not (base.FORBIDDEN_PREOUTCOME & set(row)) for row in rows
        ),
        "partial_sha256": {str(path): base.file_sha256(path) for path in paths},
    }
    expected = {
        "rows": 7160,
        "unique_tuples": 7160,
        "successful": 7140,
        "first_pass_valid": 7133,
        "terminal_error_counts": {
            "ValueError: citations outside packet: ['evidence_1']": 13,
            "ValueError: citations outside packet: ['evidence_packet']": 7,
        },
        "all_terminal_failures_are_empty_packet_remove": True,
        "record_fields_contain_no_outcomes": True,
    }
    if {key: audit[key] for key in expected} != expected:
        raise ValueError("V3.15.1 outcome-blind formal transport facts drifted")
    return audit


@contextmanager
def _configured(*, patch_validate: bool = True) -> Iterator[None]:
    old = {
        "protocol": base.PROTOCOL_VERSION,
        "root": base.DEFAULT_ROOT,
        "manifest": base.PROTOCOL_MANIFEST,
        "prereg": base.PREREGISTRATION,
        "run": base.RUN_SCRIPT,
        "parse": base.parse_ling_decision,
        "validate": base.validate_protocol,
        "implementation": base._implementation_path,
    }
    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.DEFAULT_ROOT = DEFAULT_ROOT
    base.PROTOCOL_MANIFEST = PROTOCOL_MANIFEST
    base.PREREGISTRATION = PREREGISTRATION
    base.RUN_SCRIPT = RUN_SCRIPT
    base.parse_ling_decision = parse_ling_decision
    base._implementation_path = _implementation_path
    if patch_validate:
        base.validate_protocol = validate_protocol
    try:
        yield
    finally:
        base.PROTOCOL_VERSION = old["protocol"]
        base.DEFAULT_ROOT = old["root"]
        base.PROTOCOL_MANIFEST = old["manifest"]
        base.PREREGISTRATION = old["prereg"]
        base.RUN_SCRIPT = old["run"]
        base.parse_ling_decision = old["parse"]
        base.validate_protocol = old["validate"]
        base._implementation_path = old["implementation"]


def build_protocol_manifest() -> dict[str, Any]:
    with _configured(patch_validate=False):
        manifest = _ORIGINAL_BUILD()
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["status"] = "frozen_after_v3_15_1_transport_abort_before_v3_15_2_calls"
    manifest["amendment"] = {
        "scope": "clear_citations_only_when_environment_packet_is_empty",
        "answer_and_confidence_unchanged": True,
        "nonempty_packet_citations_still_strict": True,
        "no_prompt_sample_score_metric_or_gate_change": True,
        "fresh_smoke_and_full_formal_cache": True,
        "parent_protocol": str(PARENT_PROTOCOL),
        "parent_protocol_sha256": base.file_sha256(PARENT_PROTOCOL),
        "parent_outcome_blind_formal_transport_audit": _formal_abort_audit(),
    }
    return manifest


def freeze_protocol() -> dict[str, Any]:
    if not all(path.is_file() for path in (PREREGISTRATION, RUN_SCRIPT, PARENT_PROTOCOL)):
        raise ValueError("V3.15.2 protocol files are incomplete")
    if not PROTOCOL_MANIFEST.exists() and any(DEFAULT_ROOT.glob("**/records*.jsonl")):
        raise ValueError("cannot freeze V3.15.2 after target calls")
    expected = build_protocol_manifest()
    if PROTOCOL_MANIFEST.exists():
        if json.loads(PROTOCOL_MANIFEST.read_text()) != expected:
            raise ValueError("V3.15.2 protocol drifted")
    else:
        base._write_json(PROTOCOL_MANIFEST, expected)
    return expected


def validate_protocol() -> None:
    if json.loads(PROTOCOL_MANIFEST.read_text()) != build_protocol_manifest():
        raise ValueError("V3.15.2 protocol drifted")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "freeze-protocol",
        "endpoint-check",
        "smoke",
        "formal",
        "freeze-preoutcome",
        "evaluate",
    ):
        sub.add_parser(command)
    args = parser.parse_args(argv)
    if args.command == "freeze-protocol":
        print(json.dumps(freeze_protocol(), indent=2))
        return 0
    with _configured():
        if args.command == "endpoint-check":
            validate_protocol()
            print(json.dumps({"models": sorted(base.endpoint_models())}, indent=2))
        elif args.command == "smoke":
            print(json.dumps(base.smoke(), indent=2))
        elif args.command == "formal":
            rows = base.execute(
                mode="formal",
                output_dir=DEFAULT_ROOT / "formal",
                cache_dir=DEFAULT_ROOT / "cache" / "formal",
                resume=True,
            )
            print(json.dumps({"records": len(rows)}, indent=2))
        elif args.command == "freeze-preoutcome":
            print(json.dumps(base.freeze_preoutcome(), indent=2))
        elif args.command == "evaluate":
            print(json.dumps(base.evaluate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
