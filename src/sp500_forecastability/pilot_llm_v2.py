"""Question-aligned Pilot-LLM V2 runner.

V2 preserves the V1 execution and evaluation machinery but never exposes the
generated StrategyQA ``claim``.  It answers the original yes/no question whose
``valid`` label is stored by the dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from sp500_forecastability.pilot_llm_v1 import (
    AGENT_PERSONAS,
    CONDITIONS,
    DEFAULT_DATASET,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    EXPECTED_DATASET_SHA256,
    FORMAL_EXAMPLES,
    FORMAL_PER_LABEL,
    REPAIR_SUFFIX,
    CachedChatClient,
    EvidenceView,
    StrategyQAExample,
    _agent_seed,
    _attempt_payload,
    _write_json,
    _write_jsonl,
    evidence_view,
    load_strategyqa,
    render_report,
    summarize_records,
)

PROTOCOL_VERSION = "pilot-llm-v2-2026-08-31"
DEFAULT_ROOT = Path("results/pilot_llm_v2")


def _selection_digest(qid: str) -> str:
    return sha256(f"{PROTOCOL_VERSION}\n{qid}".encode()).hexdigest()


def select_frozen_examples(
    examples: Sequence[StrategyQAExample], *, per_label: int = FORMAL_PER_LABEL
) -> list[StrategyQAExample]:
    """Select V2's independent deterministic label-balanced sample."""

    selected: list[StrategyQAExample] = []
    for label in (False, True):
        stratum = sorted(
            (example for example in examples if example.label is label),
            key=lambda example: (_selection_digest(example.qid), example.qid),
        )
        if len(stratum) < per_label:
            raise ValueError(f"not enough examples for label={label}")
        selected.extend(stratum[:per_label])
    return sorted(selected, key=lambda example: (_selection_digest(example.qid), example.qid))


def build_manifest(dataset_path: Path) -> dict[str, object]:
    examples = select_frozen_examples(load_strategyqa(dataset_path))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "selection": {
            "per_label": FORMAL_PER_LABEL,
            "total": FORMAL_EXAMPLES,
            "salt": PROTOCOL_VERSION,
        },
        "task_field": "question",
        "excluded_task_field": "claim",
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "conditions": list(CONDITIONS),
        "agents": [agent_id for agent_id, _ in AGENT_PERSONAS],
        "examples": [
            {
                "qid": example.qid,
                "label": example.label,
                "fact_count": len(example.facts),
                "selection_sha256": _selection_digest(example.qid),
            }
            for example in examples
        ],
    }


def validate_manifest(
    manifest: Mapping[str, object], dataset_path: Path
) -> list[StrategyQAExample]:
    """Recompute the V2 question-aligned sample and reject drift."""

    expected_fields = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "task_field": "question",
        "excluded_task_field": "claim",
        "model": DEFAULT_MODEL,
        "endpoint": DEFAULT_ENDPOINT,
        "conditions": list(CONDITIONS),
        "agents": [agent_id for agent_id, _ in AGENT_PERSONAS],
    }
    for field, expected_value in expected_fields.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"manifest {field} does not match Pilot-LLM V2")

    expected = select_frozen_examples(load_strategyqa(dataset_path))
    expected_rows = [
        {
            "qid": example.qid,
            "label": example.label,
            "fact_count": len(example.facts),
            "selection_sha256": _selection_digest(example.qid),
        }
        for example in expected
    ]
    if manifest.get("examples") != expected_rows:
        raise ValueError("manifest examples do not match the recomputed V2 sample")
    if Counter(example.label for example in expected) != Counter({False: 25, True: 25}):
        raise ValueError("V2 manifest must contain 25 examples per label")
    return expected


def build_messages(
    example: StrategyQAExample,
    view: EvidenceView,
    *,
    agent_id: str,
    persona: str,
    repair: bool = False,
) -> list[dict[str, str]]:
    """Build a stateless V2 prompt containing question and evidence, never claim."""

    system = (
        "You are a binary evidence judge in a provenance audit. "
        f"Your fixed decision style is: {persona} "
        "Treat packet statements as task-local evidence, including counterfactual statements. "
        "You may use general knowledge when evidence is insufficient, but cite only packet IDs. "
        "Do not provide reasoning, analysis, or chain-of-thought. Return only one JSON object."
    )
    task_payload = {
        "question": example.question,
        "evidence_packet": [
            {"evidence_id": evidence_id, "text": text} for evidence_id, text in view.items
        ],
    }
    user = (
        "Answer the original yes/no question under the task-local evidence. "
        "Use decision='answer' with answer='yes' or 'no', or decision='abstain' with "
        "answer=null and confidence=0.0. Confidence is the probability that your yes/no answer "
        "is correct. cited_evidence_ids must be a unique list drawn only from the packet.\n\n"
        f"agent_id must be exactly: {agent_id}\n"
        "Required keys, with no others: agent_id, decision, answer, confidence, "
        "cited_evidence_ids.\n\n"
        f"Task payload:\n{json.dumps(task_payload, ensure_ascii=False, indent=2)}"
    )
    if repair:
        user += REPAIR_SUFFIX
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_one_call(
    client: CachedChatClient,
    example: StrategyQAExample,
    view: EvidenceView,
    *,
    agent_index: int,
) -> dict[str, Any]:
    """Run one V2 condition with the V1-frozen retry and parser rules."""

    from sp500_forecastability.pilot_llm_v1 import parse_qa_decision

    agent_id, persona = AGENT_PERSONAS[agent_index]
    attempts: list[dict[str, object]] = []
    repair = False
    decision = None
    final_error: str | None = None
    for _ in range(2):
        messages = build_messages(
            example,
            view,
            agent_id=agent_id,
            persona=persona,
            repair=repair,
        )
        try:
            result = client.call(messages, seed=_agent_seed(agent_index))
        except (RuntimeError, TypeError, ValueError) as error:
            final_error = f"{type(error).__name__}: {error}"
            attempts.append(
                {
                    "cache_hit": False,
                    "cache_key": None,
                    "http_status": None,
                    "request_bytes": None,
                    "response_bytes": None,
                    "latency_seconds": None,
                    "usage": {},
                    "parse_error": None,
                    "transport_error": final_error,
                }
            )
            continue
        attempt = _attempt_payload(result)
        try:
            decision = parse_qa_decision(
                result.content,
                expected_agent_id=agent_id,
                allowed_evidence_ids=view.allowed_evidence_ids,
            )
        except (TypeError, ValueError) as error:
            final_error = f"{type(error).__name__}: {error}"
            attempt["parse_error"] = final_error
            attempts.append(attempt)
            repair = True
            continue
        attempts.append(attempt)
        final_error = None
        break

    return {
        "protocol_version": PROTOCOL_VERSION,
        "qid": example.qid,
        "label": example.label,
        "agent_id": agent_id,
        "condition": view.condition,
        "success": decision is not None,
        "first_pass_valid": decision is not None and len(attempts) == 1,
        "attempts": attempts,
        "decision": decision.to_payload() if decision is not None else None,
        "final_error": final_error,
    }


def execute_run(
    *,
    mode: str,
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    cache_dir: Path,
    smoke_examples: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = validate_manifest(manifest, dataset_path)
    if mode == "smoke":
        if not 1 <= smoke_examples <= 2:
            raise ValueError("smoke mode permits only one or two examples")
        examples = examples[:smoke_examples]
        agents = AGENT_PERSONAS[:1]
    elif mode == "formal":
        if len(examples) != FORMAL_EXAMPLES:
            raise ValueError("formal mode requires the exact 50-question V2 manifest")
        agents = AGENT_PERSONAS
    else:
        raise ValueError("mode must be smoke or formal")

    client = CachedChatClient(DEFAULT_ENDPOINT, DEFAULT_MODEL, cache_dir)
    records: list[dict[str, Any]] = []
    total = len(examples) * len(agents) * len(CONDITIONS)
    completed = 0
    for example in examples:
        for agent_index, _ in enumerate(agents):
            for condition in CONDITIONS:
                record = run_one_call(
                    client,
                    example,
                    evidence_view(example, condition),
                    agent_index=agent_index,
                )
                records.append(record)
                completed += 1
                print(
                    f"[{completed}/{total}] {example.qid} {record['agent_id']} "
                    f"{condition} success={record['success']}",
                    flush=True,
                )
                _write_jsonl(output_dir / "records.partial.jsonl", records)

    summary = summarize_records(
        records,
        mode=mode,
        expected_examples=len(examples),
        agent_count=len(agents),
        protocol_version=PROTOCOL_VERSION,
    )
    _write_jsonl(output_dir / "records.jsonl", records)
    _write_json(output_dir / "summary.json", summary)
    report = render_report(summary).replace("# Pilot-LLM V1", "# Pilot-LLM V2", 1)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    partial = output_dir / "records.partial.jsonl"
    if partial.exists():
        partial.unlink()
    return records, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write and validate the V2 manifest")
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")

    smoke = subparsers.add_parser("smoke", help="run at most six V2 instrumentation calls")
    smoke.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    smoke.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "smoke")
    smoke.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    smoke.add_argument("--examples", type=int, default=2)

    formal = subparsers.add_parser("run", help="run the frozen 750-call V2 pilot")
    formal.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    formal.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    formal.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "formal")
    formal.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = build_manifest(args.dataset)
        _write_json(args.output, manifest)
        validate_manifest(manifest, args.dataset)
        print(f"Wrote frozen V2 manifest: {args.output}")
        return 0
    if args.command == "smoke":
        execute_run(
            mode="smoke",
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            smoke_examples=args.examples,
        )
        return 0
    if args.command == "run":
        execute_run(
            mode="formal",
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
