"""Pilot-LLM V10.3: exact-source-token counterfactual rewrite replication.

This small versioned runner reuses V10.2's audited mechanics while injecting
V10.3's independently preregistered exact-length prompts, identity, cache
root, and lineage.  It never reads V10.2 rewrite content.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sp500_forecastability import pilot_llm_v10 as base
from sp500_forecastability import pilot_llm_v10_2 as implementation
from sp500_forecastability.pilot_llm_v1 import _write_json

PROTOCOL_VERSION = "pilot-llm-v10.3-2026-09-01"
SALT = b"pilot-llm-v10.3-2026-09-01\n"
DEFAULT_ROOT = Path("results/pilot_llm_v10_3")
INITIAL_REWRITE_SEED = 20_260_907
LENGTH_REPAIR_SEED = 20_260_908
_IMPLEMENTATION_AUDIT = implementation.run_audit


def _exact_length_prompt(item: base.BoolQItem, *, failed_candidate: str | None = None) -> str:
    source_tokens, _lower, _upper = implementation._token_bounds(item.passage)
    opposite = "no" if item.label == "yes" else "yes"
    prefix = (
        "Rewrite one task-local BoolQ evidence sentence so it supports the "
        f"opposite answer ({opposite}) to the question below. Preserve its topic "
        "and named entities; do not introduce a new entity. Return exactly one "
        "plain-text evidence sentence on one line, with no preface, Markdown, or "
        "meta-language. Before sending, count whitespace-separated tokens: your "
        f"response must contain exactly {source_tokens} tokens.\n\n"
    )
    if failed_candidate is not None:
        prefix = (
            "Produce a replacement for the unusable counterfactual evidence "
            "below. " + prefix.replace("Rewrite one", "Rewrite the one")
        )
    prompt = (
        prefix
        + f"Question: {item.question}\n"
        + f"Original evidence sentence: {item.passage}"
    )
    if failed_candidate is not None:
        prompt += f"\nUnusable prior candidate: {failed_candidate.strip()}"
    return prompt


def _initial_prompt(item: base.BoolQItem) -> str:
    return _exact_length_prompt(item)


def _length_repair_prompt(item: base.BoolQItem, failed_candidate: str) -> str:
    return _exact_length_prompt(item, failed_candidate=failed_candidate)


def run_audit(
    dataset_path: Path, selection_path: Path, output_path: Path, *, cache_dir: Path,
) -> dict[str, Any]:
    """Add V10.3's no-reuse statement after the inherited structural audit."""
    audit = _IMPLEMENTATION_AUDIT(
        dataset_path, selection_path, output_path, cache_dir=cache_dir,
    )
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    lineage = dict(manifest.get("protocol_lineage", {}))
    lineage.update({
        "v10_2_rewrite_content_reused": False,
        "v10_2_evaluation_records_reused": False,
        "rewrite_length_target": "exact_source_whitespace_token_count",
    })
    manifest["protocol_lineage"] = lineage
    base.validate_manifest(manifest, dataset_path)
    _write_json(output_path, manifest)
    return audit


def configure_v10_3() -> None:
    """Bind the shared, tested runner to the V10.3 frozen parameters."""
    implementation.PROTOCOL_VERSION = PROTOCOL_VERSION
    implementation.SALT = SALT
    implementation.DEFAULT_ROOT = DEFAULT_ROOT
    implementation.INITIAL_REWRITE_SEED = INITIAL_REWRITE_SEED
    implementation.LENGTH_REPAIR_SEED = LENGTH_REPAIR_SEED
    implementation._initial_prompt = _initial_prompt
    implementation._length_repair_prompt = _length_repair_prompt
    implementation.run_audit = run_audit
    implementation._configure_base()


def main(argv: Sequence[str] | None = None) -> int:
    configure_v10_3()
    args = implementation._parser().parse_args(argv)
    if args.command == "prepare":
        created = implementation.write_or_validate_inherited_selection(
            args.output, args.dataset, args.parent,
        )
        print(
            f"{'Wrote' if created else 'Reused'} frozen V10.3 inherited selection: "
            f"{args.output}"
        )
        return 0
    if args.command == "substitute-generation":
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        composites = base.validate_manifest(
            selection, args.dataset, require_substitutes=False,
        )
        items = [item for composite in composites for item in composite.items]
        _manifest, stats = implementation.build_substitute_manifest_v10_2(
            items, cache_dir=args.cache_dir,
        )
        for key, value in stats.items():
            print(f"[v10.3 substitute] {key}: {value}")
        return 0 if stats["passed_fail_fast"] else 2
    if args.command == "audit":
        audit = run_audit(
            args.dataset, args.selection_manifest, args.output,
            cache_dir=args.cache_dir,
        )
        for key, value in audit.items():
            print(f"{key}: {value}")
        return 0
    if args.command in {"smoke", "run"}:
        base.execute_run(
            mode="smoke" if args.command == "smoke" else "formal",
            dataset_path=args.dataset, manifest_path=args.manifest,
            output_dir=args.output_dir, cache_dir=args.cache_dir,
            smoke_examples=getattr(args, "examples", 2),
            resume=not args.no_resume,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
