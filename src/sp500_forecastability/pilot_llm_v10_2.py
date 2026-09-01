"""Pilot-LLM V10.2: fixed-length-repair BoolQ replication.

V10.2 inherits V10.1's frozen text-only selection without reranking it.  It
uses one initial counterfactual rewrite call and, only for an unusable response,
one deterministic length-repair call.  See
``docs/pilot_llm_v10_2_preregistration.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from sp500_forecastability import pilot_llm_v10 as base
from sp500_forecastability.pilot_llm_v1 import CachedChatClient, _write_json

PROTOCOL_VERSION = "pilot-llm-v10.2-2026-09-01"
SALT = b"pilot-llm-v10.2-2026-09-01\n"
PARENT_PROTOCOL_VERSION = "pilot-llm-v10.1-2026-09-01"
DEFAULT_ROOT = Path("results/pilot_llm_v10_2")
DEFAULT_DATASET = base.DEFAULT_DATASET
PARENT_SELECTION_PATH = Path("results/pilot_llm_v10_1/selection_manifest.json")
INITIAL_REWRITE_SEED = 20_260_905
LENGTH_REPAIR_SEED = 20_260_906


def _configure_base() -> None:
    """Reuse V10.1's frozen evaluation implementation under a new identity."""
    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.SALT = SALT
    base.CQID_PROTOCOL_VERSION = PROTOCOL_VERSION
    base.CQID_PREFIX = "v10_2"
    base.DEFAULT_ROOT = DEFAULT_ROOT


def _validate_parent_selection(
    parent_manifest: Mapping[str, object], dataset_path: Path,
) -> None:
    """Validate the exact V10.1 parent before inheriting its examples."""
    if parent_manifest.get("protocol_version") != PARENT_PROTOCOL_VERSION:
        raise ValueError("V10.2 parent must be the frozen V10.1 selection manifest")
    original_protocol = base.PROTOCOL_VERSION
    try:
        base.PROTOCOL_VERSION = PARENT_PROTOCOL_VERSION
        base.validate_manifest(
            parent_manifest, dataset_path, require_substitutes=False,
        )
    finally:
        base.PROTOCOL_VERSION = original_protocol


def _parent_lineage(parent_path: Path, parent: Mapping[str, object]) -> dict[str, str]:
    return {
        "parent_protocol_version": str(parent["protocol_version"]),
        "parent_selection_manifest": str(parent_path),
        "parent_selection_manifest_sha256": base.file_sha256(parent_path),
        "parent_dataset_sha256": str(parent["dataset_sha256"]),
        "inheritance_reason": "v10_1_auxiliary_rewrite_length_feasibility_abort_only",
    }


def build_inherited_selection_manifest(
    dataset_path: Path, parent_path: Path,
) -> dict[str, object]:
    """Copy only the V10.1 selection, explicitly recording its provenance."""
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    _validate_parent_selection(parent, dataset_path)
    manifest = deepcopy(parent)
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["status"] = "selection_inherited_from_v10_1_pre_substitution"
    manifest["selection_parent"] = _parent_lineage(parent_path, parent)
    manifest["substitute_manifest"] = {}
    _configure_base()
    base.validate_manifest(manifest, dataset_path, require_substitutes=False)
    return manifest


def write_or_validate_inherited_selection(
    output_path: Path, dataset_path: Path, parent_path: Path,
) -> bool:
    expected = build_inherited_selection_manifest(dataset_path, parent_path)
    if output_path.exists():
        actual = json.loads(output_path.read_text(encoding="utf-8"))
        _configure_base()
        base.validate_manifest(actual, dataset_path, require_substitutes=False)
        if actual != expected:
            raise ValueError(
                "existing V10.2 inherited selection differs from its frozen parent"
            )
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, expected)
    return True


def _token_bounds(source: str) -> tuple[int, int, int]:
    source_tokens = max(1, len(source.split()))
    return (
        source_tokens,
        max(1, math.ceil(0.5 * source_tokens)),
        max(1, math.floor(1.5 * source_tokens)),
    )


def _initial_prompt(item: base.BoolQItem) -> str:
    _src_tokens, lower, upper = _token_bounds(item.passage)
    opposite = "no" if item.label == "yes" else "yes"
    return (
        "Rewrite one task-local BoolQ evidence sentence so it supports the "
        f"opposite answer ({opposite}) to the question below. Preserve its topic "
        "and named entities; do not introduce a new entity. Return exactly one "
        "plain-text evidence sentence on one line, with no preface, Markdown, or "
        f"meta-language. It must contain between {lower} and {upper} whitespace "
        "tokens inclusive.\n\n"
        f"Question: {item.question}\n"
        f"Original evidence sentence: {item.passage}"
    )


def _length_repair_prompt(item: base.BoolQItem, failed_candidate: str) -> str:
    _src_tokens, lower, upper = _token_bounds(item.passage)
    opposite = "no" if item.label == "yes" else "yes"
    return (
        "Produce a replacement for the unusable counterfactual evidence below. "
        f"It must support the opposite BoolQ answer ({opposite}), retain the "
        "same topic and named entities, and introduce no new entity. Return "
        "exactly one plain-text evidence sentence on one line, with no preface, "
        f"Markdown, or meta-language, using between {lower} and {upper} whitespace "
        "tokens inclusive. Do not explain the repair.\n\n"
        f"Question: {item.question}\n"
        f"Original evidence sentence: {item.passage}\n"
        f"Unusable prior candidate: {failed_candidate.strip()}"
    )


def _parse_one_line_rewrite(content: str, source_tokens: int) -> str | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    return base._parse_substitute_response(lines[0], source_tokens)


def build_substitute_manifest_v10_2(
    items: Sequence[base.BoolQItem], *, client: CachedChatClient | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, Any]]:
    """Generate a frozen rewrite, then at most one fixed repair per item."""
    expected_qids = {item.qid for item in items}
    if len(expected_qids) != len(items):
        raise ValueError("V10.2 substitute input contains duplicate evidence IDs")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "substitute_manifest.json"
        stats_path = cache_dir / "substitute_generation_stats.json"
        if manifest_path.exists() and stats_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if set(manifest) != expected_qids:
                raise ValueError("cached V10.2 substitutes do not match frozen evidence IDs")
            return manifest, json.loads(stats_path.read_text(encoding="utf-8"))
    if client is None:
        client = CachedChatClient(
            base.DEFAULT_ENDPOINT, base.DEFAULT_MODEL,
            cache_dir or DEFAULT_ROOT / "cache",
        )

    manifest: dict[str, dict[str, object]] = {}
    stats: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "n_items": len(items),
        "initial_valid": 0,
        "repair_attempted": 0,
        "repair_valid": 0,
        "n_rewritten": 0,
        "n_unusable": 0,
        "transfer_bytes": 0,
        "in_window": 0,
        "out_of_window": 0,
        "max_repair_calls": len(items),
    }
    for index, item in enumerate(items, start=1):
        source_tokens = max(1, len(item.passage.split()))
        initial_content = ""
        rewrite: str | None = None
        try:
            prompt = _initial_prompt(item)
            result = client.call([{"role": "user", "content": prompt}], seed=INITIAL_REWRITE_SEED)
            initial_content = result.content
            stats["transfer_bytes"] += len(prompt) + len(result.content)
            rewrite = _parse_one_line_rewrite(result.content, source_tokens)
        except (RuntimeError, TypeError, ValueError):
            rewrite = None
        if rewrite is not None:
            mode = "initial"
            stats["initial_valid"] += 1
        else:
            mode = "failed"
            stats["repair_attempted"] += 1
            try:
                prompt = _length_repair_prompt(item, initial_content)
                result = client.call([{"role": "user", "content": prompt}], seed=LENGTH_REPAIR_SEED)
                stats["transfer_bytes"] += len(prompt) + len(result.content)
                rewrite = _parse_one_line_rewrite(result.content, source_tokens)
            except (RuntimeError, TypeError, ValueError):
                rewrite = None
            if rewrite is not None:
                mode = "length_repair"
                stats["repair_valid"] += 1
        if rewrite is None:
            manifest[item.qid] = {
                "substitute_sentence": "", "in_length_window": False,
                "deviation_log": ["initial_rewrite_unusable", "length_repair_unusable"],
                "generation_mode": mode, "source_label": item.label,
                "source_root": item.source_root,
            }
            stats["n_unusable"] += 1
            stats["out_of_window"] += 1
        else:
            manifest[item.qid] = {
                "substitute_sentence": rewrite, "in_length_window": True,
                "deviation_log": ["v10_2_fixed_length_repair_contract"],
                "generation_mode": mode, "source_label": item.label,
                "source_root": item.source_root,
            }
            stats["n_rewritten"] += 1
            stats["in_window"] += 1
        if index % 50 == 0 or index == len(items):
            print(
                f"[v10.2 substitute] {index}/{len(items)} initial={stats['initial_valid']} "
                f"repaired={stats['repair_valid']} unusable={stats['n_unusable']}",
                flush=True,
            )
    stats["unusable_fraction"] = stats["n_unusable"] / max(1, stats["n_items"])
    stats["passed_fail_fast"] = stats["n_unusable"] == 0
    if cache_dir is not None:
        _write_json(cache_dir / "substitute_manifest.json", manifest)
        _write_json(cache_dir / "substitute_generation_stats.json", stats)
    return manifest, stats


def run_audit(
    dataset_path: Path, selection_path: Path, output_path: Path, *, cache_dir: Path,
) -> dict[str, Any]:
    """Run V10.1's structural audit and restore explicit V10.2 lineage."""
    _configure_base()
    audit = base._pre_formal_audit(
        dataset_path, selection_path, output_path, cache_dir=cache_dir,
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    run_manifest["selection"] = selection["selection"]
    run_manifest["selection_parent"] = selection["selection_parent"]
    run_manifest["protocol_lineage"] = {
        "v10_1_rewrite_content_reused": False,
        "v10_1_evaluation_records_reused": False,
        "length_repair_policy": "one_fixed_repair_after_unusable_initial_only",
    }
    base.validate_manifest(run_manifest, dataset_path)
    _write_json(output_path, run_manifest)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="freeze the inherited V10.2 selection")
    prep.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prep.add_argument("--parent", type=Path, default=PARENT_SELECTION_PATH)
    prep.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg = sub.add_parser("substitute-generation", help="run fixed initial plus one repair")
    subg.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    subg.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    audit = sub.add_parser("audit", help="write and verify the V10.2 run manifest")
    audit.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    audit.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    audit.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    for command, help_text in (("smoke", "run the 40-call smoke test"), ("run", "run the 2,000-call formal test")):
        stage = sub.add_parser(command, help=help_text)
        stage.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        stage.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
        stage.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / ("smoke" if command == "smoke" else "formal"))
        stage.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
        stage.add_argument("--no-resume", action="store_true")
        if command == "smoke":
            stage.add_argument("--examples", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_base()
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_inherited_selection(args.output, args.dataset, args.parent)
        print(f"{'Wrote' if created else 'Reused'} frozen V10.2 inherited selection: {args.output}")
        return 0
    if args.command == "substitute-generation":
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        composites = base.validate_manifest(selection, args.dataset, require_substitutes=False)
        items = [item for composite in composites for item in composite.items]
        _manifest, stats = build_substitute_manifest_v10_2(items, cache_dir=args.cache_dir)
        for key, value in stats.items():
            print(f"[v10.2 substitute] {key}: {value}")
        return 0 if stats["passed_fail_fast"] else 2
    if args.command == "audit":
        audit = run_audit(args.dataset, args.selection_manifest, args.output, cache_dir=args.cache_dir)
        for key, value in audit.items():
            print(f"{key}: {value}")
        return 0
    if args.command in {"smoke", "run"}:
        base.execute_run(
            mode="smoke" if args.command == "smoke" else "formal",
            dataset_path=args.dataset, manifest_path=args.manifest,
            output_dir=args.output_dir, cache_dir=args.cache_dir,
            smoke_examples=getattr(args, "examples", 2), resume=not args.no_resume,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
