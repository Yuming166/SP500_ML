"""Pilot-LLM V10.4 deterministic short-rewrite normalization runner.

The version inherits V10.1's frozen selection and V10.1 evaluation mechanics,
but uses a fresh cache and one preregistered deterministic length adjustment.
See ``docs/pilot_llm_v10_4_preregistration.md``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sp500_forecastability import pilot_llm_v10 as base
from sp500_forecastability import pilot_llm_v10_2 as implementation
from sp500_forecastability.pilot_llm_v1 import CachedChatClient, _write_json

PROTOCOL_VERSION = "pilot-llm-v10.4-2026-09-01"
SALT = b"pilot-llm-v10.4-2026-09-01\n"
DEFAULT_ROOT = Path("results/pilot_llm_v10_4")
INITIAL_REWRITE_SEED = 20_260_909
NEUTRAL_QUALIFIER = "in the described local situation."
_IMPLEMENTATION_AUDIT = implementation.run_audit


def _initial_prompt(item: base.BoolQItem) -> str:
    source_tokens, _lower, _upper = implementation._token_bounds(item.passage)
    opposite = "no" if item.label == "yes" else "yes"
    return (
        "Rewrite one task-local BoolQ evidence sentence so it supports the "
        f"opposite answer ({opposite}) to the question below. Preserve its topic "
        "and named entities; do not introduce a new entity. Return exactly one "
        "plain-text evidence sentence on one line, with no preface, Markdown, or "
        "meta-language. Before sending, count whitespace-separated tokens: your "
        f"response must contain exactly {source_tokens} tokens.\n\n"
        f"Question: {item.question}\n"
        f"Original evidence sentence: {item.passage}"
    )


def _single_line_candidate(content: str) -> str | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    candidate = re.sub(r"^[-*]\s+", "", lines[0]).strip()
    return candidate or None


def normalize_short_candidate(candidate: str, source_tokens: int) -> tuple[str | None, str]:
    """Return a valid candidate and its fixed generation mode, or a failure mode."""
    token_count = len(candidate.split())
    if 0.5 <= token_count / source_tokens <= 1.5:
        return candidate, "initial"
    if token_count / source_tokens >= 0.5:
        return None, "failed_overlong"
    stem = re.sub(r"[.!?]+$", "", candidate).rstrip()
    normalized = f"{stem} {NEUTRAL_QUALIFIER}".strip()
    normalized_tokens = len(normalized.split())
    if 0.5 <= normalized_tokens / source_tokens <= 1.5:
        return normalized, "deterministic_short_normalization"
    return None, "failed_short_after_normalization"


def build_substitute_manifest_v10_4(
    items: Sequence[base.BoolQItem], *, client: CachedChatClient | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, Any]]:
    expected_qids = {item.qid for item in items}
    if len(expected_qids) != len(items):
        raise ValueError("V10.4 substitute input contains duplicate evidence IDs")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "substitute_manifest.json"
        stats_path = cache_dir / "substitute_generation_stats.json"
        if manifest_path.exists() and stats_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if set(manifest) != expected_qids:
                raise ValueError("cached V10.4 substitutes do not match frozen evidence IDs")
            return manifest, json.loads(stats_path.read_text(encoding="utf-8"))
    if client is None:
        client = CachedChatClient(
            base.DEFAULT_ENDPOINT, base.DEFAULT_MODEL,
            cache_dir or DEFAULT_ROOT / "cache",
        )
    manifest: dict[str, dict[str, object]] = {}
    stats: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION, "n_items": len(items),
        "initial_valid": 0, "normalized_short": 0, "n_rewritten": 0,
        "n_unusable": 0, "transfer_bytes": 0, "in_window": 0,
        "out_of_window": 0, "model_repair_calls": 0,
    }
    for index, item in enumerate(items, start=1):
        source_tokens = max(1, len(item.passage.split()))
        candidate: str | None = None
        try:
            prompt = _initial_prompt(item)
            result = client.call([{"role": "user", "content": prompt}], seed=INITIAL_REWRITE_SEED)
            stats["transfer_bytes"] += len(prompt) + len(result.content)
            candidate = _single_line_candidate(result.content)
        except (RuntimeError, TypeError, ValueError):
            candidate = None
        rewrite: str | None = None
        mode = "failed_no_single_line_candidate"
        if candidate is not None:
            rewrite, mode = normalize_short_candidate(candidate, source_tokens)
        if rewrite is None:
            manifest[item.qid] = {
                "substitute_sentence": "", "in_length_window": False,
                "deviation_log": [mode], "generation_mode": mode,
                "source_label": item.label, "source_root": item.source_root,
            }
            stats["n_unusable"] += 1
            stats["out_of_window"] += 1
        else:
            manifest[item.qid] = {
                "substitute_sentence": rewrite, "in_length_window": True,
                "deviation_log": [mode], "generation_mode": mode,
                "source_label": item.label, "source_root": item.source_root,
            }
            stats["n_rewritten"] += 1
            stats["in_window"] += 1
            if mode == "initial":
                stats["initial_valid"] += 1
            else:
                stats["normalized_short"] += 1
        if index % 50 == 0 or index == len(items):
            print(
                f"[v10.4 substitute] {index}/{len(items)} initial={stats['initial_valid']} "
                f"normalized={stats['normalized_short']} unusable={stats['n_unusable']}",
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
    audit = _IMPLEMENTATION_AUDIT(
        dataset_path, selection_path, output_path, cache_dir=cache_dir,
    )
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    lineage = dict(manifest.get("protocol_lineage", {}))
    lineage.update({
        "v10_1_v10_2_v10_3_rewrite_content_reused": False,
        "model_repair_calls": 0,
        "short_normalization": NEUTRAL_QUALIFIER,
    })
    manifest["protocol_lineage"] = lineage
    base.validate_manifest(manifest, dataset_path)
    _write_json(output_path, manifest)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="freeze V10.4's inherited selection")
    prep.add_argument("--dataset", type=Path, default=implementation.DEFAULT_DATASET)
    prep.add_argument("--parent", type=Path, default=implementation.PARENT_SELECTION_PATH)
    prep.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg = sub.add_parser("substitute-generation", help="one call plus deterministic normalization")
    subg.add_argument("--dataset", type=Path, default=implementation.DEFAULT_DATASET)
    subg.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    audit = sub.add_parser("audit", help="audit all 300 frozen rewrites")
    audit.add_argument("--dataset", type=Path, default=implementation.DEFAULT_DATASET)
    audit.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    audit.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    for command, help_text in (("smoke", "run the 40-call smoke test"), ("run", "run the formal 2,000-call test")):
        stage = sub.add_parser(command, help=help_text)
        stage.add_argument("--dataset", type=Path, default=implementation.DEFAULT_DATASET)
        stage.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
        stage.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / ("smoke" if command == "smoke" else "formal"))
        stage.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
        stage.add_argument("--no-resume", action="store_true")
        if command == "smoke":
            stage.add_argument("--examples", type=int, default=2)
    return parser


def configure_v10_4() -> None:
    implementation.PROTOCOL_VERSION = PROTOCOL_VERSION
    implementation.SALT = SALT
    implementation.DEFAULT_ROOT = DEFAULT_ROOT
    implementation._configure_base()


def main(argv: Sequence[str] | None = None) -> int:
    configure_v10_4()
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        created = implementation.write_or_validate_inherited_selection(
            args.output, args.dataset, args.parent,
        )
        print(f"{'Wrote' if created else 'Reused'} frozen V10.4 inherited selection: {args.output}")
        return 0
    if args.command == "substitute-generation":
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        composites = base.validate_manifest(selection, args.dataset, require_substitutes=False)
        items = [item for composite in composites for item in composite.items]
        _manifest, stats = build_substitute_manifest_v10_4(items, cache_dir=args.cache_dir)
        for key, value in stats.items():
            print(f"[v10.4 substitute] {key}: {value}")
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
