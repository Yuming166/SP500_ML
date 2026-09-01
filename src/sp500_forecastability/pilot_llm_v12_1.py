"""Pilot-LLM V12.1 bounded repair amendment and four-worker runner."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from sp500_forecastability import pilot_llm_v12 as v12
from sp500_forecastability.pilot_llm_v1 import (
    CachedChatClient,
    _write_json,
    _write_jsonl,
    file_sha256,
)

PROTOCOL_VERSION = "pilot-llm-v12.1-2026-09-02"
DEFAULT_ROOT = Path("results/pilot_llm_v12_1")
PARENT_SELECTION = Path("results/pilot_llm_v12/selection_manifest.json")
PARENT_SELECTION_SHA256 = (
    "1a4b953ab28e092448854a1e868fe22352e82902427634c8c85583ff7e0c5a48"
)
PARENT_SUBSTITUTES = Path("results/pilot_llm_v12/cache/substitute_manifest.json")
PARENT_SUBSTITUTES_SHA256 = (
    "779ad8caee82b029cff504fccf55bca10735f4c4bb7f38b838ff7bdf753f4ffc"
)
FAILED_QID = "boolq-1592052e5f54e039-e03"
SECOND_REPAIR_SEED = 20_260_924


def configure_v12_1() -> None:
    v12.configure_v12()
    v12.v11.PROTOCOL_VERSION = PROTOCOL_VERSION
    v12.base.PROTOCOL_VERSION = PROTOCOL_VERSION
    v12.v11.rewrite_impl.PROTOCOL_VERSION = PROTOCOL_VERSION


def _validated_parent(
    dataset_path: Path,
) -> tuple[dict[str, object], list[v12.base.CompositeQuestion]]:
    if file_sha256(PARENT_SELECTION) != PARENT_SELECTION_SHA256:
        raise ValueError("V12.1 parent selection SHA-256 drifted")
    parent = json.loads(PARENT_SELECTION.read_text(encoding="utf-8"))
    composites = v12.validate_manifest_v12(
        parent, dataset_path, require_substitutes=False,
    )
    return parent, composites


def build_inherited_selection(
    dataset_path: Path, *, status: str,
) -> tuple[dict[str, object], list[v12.base.CompositeQuestion]]:
    parent, composites = _validated_parent(dataset_path)
    manifest = deepcopy(parent)
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["status"] = status
    manifest["substitute_manifest"] = {}
    manifest["selection_parent"] = {
        "parent_protocol": v12.PROTOCOL_VERSION,
        "parent_path": str(PARENT_SELECTION),
        "parent_sha256": PARENT_SELECTION_SHA256,
        "selection_changed": False,
        "validation_agent_outputs_existed": False,
    }
    manifest["auxiliary_amendment"] = {
        "parent_substitutes": str(PARENT_SUBSTITUTES),
        "parent_substitutes_sha256": PARENT_SUBSTITUTES_SHA256,
        "inherited_usable_substitutes": 1073,
        "second_repair_qids": [FAILED_QID],
        "second_repair_seed": SECOND_REPAIR_SEED,
        "maximum_second_repair_calls": 1,
        "overlong_repairs_truncated": False,
        "selection_or_confirmatory_contract_changed": False,
    }
    manifest["rewrite_contract"] = {
        **dict(manifest["rewrite_contract"]),
        "v12_1_second_repair_qids": [FAILED_QID],
        "v12_1_second_repair_seed": SECOND_REPAIR_SEED,
        "v12_1_maximum_second_repair_calls": 1,
    }
    return manifest, composites


def validate_manifest_v12_1(
    manifest: Mapping[str, object], dataset_path: Path, *,
    require_substitutes: bool = True,
) -> list[v12.base.CompositeQuestion]:
    expected, composites = build_inherited_selection(
        dataset_path, status=str(manifest.get("status", "")),
    )
    actual_without_subs = {
        key: value for key, value in manifest.items() if key != "substitute_manifest"
    }
    expected_without_subs = {
        key: value for key, value in expected.items() if key != "substitute_manifest"
    }
    if actual_without_subs != expected_without_subs:
        differing = sorted(
            key for key in set(actual_without_subs) | set(expected_without_subs)
            if actual_without_subs.get(key) != expected_without_subs.get(key)
        )
        raise ValueError(f"V12.1 inherited manifest drifted: {differing}")
    substitutes = manifest.get("substitute_manifest", {})
    if not isinstance(substitutes, Mapping):
        raise TypeError("V12.1 substitute manifest must be a mapping")
    expected_qids = {item.qid for comp in composites for item in comp.items}
    if require_substitutes:
        if set(substitutes) != expected_qids:
            raise ValueError("V12.1 substitute IDs differ from frozen evidence IDs")
        for qid in expected_qids:
            row = substitutes[qid]
            if not row.get("substitute_sentence") or not row.get("in_length_window"):
                raise ValueError(f"V12.1 unusable substitute: {qid}")
    elif substitutes:
        raise ValueError("V12.1 selection manifest must precede its repair call")
    return composites


def write_or_validate_selection(output_path: Path, dataset_path: Path) -> bool:
    expected, _composites = build_inherited_selection(
        dataset_path, status="selection_inherited_before_v12_1_repair",
    )
    validate_manifest_v12_1(expected, dataset_path, require_substitutes=False)
    if output_path.exists():
        actual = json.loads(output_path.read_text(encoding="utf-8"))
        validate_manifest_v12_1(actual, dataset_path, require_substitutes=False)
        if actual != expected:
            raise ValueError("existing V12.1 selection differs from frozen amendment")
        return False
    _write_json(output_path, expected)
    return True


def _second_repair_prompt(item: v12.base.BoolQItem) -> str:
    source_tokens = len(item.passage.split())
    lower = (source_tokens + 1) // 2
    upper = (3 * source_tokens) // 2
    opposite = "no" if item.label == "yes" else "yes"
    return (
        "Return exactly one plain-text sentence on one line, with no explanation, "
        "Markdown, list marker, or quotation marks. Rewrite the original evidence "
        f"to support the opposite BoolQ answer ({opposite}) while preserving its "
        "topic and named entities and introducing no new entity. Your response "
        f"must contain exactly {source_tokens} whitespace-separated tokens. The "
        f"hard allowed range is {lower} through {upper} tokens; count before "
        "answering and do not exceed the upper bound.\n\n"
        f"Question: {item.question}\n"
        f"Original evidence: {item.passage}"
    )


def build_repaired_substitutes(
    composites: Sequence[v12.base.CompositeQuestion], *, cache_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    output_path = cache_dir / "substitute_manifest.json"
    stats_path = cache_dir / "substitute_generation_stats.json"
    expected_qids = {item.qid for comp in composites for item in comp.items}
    if output_path.exists() and stats_path.exists():
        manifest = json.loads(output_path.read_text(encoding="utf-8"))
        if set(manifest) != expected_qids:
            raise ValueError("cached V12.1 substitutes differ from frozen evidence IDs")
        return manifest, json.loads(stats_path.read_text(encoding="utf-8"))
    if file_sha256(PARENT_SUBSTITUTES) != PARENT_SUBSTITUTES_SHA256:
        raise ValueError("V12.1 parent substitutes SHA-256 drifted")
    parent = json.loads(PARENT_SUBSTITUTES.read_text(encoding="utf-8"))
    invalid = {
        qid for qid, row in parent.items()
        if not row.get("substitute_sentence") or not row.get("in_length_window")
    }
    if invalid != {FAILED_QID}:
        raise ValueError(f"V12.1 repair set drifted: {sorted(invalid)}")
    item = next(
        item for comp in composites for item in comp.items if item.qid == FAILED_QID
    )
    source_tokens = len(item.passage.split())
    if source_tokens != 13:
        raise ValueError("V12.1 frozen repair source length drifted")
    configure_v12_1()
    client = CachedChatClient(
        v12.base.DEFAULT_ENDPOINT, v12.base.DEFAULT_MODEL,
        cache_dir / "second_repair",
    )
    candidate: str | None = None
    prompt = _second_repair_prompt(item)
    transfer_bytes = 0
    try:
        result = client.call(
            [{"role": "user", "content": prompt}], seed=SECOND_REPAIR_SEED,
        )
        transfer_bytes = len(prompt) + len(result.content)
        candidate = v12.v11.rewrite_impl._single_line_candidate(result.content)
    except (RuntimeError, TypeError, ValueError):
        candidate = None
    rewrite: str | None = None
    mode = "v12_1_second_repair_no_single_line_candidate"
    if candidate is not None:
        rewrite, normalized_mode = v12.normalize_repaired_candidate(
            candidate, source_tokens,
        )
        mode = f"v12_1_second_{normalized_mode}"
    manifest = deepcopy(parent)
    manifest[FAILED_QID] = {
        "substitute_sentence": rewrite or "",
        "in_length_window": rewrite is not None,
        "deviation_log": [mode],
        "generation_mode": mode,
        "source_label": item.label,
        "source_root": item.source_root,
    }
    stats: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_substitutes_sha256": PARENT_SUBSTITUTES_SHA256,
        "n_items": len(expected_qids),
        "inherited_usable": len(expected_qids) - 1,
        "second_repair_attempted": 1,
        "second_repair_valid": int(rewrite is not None),
        "second_repair_mode": mode,
        "second_repair_transfer_bytes": transfer_bytes,
        "n_rewritten": sum(bool(row.get("in_length_window")) for row in manifest.values()),
    }
    stats["n_unusable"] = len(expected_qids) - int(stats["n_rewritten"])
    stats["passed_fail_fast"] = stats["n_unusable"] == 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, dict(sorted(manifest.items())))
    _write_json(stats_path, stats)
    return manifest, stats


def run_audit(
    dataset_path: Path, selection_path: Path, output_path: Path, *,
    cache_dir: Path,
) -> dict[str, object]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    composites = validate_manifest_v12_1(
        selection, dataset_path, require_substitutes=False,
    )
    stats = json.loads(
        (cache_dir / "substitute_generation_stats.json").read_text(encoding="utf-8")
    )
    substitutes = json.loads(
        (cache_dir / "substitute_manifest.json").read_text(encoding="utf-8")
    )
    if stats.get("passed_fail_fast") is not True:
        raise ValueError("V12.1 bounded repair gate failed")
    manifest, _ = build_inherited_selection(dataset_path, status="run_frozen")
    manifest["substitute_manifest"] = substitutes
    validate_manifest_v12_1(manifest, dataset_path, require_substitutes=True)
    _write_json(output_path, manifest)
    return {
        "n_composites": len(composites),
        "labels": dict(Counter(comp.label for comp in composites)),
        "n_substitutes": stats["n_rewritten"],
        "second_repair_attempted": stats["second_repair_attempted"],
        "second_repair_valid": stats["second_repair_valid"],
        "workers": v12.N_WORKERS,
        "formal_logical_calls": 7160,
        "primary_endpoint": v12.PRIMARY_ENDPOINT,
        "passes": True,
    }


def execute_parallel_v12_1(
    *, mode: str, dataset_path: Path, manifest_path: Path,
    output_dir: Path, cache_dir: Path, resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, object], list[v12.base.CompositeQuestion]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composites = validate_manifest_v12_1(manifest, dataset_path, require_substitutes=True)
    configure_v12_1()
    full_shards = v12.assign_shards(composites)
    if mode == "smoke":
        run_shards = v12._smoke_composites(full_shards)
        expected_examples = v12.SMOKE_EXAMPLES
        expected_calls = v12.SMOKE_CALLS
    elif mode == "formal":
        run_shards = full_shards
        expected_examples = v12.FORMAL_EXAMPLES
        expected_calls = 7160
    else:
        raise ValueError("V12.1 mode must be smoke or formal")
    substitutes = manifest["substitute_manifest"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_records: list[list[dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=v12.N_WORKERS) as pool:
        futures = {
            pool.submit(
                v12._worker_run,
                index,
                run_shards[index],
                substitutes,
                output_dir=output_dir,
                cache_dir=cache_dir,
                resume=resume,
            ): index
            for index in range(v12.N_WORKERS)
        }
        for future in as_completed(futures):
            index = futures[future]
            rows = future.result()
            shard_records.append(rows)
            print(f"[v12.1 shard {index}] complete records={len(rows)}", flush=True)
    records = [record for rows in shard_records for record in rows]
    records.sort(key=v12._record_key)
    keys = [v12._record_key(record) for record in records]
    if len(records) != expected_calls or len(set(keys)) != expected_calls:
        raise ValueError(
            f"V12.1 merge failed: records={len(records)}, unique={len(set(keys))}"
        )
    if Counter(record.get("protocol_version") for record in records) != Counter({
        PROTOCOL_VERSION: expected_calls,
    }):
        raise ValueError("V12.1 record protocol versions drifted")
    condition_counts = Counter(record["condition"] for record in records)
    if condition_counts != Counter({
        condition: expected_examples * v12.base.N_AGENTS
        for condition in v12.base.CONDITIONS
    }):
        raise ValueError(f"V12.1 condition counts drifted: {dict(condition_counts)}")
    agent_counts = Counter(int(record["agent_index"]) for record in records)
    if agent_counts != Counter({
        index: expected_examples * len(v12.base.CONDITIONS)
        for index in range(v12.base.N_AGENTS)
    }):
        raise ValueError(f"V12.1 agent counts drifted: {dict(agent_counts)}")
    _write_jsonl(output_dir / "records.jsonl", records)
    failures = [record for record in records if not record.get("success")]
    if failures:
        _write_json(output_dir / "formal_abort.json", {
            "protocol_version": PROTOCOL_VERSION,
            "failed_records": len(failures),
        })
        raise RuntimeError(f"V12.1 has {len(failures)} failed logical records")
    summary = v12.v11.summarize_records_v11(
        records, mode=mode, expected_examples=expected_examples,
        agent_count=v12.base.N_AGENTS, substitute_manifest=substitutes,
    )
    summary["parallel_execution"] = {
        "workers": v12.N_WORKERS,
        "endpoint_concurrency_only": True,
        "physical_gpu_count_verified": False,
        "shard_record_counts": sorted(len(rows) for rows in shard_records),
        "merge_unique_records": len(set(keys)),
    }
    _write_json(output_dir / "summary.json", summary)
    report = v12.v11.render_report_v11(summary).replace(
        "Pilot-LLM V11", "Pilot-LLM V12.1"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    for index in range(v12.N_WORKERS):
        shard_dir = output_dir / "shards" / f"shard_{index}"
        for name in ("records.partial.jsonl", "progress.json"):
            path = shard_dir / name
            if path.exists():
                path.unlink()
    return records, summary, composites


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, default=v12.DEFAULT_DATASET)
    prep.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json",
    )
    repair = sub.add_parser("substitute-generation")
    repair.add_argument("--dataset", type=Path, default=v12.DEFAULT_DATASET)
    repair.add_argument(
        "--selection-manifest", type=Path,
        default=DEFAULT_ROOT / "selection_manifest.json",
    )
    repair.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", type=Path, default=v12.DEFAULT_DATASET)
    audit.add_argument(
        "--selection-manifest", type=Path,
        default=DEFAULT_ROOT / "selection_manifest.json",
    )
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    audit.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    for command in ("smoke", "run"):
        stage = sub.add_parser(command)
        stage.add_argument("--dataset", type=Path, default=v12.DEFAULT_DATASET)
        stage.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
        stage.add_argument(
            "--output-dir", type=Path,
            default=DEFAULT_ROOT / ("smoke" if command == "smoke" else "formal"),
        )
        stage.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
        stage.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_selection(args.output, args.dataset)
        print(f"{'Wrote' if created else 'Reused'} frozen V12.1 selection: {args.output}")
        return 0
    if args.command == "substitute-generation":
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        composites = validate_manifest_v12_1(
            selection, args.dataset, require_substitutes=False,
        )
        _manifest, stats = build_repaired_substitutes(
            composites, cache_dir=args.cache_dir,
        )
        for key, value in stats.items():
            print(f"[v12.1 substitute] {key}: {value}")
        return 0 if stats["passed_fail_fast"] else 2
    if args.command == "audit":
        result = run_audit(
            args.dataset, args.selection_manifest, args.output,
            cache_dir=args.cache_dir,
        )
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0
    if args.command in {"smoke", "run"}:
        execute_parallel_v12_1(
            mode="smoke" if args.command == "smoke" else "formal",
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            resume=not args.no_resume,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
