"""Pilot-LLM V12 exhaustive held-out BoolQ replication with four workers."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any

from sp500_forecastability import pilot_llm_v10 as base
from sp500_forecastability import pilot_llm_v11 as v11
from sp500_forecastability.pilot_llm_v1 import (
    CachedChatClient,
    _write_json,
    _write_jsonl,
    file_sha256,
)

PROTOCOL_VERSION = "pilot-llm-v12-2026-09-01"
SALT = b"pilot-llm-v12-2026-09-01\n"
DEFAULT_DATASET = Path(
    "/storage/gaoym/sp500-forecastability-lab/data/boolq/validation.parquet"
)
DEFAULT_ROOT = Path("results/pilot_llm_v12")
PARENT_SELECTION = Path("results/pilot_llm_v11_1/selection_manifest.json")
PARENT_SELECTION_SHA256 = (
    "78d3e660ccbfba777296a433283f1c0045cbec280181211266280b686a5771f6"
)
DATASET_SHA256 = (
    "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8"
)
FORMAL_EXAMPLES = 358
LABEL_COUNTS = {"yes": 246, "no": 112}
ELIGIBLE_UNIVERSE = 558
PARENT_EXAMPLES = 200
N_WORKERS = 4
BOOTSTRAP_SEED = 20_260_921
BOOTSTRAP_REPLICATES = 1_000
INITIAL_REWRITE_SEED = 20_260_922
REPAIR_REWRITE_SEED = 20_260_923
SMOKE_EXAMPLES = 4
SMOKE_CALLS = 80
PRIMARY_ENDPOINT = "AUROC(R_PI,consensus_wrong|original_agreement>=0.8)"
NEUTRAL_QUALIFIER = "in the described local situation."

_PRINT_LOCK = threading.Lock()


def configure_v12() -> None:
    """Set shared implementation globals to the frozen V12 values."""
    v11.PROTOCOL_VERSION = PROTOCOL_VERSION
    v11.SALT = SALT
    v11.DEFAULT_ROOT = DEFAULT_ROOT
    v11.FORMAL_EXAMPLES = FORMAL_EXAMPLES
    v11.FORMAL_PER_LABEL = 0  # V12 uses all remaining roots, not balancing.
    v11.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    v11.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
    v11.configure_base()
    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.SALT = SALT
    base.CQID_PROTOCOL_VERSION = PROTOCOL_VERSION
    base.CQID_PREFIX = "v12"
    base.FORMAL_EXAMPLES = FORMAL_EXAMPLES
    base.FORMAL_PER_LABEL = 0
    base.DEFAULT_DATASET = DEFAULT_DATASET
    base.DEFAULT_ROOT = DEFAULT_ROOT
    base.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    base.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
    v11.rewrite_impl.PROTOCOL_VERSION = PROTOCOL_VERSION
    v11.rewrite_impl.DEFAULT_ROOT = DEFAULT_ROOT
    v11.rewrite_impl.INITIAL_REWRITE_SEED = INITIAL_REWRITE_SEED


def _parent_roots(parent_path: Path = PARENT_SELECTION) -> set[str]:
    if file_sha256(parent_path) != PARENT_SELECTION_SHA256:
        raise ValueError("V12 parent selection SHA-256 drifted")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    roots = {str(row["source_root"]) for row in parent.get("examples", [])}
    if len(roots) != PARENT_EXAMPLES:
        raise ValueError("V12 parent selection must contain exactly 200 roots")
    return roots


def build_remaining_composites(
    items: Sequence[base.BoolQItem], *, parent_path: Path = PARENT_SELECTION,
) -> list[base.CompositeQuestion]:
    """Use every eligible validation root not frozen in V11.1."""
    configure_v12()
    excluded = _parent_roots(parent_path)
    by_root: dict[str, list[base.BoolQItem]] = {}
    for item in items:
        by_root.setdefault(item.source_root, []).append(item)
    if len(by_root) != ELIGIBLE_UNIVERSE:
        raise ValueError(f"V12 eligible universe drifted: {len(by_root)}")

    composites: list[base.CompositeQuestion] = []
    for source_root, members in by_root.items():
        if source_root in excluded:
            continue
        triple = tuple(sorted(members, key=lambda item: item.evidence_index))
        if len(triple) != base.FACTS_PER_QUESTION:
            raise ValueError(f"eligible root {source_root} lacks three evidence units")
        if len({item.label for item in triple}) != 1:
            raise ValueError(f"eligible root {source_root} mixes labels")
        if len({item.question for item in triple}) != 1:
            raise ValueError(f"eligible root {source_root} mixes questions")
        label = triple[0].label
        cqid_seed = sha256(
            f"{PROTOCOL_VERSION}\n{source_root}".encode()
        ).hexdigest()[:10]
        composites.append(base.CompositeQuestion(
            cqid=f"v12-{label}-{cqid_seed}",
            question_text=base._build_composite_question_text(triple),
            items=triple,
            label=label,
        ))
    composites.sort(key=lambda comp: comp.cqid)
    counts = Counter(comp.label for comp in composites)
    if len(composites) != FORMAL_EXAMPLES or dict(counts) != LABEL_COUNTS:
        raise ValueError(
            f"V12 remaining selection drifted: n={len(composites)}, labels={dict(counts)}"
        )
    if {comp.items[0].source_root for comp in composites} & excluded:
        raise ValueError("V12 selection overlaps V11.1 provenance roots")
    return composites


def assign_shards(
    composites: Sequence[base.CompositeQuestion],
) -> list[list[base.CompositeQuestion]]:
    """Frozen label-stratified, question-block round robin assignment."""
    shards: list[list[base.CompositeQuestion]] = [[] for _ in range(N_WORKERS)]
    for label in ("no", "yes"):
        group = sorted(
            (comp for comp in composites if comp.label == label),
            key=lambda comp: comp.cqid,
        )
        for index, comp in enumerate(group):
            shards[index % N_WORKERS].append(comp)
    for shard in shards:
        shard.sort(key=lambda comp: comp.cqid)
    return shards


def _shard_contract(composites: Sequence[base.CompositeQuestion]) -> dict[str, object]:
    shards = assign_shards(composites)
    return {
        "workers": N_WORKERS,
        "endpoint_concurrency_only": True,
        "physical_gpu_count_verified": False,
        "assignment": "within_label_cqid_sorted_question_block_round_robin",
        "separate_cache_partial_progress_per_shard": True,
        "no_interim_metrics": True,
        "canonical_merge_key": ["cqid", "agent_index", "condition"],
        "shards": [
            {
                "shard": index,
                "questions": len(shard),
                "labels": dict(Counter(comp.label for comp in shard)),
                "logical_calls": len(shard) * base.N_AGENTS * len(base.CONDITIONS),
            }
            for index, shard in enumerate(shards)
        ],
    }


def build_manifest_v12(
    dataset_path: Path,
    composites: Sequence[base.CompositeQuestion],
    substitute_manifest: Mapping[str, Mapping[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    configure_v12()
    manifest = base.build_manifest(
        dataset_path, composites, substitute_manifest, status=status,
    )
    manifest["selection"] = {
        "mode": "all_eligible_validation_roots_excluding_v11_1",
        "total": FORMAL_EXAMPLES,
        "label_counts": dict(LABEL_COUNTS),
        "eligible_universe": ELIGIBLE_UNIVERSE,
        "excluded_parent_roots": PARENT_EXAMPLES,
        "parent_selection": str(PARENT_SELECTION),
        "parent_selection_sha256": PARENT_SELECTION_SHA256,
        "dataset_split": "validation",
        "source_root": "sha256(question + newline + raw_passage)",
        "sentence_rule": "first_three_8_to_80_token_sentences",
        "mean_cosine_min": base.MIN_MEAN_COSINE,
        "max_cosine_range": [base.MIN_MAX_COSINE, base.MAX_MAX_COSINE],
        "v11_1_overlap": 0,
    }
    manifest["development_contract"] = {
        "v11_1_outcome_known_before_v12": True,
        "v11_1_summary_sha256": (
            "847023eb31c188730c78fb7fcf9f264f20386006bc4f606bea4417d606e11e87"
        ),
        "role": "sample_size_motivation_only",
        "score_endpoint_threshold_unchanged": True,
    }
    manifest["co_primary_endpoints"] = []
    manifest["primary_endpoint"] = PRIMARY_ENDPOINT
    manifest["secondary_endpoints"] = [
        "D_inert_high_consensus",
        "flip_inertia_high_consensus",
        "frac_shared_high_consensus",
        "R_PI_harmful_fc_all",
        "rank_router_coverage_0.8",
        "descriptive_boolq_label_subgroups",
        "secondary_v11_1_v12_cumulative",
    ]
    manifest["risk_contract"] = {
        "name": "R_PI",
        "weights": dict(v11.RISK_WEIGHTS),
        "allowed_inputs": sorted(v11.RISK_WEIGHTS),
        "forbidden_inputs": sorted(v11.FORBIDDEN_RISK_INPUTS),
        "outcome_independent": True,
        "high_consensus_threshold": v11.HIGH_CONSENSUS_THRESHOLD,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "minimum_high_consensus": v11.PRIMARY_MIN_HIGH_CONSENSUS,
        "minimum_per_class": v11.PRIMARY_MIN_PER_CLASS,
    }
    manifest["rewrite_contract"] = {
        "initial_seed": INITIAL_REWRITE_SEED,
        "maximum_initial_calls_per_evidence": 1,
        "repair_seed": REPAIR_REWRITE_SEED,
        "maximum_repair_calls_per_initial_failure": 1,
        "short_repair_suffix": NEUTRAL_QUALIFIER,
        "overlong_repairs_truncated": False,
        "fail_if_any_unusable": True,
    }
    manifest["parallel_contract"] = _shard_contract(composites)
    manifest["smoke_contract"] = {
        "examples": SMOKE_EXAMPLES,
        "workers": N_WORKERS,
        "agents": base.N_AGENTS,
        "conditions": len(base.CONDITIONS),
        "logical_calls": SMOKE_CALLS,
    }
    return manifest


def validate_manifest_v12(
    manifest: Mapping[str, object], dataset_path: Path, *,
    require_substitutes: bool = True,
) -> list[base.CompositeQuestion]:
    configure_v12()
    if file_sha256(dataset_path) != DATASET_SHA256:
        raise ValueError("V12 dataset SHA-256 drifted")
    composites = build_remaining_composites(base.load_boolq(dataset_path))
    expected = build_manifest_v12(
        dataset_path, composites, {}, status=str(manifest.get("status", "")),
    )
    fixed_keys = {
        "protocol_version", "dataset_path", "dataset_sha256", "selection",
        "status", "model", "endpoint", "n_agents", "facts_per_question",
        "facts_per_agent", "partition_table", "conditions", "agents",
        "co_primary_endpoints", "secondary_endpoints", "examples",
        "development_contract", "primary_endpoint", "risk_contract",
        "rewrite_contract", "parallel_contract", "smoke_contract",
    }
    for key in fixed_keys:
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"V12 manifest field drifted: {key}")
    if manifest.get("primary_endpoint") != PRIMARY_ENDPOINT:
        raise ValueError("V12 primary endpoint drifted")
    if manifest.get("co_primary_endpoints") != []:
        raise ValueError("V12 must have exactly one primary endpoint")
    substitutes = manifest.get("substitute_manifest", {})
    if not isinstance(substitutes, Mapping):
        raise TypeError("V12 substitute manifest must be a mapping")
    expected_qids = {item.qid for comp in composites for item in comp.items}
    if require_substitutes:
        if set(substitutes) != expected_qids:
            raise ValueError("V12 substitute IDs differ from 1,074 frozen evidence IDs")
        for qid in expected_qids:
            row = substitutes[qid]
            if not row.get("substitute_sentence") or not row.get("in_length_window"):
                raise ValueError(f"V12 unusable frozen substitute: {qid}")
    elif substitutes:
        raise ValueError("V12 selection manifest must precede auxiliary substitutes")
    return composites


def write_or_validate_selection(output_path: Path, dataset_path: Path) -> bool:
    composites = build_remaining_composites(base.load_boolq(dataset_path))
    expected = build_manifest_v12(
        dataset_path, composites, {}, status="selection_frozen_before_v12_calls",
    )
    validate_manifest_v12(expected, dataset_path, require_substitutes=False)
    if output_path.exists():
        actual = json.loads(output_path.read_text(encoding="utf-8"))
        validate_manifest_v12(actual, dataset_path, require_substitutes=False)
        if actual != expected:
            raise ValueError("existing V12 selection differs from frozen design")
        return False
    _write_json(output_path, expected)
    return True


def _repair_prompt(item: base.BoolQItem, failed: str) -> str:
    source_tokens, lower, upper = v11.rewrite_impl.implementation._token_bounds(
        item.passage
    )
    opposite = "no" if item.label == "yes" else "yes"
    return (
        "Repair the counterfactual BoolQ evidence below. Return exactly one "
        "plain-text sentence on one line with no explanation or Markdown. It "
        f"must support the opposite answer ({opposite}), preserve the same topic "
        "and named entities, and introduce no new entity. The allowed length is "
        f"{lower} through {upper} whitespace tokens inclusive; target exactly "
        f"{source_tokens} tokens and count before answering.\n\n"
        f"Question: {item.question}\n"
        f"Original evidence: {item.passage}\n"
        f"Unusable candidate: {failed.strip() or '[no usable prior candidate]'}"
    )


def normalize_repaired_candidate(
    candidate: str, source_tokens: int,
) -> tuple[str | None, str]:
    count = len(candidate.split())
    lower = max(1, (source_tokens + 1) // 2)
    upper = max(1, (3 * source_tokens) // 2)
    if lower <= count <= upper:
        return candidate, "repair"
    if count > upper:
        return None, "repair_overlong"
    stem = candidate.rstrip().rstrip(".!?").rstrip()
    repeats = 0
    while len(stem.split()) < lower:
        proposed = f"{stem} {NEUTRAL_QUALIFIER}".strip()
        if len(proposed.split()) > upper:
            return None, "repair_short_unfixable"
        stem = proposed.rstrip().rstrip(".!?").rstrip()
        repeats += 1
    return f"{stem}.", f"repair_neutral_suffix_x{repeats}"


def _rewrite_one(
    item: base.BoolQItem, client: CachedChatClient,
) -> tuple[str, dict[str, object], dict[str, int]]:
    source_tokens = max(1, len(item.passage.split()))
    candidate: str | None = None
    failed_text = ""
    transfer_bytes = 0
    initial_calls = 0
    repair_calls = 0
    try:
        prompt = v11.rewrite_impl._initial_prompt(item)
        initial_calls = 1
        result = client.call(
            [{"role": "user", "content": prompt}], seed=INITIAL_REWRITE_SEED,
        )
        transfer_bytes += len(prompt) + len(result.content)
        candidate = v11.rewrite_impl._single_line_candidate(result.content)
        failed_text = result.content
    except (RuntimeError, TypeError, ValueError):
        candidate = None
    rewrite: str | None = None
    mode = "initial_no_single_line_candidate"
    if candidate is not None:
        rewrite, mode = v11.rewrite_impl.normalize_short_candidate(
            candidate, source_tokens,
        )
    if rewrite is None:
        try:
            prompt = _repair_prompt(item, failed_text)
            repair_calls = 1
            result = client.call(
                [{"role": "user", "content": prompt}], seed=REPAIR_REWRITE_SEED,
            )
            transfer_bytes += len(prompt) + len(result.content)
            candidate = v11.rewrite_impl._single_line_candidate(result.content)
        except (RuntimeError, TypeError, ValueError):
            candidate = None
        if candidate is None:
            mode = "repair_no_single_line_candidate"
        else:
            rewrite, mode = normalize_repaired_candidate(candidate, source_tokens)
    payload: dict[str, object] = {
        "substitute_sentence": rewrite or "",
        "in_length_window": rewrite is not None,
        "deviation_log": [mode],
        "generation_mode": mode,
        "source_label": item.label,
        "source_root": item.source_root,
    }
    return item.qid, payload, {
        "initial_calls": initial_calls,
        "repair_calls": repair_calls,
        "transfer_bytes": transfer_bytes,
    }


def build_substitutes_parallel(
    composites: Sequence[base.CompositeQuestion], *, cache_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    configure_v12()
    manifest_path = cache_dir / "substitute_manifest.json"
    stats_path = cache_dir / "substitute_generation_stats.json"
    expected_qids = {item.qid for comp in composites for item in comp.items}
    if manifest_path.exists() and stats_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(manifest) != expected_qids:
            raise ValueError("cached V12 substitutes differ from frozen evidence IDs")
        return manifest, json.loads(stats_path.read_text(encoding="utf-8"))

    question_shards = assign_shards(composites)

    def worker(shard_index: int) -> list[tuple[str, dict[str, object], dict[str, int]]]:
        shard_cache = cache_dir / "rewrite_shards" / f"shard_{shard_index}"
        client = CachedChatClient(base.DEFAULT_ENDPOINT, base.DEFAULT_MODEL, shard_cache)
        items = [item for comp in question_shards[shard_index] for item in comp.items]
        rows = []
        for index, item in enumerate(items, start=1):
            rows.append(_rewrite_one(item, client))
            if index % 50 == 0 or index == len(items):
                with _PRINT_LOCK:
                    print(
                        f"[v12 rewrite shard {shard_index}] {index}/{len(items)}",
                        flush=True,
                    )
        return rows

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(worker, index) for index in range(N_WORKERS)]
        for future in as_completed(futures):
            results.extend(future.result())
    manifest = {qid: payload for qid, payload, _usage in results}
    if set(manifest) != expected_qids:
        raise ValueError("V12 parallel rewrite merge lost or duplicated evidence IDs")
    modes = Counter(str(payload["generation_mode"]) for payload in manifest.values())
    stats: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "workers": N_WORKERS,
        "n_items": len(expected_qids),
        "initial_calls": sum(usage["initial_calls"] for _, _, usage in results),
        "repair_calls": sum(usage["repair_calls"] for _, _, usage in results),
        "transfer_bytes": sum(usage["transfer_bytes"] for _, _, usage in results),
        "generation_modes": dict(sorted(modes.items())),
        "n_rewritten": sum(bool(row["in_length_window"]) for row in manifest.values()),
        "n_unusable": sum(not bool(row["in_length_window"]) for row in manifest.values()),
    }
    stats["passed_fail_fast"] = stats["n_unusable"] == 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, dict(sorted(manifest.items())))
    _write_json(stats_path, stats)
    return manifest, stats


def run_audit(
    dataset_path: Path, selection_path: Path, output_path: Path, *,
    cache_dir: Path,
) -> dict[str, object]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    composites = validate_manifest_v12(
        selection, dataset_path, require_substitutes=False,
    )
    sub_path = cache_dir / "substitute_manifest.json"
    stats_path = cache_dir / "substitute_generation_stats.json"
    if not sub_path.exists() or not stats_path.exists():
        raise ValueError("V12 substitute artifacts are missing")
    substitutes = json.loads(sub_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("passed_fail_fast") is not True:
        raise ValueError("V12 auxiliary rewrite gate failed")
    manifest = build_manifest_v12(
        dataset_path, composites, substitutes, status="run_frozen",
    )
    validate_manifest_v12(manifest, dataset_path, require_substitutes=True)
    _write_json(output_path, manifest)
    shards = assign_shards(composites)
    audit = {
        "n_composites": len(composites),
        "labels": dict(Counter(comp.label for comp in composites)),
        "v11_1_root_overlap": 0,
        "n_evidence": sum(len(comp.items) for comp in composites),
        "n_substitutes": stats["n_rewritten"],
        "workers": N_WORKERS,
        "shard_questions": [len(shard) for shard in shards],
        "formal_logical_calls": FORMAL_EXAMPLES * base.N_AGENTS * len(base.CONDITIONS),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "risk_inputs": sorted(v11.RISK_WEIGHTS),
        "forbidden_risk_inputs": sorted(v11.FORBIDDEN_RISK_INPUTS),
        "passes": True,
    }
    return audit


def _record_key(record: Mapping[str, object]) -> tuple[str, int, str]:
    return str(record["cqid"]), int(record["agent_index"]), str(record["condition"])


def _worker_run(
    shard_index: int,
    composites: Sequence[base.CompositeQuestion],
    substitutes: Mapping[str, Mapping[str, str]],
    *,
    output_dir: Path,
    cache_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    shard_dir = output_dir / "shards" / f"shard_{shard_index}"
    shard_cache = cache_dir / "evaluation_shards" / f"shard_{shard_index}"
    partial_path = shard_dir / "records.partial.jsonl"
    progress_path = shard_dir / "progress.json"
    shard_dir.mkdir(parents=True, exist_ok=True)
    client = CachedChatClient(base.DEFAULT_ENDPOINT, base.DEFAULT_MODEL, shard_cache)
    records = base._load_partial_records(partial_path) if resume else []
    done = {_record_key(record) for record in records}
    if len(done) != len(records):
        raise ValueError(f"V12 shard {shard_index} partial contains duplicates")
    allowed = {
        (comp.cqid, agent, condition)
        for comp in composites
        for agent in range(base.N_AGENTS)
        for condition in base.CONDITIONS
    }
    if done - allowed:
        raise ValueError(f"V12 shard {shard_index} partial contains foreign tuples")
    total = len(allowed)
    started = time.time()
    for comp in composites:
        for agent_index in range(base.N_AGENTS):
            for condition in base.CONDITIONS:
                key = (comp.cqid, agent_index, condition)
                if key in done:
                    continue
                view = base.build_evidence_view(
                    comp, agent_index, condition, substitutes,
                )
                record = base.run_one_call(
                    client, comp, view, agent_index=agent_index,
                    substitute_manifest=substitutes,
                )
                records.append(record)
                done.add(key)
                elapsed = time.time() - started
                fresh = max(1, len(done))
                rate = fresh / elapsed if elapsed else 0.0
                eta = (total - len(done)) / rate if rate else float("inf")
                with _PRINT_LOCK:
                    print(
                        f"[v12 shard {shard_index} {len(done)}/{total}] "
                        f"{comp.cqid} {record['agent_id']} {condition} "
                        f"success={record['success']} eta={eta:.0f}s",
                        flush=True,
                    )
                _write_jsonl(partial_path, records)
                _write_json(progress_path, {
                    "protocol_version": PROTOCOL_VERSION,
                    "shard": shard_index,
                    "completed": len(done),
                    "total": total,
                    "eta_seconds": eta,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
    records.sort(key=_record_key)
    _write_jsonl(shard_dir / "records.jsonl", records)
    return records


def _smoke_composites(
    shards: Sequence[Sequence[base.CompositeQuestion]],
) -> list[list[base.CompositeQuestion]]:
    selected: list[list[base.CompositeQuestion]] = []
    for shard in shards:
        if not shard:
            raise ValueError("V12 smoke requires one question per shard")
        selected.append([shard[0]])
    return selected


def execute_parallel(
    *,
    mode: str,
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    cache_dir: Path,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, object], list[base.CompositeQuestion]]:
    configure_v12()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composites = validate_manifest_v12(manifest, dataset_path, require_substitutes=True)
    full_shards = assign_shards(composites)
    if mode == "smoke":
        run_shards = _smoke_composites(full_shards)
        expected_examples = SMOKE_EXAMPLES
        expected_calls = SMOKE_CALLS
    elif mode == "formal":
        run_shards = full_shards
        expected_examples = FORMAL_EXAMPLES
        expected_calls = FORMAL_EXAMPLES * base.N_AGENTS * len(base.CONDITIONS)
    else:
        raise ValueError("V12 mode must be smoke or formal")
    substitutes = manifest["substitute_manifest"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_records: list[list[dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                _worker_run,
                index,
                run_shards[index],
                substitutes,
                output_dir=output_dir,
                cache_dir=cache_dir,
                resume=resume,
            ): index
            for index in range(N_WORKERS)
        }
        for future in as_completed(futures):
            index = futures[future]
            rows = future.result()
            shard_records.append(rows)
            with _PRINT_LOCK:
                print(f"[v12 shard {index}] complete records={len(rows)}", flush=True)
    records = [record for rows in shard_records for record in rows]
    records.sort(key=_record_key)
    keys = [_record_key(record) for record in records]
    if len(records) != expected_calls or len(set(keys)) != expected_calls:
        raise ValueError(
            f"V12 merge integrity failed: records={len(records)}, unique={len(set(keys))}, "
            f"expected={expected_calls}"
        )
    protocol_counts = Counter(record.get("protocol_version") for record in records)
    condition_counts = Counter(record["condition"] for record in records)
    agent_counts = Counter(int(record["agent_index"]) for record in records)
    if protocol_counts != Counter({PROTOCOL_VERSION: expected_calls}):
        raise ValueError(f"V12 protocol counts drifted: {dict(protocol_counts)}")
    per_condition = expected_examples * base.N_AGENTS
    if condition_counts != Counter({condition: per_condition for condition in base.CONDITIONS}):
        raise ValueError(f"V12 condition counts drifted: {dict(condition_counts)}")
    per_agent = expected_examples * len(base.CONDITIONS)
    if agent_counts != Counter({index: per_agent for index in range(base.N_AGENTS)}):
        raise ValueError(f"V12 agent counts drifted: {dict(agent_counts)}")
    _write_jsonl(output_dir / "records.jsonl", records)
    failures = [record for record in records if not record.get("success")]
    if failures:
        _write_json(output_dir / "formal_abort.json", {
            "protocol_version": PROTOCOL_VERSION,
            "reason": "one_or_more_logical_calls_failed_frozen_two_attempt_contract",
            "failed_records": len(failures),
        })
        raise RuntimeError(f"V12 has {len(failures)} failed logical records")
    summary = v11.summarize_records_v11(
        records,
        mode=mode,
        expected_examples=expected_examples,
        agent_count=base.N_AGENTS,
        substitute_manifest=substitutes,
    )
    summary["parallel_execution"] = {
        "workers": N_WORKERS,
        "endpoint_concurrency_only": True,
        "physical_gpu_count_verified": False,
        "shard_record_counts": sorted(len(rows) for rows in shard_records),
        "merge_unique_records": len(set(keys)),
    }
    _write_json(output_dir / "summary.json", summary)
    report = v11.render_report_v11(summary).replace(
        "Pilot-LLM V11", "Pilot-LLM V12"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    for index in range(N_WORKERS):
        shard_dir = output_dir / "shards" / f"shard_{index}"
        partial = shard_dir / "records.partial.jsonl"
        progress = shard_dir / "progress.json"
        if partial.exists():
            partial.unlink()
        if progress.exists():
            progress.unlink()
    return records, summary, composites


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json",
    )
    rewrite = sub.add_parser("substitute-generation")
    rewrite.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    rewrite.add_argument(
        "--selection-manifest", type=Path,
        default=DEFAULT_ROOT / "selection_manifest.json",
    )
    rewrite.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    audit.add_argument(
        "--selection-manifest", type=Path,
        default=DEFAULT_ROOT / "selection_manifest.json",
    )
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    audit.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    for command in ("smoke", "run"):
        stage = sub.add_parser(command)
        stage.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        stage.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
        stage.add_argument(
            "--output-dir", type=Path,
            default=DEFAULT_ROOT / ("smoke" if command == "smoke" else "formal"),
        )
        stage.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
        stage.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_v12()
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_selection(args.output, args.dataset)
        print(f"{'Wrote' if created else 'Reused'} frozen V12 selection: {args.output}")
        return 0
    if args.command == "substitute-generation":
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        composites = validate_manifest_v12(
            selection, args.dataset, require_substitutes=False,
        )
        _manifest, stats = build_substitutes_parallel(
            composites, cache_dir=args.cache_dir,
        )
        for key, value in stats.items():
            print(f"[v12 substitute] {key}: {value}")
        return 0 if stats["passed_fail_fast"] else 2
    if args.command == "audit":
        result = run_audit(
            args.dataset,
            args.selection_manifest,
            args.output,
            cache_dir=args.cache_dir,
        )
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0
    if args.command in {"smoke", "run"}:
        execute_parallel(
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
