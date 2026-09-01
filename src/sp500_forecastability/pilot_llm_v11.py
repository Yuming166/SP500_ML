"""Pilot-LLM V11 held-out BoolQ validation confirmation.

V11 freezes one outcome-independent development-selected score before any
validation model output.  See ``docs/pilot_llm_v11_preregistration.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from sp500_forecastability import pilot_llm_v10 as base
from sp500_forecastability import pilot_llm_v10_4 as rewrite_impl
from sp500_forecastability.pilot_llm_v1 import (
    CachedChatClient,
    _write_json,
    _write_jsonl,
)

PROTOCOL_VERSION = "pilot-llm-v11-2026-09-01"
SALT = b"pilot-llm-v11-2026-09-01\n"
DEFAULT_DATASET = Path(
    "/storage/gaoym/sp500-forecastability-lab/data/boolq/validation.parquet"
)
DEFAULT_ROOT = Path("results/pilot_llm_v11")
DEVELOPMENT_RECORDS = Path("results/pilot_llm_v10_4/formal/records.jsonl")
FORMAL_EXAMPLES = 200
FORMAL_PER_LABEL = 100
BOOTSTRAP_SEED = 20_260_902
BOOTSTRAP_REPLICATES = 1_000
PRIMARY_MIN_HIGH_CONSENSUS = 80
PRIMARY_MIN_PER_CLASS = 10
HIGH_CONSENSUS_THRESHOLD = 0.8
ROUTER_COVERAGE = 0.8
INITIAL_REWRITE_SEED = 20_260_911
RISK_WEIGHTS = {
    "D_inert": 0.1,
    "flip_inertia": 0.3,
    "frac_shared": 0.6,
}
FORBIDDEN_RISK_INPUTS = {
    "label", "gold_binary", "correct", "any_wrong", "harmful_fc",
}


def configure_base() -> None:
    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.SALT = SALT
    base.CQID_PROTOCOL_VERSION = PROTOCOL_VERSION
    base.CQID_PREFIX = "v11"
    base.FORMAL_EXAMPLES = FORMAL_EXAMPLES
    base.FORMAL_PER_LABEL = FORMAL_PER_LABEL
    base.DEFAULT_DATASET = DEFAULT_DATASET
    base.DEFAULT_ROOT = DEFAULT_ROOT
    base.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    base.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
    rewrite_impl.PROTOCOL_VERSION = PROTOCOL_VERSION
    rewrite_impl.DEFAULT_ROOT = DEFAULT_ROOT
    rewrite_impl.INITIAL_REWRITE_SEED = INITIAL_REWRITE_SEED


def risk_score_from_preoutcome(row: Mapping[str, object]) -> float:
    """Compute the frozen score while deliberately ignoring outcome fields."""
    return float(
        RISK_WEIGHTS["D_inert"] * float(row["D_inert"])
        + RISK_WEIGHTS["flip_inertia"] * float(row["flip_inertia"])
        + RISK_WEIGHTS["frac_shared"] * float(row["frac_shared"])
    )


def build_manifest_v11(
    dataset_path: Path,
    composites: Sequence[base.CompositeQuestion],
    substitute_manifest: Mapping[str, Mapping[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    configure_base()
    manifest = base.build_manifest(
        dataset_path, composites, substitute_manifest, status=status,
    )
    manifest["selection"]["dataset_split"] = "validation"
    manifest["selection"]["development_split_excluded"] = "train"
    manifest["development_contract"] = {
        "records": str(DEVELOPMENT_RECORDS),
        "role": "feature_weight_selection_only",
        "grid": "66 convex combinations on 0.1 simplex",
        "selected_development_high_consensus_auroc": 0.7338709677419355,
    }
    manifest["co_primary_endpoints"] = []
    manifest["primary_endpoint"] = (
        "AUROC(R_PI,consensus_wrong|original_agreement>=0.8)"
    )
    manifest["secondary_endpoints"] = [
        "D_inert_high_consensus", "flip_inertia_high_consensus",
        "frac_shared_high_consensus", "R_PI_harmful_fc_all",
        "rank_router_coverage_0.8",
    ]
    manifest["risk_contract"] = {
        "name": "R_PI",
        "weights": dict(RISK_WEIGHTS),
        "allowed_inputs": sorted(RISK_WEIGHTS),
        "forbidden_inputs": sorted(FORBIDDEN_RISK_INPUTS),
        "outcome_independent": True,
        "high_consensus_threshold": HIGH_CONSENSUS_THRESHOLD,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "minimum_high_consensus": PRIMARY_MIN_HIGH_CONSENSUS,
        "minimum_per_class": PRIMARY_MIN_PER_CLASS,
    }
    manifest["smoke_contract"] = {
        "examples": 2, "agents": base.N_AGENTS,
        "conditions": len(base.CONDITIONS), "logical_calls": 40,
    }
    return manifest


def validate_manifest_v11(
    manifest: Mapping[str, object], dataset_path: Path, *,
    require_substitutes: bool = True,
) -> list[base.CompositeQuestion]:
    configure_base()
    composites = base.validate_manifest(
        manifest, dataset_path, require_substitutes=require_substitutes,
    )
    if manifest.get("primary_endpoint") != (
        "AUROC(R_PI,consensus_wrong|original_agreement>=0.8)"
    ):
        raise ValueError("V11 primary endpoint drifted")
    if manifest.get("co_primary_endpoints") != []:
        raise ValueError("V11 must have exactly one primary endpoint")
    contract = manifest.get("risk_contract", {})
    if not isinstance(contract, Mapping):
        raise TypeError("V11 risk contract is missing")
    if contract.get("weights") != RISK_WEIGHTS:
        raise ValueError("V11 risk weights drifted")
    if contract.get("allowed_inputs") != sorted(RISK_WEIGHTS):
        raise ValueError("V11 allowed risk inputs drifted")
    if contract.get("outcome_independent") is not True:
        raise ValueError("V11 risk score must be outcome independent")
    forbidden = set(contract.get("forbidden_inputs", []))
    if forbidden != FORBIDDEN_RISK_INPUTS:
        raise ValueError("V11 forbidden risk inputs drifted")
    if manifest.get("smoke_contract", {}).get("logical_calls") != 40:
        raise ValueError("V11 smoke contract must contain 40 calls")
    return composites


def write_or_validate_selection(
    output_path: Path, dataset_path: Path,
) -> bool:
    configure_base()
    items = base.load_boolq(dataset_path)
    composites = base.build_composite_questions(items)
    expected = build_manifest_v11(
        dataset_path, composites, {},
        status="selection_frozen_before_v11_model_calls",
    )
    validate_manifest_v11(expected, dataset_path, require_substitutes=False)
    if output_path.exists():
        actual = json.loads(output_path.read_text(encoding="utf-8"))
        validate_manifest_v11(actual, dataset_path, require_substitutes=False)
        if actual != expected:
            raise ValueError("existing V11 selection differs from frozen design")
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, expected)
    return True


def build_substitutes(
    items: Sequence[base.BoolQItem], *, cache_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, Any]]:
    configure_base()
    return rewrite_impl.build_substitute_manifest_v10_4(
        items, cache_dir=cache_dir,
    )


def run_audit(
    dataset_path: Path, selection_path: Path, output_path: Path, *,
    cache_dir: Path,
) -> dict[str, object]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    composites = validate_manifest_v11(
        selection, dataset_path, require_substitutes=False,
    )
    sub_path = cache_dir / "substitute_manifest.json"
    if not sub_path.exists():
        raise ValueError("V11 substitute manifest is missing")
    substitutes = json.loads(sub_path.read_text(encoding="utf-8"))
    expected_qids = {item.qid for comp in composites for item in comp.items}
    if set(substitutes) != expected_qids:
        raise ValueError("V11 substitutes differ from exactly 600 frozen items")
    manifest = build_manifest_v11(
        dataset_path, composites, substitutes, status="run_frozen",
    )
    validate_manifest_v11(manifest, dataset_path, require_substitutes=True)
    _write_json(output_path, manifest)
    audit = {
        "n_composites": len(composites),
        "balance": dict(Counter(comp.label for comp in composites)),
        "n_evidence": len(expected_qids),
        "n_substitutes": sum(
            bool(value.get("substitute_sentence"))
            and bool(value.get("in_length_window"))
            for value in substitutes.values()
        ),
        "risk_inputs": sorted(RISK_WEIGHTS),
        "forbidden_risk_inputs": sorted(FORBIDDEN_RISK_INPUTS),
        "passes": True,
    }
    return audit


def _risk_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    grouped = base._group_by_question(records)
    base_rows = {
        row["cqid"]: row for row in base._per_question_risks(grouped)
    }
    rows: list[dict[str, Any]] = []
    for cqid, recs in sorted(grouped.items()):
        if cqid not in base_rows:
            continue
        signals = [base._agent_signal(group) for group in base._group_agents(recs)]
        if len(signals) != base.N_AGENTS or any(
            not signal.get("complete") for signal in signals
        ):
            continue
        flip_count = sum(
            sum(int(value) for value in signal["flips"].values())
            for signal in signals
        )
        flip_inertia = 1.0 - flip_count / (base.N_AGENTS * 3)
        source = base_rows[cqid]
        preoutcome = {
            "D_inert": float(source["D_inert"]),
            "flip_inertia": float(flip_inertia),
            "frac_shared": float(source["frac_shared"]),
        }
        rows.append({
            "cqid": cqid,
            **preoutcome,
            "R_PI": risk_score_from_preoutcome(preoutcome),
            "agreement": float(source["agreement"]),
            "consensus_wrong": int(source["any_wrong"]),
            "harmful_fc": int(source["harmful_fc"]),
        })
    return rows


def _metric(rows: Sequence[Mapping[str, object]], field: str, target: str) -> dict[str, object]:
    scores = [float(row[field]) for row in rows]
    labels = [int(row[target]) for row in rows]
    return {
        "auroc": base._auroc(scores, labels),
        "auroc_ci": list(base._per_question_metric_bootstrap(
            base._auroc, rows, field, target,
            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
        )),
        "auprc": base._safe_auprc(scores, labels),
        "auprc_ci": list(base._per_question_metric_bootstrap(
            lambda score, label: base._safe_auprc(score, label),
            rows, field, target,
            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
        )),
        "n": len(rows),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
    }


def _selective_router(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def evaluate(sample: Sequence[Mapping[str, object]]) -> tuple[float, float, float]:
        baseline = sum(int(row["consensus_wrong"]) for row in sample) / len(sample)
        keep_n = max(1, round(len(sample) * ROUTER_COVERAGE))
        kept = sorted(sample, key=lambda row: float(row["R_PI"]))[:keep_n]
        routed = sum(int(row["consensus_wrong"]) for row in kept) / len(kept)
        return baseline, routed, baseline - routed

    if not rows:
        return {"n": 0, "coverage": ROUTER_COVERAGE}
    baseline, routed, improvement = evaluate(rows)
    rng = random.Random(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        boot = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        samples.append(evaluate(boot)[2])
    samples.sort()
    return {
        "n": len(rows), "coverage": ROUTER_COVERAGE,
        "baseline_error": baseline, "routed_error": routed,
        "error_reduction": improvement,
        "error_reduction_ci": [samples[25], samples[975]],
    }


def summarize_records_v11(
    records: Sequence[Mapping[str, object]], *, mode: str,
    expected_examples: int, agent_count: int,
    substitute_manifest: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows = _risk_rows(records)
    high = [row for row in rows if row["agreement"] >= HIGH_CONSENSUS_THRESHOLD]
    primary = _metric(high, "R_PI", "consensus_wrong")
    ci_lo = primary["auroc_ci"][0]
    count_gate = (
        len(high) >= PRIMARY_MIN_HIGH_CONSENSUS
        and primary["positives"] >= PRIMARY_MIN_PER_CLASS
        and primary["negatives"] >= PRIMARY_MIN_PER_CLASS
    )
    passed = bool(
        count_gate and isinstance(ci_lo, (int, float))
        and isfinite(float(ci_lo)) and float(ci_lo) > 0.5
    )
    secondary = {
        f"{field}__wrong_high_consensus": _metric(
            high, field, "consensus_wrong",
        )
        for field in ("D_inert", "flip_inertia", "frac_shared")
    }
    secondary["R_PI__harmful_fc_all"] = _metric(
        rows, "R_PI", "harmful_fc",
    )
    valid_records = [record for record in records if record.get("success")]
    transfer_bytes = sum(
        int(attempt.get("request_bytes", 0) or 0)
        + int(attempt.get("response_bytes", 0) or 0)
        for record in records for attempt in record.get("attempts", [])
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "expected_examples": expected_examples,
        "agent_count": agent_count,
        "expected_calls": expected_examples * agent_count * len(base.CONDITIONS),
        "records": len(records),
        "instrumentation": {
            "valid_records": len(valid_records),
            "valid_rate": len(valid_records) / max(1, len(records)),
            "first_pass_valid_rate": sum(
                bool(record.get("first_pass_valid")) for record in records
            ) / max(1, len(records)),
            "transfer_bytes": transfer_bytes,
        },
        "primary": {
            "name": "R_PI_wrong_high_consensus",
            **primary,
            "count_gate_passed": count_gate,
            "passes_ci_lower_above_0_5": passed,
            "verdict": "PASS" if passed else (
                "STRUCTURAL_BOUNDARY" if not count_gate else "FAIL"
            ),
        },
        "high_consensus": {
            "threshold": HIGH_CONSENSUS_THRESHOLD,
            "n": len(high),
            "fraction": len(high) / max(1, len(rows)),
            "wrong": sum(int(row["consensus_wrong"]) for row in high),
            "wrong_prevalence": sum(
                int(row["consensus_wrong"]) for row in high
            ) / max(1, len(high)),
        },
        "secondary": secondary,
        "router": _selective_router(high),
        "intervention_flip_rates": base._intervention_flip_rates(
            base._group_by_question(records)
        ),
        "risk_contract": {
            "weights": dict(RISK_WEIGHTS),
            "allowed_inputs": sorted(RISK_WEIGHTS),
            "forbidden_inputs": sorted(FORBIDDEN_RISK_INPUTS),
            "outcome_independent": True,
        },
        "substitute_summary": dict(Counter(
            str(value.get("generation_mode"))
            for value in substitute_manifest.values()
        )),
    }


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report_v11(summary: Mapping[str, object]) -> str:
    inst = summary["instrumentation"]
    primary = summary["primary"]
    high = summary["high_consensus"]
    lines = [f"# Pilot-LLM V11 {summary['mode']} report", ""]
    lines.extend([
        "## Integrity", "",
        f"- Records: {summary['records']} / {summary['expected_calls']}",
        f"- Valid rate: {_fmt(inst['valid_rate'])}",
        f"- First-pass valid rate: {_fmt(inst['first_pass_valid_rate'])}",
        "",
        "## Single confirmatory endpoint", "",
        f"- Verdict: **{primary['verdict']}**",
        f"- AUROC: {_fmt(primary['auroc'])}",
        f"- 95% CI: [{_fmt(primary['auroc_ci'][0])}, {_fmt(primary['auroc_ci'][1])}]",
        f"- Count gate passed: {primary['count_gate_passed']}",
        f"- High-consensus N: {high['n']} ({_fmt(high['fraction'])})",
        f"- Wrong high consensus: {high['wrong']} ({_fmt(high['wrong_prevalence'])})",
        "",
        "## Frozen pre-outcome score", "",
        "`R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared`",
        "",
        "## Secondary metrics", "",
    ])
    for name, metric in summary["secondary"].items():
        lines.append(
            f"- {name}: AUROC {_fmt(metric['auroc'])} "
            f"[{_fmt(metric['auroc_ci'][0])}, {_fmt(metric['auroc_ci'][1])}]"
        )
    router = summary["router"]
    lines.extend([
        "", "## Rank router at 80% coverage", "",
        f"- Baseline high-consensus error: {_fmt(router.get('baseline_error'))}",
        f"- Routed retained error: {_fmt(router.get('routed_error'))}",
        f"- Error reduction: {_fmt(router.get('error_reduction'))}",
        f"- Error-reduction CI: {_fmt(router.get('error_reduction_ci'))}",
        "", "## Interpretation boundary", "",
        (
            "This held-out result confirms or rejects only the frozen BoolQ validation "
            "paired-intervention score for this model and evidence regime. It does not "
            "establish cross-model, cross-domain, financial, or general factuality claims."
        ),
    ])
    return "\n".join(lines) + "\n"


def execute_run_v11(
    *, mode: str, dataset_path: Path, manifest_path: Path,
    output_dir: Path, cache_dir: Path, smoke_examples: int = 2,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, object], list[base.CompositeQuestion]]:
    configure_base()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composites = validate_manifest_v11(manifest, dataset_path)
    if mode == "smoke":
        if smoke_examples != 2:
            raise ValueError("V11 smoke requires exactly two questions")
        composites = composites[:smoke_examples]
    elif mode == "formal":
        if len(composites) != FORMAL_EXAMPLES:
            raise ValueError("V11 formal requires exactly 200 questions")
    else:
        raise ValueError("mode must be smoke or formal")
    agents = list(range(base.N_AGENTS))
    substitutes = manifest["substitute_manifest"]
    client = CachedChatClient(base.DEFAULT_ENDPOINT, base.DEFAULT_MODEL, cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "records.partial.jsonl"
    progress_path = output_dir / "progress.json"
    records: list[dict[str, Any]] = []
    done: set[tuple[str, int, str]] = set()
    if resume:
        records = base._load_partial_records(partial_path)
        done = {
            (str(record["cqid"]), int(record["agent_index"]), str(record["condition"]))
            for record in records
        }
        if len(done) != len(records):
            raise ValueError("V11 partial records contain duplicate tuples")
        if done:
            print(f"[resume] loaded {len(done)} completed tuples", flush=True)
    total = len(composites) * len(agents) * len(base.CONDITIONS)
    allowed = {
        (comp.cqid, agent, condition)
        for comp in composites for agent in agents for condition in base.CONDITIONS
    }
    if done - allowed:
        raise ValueError("V11 partial records contain tuples outside this run")
    started = time.time()
    for comp in composites:
        for agent_index in agents:
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
                rate = len(done) / elapsed if elapsed else 0.0
                eta = (total - len(done)) / rate if rate else float("inf")
                print(
                    f"[{len(done)}/{total}] {comp.cqid} "
                    f"{record['agent_id']} {condition} "
                    f"success={record['success']} rate={rate:.1f}/s eta={eta:.0f}s",
                    flush=True,
                )
                _write_jsonl(partial_path, records)
                _write_json(progress_path, {
                    "mode": mode, "completed": len(done), "total": total,
                    "rate_per_second": rate, "eta_seconds": eta,
                    "last_cqid": comp.cqid, "last_agent": record["agent_id"],
                    "last_condition": condition,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
    summary = summarize_records_v11(
        records, mode=mode, expected_examples=len(composites),
        agent_count=len(agents), substitute_manifest=substitutes,
    )
    _write_jsonl(output_dir / "records.jsonl", records)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_report_v11(summary), encoding="utf-8",
    )
    if partial_path.exists():
        partial_path.unlink()
    if progress_path.exists():
        progress_path.unlink()
    return records, summary, composites


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prep.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg = sub.add_parser("substitute-generation")
    subg.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    subg.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    audit.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    audit.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    for command in ("smoke", "run"):
        stage = sub.add_parser(command)
        stage.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        stage.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
        stage.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / ("smoke" if command == "smoke" else "formal"))
        stage.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
        stage.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_base()
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_selection(args.output, args.dataset)
        print(f"{'Wrote' if created else 'Reused'} frozen V11 selection: {args.output}")
        return 0
    if args.command == "substitute-generation":
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        comps = validate_manifest_v11(selection, args.dataset, require_substitutes=False)
        items = [item for comp in comps for item in comp.items]
        _manifest, stats = build_substitutes(items, cache_dir=args.cache_dir)
        for key, value in stats.items():
            print(f"[v11 substitute] {key}: {value}")
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
        execute_run_v11(
            mode="smoke" if args.command == "smoke" else "formal",
            dataset_path=args.dataset, manifest_path=args.manifest,
            output_dir=args.output_dir, cache_dir=args.cache_dir,
            resume=not args.no_resume,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
