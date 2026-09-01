"""Pilot-LLM V11.1 auxiliary length-repair amendment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from sp500_forecastability import pilot_llm_v11 as v11
from sp500_forecastability.pilot_llm_v1 import CachedChatClient, _write_json

PROTOCOL_VERSION = "pilot-llm-v11.1-2026-09-01"
SALT = b"pilot-llm-v11.1-2026-09-01\n"
DEFAULT_ROOT = Path("results/pilot_llm_v11_1")
PARENT_ROOT = Path("results/pilot_llm_v11")
PARENT_SELECTION = PARENT_ROOT / "selection_manifest.json"
PARENT_SUBSTITUTES = PARENT_ROOT / "cache" / "substitute_manifest.json"
REPAIR_SEED = 20_260_912
EXPECTED_REPAIR_QIDS = {
    "boolq-1e583402fdb107ef-e01",
    "boolq-095374f01188fd3e-e01",
    "boolq-2e961908ed018a35-e02",
}


def configure_v11_1() -> None:
    v11.PROTOCOL_VERSION = PROTOCOL_VERSION
    v11.SALT = SALT
    v11.DEFAULT_ROOT = DEFAULT_ROOT
    v11.configure_base()


def _lineage(path: Path) -> dict[str, object]:
    return {
        "parent_protocol": "pilot-llm-v11-2026-09-01",
        "parent_path": str(path),
        "parent_sha256": v11.base.file_sha256(path),
        "parent_has_validation_agent_outputs": False,
        "selection_changed": False,
    }


def build_inherited_selection(parent_path: Path, dataset_path: Path) -> dict[str, object]:
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("protocol_version") != "pilot-llm-v11-2026-09-01":
        raise ValueError("V11.1 requires the frozen V11 selection parent")
    manifest = deepcopy(parent)
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["status"] = "selection_inherited_from_v11_preformal_abort"
    manifest["selection_parent"] = _lineage(parent_path)
    manifest["substitute_manifest"] = {}
    configure_v11_1()
    v11.validate_manifest_v11(manifest, dataset_path, require_substitutes=False)
    return manifest


def write_or_validate_selection(
    output_path: Path, parent_path: Path, dataset_path: Path,
) -> bool:
    expected = build_inherited_selection(parent_path, dataset_path)
    if output_path.exists():
        actual = json.loads(output_path.read_text(encoding="utf-8"))
        v11.validate_manifest_v11(actual, dataset_path, require_substitutes=False)
        if actual != expected:
            raise ValueError("existing V11.1 selection differs from frozen V11 parent")
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, expected)
    return True


def _repair_prompt(item: v11.base.BoolQItem, failed: str) -> str:
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
        f"Unusable candidate: {failed.strip()}"
    )


def normalize_repaired_candidate(
    candidate: str, source_tokens: int,
) -> tuple[str | None, str]:
    count = len(candidate.split())
    lower = max(1, (source_tokens + 1) // 2)
    upper = max(1, (3 * source_tokens) // 2)
    if lower <= count <= upper:
        return candidate, "v11_1_repair"
    if count > upper:
        return None, "v11_1_repair_overlong"
    stem = candidate.rstrip().rstrip(".!?").rstrip()
    suffix = v11.rewrite_impl.NEUTRAL_QUALIFIER
    repeats = 0
    while len(stem.split()) < lower:
        proposed = f"{stem} {suffix}".strip()
        if len(proposed.split()) > upper:
            return None, "v11_1_repair_short_unfixable"
        stem = proposed.rstrip().rstrip(".!?").rstrip()
        repeats += 1
    return f"{stem}.", f"v11_1_repair_neutral_suffix_x{repeats}"


def build_repaired_substitutes(
    items: Sequence[v11.base.BoolQItem], *, parent_substitutes_path: Path,
    cache_dir: Path, client: CachedChatClient | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "substitute_manifest.json"
    stats_path = cache_dir / "substitute_generation_stats.json"
    expected_qids = {item.qid for item in items}
    if output_path.exists() and stats_path.exists():
        manifest = json.loads(output_path.read_text(encoding="utf-8"))
        if set(manifest) != expected_qids:
            raise ValueError("cached V11.1 substitutes differ from frozen selection")
        return manifest, json.loads(stats_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_substitutes_path.read_text(encoding="utf-8"))
    if set(parent) != expected_qids:
        raise ValueError("V11 parent substitutes differ from frozen selection")
    invalid = {
        qid for qid, value in parent.items()
        if not value.get("substitute_sentence") or not value.get("in_length_window")
    }
    if invalid != EXPECTED_REPAIR_QIDS:
        raise ValueError(f"V11.1 repair set drifted: {sorted(invalid)}")
    if client is None:
        client = CachedChatClient(
            v11.base.DEFAULT_ENDPOINT, v11.base.DEFAULT_MODEL, cache_dir,
        )
    by_qid = {item.qid: item for item in items}
    manifest = deepcopy(parent)
    stats: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "n_items": len(items), "parent_valid_inherited": len(items) - len(invalid),
        "repair_attempted": 0, "repair_valid": 0, "n_unusable": 0,
        "parent_substitutes_sha256": v11.base.file_sha256(parent_substitutes_path),
    }
    for qid in sorted(invalid):
        item = by_qid[qid]
        failed = str(parent[qid].get("substitute_sentence") or "")
        if not failed:
            # Recover the frozen one-line response from the V11 diagnostic field
            # is deliberately forbidden; an empty parent requires a fresh repair
            # from the original evidence alone.
            failed = "[no usable prior candidate]"
        prompt = _repair_prompt(item, failed)
        stats["repair_attempted"] = int(stats["repair_attempted"]) + 1
        try:
            result = client.call(
                [{"role": "user", "content": prompt}], seed=REPAIR_SEED,
            )
            candidate = v11.rewrite_impl._single_line_candidate(result.content)
        except (RuntimeError, TypeError, ValueError):
            candidate = None
        rewrite: str | None = None
        mode = "v11_1_repair_failed"
        if candidate:
            rewrite, mode = normalize_repaired_candidate(
                candidate, max(1, len(item.passage.split())),
            )
        if rewrite is None:
            stats["n_unusable"] = int(stats["n_unusable"]) + 1
            manifest[qid] = {
                "substitute_sentence": "", "in_length_window": False,
                "generation_mode": mode, "deviation_log": [mode],
                "source_label": item.label, "source_root": item.source_root,
            }
        else:
            stats["repair_valid"] = int(stats["repair_valid"]) + 1
            manifest[qid] = {
                "substitute_sentence": rewrite, "in_length_window": True,
                "generation_mode": mode, "deviation_log": [mode],
                "source_label": item.label, "source_root": item.source_root,
            }
    stats["n_rewritten"] = sum(
        bool(value.get("substitute_sentence"))
        and bool(value.get("in_length_window"))
        for value in manifest.values()
    )
    stats["passed_fail_fast"] = stats["n_rewritten"] == len(items)
    _write_json(output_path, manifest)
    _write_json(stats_path, stats)
    return manifest, stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, default=v11.DEFAULT_DATASET)
    prep.add_argument("--parent", type=Path, default=PARENT_SELECTION)
    prep.add_argument("--output", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg = sub.add_parser("substitute-generation")
    subg.add_argument("--dataset", type=Path, default=v11.DEFAULT_DATASET)
    subg.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    subg.add_argument("--parent-substitutes", type=Path, default=PARENT_SUBSTITUTES)
    subg.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", type=Path, default=v11.DEFAULT_DATASET)
    audit.add_argument("--selection-manifest", type=Path, default=DEFAULT_ROOT / "selection_manifest.json")
    audit.add_argument("--output", type=Path, default=DEFAULT_ROOT / "manifest.json")
    audit.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
    for command in ("smoke", "run"):
        stage = sub.add_parser(command)
        stage.add_argument("--dataset", type=Path, default=v11.DEFAULT_DATASET)
        stage.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
        stage.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / ("smoke" if command == "smoke" else "formal"))
        stage.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT / "cache")
        stage.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_v11_1()
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        created = write_or_validate_selection(
            args.output, args.parent, args.dataset,
        )
        print(f"{'Wrote' if created else 'Reused'} frozen V11.1 selection: {args.output}")
        return 0
    if args.command == "substitute-generation":
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        comps = v11.validate_manifest_v11(
            selection, args.dataset, require_substitutes=False,
        )
        items = [item for comp in comps for item in comp.items]
        _manifest, stats = build_repaired_substitutes(
            items, parent_substitutes_path=args.parent_substitutes,
            cache_dir=args.cache_dir,
        )
        for key, value in stats.items():
            print(f"[v11.1 substitute] {key}: {value}")
        return 0 if stats["passed_fail_fast"] else 2
    if args.command == "audit":
        audit = v11.run_audit(
            args.dataset, args.selection_manifest, args.output,
            cache_dir=args.cache_dir,
        )
        selection = json.loads(
            args.selection_manifest.read_text(encoding="utf-8")
        )
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        manifest["selection"] = selection["selection"]
        manifest["selection_parent"] = selection["selection_parent"]
        manifest["auxiliary_lineage"] = {
            "v11_valid_substitutes_inherited": 597,
            "v11_repair_items": sorted(EXPECTED_REPAIR_QIDS),
            "validation_agent_outputs_existed_before_repair": False,
        }
        v11.validate_manifest_v11(manifest, args.dataset)
        _write_json(args.output, manifest)
        for key, value in audit.items():
            print(f"{key}: {value}")
        return 0
    if args.command in {"smoke", "run"}:
        v11.execute_run_v11(
            mode="smoke" if args.command == "smoke" else "formal",
            dataset_path=args.dataset, manifest_path=args.manifest,
            output_dir=args.output_dir, cache_dir=args.cache_dir,
            resume=not args.no_resume,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
