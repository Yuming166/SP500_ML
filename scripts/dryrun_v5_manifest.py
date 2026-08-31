"""V5 manifest dry-run.

Local-only check: verifies that the FEVER validation set has enough
clusters and length diversity to support V5's pre-registered manifest
construction (§4.3 of docs/pilot_llm_v5_preregistration.md). Does NOT
make any LLM calls.

Usage:
    python scripts/dryrun_v5_manifest.py
    python scripts/dryrun_v5_manifest.py --json   # machine-readable output

Outputs the following pre-registered gates:

  G1: how many clusters have ≥3 SUPPORTS and ≥3 REFUTES rows each
  G2: total C(n,3) composite budget per label after NEI exclusion
  G3: whether salt-sorted top-25+25 is feasible
  G4: evidence sentence length distribution + ±50% window hit-rate
      within cluster
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Pre-registered V5 constants
SALT = b"pilot-llm-v5-2026-08-31\n"
TARGET_N_PER_LABEL = 25
COMPOSITE_ROWS = 3
LENGTH_WINDOW_PCT = 0.50
DATA_PATH = Path("/storage/gaoym/sp500-forecastability-lab/data/fever/fever-validation.jsonl")


def _sha_sort_key(qid: str) -> str:
    """Pre-registered salt sort key for a row qid."""
    h = hashlib.sha256(SALT + qid.encode()).hexdigest()
    return h


def _entity_of(row: dict) -> str | None:
    """Cluster key = first evidence entity, or fallback to claim-id."""
    ev = row.get("evidence") or []
    if ev and isinstance(ev[0], list) and ev[0]:
        return ev[0][0]
    return row.get("id")


def _evidence_sentence(row: dict) -> str | None:
    """First evidence sentence (V5 §4.2 says use first annotated)."""
    ev = row.get("evidence") or []
    if ev and isinstance(ev[0], list) and len(ev[0]) >= 3:
        return ev[0][2]
    return None


def _tokens(s: str) -> list[str]:
    return s.split()


def load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    rows = load(args.data)
    n_total = len(rows)
    label_counts = Counter(r.get("label") for r in rows)

    # NEI exclusion (§4.1)
    binary_rows = [r for r in rows if r.get("label") in ("SUPPORTS", "REFUTES")]
    n_binary = len(binary_rows)

    # Group by entity
    by_entity: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"SUPPORTS": [], "REFUTES": []}
    )
    for r in binary_rows:
        ent = _entity_of(r)
        if ent is None:
            continue
        label = r["label"]
        by_entity[ent][label].append(r)
    n_clusters = len(by_entity)

    # Gate G1: clusters with ≥3 SUPPORTS AND ≥3 REFUTES
    clusters_dual = [e for e, d in by_entity.items()
                     if len(d["SUPPORTS"]) >= COMPOSITE_ROWS
                     and len(d["REFUTES"]) >= COMPOSITE_ROWS]
    clusters_sup_only = [e for e, d in by_entity.items()
                         if len(d["SUPPORTS"]) >= COMPOSITE_ROWS
                         and len(d["REFUTES"]) < COMPOSITE_ROWS]
    clusters_ref_only = [e for e, d in by_entity.items()
                         if len(d["REFUTES"]) >= COMPOSITE_ROWS
                         and len(d["SUPPORTS"]) < COMPOSITE_ROWS]

    # Gate G2: composite budget (C(n,3) per cluster per label)
    def nCr(n: int, r: int) -> int:
        if n < r:
            return 0
        return math.comb(n, r)

    budget_sup = sum(nCr(len(d["SUPPORTS"]), COMPOSITE_ROWS)
                     for d in by_entity.values())
    budget_ref = sum(nCr(len(d["REFUTES"]), COMPOSITE_ROWS)
                     for d in by_entity.values())

    # Gate G3: salt-sorted top-25+25 feasibility
    # Correct pre-registered interpretation of V5 §4.3:
    #   step 1: per-cluster, salt-sort rows of one label
    #   step 2: per-cluster, every COMPOSITE_ROWS consecutive rows form a composite
    #   step 3: collect all composites, salt-sort them by their cluster qid,
    #           take the first TARGET_N_PER_LABEL
    def collect_composites(label: str) -> tuple[list[list[dict]], list[str]]:
        per_cluster: dict[str, list[dict]] = defaultdict(list)
        for r in binary_rows:
            if r.get("label") != label:
                continue
            ent = _entity_of(r)
            if ent is None:
                continue
            per_cluster[ent].append(r)
        composites: list[list[dict]] = []
        unusable: list[str] = []
        for ent, members in per_cluster.items():
            members_sorted = sorted(members, key=lambda r: _sha_sort_key(r["id"]))
            for i in range(0, len(members_sorted) - COMPOSITE_ROWS + 1, COMPOSITE_ROWS):
                composites.append(members_sorted[i : i + COMPOSITE_ROWS])
            leftover = len(members_sorted) % COMPOSITE_ROWS
            if 0 < leftover < COMPOSITE_ROWS:
                unusable.append(ent)
        # salt-sort composites by the smallest member qid, then keep top N
        composites.sort(key=lambda c: min(_sha_sort_key(r["id"]) for r in c))
        return composites, unusable

    sup_composites, unsup_ents = collect_composites("SUPPORTS")
    ref_composites, unref_ents = collect_composites("REFUTES")

    top_sup = sup_composites[:TARGET_N_PER_LABEL]
    top_ref = ref_composites[:TARGET_N_PER_LABEL]
    formed_sup = len(top_sup)
    formed_ref = len(top_ref)
    unsup_cnt = len(unsup_ents)
    unref_cnt = len(unref_ents)

    # Gate G4: length distribution + within-cluster ±50% window feasibility
    sent_lengths: list[int] = []
    for r in binary_rows:
        s = _evidence_sentence(r)
        if s is not None:
            sent_lengths.append(len(_tokens(s)))

    # Within-cluster length-window feasibility
    window_hits = 0
    window_misses = 0
    for ent, d in by_entity.items():
        all_rows = d["SUPPORTS"] + d["REFUTES"]
        if len(all_rows) < 2:
            continue
        for r in all_rows:
            s = _evidence_sentence(r)
            if s is None:
                continue
            L = len(_tokens(s))
            lo = int(L * (1 - LENGTH_WINDOW_PCT))
            hi = int(L * (1 + LENGTH_WINDOW_PCT))
            in_window = False
            for r2 in all_rows:
                if r2 is r:
                    continue
                s2 = _evidence_sentence(r2)
                if s2 is None:
                    continue
                L2 = len(_tokens(s2))
                if lo <= L2 <= hi:
                    in_window = True
                    break
            if in_window:
                window_hits += 1
            else:
                window_misses += 1

    summary = {
        "input": {
            "path": str(args.data),
            "rows_total": n_total,
            "label_counts": dict(label_counts),
            "rows_binary": n_binary,
            "rows_excluded_NEI": n_total - n_binary,
        },
        "G1_cluster_availability": {
            "n_clusters_total": n_clusters,
            "clusters_with_dual_label": len(clusters_dual),
            "clusters_sup_only_ge3": len(clusters_sup_only),
            "clusters_ref_only_ge3": len(clusters_ref_only),
            "interp": (
                "dual_label clusters can contribute both SUPPORTS and REFUTES composites;"
                " sup_only/ref_only contribute to one side only"
            ),
        },
        "G2_composite_budget": {
            "SUPPORTS_C3_total": budget_sup,
            "REFUTES_C3_total": budget_ref,
            "interp": (
                f"target = {TARGET_N_PER_LABEL} per label = {TARGET_N_PER_LABEL * COMPOSITE_ROWS} rows per label;"
                f" budget_sup/budget_ref are the pool size before salt sort"
            ),
        },
        "G3_salt_sorted_top": {
            "target_per_label": TARGET_N_PER_LABEL,
            "total_composites_pool_SUPPORTS": len(sup_composites),
            "total_composites_pool_REFUTES": len(ref_composites),
            "composites_formed_SUPPORTS": formed_sup,
            "composites_formed_REFUTES": formed_ref,
            "shortfall_SUPPORTS": max(0, TARGET_N_PER_LABEL - formed_sup),
            "shortfall_REFUTES": max(0, TARGET_N_PER_LABEL - formed_ref),
            "unusable_partial_clusters_SUPPORTS": unsup_cnt,
            "unusable_partial_clusters_REFUTES": unref_cnt,
            "verdict": (
                "PASS" if (formed_sup >= TARGET_N_PER_LABEL and formed_ref >= TARGET_N_PER_LABEL)
                else "FAIL — need to relax §4.3 or pool more rows"
            ),
        },
        "G4_length_window": {
            "evidence_sentence_token_count": {
                "n": len(sent_lengths),
                "mean": round(statistics.fmean(sent_lengths), 2),
                "median": statistics.median(sent_lengths),
                "p25": sorted(sent_lengths)[len(sent_lengths) // 4],
                "p75": sorted(sent_lengths)[3 * len(sent_lengths) // 4],
                "min": min(sent_lengths),
                "max": max(sent_lengths),
            },
            "within_cluster_pm50pct_window": {
                "rows_with_a_same_cluster_match": window_hits,
                "rows_without": window_misses,
                "hit_rate": round(window_hits / max(1, window_hits + window_misses), 4),
                "interp": (
                    "this measures how often a substitute can be sourced from within the same"
                    " cluster with ±50% length; hit_rate < 0.5 means fallback to nearest-length"
                    " (V4 D1_v4 pattern) is essential"
                ),
            },
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        print(f"V5 MANIFEST DRY-RUN  (V5 preregistration §4.2, §4.3)")
        print("=" * 70)
        s = summary
        print("\n[Input]")
        print(f"  path                  : {s['input']['path']}")
        print(f"  rows_total            : {s['input']['rows_total']}")
        print(f"  label_counts          : {s['input']['label_counts']}")
        print(f"  rows_binary           : {s['input']['rows_binary']}  "
              f"(NEI excluded per §4.1: {s['input']['rows_excluded_NEI']})")
        print("\n[G1 — Cluster availability]")
        print(f"  n_clusters_total      : {s['G1_cluster_availability']['n_clusters_total']}")
        print(f"  clusters with ≥3 SUP AND ≥3 REF : "
              f"{s['G1_cluster_availability']['clusters_with_dual_label']}")
        print(f"  clusters with ≥3 SUP only       : "
              f"{s['G1_cluster_availability']['clusters_sup_only_ge3']}")
        print(f"  clusters with ≥3 REF only       : "
              f"{s['G1_cluster_availability']['clusters_ref_only_ge3']}")
        print("\n[G2 — Composite budget (C(n,3) per cluster per label)]")
        print(f"  SUPPORTS composites available : "
              f"{s['G2_composite_budget']['SUPPORTS_C3_total']}")
        print(f"  REFUTES  composites available : "
              f"{s['G2_composite_budget']['REFUTES_C3_total']}")
        print("\n[G3 — Salt-sorted top 25+25 feasibility]")
        print(f"  target per label       : {s['G3_salt_sorted_top']['target_per_label']}")
        print(f"  composite pool SUPPORTS: {s['G3_salt_sorted_top']['total_composites_pool_SUPPORTS']}")
        print(f"  composite pool REFUTES : {s['G3_salt_sorted_top']['total_composites_pool_REFUTES']}")
        print(f"  composites formed SUP  : {s['G3_salt_sorted_top']['composites_formed_SUPPORTS']}")
        print(f"  composites formed REF  : {s['G3_salt_sorted_top']['composites_formed_REFUTES']}")
        print(f"  shortfall SUPPORTS     : {s['G3_salt_sorted_top']['shortfall_SUPPORTS']}")
        print(f"  shortfall REFUTES      : {s['G3_salt_sorted_top']['shortfall_REFUTES']}")
        print(f"  unusable partial clusters (SUP) : "
              f"{s['G3_salt_sorted_top']['unusable_partial_clusters_SUPPORTS']}")
        print(f"  unusable partial clusters (REF) : "
              f"{s['G3_salt_sorted_top']['unusable_partial_clusters_REFUTES']}")
        print(f"  >>> VERDICT : {s['G3_salt_sorted_top']['verdict']}")
        print("\n[G4 — Evidence-sentence length & within-cluster length-window]")
        es = s['G4_length_window']['evidence_sentence_token_count']
        print(f"  n sent_length samples  : {es['n']}")
        print(f"  min/median/mean/max    : "
              f"{es['min']} / {es['median']} / {es['mean']} / {es['max']}")
        ww = s['G4_length_window']['within_cluster_pm50pct_window']
        print(f"  within-cluster ±50% window:")
        print(f"    rows with a match     : {ww['rows_with_a_same_cluster_match']}")
        print(f"    rows without a match  : {ww['rows_without']}")
        print(f"    hit_rate              : {ww['hit_rate']}")
        print("=" * 70)

    # Exit code: 0 if G3 passes, 1 otherwise (so CI/audit gates can use it)
    return 0 if summary["G3_salt_sorted_top"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
