#!/usr/bin/env python3
"""Re-derive V7's summary.json + report.md from records.partial.jsonl.

Use case: vLLM endpoint died mid-run; the formal run produced 1840 valid
records + 160 failed (Connection refused) before crashing in
_per_question_risks on empty `answers` lists. The _per_question_risks
bug is fixed in pilot_llm_v7; this script reads the partial records
and runs summarize_records to produce the formal artifacts without
re-running any LLM calls.

Reads: results/pilot_llm_v7/formal/records.partial.jsonl
       results/pilot_llm_v7/manifest.json
Writes: results/pilot_llm_v7/formal/records.jsonl
        results/pilot_llm_v7/formal/summary.json
        results/pilot_llm_v7/formal/report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/storage/gaoym/sp500-forecastability-lab")
sys.path.insert(0, str(ROOT / "src"))

from sp500_forecastability.pilot_llm_v7 import (
    PROTOCOL_VERSION,
    FORMAL_EXAMPLES,
    summarize_records,
    render_report,
    validate_manifest,
)


def main():
    formal_dir = ROOT / "results/pilot_llm_v7/formal"
    partial_path = formal_dir / "records.partial.jsonl"
    manifest_path = ROOT / "results/pilot_llm_v7/manifest.json"
    dataset_path = ROOT / "data/fever/fever-validation.jsonl"

    # 1. Load manifest
    manifest = json.loads(manifest_path.read_text())
    composites = validate_manifest(manifest, dataset_path)
    substitute_manifest = manifest["substitute_manifest"]

    # 2. Load partial records
    records = []
    with partial_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {partial_path.name}")

    # 3. Summarize (uses fixed _per_question_risks, skips questions with no decisions)
    summary = summarize_records(
        records, mode="formal",
        expected_examples=FORMAL_EXAMPLES,
        agent_count=5,
        substitute_manifest=substitute_manifest,
    )

    # 4. Promote partial to final
    records_path = formal_dir / "records.jsonl"
    with records_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {records_path}")

    summary_path = formal_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    report_path = formal_dir / "report.md"
    report_path.write_text(render_report(summary))
    print(f"Wrote {report_path}")

    # Print headline
    print()
    print("=" * 60)
    cpv = summary.get("co_primary_verdict")
    if cpv:
        print(f"Verdict: {cpv['verdict']}")
        for k in ("D_OR", "shared_weighted"):
            v = cpv.get(k, {})
            print(f"  {k}: {v.get('auroc', 'NA'):.4f} [{v.get('ci_lo', 'NA'):.3f}, {v.get('ci_hi', 'NA'):.3f}]  "
                  f"passes_lo>0.5={v.get('passes_lower_bound_above_0_5', False)}")
    print()
    print(f"N questions analyzed: {summary['outcome_prevalence']['n_questions']}")
    print(f"  harmful_fc: {summary['outcome_prevalence']['harmful_fc']}")
    print(f"  any_wrong: {summary['outcome_prevalence']['any_wrong_consensus']}")
    print(f"  valid records: {summary['instrumentation']['valid_records']}")
    print(f"  invalid records: {len(records) - summary['instrumentation']['valid_records']}")


if __name__ == "__main__":
    main()