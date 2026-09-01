#!/usr/bin/env python3
"""
Individual-agent reliability analysis across Pilot-LLM versions.

Per-agent metrics (computed on existing records, no new LLM calls):
  - correct_rate: fraction of questions where this agent's original-condition
    answer matches gold_label
  - conf_stable_rate: fraction of questions where this agent's confidence
    drop |orig_conf - other_conf| < 0.05 in ALL 3 conditions
  - per_condition_flip_rate[c]: fraction of questions where this agent
    flipped answer under condition c ∈ {remove, reverse, substitute}
  - per_agent_auroc_fragility: AUROC of per-question
    fragility[q] = 1 - (inert OR conf_stable) → harmful_fc label

Reads from existing records.jsonl files (zero bandwidth):
    results/pilot_llm_v7/formal/records.jsonl  (preferred: N=100, V5 salt)
    results/pilot_llm_v6/formal/records.jsonl
    results/pilot_llm_v5/formal/records.jsonl
    results/pilot_llm_v4/formal/records.jsonl

Writes:
    analysis/individual_agent_reliability.md   (paper-ready table)
    analysis/individual_agent_reliability.json (machine-readable)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/storage/gaoym/sp500-forecastability-lab")
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_BAND = 0.05  # V4 §8 / V5 §8 / V6 §8 / V7 §8
AGENT_INDICES = [0, 1, 2, 3, 4]
CONDITIONS = ["remove", "reverse", "substitute"]
AGENT_NAMES = {
    0: "literal_evidence",
    1: "skeptical_auditor",
    2: "consistency_checker",
    3: "counterfactual_reasoner",
    4: "minimal_judge",
}


def _auroc(scores, labels):
    if not scores or not labels:
        return None
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def _bootstrap_auroc_ci(scores, labels, n_replicates=1000, seed=20_260_902):
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n < 5:
        return (float("nan"), float("nan"))
    boot = []
    for _ in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        b_scores = [scores[i] for i in idx]
        b_labels = [labels[i] for i in idx]
        a = _auroc(b_scores, b_labels)
        if a is not None:
            boot.append(a)
    if not boot:
        return (float("nan"), float("nan"))
    boot.sort()
    return (boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))])


def load_records(version: str) -> list[dict]:
    """Load records.jsonl from a Pilot-LLM formal run."""
    rec_path = ROOT / f"results/pilot_llm_{version}/formal/records.jsonl"
    if not rec_path.exists():
        return []
    out = []
    with rec_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def compute_q_harmful_fc(records: list[dict]) -> dict[str, int]:
    """For each question, compute harmful_fc from the 5 agents' original answers.

    Returns: {cqid: 1 if (consensus != gold AND agreement >= 0.8) else 0}
    """
    q_orig: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in records:
        if r.get("condition") != "original":
            continue
        if not r.get("success") or not r.get("decision"):
            continue
        ans = r["decision"].get("answer")
        gold = int(r.get("gold_binary", 0))
        q_orig[r["cqid"]].append((ans, gold))

    out: dict[str, int] = {}
    for cqid, ans_gold in q_orig.items():
        if len(ans_gold) < 5:
            continue
        gold = ans_gold[0][1]
        answers = [a for a, _ in ans_gold]
        cnt = Counter(answers)
        cons, n = cnt.most_common(1)[0]
        if n < 4:
            out[cqid] = 0
            continue
        correct_consensus = int((cons == "yes") == bool(gold))
        out[cqid] = int(correct_consensus == 0 and (n / 5) >= 0.8)
    return out


def per_agent_metrics(records: list[dict], q_harmful_fc: dict[str, int]) -> dict[int, dict[str, Any]]:
    """Compute per-agent reliability metrics.

    For each (cqid, agent_index), look up:
      - original record → orig_answer, orig_conf, gold_binary
      - remove / reverse / substitute records → flips, conf_drops
    Then derive inert / conf_stable / fragility.
    """
    by_qa: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in records:
        if not r.get("success") or not r.get("decision"):
            continue
        key = (r["cqid"], int(r["agent_index"]))
        by_qa[key][r["condition"]] = r

    per_agent: dict[int, dict[str, list]] = {
        i: {
            "correct": [],
            "flip": [],
            "conf_stable": [],
            "fragility_label_pairs": [],
        }
        for i in AGENT_INDICES
    }

    for (cqid, ai), by_cond in by_qa.items():
        orig = by_cond.get("original")
        if orig is None:
            continue
        orig_answer = orig["decision"]["answer"]
        orig_conf = float(orig["decision"].get("confidence", 0.0) or 0.0)
        gold_binary = int(orig.get("gold_binary", 0))
        is_correct = int((orig_answer == "yes") == bool(gold_binary))

        flips = {}
        conf_drops = {}
        for cond in CONDITIONS:
            other = by_cond.get(cond)
            if other is None:
                continue
            flips[cond] = int(orig_answer != other["decision"]["answer"])
            conf_drops[cond] = orig_conf - float(other["decision"].get("confidence", 0.0) or 0.0)

        inert = int(all(flips.get(c, 0) == 0 for c in CONDITIONS))
        conf_stable = int(
            all(abs(conf_drops.get(c, 0)) < CONFIDENCE_BAND
                for c in CONDITIONS)
        )

        per_agent[ai]["correct"].append(is_correct)
        per_agent[ai]["flip"].append(flips)
        per_agent[ai]["conf_stable"].append(conf_stable)

        fragility = int(not (inert or conf_stable))
        label = q_harmful_fc.get(cqid, 0)
        per_agent[ai]["fragility_label_pairs"].append((fragility, label))

    out: dict[int, dict[str, Any]] = {}
    for ai in AGENT_INDICES:
        n = len(per_agent[ai]["correct"])
        if n == 0:
            continue
        scores = [s for s, _ in per_agent[ai]["fragility_label_pairs"]]
        labels = [l for _, l in per_agent[ai]["fragility_label_pairs"]]
        auroc = _auroc(scores, labels)
        ci_lo, ci_hi = _bootstrap_auroc_ci(scores, labels)

        flip_rate = {}
        for cond in CONDITIONS:
            flip_rate[cond] = sum(
                per_agent[ai]["flip"][i].get(cond, 0) for i in range(n)
            ) / max(1, n)

        out[ai] = {
            "n_questions": n,
            "correct_rate": sum(per_agent[ai]["correct"]) / max(1, n),
            "conf_stable_rate": sum(per_agent[ai]["conf_stable"]) / max(1, n),
            "per_condition_flip_rate": flip_rate,
            "per_agent_auroc_fragility": auroc,
            "auroc_ci": (ci_lo, ci_hi),
        }
    return out


def render_version(version: str, m: dict[int, dict[str, Any]], n_q: int) -> str:
    lines = [f"## {version.upper()} — per-agent reliability (N = {n_q} questions)\n"]
    lines.append("| Agent | n | correct_rate | conf_stable | flip_remove | flip_reverse | flip_substitute | AUROC_fragility | CI_lo | CI_hi |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ai in AGENT_INDICES:
        d = m.get(ai)
        if d is None:
            continue
        auroc = d["per_agent_auroc_fragility"]
        ci_lo, ci_hi = d["auroc_ci"]
        auroc_s = f"{auroc:.3f}" if auroc is not None else "NA"
        # NaN check (nan != nan)
        ci_lo_s = f"{ci_lo:.3f}" if ci_lo == ci_lo else "NA"
        ci_hi_s = f"{ci_hi:.3f}" if ci_hi == ci_hi else "NA"
        lines.append(
            f"| {AGENT_NAMES[ai]} | {d['n_questions']} | "
            f"{d['correct_rate']:.3f} | {d['conf_stable_rate']:.3f} | "
            f"{d['per_condition_flip_rate']['remove']:.3f} | "
            f"{d['per_condition_flip_rate']['reverse']:.3f} | "
            f"{d['per_condition_flip_rate']['substitute']:.3f} | "
            f"{auroc_s} | {ci_lo_s} | {ci_hi_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_cross_version(all_metrics: dict) -> str:
    """Cross-version per-agent AUROC table (V4 vs V5 vs V6 vs V7)."""
    lines = ["## Cross-version per-agent AUROC (fragility → harmful_fc)\n"]
    lines.append("| Agent | V7 (N=100, V5 salt) | V6 (N=100, V6 salt) | V5 (N=50, V5 salt) | V4 (N=50, V4 salt) |")
    lines.append("|---|---|---|---|---|")
    for ai in AGENT_INDICES:
        row = [AGENT_NAMES[ai]]
        for v in ["v7", "v6", "v5", "v4"]:
            d = all_metrics.get(v, {}).get("per_agent", {}).get(ai)
            if d is None or d["per_agent_auroc_fragility"] is None:
                row.append("NA")
            else:
                a = d["per_agent_auroc_fragility"]
                ci = d["auroc_ci"]
                row.append(f"{a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_interpretation() -> str:
    lines = [
        "## Interpretation\n",
        "Per-agent AUROC measures how well each agent's *fragility* (1 - (inert OR conf_stable)) "
        "predicts the question-level `harmful_fc` outcome.",
        "",
        "**Reading the tables**",
        "- Higher AUROC = this agent's fragility is a better per-question predictor of harm",
        "- `correct_rate` reveals baseline accuracy (this is the original-condition answer only)",
        "- `conf_stable_rate` reveals confidence discipline (how often this agent's confidence "
        "is robust to interventions regardless of answer flips)",
        "- `per_condition_flip_rate[c]` decomposes fragility by intervention type",
        "",
        "**Use cases for downstream paper sections**",
        "- **Agent-level router (§11.b)**: weight answers by per-agent AUROC, or pick the "
        "most-reliable agent per question. Per-agent AUROC provides the diagnostic table that "
        "justifies the routing weights.",
        "- **Persona engineering (§11.c)**: if `minimal_judge` is consistently more fragile "
        "(lower AUROC) than `skeptical_auditor`, this is empirical evidence that the `minimal` "
        "persona is too credulous under evidence removal.",
        "- **Selection-fixed stability (§10)**: comparing per-agent AUROC across V5 (N=50) and "
        "V7 (N=100, V5 salt) tests whether each individual agent's reliability is selection-fixed-stable. "
        "If the SAME agent has consistent AUROC across V5 and V7 but D_OR (the unweighted mean) "
        "varies, this isolates the aggregation as the source of V5→V6 regression.",
        "",
    ]
    return "\n".join(lines)


def main():
    all_metrics: dict[str, dict] = {}
    for v in ["v7", "v6", "v5", "v4"]:
        records = load_records(v)
        if not records:
            print(f"[warn] {v}: no records found, skipping")
            continue
        q_harmful_fc = compute_q_harmful_fc(records)
        m = per_agent_metrics(records, q_harmful_fc)
        n_q = len({r["cqid"] for r in records if r.get("cqid")})
        all_metrics[v] = {"n_questions": n_q, "per_agent": m}
        print(f"[done] {v}: N={n_q} questions, {len(m)} agents computed")

    md = ["# Individual-agent reliability analysis\n"]
    md.append("Per-agent breakdown of Pilot-LLM V4/V5/V6/V7. "
              "Each row computes fragility = 1 - (inert OR conf_stable) "
              "for every (cqid, agent_index) pair, then computes AUROC of fragility → "
              "harmful_fc label (consensus answer is wrong with ≥0.8 agreement).\n")
    for v in ["v7", "v6", "v5", "v4"]:
        if v in all_metrics:
            md.append(render_version(v, all_metrics[v]["per_agent"],
                                     all_metrics[v]["n_questions"]))
    md.append(render_cross_version(all_metrics))
    md.append(render_interpretation())

    out_md = ANALYSIS_DIR / "individual_agent_reliability.md"
    out_md.write_text("\n".join(md))
    print(f"\n[wrote] {out_md}")

    out_json = ANALYSIS_DIR / "individual_agent_reliability.json"
    json_dump = {}
    for v, vm in all_metrics.items():
        json_dump[v] = {
            "n_questions": vm["n_questions"],
            "per_agent": {
                str(ai): {
                    "n_questions": d["n_questions"],
                    "correct_rate": d["correct_rate"],
                    "conf_stable_rate": d["conf_stable_rate"],
                    "per_condition_flip_rate": d["per_condition_flip_rate"],
                    "per_agent_auroc_fragility": d["per_agent_auroc_fragility"],
                    "auroc_ci": list(d["auroc_ci"]),
                }
                for ai, d in vm["per_agent"].items()
            },
        }
    out_json.write_text(json.dumps(json_dump, indent=2))
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()