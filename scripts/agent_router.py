#!/usr/bin/env python3
"""
Agent-level router comparison across Pilot-LLM versions.

Pre-registered design (locked 2026-09-01 BEFORE running this script):

ROUTER VARIANTS (all 3 reported, no cherry-picking):
  R1_top_auroc:  For each question, pick the answer from the agent with
                  the highest V7 per-agent AUROC_fragility. Tie-break by
                  this-question's agent fragility (lower wins).
  R2_weighted:   Weighted majority vote. Per-agent weight = V7
                  per-agent AUROC (clipped to [0.5, 1.0] to avoid zero-
                  weighting). Weighted score for "yes" answers, choose
                  "yes" if total weighted yes > total weighted no.
  R3_min_frag:   For each question, pick the answer from the agent
                  with the lowest fragility on that question
                  (most "inert OR conf_stable").

BASELINES:
  BL_majority:   Simple unweighted majority vote of 5 agents' original
                  answers (the implicit baseline of every prior version).
  BL_D_OR:       D_OR-style fragility score per question (mean of 5
                  agents' (inert OR conf_stable)) — predicts harmful_fc.

EVALUATION:
  - harmful_fc label = consensus wrong AND agreement ≥ 0.8 (V7 §8).
  - Per-question "score" for each method (router returns answer + score
    for AUROC computation). For BL_majority: vote agreement (n/5).
    For BL_D_OR: mean fragility. For R1: 1 - this question's fragility of
    the chosen agent. For R2: weighted vote sum. For R3: 1 - this
    question's fragility of the chosen agent.
  - AUROC of per-question score → harmful_fc.

REPORTED:
  - 4 versions × 5 methods = 20 (AUROC, 95% bootstrap CI) cells.
  - No method/version is hidden; no version is selected based on result.

Inputs:
    analysis/individual_agent_reliability.json   (for V7 per-agent AUROC)
    results/pilot_llm_v{4,5,6,7}/formal/records.jsonl

Outputs:
    analysis/agent_router_comparison.md
    analysis/agent_router_comparison.json
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

CONFIDENCE_BAND = 0.05
CONDITIONS = ["remove", "reverse", "substitute"]
AGENT_NAMES = {
    0: "literal_evidence", 1: "skeptical_auditor",
    2: "consistency_checker", 3: "counterfactual_reasoner",
    4: "minimal_judge",
}


# ----- pre-registered inputs -----

# V7 per-agent AUROC is the LEARNED signal for R1 and R2 weights.
# Frozen here so the script is reproducible from a single recorded input.
V7_PER_AGENT_AUROC = {
    0: 0.423,  # literal_evidence
    1: 0.493,  # skeptical_auditor   <- highest V7 AUROC
    2: 0.439,
    3: 0.427,
    4: 0.468,
}
R1_BEST_AGENT = max(V7_PER_AGENT_AUROC, key=V7_PER_AGENT_AUROC.get)  # 1 (skeptical_auditor)

# R2 weight: clip AUROC to [0.5, 1.0] (so no agent is zeroed out).
# Weights:
def _r2_weight(auroc: float) -> float:
    return max(0.5, min(1.0, auroc))

R2_WEIGHTS = {ai: _r2_weight(a) for ai, a in V7_PER_AGENT_AUROC.items()}


# ----- helpers -----

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


def _bootstrap_auroc_ci(scores, labels, n_replicates=2000, seed=20_260_902):
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


def per_question_agent_table(records: list[dict]) -> dict[str, dict[int, dict[str, Any]]]:
    """For each cqid, build per-agent view:
       {cqid: {ai: {answer, conf, gold_binary, fragile, inertia, conf_stable}}}
    """
    by_qa: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in records:
        if not r.get("success") or not r.get("decision"):
            continue
        key = (r["cqid"], int(r["agent_index"]))
        by_qa[key][r["condition"]] = r

    out: dict[str, dict[int, dict[str, Any]]] = {}
    for (cqid, ai), by_cond in by_qa.items():
        orig = by_cond.get("original")
        if orig is None:
            continue
        d = {
            "answer": orig["decision"]["answer"],
            "conf": float(orig["decision"].get("confidence", 0.0) or 0.0),
            "gold_binary": int(orig.get("gold_binary", 0)),
            "inert": 0, "conf_stable": 0, "fragile": 1,
            "orig_label_correct": int((orig["decision"]["answer"] == "yes")
                                     == bool(int(orig.get("gold_binary", 0)))),
        }
        # Compute fragility from siblings
        flips = {}
        conf_drops = {}
        for cond in CONDITIONS:
            other = by_cond.get(cond)
            if other is None:
                continue
            flips[cond] = int(orig["decision"]["answer"] != other["decision"]["answer"])
            conf_drops[cond] = d["conf"] - float(other["decision"].get("confidence", 0.0) or 0.0)
        d["inert"] = int(all(flips.get(c, 0) == 0 for c in CONDITIONS))
        d["conf_stable"] = int(all(abs(conf_drops.get(c, 0)) < CONFIDENCE_BAND for c in CONDITIONS))
        d["fragile"] = int(not (d["inert"] or d["conf_stable"]))
        out.setdefault(cqid, {})[ai] = d
    return out


# ----- router variants (pre-registered) -----

def router_top_auroc(q_data: dict[int, dict], best_agent: int) -> tuple[str, float]:
    """R1: pick best_agent's answer. Score = 1 - this_agent's fragility."""
    chosen = q_data[best_agent]
    score = 1.0 - chosen["fragile"]
    return chosen["answer"], score


def router_weighted(q_data: dict[int, dict], weights: dict[int, float]) -> tuple[str, float]:
    """R2: weighted majority vote. Score = weighted vote sum (yes_weight / total)."""
    yes_w = sum(weights[ai] for ai in q_data if q_data[ai]["answer"] == "yes")
    total_w = sum(weights[ai] for ai in q_data)
    chosen = "yes" if yes_w > total_w / 2 else "no"
    score = yes_w / max(1, total_w)
    return chosen, score


def router_min_fragility(q_data: dict[int, dict]) -> tuple[str, float]:
    """R3: pick agent with lowest fragility on this question."""
    best_ai = min(q_data.keys(), key=lambda ai: q_data[ai]["fragile"])
    score = 1.0 - q_data[best_ai]["fragile"]
    return q_data[best_ai]["answer"], score


def baseline_majority(q_data: dict[int, dict]) -> tuple[str, float]:
    """BL_majority: simple unweighted majority. Score = vote agreement (n/5)."""
    answers = [d["answer"] for d in q_data.values()]
    cnt = Counter(answers)
    cons, n = cnt.most_common(1)[0]
    return cons, n / len(q_data)


def baseline_d_or(q_data: dict[int, dict]) -> tuple[str, float]:
    """BL_D_OR: D_OR-style fragility score. Score = 1 - mean(inert OR conf_stable)."""
    frags = [d["fragile"] for d in q_data.values()]
    mean_frag = sum(frags) / max(1, len(frags))
    # pick majority answer for the binary output (score is used for AUROC)
    answers = [d["answer"] for d in q_data.values()]
    cons, _ = Counter(answers).most_common(1)[0]
    return cons, 1.0 - mean_frag


# ----- per-version analysis -----

def evaluate_version(version: str, q_harmful_fc: dict[str, int],
                     q_data: dict[str, dict[int, dict]]) -> dict[str, dict[str, Any]]:
    methods = {
        "BL_majority": baseline_majority,
        "BL_D_OR": baseline_d_or,
        "R1_top_auroc": lambda qd: router_top_auroc(qd, R1_BEST_AGENT),
        "R2_weighted": lambda qd: router_weighted(qd, R2_WEIGHTS),
        "R3_min_frag": router_min_fragility,
    }

    results = {}
    for name, fn in methods.items():
        labels = []
        scores = []
        for cqid, qd in q_data.items():
            if len(qd) < 5:
                continue
            ans, score = fn(qd)
            label = q_harmful_fc.get(cqid, 0)
            labels.append(label)
            scores.append(score)
        auroc = _auroc(scores, labels)
        ci_lo, ci_hi = _bootstrap_auroc_ci(scores, labels)
        results[name] = {
            "auroc": auroc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_questions": len(labels),
            "positive_class_n": sum(labels),
        }
    return results


def render_table(version: str, res: dict[str, dict[str, Any]], n_q: int) -> str:
    lines = [f"## {version.upper()} — N = {n_q} questions\n"]
    lines.append("| Method | AUROC | CI lo | CI hi |")
    lines.append("|---|---:|---:|---:|")
    for name in ["BL_majority", "BL_D_OR", "R1_top_auroc", "R2_weighted", "R3_min_frag"]:
        r = res.get(name)
        if r is None:
            continue
        a = r["auroc"]
        ci_lo = r["ci_lo"]
        ci_hi = r["ci_hi"]
        a_s = f"{a:.3f}" if a is not None else "NA"
        # NaN-safe formatting
        lo_s = f"{ci_lo:.3f}" if ci_lo == ci_lo else "NA"
        hi_s = f"{ci_hi:.3f}" if ci_hi == ci_hi else "NA"
        lines.append(f"| {name} | {a_s} | {lo_s} | {hi_s} |")
    lines.append("")
    return "\n".join(lines)


def render_cross_version(all_results: dict[str, dict[str, dict]]) -> str:
    """Single big table: rows = versions, cols = methods."""
    lines = ["## Cross-version comparison (rows = Pilot-LLM version, cols = method)\n"]
    method_order = ["BL_majority", "BL_D_OR", "R1_top_auroc", "R2_weighted", "R3_min_frag"]
    header = "| Version | N | " + " | ".join(method_order) + " |"
    sep = "|---|---:|" + "---:|" * len(method_order)
    lines.append(header)
    lines.append(sep)
    for v in ["v7", "v6", "v5", "v4"]:
        if v not in all_results:
            continue
        res = all_results[v]
        n_q = next(iter(res.values()))["n_questions"] if res else 0
        row = [v.upper(), str(n_q)]
        for m in method_order:
            r = res.get(m, {})
            a = r.get("auroc")
            row.append(f"{a:.3f}" if a is not None else "NA")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_interpretation() -> str:
    return """
## Interpretation

This table reports **5 methods × 4 versions = 20 cells**, none hidden.

**Reading the table**
- AUROC = P(method's score higher on harmful_fc questions than on non-harmful_fc questions).
- Higher is better. CI lo > 0.5 means the method's bar at §9.2-equivalent is cleared.
- All methods use the same per-question `harmful_fc` label (V7 §8: consensus wrong AND agreement ≥ 0.8).

**Pre-registered router design (locked 2026-09-01)**
- **R1 (top-AUROC agent)**: pick the answer from the agent with the highest V7 per-agent AUROC_fragility (0.493, `skeptical_auditor`). Tie-break by per-question agent fragility.
- **R2 (weighted vote)**: per-agent weight = clip(V7 per-agent AUROC, [0.5, 1.0]). Weighted majority vote among 5 agents' original answers.
- **R3 (min-fragility agent)**: for each question, pick the answer from the agent with the lowest fragility (most inert/conf-stable) on that question.

**Baselines**
- **BL_majority**: unweighted majority of 5 agents (the implicit baseline of every prior version).
- **BL_D_OR**: per-question D_OR-style score = 1 - mean(inert OR conf_stable) across 5 agents.

**Reporting policy**
- All 5 methods reported for all 4 versions. No version hidden. No method hidden.
- If a router loses on a version, this is reported as-is. The methodological lesson
  (router efficacy is selection- and prevalence-dependent) is more interesting than
  cherry-picking a winning variant.
"""


def main():
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    all_q_data: dict[str, dict[str, dict[int, dict]]] = {}
    for v in ["v7", "v6", "v5", "v4"]:
        records = load_records(v)
        if not records:
            print(f"[warn] {v}: no records, skipping")
            continue
        q_harmful_fc = compute_q_harmful_fc(records)
        q_data = per_question_agent_table(records)
        all_q_data[v] = q_data
        res = evaluate_version(v, q_harmful_fc, q_data)
        n_q = sum(1 for _ in q_data.values())
        all_results[v] = res
        print(f"[done] {v}: N={n_q} questions, methods={list(res.keys())}")

    md = ["# Agent-level router comparison (pre-registered 2026-09-01)\n"]
    md.append("5 methods × 4 versions = 20 cells, all reported.\n")
    md.append("Pre-registered router variants: R1 = top-V7-AUROC agent, R2 = "
              "AUROC-weighted majority, R3 = min-fragility agent.\n")
    md.append(render_cross_version(all_results))
    md.append("")
    for v in ["v7", "v6", "v5", "v4"]:
        if v in all_results:
            n_q = len(all_q_data[v])
            md.append(render_table(v, all_results[v], n_q))
    md.append(render_interpretation())

    out_md = ANALYSIS_DIR / "agent_router_comparison.md"
    out_md.write_text("\n".join(md))
    print(f"\n[wrote] {out_md}")

    out_json = ANALYSIS_DIR / "agent_router_comparison.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()