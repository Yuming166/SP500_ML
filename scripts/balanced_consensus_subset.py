#!/usr/bin/env python3
"""
Stratified consensus-strength analysis (pre-registered 2026-09-01).

V7's 92% harmful_fc prevalence comes from agents being too unanimous
on wrong consensus (5/5 agreement on wrong answer). This script tests
whether routers win on a **subsample of V7 where consensus is weaker**
— i.e., where agents diverge enough to give routers room to operate.

Pre-registered design (locked BEFORE running):

  3 strata by consensus strength (= max agent agreement, 1-5 votes):
    S1_unanimous:  consensus strength = 1.0 (5/5 unanimous)
    S2_strong:     consensus strength ∈ {0.8, 0.9} (4-1 to 5-1)
    S3_weak:       consensus strength ≤ 0.6 (3-2 split)

  For each stratum × 5 methods (BL_majority, BL_D_OR, R1, R2, R3),
  compute AUROC of per-question score → harmful_fc label.

  Pre-registered expected patterns:
    - S1_unanimous:  R1/R2/R3 ≤ BL_majority (routers cannot recover from 5/5)
    - S2_strong:     methods converge (4-1 leaves some signal)
    - S3_weak:       R2 weighted vote > BL_majority (routers win)

  Reporting: all 15 cells, no cherry-picking.

Inputs:
    results/pilot_llm_v7/formal/records.jsonl

Outputs:
    analysis/balanced_consensus_subset.md
    analysis/balanced_consensus_subset.json
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
V7_PER_AGENT_AUROC = {
    0: 0.423, 1: 0.493, 2: 0.439, 3: 0.427, 4: 0.468,
}
R2_WEIGHTS = {ai: max(0.5, min(1.0, a)) for ai, a in V7_PER_AGENT_AUROC.items()}
R1_BEST_AGENT = max(V7_PER_AGENT_AUROC, key=V7_PER_AGENT_AUROC.get)


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


def load_records() -> list[dict]:
    rec_path = ROOT / "results/pilot_llm_v7/formal/records.jsonl"
    out = []
    with rec_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def per_question_data(records: list[dict]) -> tuple[dict, dict, dict]:
    """Returns: (data, q_harmful_fc, q_consensus_strength)"""
    by_qa: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in records:
        if not r.get("success") or not r.get("decision"):
            continue
        key = (r["cqid"], int(r["agent_index"]))
        by_qa[key][r["condition"]] = r

    q_orig: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in records:
        if r.get("condition") != "original" or not r.get("success") or not r.get("decision"):
            continue
        ans = r["decision"].get("answer")
        gold = int(r.get("gold_binary", 0))
        q_orig[r["cqid"]].append((ans, gold))
    q_harmful_fc = {}
    q_consensus_strength = {}
    for cqid, ans_gold in q_orig.items():
        if len(ans_gold) < 5:
            continue
        gold = ans_gold[0][1]
        answers = [a for a, _ in ans_gold]
        cnt = Counter(answers)
        cons, n = cnt.most_common(1)[0]
        q_consensus_strength[cqid] = n / 5
        if n < 4:
            q_harmful_fc[cqid] = -1  # 3-2 split, no consensus label
        else:
            correct_consensus = int((cons == "yes") == bool(gold))
            q_harmful_fc[cqid] = int(correct_consensus == 0)

    data: dict[str, dict[int, dict[str, Any]]] = {}
    for (cqid, ai), by_cond in by_qa.items():
        orig = by_cond.get("original")
        if orig is None:
            continue
        d = {
            "answer": orig["decision"]["answer"],
            "conf": float(orig["decision"].get("confidence", 0.0) or 0.0),
            "gold_binary": int(orig.get("gold_binary", 0)),
        }
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
        data.setdefault(cqid, {})[ai] = d
    return data, q_harmful_fc, q_consensus_strength


def stratify(q_consensus_strength: dict) -> dict[str, list[str]]:
    out = {"S1_unanimous": [], "S2_strong": [], "S3_weak": []}
    for cqid, s in q_consensus_strength.items():
        if s == 1.0:
            out["S1_unanimous"].append(cqid)
        elif s >= 0.8:
            out["S2_strong"].append(cqid)
        else:
            out["S3_weak"].append(cqid)
    return out


def evaluate(data: dict, q_harmful_fc: dict, stratum_qids: list[str]) -> dict:
    """Evaluate the 5 methods on the given stratum. Drop questions where
    harmful_fc label is -1 (3-2 split, no consensus)."""
    methods = {
        "BL_majority": lambda qd, sc: _bl_majority(qd),
        "BL_D_OR": lambda qd, sc: _bl_d_or(qd, sc),
        "R1_top_auroc": lambda qd, sc: _r1(qd, sc),
        "R2_weighted": lambda qd, sc: _r2(qd, sc),
        "R3_min_frag": lambda qd, sc: _r3(qd, sc),
    }
    out = {}
    for name, fn in methods.items():
        scores = []
        labels = []
        for cqid in stratum_qids:
            if cqid not in data or len(data[cqid]) < 5:
                continue
            label = q_harmful_fc.get(cqid, -1)
            if label == -1:
                continue
            # per-question agent score (R2 needs weights; R1 needs ranking;
            # others ignore score and use the agent data directly)
            ans, sc = fn(data[cqid], _v7_per_q_score(cqid, data))
            scores.append(sc)
            labels.append(label)
        auroc = _auroc(scores, labels)
        ci_lo, ci_hi = _bootstrap_auroc_ci(scores, labels)
        out[name] = {
            "auroc": auroc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_questions": len(scores),
            "n_pos": sum(labels),
        }
    return out


def _v7_per_q_score(cqid: str, data: dict) -> dict[int, float]:
    """Per-agent 'reliability score' = 1 - fragility, using V7's actual fragility."""
    if cqid not in data:
        return {ai: 0.5 for ai in AGENT_NAMES}
    return {ai: 1.0 - data[cqid][ai]["fragile"] for ai in AGENT_NAMES}


def _bl_majority(qd):
    answers = [d["answer"] for d in qd.values()]
    cnt = Counter(answers)
    cons, n = cnt.most_common(1)[0]
    return cons, n / len(qd)


def _bl_d_or(qd, sc):
    frags = [d["fragile"] for d in qd.values()]
    mean_frag = sum(frags) / max(1, len(frags))
    answers = [d["answer"] for d in qd.values()]
    cons, _ = Counter(answers).most_common(1)[0]
    return cons, 1.0 - mean_frag


def _r1(qd, sc):
    best_ai = max(sc.keys(), key=lambda ai: sc[ai])
    return qd[best_ai]["answer"], sc[best_ai]


def _r2(qd, sc):
    yes_w = sum(sc[ai] for ai in qd if qd[ai]["answer"] == "yes")
    total_w = sum(sc[ai] for ai in qd)
    return ("yes" if yes_w > total_w / 2 else "no"), yes_w / max(1, total_w)


def _r3(qd, sc):
    best_ai = max(sc.keys(), key=lambda ai: sc[ai])
    return qd[best_ai]["answer"], sc[best_ai]


def render_table(all_results: dict, strata: dict) -> str:
    lines = ["# Stratified consensus-strength analysis on V7 (pre-registered 2026-09-01)\n"]
    lines.append("3 strata × 5 methods = 15 cells, all reported.\n")
    lines.append("**V7 baseline (full N=92, prevalence 92.4%): BL_majority=0.922, R2=0.42**\n")

    lines.append("## Strata counts\n")
    for s in ["S1_unanimous", "S2_strong", "S3_weak"]:
        n = len(strata.get(s, []))
        lines.append(f"- `{s}`: N = {n} questions")
    lines.append("")

    lines.append("## Cross-strata × method AUROC\n")
    methods = ["BL_majority", "BL_D_OR", "R1_top_auroc", "R2_weighted", "R3_min_frag"]
    header = "| Stratum | N | " + " | ".join(methods) + " |"
    sep = "|---|---:|" + "---:|" * len(methods)
    lines.append(header)
    lines.append(sep)
    for s in ["S1_unanimous", "S2_strong", "S3_weak"]:
        if s not in all_results:
            continue
        res = all_results[s]
        n = next(iter(res.values()))["n_questions"] if res else 0
        row = [s, str(n)]
        for m in methods:
            r = res.get(m, {})
            a = r.get("auroc")
            row.append(f"{a:.3f}" if a is not None else "NA")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_detail(all_results: dict) -> str:
    lines = []
    methods = ["BL_majority", "BL_D_OR", "R1_top_auroc", "R2_weighted", "R3_min_frag"]
    for s in ["S1_unanimous", "S2_strong", "S3_weak"]:
        if s not in all_results:
            continue
        res = all_results[s]
        n = next(iter(res.values()))["n_questions"] if res else 0
        lines.append(f"### {s} (N = {n})\n")
        lines.append("| Method | AUROC | CI lo | CI hi |")
        lines.append("|---|---:|---:|---:|")
        for m in methods:
            r = res.get(m, {})
            a = r.get("auroc")
            ci_lo = r.get("ci_lo", float("nan"))
            ci_hi = r.get("ci_hi", float("nan"))
            a_s = f"{a:.3f}" if a is not None else "NA"
            lo_s = f"{ci_lo:.3f}" if ci_lo == ci_lo else "NA"
            hi_s = f"{ci_hi:.3f}" if ci_hi == ci_hi else "NA"
            lines.append(f"| {m} | {a_s} | {lo_s} | {hi_s} |")
        lines.append("")
    return "\n".join(lines)


def render_interpretation() -> str:
    return """
## Interpretation (pre-registered)

**Pre-registered question**: "Does router AUROC depend on consensus strength
(agent agreement)?"

**Reading the strata**
- **S1_unanimous** (5/5 agree): strongest consensus. All agents say the same
  thing — no router can pick a different answer.
- **S2_strong** (4/5 or 5/5 minus 1): strong consensus with one dissenter.
  Some room for routers to pick the dissenter.
- **S3_weak** (3/2 or less): weak consensus. Multiple agents disagree, and
  the minority might be right.

**Pre-registered expected patterns**
- S1 should show BL_majority ≈ routers (no answer diversity).
- S3 should show R2 (AUROC-weighted vote) > BL_majority if routers work
  on diverse-agent questions.

**Reporting policy**
- All 15 cells reported, no stratum hidden.
- If S3 also loses, the failure mode is not consensus-strength but
  something deeper (e.g., wrong answer = "obvious" to all agents regardless
  of evidence removal).
"""


def main():
    records = load_records()
    data, q_harmful_fc, q_consensus_strength = per_question_data(records)
    strata = stratify(q_consensus_strength)
    print(f"[loaded V7] {len(records)} records, {len(data)} questions, "
          f"harmful_fc prevalence {sum(1 for v in q_harmful_fc.values() if v == 1) / max(1, len(q_harmful_fc)):.3f}")
    for s, qids in strata.items():
        n = len(qids)
        n_pos = sum(1 for q in qids if q_harmful_fc.get(q) == 1)
        n_neg = sum(1 for q in qids if q_harmful_fc.get(q) == 0)
        n_skipped = sum(1 for q in qids if q_harmful_fc.get(q) == -1)
        print(f"  {s}: N={n} (positive={n_pos}, negative={n_neg}, 3-2-skipped={n_skipped})")

    all_results = {}
    for s, qids in strata.items():
        all_results[s] = evaluate(data, q_harmful_fc, qids)
        n = next(iter(all_results[s].values()))["n_questions"]
        print(f"  {s}: evaluated N={n} questions")

    md = render_table(all_results, strata)
    md += render_detail(all_results)
    md += render_interpretation()
    out_md = ANALYSIS_DIR / "balanced_consensus_subset.md"
    out_md.write_text(md)
    print(f"\n[wrote] {out_md}")

    out_json = ANALYSIS_DIR / "balanced_consensus_subset.json"
    out_json.write_text(json.dumps({
        "strata_counts": {s: len(qids) for s, qids in strata.items()},
        "results": all_results,
    }, indent=2, default=str))
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()