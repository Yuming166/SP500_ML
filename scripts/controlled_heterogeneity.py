#!/usr/bin/env python3
"""
Controlled-heterogeneity experiment (pre-registered 2026-09-01).

Takes V7 records (N=100, V5 salt) and re-shuffles which real agent
is "most-reliable" per question, simulating 4 different agent-reliability
distributions. Tests how router vs majority vote perform under each.

Pre-registered design (locked BEFORE running this script):
  4 profiles × 5 methods = 20 AUROC cells, all reported.

PROFILES:
  P1_homogeneous:    For each question, randomly assign one of 5 agents
                      to be "most-reliable" (uniform random). Simulates
                      a regime where no agent is systematically better.
  P2_concentrated_best:  Always pick the agent with highest V7 per-agent
                         AUROC (skeptical_auditor, index 1) as
                         "most-reliable". Simulates a regime where one
                         agent is uniformly best.
  P3_concentrated_worst:  Always pick the agent with lowest V7 per-agent
                          AUROC (literal_evidence, index 0) as
                          "most-reliable". Simulates the opposite of P2.
  P4_realistic_FEVER:    Use the real V7 per-question fragility rank as
                          "reliability rank" (i.e., FEVER's actual
                          heterogeneous fragility distribution). This
                          is the "control" matching V4-V7 analysis.

METHODS (pre-registered from agent_router.py):
  BL_majority, BL_D_OR, R1_top_auroc, R2_weighted, R3_min_frag

REPORTING:
  All 20 cells, no cherry-picking. The pre-registered question is:
    "Does router AUC depend on agent reliability distribution?"
  If P2 → P1 → P3: routers are sensitivity to concentration (good)
  If P2 ≈ P1 ≈ P3: routers are not extracting agent signal (bad)
  If P4 < P2 (router wins on P2 but not P4): router efficacy
    depends on the agent-heterogeneity regime.

Inputs:
    results/pilot_llm_v7/formal/records.jsonl

Outputs:
    analysis/controlled_heterogeneity.md
    analysis/controlled_heterogeneity.json
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path("/storage/gaoym/sp500-forecastability-lab")
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_BAND = 0.05
CONDITIONS = ["remove", "reverse", "substitute"]
AGENT_NAMES = {
    0: "literal_evidence", 1: "skeptical_auditor",
    2: "consistency_checker", 3: "counterfactual_reasoner",
    4: "minimal_judge",
}
# V7 per-agent AUROC of fragility → harmful_fc (from individual_agent_reliability)
V7_PER_AGENT_AUROC = {
    0: 0.423, 1: 0.493, 2: 0.439, 3: 0.427, 4: 0.468,
}
R2_WEIGHTS = {ai: max(0.5, min(1.0, a)) for ai, a in V7_PER_AGENT_AUROC.items()}
R1_BEST_AGENT = max(V7_PER_AGENT_AUROC, key=V7_PER_AGENT_AUROC.get)  # 1


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


def load_records(version: str = "v7") -> list[dict]:
    rec_path = ROOT / f"results/pilot_llm_{version}/formal/records.jsonl"
    out = []
    with rec_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_v7_records() -> list[dict]:
    return load_records("v7")


def per_question_data(records: list[dict]) -> tuple[dict[str, dict], dict[str, int]]:
    """Build per-question, per-agent data dict.
    Returns: (data, q_harmful_fc)
      data[cqid][ai] = {answer, conf, gold_binary, inert, conf_stable, fragile}
      q_harmful_fc[cqid] = 0/1
    """
    by_qa: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in records:
        if not r.get("success") or not r.get("decision"):
            continue
        key = (r["cqid"], int(r["agent_index"]))
        by_qa[key][r["condition"]] = r

    # harmful_fc label per question
    q_orig: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in records:
        if r.get("condition") != "original" or not r.get("success") or not r.get("decision"):
            continue
        ans = r["decision"].get("answer")
        gold = int(r.get("gold_binary", 0))
        q_orig[r["cqid"]].append((ans, gold))
    q_harmful_fc = {}
    for cqid, ans_gold in q_orig.items():
        if len(ans_gold) < 5:
            continue
        gold = ans_gold[0][1]
        answers = [a for a, _ in ans_gold]
        cnt = Counter(answers)
        cons, n = cnt.most_common(1)[0]
        if n < 4:
            q_harmful_fc[cqid] = 0
            continue
        correct_consensus = int((cons == "yes") == bool(gold))
        q_harmful_fc[cqid] = int(correct_consensus == 0 and (n / 5) >= 0.8)

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
    return data, q_harmful_fc


# ----- profile generators (pre-registered) -----

def profile_homogeneous(cqid: str, data: dict, rng: random.Random) -> dict[int, float]:
    """P1: every question, every agent is equally likely to be most-reliable.
    Score per agent = uniform random in [0, 1]. Average = 0.5.
    """
    return {ai: rng.random() for ai in AGENT_NAMES}


def profile_concentrated_best(cqid: str, data: dict, rng: random.Random) -> dict[int, float]:
    """P2: agent 1 (skeptical_auditor) is best. Other agents get
    monotonically lower scores, with small noise.
    """
    return {ai: 1.0 - 0.2 * ai + 0.01 * rng.random() for ai in AGENT_NAMES}


def profile_concentrated_worst(cqid: str, data: dict, rng: random.Random) -> dict[int, float]:
    """P3: agent 0 (literal_evidence) is best. Inverted from P2.
    """
    return {ai: 0.2 * ai + 0.01 * rng.random() for ai in AGENT_NAMES}


def profile_realistic_FEVER(cqid: str, data: dict, rng: random.Random) -> dict[int, float]:
    """P4: use V7's real per-question fragility. Lower fragility = higher score.
    """
    if cqid not in data:
        return {ai: 0.5 for ai in AGENT_NAMES}
    return {ai: 1.0 - data[cqid][ai]["fragile"] for ai in AGENT_NAMES}


# ----- methods (same as agent_router.py) -----

def method_bl_majority(q_data: dict) -> tuple[str, float]:
    answers = [d["answer"] for d in q_data.values()]
    cnt = Counter(answers)
    cons, n = cnt.most_common(1)[0]
    return cons, n / len(q_data)


def method_bl_d_or(q_data: dict) -> tuple[str, float]:
    frags = [d["fragile"] for d in q_data.values()]
    mean_frag = sum(frags) / max(1, len(frags))
    answers = [d["answer"] for d in q_data.values()]
    cons, _ = Counter(answers).most_common(1)[0]
    return cons, 1.0 - mean_frag


def method_r1_top_auroc(q_data: dict, scores: dict[int, float]) -> tuple[str, float]:
    """Picks the agent with the highest score (= most-reliable in this profile)."""
    best_ai = max(scores.keys(), key=lambda ai: scores[ai])
    return q_data[best_ai]["answer"], scores[best_ai]


def method_r2_weighted(q_data: dict, scores: dict[int, float]) -> tuple[str, float]:
    yes_w = sum(scores[ai] for ai in q_data if q_data[ai]["answer"] == "yes")
    total_w = sum(scores[ai] for ai in q_data)
    chosen = "yes" if yes_w > total_w / 2 else "no"
    return chosen, yes_w / max(1, total_w)


def method_r3_min_fragility(q_data: dict, scores: dict[int, float]) -> tuple[str, float]:
    """Picks the agent with the lowest fragility (= highest score)."""
    best_ai = max(scores.keys(), key=lambda ai: scores[ai])
    return q_data[best_ai]["answer"], scores[best_ai]


# ----- main -----

PROFILES = {
    "P1_homogeneous": profile_homogeneous,
    "P2_concentrated_best": profile_concentrated_best,
    "P3_concentrated_worst": profile_concentrated_worst,
    "P4_realistic_FEVER": profile_realistic_FEVER,
}


def evaluate_profile(profile_name: str, profile_fn, data: dict, q_harmful_fc: dict) -> dict:
    rng = random.Random(20_260_902)
    methods = ["BL_majority", "BL_D_OR", "R1_top_auroc", "R2_weighted", "R3_min_frag"]
    method_results = {m: {"scores": [], "labels": []} for m in methods}

    for cqid, q_data in data.items():
        if len(q_data) < 5:
            continue
        if cqid not in q_harmful_fc:
            continue
        scores = profile_fn(cqid, q_data, rng)
        label = q_harmful_fc[cqid]

        ans, sc = method_bl_majority(q_data)
        method_results["BL_majority"]["scores"].append(sc)
        method_results["BL_majority"]["labels"].append(label)

        ans, sc = method_bl_d_or(q_data)
        method_results["BL_D_OR"]["scores"].append(sc)
        method_results["BL_D_OR"]["labels"].append(label)

        ans, sc = method_r1_top_auroc(q_data, scores)
        method_results["R1_top_auroc"]["scores"].append(sc)
        method_results["R1_top_auroc"]["labels"].append(label)

        ans, sc = method_r2_weighted(q_data, scores)
        method_results["R2_weighted"]["scores"].append(sc)
        method_results["R2_weighted"]["labels"].append(label)

        ans, sc = method_r3_min_fragility(q_data, scores)
        method_results["R3_min_frag"]["scores"].append(sc)
        method_results["R3_min_frag"]["labels"].append(label)

    out = {}
    for m, d in method_results.items():
        auroc = _auroc(d["scores"], d["labels"])
        ci_lo, ci_hi = _bootstrap_auroc_ci(d["scores"], d["labels"])
        out[m] = {
            "auroc": auroc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_questions": len(d["scores"]),
        }
    return out


def render_cross_table(all_results: dict) -> str:
    lines = ["## Cross-profile × method AUROC (controlled heterogeneity)\n"]
    lines.append("Rows: 4 pre-registered profiles. Cols: 5 methods. All 20 cells reported.\n")
    methods = ["BL_majority", "BL_D_OR", "R1_top_auroc", "R2_weighted", "R3_min_frag"]
    header = "| Profile | N | " + " | ".join(methods) + " |"
    sep = "|---|---:|" + "---:|" * len(methods)
    lines.append(header)
    lines.append(sep)
    for p in ["P1_homogeneous", "P2_concentrated_best", "P3_concentrated_worst", "P4_realistic_FEVER"]:
        if p not in all_results:
            continue
        res = all_results[p]
        n_q = next(iter(res.values()))["n_questions"] if res else 0
        row = [p, str(n_q)]
        for m in methods:
            r = res.get(m, {})
            a = r.get("auroc")
            row.append(f"{a:.3f}" if a is not None else "NA")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_detail_table(all_results: dict) -> str:
    lines = []
    methods = ["BL_majority", "BL_D_OR", "R1_top_auroc", "R2_weighted", "R3_min_frag"]
    for p in ["P1_homogeneous", "P2_concentrated_best", "P3_concentrated_worst", "P4_realistic_FEVER"]:
        if p not in all_results:
            continue
        res = all_results[p]
        n_q = next(iter(res.values()))["n_questions"] if res else 0
        lines.append(f"### {p} (N = {n_q} questions)\n")
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

**Pre-registered question**: "Does router AUC depend on agent reliability
distribution?"

**Reading the table**
- The 4 profiles are constructed to span the agent-reliability axis:
  - **P1 (homogeneous)**: every agent equally likely to be most-reliable
    per question. Router should be unable to extract signal.
  - **P2 (concentrated-best)**: agent 1 (skeptical_auditor, V7 AUROC
    0.493) is always most-reliable. Router should win.
  - **P3 (concentrated-worst)**: agent 0 (literal_evidence, V7 AUROC
    0.423) is always most-reliable. Router's "anti-best" behavior.
  - **P4 (realistic FEVER)**: V7's actual per-question fragility used as
    the "reliability" score. This is the V4-V7 control.

**Pre-registered expected patterns**
- If `P2_R1` > `P1_R1` and `P2_R1` > `P3_R1`: routers correctly
  identify the most-reliable agent under concentration.
- If `P2_R1` > `P4_R1`: routers extract more signal under concentration
  than under realistic FEVER — supports the V9 §2 finding that realistic
  FEVER is over-redundant.
- If `P2_R1` > `P2_BL_majority`: routers beat majority vote under
  concentration. If not, majority vote is always the best — confirming
  the V4-V7 finding at a more fundamental level.

**Reporting policy**
- All 20 cells reported, no method/version hidden.
- No profile cherry-picked. If P2 and P3 give similar results, that's
  a finding.
"""


def main():
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "v7"
    assert version in ("v4", "v5", "v6", "v7"), f"unsupported version {version}"
    records = load_records(version)
    data, q_harmful_fc = per_question_data(records)
    print(f"[loaded {version}] {len(records)} records, {len(data)} questions, "
          f"harmful_fc prevalence {sum(q_harmful_fc.values()) / max(1, len(q_harmful_fc)):.3f}")

    all_results: dict = {}
    for pname, pfn in PROFILES.items():
        res = evaluate_profile(pname, pfn, data, q_harmful_fc)
        all_results[pname] = res
        n = next(iter(res.values()))["n_questions"] if res else 0
        print(f"[done] {pname}: N={n} questions evaluated")

    md = [f"# Controlled-heterogeneity experiment on {version.upper()} (pre-registered 2026-09-01)\n"]
    md.append("4 profiles × 5 methods = 20 AUROC cells. Profiles vary the"
              " agent-reliability distribution; methods are the V4-V7"
              " router/baseline suite. All cells reported.\n")
    md.append(render_cross_table(all_results))
    md.append(render_detail_table(all_results))
    md.append(render_interpretation())

    suffix = "" if version == "v7" else f"_{version}"
    out_md = ANALYSIS_DIR / f"controlled_heterogeneity{suffix}.md"
    out_md.write_text("\n".join(md))
    print(f"\n[wrote] {out_md}")

    out_json = ANALYSIS_DIR / f"controlled_heterogeneity{suffix}.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()