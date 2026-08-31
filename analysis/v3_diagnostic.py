"""
V3 diagnostic reanalysis.

Reuses the existing 750 V3 records (no LLM calls). Computes:
  - Adjustment 5: alternative causal-risk definitions vs. AUROC for harmful false consensus
  - Adjustment 6: shared-citation-cluster detector vs. AUROC for harmful false consensus

All risk definitions predict the SAME outcome: per-question "is the consensus wrong?"
so AUROC is over the 50 questions (5 agents per question).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

RECORDS = Path("/storage/gaoym/sp500-forecastability-lab/results/pilot_llm_v3/formal/records.jsonl")
OUT = Path("/storage/gaoym/sp500-forecastability-lab/analysis/v3_diagnostic_report.md")

records = [json.loads(l) for l in RECORDS.open()]
records = [r for r in records if r.get("success")]

# Build (qid, agent_id) -> {original, remove, reverse} triplets
triplets: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
for r in records:
    triplets[(r["qid"], r["agent_id"])][r["condition"]] = r

complete = {k: v for k, v in triplets.items() if all(c in v for c in ("original", "remove", "reverse"))}
print(f"Total records: {len(records)}; complete triplets: {len(complete)}")

# ---------- 1. Per-question aggregation ----------
def answer_str(label: bool) -> str:
    return "yes" if label else "no"

per_q: dict[str, dict] = {}
for (qid, agent), t in complete.items():
    if qid not in per_q:
        per_q[qid] = {
            "label": t["original"]["label"],
            "agents": {},
        }
    orig_ans = t["original"]["decision"]["answer"]
    rm_ans = t["remove"]["decision"]["answer"]
    rv_ans = t["reverse"]["decision"]["answer"]
    orig_conf = t["original"]["decision"]["confidence"]
    rm_conf = t["remove"]["decision"]["confidence"]
    rv_conf = t["reverse"]["decision"]["confidence"]

    # answer-flip indicators (1 = flipped)
    rm_flipped = int(orig_ans != rm_ans)
    rv_flipped = int(orig_ans != rv_ans)
    inert = int(rm_flipped == 0 and rv_flipped == 0)

    # confidence drops (max drop across interventions)
    conf_drop_rm = orig_conf - rm_conf
    conf_drop_rv = orig_conf - rv_conf
    max_conf_drop = max(conf_drop_rm, conf_drop_rv)

    # confidence stability = 1 if both drops < 0.05 (kept within 5%)
    conf_stable = int(abs(conf_drop_rm) < 0.05 and abs(conf_drop_rv) < 0.05)

    per_q[qid]["agents"][agent] = {
        "original_answer": orig_ans,
        "remove_answer": rm_ans,
        "reverse_answer": rv_ans,
        "original_confidence": orig_conf,
        "remove_confidence": rm_conf,
        "reverse_confidence": rv_conf,
        "rm_flipped": rm_flipped,
        "rv_flipped": rv_flipped,
        "inert": inert,
        "conf_drop_rm": conf_drop_rm,
        "conf_drop_rv": conf_drop_rv,
        "max_conf_drop": max_conf_drop,
        "conf_stable": conf_stable,
        "correct": int(orig_ans == answer_str(t["original"]["label"])),
    }

print(f"Unique questions: {len(per_q)}")

# ---------- 2. Per-question outcome (target for AUROC) ----------
# Original V3 target = "harmful false consensus" = consensus is wrong AND agreement >= 0.8
def majority(answers: list[str]) -> tuple[str, float]:
    cnt = Counter(answers)
    top, n = cnt.most_common(1)[0]
    return top, n / len(answers)

# Build per-question outputs
q_rows = []
for qid, info in per_q.items():
    agents = info["agents"]
    answers = [a["original_answer"] for a in agents.values()]
    cons, agr = majority(answers)
    correct = int(cons == answer_str(info["label"]))
    harmful_fc = int(correct == 0 and agr >= 0.8)

    # shared-citation stats
    citations = [set(a.get("cited_evidence_ids", []) or []) for a in agents.values()]
    # evidence-id cited by ≥2 agents (i.e. shared cluster, ignoring empty)
    nonempty = [c for c in citations if c]
    flat = []
    for c in nonempty:
        flat.extend(c)
    cite_counts = Counter(flat)
    n_shared = sum(1 for v in cite_counts.values() if v >= 2)
    max_shared_citations = max(cite_counts.values()) if cite_counts else 0
    frac_shared_agents = (
        sum(1 for c in citations if any(cite_counts[e] >= 2 for e in c)) / len(citations)
        if citations else 0.0
    )

    q_rows.append({
        "qid": qid,
        "label": info["label"],
        "consensus": cons,
        "agreement": agr,
        "correct": correct,
        "harmful_fc": harmful_fc,
        "agents": agents,
        "n_shared_citations": n_shared,
        "max_shared_citations": max_shared_citations,
        "frac_shared_agents": frac_shared_agents,
    })

# ---------- 3. Risk-score definitions (Adjustment 5) ----------
def risk_inert(q):
    a = q["agents"]
    return sum(v["inert"] for v in a.values()) / len(a)

def risk_conf_stable(q):
    a = q["agents"]
    return sum(v["conf_stable"] for v in a.values()) / len(a)

def risk_inert_and_confstable(q):
    a = q["agents"]
    return sum(int(v["inert"] and v["conf_stable"]) for v in a.values()) / len(a)

def risk_inert_or_confstable(q):
    a = q["agents"]
    return sum(int(v["inert"] or v["conf_stable"]) for v in a.values()) / len(a)

def risk_composite_5050(q):
    return 0.5 * risk_inert(q) + 0.5 * risk_conf_stable(q)

def risk_min_conf_drop(q):
    a = q["agents"]
    return 1.0 - mean(max(v["conf_drop_rm"], v["conf_drop_rv"]) for v in a.values())

def risk_min_conf_drop_clip(q):
    # clipped: max drop per agent ∈ [0,1]; risk = 1 - clipped
    a = q["agents"]
    return 1.0 - mean(min(1.0, max(v["conf_drop_rm"], v["conf_drop_rv"])) for v in a.values())

def risk_n_inert(q):
    a = q["agents"]
    return float(sum(v["inert"] for v in a.values()))

RISK_FNS = {
    "A_orig_inert_only":              risk_inert,
    "B_conf_stable_only":             risk_conf_stable,
    "C_inert_AND_confstable":         risk_inert_and_confstable,
    "D_inert_OR_confstable":          risk_inert_or_confstable,
    "E_composite_50_50":              risk_composite_5050,
    "F_1_minus_max_conf_drop":        risk_min_conf_drop,
    "G_1_minus_max_conf_drop_clipped": risk_min_conf_drop_clip,
    "H_n_inert_count":                risk_n_inert,
}

# ---------- 4. Shared-citation detector scores (Adjustment 6) ----------
def det_max_shared(q):
    # count of agents sharing the most-cited evidence ID (≥2 sharing)
    return float(q["max_shared_citations"])

def det_n_shared_ids(q):
    return float(q["n_shared_citations"])

def det_frac_shared_agents(q):
    return q["frac_shared_agents"]

def det_shared_x_wrong(q):
    # if consensus wrong AND shared cluster exists -> high score
    base = q["frac_shared_agents"]
    return base * (1 - q["correct"]) + 0.5 * base * q["correct"]  # wrong boosts, but presence still counts

DET_FNS = {
    "S1_max_shared_citation_count": det_max_shared,
    "S2_n_shared_evidence_ids":     det_n_shared_ids,
    "S3_frac_agents_with_shared":   det_frac_shared_agents,
    "S4_shared_x_wrong":            det_shared_x_wrong,
}

# ---------- 5. AUROC ----------
def auroc(scores: list[float], labels: list[int]) -> float | None:
    """AUROC for binary label; returns None if all labels same."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    n_pos, n_neg = len(pos), len(neg)
    # Mann-2 wins / total pairs
    wins = 0.0
    for s_pos in pos:
        for s_neg in neg:
            if s_pos > s_neg: wins += 1
            elif s_pos == s_neg: wins += 0.5
    return wins / (n_pos * n_neg)

print("\n========== ADJUSTMENT 5: Risk definitions vs. harmful false consensus ==========")
labels = [q["harmful_fc"] for q in q_rows]
print(f"Questions: {len(q_rows)}; harmful false consensus count: {sum(labels)}; prevalence: {sum(labels)/len(labels)*100:.1f}%")
results5 = []
for name, fn in RISK_FNS.items():
    scores = [fn(q) for q in q_rows]
    # Higher score = higher risk; positive class = harmful_fc==1
    a = auroc(scores, labels)
    results5.append((name, a))
    print(f"  {name:40s} AUROC = {a:.3f}" if a is not None else f"  {name:40s} AUROC = N/A")

print("\n========== ADJUSTMENT 6: Shared-citation detector vs. harmful false consensus ==========")
results6 = []
for name, fn in DET_FNS.items():
    scores = [fn(q) for q in q_rows]
    a = auroc(scores, labels)
    results6.append((name, a))
    print(f"  {name:40s} AUROC = {a:.3f}" if a is not None else f"  {name:40s} AUROC = N/A")

# ---------- 6. Cross-check: also try predicting raw consensus correctness (not just harmful_fc) ----------
print("\n========== Cross-check: predicting consensus correctness (any wrong consensus) ==========")
labels_any_wrong = [int(q["correct"] == 0) for q in q_rows]
print(f"Wrong-consensus count: {sum(labels_any_wrong)}; prevalence: {sum(labels_any_wrong)/len(labels_any_wrong)*100:.1f}%")
print("\nRisk definitions:")
results5b = []
for name, fn in RISK_FNS.items():
    scores = [fn(q) for q in q_rows]
    a = auroc(scores, labels_any_wrong)
    results5b.append((name, a))
    print(f"  {name:40s} AUROC = {a:.3f}" if a is not None else f"  {name:40s} AUROC = N/A")
print("\nShared-citation detectors:")
results6b = []
for name, fn in DET_FNS.items():
    scores = [fn(q) for q in q_rows]
    a = auroc(scores, labels_any_wrong)
    results6b.append((name, a))
    print(f"  {name:40s} AUROC = {a:.3f}" if a is not None else f"  {name:40s} AUROC = N/A")

# ---------- 7. Worst-case baselines (sanity) ----------
print("\n========== Sanity baselines (random / fixed scores) ==========")
import random
random.seed(0)
random_scores = [random.random() for _ in q_rows]
print(f"  Random score AUROC on harmful_fc:        {auroc(random_scores, labels):.3f}")
print(f"  Random score AUROC on any-wrong:         {auroc(random_scores, labels_any_wrong):.3f}")
# Confidence-mean (no intervention): high confidence wrong is a known weak signal
mean_conf_scores = [mean(v["original_confidence"] for v in q["agents"].values()) for q in q_rows]
print(f"  Mean original confidence AUROC (any-wrong): {auroc(mean_conf_scores, labels_any_wrong):.3f}")
# Agreement: high agreement wrong = harmful_fc definition itself
agree_scores = [q["agreement"] for q in q_rows]
print(f"  Agreement AUROC on harmful_fc:           {auroc(agree_scores, labels):.3f}")

# ---------- 8. Persist a small JSON next to the report for reproducibility ----------
results_blob = {
    "n_records": len(records),
    "n_complete_triplets": len(complete),
    "n_questions": len(per_q),
    "harmful_fc_count": sum(labels),
    "any_wrong_consensus_count": sum(labels_any_wrong),
    "adjustment_5_auroc_for_harmful_fc":  dict(results5),
    "adjustment_6_auroc_for_harmful_fc":  dict(results6),
    "adjustment_5_auroc_for_any_wrong":   dict(results5b),
    "adjustment_6_auroc_for_any_wrong":   dict(results6b),
}
(Path("/storage/gaoym/sp500-forecastability-lab/analysis") / "v3_diagnostic_results.json").write_text(
    json.dumps(results_blob, ensure_ascii=False, indent=2)
)
print(f"\nWrote results JSON to analysis/v3_diagnostic_results.json")