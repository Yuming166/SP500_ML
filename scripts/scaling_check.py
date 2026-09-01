#!/usr/bin/env python3
"""
V5 PARTIAL_PASS 修复方案模拟（快速版）

用单层 question-cluster bootstrap（每个 N 跑 N_replicates 个 resample），
直接报告 CI lo / hi / width。要点：bootstrap 分布的 2.5% 分位数就是该 N
下 95% CI 下界的期望 —— 不需要嵌套。

总计算量：~7000 AUROC 评估，应在 <2 分钟内完成。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

CONFIDENCE_BAND = 0.05
N_AGENTS = 5
BOOTSTRAP_SEED = 20260902

V5_RECORDS = Path("/storage/gaoym/sp500-forecastability-lab/results/pilot_llm_v5/formal/records.jsonl")
OUT_JSON = Path("/storage/gaoym/sp500-forecastability-lab/results/pilot_llm_v5/scaling_check.json")


def _agent_signal(agent_recs):
    by_cond = {r["condition"]: r for r in agent_recs}
    for c in ("original", "remove", "reverse", "substitute"):
        if c not in by_cond or not by_cond[c].get("decision"):
            return {"complete": False}
    original = by_cond["original"]["decision"]
    orig_answer = original["answer"]
    orig_conf = float(original["confidence"])

    flips = {}
    conf_drops = {}
    citations = {}
    for cond in ("original", "remove", "reverse", "substitute"):
        d = by_cond[cond]["decision"]
        if cond == "original":
            flips[cond] = 0
            conf_drops[cond] = 0.0
        else:
            flips[cond] = int(d["answer"] != orig_answer)
            conf_drops[cond] = orig_conf - float(d["confidence"])
        citations[cond] = set(d.get("cited_evidence_ids", []) or [])

    inert = int(all(flips[c] == 0 for c in ("remove", "reverse", "substitute")))
    conf_stable = int(
        all(abs(conf_drops[c]) < CONFIDENCE_BAND
            for c in ("remove", "reverse", "substitute"))
    )
    return {"complete": True, "inert": inert, "conf_stable": conf_stable, "citations": citations}


def _per_question_risks(records):
    by_cqid = {}
    for r in records:
        by_cqid.setdefault(r["cqid"], []).append(r)

    rows = []
    for cqid, recs in sorted(by_cqid.items()):
        per_agent_orig = [r for r in recs if r["condition"] == "original"]
        if len(per_agent_orig) != N_AGENTS:
            continue
        gold_binary = int(per_agent_orig[0]["gold_binary"])
        answers = [r["decision"]["answer"] for r in per_agent_orig if r["decision"]]
        cnt = Counter(answers)
        cons, n = cnt.most_common(1)[0]
        agreement = n / len(answers)
        correct = int((cons == "yes") == bool(gold_binary))
        harmful_fc = int(correct == 0 and agreement >= 0.8)

        by_ai = {}
        for r in recs:
            by_ai.setdefault(r["agent_index"], []).append(r)

        agent_signals = []
        shared_count = 0
        for ai in sorted(by_ai.keys()):
            sig = _agent_signal(by_ai[ai])
            if not sig.get("complete"):
                continue
            agent_signals.append(sig)
            orig_cites = sig["citations"]["original"]
            for other_sig in agent_signals[:-1]:
                if orig_cites & other_sig["citations"]["original"]:
                    shared_count += 1
                    break
        if len(agent_signals) != N_AGENTS:
            continue

        d_or = sum(int(s["inert"] or s["conf_stable"]) for s in agent_signals) / N_AGENTS
        frac_shared = shared_count / N_AGENTS
        shared_weighted = frac_shared * (1 - correct) + 0.5 * frac_shared * correct

        rows.append({
            "cqid": cqid, "harmful_fc": harmful_fc,
            "D_OR": d_or, "shared_weighted": shared_weighted,
        })
    return rows


def _bootstrap_at_n(rows, target_n, n_replicates, seed):
    """
    Single-layer question-cluster bootstrap.
    Stratify by harmful_fc to preserve prevalence.
    Returns: dict with CI stats and power estimates.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.array([i for i, r in enumerate(rows) if r["harmful_fc"] == 1])
    neg_idx = np.array([i for i, r in enumerate(rows) if r["harmful_fc"] == 0])

    pos_frac = len(pos_idx) / (len(pos_idx) + len(neg_idx))
    n_pos_take = max(1, round(target_n * pos_frac))
    n_neg_take = max(1, target_n - n_pos_take)

    dor_scores = np.empty(n_replicates)
    sw_scores = np.empty(n_replicates)

    rows_arr = np.asarray(rows, dtype=object)

    for rep in range(n_replicates):
        # Stratified sampling with replacement
        sp = rng.choice(pos_idx, size=n_pos_take, replace=True)
        sn = rng.choice(neg_idx, size=n_neg_take, replace=True)
        sampled = np.concatenate([sp, sn])
        sub = rows_arr[sampled]

        scores_dor = np.asarray([r["D_OR"] for r in sub])
        scores_sw = np.asarray([r["shared_weighted"] for r in sub])
        labels = np.asarray([r["harmful_fc"] for r in sub])

        if len(set(labels.tolist())) < 2:
            dor_scores[rep] = np.nan
            sw_scores[rep] = np.nan
            continue

        dor_scores[rep] = roc_auc_score(labels, scores_dor)
        sw_scores[rep] = roc_auc_score(labels, scores_sw)

    valid_dor = dor_scores[~np.isnan(dor_scores)]
    valid_sw = sw_scores[~np.isnan(sw_scores)]

    out = {
        "target_n": target_n,
        "n_pos_take": n_pos_take,
        "n_neg_take": n_neg_take,
        "n_replicates": n_replicates,
        "n_valid_dor": int(len(valid_dor)),
        "n_valid_sw": int(len(valid_sw)),
    }
    if len(valid_dor):
        out["D_OR"] = {
            "mean": float(valid_dor.mean()),
            "median": float(np.median(valid_dor)),
            "ci_lo_2.5pct": float(np.quantile(valid_dor, 0.025)),
            "ci_hi_97.5pct": float(np.quantile(valid_dor, 0.975)),
            "ci_width": float(np.quantile(valid_dor, 0.975) - np.quantile(valid_dor, 0.025)),
            "p_point_above_0.5": float((valid_dor > 0.5).mean()),
        }
    if len(valid_sw):
        out["shared_weighted"] = {
            "mean": float(valid_sw.mean()),
            "median": float(np.median(valid_sw)),
            "ci_lo_2.5pct": float(np.quantile(valid_sw, 0.025)),
            "ci_hi_97.5pct": float(np.quantile(valid_sw, 0.975)),
            "ci_width": float(np.quantile(valid_sw, 0.975) - np.quantile(valid_sw, 0.025)),
            "p_point_above_0.5": float((valid_sw > 0.5).mean()),
        }

    # Power estimation: split the n_replicates into chunks of ~40
    # Each chunk's 2.5%ile approximates a fresh CI lo
    chunk_size = 40
    if len(valid_dor) >= chunk_size:
        n_chunks = len(valid_dor) // chunk_size
        chunk_cis = []
        for k in range(n_chunks):
            chunk = valid_dor[k*chunk_size:(k+1)*chunk_size]
            chunk_cis.append(np.quantile(chunk, 0.025))
        out["D_OR_p_chunk_above_0.5"] = float(np.mean([c > 0.5 for c in chunk_cis]))
        out["D_OR_n_chunks"] = n_chunks
    if len(valid_sw) >= chunk_size:
        n_chunks = len(valid_sw) // chunk_size
        chunk_cis = []
        for k in range(n_chunks):
            chunk = valid_sw[k*chunk_size:(k+1)*chunk_size]
            chunk_cis.append(np.quantile(chunk, 0.025))
        out["shared_weighted_p_chunk_above_0.5"] = float(np.mean([c > 0.5 for c in chunk_cis]))
        out["shared_weighted_n_chunks"] = n_chunks
    return out


def main():
    print("=" * 70)
    print("V5 PARTIAL_PASS 修复方案模拟（快速版）")
    print("=" * 70)

    records = []
    with V5_RECORDS.open() as f:
        for line in f:
            records.append(json.loads(line))
    print(f"\n加载 {len(records)} records")

    rows = _per_question_risks(records)
    n_pos = sum(1 for r in rows if r["harmful_fc"] == 1)
    n_neg = sum(1 for r in rows if r["harmful_fc"] == 0)
    print(f"N = {len(rows)} questions  (pos={n_pos}, neg={n_neg}, prev={n_pos/len(rows):.3f})")

    det_dor = roc_auc_score([r["harmful_fc"] for r in rows], [r["D_OR"] for r in rows])
    det_sw = roc_auc_score([r["harmful_fc"] for r in rows], [r["shared_weighted"] for r in rows])
    print(f"\nDeterministic AUROC on full N=50:")
    print(f"  D_OR:            {det_dor:.4f}")
    print(f"  shared_weighted: {det_sw:.4f}")

    target_ns = [25, 50, 75, 100, 125, 150, 200]
    n_replicates = 2000  # per N
    print(f"\n跑单层 bootstrap: {n_replicates} replicates × {len(target_ns)} 个 N 值...")
    print("=" * 70)

    results = []
    for n in target_ns:
        res = _bootstrap_at_n(rows, n, n_replicates=n_replicates, seed=BOOTSTRAP_SEED + n)
        results.append(res)
        d = res.get("D_OR", {})
        s = res.get("shared_weighted", {})
        line = (f"\n[N={n:3d}]  n_pos={res['n_pos_take']}, n_neg={res['n_neg_take']}\n"
                f"  D_OR:            med={d.get('median', 0):.3f}  CI=[{d.get('ci_lo_2.5pct', 0):.3f}, {d.get('ci_hi_97.5pct', 0):.3f}]  "
                f"width={d.get('ci_width', 0):.3f}  P(lo>0.5)={res.get('D_OR_p_chunk_above_0.5', 'NA')}\n"
                f"  shared_weighted: med={s.get('median', 0):.3f}  CI=[{s.get('ci_lo_2.5pct', 0):.3f}, {s.get('ci_hi_97.5pct', 0):.3f}]  "
                f"width={s.get('ci_width', 0):.3f}  P(lo>0.5)={res.get('shared_weighted_p_chunk_above_0.5', 'NA')}")
        print(line)

    print("\n" + "=" * 70)
    print("V6 N 选择决策表（P(chunk CI lo > 0.5)）")
    print("=" * 70)
    print(f"{'N':>4}  {'D_OR':>8}  {'sw':>8}  {'verdict':>15}")
    for r in results:
        n = r["target_n"]
        dp = r.get("D_OR_p_chunk_above_0.5", 0) or 0
        sp = r.get("shared_weighted_p_chunk_above_0.5", 0) or 0
        if dp >= 0.8 and sp >= 0.8:
            verdict = "✅ 双过 (推荐)"
        elif dp >= 0.5 and sp >= 0.5:
            verdict = "🟡 边界"
        else:
            verdict = "❌ 不过"
        print(f"{n:>4}  {dp:>8.2f}  {sp:>8.2f}  {verdict:>15}")

    print("\n" + "=" * 70)
    print("CI 宽度收敛趋势")
    print("=" * 70)
    print(f"{'N':>4}  {'D_OR  width':>15}  {'sw  width':>15}  {'D_OR  CI lo':>15}  {'sw  CI lo':>15}")
    for r in results:
        n = r["target_n"]
        d = r.get("D_OR", {})
        s = r.get("shared_weighted", {})
        print(f"{n:>4}  {d.get('ci_width', 0):>15.3f}  {s.get('ci_width', 0):>15.3f}  "
              f"{d.get('ci_lo_2.5pct', 0):>15.3f}  {s.get('ci_lo_2.5pct', 0):>15.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump({
            "n_questions_total": len(rows),
            "n_pos": n_pos, "n_neg": n_neg,
            "deterministic": {"D_OR": det_dor, "shared_weighted": det_sw},
            "subsample_results": results,
        }, f, indent=2)
    print(f"\nSaved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()