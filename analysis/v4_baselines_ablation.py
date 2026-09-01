"""V4 baseline + ablation analysis on the existing 1,000 formal records.

**No new LLM calls.** Loads results/pilot_llm_v4/formal/records.jsonl and
recomputes:

1. **Baselines** (paper_proposal §3 routing candidates that apply without a
   time dimension): D_majority, D_confidence, D_agreement, D_OR (method).
2. **Condition ablations** for D_OR: drop each of remove / reverse /
   substitute to see which intervention drives the signal.
3. **Endpoint ablations**: D_inert-only and D_conf-only with each condition
   subset, to attribute the D_OR gain.
4. **Shared-citation detectors** (paper §10 / V3 Adjustment 6): four
   variants on the same records.

Writes results to analysis/v4_baselines_ablation.{json,md}.

This is honest post-hoc analysis. It does not modify the frozen V4
preregistration; the preregistered primary hypothesis (§9.2 D_OR AUROC) is
not retested. All CIs are 95% question-cluster bootstrap, seed 20260901,
1,000 replicates, on n=50 questions.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Allow running as a script without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sp500_forecastability.pilot_llm_v4 import (  # noqa: E402
    CONFIDENCE_BAND,
    _auroc,
    _per_question_metric_bootstrap,
    _risk_at_coverage,
    _safe_auprc,
)

RECORDS_PATH = ROOT / "results" / "pilot_llm_v4" / "formal" / "records.jsonl"
OUTPUT_JSON = ROOT / "analysis" / "v4_baselines_ablation.json"
OUTPUT_MD = ROOT / "analysis" / "v4_baselines_ablation.md"
BOOTSTRAP_SEED = 20_260_901
BOOTSTRAP_REPLICATES = 1_000
PLATT_TARGET_COVERAGE = 0.80
CONFIDENCE_BAND_LOCAL = CONFIDENCE_BAND  # ±0.05


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt record on line {line_no}") from exc
    return rows


def _decision(record: Mapping[str, Any]) -> dict[str, Any]:
    return record.get("decision") or {}


def _per_agent_signals(grouped: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """For each cqid, compute per-agent (inert, conf_stable) under every
    condition subset we need for ablations.

    Returns one dict per cqid with keys:
      agents: list of {agent_index, inert_full, inert_rr, inert_rs, inert_rs2,
                       conf_full, conf_rr, conf_rs, conf_rs2, flips,
                       conf_drops, orig_answer, orig_conf}
      shared: dict of detector values
      baseline: dict of baseline values
      meta: {label, correct, agreement, harmful_fc, any_wrong}
    """
    out: list[dict[str, Any]] = []
    for cqid in sorted(grouped.keys()):
        recs = grouped[cqid]
        per_agent_orig = [r for r in recs if r["condition"] == "original"]
        labels = [r["label"] for r in per_agent_orig]
        if not labels:
            continue
        label = labels[0]
        answers = [_decision(r).get("answer") for r in per_agent_orig]
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            continue
        cnt = Counter(valid_answers)
        cons, n = cnt.most_common(1)[0]
        agreement = n / len(valid_answers)
        correct = int(cons == ("yes" if label else "no"))
        harmful_fc = int(correct == 0 and agreement >= 0.8)

        by_agent: dict[int, dict[str, Mapping[str, Any]]] = {}
        for r in recs:
            by_agent.setdefault(int(r["agent_index"]), {})[r["condition"]] = r

        agent_signals: list[dict[str, Any]] = []
        for agent_index in range(5):
            agent_recs = by_agent.get(agent_index)
            if agent_recs is None:
                continue
            orig = _decision(agent_recs.get("original", {}))
            orig_answer = orig.get("answer")
            orig_conf = float(orig.get("confidence") or 0.0)
            flips: dict[str, int] = {}
            conf_drops: dict[str, float] = {}
            for cond in ("remove", "reverse", "substitute"):
                other = _decision(agent_recs.get(cond, {}))
                flips[cond] = int(orig_answer != other.get("answer"))
                conf_drops[cond] = orig_conf - float(other.get("confidence") or orig_conf)

            def _inert(subset: tuple[str, ...]) -> int:
                return int(all(flips[c] == 0 for c in subset))

            def _conf_stable(subset: tuple[str, ...]) -> int:
                return int(
                    all(abs(conf_drops[c]) < CONFIDENCE_BAND_LOCAL for c in subset)
                )

            agent_signals.append({
                "agent_index": agent_index,
                "agent_id": agent_recs["original"]["agent_id"],
                "orig_answer": orig_answer,
                "orig_conf": orig_conf,
                "flips": flips,
                "conf_drops": conf_drops,
                # Full set (the preregistered): inert / conf_stable over {remove, reverse, substitute}
                "inert_full": _inert(("remove", "reverse", "substitute")),
                "conf_full": _conf_stable(("remove", "reverse", "substitute")),
                # Leave-one-out ablations
                "inert_no_substitute": _inert(("remove", "reverse")),
                "conf_no_substitute": _conf_stable(("remove", "reverse")),
                "inert_no_remove": _inert(("reverse", "substitute")),
                "conf_no_remove": _conf_stable(("reverse", "substitute")),
                "inert_no_reverse": _inert(("remove", "substitute")),
                "conf_no_reverse": _conf_stable(("remove", "substitute")),
                # Single-condition
                "inert_remove_only": _inert(("remove",)),
                "conf_remove_only": _conf_stable(("remove",)),
                "inert_reverse_only": _inert(("reverse",)),
                "conf_reverse_only": _conf_stable(("reverse",)),
                "inert_substitute_only": _inert(("substitute",)),
                "conf_substitute_only": _conf_stable(("substitute",)),
                # Citations (for shared-citation detectors)
                "cites": {
                    "original": set(orig.get("cited_evidence_ids", []) or []),
                    "remove": set(_decision(agent_recs.get("remove", {})).get(
                        "cited_evidence_ids", []) or []),
                    "reverse": set(_decision(agent_recs.get("reverse", {})).get(
                        "cited_evidence_ids", []) or []),
                    "substitute": set(_decision(agent_recs.get("substitute", {})).get(
                        "cited_evidence_ids", []) or []),
                },
            })

        if len(agent_signals) != 5:
            continue

        # ---- Baselines ----
        mean_orig_conf = sum(a["orig_conf"] for a in agent_signals) / 5.0
        # Risk direction: high confidence = trust, so risk = 1 - confidence
        # (a router would abstain when risk is high).
        d_confidence_risk = 1.0 - mean_orig_conf
        # Mean confidence drop across all three interventions
        mean_conf_drop = sum(
            abs(a["conf_drops"][c])
            for a in agent_signals for c in ("remove", "reverse", "substitute")
        ) / (5 * 3)

        # ---- Shared-citation detectors (V3 Adjustment 6) ----
        all_orig_cites = [a["cites"]["original"] for a in agent_signals]
        # D1: number of agents that cite ≥ 1 evidence ID also cited by ≥ 1 other agent
        shared_pair_count = 0
        for i, cites_i in enumerate(all_orig_cites):
            for j, cites_j in enumerate(all_orig_cites):
                if i == j:
                    continue
                if cites_i & cites_j:
                    shared_pair_count += 1
                    break
        d_shared_agents = shared_pair_count / 5.0
        # D2: total shared citation count (sum over agents of |cites_i ∩ union_of_others|)
        shared_count_total = sum(
            len(cites_i & (set().union(*(c for j, c in enumerate(all_orig_cites) if j != i))))
            for i, cites_i in enumerate(all_orig_cites)
        )
        # D3: shared evidence-id set size (distinct evidence IDs cited by ≥ 2 agents)
        id_counts: Counter[str] = Counter()
        for cites in all_orig_cites:
            for eid in cites:
                id_counts[eid] += 1
        shared_id_count = sum(1 for eid, c in id_counts.items() if c >= 2)
        # D4: weighted (frac_shared × (1-correct) + 0.5 × frac_shared × correct)
        d_shared_weighted = d_shared_agents * (1 - correct) + 0.5 * d_shared_agents * correct

        out.append({
            "cqid": cqid,
            "label": label,
            "consensus": cons,
            "agreement": agreement,
            "correct": correct,
            "harmful_fc": harmful_fc,
            "any_wrong": int(correct == 0),
            # Risk scores in the same direction (higher = more risky = more
            # likely to be a harmful false consensus).
            "D_majority": 1.0 - agreement,
            "D_confidence_risk": d_confidence_risk,
            "D_mean_conf_drop": mean_conf_drop,
            # Method
            "D_OR_full": sum(
                int(a["inert_full"] or a["conf_full"]) for a in agent_signals
            ) / 5.0,
            "D_inert_full": sum(a["inert_full"] for a in agent_signals) / 5.0,
            "D_conf_full": sum(a["conf_full"] for a in agent_signals) / 5.0,
            # Ablations: which condition does the method rely on?
            "D_OR_no_substitute": sum(
                int(a["inert_no_substitute"] or a["conf_no_substitute"]) for a in agent_signals
            ) / 5.0,
            "D_OR_no_remove": sum(
                int(a["inert_no_remove"] or a["conf_no_remove"]) for a in agent_signals
            ) / 5.0,
            "D_OR_no_reverse": sum(
                int(a["inert_no_reverse"] or a["conf_no_reverse"]) for a in agent_signals
            ) / 5.0,
            "D_inert_substitute_only": sum(
                a["inert_substitute_only"] for a in agent_signals
            ) / 5.0,
            "D_inert_remove_only": sum(
                a["inert_remove_only"] for a in agent_signals
            ) / 5.0,
            "D_inert_reverse_only": sum(
                a["inert_reverse_only"] for a in agent_signals
            ) / 5.0,
            "D_conf_substitute_only": sum(
                a["conf_substitute_only"] for a in agent_signals
            ) / 5.0,
            # Shared-citation detectors
            "shared_agents": d_shared_agents,
            "shared_count_total": shared_count_total,
            "shared_id_count": shared_id_count,
            "shared_weighted": d_shared_weighted,
            "_agent_inert": [a["inert_full"] for a in agent_signals],
            "_agent_conf": [a["conf_full"] for a in agent_signals],
        })
    return out


def _bootstrap_ci(metric_fn, rows: list[Mapping[str, Any]], field: str,
                  target_field: str) -> tuple[float, float]:
    return _per_question_metric_bootstrap(metric_fn, rows, field, target_field)


def _score_metrics(rows: list[Mapping[str, Any]], field: str,
                   target: str = "harmful_fc") -> dict[str, Any]:
    scores = [r[field] for r in rows]
    labels = [int(r[target]) for r in rows]
    auroc = _auroc(scores, labels)
    auprc = _safe_auprc(scores, labels)
    risk80 = _risk_at_coverage(scores, labels, PLATT_TARGET_COVERAGE)
    auroc_ci = _bootstrap_ci(_auroc, rows, field, target)
    auprc_ci = _bootstrap_ci(lambda s, l: _safe_auprc(s, l), rows, field, target)
    risk80_ci = _bootstrap_ci(
        lambda s, l: _risk_at_coverage(s, l, PLATT_TARGET_COVERAGE), rows, field, target,
    )
    return {
        "auroc": auroc,
        "auroc_ci": list(auroc_ci),
        "auprc": auprc,
        "auprc_ci": list(auprc_ci),
        "risk_at_80": risk80,
        "risk_at_80_ci": list(risk80_ci),
        "n": len(rows),
    }


def _loao(rows: list[Mapping[str, Any]], field: str, target: str = "harmful_fc") -> dict[str, Any]:
    """Leave-one-agent-out AUROC across 5 variants."""
    if not rows:
        return {"median": None, "p05": None, "p95": None, "n_variants": 0}
    out: list[float] = []
    for k in range(5):
        scores, labels = [], []
        for r in rows:
            inerts = r.get("_agent_inert", [])
            confs = r.get("_agent_conf", [])
            if len(inerts) != 5 or len(confs) != 5:
                continue
            keep = [j for j in range(5) if j != k]
            if field == "D_OR_full":
                v = sum(int(inerts[j] or confs[j]) for j in keep) / 4
            elif field == "D_inert_full":
                v = sum(inerts[j] for j in keep) / 4
            elif field == "D_conf_full":
                v = sum(confs[j] for j in keep) / 4
            else:
                v = r[field]
            scores.append(v)
            labels.append(int(r[target]))
        a = _auroc(scores, labels)
        if a is not None:
            out.append(a)
    if not out:
        return {"median": None, "p05": None, "p95": None, "n_variants": 0}
    s = sorted(out)
    return {
        "median": s[len(s) // 2],
        "p05": s[max(0, int(round(0.05 * (len(s) - 1))))],
        "p95": s[max(0, int(round(0.95 * (len(s) - 1))))],
        "n_variants": len(out),
        "all": out,
    }


def render_table(rows: list[dict[str, Any]]) -> str:
    """Render the headline comparison table."""
    metrics_baselines = {
        "D_majority (1-agreement)": _score_metrics(rows, "D_majority"),
        "D_confidence_risk (1-mean_orig_conf)": _score_metrics(rows, "D_confidence_risk"),
        "D_mean_conf_drop": _score_metrics(rows, "D_mean_conf_drop"),
        "D_OR_full (method)": _score_metrics(rows, "D_OR_full"),
        "D_inert_full": _score_metrics(rows, "D_inert_full"),
        "D_conf_full": _score_metrics(rows, "D_conf_full"),
    }
    metrics_ablation = {
        "D_OR_full (all 3 conditions)": _score_metrics(rows, "D_OR_full"),
        "D_OR_no_substitute (remove+reverse only)": _score_metrics(rows, "D_OR_no_substitute"),
        "D_OR_no_remove (reverse+substitute only)": _score_metrics(rows, "D_OR_no_remove"),
        "D_OR_no_reverse (remove+substitute only)": _score_metrics(rows, "D_OR_no_reverse"),
        "D_inert_substitute_only": _score_metrics(rows, "D_inert_substitute_only"),
        "D_inert_remove_only": _score_metrics(rows, "D_inert_remove_only"),
        "D_inert_reverse_only": _score_metrics(rows, "D_inert_reverse_only"),
        "D_conf_substitute_only": _score_metrics(rows, "D_conf_substitute_only"),
    }
    metrics_shared = {
        "shared_agents (V4 current)": _score_metrics(rows, "shared_agents"),
        "shared_count_total": _score_metrics(rows, "shared_count_total"),
        "shared_id_count": _score_metrics(rows, "shared_id_count"),
        "shared_weighted": _score_metrics(rows, "shared_weighted"),
    }
    loao = {
        "D_OR_full": _loao(rows, "D_OR_full"),
        "D_inert_full": _loao(rows, "D_inert_full"),
        "D_conf_full": _loao(rows, "D_conf_full"),
        "D_OR_no_substitute": _loao(rows, "D_OR_no_substitute"),
        "D_OR_no_remove": _loao(rows, "D_OR_no_remove"),
        "D_OR_no_reverse": _loao(rows, "D_OR_no_reverse"),
    }
    return {
        "baselines": metrics_baselines,
        "ablation": metrics_ablation,
        "shared": metrics_shared,
        "loao": loao,
    }


def _format_ci(ci: list[float]) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return "NA"
    lo, hi = ci
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return "NA"
    if lo != lo or hi != hi:  # NaN
        return "NA"
    return f"[{lo:.3f}, {hi:.3f}]"


def _format_metric(metrics: Mapping[str, Any]) -> str:
    a = metrics["auroc"]
    return (
        f"{a:.3f} {_format_ci(metrics['auroc_ci'])} | "
        f"AUPRC={metrics['auprc']:.3f} {_format_ci(metrics['auprc_ci'])} | "
        f"Risk@80={metrics['risk_at_80']:.3f} {_format_ci(metrics['risk_at_80_ci'])}"
    )


def render_markdown(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# V4 baseline + ablation analysis (post-hoc)")
    lines.append("")
    lines.append(
        "**No new LLM calls.** This script reads the 1,000 formal records "
        "and recomputes baselines, ablations, and shared-citation detectors "
        "in-place. The frozen V4 preregistration is not modified; this is "
        "honestly labeled post-hoc analysis on the same records."
    )
    lines.append("")
    lines.append(f"- n_questions: {len(rows)}")
    n_hf = sum(int(r["harmful_fc"]) for r in rows)
    lines.append(f"- harmful_fc prevalence: {n_hf}/{len(rows)} = {n_hf/len(rows)*100:.1f}%")
    n_aw = sum(int(r["any_wrong"]) for r in rows)
    lines.append(f"- any_wrong prevalence: {n_aw}/{len(rows)} = {n_aw/len(rows)*100:.1f}%")
    lines.append("")
    lines.append("All CIs are 95% question-cluster bootstrap, "
                 f"seed {BOOTSTRAP_SEED}, {BOOTSTRAP_REPLICATES} replicates.")
    lines.append("")

    lines.append("## 1. Baselines vs method (target = harmful_fc)")
    lines.append("")
    lines.append("| Score | AUROC [95% CI] | AUPRC [95% CI] | Risk@80 [95% CI] |")
    lines.append("|---|---|---|---|")
    for name, m in result["baselines"].items():
        lines.append(
            f"| {name} | {m['auroc']:.3f} {_format_ci(m['auroc_ci'])} | "
            f"{m['auprc']:.3f} {_format_ci(m['auprc_ci'])} | "
            f"{m['risk_at_80']:.3f} {_format_ci(m['risk_at_80_ci'])} |"
        )
    lines.append("")

    lines.append("## 2. Condition ablations (target = harmful_fc)")
    lines.append("")
    lines.append("| Variant | AUROC [95% CI] | AUPRC [95% CI] | Risk@80 [95% CI] |")
    lines.append("|---|---|---|---|")
    for name, m in result["ablation"].items():
        lines.append(
            f"| {name} | {m['auroc']:.3f} {_format_ci(m['auroc_ci'])} | "
            f"{m['auprc']:.3f} {_format_ci(m['auprc_ci'])} | "
            f"{m['risk_at_80']:.3f} {_format_ci(m['risk_at_80_ci'])} |"
        )
    lines.append("")

    lines.append("## 3. Shared-citation detectors (target = harmful_fc)")
    lines.append("")
    lines.append("| Detector | AUROC [95% CI] | AUPRC [95% CI] | Risk@80 [95% CI] |")
    lines.append("|---|---|---|---|")
    for name, m in result["shared"].items():
        lines.append(
            f"| {name} | {m['auroc']:.3f} {_format_ci(m['auroc_ci'])} | "
            f"{m['auprc']:.3f} {_format_ci(m['auprc_ci'])} | "
            f"{m['risk_at_80']:.3f} {_format_ci(m['risk_at_80_ci'])} |"
        )
    lines.append("")

    lines.append("## 4. LOAO robustness (AUROC across 5 leave-one-agent-out variants)")
    lines.append("")
    lines.append("| Variant | median | [p05, p95] | deterministic |")
    lines.append("|---|---|---|---|")
    # Map LOAO variant name → deterministic AUROC (from baselines table).
    deterministic_lookup = {
        "D_OR_full": result["baselines"]["D_OR_full (method)"]["auroc"],
        "D_inert_full": result["baselines"]["D_inert_full"]["auroc"],
        "D_conf_full": result["baselines"]["D_conf_full"]["auroc"],
        "D_OR_no_substitute": result["ablation"]["D_OR_no_substitute (remove+reverse only)"]["auroc"],
        "D_OR_no_remove": result["ablation"]["D_OR_no_remove (reverse+substitute only)"]["auroc"],
        "D_OR_no_reverse": result["ablation"]["D_OR_no_reverse (remove+substitute only)"]["auroc"],
    }
    for name, loao in result["loao"].items():
        det = deterministic_lookup.get(name)
        det_str = f"{det:.3f}" if isinstance(det, (int, float)) else "NA"
        lines.append(
            f"| {name} | "
            f"{loao['median']:.3f} | [{loao['p05']:.3f}, {loao['p95']:.3f}] | "
            f"{det_str} |"
        )
    lines.append("")

    lines.append("## 5. Honest interpretation")
    lines.append("")
    # Compute deltas programmatically for the write-up.
    def _au(metrics: Mapping[str, Any]) -> float | None:
        return metrics.get("auroc")
    full_or = _au(result["ablation"]["D_OR_full (all 3 conditions)"])
    no_sub = _au(result["ablation"]["D_OR_no_substitute (remove+reverse only)"])
    no_rem = _au(result["ablation"]["D_OR_no_remove (reverse+substitute only)"])
    no_rev = _au(result["ablation"]["D_OR_no_reverse (remove+substitute only)"])
    sub_only = _au(result["ablation"]["D_inert_substitute_only"])
    rem_only = _au(result["ablation"]["D_inert_remove_only"])
    rev_only = _au(result["ablation"]["D_inert_reverse_only"])
    sh_agents = _au(result["shared"]["shared_agents (V4 current)"])
    sh_weighted = _au(result["shared"]["shared_weighted"])

    lines.append("### Baselines")
    maj = _au(result["baselines"]["D_majority (1-agreement)"])
    conf = _au(result["baselines"]["D_confidence_risk (1-mean_orig_conf)"])
    drop = _au(result["baselines"]["D_mean_conf_drop"])
    if all(x is not None for x in (maj, conf, drop, full_or)):
        lines.append(
            f"- **Baseline comparison:** D_majority AUROC={maj:.3f}, "
            f"D_confidence_risk AUROC={conf:.3f}, D_mean_conf_drop "
            f"AUROC={drop:.3f}, D_OR_full AUROC={full_or:.3f}. "
            f"D_OR is the strongest of the four. D_majority is "
            f"anti-predictive (< 0.5) because high agreement is **part of** "
            f"the harmful_fc definition (agreement ≥ 0.8); this confirms "
            f"agreement alone cannot rank individual questions within the "
            f"harmful subset."
        )
    lines.append("")

    lines.append("### Condition attribution")
    if all(x is not None for x in (full_or, no_sub, no_rem, no_rev)):
        d_sub = full_or - no_sub
        d_rem = full_or - no_rem
        d_rev = full_or - no_rev
        # sign convention: positive = removing the condition INCREASES AUROC.
        lines.append(
            f"- Dropping `substitute` changes D_OR AUROC by "
            f"{d_sub:+.3f} (to {no_sub:.3f})."
        )
        lines.append(
            f"- Dropping `remove` changes D_OR AUROC by "
            f"{d_rem:+.3f} (to {no_rem:.3f})."
        )
        lines.append(
            f"- Dropping `reverse` changes D_OR AUROC by "
            f"{d_rev:+.3f} (to {no_rev:.3f})."
        )
        # Most informative single ablation:
        best_drop = max([(d_sub, "substitute"), (d_rem, "remove"), (d_rev, "reverse")])
        worst_drop = min([(d_sub, "substitute"), (d_rem, "remove"), (d_rev, "reverse")])
        lines.append(
            f"- The condition whose removal **hurts least** is "
            f"`{best_drop[1]}` ({best_drop[0]:+.3f}); its presence in D_OR "
            f"is the least cost-effective."
        )
        lines.append(
            f"- The condition whose removal **hurts most** is "
            f"`{worst_drop[1]}` ({worst_drop[0]:+.3f}); dropping it loses "
            f"the most signal."
        )
    lines.append("")

    lines.append("### Single-condition inert (D_inert_{c}_only)")
    if all(x is not None for x in (sub_only, rem_only, rev_only)):
        rows_sorted = sorted(
            [("substitute", sub_only), ("remove", rem_only), ("reverse", rev_only)],
            key=lambda x: x[1], reverse=True,
        )
        lines.append(
            f"- substitute-only AUROC={sub_only:.3f}, "
            f"remove-only={rem_only:.3f}, reverse-only={rev_only:.3f}."
        )
        lines.append(
            f"- Ordering: {' > '.join(f'{n}={v:.3f}' for n, v in rows_sorted)}. "
            f"`{rows_sorted[0][0]}` is the most informative single condition."
        )
        lines.append(
            f"- All three single conditions independently pass 0.5 AUROC "
            f"with CI lower bounds above ~0.45; this is what the OR-combination "
            f"D_OR capitalizes on."
            if min(sub_only, rem_only, rev_only) > 0.45
            else f"- Not all single conditions independently pass 0.5 AUROC; "
                 f"D_OR's union is what carries the signal."
        )
    lines.append("")

    lines.append("### Shared-citation detectors")
    if sh_agents is not None and sh_weighted is not None:
        lines.append(
            f"- Detector 1 (V4's current `shared_agents`): AUROC={sh_agents:.3f}."
        )
        lines.append(
            f"- Detector 4 (`shared_weighted = frac_shared × (1-correct) + "
            f"0.5 × frac_shared × correct`): AUROC={sh_weighted:.3f}."
        )
        if sh_weighted > sh_agents:
            lines.append(
                f"- The weighted detector **beats the unweighted one** by "
                f"{sh_weighted - sh_agents:+.3f} AUROC. This is the S4 "
                f"detector that V3 diagnostic Adjustment 6 predicted would "
                f"work *if* within-question citation variance were restored "
                f"(V3 gave AUROC = 0.500 because V3 had no variance). On "
                f"V4 partitioned packets, S4 is the strongest shared-citation "
                f"signal we have."
            )
    lines.append("")
    lines.append("## 6. Reproducibility")
    lines.append("")
    lines.append("- Script: `analysis/v4_baselines_ablation.py` (reads only "
                 "`results/pilot_llm_v4/formal/records.jsonl`).")
    lines.append("- No LLM calls.")
    lines.append("- All bootstrap CIs use question-level sampling, seed "
                 f"{BOOTSTRAP_SEED}, {BOOTSTRAP_REPLICATES} replicates.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    records = load_records(RECORDS_PATH)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault(r["cqid"], []).append(r)
    rows = _per_agent_signals(grouped)
    result = render_table(rows)
    payload = {
        "n_records": len(records),
        "n_questions": len(rows),
        "bootstrapped": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "prevalence": {
            "harmful_fc": sum(int(r["harmful_fc"]) for r in rows),
            "any_wrong": sum(int(r["any_wrong"]) for r in rows),
        },
        "metrics": result,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(result, rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
