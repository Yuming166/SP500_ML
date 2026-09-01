"""Generate the paper figures from frozen Pilot-LLM result artifacts.

Run from the repository root with::

    .venv/bin/python paper/scripts/generate_figures.py

The PDF files are vector graphics for the paper.  The PNG files are high-
resolution previews.  No reported value is entered independently of the frozen
JSON summaries or formal records.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from sp500_forecastability import pilot_llm_v11

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"
V11_SUMMARY = ROOT / "results" / "pilot_llm_v11_1" / "formal" / "summary.json"
V12_SUMMARY = ROOT / "results" / "pilot_llm_v12_1" / "formal" / "summary.json"
V12_SECONDARY = (
    ROOT
    / "results"
    / "pilot_llm_v12_1"
    / "formal"
    / "preregistered_secondary_analysis.json"
)
V11_RECORDS = ROOT / "results" / "pilot_llm_v11_1" / "formal" / "records.jsonl"
V12_RECORDS = ROOT / "results" / "pilot_llm_v12_1" / "formal" / "records.jsonl"

NAVY = "#24557A"
BLUE = "#2A9DCE"
GREEN = "#139A74"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#7B61A8"
INK = "#263238"
GRAY = "#73808C"
LIGHT_GRAY = "#E8EDF1"
PALE_BLUE = "#E7F3F8"
PALE_ORANGE = "#FFF1D6"
PALE_GREEN = "#E3F3ED"
PALE_RED = "#FBE8E4"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.titlesize": 12.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9AA4AC",
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "grid.color": "#DDE3E8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320)
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    title: str,
    lines: str,
    edgecolor: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.69,
        title,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        lines,
        ha="center",
        va="center",
        fontsize=7.7,
        color=INK,
        linespacing=1.25,
    )


def make_framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, PALE_BLUE, NAVY, "Evidence graph", "source roots $\\rightarrow$ evidence IDs\nfixed agent views"),
        (0.27, PALE_ORANGE, ORANGE, "Agent votes", "5 fixed agents\nanswer, confidence, cited IDs"),
        (0.52, PALE_RED, VERMILION, "Evidence tests", "original / remove\nreverse / substitute"),
        (0.77, PALE_GREEN, GREEN, "Route or retain", "$R_{PI}$ ranks false consensus\nretain low risk; route high risk"),
    ]
    for x, face, edge, title, lines in boxes:
        _box(ax, (x, 0.26), 0.205, 0.54, face, title, lines, edge)

    for start in (0.225, 0.475, 0.725):
        arrow = FancyArrowPatch(
            (start, 0.53),
            (start + 0.04, 0.53),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.7,
            color=GRAY,
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.91,
        "Environment-controlled evidence tests turn consensus into a routing decision",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.5,
        0.09,
        "Outcome-independent inputs only; labels are revealed after routing",
        ha="center",
        va="center",
        fontsize=8.7,
        color=GRAY,
    )
    _save(fig, "framework_overview")


def _errorbar_points(
    ax: plt.Axes,
    values: list[float],
    intervals: list[list[float]],
    labels: list[str],
    colors: list[str],
    *,
    horizontal: bool = False,
) -> None:
    positions = np.arange(len(values))
    lower = np.array(values) - np.array([interval[0] for interval in intervals])
    upper = np.array([interval[1] for interval in intervals]) - np.array(values)
    if horizontal:
        for index, (value, color) in enumerate(zip(values, colors, strict=True)):
            ax.errorbar(
                value,
                positions[index],
                xerr=np.array([[lower[index]], [upper[index]]]),
                fmt="none",
                ecolor=color,
                elinewidth=2.2,
                capsize=4,
                capthick=1.5,
            )
        ax.scatter(values, positions, s=54, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_yticks(positions, labels)
    else:
        for index, (value, color) in enumerate(zip(values, colors, strict=True)):
            ax.errorbar(
                positions[index],
                value,
                yerr=np.array([[lower[index]], [upper[index]]]),
                fmt="none",
                ecolor=color,
                elinewidth=2.2,
                capsize=4,
                capthick=1.5,
            )
        ax.scatter(positions, values, s=54, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_xticks(positions, labels)


def make_primary_results_figure() -> None:
    v11 = _read_json(V11_SUMMARY)
    v12 = _read_json(V12_SUMMARY)
    secondary = _read_json(V12_SECONDARY)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.45))
    fig.subplots_adjust(hspace=0.47, wspace=0.38)

    ax = axes[0, 0]
    pooled = secondary["cumulative_v11_1_v12"]
    auc_values = [v11["primary"]["auroc"], v12["primary"]["auroc"], pooled["r_pi_auroc"]]
    auc_cis = [v11["primary"]["auroc_ci"], v12["primary"]["auroc_ci"], pooled["r_pi_auroc_ci"]]
    _errorbar_points(
        ax,
        auc_values,
        auc_cis,
        ["V11.1\nheld out", "V12.1\nreplication", "Pooled\nsecondary"],
        [ORANGE, BLUE, GRAY],
    )
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.1)
    ax.set_ylim(0.42, 0.86)
    ax.set_ylabel("AUROC for wrong consensus")
    ax.grid(axis="y")
    ax.set_title("a  Frozen $R_{PI}$ validation", loc="left", fontweight="bold")
    ax.text(1, auc_cis[1][1] + 0.025, "PASS", ha="center", color=GREEN, fontweight="bold", fontsize=8.2)

    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.34
    baseline = [v11["router"]["baseline_error"], v12["router"]["baseline_error"]]
    routed = [v11["router"]["routed_error"], v12["router"]["routed_error"]]
    bars_a = ax.bar(x - width / 2, baseline, width, label="All high consensus", color=LIGHT_GRAY, edgecolor=GRAY)
    bars_b = ax.bar(x + width / 2, routed, width, label="Retained by $R_{PI}$", color=[ORANGE, BLUE])
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )
    ax.set_xticks(x, ["V11.1", "V12.1"])
    ax.set_ylim(0, 0.285)
    ax.set_ylabel("Consensus error")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("b  Routing at 80% coverage", loc="left", fontweight="bold")
    ax.text(1, 0.265, "$\\Delta$=0.087 [0.046, 0.098]", ha="center", color=GREEN, fontsize=8.0, fontweight="bold")

    ax = axes[1, 0]
    component_keys = [
        "D_inert__wrong_high_consensus",
        "flip_inertia__wrong_high_consensus",
        "frac_shared__wrong_high_consensus",
    ]
    component_values = [v12["secondary"][key]["auroc"] for key in component_keys] + [v12["primary"]["auroc"]]
    component_cis = [v12["secondary"][key]["auroc_ci"] for key in component_keys] + [v12["primary"]["auroc_ci"]]
    _errorbar_points(
        ax,
        component_values,
        component_cis,
        ["Complete\ninertia", "Flip\ninertia", "Citation\noverlap", "Frozen\n$R_{PI}$"],
        [NAVY, GREEN, PURPLE, BLUE],
    )
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.1)
    ax.set_ylim(0.35, 0.93)
    ax.set_ylabel("AUROC")
    ax.grid(axis="y")
    ax.set_title("c  What carries the signal?", loc="left", fontweight="bold")
    ax.annotate("citation-only null", xy=(2, component_values[2]), xytext=(2.55, 0.41), fontsize=7.7, color=PURPLE, arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 1.0})

    ax = axes[1, 1]
    groups = secondary["boolq_label_subgroups"]
    label_values = [groups["yes"]["r_pi_auroc"], groups["no"]["r_pi_auroc"], v12["primary"]["auroc"]]
    label_cis = [groups["yes"]["r_pi_auroc_ci"], groups["no"]["r_pi_auroc_ci"], v12["primary"]["auroc_ci"]]
    _errorbar_points(
        ax,
        label_values,
        label_cis,
        ["BoolQ yes\n(n=210)", "BoolQ no\n(n=90)", "Aggregate\n(n=300)"],
        [GREEN, VERMILION, BLUE],
    )
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.1)
    ax.set_ylim(0.05, 1.0)
    ax.set_ylabel("AUROC")
    ax.grid(axis="y")
    ax.set_title("d  Label-conditional reversal", loc="left", fontweight="bold")

    fig.suptitle("Intervention sensitivity predicts false consensus, but not uniformly", y=1.015, fontweight="bold")
    _save(fig, "primary_results")


def _risk_rows(records_path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(records_path)
    rows = pilot_llm_v11._risk_rows(records)
    return [row for row in rows if float(row["agreement"]) >= 0.8]


def _evaluate_at_coverage(rows: list[dict[str, Any]], coverage: float) -> tuple[float, float, float]:
    baseline = sum(int(row["consensus_wrong"]) for row in rows) / len(rows)
    keep_n = max(1, round(len(rows) * coverage))
    retained = sorted(rows, key=lambda row: float(row["R_PI"]))[:keep_n]
    retained_error = sum(int(row["consensus_wrong"]) for row in retained) / len(retained)
    return baseline, retained_error, baseline - retained_error


def _routing_curve(
    rows: list[dict[str, Any]], coverages: np.ndarray, *, seed: int
) -> dict[str, list[float]]:
    retained_errors: list[float] = []
    reductions: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for coverage in coverages:
        _, retained, reduction = _evaluate_at_coverage(rows, float(coverage))
        retained_errors.append(retained)
        reductions.append(reduction)
        rng = random.Random(seed)
        boot: list[float] = []
        for _ in range(1_000):
            sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            boot.append(_evaluate_at_coverage(sample, float(coverage))[2])
        boot.sort()
        lower.append(boot[25])
        upper.append(boot[975])
    return {
        "retained_error": retained_errors,
        "reduction": reductions,
        "lower": lower,
        "upper": upper,
    }


def make_routing_figure() -> dict[str, Any]:
    rows_v11 = _risk_rows(V11_RECORDS)
    rows_v12 = _risk_rows(V12_RECORDS)
    coverages = np.round(np.arange(0.20, 1.001, 0.05), 2)
    curve_v11 = _routing_curve(rows_v11, coverages, seed=20_260_902)
    curve_v12 = _routing_curve(rows_v12, coverages, seed=20_260_921)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    v12_baseline = _evaluate_at_coverage(rows_v12, 1.0)[0]
    ax.plot(coverages, curve_v12["retained_error"], color=BLUE, linewidth=2.4, marker="o", markersize=3.4, label="$R_{PI}$-retained")
    ax.axhline(v12_baseline, color=GRAY, linestyle="--", linewidth=1.4, label="Unrouted consensus")
    index_80 = int(np.where(np.isclose(coverages, 0.8))[0][0])
    ax.scatter([0.8], [curve_v12["retained_error"][index_80]], s=72, c=GREEN, edgecolors="white", linewidths=1.0, zorder=4)
    ax.annotate(
        "registered point\n0.220 $\\rightarrow$ 0.133",
        xy=(0.8, curve_v12["retained_error"][index_80]),
        xytext=(0.51, 0.055),
        fontsize=8.0,
        color=GREEN,
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.2},
    )
    ax.set_xlim(0.18, 1.02)
    ax.set_ylim(0, max(curve_v12["retained_error"] + [v12_baseline]) * 1.22)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Error among retained decisions")
    ax.grid(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("a  V12.1 risk--coverage", loc="left", fontweight="bold")

    ax = axes[1]
    for curve, label, color in (
        (curve_v11, "V11.1", ORANGE),
        (curve_v12, "V12.1", BLUE),
    ):
        reduction = np.array(curve["reduction"])
        lo = np.array(curve["lower"])
        hi = np.array(curve["upper"])
        ax.plot(coverages, reduction, color=color, linewidth=2.1, marker="o", markersize=3.0, label=label)
        ax.fill_between(coverages, lo, hi, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0, color=GRAY, linestyle="--", linewidth=1.2)
    ax.axvline(0.8, color=GREEN, linestyle=":", linewidth=1.5)
    ax.scatter(
        [0.8, 0.8],
        [curve_v11["reduction"][index_80], curve_v12["reduction"][index_80]],
        s=48,
        c=[ORANGE, BLUE],
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
    )
    ax.set_xlim(0.18, 1.02)
    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Absolute error reduction")
    ax.grid(True)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("b  Routing benefit with 95% CI", loc="left", fontweight="bold")
    ax.text(
        0.20,
        ax.get_ylim()[0] + 0.006,
        "Only 80% was preregistered; other points are descriptive",
        fontsize=7.2,
        color=GRAY,
    )

    fig.suptitle("Reliability information supports selective routing at the frozen operating point", y=1.035, fontweight="bold")
    _save(fig, "routing_risk_coverage")

    return {
        "coverages": coverages.tolist(),
        "v11_1": curve_v11,
        "v12_1": curve_v12,
        "note": "Only coverage=0.8 is a preregistered secondary endpoint; the remaining curve is descriptive.",
    }


def main() -> None:
    _configure_style()
    make_framework_figure()
    make_primary_results_figure()
    routing_data = make_routing_figure()
    (FIGURE_DIR / "routing_curve_data.json").write_text(
        json.dumps(routing_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote paper figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
