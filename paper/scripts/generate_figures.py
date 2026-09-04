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
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

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


def _save(fig: plt.Figure, stem: str, *, write_svg: bool = False) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    facecolor = fig.get_facecolor()
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", facecolor=facecolor)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, facecolor=facecolor)
    if write_svg:
        svg_path = FIGURE_DIR / f"{stem}.svg"
        fig.savefig(svg_path, facecolor=facecolor)
        svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
        svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
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
    background = "#FFFDF7"
    soft_ink = "#35414A"
    muted = "#83909A"
    shadow = "#E9E3D7"
    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    fig.patch.set_facecolor(background)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def rounded_card(
        left: float,
        bottom: float,
        width: float,
        height: float,
        face: str,
        edge: str,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (left + 0.006, bottom - 0.009),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.026",
                linewidth=0,
                facecolor=shadow,
                alpha=0.75,
                zorder=0,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.026",
                linewidth=1.5,
                edgecolor=edge,
                facecolor=face,
                zorder=1,
            )
        )

    def step_badge(center: tuple[float, float], number: str, color: str) -> None:
        center_x, center_y = center
        ax.add_patch(Circle((center_x, center_y), 0.022, facecolor=color, edgecolor="white", linewidth=1.0, zorder=4))
        ax.text(
            center_x,
            center_y - 0.001,
            number,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="white",
            zorder=5,
        )

    def face(center: tuple[float, float], size: float, color: str, wink: bool = False) -> None:
        center_x, center_y = center
        head_height = size * 0.80
        ax.add_patch(
            Ellipse(
                (center_x, center_y),
                size,
                head_height,
                facecolor=color,
                edgecolor=soft_ink,
                linewidth=0.8,
                zorder=4,
            )
        )
        eye_offset = size * 0.19
        eye_y = center_y + size * 0.08
        if wink:
            ax.plot(
                [center_x - eye_offset * 1.35, center_x - eye_offset * 0.75],
                [eye_y, eye_y + size * 0.01],
                color=soft_ink,
                linewidth=0.8,
                zorder=5,
            )
        else:
            ax.add_patch(Circle((center_x - eye_offset, eye_y), size * 0.035, color=soft_ink, zorder=5))
        ax.add_patch(Circle((center_x + eye_offset, eye_y), size * 0.035, color=soft_ink, zorder=5))
        ax.add_patch(
            Arc(
                (center_x, center_y - size * 0.01),
                size * 0.28,
                size * 0.18,
                theta1=205,
                theta2=335,
                color=soft_ink,
                linewidth=0.8,
                zorder=5,
            )
        )
        ax.add_patch(Ellipse((center_x - size * 0.30, center_y - size * 0.08), size * 0.10, size * 0.045, color="#F4A6A0", alpha=0.75, zorder=5))
        ax.add_patch(Ellipse((center_x + size * 0.30, center_y - size * 0.08), size * 0.10, size * 0.045, color="#F4A6A0", alpha=0.75, zorder=5))

    def document(center: tuple[float, float], width: float, height: float, color: str) -> None:
        center_x, center_y = center
        left = center_x - width / 2
        bottom = center_y - height / 2
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (left + width * 0.66, bottom + height),
                    (left + width, bottom + height * 0.66),
                    (left + width * 0.66, bottom + height * 0.66),
                ],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.3,
                zorder=4,
            )
        )
        for line_index in range(2):
            line_y = bottom + height * (0.43 - line_index * 0.16)
            ax.plot(
                [left + width * 0.18, left + width * 0.78],
                [line_y, line_y],
                color=color,
                linewidth=0.8,
                alpha=0.75,
                zorder=4,
            )

    def sparkle(center: tuple[float, float], color: str, size: float = 0.014) -> None:
        center_x, center_y = center
        ax.plot([center_x - size, center_x + size], [center_y, center_y], color=color, linewidth=1.1, zorder=4)
        ax.plot([center_x, center_x], [center_y - size, center_y + size], color=color, linewidth=1.1, zorder=4)

    def pill(left: float, bottom: float, width: float, text: str, face: str, edge: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                0.052,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8,
                zorder=3,
            )
        )
        ax.text(
            left + width / 2,
            bottom + 0.026,
            text,
            ha="center",
            va="center",
            fontsize=6.8,
            color=soft_ink,
            zorder=4,
        )

    def arrow(start_x: float, end_x: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (start_x, 0.535),
                (end_x, 0.535),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.6,
                color=muted,
                zorder=2,
            )
        )

    title = "How the evidence-aware audit catches false consensus"
    ax.text(0.5, 0.955, title, ha="center", va="center", fontsize=12.0, fontweight="bold", color=soft_ink)
    ax.text(
        0.5,
        0.895,
        "Change the evidence, watch the behavior, then unlock the labels",
        ha="center",
        va="center",
        fontsize=8.5,
        color=muted,
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.405, 0.825),
            0.19,
            0.038,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            facecolor="#F2EEE5",
            edgecolor="#D8D0C3",
            linewidth=0.7,
            zorder=2,
        )
    )
    ax.text(0.5, 0.844, "OUTCOME-FIREWALLED PIPELINE", ha="center", va="center", fontsize=6.6, fontweight="bold", color=muted, zorder=3)

    card_bottom = 0.255
    card_height = 0.515
    card_specs = [
        (0.03, 0.205, PALE_BLUE, NAVY),
        (0.27, 0.205, PALE_ORANGE, ORANGE),
        (0.51, 0.205, PALE_RED, VERMILION),
        (0.75, 0.22, PALE_GREEN, GREEN),
    ]
    for left, width, face_color, edge_color in card_specs:
        rounded_card(left, card_bottom, width, card_height, face_color, edge_color)

    step_badge((0.06, 0.725), "1", NAVY)
    ax.text(0.1325, 0.719, "Evidence\ngarden", ha="center", va="center", fontsize=8.5, fontweight="bold", color=soft_ink, linespacing=0.95)
    ax.text(0.1325, 0.647, "roots  →  evidence IDs", ha="center", va="center", fontsize=6.8, color=soft_ink)
    document((0.085, 0.525), 0.052, 0.092, BLUE)
    document((0.145, 0.525), 0.052, 0.092, NAVY)
    ax.plot([0.111, 0.119, 0.135], [0.483, 0.458, 0.483], color=GREEN, linewidth=1.4, zorder=3)
    ax.plot([0.119, 0.119], [0.458, 0.418], color=GREEN, linewidth=1.4, zorder=3)
    ax.add_patch(Ellipse((0.101, 0.469), 0.026, 0.052, angle=35, facecolor=PALE_GREEN, edgecolor=GREEN, linewidth=0.8, zorder=3))
    ax.add_patch(Ellipse((0.139, 0.469), 0.026, 0.052, angle=-35, facecolor=PALE_GREEN, edgecolor=GREEN, linewidth=0.8, zorder=3))
    ax.text(0.1325, 0.365, "source roots\nfixed agent views", ha="center", va="center", fontsize=6.8, color=soft_ink, linespacing=1.2)

    step_badge((0.30, 0.725), "2", ORANGE)
    ax.text(0.3725, 0.719, "Agent\nchoir", ha="center", va="center", fontsize=8.5, fontweight="bold", color=soft_ink, linespacing=0.95)
    ax.text(0.3725, 0.647, "5 fixed personas", ha="center", va="center", fontsize=6.8, color=soft_ink)
    face((0.316, 0.535), 0.052, "#F5C56B")
    face((0.346, 0.535), 0.052, "#F5D98D", wink=True)
    face((0.376, 0.535), 0.052, "#F2B2A8")
    face((0.406, 0.535), 0.052, "#B9DCCF", wink=True)
    face((0.436, 0.535), 0.052, "#B8D8EE")
    ax.text(0.3725, 0.365, "answer  ·  confidence\ncited IDs", ha="center", va="center", fontsize=6.9, color=soft_ink, linespacing=1.2)

    step_badge((0.54, 0.725), "3", VERMILION)
    ax.text(0.6125, 0.719, "Gentle\nnudges", ha="center", va="center", fontsize=8.5, fontweight="bold", color=soft_ink, linespacing=0.95)
    ax.text(0.6125, 0.647, "one paired change", ha="center", va="center", fontsize=6.8, color=soft_ink)
    sparkle((0.55, 0.575), ORANGE, 0.018)
    sparkle((0.675, 0.585), PURPLE, 0.012)
    pill(0.526, 0.525, 0.081, "original", "#E8F3F8", BLUE)
    pill(0.612, 0.525, 0.081, "remove", "#F0EEEE", GRAY)
    pill(0.526, 0.455, 0.081, "reverse", "#FBE7E2", VERMILION)
    pill(0.612, 0.455, 0.081, "substitute", "#EEE8F7", PURPLE)
    ax.text(0.6125, 0.365, "paired, visible\nanswer changes", ha="center", va="center", fontsize=6.9, color=soft_ink, linespacing=1.2)

    step_badge((0.78, 0.725), "4", GREEN)
    ax.text(0.86, 0.719, "Risk  →\naction", ha="center", va="center", fontsize=8.5, fontweight="bold", color=soft_ink, linespacing=0.95)
    ax.text(0.86, 0.647, "pre-outcome risk", ha="center", va="center", fontsize=6.8, color=soft_ink)
    gauge_center = (0.86, 0.545)
    ax.add_patch(Arc(gauge_center, 0.135, 0.115, theta1=0, theta2=180, color="#C5DCD1", linewidth=5.0, zorder=3))
    ax.add_patch(Arc(gauge_center, 0.135, 0.115, theta1=0, theta2=72, color=GREEN, linewidth=5.0, zorder=4))
    ax.plot([0.86, 0.897], [0.545, 0.59], color=VERMILION, linewidth=1.4, zorder=5)
    ax.add_patch(Circle(gauge_center, 0.009, facecolor=VERMILION, edgecolor="white", linewidth=0.6, zorder=6))
    ax.text(0.86, 0.505, "$R_{\\mathrm{PI}}$ / $R_{\\mathrm{sym}}$", ha="center", va="center", fontsize=7.3, fontweight="bold", color=soft_ink)
    ax.add_patch(
        FancyBboxPatch((0.765, 0.345), 0.085, 0.062, boxstyle="round,pad=0.006,rounding_size=0.013", facecolor="#E3F3ED", edgecolor=GREEN, linewidth=0.8, zorder=3)
    )
    ax.add_patch(
        FancyBboxPatch((0.87, 0.345), 0.085, 0.062, boxstyle="round,pad=0.006,rounding_size=0.013", facecolor="#FBE8E4", edgecolor=VERMILION, linewidth=0.8, zorder=3)
    )
    ax.text(0.8075, 0.382, "LOW RISK\nretain", ha="center", va="center", fontsize=6.4, fontweight="bold", color=GREEN, linespacing=1.1, zorder=4)
    ax.text(0.9125, 0.382, "HIGH RISK\nreview / route", ha="center", va="center", fontsize=6.4, fontweight="bold", color=VERMILION, linespacing=1.1, zorder=4)
    ax.text(0.86, 0.298, "ranks the tail\nnot the truth", ha="center", va="center", fontsize=5.8, color=muted, linespacing=1.0)

    arrow(0.238, 0.258)
    arrow(0.478, 0.498)
    arrow(0.718, 0.738)
    ax.add_patch(FancyArrowPatch((0.50, 0.17), (0.50, 0.095), arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=muted, zorder=2))
    ax.add_patch(
        FancyBboxPatch((0.20, 0.035), 0.62, 0.06, boxstyle="round,pad=0.008,rounding_size=0.018", facecolor="#F2EEE5", edgecolor="#D8D0C3", linewidth=0.8, zorder=2)
    )
    lock_left = 0.225
    ax.add_patch(Rectangle((lock_left, 0.049), 0.027, 0.023, facecolor=muted, edgecolor=muted, linewidth=0.5, zorder=4))
    ax.add_patch(Arc((lock_left + 0.0135, 0.072), 0.022, 0.025, theta1=0, theta2=180, color=muted, linewidth=1.2, zorder=4))
    ax.text(0.58, 0.065, "labels unlock after routing  →  evaluate consensus error", ha="center", va="center", fontsize=6.6, fontweight="bold", color=soft_ink, zorder=3)
    sparkle((0.72, 0.86), ORANGE, 0.012)
    sparkle((0.255, 0.86), BLUE, 0.010)
    _save(fig, "framework_overview", write_svg=True)


def make_methodology_figure() -> None:
    background = "#FFFFFF"
    panel_navy = "#214E6B"
    soft_ink = "#253746"
    muted = "#6F7F8C"
    warning = "#C4514D"
    stable = "#159A74"
    yellow = "#F3B21A"
    lavender = "#7A5AA6"
    pale_yellow = "#FFF5D9"
    pale_gray = "#F1F4F6"

    fig, ax = plt.subplots(figsize=(7.15, 5.35))
    fig.patch.set_facecolor(background)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def frame(
        left: float,
        bottom: float,
        width: float,
        height: float,
        title: str,
        subtitle: str,
        accent: str,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                facecolor="white",
                edgecolor=panel_navy,
                linewidth=1.2,
                linestyle=(0, (4, 2)),
                zorder=0,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (left + 0.012, bottom + height - 0.046),
                width - 0.024,
                0.032,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor=accent,
                edgecolor=panel_navy,
                linewidth=0.6,
                zorder=1,
            )
        )
        ax.text(
            left + width / 2,
            bottom + height - 0.030,
            title,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=soft_ink,
            zorder=2,
        )
        ax.text(
            left + width / 2,
            bottom + height - 0.063,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.8,
            color=muted,
            zorder=2,
        )

    def card(
        left: float,
        bottom: float,
        width: float,
        height: float,
        text: str,
        face_color: str,
        edge_color: str,
        *,
        fontsize: float = 6.5,
        dashed: bool = False,
        weight: str = "normal",
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=0.85,
                linestyle=(0, (3, 2)) if dashed else "-",
                zorder=2,
            )
        )
        ax.text(
            left + width / 2,
            bottom + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=soft_ink,
            linespacing=1.05,
            zorder=3,
        )

    def flow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = muted,
        dashed: bool = False,
        linewidth: float = 1.15,
        mutation_scale: float = 10,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=mutation_scale,
                linewidth=linewidth,
                linestyle="--" if dashed else "-",
                color=color,
                zorder=4,
            )
        )

    def document(center: tuple[float, float], width: float, height: float, color: str) -> None:
        center_x, center_y = center
        left = center_x - width / 2
        bottom = center_y - height / 2
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (left + width * 0.64, bottom + height),
                    (left + width, bottom + height * 0.64),
                    (left + width * 0.64, bottom + height * 0.64),
                ],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.3,
                zorder=4,
            )
        )
        for line_index in range(2):
            line_y = bottom + height * (0.39 - line_index * 0.16)
            ax.plot(
                [left + width * 0.18, left + width * 0.78],
                [line_y, line_y],
                color=color,
                linewidth=0.65,
                zorder=4,
            )

    def source(center: tuple[float, float], width: float, height: float, color: str) -> None:
        center_x, center_y = center
        left = center_x - width / 2
        bottom = center_y - height / 2
        ax.add_patch(Rectangle((left, bottom), width, height * 0.72, facecolor=color, edgecolor=color, linewidth=0.7, zorder=3))
        ax.add_patch(Ellipse((center_x, bottom + height * 0.72), width, height * 0.32, facecolor=color, edgecolor=panel_navy, linewidth=0.7, zorder=4))
        ax.add_patch(Ellipse((center_x, bottom), width, height * 0.32, facecolor=color, edgecolor=panel_navy, linewidth=0.7, zorder=4))
        ax.add_patch(Ellipse((center_x, bottom + height * 0.72), width * 0.62, height * 0.12, facecolor="white", edgecolor="none", alpha=0.55, zorder=5))

    def robot(center: tuple[float, float], size: float, color: str, label: str | None = None) -> None:
        center_x, center_y = center
        ax.plot([center_x, center_x], [center_y + size * 0.35, center_y + size * 0.56], color=panel_navy, linewidth=0.7, zorder=4)
        ax.add_patch(Circle((center_x, center_y + size * 0.61), size * 0.045, facecolor=yellow, edgecolor=panel_navy, linewidth=0.5, zorder=5))
        ax.add_patch(FancyBboxPatch((center_x - size * 0.32, center_y - size * 0.38), size * 0.64, size * 0.46, boxstyle="round,pad=0.002,rounding_size=0.006", facecolor=color, edgecolor=panel_navy, linewidth=0.7, zorder=3))
        ax.add_patch(Ellipse((center_x, center_y + size * 0.10), size * 0.76, size * 0.58, facecolor=color, edgecolor=panel_navy, linewidth=0.8, zorder=4))
        ax.add_patch(Circle((center_x - size * 0.18, center_y + size * 0.15), size * 0.045, facecolor=soft_ink, zorder=5))
        ax.add_patch(Circle((center_x + size * 0.18, center_y + size * 0.15), size * 0.045, facecolor=soft_ink, zorder=5))
        ax.add_patch(Arc((center_x, center_y + size * 0.05), size * 0.28, size * 0.16, theta1=205, theta2=335, color=soft_ink, linewidth=0.65, zorder=5))
        if label:
            ax.text(center_x, center_y - size * 0.58, label, ha="center", va="center", fontsize=5.2, color=muted, zorder=5)

    def decision_token(center: tuple[float, float], text: str, color: str, *, width: float = 0.042) -> None:
        center_x, center_y = center
        card(center_x - width / 2, center_y - 0.014, width, 0.028, text, "white", color, fontsize=5.3, weight="bold")

    def matrix_cell(left: float, bottom: float, value: str, color: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                0.034,
                0.027,
                boxstyle="round,pad=0.002,rounding_size=0.005",
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        )
        ax.text(left + 0.017, bottom + 0.0135, value, ha="center", va="center", fontsize=5.4, fontweight="bold", color=soft_ink, zorder=4)

    def shield(center: tuple[float, float], size: float) -> None:
        center_x, center_y = center
        points = [
            (center_x - size * 0.34, center_y + size * 0.37),
            (center_x + size * 0.34, center_y + size * 0.37),
            (center_x + size * 0.29, center_y - size * 0.10),
            (center_x, center_y - size * 0.43),
            (center_x - size * 0.29, center_y - size * 0.10),
        ]
        ax.add_patch(Polygon(points, closed=True, facecolor="#DCEFE8", edgecolor=stable, linewidth=1.0, zorder=4))
        ax.plot([center_x - size * 0.13, center_x - size * 0.02, center_x + size * 0.18], [center_y - size * 0.02, center_y - size * 0.16, center_y + size * 0.15], color=stable, linewidth=1.2, zorder=5)

    def warning_icon(center: tuple[float, float], size: float) -> None:
        center_x, center_y = center
        ax.add_patch(Polygon([(center_x, center_y + size * 0.42), (center_x - size * 0.40, center_y - size * 0.33), (center_x + size * 0.40, center_y - size * 0.33)], closed=True, facecolor="#FBE8E4", edgecolor=warning, linewidth=0.9, zorder=4))
        ax.plot([center_x, center_x], [center_y + size * 0.15, center_y - size * 0.10], color=warning, linewidth=1.0, zorder=5)
        ax.add_patch(Circle((center_x, center_y - size * 0.21), size * 0.035, facecolor=warning, zorder=5))

    ax.text(0.5, 0.985, "From shared evidence to a reliability-aware decision", ha="center", va="center", fontsize=12.0, fontweight="bold", color=soft_ink)
    ax.text(0.5, 0.958, "The environment changes the evidence, watches the response, and reveals labels last", ha="center", va="center", fontsize=7.4, color=muted)

    frame(0.025, 0.53, 0.285, 0.40, "1. Decision setup", "question → agents → consensus", PALE_BLUE)
    frame(0.33, 0.53, 0.30, 0.40, "2. Provenance", "source → evidence → agents", pale_yellow)
    frame(0.65, 0.53, 0.325, 0.40, "3. Counterfactual tests", "controlled evidence intervention", PALE_RED)
    frame(0.025, 0.075, 0.95, 0.385, "4. Reliability & routing", "", PALE_GREEN)

    card(0.047, 0.802, 0.241, 0.050, 'Question: "Accept this\nmulti-agent decision?"', pale_yellow, yellow, fontsize=6.3, weight="bold")
    agent_centers = [0.070, 0.117, 0.164, 0.211, 0.258]
    robot_colors = ["#F4C86A", "#F2D98A", "#F1B1A8", "#BBDDCF", "#B9D9EF"]
    for index, (agent_x, robot_color) in enumerate(zip(agent_centers, robot_colors, strict=True), start=1):
        robot((agent_x, 0.748), 0.045, robot_color, f"A{index}")
        decision_token((agent_x, 0.678), "YES", stable)
    card(0.061, 0.603, 0.213, 0.046, "HIGH CONSENSUS  ·  5/5 YES\nconsensus  ≠  reliability", "#FBE8E4", warning, fontsize=6.0, dashed=True, weight="bold")
    source((0.063, 0.565), 0.026, 0.034, BLUE)
    document((0.111, 0.565), 0.033, 0.039, BLUE)
    card(0.139, 0.554, 0.041, 0.022, "claim", PALE_ORANGE, ORANGE, fontsize=4.8)
    robot((0.216, 0.566), 0.022, "#BBDDCF")
    decision_token((0.270, 0.566), "YES", stable, width=0.035)
    flow((0.078, 0.565), (0.094, 0.565), color=muted, mutation_scale=7)
    flow((0.128, 0.565), (0.139, 0.565), color=muted, mutation_scale=7)
    flow((0.181, 0.565), (0.203, 0.565), color=muted, mutation_scale=7)
    flow((0.229, 0.565), (0.252, 0.565), color=muted, mutation_scale=7)
    ax.text(0.167, 0.541, "SRC  →  EVID  →  CLAIM  →  AGENT  →  DEC", ha="center", va="center", fontsize=4.8, color=muted)

    source_centers = [(0.375, "A", BLUE), (0.455, "B", ORANGE), (0.535, "C", PURPLE)]
    evidence_centers = [(0.365, "E1", BLUE), (0.405, "E2", BLUE), (0.445, "E3", ORANGE), (0.485, "E4", PURPLE), (0.525, "E5", PURPLE)]
    for source_x, source_label, source_color in source_centers:
        source((source_x, 0.827), 0.034, 0.035, source_color)
        ax.text(source_x, 0.795, f"Source {source_label}", ha="center", va="center", fontsize=5.0, color=soft_ink)
    for evidence_x, evidence_label, evidence_color in evidence_centers:
        document((evidence_x, 0.744), 0.030, 0.035, evidence_color)
        ax.text(evidence_x, 0.713, evidence_label, ha="center", va="center", fontsize=4.8, color=muted)
    for start_x, end_x in ((0.375, 0.365), (0.375, 0.405), (0.455, 0.445), (0.535, 0.485), (0.535, 0.525)):
        flow((start_x, 0.805), (end_x, 0.765), color=muted, mutation_scale=6)
    agent_graph = [(0.385, "A1"), (0.465, "A3"), (0.545, "A5")]
    for agent_x, agent_label in agent_graph:
        robot((agent_x, 0.665), 0.029, "#F2D98A", agent_label)
    for evidence_x, agent_x in ((0.365, 0.385), (0.445, 0.465), (0.525, 0.545)):
        flow((evidence_x, 0.725), (agent_x, 0.684), color=muted, dashed=True, mutation_scale=6)
    card(0.352, 0.565, 0.256, 0.072, "SHARED PROVENANCE\none source  →  correlated views", pale_yellow, yellow, fontsize=5.9, dashed=True, weight="bold")
    ax.text(0.480, 0.548, "different-looking citations can share one root", ha="center", va="center", fontsize=5.0, color=warning)

    card(0.670, 0.805, 0.285, 0.040, "Original:  E={E1,E2,E3}  ·  A(E)=YES", "#FFFFFF", panel_navy, fontsize=5.5, weight="bold")
    branch_centers = [0.704, 0.812, 0.920]
    branch_names = ["REMOVE", "REVERSE", "SUBSTITUTE"]
    branch_text = ["Ej  →  ∅", "Ej  →  Ēj", "Ej  →  E′j"]
    branch_colors = [BLUE, warning, lavender]
    for branch_x, branch_name, branch_formula, branch_color in zip(branch_centers, branch_names, branch_text, branch_colors, strict=True):
        card(branch_x - 0.045, 0.773, 0.090, 0.026, branch_name, branch_color, panel_navy, fontsize=5.3, weight="bold")
        card(branch_x - 0.045, 0.711, 0.090, 0.040, branch_formula, pale_gray, branch_color, fontsize=6.3, weight="bold")
        flow((branch_x, 0.771), (branch_x, 0.753), color=branch_color, mutation_scale=6)
    response_texts = ["YES  →  NO\nflip", "YES  →  YES\ninertia", "YES  →  NO\nflip"]
    response_colors = [warning, stable, warning]
    for branch_x, response_text, response_color in zip(branch_centers, response_texts, response_colors, strict=True):
        card(branch_x - 0.045, 0.629, 0.090, 0.052, response_text, "#FFFFFF", response_color, fontsize=5.5, weight="bold")
        flow((branch_x, 0.709), (branch_x, 0.684), color=response_color, mutation_scale=6)
    ax.text(0.812, 0.602, r"$\Delta_{ij}=A_i(E)-A_i(E')$", ha="center", va="center", fontsize=6.4, color=panel_navy)
    card(0.674, 0.543, 0.277, 0.045, "Does the decision respond to evidence?\nfragility  →  false-consensus risk", pale_yellow, warning, fontsize=5.0, dashed=True, weight="bold")

    flow((0.310, 0.752), (0.328, 0.752), color=panel_navy, linewidth=1.4, mutation_scale=10)
    flow((0.630, 0.752), (0.648, 0.752), color=panel_navy, linewidth=1.4, mutation_scale=10)
    flow((0.812, 0.535), (0.812, 0.468), color=panel_navy, linewidth=1.4, mutation_scale=10)
    ax.text(0.319, 0.768, "trace", ha="center", va="center", fontsize=4.8, color=muted)
    ax.text(0.639, 0.768, "test", ha="center", va="center", fontsize=4.8, color=muted)
    ax.text(0.829, 0.503, "observe", ha="left", va="center", fontsize=4.8, color=muted)

    card(0.046, 0.381, 0.274, 0.036, "INTERVENTION MATRIX", "#EAF4F8", panel_navy, fontsize=6.0, weight="bold")
    matrix_left = 0.142
    matrix_bottom = 0.205
    for column_index, column_label in enumerate(("E1", "E2", "E3")):
        ax.text(matrix_left + column_index * 0.042 + 0.017, 0.347, column_label, ha="center", va="center", fontsize=5.2, fontweight="bold", color=panel_navy)
    matrix_values = ((0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 1, 0), (0, 0, 1))
    for row_index, row_values in enumerate(matrix_values):
        row_y = matrix_bottom + (4 - row_index) * 0.030
        ax.text(0.106, row_y + 0.0135, f"A{row_index + 1}", ha="right", va="center", fontsize=5.0, color=muted)
        for column_index, value in enumerate(row_values):
            cell_color = "#CFEBDD" if value else "#F9D6C9"
            matrix_cell(matrix_left + column_index * 0.042, row_y, str(value), cell_color)
    ax.text(0.182, 0.174, "1 = evidence-sensitive    0 = suspicious inertia", ha="center", va="center", fontsize=4.9, color=muted)
    ax.text(0.182, 0.145, r"$\Delta_{ij}=A_i(E)-A_i(E')$", ha="center", va="center", fontsize=5.8, color=panel_navy)
    card(0.046, 0.103, 0.274, 0.028, "green = flip   ·   orange = inertia", "#FFFFFF", muted, fontsize=4.9)

    card(0.350, 0.381, 0.270, 0.036, "INTERPRETABLE RISK ROUTER", "#EAF4F8", panel_navy, fontsize=6.0, weight="bold")
    card(0.370, 0.319, 0.095, 0.040, "answer-flip\ninertia", PALE_BLUE, BLUE, fontsize=5.3, weight="bold")
    card(0.505, 0.319, 0.095, 0.040, "complete\ninertia", PALE_ORANGE, ORANGE, fontsize=5.3, weight="bold")
    flow((0.418, 0.318), (0.466, 0.267), color=BLUE, mutation_scale=7)
    flow((0.553, 0.318), (0.502, 0.267), color=ORANGE, mutation_scale=7)
    shield((0.485, 0.239), 0.060)
    ax.text(0.485, 0.184, r"$R_{\mathrm{PI}}$ / $R_{\mathrm{sym}}$", ha="center", va="center", fontsize=7.2, fontweight="bold", color=panel_navy)
    ax.text(0.485, 0.158, "pre-outcome risk", ha="center", va="center", fontsize=5.0, color=muted)
    card(0.370, 0.103, 0.230, 0.034, "shared-source fraction  ·  secondary signal", pale_yellow, yellow, fontsize=5.0, dashed=True)
    ax.text(0.485, 0.083, r"$R_{\mathrm{PI}}=0.1D_{\mathrm{inert}}+0.3I_{\mathrm{flip}}+0.6F_{\mathrm{shared}}$", ha="center", va="center", fontsize=4.7, color=muted)

    card(0.650, 0.375, 0.300, 0.030, "SELECTIVE ROUTING", "#EAF4F8", panel_navy, fontsize=5.8, weight="bold")
    card(0.660, 0.319, 0.132, 0.050, "", "#E3F3ED", stable, fontsize=6.1, weight="bold")
    shield((0.678, 0.344), 0.026)
    ax.text(0.742, 0.344, "LOW RISK\nACCEPT", ha="center", va="center", fontsize=5.8, fontweight="bold", color=soft_ink, linespacing=1.05)
    card(0.808, 0.319, 0.132, 0.050, "", "#FBE8E4", warning, fontsize=5.6, weight="bold")
    warning_icon((0.818, 0.344), 0.026)
    ax.text(0.895, 0.344, "HIGH RISK\nABSTAIN / ESCALATE", ha="center", va="center", fontsize=5.0, fontweight="bold", color=soft_ink, linespacing=1.05)
    ax.text(0.800, 0.286, "predict reliability before labels are revealed", ha="center", va="center", fontsize=5.0, color=panel_navy, fontweight="bold")
    card(0.660, 0.240, 0.280, 0.028, "BoolQ V12.1  ·  compact validation check", pale_yellow, yellow, fontsize=5.0, dashed=True, weight="bold")
    card(0.660, 0.199, 0.132, 0.026, "flip inertia: 0.807", "#FFFFFF", BLUE, fontsize=4.7, weight="bold")
    card(0.808, 0.199, 0.132, 0.026, "shared source: 0.498", "#FFFFFF", lavender, fontsize=4.5, weight="bold")
    ax.text(0.660, 0.160, "Risk@80", ha="left", va="center", fontsize=5.0, color=muted, fontweight="bold")
    ax.add_patch(Rectangle((0.715, 0.151), 0.055, 0.018, facecolor="#E5EBEF", edgecolor=muted, linewidth=0.5, zorder=3))
    ax.add_patch(Rectangle((0.715, 0.151), 0.036, 0.018, facecolor=stable, edgecolor="none", zorder=4))
    ax.text(0.785, 0.160, "22.0%  →  13.3%", ha="left", va="center", fontsize=5.1, color=stable, fontweight="bold")
    ax.text(0.800, 0.125, "selective routing reduces retained error at 80% coverage", ha="center", va="center", fontsize=4.6, color=muted)

    flow((0.322, 0.267), (0.345, 0.267), color=panel_navy, linewidth=1.2, mutation_scale=8)
    flow((0.622, 0.267), (0.645, 0.267), color=panel_navy, linewidth=1.2, mutation_scale=8)
    ax.text(0.334, 0.282, "aggregate", ha="center", va="center", fontsize=4.6, color=muted)
    ax.text(0.633, 0.282, "route", ha="center", va="center", fontsize=4.6, color=muted)

    card(0.185, 0.018, 0.290, 0.032, "HIGH CONSENSUS  ≠  RELIABLE", "#FBE8E4", warning, fontsize=5.8, weight="bold")
    card(0.525, 0.018, 0.290, 0.032, "INTERVENTION  >  CITATIONS", "#E3F3ED", stable, fontsize=5.8, weight="bold")

    _save(fig, "framework_overview", write_svg=True)


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
    make_methodology_figure()
    make_primary_results_figure()
    routing_data = make_routing_figure()
    (FIGURE_DIR / "routing_curve_data.json").write_text(
        json.dumps(routing_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote paper figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
