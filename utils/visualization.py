"""
Visualisation Utilities
========================
Matplotlib-based charts for the Streamlit dashboard:
  • Risk-score gauge
  • Hazard distribution chart
  • Detection statistics summary
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List
from utils.config import RISK_LEVELS


# ---------------------------------------------------------------------------
# Risk-Score Gauge
# ---------------------------------------------------------------------------
def create_risk_gauge(score: int, level: str, color: str) -> plt.Figure:
    """
    Create a semi-circular gauge chart showing the risk score 0–100.

    Parameters
    ----------
    score : int    – Normalised risk score.
    level : str    – Risk level label (Low / Medium / High / Critical).
    color : str    – Hex colour for the score arc.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Draw background arc segments for each risk level
    for lvl in RISK_LEVELS:
        start = np.radians(180 - lvl["min"] * 1.8)
        end = np.radians(180 - lvl["max"] * 1.8)
        theta = np.linspace(start, end, 50)
        ax.fill_between(theta, 0.7, 1.0, color=lvl["color"], alpha=0.20)

    # Score arc
    score_end = np.radians(180 - score * 1.8)
    theta_score = np.linspace(np.radians(180), score_end, 100)
    ax.fill_between(theta_score, 0.72, 0.98, color=color, alpha=0.85)

    # Needle
    ax.plot([score_end, score_end], [0, 0.7], color="white", linewidth=2)
    ax.plot(score_end, 0.7, "o", color="white", markersize=6)

    # Centre text
    ax.text(
        np.radians(90), 0.15, f"{score}",
        ha="center", va="center", fontsize=36, fontweight="bold",
        color="white", family="monospace",
    )
    ax.text(
        np.radians(90), -0.15, level.upper(),
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=color,
    )

    # Labels
    ax.text(np.radians(180), 1.15, "0", ha="center", color="grey", fontsize=9)
    ax.text(np.radians(0), 1.15, "100", ha="center", color="grey", fontsize=9)

    ax.set_ylim(0, 1.3)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hazard Distribution Chart
# ---------------------------------------------------------------------------
def create_hazard_chart(hazard_breakdown: List[Dict]) -> plt.Figure:
    """Horizontal bar chart of hazard contributions."""
    if not hazard_breakdown:
        return _empty_chart("No hazards to display")

    hazards = [h["hazard"] for h in hazard_breakdown]
    contributions = [h["risk_contribution"] for h in hazard_breakdown]
    severities = [h["severity"] for h in hazard_breakdown]

    severity_colors = {
        "High": "#ef4444",
        "Medium-High": "#f97316",
        "Medium": "#f59e0b",
        "Low": "#22c55e",
    }
    colors = [severity_colors.get(s, "#6b7280") for s in severities]

    fig, ax = plt.subplots(figsize=(7, max(3, len(hazards) * 0.7)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(hazards, contributions, color=colors, edgecolor="white", linewidth=0.5, height=0.5)

    for bar, val in zip(bars, contributions):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val}", va="center", color="white", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Risk Contribution", color="white", fontsize=11)
    ax.set_title("Hazard Distribution", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Detection Statistics
# ---------------------------------------------------------------------------
def create_detection_stats_chart(detections: List[Dict]) -> plt.Figure:
    """Bar chart showing count of each detected class."""
    if not detections:
        return _empty_chart("No detections to display")

    from collections import Counter
    counts = Counter(d["class_name"] for d in detections)
    names = list(counts.keys())
    values = list(counts.values())

    # Colour by semantic category
    from utils.config import HAZARD_CLASSES, SAFETY_EQUIPMENT_CLASSES, WORKER_CLASSES, HEAVY_OBJECTS
    def _color(name):
        if name in HAZARD_CLASSES:
            return "#ef4444"
        if name in SAFETY_EQUIPMENT_CLASSES:
            return "#22c55e"
        if name in WORKER_CLASSES:
            return "#f59e0b"
        if name in HEAVY_OBJECTS:
            return "#f97316"
        return "#6b7280"

    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.7)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5, height=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", color="white", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Count", color="white", fontsize=11)
    ax.set_title("Detected Objects", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Legend
    patches = [
        mpatches.Patch(color="#ef4444", label="Hazard"),
        mpatches.Patch(color="#22c55e", label="Safety Equipment"),
        mpatches.Patch(color="#f59e0b", label="Worker"),
        mpatches.Patch(color="#f97316", label="Heavy Object"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Risk Trend (for video analysis)
# ---------------------------------------------------------------------------
def create_risk_trend_chart(scores: List[int], frames: List[int]) -> plt.Figure:
    """Line chart showing risk score across video frames."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(frames, scores, color="#818cf8", linewidth=2, marker="o", markersize=4)
    ax.fill_between(frames, scores, alpha=0.15, color="#818cf8")

    # Risk-level bands
    for lvl in RISK_LEVELS:
        ax.axhspan(lvl["min"], lvl["max"], color=lvl["color"], alpha=0.08)

    ax.set_xlabel("Frame", color="white", fontsize=11)
    ax.set_ylabel("Risk Score", color="white", fontsize=11)
    ax.set_title("Risk Score Trend", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Milestone 2 Analytics Charts
# ---------------------------------------------------------------------------
def create_compliance_pie_chart(compliant_count: int, violation_count: int) -> plt.Figure:
    """Donut chart showing the current status of tracked workers."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    total = compliant_count + violation_count
    if total == 0:
        return _empty_chart("No worker data available")

    sizes = [compliant_count, violation_count]
    labels = [f"Compliant ({compliant_count})", f"Violations ({violation_count})"]
    colors = ["#22c55e", "#ef4444"]
    explode = (0.05, 0.05) if violation_count > 0 else (0, 0)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
        wedgeprops={"width": 0.42, "edgecolor": "#0e1117", "linewidth": 2},
        textprops=dict(color="white", fontweight="bold"),
    )
    for autotext in autotexts:
        autotext.set_color("white")

    ax.set_title("Current Worker PPE Status", color="white", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def create_violations_by_type_chart(violation_types: Dict[str, int]) -> plt.Figure:
    """Horizontal bar chart for violations broken down by type."""
    if not violation_types:
        return _empty_chart("No violations recorded")

    ordered = sorted(violation_types.items(), key=lambda item: item[1])
    types = [item[0] for item in ordered]
    counts = [item[1] for item in ordered]

    fig, ax = plt.subplots(figsize=(7, max(3.2, len(types) * 0.7)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(types, counts, color="#f97316", edgecolor="white", linewidth=0.5, height=0.5)

    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", color="white", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Violation records", color="white", fontsize=10)
    ax.set_title("Missing PPE by Type", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def create_risk_distribution_chart(risk_dist: Dict[str, int]) -> plt.Figure:
    """Bar chart showing risk level distribution."""
    if not risk_dist:
        return _empty_chart("No risk data recorded")

    levels = ["Low", "Medium", "High", "Critical"]
    color_map = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}

    present_levels = levels
    counts = [risk_dist.get(l, risk_dist.get(l.upper(), 0)) for l in present_levels]
    colors = [color_map[l] for l in present_levels]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.bar(present_levels, counts, color=colors, edgecolor="white", linewidth=0.5, width=0.5)

    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(val), ha="center", color="white", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Violation records", color="white", fontsize=10)
    ax.set_title("Risk Level of PPE Violations", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _empty_chart(message: str) -> plt.Figure:
    """Return a placeholder chart with a centred message."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.text(0.5, 0.5, message, ha="center", va="center", color="grey", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    return fig

