"""
Piezo.AI — Chart Generator for PDF Reports
=============================================
Uses Matplotlib to generate chart images for embedding in ReportLab PDFs.
Charts are saved as temporary PNG files and returned as paths.

LIGHT THEME — white backgrounds, professional styling for PDF embedding.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import numpy as np


# ── Chart Color Palette (matches Piezo.AI theme) ────────────
COLORS = {
    "indigo": "#4F46E5",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "pink": "#EC4899",
    "violet": "#8B5CF6",
    "slate": "#64748B",
    "blue": "#3B82F6",
    "red": "#EF4444",
}

TARGET_COLORS = {
    "d33": COLORS["indigo"],
    "tc": COLORS["emerald"],
    "vickers_hardness": COLORS["amber"],
}

TARGET_LABELS = {
    "d33": "d₃₃ (pC/N)",
    "tc": "Tc (°C)",
    "vickers_hardness": "Hardness (HV)",
}


def _setup_style():
    """Apply clean light-theme chart styling for PDF reports."""
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FAFBFC",
        "axes.edgecolor": "#D1D5DB",
        "axes.labelcolor": "#1F2937",
        "text.color": "#1F2937",
        "xtick.color": "#6B7280",
        "ytick.color": "#6B7280",
        "grid.color": "#E5E7EB",
        "grid.alpha": 0.7,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
    })


def generate_r2_rmse_chart(models: list[dict[str, Any]]) -> str | None:
    """Generate R²/RMSE bar chart comparing models. Returns path to PNG."""
    if not models:
        return None

    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    names = [m.get("display_name", "Model")[:20] for m in models]
    r2_scores = [m.get("r2_score", 0) for m in models]
    rmse_scores = [m.get("rmse", 0) for m in models]
    bar_colors = [TARGET_COLORS.get(m.get("target", ""), COLORS["slate"]) for m in models]

    x = np.arange(len(names))
    width = 0.55

    # R² chart
    bars1 = ax1.bar(x, r2_scores, width, color=bar_colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Model", fontweight="medium")
    ax1.set_ylabel("R² Score", fontweight="medium")
    ax1.set_title("R² Score Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax1.set_ylim(min(0, min(r2_scores) - 0.1) if r2_scores else 0, 1.15)
    ax1.axhline(y=0.8, color=COLORS["emerald"], linestyle="--", alpha=0.4, linewidth=0.8, label="Good (0.8)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=7, loc="upper right")
    for bar, val in zip(bars1, r2_scores):
        y_offset = 0.03 if val >= 0 else -0.08
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color="#374151")

    # RMSE chart
    bars2 = ax2.bar(x, rmse_scores, width, color=bar_colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Model", fontweight="medium")
    ax2.set_ylabel("RMSE", fontweight="medium")
    ax2.set_title("RMSE Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmse_scores) * 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color="#374151")

    fig.tight_layout(pad=2.0)
    path = _save_temp_chart("r2_rmse")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return path


def generate_target_distribution_chart(models: list[dict[str, Any]]) -> str | None:
    """Generate donut chart of model target distribution. Returns path to PNG."""
    if not models:
        return None

    _setup_style()
    targets = {}
    for m in models:
        t = m.get("target", "unknown")
        targets[t] = targets.get(t, 0) + 1

    labels = [TARGET_LABELS.get(k, k) for k in targets.keys()]
    sizes = list(targets.values())
    colors = [TARGET_COLORS.get(k, COLORS["slate"]) for k in targets.keys()]
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2.5),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("#FFFFFF")
    for t in texts:
        t.set_fontsize(10)
        t.set_color("#374151")

    # Center label
    ax.text(0, 0, str(total), ha="center", va="center",
            fontsize=24, fontweight="bold", color="#1F2937")
    ax.text(0, -0.12, "Models", ha="center", va="center",
            fontsize=9, color="#6B7280")

    ax.set_title("Model Target Distribution", pad=20)

    path = _save_temp_chart("target_dist")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return path


def generate_model_performance_chart(models: list[dict[str, Any]]) -> str | None:
    """Generate horizontal bar chart of model performance overview."""
    if not models:
        return None

    _setup_style()
    n = len(models)
    fig_height = max(3, n * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    names = [f"{m.get('display_name', '?')[:22]} ({TARGET_LABELS.get(m.get('target', ''), m.get('target', '?'))})" for m in models]
    r2_scores = [m.get("r2_score", 0) for m in models]
    bar_colors = [TARGET_COLORS.get(m.get("target", ""), COLORS["slate"]) for m in models]

    y = np.arange(n)
    bars = ax.barh(y, r2_scores, color=bar_colors, alpha=0.9, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("R² Score", fontweight="medium")
    ax.set_title("Model Performance Overview")
    ax.set_xlim(min(0, min(r2_scores) - 0.1) if r2_scores else 0, 1.15)
    ax.axvline(x=0.8, color=COLORS["emerald"], linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, r2_scores):
        x_offset = 0.02 if val >= 0 else -0.08
        ax.text(max(bar.get_width(), 0) + x_offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", fontsize=9, fontweight="bold",
                color="#374151")

    fig.tight_layout(pad=1.5)
    path = _save_temp_chart("model_perf")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return path


def generate_convergence_chart(
    convergence_data: dict[str, list[dict]],
    model_name: str = "",
) -> str | None:
    """Generate convergence chart for a trained model. Returns path to PNG."""
    if not convergence_data:
        return None

    _setup_style()
    n_targets = len(convergence_data)
    if n_targets == 0:
        return None

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets + 1, 4), squeeze=False)

    for idx, (target, data_points) in enumerate(convergence_data.items()):
        ax = axes[0][idx]
        if not data_points:
            continue
        iters = [d["iteration"] for d in data_points]
        metrics = [d["metric"] for d in data_points]
        color = TARGET_COLORS.get(target, COLORS["slate"])
        label = TARGET_LABELS.get(target, target)

        ax.plot(iters, metrics, color=color, linewidth=2, alpha=0.9, label=label)
        ax.fill_between(iters, metrics, alpha=0.1, color=color)
        ax.set_xlabel("Iteration", fontweight="medium")
        ax.set_ylabel("Metric (Loss / RMSE)", fontweight="medium")
        ax.set_title(f"Convergence — {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Training Convergence{f' — {model_name}' if model_name else ''}",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout(pad=1.5)
    path = _save_temp_chart("convergence")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return path


def generate_usage_chart(usage_data: list[dict]) -> str | None:
    """Generate bar chart of usage/use-case predictions. Returns path to PNG."""
    if not usage_data:
        return None

    _setup_style()
    # Take top 8 use cases
    sorted_data = sorted(usage_data, key=lambda x: x.get("score", 0), reverse=True)[:8]
    if not sorted_data:
        return None

    names = [d.get("use_case", "?")[:25] for d in sorted_data]
    scores = [d.get("score", 0) for d in sorted_data]

    # Color by tier
    bar_colors = []
    for s in scores:
        if s >= 70:
            bar_colors.append(COLORS["emerald"])
        elif s >= 45:
            bar_colors.append(COLORS["amber"])
        else:
            bar_colors.append(COLORS["slate"])

    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.7 + 1)))
    y = np.arange(len(names))
    bars = ax.barh(y, scores, color=bar_colors, alpha=0.9, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score", fontweight="medium")
    ax.set_title("Material Use Case Predictions")
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, scores):
        tier = "Primary" if val >= 70 else ("Secondary" if val >= 45 else "Possible")
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} ({tier})", ha="left", va="center", fontsize=8, color="#374151")

    fig.tight_layout(pad=1.5)
    path = _save_temp_chart("usage")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return path


def _save_temp_chart(prefix: str) -> str:
    """Create a temp file path for chart output."""
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"piezo_{prefix}_", suffix=".png", delete=False,
    )
    tmp.close()
    return tmp.name
