"""Visualization utilities for similarity analysis reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .config import RESEARCH_COLORS
from .statistics import SIMILARITY_BIN_LABELS


def create_combined_visualizations(
    all_best_score_data: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    dataset_name_map: Mapping[str, str],
    color_cycle = tuple(RESEARCH_COLORS.values()),
    output_name: str = "combined_pred_best_score_1_analysis",
) -> None:
    if not all_best_score_data:
        return

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "Arial",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    dataset_keys = list(all_best_score_data.keys())
    dataset_titles = [dataset_name_map.get(key, key) for key in dataset_keys]
    colors = list(color_cycle)

    ax1 = axes[0]
    x = np.arange(len(SIMILARITY_BIN_LABELS))
    width = 0.7 / max(1, len(dataset_keys))
    max_percentage = 0.0

    for idx, key in enumerate(dataset_keys):
        stats = all_best_score_data[key]
        distribution = stats.get("similarity_distribution")
        if not distribution:
            continue
        percentages = [distribution["percentages"].get(bin_label, 0) for bin_label in SIMILARITY_BIN_LABELS]
        offset = (idx - (len(dataset_keys) - 1) / 2) * width
        ax1.bar(
            x + offset,
            percentages,
            width,
            label=dataset_titles[idx],
            color=colors[idx % len(colors)],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        max_percentage = max(max_percentage, max(percentages, default=0))

    ax1.set_xlabel("Similarity Range", fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    ax1.set_title("(a) Similarity Range Distribution", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SIMILARITY_BIN_LABELS, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    if max_percentage:
        ax1.set_ylim(0, max_percentage * 1.15)
    ax1.legend(title="Dataset", loc="upper right", frameon=True, fancybox=True, framealpha=0.9)

    ax2 = axes[1]
    positions = []
    labels = []
    box_data = []
    box_colors = []
    pos_counter = 0

    for idx, key in enumerate(dataset_keys):
        stats = all_best_score_data[key]
        by_predictor = stats.get("by_predictor")
        if not by_predictor:
            continue
        display_name = dataset_titles[idx]
        for predictor in ("S1", "S2"):
            if predictor not in by_predictor:
                continue
            predictor_stats = by_predictor[predictor]
            mean_value = predictor_stats.get("max_similarity_mean")
            std_value = predictor_stats.get("max_similarity_std") or 0.0
            count_value = predictor_stats.get("max_similarity_count") or 0
            if mean_value is None or count_value == 0:
                continue
            np.random.seed(42 + idx)
            simulated = np.random.normal(mean_value, std_value if std_value else 1e-6, min(count_value, 1000))
            simulated = np.clip(simulated, 0, 1)
            positions.append(pos_counter)
            labels.append(f"{display_name}\n{predictor}")
            box_data.append(simulated)
            box_colors.append(colors[idx % len(colors)])
            pos_counter += 1

    if box_data:
        box_plot = ax2.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1},
            capprops={"linewidth": 1},
        )
        for patch, color in zip(box_plot["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        ax2.set_xlabel("Dataset and Predictor", fontweight="bold")
        ax2.set_ylabel("Maximum Similarity", fontweight="bold")
        ax2.set_title("(b) Similarity by Predictor", fontweight="bold", fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        all_values = np.concatenate(box_data)
        if len(all_values):
            ax2.set_ylim(max(0, all_values.min() - 0.05), min(1, all_values.max() + 0.05))
            for level in (0.5, 0.7, 0.9):
                ax2.axhline(y=level, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()


__all__ = ["create_combined_visualizations"]
