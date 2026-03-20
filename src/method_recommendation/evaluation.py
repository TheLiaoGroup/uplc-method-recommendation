"""Unified evaluation utilities for retention-time predictions."""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIXED_METHOD_ORDER, METHOD_RANGE_CONFIG

logger = logging.getLogger(__name__)


class UnifiedEvaluationSystem:
    """Scores predicted retention times with range and interval checks."""

    def __init__(
        self,
        min_interval: float = 9.0,
        distance_weight: float = 10.0,
        range_weight: float = 0.6,
        importance_weight: float = 5.0,
        strict_penalty: bool = True,
        default_range: Tuple[float, float] = (30.0, 120.0),
        method_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    ) -> None:
        self.min_interval = min_interval
        self.distance_weight = distance_weight
        self.range_weight = range_weight
        self.importance_weight = importance_weight
        self.strict_penalty = strict_penalty
        self.default_range = default_range
        self.method_ranges = dict(method_ranges or METHOD_RANGE_CONFIG)

    # ---- core scoring helpers -------------------------------------------------
    def _calculate_interval_score(self, values: Sequence[float]) -> tuple[float, List[Dict[str, Any]]]:
        violations: List[Dict[str, Any]] = []
        penalty = 0.0
        sorted_vals = sorted(values)

        for idx in range(len(sorted_vals) - 1):
            gap = sorted_vals[idx + 1] - sorted_vals[idx]
            if gap >= self.min_interval:
                continue

            shortage = self.min_interval - gap
            w1 = (1 / (values.index(sorted_vals[idx]) + 1)) ** 3
            w2 = (1 / (values.index(sorted_vals[idx + 1]) + 1)) ** 3
            weight = max(w1, w2) * self.importance_weight
            penalty_delta = shortage * weight * self.distance_weight / len(values)
            penalty += penalty_delta
            violations.append(
                {
                    "type": "interval",
                    "values": [sorted_vals[idx], sorted_vals[idx + 1]],
                    "required": self.min_interval,
                    "actual": gap,
                    "penalty": penalty_delta,
                }
            )

        return penalty, violations

    def _calculate_range_score(
        self,
        values: Sequence[float],
        value_range: Tuple[float, float],
    ) -> tuple[float, List[Dict[str, Any]]]:
        violations: List[Dict[str, Any]] = []
        penalty = 0.0
        min_val, max_val = value_range

        for idx, value in enumerate(values):
            if min_val <= value <= max_val:
                continue

            importance = idx + 1
            importance_weight = math.exp(-0.5 * (importance - 1)) * self.importance_weight
            distance = min_val - value if value < min_val else value - max_val
            penalty_delta = distance * importance_weight * self.range_weight / len(values)
            penalty += penalty_delta
            violations.append(
                {
                    "type": "range",
                    "value": value,
                    "importance": importance,
                    "distance": distance,
                    "penalty": penalty_delta,
                }
            )

            if idx == 0 and self.strict_penalty:
                return -1.0, violations

        return penalty, violations

    def _normalize_score(
        self,
        interval_penalty: float,
        range_penalty: float,
        n_points: int,
        value_range: Tuple[float, float],
    ) -> float:
        if range_penalty == -1:
            return -1.0

        max_distance_penalty = self.min_interval * (n_points - 1) * sum(1 / i for i in range(1, n_points + 1))
        max_range_penalty = max(abs(value_range[0]), abs(value_range[1])) * sum(1 / i for i in range(1, n_points + 1))

        total_penalty = self.distance_weight * interval_penalty + self.range_weight * range_penalty
        max_total_penalty = self.distance_weight * max_distance_penalty + self.range_weight * max_range_penalty
        if max_total_penalty == 0:
            return 1.0

        return max(0.0, 1 - total_penalty / max_total_penalty)

    # ---- public API ------------------------------------------------------------
    def evaluate(self, values: Sequence[float], value_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        value_range = value_range or self.default_range

        interval_penalty, interval_violations = self._calculate_interval_score(values)
        range_penalty, range_violations = self._calculate_range_score(values, value_range)
        score = self._normalize_score(interval_penalty, range_penalty, len(values), value_range)

        return {
            "values": list(values),
            "distance_penalty": interval_penalty,
            "range_penalty": range_penalty,
            "final_score": score,
            "distance_violations": interval_violations,
            "range_violations": range_violations,
            "is_strict_penalty": score == -1,
            "value_range": value_range,
        }

    def evaluate_datasets(
        self,
        datasets: Iterable[Sequence[float]],
        method_names: Sequence[str],
        save_csv: bool = True,
        save_plot: bool = True,
        output_dir: str | Path = "./4-All-Reaction-data-results/",
    ) -> List[Dict[str, Any]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for idx, values in enumerate(datasets):
            method = method_names[idx]
            value_range = self.method_ranges.get(method, self.default_range)
            evaluation = self.evaluate(values, value_range)
            evaluation["dataset_id"] = idx
            evaluation["method_name"] = method
            evaluation["xmax"] = 210 if value_range[1] > 120 else 180
            results.append(evaluation)

        if save_csv:
            self._save_csv_report(results, output_path)
        if save_plot:
            self._create_visualization(results, output_path)
        return results

    # ---- reporting -------------------------------------------------------------
    def _save_csv_report(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        rows = []
        for result in results:
            violations: list[str] = []
            for violation in result["distance_violations"]:
                violations.append(
                    f"Interval: {violation['values']} req={violation['required']} act={violation['actual']:.2f}"
                )
            for violation in result["range_violations"]:
                violations.append(
                    f"Range: {violation['value']} imp={violation['importance']} dist={violation['distance']:.2f}"
                )

            rows.append(
                {
                    "Dataset_ID": result["dataset_id"],
                    "Method": result["method_name"],
                    "Values": str(result["values"]),
                    "Distance_Penalty": result["distance_penalty"],
                    "Range_Penalty": result["range_penalty"],
                    "Final_Score": result["final_score"],
                    "Is_Strict_Penalty": result["is_strict_penalty"],
                    "Value_Range": str(result["value_range"]),
                    "Violations": "; ".join(violations) if violations else "None",
                }
            )

        report_path = output_dir / "evaluation_results.csv"
        pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
        logger.info("Evaluation report saved to %s", report_path)

    def _create_visualization(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        if not results:
            return

        results_sorted = sorted(results, key=lambda item: FIXED_METHOD_ORDER.index(item["method_name"]))
        n_rows = len(results_sorted)
        max_points = max(len(result["values"]) for result in results_sorted)

        fig, ax = plt.subplots(figsize=(14, max(5, n_rows * 1.1)))
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        colors = plt.cm.tab10.colors

        all_values = [value for result in results_sorted for value in result["values"]]
        global_min = min(min(all_values), 30) - 5
        xmax = max(result["xmax"] for result in results_sorted)
        ax.set_xlim(global_min, xmax)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.axvspan(30, 180, color="#D9D9D9", alpha=0.45, zorder=0)

        for idx, result in enumerate(results_sorted):
            y = n_rows - idx - 1
            values = result["values"]
            value_range = result["value_range"]

            for point_idx, value in enumerate(values):
                size = 300 / (point_idx + 1)
                color = colors[point_idx % 10]
                marker = "o" if value_range[0] <= value <= value_range[1] else "X"
                ax.scatter(
                    value,
                    y,
                    s=size,
                    c=[color],
                    marker=marker,
                    alpha=0.9,
                    edgecolors="k",
                    linewidths=1.5,
                    zorder=3,
                )

            sorted_values = sorted(values)
            for point_idx in range(len(sorted_values) - 1):
                if sorted_values[point_idx + 1] - sorted_values[point_idx] < self.min_interval:
                    ax.plot(sorted_values[point_idx : point_idx + 2], [y, y], "r-", lw=3, alpha=0.7, zorder=2)

            score_txt = f"{result['final_score']:.3f}" if result["final_score"] >= 0 else "Penalty"
            ax.text(
                xmax * 0.99,
                y,
                score_txt,
                ha="right",
                va="center",
                fontsize=15,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
            )

        ax.set_xlabel("Retention Time (s)", fontsize=17)
        ax.set_ylabel("UPLC Method", fontsize=17)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([result["method_name"] for result in reversed(results_sorted)], fontsize=16)
        ax.tick_params(axis="y", labelsize=16, length=8, width=1.5)
        ax.tick_params(axis="x", labelsize=16, length=8, width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=1.2)

        labels = ["P", "S1", "S2"][:max_points]
        legend_items = [
            plt.scatter([], [], s=250 // (idx + 1), c=[colors[idx % 10]], label=label)
            for idx, label in enumerate(labels)
        ]
        legend_items += [
            plt.scatter([], [], marker="X", c="gray", s=120, label="Out Range"),
            plt.Line2D([0], [0], color="red", lw=3, label="Interval Violation"),
        ]

        ax.legend(
            handles=legend_items,
            bbox_to_anchor=(0.5, 1.05),
            loc="lower center",
            ncol=len(legend_items),
            fontsize=15,
        )

        plot_path = output_dir / "comparison_chart.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=600, bbox_inches="tight")
        plt.close()
        logger.info("Comparison chart saved to %s", plot_path)


__all__ = ["UnifiedEvaluationSystem"]
