"""Statistical helpers for similarity analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SIMILARITY_BINS: List[float] = [round(i * 0.1, 1) for i in range(11)]
SIMILARITY_BIN_LABELS: List[str] = [
    "0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0",
]


def compute_similarity_statistics(
    predictable_df: pd.DataFrame,
    thresholds: Sequence[float],
    percentiles: Sequence[int],
) -> Dict[str, Any]:
    stats_results: Dict[str, Any] = {}
    if predictable_df.empty:
        return stats_results

    values = predictable_df["max_similarity"].to_numpy()
    stats_results["basic_stats"] = {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }

    stats_results["percentiles"] = {f"P{p}": float(np.percentile(values, p)) for p in percentiles}

    threshold_counts: Dict[float, Dict[str, float]] = {}
    for threshold in thresholds:
        mask = predictable_df["max_similarity"] >= threshold
        count = int(mask.sum())
        threshold_counts[threshold] = {
            "count": count,
            "proportion": float(count / len(values)),
        }
    stats_results["threshold_counts"] = threshold_counts

    by_predictor = predictable_df.groupby("predicted_by")["max_similarity"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    stats_results["by_predictor"] = by_predictor.to_dict("index")

    digitized = np.digitize(values, SIMILARITY_BINS, right=False) - 1
    bin_counts: Dict[str, int] = {}
    for idx, label in enumerate(SIMILARITY_BIN_LABELS):
        bin_counts[label] = int((digitized == idx).sum())
    stats_results["bin_distribution"] = bin_counts

    if "S1" in by_predictor.index and "S2" in by_predictor.index:
        s1_vals = predictable_df[predictable_df["predicted_by"] == "S1"]["max_similarity"].to_numpy()
        s2_vals = predictable_df[predictable_df["predicted_by"] == "S2"]["max_similarity"].to_numpy()
        if len(s1_vals) > 1 and len(s2_vals) > 1:
            t_stat, p_val = stats.ttest_ind(s1_vals, s2_vals, equal_var=False)
            stats_results["ttest"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }

    return stats_results


def analyze_best_score_data(
    df: pd.DataFrame,
    base_name: str,
    thresholds: Sequence[float],
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    subset = df[df["pred_best_score"] == 1].copy()
    if subset.empty:
        return None

    total = len(subset)
    predictable = subset[subset["max_similarity"] > 0].copy()
    stats_results: Dict[str, Any] = {
        "total_count": total,
        "predictable_count": len(predictable),
        "predictable_proportion": (len(predictable) / total) if total else 0,
    }

    if predictable.empty:
        return stats_results

    predictable["similarity_bin"] = pd.cut(
        predictable["max_similarity"],
        bins=SIMILARITY_BINS,
        labels=SIMILARITY_BIN_LABELS,
        include_lowest=True,
    )

    bin_counts = predictable["similarity_bin"].value_counts().reindex(SIMILARITY_BIN_LABELS, fill_value=0)
    bin_percentages = (bin_counts / len(predictable) * 100).round(2)
    stats_results["similarity_distribution"] = {
        "bins": SIMILARITY_BIN_LABELS,
        "counts": bin_counts.to_dict(),
        "percentages": bin_percentages.to_dict(),
    }

    stats_results["similarity_measures"] = {
        "mean": float(predictable["max_similarity"].mean()),
        "median": float(predictable["max_similarity"].median()),
        "std": float(predictable["max_similarity"].std()),
        "min": float(predictable["max_similarity"].min()),
        "max": float(predictable["max_similarity"].max()),
        "q1": float(predictable["max_similarity"].quantile(0.25)),
        "q3": float(predictable["max_similarity"].quantile(0.75)),
    }

    threshold_stats: Dict[float, Dict[str, float]] = {}
    for threshold in [thr for thr in thresholds if thr <= 1]:
        count = int((predictable["max_similarity"] >= threshold).sum())
        threshold_stats[threshold] = {
            "count": count,
            "proportion_of_total": count / total if total else 0,
            "proportion_of_predictable": count / len(predictable) if len(predictable) else 0,
        }
    stats_results["threshold_analysis"] = threshold_stats

    if "predicted_by" in predictable.columns:
        grouped = predictable.groupby("predicted_by")["max_similarity"].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )
        stats_results["by_predictor"] = grouped.round(3).to_dict("index")

    save_best_score_details(predictable, thresholds, output_dir, base_name)
    return stats_results


def save_best_score_details(
    predictable_df: pd.DataFrame,
    thresholds: Sequence[float],
    output_dir: Path,
    base_name: str,
) -> None:
    if predictable_df.empty:
        return

    output_csv = output_dir / f"{base_name}_pred_best_score_1_analysis.csv"
    detailed_df = predictable_df.copy()
    available_columns = [
        col
        for col in [
            "P",
            "S1",
            "S2",
            "pred_best_score",
            "sim_S1_P",
            "sim_S2_P",
            "max_similarity",
            "predicted_by",
            "similarity_bin",
        ]
        if col in detailed_df.columns
    ]
    detailed_df = detailed_df[available_columns]
    for threshold in thresholds:
        if threshold <= 1:
            detailed_df[f"similarity_geq_{int(threshold * 100)}"] = (
                detailed_df["max_similarity"] >= threshold
            ).astype(int)
    detailed_df.to_csv(output_csv, index=False)


def save_statistics_workbook(
    stats_results: Mapping[str, Any],
    best_score_stats: Optional[Mapping[str, Any]],
    output_dir: Path,
    base_name: str,
) -> Path:
    output_path = output_dir / f"{base_name}_similarity_statistics.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        basic = stats_results.get("basic_stats")
        if basic:
            pd.DataFrame([basic]).to_excel(writer, sheet_name="Basic_Statistics", index=False)

        percentiles = stats_results.get("percentiles")
        if percentiles:
            pd.DataFrame(percentiles.items(), columns=["Percentile", "Value"]).to_excel(
                writer, sheet_name="Percentiles", index=False
            )

        thresholds = stats_results.get("threshold_counts")
        if thresholds:
            threshold_df = pd.DataFrame(
                [{"Threshold": thr, **data} for thr, data in thresholds.items()]
            )
            threshold_df.to_excel(writer, sheet_name="Threshold_Analysis", index=False)

        by_predictor = stats_results.get("by_predictor")
        if by_predictor:
            pd.DataFrame(by_predictor).T.to_excel(writer, sheet_name="Predictor_Distribution")

        bin_distribution = stats_results.get("bin_distribution")
        if bin_distribution:
            bin_df = pd.DataFrame(
                [
                    {
                        "Bin": bin_label,
                        "Count": count,
                        "Proportion": count / sum(bin_distribution.values())
                        if sum(bin_distribution.values())
                        else 0,
                    }
                    for bin_label, count in bin_distribution.items()
                ]
            )
            bin_df.to_excel(writer, sheet_name="Bin_Distribution", index=False)

        ttest = stats_results.get("ttest")
        if ttest:
            pd.DataFrame([ttest]).to_excel(writer, sheet_name="Statistical_Tests", index=False)

        if best_score_stats:
            overview = pd.DataFrame(
                [
                    {
                        "Total_reactions_pred_best_score_1": best_score_stats.get("total_count", 0),
                        "Predictable_reactions": best_score_stats.get("predictable_count", 0),
                        "Predictable_proportion": best_score_stats.get("predictable_proportion", 0),
                    }
                ]
            )
            overview.to_excel(writer, sheet_name="PredBestScore1_Overview", index=False)

            similarity_measures = best_score_stats.get("similarity_measures")
            if similarity_measures:
                pd.DataFrame([similarity_measures]).to_excel(
                    writer, sheet_name="PredBestScore1_SimilarityStats", index=False
                )

            similarity_distribution = best_score_stats.get("similarity_distribution")
            if similarity_distribution:
                dist_df = pd.DataFrame(
                    [
                        {
                            "Similarity_Range": bin_label,
                            "Count": similarity_distribution["counts"].get(bin_label, 0),
                            "Percentage": similarity_distribution["percentages"].get(bin_label, 0),
                        }
                        for bin_label in similarity_distribution["bins"]
                    ]
                )
                dist_df.to_excel(writer, sheet_name="PredBestScore1_Distribution", index=False)

            threshold_analysis = best_score_stats.get("threshold_analysis")
            if threshold_analysis:
                thresh_df = pd.DataFrame(
                    [
                        {
                            "Threshold": f"≥{int(thr * 100)}%",
                            "Count": data["count"],
                            "Proportion_of_Total": data["proportion_of_total"],
                            "Proportion_of_Predictable": data["proportion_of_predictable"],
                        }
                        for thr, data in threshold_analysis.items()
                    ]
                )
                thresh_df.to_excel(writer, sheet_name="PredBestScore1_Thresholds", index=False)

            best_by_predictor = best_score_stats.get("by_predictor")
            if best_by_predictor:
                pd.DataFrame(best_by_predictor).T.to_excel(writer, sheet_name="PredBestScore1_ByPredictor")

    return output_path


__all__ = [
    "analyze_best_score_data",
    "compute_similarity_statistics",
    "save_best_score_details",
    "save_statistics_workbook",
]
