"""High-level orchestrator for batch processing reaction data files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .config import (
    DEFAULT_OUTPUT_DIR,
    EvaluationSettings,
    METHOD_RANGE_CONFIG,
    MODEL_NAME_PATTERNS,
    resolve_model_dirs,
    resolve_smarts_file,
)
from .evaluation import UnifiedEvaluationSystem
from .features import FeatureCalculator
from .model_hub import ModelHub
from .prediction import PredictionEvaluator

logger = logging.getLogger(__name__)


class ReactionDataProcessor:
    """Processes reaction CSV files and produces evaluation artifacts."""

    def __init__(
        self,
        evaluation_settings: EvaluationSettings = EvaluationSettings(),
        smarts_file: Optional[str | Path] = None,
        model_dir_overrides: Optional[Mapping[str, str | Path]] = None,
    ) -> None:
        smarts_path = resolve_smarts_file(smarts_file)
        self.feature_calculator = FeatureCalculator(smarts_path)

        model_dirs = resolve_model_dirs(model_dir_overrides)
        self.model_hub = ModelHub(model_dirs, MODEL_NAME_PATTERNS, self.feature_calculator)

        self.evaluator = UnifiedEvaluationSystem(
            min_interval=evaluation_settings.min_interval,
            distance_weight=evaluation_settings.distance_weight,
            range_weight=evaluation_settings.range_weight,
            importance_weight=evaluation_settings.importance_weight,
            strict_penalty=evaluation_settings.strict_penalty,
            default_range=evaluation_settings.default_range,
            method_ranges=METHOD_RANGE_CONFIG,
        )
        self.prediction_evaluator = PredictionEvaluator(self.model_hub, self.evaluator)
        self.settings = evaluation_settings

    def process_row(self, row: pd.Series, row_idx: int, output_dir: str | Path) -> Dict[str, Any]:
        smiles_list = [row.get("P"), row.get("S1"), row.get("S2")]
        prediction_result = self.prediction_evaluator.evaluate_predictions(
            smiles_list,
            row_index=row_idx,
            output_dir=output_dir,
        )

        result: Dict[str, Any] = {
            "pred_best_methods": prediction_result["best_methods"],
            "pred_best_methods_str": prediction_result.get("best_methods_str"),
            "pred_best_score": prediction_result.get("best_score"),
            "error": prediction_result.get("error"),
        }

        best_methods = prediction_result.get("best_methods") or []
        best_method_values = prediction_result.get("best_method_values", {})
        for method in best_methods:
            values = best_method_values.get(method)
            if not values:
                continue
            result[f"pred_{method}_P"] = values[0] if len(values) > 0 else None
            result[f"pred_{method}_S1"] = values[1] if len(values) > 1 else None
            result[f"pred_{method}_S2"] = values[2] if len(values) > 2 else None

        all_scores = prediction_result.get("all_scores", {})
        for method in self.evaluator.method_ranges.keys():
            details = all_scores.get(method)
            if details and details.get("valid"):
                result[f"pred_score_{method}"] = details.get("score")
            else:
                result[f"pred_score_{method}"] = None

        return result

    def process_file(
        self,
        input_file: str | Path,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> Optional[Path]:
        input_path = Path(input_file).expanduser().resolve()
        output_base = Path(output_dir).expanduser().resolve()
        output_base.mkdir(parents=True, exist_ok=True)

        logger.info("Reading file: %s", input_path)
        try:
            df = pd.read_csv(input_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read %s: %s", input_path, exc)
            return None

        missing_cols = [col for col in ["P", "S1", "S2"] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"File missing required columns: {missing_cols}")

        logger.info("Successfully read %d rows", len(df))
        results: list[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            logger.info("Processing row %d/%d", idx + 1, len(df))
            try:
                row_result = self.process_row(row, idx, output_base)
                results.append(row_result)
                logger.info(
                    "  Best method(s): %s | Score: %s",
                    row_result.get("pred_best_methods_str"),
                    row_result.get("pred_best_score"),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Error processing row %d: %s", idx + 1, exc)
                results.append(
                    {
                        "pred_best_methods": None,
                        "pred_best_methods_str": "None",
                        "pred_best_score": None,
                        "error": str(exc),
                    }
                )

        results_df = pd.DataFrame(results)
        for column in results_df.columns:
            df[column] = results_df[column]

        input_name = input_path.stem
        output_file = output_base / f"{input_name}_evaluated.csv"

        df.to_csv(output_file, index=False)
        logger.info("Results saved to %s", output_file)

        self._generate_statistics_report(df, output_base, input_name)
        return output_file

    def _generate_statistics_report(self, df: pd.DataFrame, output_dir: Path, input_name: str) -> None:
        stats: Dict[str, Any] = {
            "total_rows": len(df),
            "rows_with_prediction": df["pred_best_methods_str"].notna().sum(),
            "avg_pred_score": df["pred_best_score"].mean() if df["pred_best_score"].notna().any() else None,
        }

        stats["pred_score_distribution"] = {
            "excellent(0.9-1.0)": ((df["pred_best_score"] >= 0.9) & (df["pred_best_score"] <= 1.0)).sum(),
            "good(0.7-0.9)": ((df["pred_best_score"] >= 0.7) & (df["pred_best_score"] < 0.9)).sum(),
            "fair(0.5-0.7)": ((df["pred_best_score"] >= 0.5) & (df["pred_best_score"] < 0.7)).sum(),
            "poor(<0.5)": (df["pred_best_score"] < 0.5).sum(),
            "penalty(-1)": (df["pred_best_score"] == -1).sum()
            if df["pred_best_score"].notna().any()
            else 0,
        }

        if df["pred_best_methods_str"].notna().any():
            recommendations: list[str] = []
            for methods_str in df["pred_best_methods_str"].dropna():
                if methods_str == "None":
                    continue
                recommendations.extend(method.strip() for method in methods_str.split(","))

            if recommendations:
                from collections import Counter

                method_counts = Counter(recommendations)
                stats["method_recommendation_distribution"] = dict(method_counts)
                stats["method_recommendation_percentage"] = {
                    method: count / len(df) * 100 for method, count in method_counts.items()
                }

        stats_df = pd.DataFrame([stats])
        stats_file = output_dir / f"{input_name}_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info("Statistics report saved to %s", stats_file)

        logger.info("%s", "=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("%s", "=" * 60)
        logger.info("Total rows: %s", stats["total_rows"])
        logger.info("Successful prediction rows: %s", stats["rows_with_prediction"])
        avg_score = stats.get("avg_pred_score")
        if avg_score is not None:
            logger.info("Average prediction score: %.3f", avg_score)

        distribution = stats.get("method_recommendation_distribution")
        percentages = stats.get("method_recommendation_percentage")
        if distribution and percentages:
            logger.info("Method recommendation distribution (including ties):")
            for method, count in distribution.items():
                logger.info("  %s: %d times (%.1f%% of rows)", method, count, percentages[method])

        logger.info("Prediction score distribution:")
        if stats["rows_with_prediction"]:
            for category, count in stats["pred_score_distribution"].items():
                pct = count / stats["rows_with_prediction"] * 100 if stats["rows_with_prediction"] else 0
                logger.info("  %s: %d rows (%.1f%%)", category, count, pct)


__all__ = ["ReactionDataProcessor"]
