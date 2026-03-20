"""Prediction utilities to pick the best UPLC method for a reaction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .config import FIXED_METHOD_ORDER, METHOD_RANGE_CONFIG
from .evaluation import UnifiedEvaluationSystem
from .model_hub import ModelHub

logger = logging.getLogger(__name__)


class PredictionEvaluator:
    """Evaluates retention-time predictions across all supported methods."""

    def __init__(self, model_hub: ModelHub, evaluator: UnifiedEvaluationSystem) -> None:
        self.model_hub = model_hub
        self.evaluator = evaluator

    def evaluate_predictions(
        self,
        smiles_list: Sequence[Optional[str]],
        row_index: int,
        output_dir: str | Path = "./4-All-Reaction-data-results",
    ) -> Dict[str, Any]:
        valid_smiles: List[str] = []
        all_predictions: List[Optional[Dict[str, Optional[float]]]] = []

        for smiles in smiles_list:
            if smiles and pd.notna(smiles):
                preds = self.model_hub.predict(smiles)
                all_predictions.append(preds)
                valid_smiles.append(smiles)
            else:
                all_predictions.append(None)

        if len(valid_smiles) < 3:
            return {
                "best_methods": None,
                "best_score": None,
                "all_scores": {},
                "predictions": all_predictions,
                "predicted_values": {},
                "error": f"Insufficient valid SMILES: {len(valid_smiles)}/3",
            }

        method_predictions: Dict[str, Optional[List[float]]] = {}
        for method in FIXED_METHOD_ORDER:
            method_values: List[float] = []
            complete = True
            for prediction in all_predictions:
                if prediction and method in prediction and prediction[method] is not None:
                    method_values.append(prediction[method])
                else:
                    complete = False
                    break
            method_predictions[method] = method_values if complete and len(method_values) == 3 else None

        method_scores: Dict[str, Dict[str, Any]] = {}
        for method, values in method_predictions.items():
            if values is None:
                method_scores[method] = {
                    "score": None,
                    "values": None,
                    "range": METHOD_RANGE_CONFIG.get(method, (30, 120)),
                    "valid": False,
                    "error": "Incomplete predictions",
                }
                continue

            value_range = METHOD_RANGE_CONFIG.get(method, (30, 120))
            evaluation = self.evaluator.evaluate(values, value_range)
            method_scores[method] = {
                "score": evaluation["final_score"],
                "values": values,
                "range": value_range,
                "valid": True,
            }

        valid_scores = {
            method: details
            for method, details in method_scores.items()
            if details["valid"] and details["score"] is not None and details["score"] >= 0
        }
        if valid_scores:
            best_score = max(details["score"] for details in valid_scores.values())
            best_methods = [method for method, details in valid_scores.items() if details["score"] == best_score]
            best_methods.sort(key=lambda method: FIXED_METHOD_ORDER.index(method))
        else:
            best_methods = []
            best_score = None

        datasets: List[List[float]] = []
        method_names: List[str] = []
        for method in FIXED_METHOD_ORDER:
            if method_scores[method]["valid"]:
                datasets.append(method_scores[method]["values"])
                method_names.append(method)

        if datasets:
            row_output_dir = Path(output_dir) / f"row_{row_index}"
            row_output_dir.mkdir(parents=True, exist_ok=True)
            self.evaluator.evaluate_datasets(
                datasets=datasets,
                method_names=method_names,
                save_csv=True,
                save_plot=True,
                output_dir=row_output_dir,
            )

        best_method_values = {
            method: method_predictions[method]
            for method in best_methods
            if method in method_predictions and method_predictions[method] is not None
        }

        return {
            "best_methods": best_methods,
            "best_methods_str": ", ".join(best_methods) if best_methods else "None",
            "best_score": best_score,
            "all_scores": method_scores,
            "predictions": all_predictions,
            "method_predictions": method_predictions,
            "best_method_values": best_method_values,
            "error": None,
        }


__all__ = ["PredictionEvaluator"]
