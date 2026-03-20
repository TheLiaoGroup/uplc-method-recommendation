"""Model loading utilities for the method recommendation system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np

from .features import FeatureCalculator

logger = logging.getLogger(__name__)


class ModelHub:
    """Loads and serves SVR models for every analytical method."""

    def __init__(
        self,
        model_dirs: Mapping[str, str | Path],
        model_name_patterns: Mapping[str, str],
        feature_calculator: FeatureCalculator,
    ) -> None:
        self.model_dirs = {method: Path(path).expanduser().resolve() for method, path in model_dirs.items()}
        self.model_name_patterns = model_name_patterns
        self.feature_calculator = feature_calculator
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

        self._load_models()

    def _locate_artifacts(self, method: str, directory: Path) -> tuple[Optional[Path], Optional[Path]]:
        if not directory.exists():
            return None, None

        model_path: Optional[Path] = None
        scaler_path: Optional[Path] = None

        if method in {"AM-I", "AM-II"}:
            for file in sorted(directory.glob("*.joblib")):
                if file.name.endswith("_scaler.joblib"):
                    scaler_path = file
                else:
                    model_path = file
        else:
            pattern = self.model_name_patterns.get(method, method)
            model_path = directory / f"{pattern}_svr_model.joblib"
            scaler_path = directory / f"{pattern}_scaler.joblib"

        if model_path and not model_path.exists():
            model_path = None
        if scaler_path and not scaler_path.exists():
            scaler_path = None

        return model_path, scaler_path

    def _load_models(self) -> None:
        for method, directory in self.model_dirs.items():
            model_path, scaler_path = self._locate_artifacts(method, directory)

            if not model_path or not scaler_path:
                logger.warning(
                    "Model artifacts missing for %s (model: %s, scaler: %s)",
                    method,
                    model_path or "missing",
                    scaler_path or "missing",
                )
                continue

            try:
                self.models[method] = joblib.load(model_path)
                self.scalers[method] = joblib.load(scaler_path)
                logger.info("Loaded %s artifacts from %s", method, directory)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to load %s artifacts: %s", method, exc)

    def predict(self, smiles: str) -> Dict[str, Optional[float]]:
        """Return predicted retention times across all loaded methods."""

        if not self.models:
            return {}

        features = self.feature_calculator(smiles)
        if features is None:
            logger.warning("Failed to calculate features for SMILES '%s'", smiles)
            return {method: None for method in self.models}

        base = features[:5]
        rest = features[5:]

        predictions: Dict[str, Optional[float]] = {}
        for method, model in self.models.items():
            scaler = self.scalers.get(method)
            if scaler is None:
                predictions[method] = None
                continue

            try:
                base_scaled = scaler.transform(base.reshape(1, -1))[0]
                full_vector = np.concatenate((base_scaled, rest)).reshape(1, -1)
                predictions[method] = float(model.predict(full_vector)[0])
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Prediction failed for %s: %s", method, exc)
                predictions[method] = None

        return predictions


__all__ = ["ModelHub"]
