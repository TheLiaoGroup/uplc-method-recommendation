"""Feature calculation utilities for the recommendation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ml import calc_features_from_smarts, load_smarts_patterns


class FeatureCalculator:
    """Caches SMARTS patterns for repeated feature extraction."""

    def __init__(self, smarts_file: str | Path):
        self.smarts_file = Path(smarts_file).expanduser().resolve()
        if not self.smarts_file.exists():
            raise FileNotFoundError(f"SMARTS file not found: {self.smarts_file}")

        self._patterns: Sequence[str] = load_smarts_patterns(str(self.smarts_file))

    @property
    def patterns(self) -> Sequence[str]:
        return self._patterns

    def __call__(self, smiles: str) -> Optional[np.ndarray]:
        return calc_features_from_smarts(smiles, self._patterns)


__all__ = ["FeatureCalculator"]
