"""Shared configuration for similarity analysis between product and substrates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0)
DEFAULT_PERCENTILES: Tuple[int, ...] = (50, 80, 95)
DEFAULT_INPUT_FILES: Tuple[str, ...] = (
    "./4-All-Reaction-data-results/4-all-reactiondata_evaluated.csv",
    "./4-Exp-Reaction-data-results/Exp-Reaction-data_evaluated.csv",
)

RESEARCH_COLORS: Dict[str, str] = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

DEFAULT_DATASET_NAME_MAP: Dict[str, str] = {
    "4-all-reactiondata_evaluated": "All data",
    "Exp-Reaction-data_evaluated": "Experimental data",
}


@dataclass(frozen=True)
class SimilarityConfig:
    """Configuration bundle for similarity analysis."""

    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS
    percentiles: Tuple[int, ...] = DEFAULT_PERCENTILES
    max_cores: int = 26
    output_dir: Path = Path("./5-similarity-betweenPandS")
    input_files: Tuple[str, ...] = DEFAULT_INPUT_FILES
    dataset_name_map: Dict[str, str] = field(
        default_factory=lambda: DEFAULT_DATASET_NAME_MAP.copy()
    )

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


__all__ = [
    "DEFAULT_DATASET_NAME_MAP",
    "DEFAULT_INPUT_FILES",
    "DEFAULT_PERCENTILES",
    "DEFAULT_THRESHOLDS",
    "RESEARCH_COLORS",
    "SimilarityConfig",
]
