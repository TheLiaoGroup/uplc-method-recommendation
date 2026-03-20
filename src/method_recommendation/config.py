"""Shared configuration for similarity analysis between product and substrates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_THRESHOLDS: Tuple[float, ...] = (
    0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0
)
DEFAULT_PERCENTILES: Tuple[int, ...] = (50, 80, 95)

# 根据你现在的目录结构设置默认输入文件
"""Configuration for similarity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Assume this file is located at:
# src/similarity_analysis/config.py
# Then project root is:
# ../../ from this file
"""Configuration for similarity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Assume this file is located at:
# src/similarity_analysis/config.py
# Project root is:
# ../../ from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "method_recommendation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "similarity_analysis"

DEFAULT_INPUT_FILES = (
    RESULTS_DIR / "All-Reaction" / "all_reaction_evaluated.csv",
    RESULTS_DIR / "Exp-Reaction" / "exp_reaction_evaluated.csv",
)


@dataclass(slots=True)
class SimilarityConfig:
    """Configuration container for similarity analysis."""

    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    percentiles: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)
    max_cores: int | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    input_files: tuple[str, ...] = field(
        default_factory=lambda: tuple(str(path.resolve()) for path in DEFAULT_INPUT_FILES)
    )
    dataset_name_map: dict[str, str] = field(
        default_factory=lambda: {
            "all_reaction_evaluated.csv": "All Reaction Data",
            "exp_reaction_evaluated.csv": "Experimental Reaction Data",
        }
    )

    def __post_init__(self) -> None:
        """Normalize paths after initialization."""
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.input_files = tuple(
            str(Path(path).expanduser().resolve()) for path in self.input_files
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
    "all_reaction_evaluated": "All data",
    "exp_reaction_evaluated": "Experimental data",
}


__all__ = [
    "DEFAULT_DATASET_NAME_MAP",
    "DEFAULT_INPUT_FILES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PERCENTILES",
    "DEFAULT_THRESHOLDS",
    "PROJECT_ROOT",
    "RESEARCH_COLORS",
    "SimilarityConfig",
]