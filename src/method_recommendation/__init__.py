"""Public API for the method recommendation toolkit."""

from .cli import main
from .config import (
    ALL_FEATURES,
    EvaluationSettings,
    FEATURE_COLS,
    FIXED_METHOD_ORDER,
    METHOD_RANGE_CONFIG,
    MODEL_NAME_PATTERNS,
    resolve_model_dirs,
    resolve_smarts_file,
)
from .evaluation import UnifiedEvaluationSystem
from .model_hub import ModelHub
from .prediction import PredictionEvaluator
from .processor import ReactionDataProcessor

__all__ = [
    "ALL_FEATURES",
    "EvaluationSettings",
    "FEATURE_COLS",
    "FIXED_METHOD_ORDER",
    "METHOD_RANGE_CONFIG",
    "MODEL_NAME_PATTERNS",
    "ModelHub",
    "PredictionEvaluator",
    "ReactionDataProcessor",
    "UnifiedEvaluationSystem",
    "main",
    "resolve_model_dirs",
    "resolve_smarts_file",
]
