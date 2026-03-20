"""Public API for similarity analysis helpers."""

from .cli import main
from .config import SimilarityConfig
from .pipeline import SimilarityAnalyzer

__all__ = ["SimilarityAnalyzer", "SimilarityConfig", "main"]
