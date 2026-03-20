"""Command-line interface for similarity analysis."""

from __future__ import annotations

import logging

from .config import SimilarityConfig
from .pipeline import SimilarityAnalyzer

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def main() -> None:
    """Run similarity analysis with fixed configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    logger = logging.getLogger(__name__)

    config = SimilarityConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using fixed similarity analysis configuration.")
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Input files to analyze:")
    for input_file in config.input_files:
        logger.info("  - %s", input_file)

    analyzer = SimilarityAnalyzer(config=config)
    analyzer.run(list(config.input_files))


if __name__ == "__main__":  # pragma: no cover
    main()