"""High-level orchestration for similarity analysis pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from .config import SimilarityConfig
from .fingerprints import (
    build_fingerprint_cache,
    compute_similarities,
    split_column_caches,
)
from .statistics import (
    analyze_best_score_data,
    compute_similarity_statistics,
    save_statistics_workbook,
)
from .visualization import create_combined_visualizations

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """Runs similarity analysis between product and substrates for CSV files."""

    def __init__(self, config: SimilarityConfig | None = None) -> None:
        self.config = config or SimilarityConfig()
        self.output_dir = self.config.ensure_output_dir()

    # ------------------------------------------------------------------
    def analyze_file(self, csv_path: str | Path) -> Dict[str, object]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(path)

        logger.info("Reading %s", path)
        df = pd.read_csv(path)
        if "pred_best_score" not in df.columns and "best_score" in df.columns:
            df = df.rename(columns={"best_score": "pred_best_score"})

        required = {"P", "S1", "S2", "pred_best_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        cache = build_fingerprint_cache(df, ["P", "S1", "S2"], self.config.max_cores)
        p_cache, s1_cache, s2_cache = split_column_caches(df, cache)
        sim_s1, sim_s2, max_sim, predicted_by = compute_similarities(
            df,
            p_cache,
            s1_cache,
            s2_cache,
            self.config.max_cores,
        )

        df["sim_S1_P"] = sim_s1
        df["sim_S2_P"] = sim_s2
        df["max_similarity"] = max_sim
        df["predicted_by"] = predicted_by

        predictable_df = df[df["max_similarity"] > 0].copy()
        similarity_stats = compute_similarity_statistics(
            predictable_df,
            self.config.thresholds,
            self.config.percentiles,
        )

        base_name = path.stem
        best_score_stats = analyze_best_score_data(
            df,
            base_name,
            self.config.thresholds,
            self.output_dir,
        )

        enriched_path = self.output_dir / f"{base_name}_with_similarity.csv"
        df.to_csv(enriched_path, index=False)
        logger.info("Saved enriched data to %s", enriched_path)

        if not predictable_df.empty:
            predictable_path = self.output_dir / f"{base_name}_predictable_reactions.csv"
            predictable_df.to_csv(predictable_path, index=False)
            logger.info("Saved predictable subset to %s", predictable_path)

        workbook_path = save_statistics_workbook(
            similarity_stats,
            best_score_stats,
            self.output_dir,
            base_name,
        )
        logger.info("Saved statistics workbook to %s", workbook_path)

        return {
            "dataframe": df,
            "predictable_subset": predictable_df,
            "similarity_stats": similarity_stats,
            "best_score_stats": best_score_stats,
            "input": path,
            "enriched_path": enriched_path,
            "workbook_path": workbook_path,
        }

    # ------------------------------------------------------------------
    def run(self, csv_files: Optional[Sequence[str | Path]] = None) -> List[Dict[str, object]]:
        files = list(csv_files) if csv_files else list(self.config.input_files)
        files = [Path(f) for f in files]

        existing_files = [path for path in files if path.exists()]
        missing_files = [path for path in files if not path.exists()]

        for missing in missing_files:
            logger.warning("Missing file: %s", missing)
        if not existing_files:
            logger.error("No valid CSV files provided.")
            return []

        logger.info("Analyzing %d CSV file(s)", len(existing_files))
        results: List[Dict[str, object]] = []
        combined_best_scores: Dict[str, Mapping[str, object]] = {}

        for path in existing_files:
            logger.info("%s", "=" * 60)
            logger.info("Starting analysis for %s", path.name)
            logger.info("%s", "=" * 60)
            try:
                result = self.analyze_file(path)
                results.append(result)
                if result.get("best_score_stats"):
                    combined_best_scores[path.stem] = result["best_score_stats"]  # type: ignore[index]
            except Exception as exc:
                logger.exception("Failed to analyze %s: %s", path, exc)

        if combined_best_scores:
            create_combined_visualizations(
                combined_best_scores,
                self.output_dir,
                self.config.dataset_name_map,
            )

        saved_files = sorted(self.output_dir.glob("*"))
        logger.info("Generated %d file(s) in %s", len(saved_files), self.output_dir)
        return results


__all__ = ["SimilarityAnalyzer"]
