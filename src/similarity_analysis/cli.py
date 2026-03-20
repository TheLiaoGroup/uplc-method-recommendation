from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .config import SimilarityConfig
from .pipeline import SimilarityAnalyzer

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# 基于当前文件位置自动定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "similarity_analysis"


def _normalize_input_paths(inputs: Sequence[str] | None) -> list[Path] | None:
    """将命令行输入路径统一解析为绝对路径；未提供则返回 None。"""
    if not inputs:
        return None
    return [Path(path).expanduser().resolve() for path in inputs]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze P/S similarity statistics for evaluated reaction CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=None,
        help=(
            "Optional list of CSV files. "
            "Defaults to configured input files if omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where analysis artifacts will be stored.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=None,
        help="Optional override for maximum CPU cores used during fingerprint generation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )

    base_config = SimilarityConfig()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_inputs = _normalize_input_paths(args.inputs)

    # 如果命令行没有传 inputs，就把 config 里的 input_files 也转成绝对路径
    config_inputs = (
        cli_inputs
        if cli_inputs is not None
        else [Path(path).expanduser().resolve() for path in base_config.input_files]
    )

    config = SimilarityConfig(
        thresholds=base_config.thresholds,
        percentiles=base_config.percentiles,
        max_cores=args.max_cores or base_config.max_cores,
        output_dir=output_dir,
        input_files=config_inputs,
        dataset_name_map=base_config.dataset_name_map.copy(),
    )

    analyzer = SimilarityAnalyzer(config=config)
    analyzer.run(config.input_files)


if __name__ == "__main__":  # pragma: no cover
    main()