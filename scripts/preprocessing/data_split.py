from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CLUSTER_RESULT_DIR = PROJECT_ROOT / "results" / "preprocessing" / "clustering"
OUTPUT_SPLIT_DIR = PROJECT_ROOT / "data" / "train_test_split"
FILES = [
    "AM-I-filtered/AM-I-filtered_with_labels_k4.csv",
    "AM-II-filtered/AM-II-filtered_with_labels_k4.csv",
]
RANDOM_SEED = 42
TEST_SIZE = 0.1
CLUSTER_COL = "UMAP_Cluster"


def split_single_file(file_rel_path: str) -> None:
    file_path = INPUT_CLUSTER_RESULT_DIR / file_rel_path
    if not file_path.exists():
        print(f"[SKIP] File not found: {file_path}")
        return

    df = pd.read_csv(file_path).dropna().reset_index(drop=True)
    if CLUSTER_COL not in df.columns:
        print(f"[SKIP] Missing column '{CLUSTER_COL}' in {file_path.name}")
        return

    train_parts = []
    test_parts = []

    # Split within each cluster to preserve cluster composition.
    for cluster in sorted(df[CLUSTER_COL].dropna().unique()):
        cluster_df = df[df[CLUSTER_COL] == cluster].reset_index(drop=True)
        if len(cluster_df) < 2:
            train_parts.append(cluster_df)
            continue

        train_df, test_df = train_test_split(
            cluster_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
        )
        train_parts.append(train_df)
        test_parts.append(test_df)

    train_data = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_data = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    duplicate_rows = train_data.merge(test_data, how="inner")
    if duplicate_rows.empty:
        print(f"[OK] No duplicate samples between train and test: {file_rel_path}")
    else:
        print(
            f"[WARN] Potential leakage for {file_rel_path}: "
            f"{len(duplicate_rows)} duplicate rows"
        )

    base_name = file_path.stem
    train_output_path = OUTPUT_SPLIT_DIR / f"{base_name}_train.csv"
    test_output_path = OUTPUT_SPLIT_DIR / f"{base_name}_test.csv"

    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)
    print(f"Saved train: {train_output_path}")
    print(f"Saved test:  {test_output_path}")


def main() -> None:
    OUTPUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for file_rel_path in FILES:
        split_single_file(file_rel_path)


if __name__ == "__main__":
    main()