from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_SPLIT_DIR = PROJECT_ROOT / "data" / "train_test_split"
RESULT_BASE_DIR = PROJECT_ROOT / "results" / "preprocessing" / "train-test-distribution"
RT_RESULT_FOLDER = RESULT_BASE_DIR / "rt-comparison"
PCA_RESULT_FOLDER = RESULT_BASE_DIR / "pca-structure-distribution"
TARGET_COL = "UV_RT-s"
ALPHA = 0.05
ALL_FEATURES = (
    ["MolWt", "logP", "TPSA", "H_bond_donors", "H_bond_acceptors"]
    + [f"col{i}" for i in range(823)]
    + [f"fp_{i}" for i in range(1024)]
)


def configure_plot_style() -> None:
    sns.set_theme(style="white")
    sns.set_context(
        "paper",
        rc={
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.5,
        },
    )
    plt.rcParams["font.family"] = "DejaVu Sans"
    np.random.seed(42)


def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path).dropna(subset=[TARGET_COL] + ALL_FEATURES)
    test_df = pd.read_csv(test_path).dropna(subset=[TARGET_COL] + ALL_FEATURES)
    return train_df, test_df


def load_fingerprints_from_csv(file_path: Path) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(file_path)
        fp_columns = [f"fp_{i}" for i in range(1024)]
        if all(col in df.columns for col in fp_columns):
            return df[fp_columns].values
        return None
    except Exception as exc:
        logging.error("Failed to load file %s: %s", file_path, exc)
        return None


def analyze_retention_time_pair(train_path: Path, test_path: Path) -> Dict[str, object]:
    base_name = train_path.stem.replace("_train", "")
    train_df, test_df = load_data(train_path, test_path)

    train_desc = train_df[TARGET_COL].describe()
    test_desc = test_df[TARGET_COL].describe()
    ks_stat, p_val = ks_2samp(train_df[TARGET_COL], test_df[TARGET_COL])

    plt.figure(figsize=(6, 4))
    sns.kdeplot(train_df[TARGET_COL], label="Train", fill=True, alpha=0.25, color="tab:blue")
    sns.kdeplot(test_df[TARGET_COL], label="Test", fill=True, alpha=0.25, color="tab:orange")
    plt.xlabel("Retention Time (s)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RT_RESULT_FOLDER / f"{base_name}_kde.png", dpi=600, bbox_inches="tight")
    plt.close()

    return {
        "dataset": base_name,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_mean": float(train_desc["mean"]),
        "train_std": float(train_desc["std"]),
        "test_mean": float(test_desc["mean"]),
        "test_std": float(test_desc["std"]),
        "ks_stat": float(ks_stat),
        "ks_p_value": float(p_val),
        "ks_reject": bool(p_val < ALPHA),
    }


def perform_pca_analysis(
    fps_train: np.ndarray,
    fps_test: np.ndarray,
    train_label: str,
    test_label: str,
) -> Optional[pd.DataFrame]:
    if fps_train is None or fps_test is None:
        return None
    if fps_train.ndim != 2 or fps_train.shape[1] != 1024:
        return None
    if fps_test.ndim != 2 or fps_test.shape[1] != 1024:
        return None

    all_fps = np.vstack([fps_train, fps_test])
    reduced = PCA(n_components=2).fit_transform(all_fps)

    df_pca = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_pca["Label"] = np.concatenate(
        [np.full(len(fps_train), train_label), np.full(len(fps_test), test_label)]
    )

    base_name = train_label.replace("_train.csv", "")
    df_pca.to_csv(PCA_RESULT_FOLDER / f"pca_reduced_data_{base_name}.csv", index=False)
    return df_pca


def plot_pca(df_pca: pd.DataFrame, train_label: str, test_label: str) -> None:
    base_name = train_label.replace("_train.csv", "")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Label",
        palette={train_label: "#007AFF", test_label: "#FFCC00"},
        s=110,
        alpha=0.7,
        linewidth=0.9,
    )

    plt.xlabel("PCA 1", fontsize=24)
    plt.ylabel("PCA 2", fontsize=24)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2, length=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    mapped_labels = ["train" if l.endswith("_train.csv") else "test" for l in labels]
    ax.legend(handles=handles, labels=mapped_labels, title=None, fontsize=20, markerscale=1)

    plt.tight_layout()
    plt.savefig(PCA_RESULT_FOLDER / f"pca_plot_{base_name}.png", dpi=600)
    plt.close()


def analyze_structure_pair(train_path: Path, test_path: Path) -> None:
    fps_train = load_fingerprints_from_csv(train_path)
    fps_test = load_fingerprints_from_csv(test_path)
    df_pca = perform_pca_analysis(fps_train, fps_test, train_path.name, test_path.name)
    if df_pca is not None:
        plot_pca(df_pca, train_path.name, test_path.name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    configure_plot_style()

    RT_RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
    PCA_RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not INPUT_SPLIT_DIR.exists():
        logging.error("Data folder does not exist: %s", INPUT_SPLIT_DIR)
        return

    train_files = sorted(INPUT_SPLIT_DIR.glob("*_train.csv"))
    if not train_files:
        logging.error("No training files found in %s", INPUT_SPLIT_DIR)
        return

    pairs: List[Tuple[Path, Path]] = []
    for train_path in train_files:
        test_path = train_path.with_name(train_path.name.replace("_train.csv", "_test.csv"))
        if test_path.exists():
            pairs.append((train_path, test_path))

    if not pairs:
        logging.error("No valid train/test pairs found")
        return

    rt_summary_rows = []
    for train_path, test_path in pairs:
        try:
            rt_summary_rows.append(analyze_retention_time_pair(train_path, test_path))
        except Exception as exc:
            logging.error("Retention-time analysis failed for %s: %s", train_path.name, exc)

    if rt_summary_rows:
        pd.DataFrame(rt_summary_rows).to_csv(
            RT_RESULT_FOLDER / "train_test_rt_summary.csv",
            index=False,
            float_format="%.4f",
        )

    for train_path, test_path in pairs:
        try:
            analyze_structure_pair(train_path, test_path)
        except Exception as exc:
            logging.error("Structure analysis failed for %s: %s", train_path.name, exc)

    logging.info("All analyses completed")


if __name__ == "__main__":
    main()
