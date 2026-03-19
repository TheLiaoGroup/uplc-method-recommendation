"""Reusable utilities extracted from ML notebooks.

This module centralizes plotting, evaluation and lightweight chemistry helpers
that were repeatedly defined across notebooks under src/ml.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chardet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import KFold, learning_curve

IPHONE_COLORS: Dict[str, str] = {
    "scatter": "#007AFF",
    "line": "#AEAEB2",
    "text": "#000000",
}


def set_all_seeds(seed: int = 42, deterministic_threads: bool = True) -> None:
    """Set global random seeds for reproducible notebook runs."""
    if deterministic_threads:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed: int = 42, num_threads: int = 20) -> None:
    """Compatibility helper used by LightGBM transfer-learning notebooks."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["LIGHTGBM_NUM_THREADS"] = str(num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))


def evaluate_regression(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute common regression metrics used across notebooks."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def iphone_style_ax(
    ax: plt.Axes,
    labelsize: int = 16,
    tick_length: float = 6,
    tick_width: float = 2,
    spine_width: float = 2,
) -> None:
    """Apply the iPhone-style axis aesthetics reused in multiple notebooks."""
    ax.tick_params(
        axis="both",
        direction="out",
        length=tick_length,
        width=tick_width,
        labelsize=labelsize,
    )
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(spine_width)
    ax.grid(False)


def plot_scatter(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    save_path: str,
    xlabel: str = "True RT (s)",
    ylabel: str = "Predicted RT (s)",
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    dpi: int = 600,
) -> None:
    """Create a standard true-vs-predicted scatter plot with R2/MAE text."""
    use_colors = colors or IPHONE_COLORS
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    iphone_style_ax(ax)
    ax.set_aspect("equal", adjustable="box")

    plt.scatter(
        y_true_arr,
        y_pred_arr,
        alpha=0.8,
        s=70,
        color=use_colors["scatter"],
        edgecolors="none",
    )

    lims = [
        float(min(y_true_arr.min(), y_pred_arr.min())),
        float(max(y_true_arr.max(), y_pred_arr.max())),
    ]
    plt.plot(lims, lims, linestyle="--", color=use_colors["line"], linewidth=3)

    metrics = evaluate_regression(y_true_arr, y_pred_arr)
    plt.text(
        0.05,
        0.95,
        f"R² = {metrics['R2']:.3f}\\nMAE = {metrics['MAE']:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=16,
        color=use_colors["text"],
    )

    plt.xlabel(xlabel, fontsize=18, fontweight="bold")
    plt.ylabel(ylabel, fontsize=18, fontweight="bold")
    if title:
        plt.title(title, fontsize=17, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    save_path: str,
    xlabel: str = "Predicted RT (s)",
    ylabel: str = "Residuals (Predicted - True)",
    colors: Optional[Dict[str, str]] = None,
    dpi: int = 600,
) -> None:
    """Create a residual plot with horizontal zero reference line."""
    use_colors = colors or IPHONE_COLORS
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    residuals = y_pred_arr - y_true_arr

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    iphone_style_ax(ax)
    ax.set_aspect("equal", adjustable="box")

    plt.scatter(
        y_pred_arr,
        residuals,
        alpha=0.8,
        s=70,
        color=use_colors["scatter"],
        edgecolors="none",
    )
    plt.axhline(y=0, linestyle="--", color=use_colors["line"], linewidth=3)

    metrics = evaluate_regression(y_true_arr, y_pred_arr)
    plt.text(
        0.05,
        0.95,
        f"R² = {metrics['R2']:.3f} \nMAE = {metrics['MAE']:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=16,
        color=use_colors["text"],
    )

    plt.xlabel(xlabel, fontsize=18, fontweight="bold")
    plt.ylabel(ylabel, fontsize=18, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_scatter_and_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    save_folder: str,
    base_name: str,
    scatter_name: Optional[str] = None,
    residual_name: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    dpi: int = 600,
) -> tuple[str, str]:
    """Create paired scatter/residual plots and return saved file paths."""
    os.makedirs(save_folder, exist_ok=True)
    scatter_file = scatter_name or f"{base_name}_scatter.png"
    residual_file = residual_name or f"{base_name}_residuals.png"
    scatter_path = os.path.join(save_folder, scatter_file)
    residual_path = os.path.join(save_folder, residual_file)

    plot_scatter(y_true, y_pred, scatter_path, colors=colors, dpi=dpi)
    plot_residuals(y_true, y_pred, residual_path, colors=colors, dpi=dpi)
    return scatter_path, residual_path


def plot_learning_curve(
    train_sizes: Sequence[float],
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    save_path: str,
    title: str = "Learning Curve",
    colors: Optional[Dict[str, str]] = None,
    dpi: int = 600,
) -> None:
    """Plot train/validation learning curves from sklearn.learning_curve outputs."""
    use_colors = colors or IPHONE_COLORS

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    iphone_style_ax(ax)

    plt.plot(
        train_sizes,
        np.mean(train_scores, axis=1),
        "o-",
        color=use_colors["scatter"],
        linewidth=3,
        label="Train R²",
    )
    plt.plot(
        train_sizes,
        np.mean(val_scores, axis=1),
        "o-",
        color=use_colors["line"],
        linewidth=3,
        label="Val R²",
    )

    plt.xlabel("Training examples", fontsize=18, fontweight="bold")
    plt.ylabel("R²", fontsize=18, fontweight="bold")
    plt.title(title, fontsize=17, fontweight="bold")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_learning_curve_from_estimator(
    estimator: Any,
    X: Any,
    y: Any,
    save_path: str,
    title: str = "Learning Curve",
    cv: Optional[Any] = None,
    scoring: str = "r2",
    n_jobs: int = 1,
    train_sizes: Optional[Sequence[float]] = None,
) -> None:
    """Compute and draw learning curve directly from estimator and data."""
    actual_train_sizes = train_sizes or np.linspace(0.1, 1.0, 5)
    actual_cv = cv or KFold(n_splits=5, shuffle=True, random_state=42)
    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=actual_cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=actual_train_sizes,
    )
    plot_learning_curve(
        sizes,
        train_scores,
        val_scores,
        save_path=save_path,
        title=title,
    )


def check_train_test_files(
    train_file: str,
    test_file: str,
    target_col: str,
    all_features: Optional[Sequence[str]] = None,
    preview_n_cols: int = 10,
) -> bool:
    """Validate train/test CSV paths and minimal schema consistency."""
    print(f"   Train file: {train_file}")
    print(f"   Test file: {test_file}")

    if not os.path.exists(train_file):
        print("   Train file not found")
        return False
    if not os.path.exists(test_file):
        print("   Test file not found")
        return False

    try:
        train_df_head = pd.read_csv(train_file, nrows=1)
        test_df_head = pd.read_csv(test_file, nrows=1)

        train_shape = pd.read_csv(train_file).shape
        test_shape = pd.read_csv(test_file).shape
        print(f"   Train shape: {train_shape}")
        print(f"   Test shape: {test_shape}")

        if target_col not in train_df_head.columns:
            print(f"   Target column '{target_col}' not found in train data")
            print(f"   Available columns: {list(train_df_head.columns[:preview_n_cols])}")
            return False

        if all_features:
            missing_features = [
                col
                for col in all_features[:preview_n_cols]
                if col not in train_df_head.columns
            ]
            if missing_features:
                print(f"   Some features missing (preview): {missing_features}")

        _ = test_df_head
        return True
    except Exception as exc:
        print(f"   Error reading files: {exc}")
        return False


def plot_feature_importance_barh(
    importance_df: pd.DataFrame,
    save_path: str,
    dataset_name: str,
    top_n: int = 20,
) -> None:
    """Plot top-N feature importances as a horizontal bar chart."""
    if importance_df.empty:
        return

    plot_df = importance_df.head(top_n) if len(importance_df) > top_n else importance_df

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    iphone_style_ax(ax, labelsize=10)

    colors = plt.cm.Blues(np.linspace(0.3, 1, len(plot_df)))
    ax.barh(range(len(plot_df)), plot_df["importance"], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"], fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Top {len(plot_df)} Feature Importance - {dataset_name}",
        fontsize=16,
        fontweight="bold",
    )
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Convert SMILES into Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def compute_similarity(fp1: Any, fp2: Any) -> float:
    """Compute Tanimoto similarity between two RDKit fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def tanimoto_similarity_from_smiles(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity directly from two SMILES strings."""
    fp1 = smiles_to_fp(smiles_a)
    fp2 = smiles_to_fp(smiles_b)
    return compute_similarity(fp1, fp2)


def safe_tag(tag: str) -> str:
    """Create filesystem-safe identifiers for experiment names."""
    return "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in tag)


def load_smarts_patterns(smarts_file: str) -> list[str]:
    """Load SMARTS patterns with automatic encoding detection."""
    with open(smarts_file, "rb") as fp:
        raw = fp.read()
    encoding = chardet.detect(raw).get("encoding") or "utf-8"
    with open(smarts_file, encoding=encoding, errors="ignore") as fp:
        return [line.strip() for line in fp if line.strip()]


def calc_features_from_smarts(
    smiles: str,
    smarts_patterns: Sequence[str],
    morgan_radius: int = 2,
    morgan_n_bits: int = 1024,
    smarts_vector_size: int = 823,
) -> Optional[np.ndarray]:
    """Calculate descriptor + SMARTS + Morgan features used in recommendation notebooks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    base = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
    ]

    fp_823 = [0] * smarts_vector_size
    for idx, sma in enumerate(smarts_patterns[:smarts_vector_size]):
        patt = Chem.MolFromSmarts(sma)
        if patt and mol.HasSubstructMatch(patt):
            fp_823[idx] = 1

    mg = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=morgan_radius,
        nBits=morgan_n_bits,
    )
    return np.array(base + fp_823 + list(mg), dtype=np.float32)
