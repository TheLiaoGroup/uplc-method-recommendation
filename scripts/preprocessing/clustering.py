import os
from pathlib import Path
import matplotlib
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from tqdm import tqdm

RDLogger.DisableLog("rdApp.warning")
matplotlib.use("Agg")
sns.set(style="white")  # White background, remove grid lines

# Set file paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
input_data_dir = str(PROJECT_ROOT / "data" / "processed")  # Data folder
output_result_dir = str(PROJECT_ROOT / "results" / "preprocessing" / "clustering")
os.makedirs(output_result_dir, exist_ok=True)

# =========================================================
# Utility Functions
# =========================================================
def smiles_to_fp(smiles, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return None

def compute_fingerprints(df):
    fps = []
    valid_indices = []
    for i, smi in enumerate(df['SMILES']):
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    return np.array(fps), df.iloc[valid_indices].reset_index(drop=True)

def compute_cluster_metrics(X, labels):
    silhouette = silhouette_score(X, labels)
    centers = np.array([X[labels == i].mean(axis=0) for i in np.unique(labels)])
    dist_mat = pairwise_distances(centers)
    separation = dist_mat[np.triu_indices_from(dist_mat, 1)].mean()
    return silhouette, separation

def get_cluster_centroids(X, labels, fps, smiles):
    centroids = []
    for i in np.unique(labels):
        cluster_idx = np.where(labels == i)[0]
        cluster_vecs = fps[cluster_idx]
        center = cluster_vecs.mean(axis=0)
        dists = np.linalg.norm(cluster_vecs - center, axis=1)
        min_idx = cluster_idx[np.argmin(dists)]
        centroids.append({
            "cluster": i,
            "center_smile": smiles[min_idx],
            "center_index": min_idx
        })
    return pd.DataFrame(centroids)

# =========================================================
# Plotting Functions - All elements scaled up 2x
# =========================================================
def plot_clusters(X, labels, method_name, k, silhouette, separation, file_prefix, cluster_sizes):
    # Base size scaled up 2x
    fig, ax = plt.subplots(figsize=(8, 6))   # Originally (8,6)

    palette = sns.color_palette("pastel", len(np.unique(labels)))
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1],
        hue=labels, palette=palette,
        s=50,               # Originally 30
        legend=False, ax=ax
    )

    # Label font size scaled up 2x
    for i in np.unique(labels):
        ax.text(
            X[labels == i, 0].mean(),
            X[labels == i, 1].mean(),
            str(cluster_sizes[i]),
            fontsize=17,            # Originally 10
            ha='center', va='center',
            color='black', fontweight='bold'
        )

    ax.set_xlabel(f"{method_name} Dim 1", fontsize=17)   # Originally 13
    ax.set_ylabel(f"{method_name} Dim 2", fontsize=17)

    # Tick length & width scaled up 2x
    ax.tick_params(
        direction='out',
        length=8,       # Originally 6
        width=2,         # Originally 1.5
        color='black',
        bottom=True, left=True, top=False, right=False,
        labelsize=16     # Tick label font size enlarged
    )

    ax.grid(False)
    ax.locator_params(axis='both', nbins=5)

    # Title font size scaled up 2x
    plt.figtext(
        0.5, -0.04,
        f"{method_name} | k={k} | silhouette={silhouette:.2f} | separation={separation:.2f}",
        ha='center', fontsize=17  # Originally 13
    )

    plt.tight_layout()
    plt.savefig(
        f"{file_prefix}_{method_name}_k{k}.png",
        dpi=600, bbox_inches='tight'
    )
    plt.close()

# =========================================================
# Main Workflow
# =========================================================
# Check if data directory exists
if not os.path.exists(input_data_dir):
    print(f"Error: Data directory '{input_data_dir}' does not exist!")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Please create the './data' folder and place your CSV files there.")
    exit(1)

# 只处理指定的2个文件
target_files = ["AM-I-filtered.csv", "AM-II-filtered.csv"]

# 检查文件是否存在
existing_files = []
missing_files = []

for file_name in target_files:
    file_path = os.path.join(input_data_dir, file_name)
    if os.path.exists(file_path):
        existing_files.append(file_name)
    else:
        missing_files.append(file_name)

if missing_files:
    print(f"Warning: The following files are missing in '{input_data_dir}':")
    for missing in missing_files:
        print(f"  - {missing}")
    print(f"Only existing files will be processed.")

if not existing_files:
    print(f"Error: None of the target files found in '{input_data_dir}'!")
    print(f"Files in directory: {os.listdir(input_data_dir)}")
    exit(1)

print(f"Found {len(existing_files)} out of {len(target_files)} target CSV file(s) in {input_data_dir}:")
for csv_file in existing_files:
    print(f"  - {csv_file}")

for file_name in existing_files:
    file_path = os.path.join(input_data_dir, file_name)
    print(f"\nProcessing file: {file_name}")
    
    try:
        df_full = pd.read_csv(file_path)
        print(f"  Loaded {len(df_full)} rows")
        
        if "SMILES" not in df_full.columns:
            print(f"  Warning: 'SMILES' column not found in {file_name}")
            print(f"  Available columns: {list(df_full.columns)}")
            continue
        
        smiles_list = df_full["SMILES"].dropna().unique()
        df = pd.DataFrame({"SMILES": smiles_list})
        print(f"  Unique SMILES: {len(smiles_list)}")

        fps, df_valid = compute_fingerprints(df)
        print(f"  Valid fingerprints computed: {len(fps)}")

        # PCA & UMAP
        pca_proj = PCA(n_components=2).fit_transform(fps)
        umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(fps)

        results = []

        folder_name = file_name.replace('.csv', '').replace('dedup_f-', '').replace('_cleaned', '')
        output_subdir = os.path.join(output_result_dir, folder_name)
        os.makedirs(output_subdir, exist_ok=True)
        print(f"  Output directory: {output_subdir}")

        for k in tqdm([2, 3, 4, 5, 6], desc=f"  Clustering k=[2,3,4,5,6]"):
            # PCA clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels_pca = kmeans.fit_predict(pca_proj)
            sil_pca, sep_pca = compute_cluster_metrics(pca_proj, labels_pca)

            # UMAP clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels_umap = kmeans.fit_predict(umap_proj)
            sil_umap, sep_umap = compute_cluster_metrics(umap_proj, labels_umap)

            results.append({
                "k": k,
                "pca_silhouette": sil_pca,
                "pca_separation": sep_pca,
                "umap_silhouette": sil_umap,
                "umap_separation": sep_umap
            })

            prefix = os.path.join(output_subdir, file_name.replace('.csv', ''))

            # Map back to full data
            pca_cluster_map = dict(zip(df_valid['SMILES'], labels_pca))
            umap_cluster_map = dict(zip(df_valid['SMILES'], labels_umap))
            df_full['PCA_Cluster'] = df_full['SMILES'].map(pca_cluster_map).fillna(-1).astype(int)
            df_full['UMAP_Cluster'] = df_full['SMILES'].map(umap_cluster_map).fillna(-1).astype(int)
            df_full.to_csv(f"{prefix}_with_labels_k{k}.csv", index=False)

            # Centroid molecules
            df_centroid_pca = get_cluster_centroids(pca_proj, labels_pca, fps, df_valid["SMILES"].values)
            df_centroid_umap = get_cluster_centroids(umap_proj, labels_umap, fps, df_valid["SMILES"].values)
            df_centroid_pca.to_csv(f"{prefix}_PCA_k{k}_centroids.csv", index=False)
            df_centroid_umap.to_csv(f"{prefix}_UMAP_k{k}_centroids.csv", index=False)

            cluster_sizes_pca = {i: np.sum(labels_pca == i) for i in np.unique(labels_pca)}
            cluster_sizes_umap = {i: np.sum(labels_umap == i) for i in np.unique(labels_umap)}

            # Plotting
            plot_clusters(pca_proj, labels_pca, "PCA", k, sil_pca, sep_pca, prefix, cluster_sizes_pca)
            plot_clusters(umap_proj, labels_umap, "UMAP", k, sil_umap, sep_umap, prefix, cluster_sizes_umap)

        # Save evaluation results
        df_result = pd.DataFrame(results)
        df_result.to_csv(
            os.path.join(output_subdir, f"{file_name.replace('.csv','')}_cluster_evaluation.csv"),
            index=False
        )
        
        print(f"  Completed processing {file_name}")
        
    except Exception as e:
        print(f"  Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*60)
print("Processing completed!")
print(f"Results saved to: {output_result_dir}")
print("="*60)

# List output directory structure
print("\nOutput directory structure:")
for root, dirs, files in os.walk(output_result_dir):
    level = root.replace(output_result_dir, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for file in files[:10]:  # Show first 10 files per directory
        print(f"{subindent}{file}")
    if len(files) > 10:
        print(f"{subindent}... and {len(files)-10} more files")