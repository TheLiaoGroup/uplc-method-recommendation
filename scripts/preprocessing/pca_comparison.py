"""
PCA Comparison Analysis Script - Separate Figures Version
Usage: python pca-comparison-separate.py
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

def load_fingerprints_from_csv(file_path):
    """Load fingerprint data from CSV file"""
    try:
        fp_columns = [f'fp_{i}' for i in range(1024)]
        df = pd.read_csv(file_path)
        return df[fp_columns].values
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def perform_pca(fps1, fps2, label1, label2, n_components=2):
    """Perform PCA on two datasets and return DataFrame"""
    all_fps = np.vstack([fps1, fps2])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(all_fps)

    df = pd.DataFrame(reduced, columns=[f'PCA{i+1}' for i in range(n_components)])
    df['Label'] = np.concatenate([np.full(len(fps1), label1),
                                  np.full(len(fps2), label2)])
    return df

def generate_filename_from_labels(label1, label2):
    """Generate filename from labels (e.g., AMIvsAMII)"""
    # Remove file extensions and clean up labels
    clean_label1 = label1.replace('.csv', '').replace('-filtered', '').replace('-', '')
    clean_label2 = label2.replace('.csv', '').replace('-filtered', '').replace('-', '')
    return f"{clean_label1}vs{clean_label2}"

def plot_individual_pca(df, label1, label2, output_path, colors):
    """Create and save individual PCA plot"""
    plt.figure(figsize=(12, 10))
    
    # Plot scatter plot
    ax = plt.gca()
    sns.scatterplot(data=df,
                    x='PCA1', y='PCA2',
                    hue='Label',
                    palette=colors,
                    s=120,
                    alpha=0.7,
                    ax=ax)
    
    # Set labels
    ax.set_xlabel("PCA 1", fontsize=20)
    ax.set_ylabel("PCA 2", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    
    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    legend_labels = []
    for l in labels:
        l_clean = l.replace('.csv', '').replace('-filtered', '')
        legend_labels.append(l_clean)
    
    ax.legend(handles=handles,
              labels=legend_labels,
              title=None,
              fontsize=18,
              loc='best')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to: {output_path}")

def main():
    """Main function"""
    print("Starting PCA comparison analysis (separate figures)...")

    project_root = Path(__file__).resolve().parents[2]
    
    # Set paths
    input_folder = str(project_root / "data" / "processed")
    
    # Reference file
    reference_file = "AM-I-filtered.csv"
    reference_path = os.path.join(input_folder, reference_file)
    
    # Target files list
    target_files = [
        "AM-II-filtered.csv",
        "AM-III-filtered.csv",
        "AM-IV-filtered.csv",
        "AM-V-filtered.csv",
        "AM-VI-filtered.csv"
    ]
    
    # Create output folder
    output_folder = str(project_root / "results" / "preprocessing" / "pca-comparison")
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if reference file exists
    if not os.path.exists(reference_path):
        print(f"Error: Reference file '{reference_file}' does not exist in folder '{input_folder}'")
        print(f"Looking in: {os.path.abspath(input_folder)}")
        sys.exit(1)
    
    # Load reference fingerprints
    print(f"Loading reference file: {reference_file}")
    reference_fps = load_fingerprints_from_csv(reference_path)
    if reference_fps is None:
        print("Error: Reference file loading failed, please check file format")
        sys.exit(1)
    
    # Check target files
    missing_files = []
    for file_name in target_files:
        file_path = os.path.join(input_folder, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print("Warning: The following files do not exist:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please ensure all files are in the correct folder")
    
    # Process each target file sequentially
    processed_count = 0
    
    for idx, file_name in enumerate(target_files):
        file_path = os.path.join(input_folder, file_name)
        
        if not os.path.exists(file_path):
            print(f"Skipping {file_name}: File not found")
            continue
        
        print(f"\nProcessing {file_name}...")
        
        # Load target fingerprints
        fps = load_fingerprints_from_csv(file_path)
        if fps is None:
            print(f"  Skipping {file_name}: Loading failed")
            continue
        
        # Perform PCA
        df = perform_pca(reference_fps, fps, reference_file, file_name)
        processed_count += 1
        
        # Define colors
        colors = {
            reference_file: "#007AFF",  # Blue for reference
            file_name: "#FFCC00"        # Yellow for target
        }
        
        # Generate filename
        filename = generate_filename_from_labels(reference_file, file_name)
        output_path = os.path.join(output_folder, f"{filename}.png")
        
        # Create and save individual plot
        plot_individual_pca(df, reference_file, file_name, output_path, colors)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Analysis completed!")
    print(f"Successfully processed {processed_count} comparisons")
    print(f"All plots saved to: {os.path.abspath(output_folder)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()