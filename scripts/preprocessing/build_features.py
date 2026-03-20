import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
from typing import Optional, List, Dict, Any
import chardet
import os
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')
RDLogger.DisableLog("rdApp.warning")

# ---------- Configuration Parameters ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMARTS_FILE = PROJECT_ROOT / 'data' / 'smarts' / 'priority_fgs_823_newnew.txt'

FEATURE_COLS = ['MolWt', 'logP', 'TPSA', 'H_bond_donors', 'H_bond_acceptors']
FP_COLS = [f'col{i}' for i in range(823)]
MG_COLS = [f'fp_{i}' for i in range(1024)]
ALL_FEATURES = FEATURE_COLS + FP_COLS + MG_COLS

# Data directory containing CSV files
INPUT_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'

# Column names to check for SMILES data (in order of priority)
SMILES_CANDIDATE_COLS = ['SMILES', 'Smiles', 'smiles', 'SMILE', 'Smile', 'smile']
TARGET_COL = 'UV_RT-s'  # Column for filtering

# Output directory
OUTPUT_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

# ---------- Helper Functions ----------
def ensure_output_dir() -> str:
    """Ensure the output directory exists"""
    if not OUTPUT_PROCESSED_DIR.exists():
        OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {OUTPUT_PROCESSED_DIR}")
    return str(OUTPUT_PROCESSED_DIR)

def detect_file_encoding(file_path: str) -> str:
    """Automatically detect file encoding"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            
            # Handle common encoding aliases
            encoding_map = {
                'GB2312': 'GB18030',
                'gb2312': 'GB18030',
                'GBK': 'GB18030',
                'gbk': 'GB18030',
                'ISO-8859-1': 'latin1',
                'iso-8859-1': 'latin1',
                'Windows-1252': 'cp1252',
                'windows-1252': 'cp1252'
            }
            
            if encoding in encoding_map:
                encoding = encoding_map[encoding]
            
            print(f"Detected file {file_path} encoding: {encoding} (confidence: {result.get('confidence', 0):.2f})")
            return encoding
    except Exception as e:
        print(f"Error detecting file encoding ({file_path}): {e}, using default encoding utf-8")
        return 'utf-8'

def read_csv_with_encoding(file_path: str, encodings_to_try: List[str] = None) -> Optional[pd.DataFrame]:
    """Read CSV file with multiple encoding attempts"""
    if encodings_to_try is None:
        encodings_to_try = ['utf-8', 'GB18030', 'latin1', 'cp1252', 'ISO-8859-1']
    
    detected_encoding = detect_file_encoding(file_path)
    if detected_encoding not in encodings_to_try:
        encodings_to_try.insert(0, detected_encoding)
    
    for encoding in encodings_to_try:
        try:
            print(f"Attempting to read {file_path} with {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='warn')
            print(f"Successfully read file with {encoding} encoding")
            return df
        except UnicodeDecodeError as e:
            print(f"Encoding {encoding} failed: {e}")
            continue
        except Exception as e:
            print(f"Error reading file (encoding: {encoding}): {e}")
            continue
    
    print(f"All encoding attempts failed: {encodings_to_try}")
    return None

def find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Find the appropriate SMILES column in the DataFrame"""
    for col_name in SMILES_CANDIDATE_COLS:
        if col_name in df.columns:
            print(f"Found SMILES column: {col_name}")
            return col_name
    
    # If no standard column found, check for any column containing 'smiles' (case-insensitive)
    for col in df.columns:
        if 'smiles' in col.lower():
            print(f"Found alternative SMILES column: {col}")
            return col
    
    print(f"No SMILES column found. Available columns: {list(df.columns)}")
    return None

# ---------- Automatically Read SMARTS ----------
print("Reading SMARTS file...")
try:
    with open(SMARTS_FILE, 'rb') as f:
        raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
        if enc.lower() in ['gb2312', 'gbk']:
            enc = 'GB18030'

    with open(SMARTS_FILE, encoding=enc, errors='ignore') as f:
        SMARTS_PATTERNS = [l.strip() for l in f if l.strip()]

    print(f"Successfully loaded {len(SMARTS_PATTERNS)} SMARTS patterns (using encoding: {enc})")
except Exception as e:
    print(f"Error reading SMARTS file: {e}")
    encodings_to_try = ['utf-8', 'GB18030', 'latin1']
    for enc in encodings_to_try:
        try:
            with open(SMARTS_FILE, encoding=enc, errors='ignore') as f:
                SMARTS_PATTERNS = [l.strip() for l in f if l.strip()]
            print(f"Successfully loaded {len(SMARTS_PATTERNS)} SMARTS patterns (using fallback encoding: {enc})")
            break
        except:
            continue
    else:
        print("Cannot read SMARTS file, exiting program")
        exit(1)

# ---------- Feature Calculation Functions ----------
def calc_features(smiles: str) -> Optional[np.ndarray]:
    """Calculate molecular features"""
    try:
        smiles_str = str(smiles).strip()
        if not smiles_str or smiles_str.lower() in ['nan', 'none', 'null', '']:
            print(f"  Warning: Empty SMILES value")
            return None
            
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            print(f"  Warning: Cannot parse SMILES: {smiles_str[:50]}...")
            return None
        
        # Basic descriptors
        try:
            base = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol)
            ]
        except Exception as e:
            print(f"  Warning: Failed to calculate basic descriptors (SMILES: {smiles_str[:50]}...): {e}")
            return None
        
        # SMARTS fingerprints (823 dimensions)
        fp_823 = [0] * 823
        patterns_to_check = SMARTS_PATTERNS[:823]
        
        for i, sma in enumerate(patterns_to_check):
            try:
                if not sma:
                    continue
                patt = Chem.MolFromSmarts(sma)
                if patt and mol.HasSubstructMatch(patt):
                    fp_823[i] = 1
            except Exception as e:
                continue
        
        # Morgan fingerprints (1024 dimensions)
        try:
            mg = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            mg_bits = list(mg)
        except Exception as e:
            print(f"  Warning: Failed to calculate Morgan fingerprints (SMILES: {smiles_str[:50]}...): {e}")
            return None
        
        return np.array(base + fp_823 + mg_bits, dtype=np.float32)
        
    except Exception as e:
        print(f"Error calculating features (SMILES: {str(smiles)[:50]}...): {e}")
        return None

def process_csv_file(file_path: str) -> bool:
    """Process a single CSV file"""
    try:
        print(f"\nProcessing file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return False
        
        df = read_csv_with_encoding(file_path)
        if df is None:
            print(f"Error: Cannot read file {file_path}")
            return False
        
        original_rows = len(df)
        print(f"Original number of rows: {original_rows}")
        
        # Find SMILES column
        smiles_col = find_smiles_column(df)
        if smiles_col is None:
            print(f"Error: No SMILES column found in {file_path}")
            return False
        
        # Check for target column
        if TARGET_COL not in df.columns:
            print(f"Warning: Column '{TARGET_COL}' not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            
            possible_cols = [col for col in df.columns if 'RT' in col or 'rt' in col or 'UV' in col or 'uv' in col]
            if possible_cols:
                print(f"Possible alternative columns: {possible_cols}")
            
            print(f"Skipping file {file_path}")
            return False
        
        # Convert target column to numeric if needed
        try:
            if not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
                print(f"Converting column '{TARGET_COL}' to numeric type...")
                df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
        except Exception as e:
            print(f"Error converting column '{TARGET_COL}': {e}")
        
        # Filter rows with retention time >= 30 seconds
        before_filter = len(df)
        df = df[df[TARGET_COL] >= 30].copy()
        filtered_rows = len(df)
        print(f"Rows after filtering: {filtered_rows} (removed {before_filter - filtered_rows} rows)")
        
        if filtered_rows == 0:
            print(f"Warning: No data left after filtering in {file_path}, skipping")
            return False
        
        print(f"Using SMILES column: {smiles_col}")
        print(f"Using filter column: {TARGET_COL} (threshold: 30 seconds)")
        
        # Calculate features
        print("Calculating molecular features...")
        features_list = []
        valid_indices = []
        error_count = 0
        

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_col]
            features = calc_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
            else:
                error_count += 1
                if error_count <= 5:
                    print(f"  Warning: Row {idx+2} (SMILES: {str(smiles)[:50]}...) failed feature calculation")
        
        if error_count > 5:
            print(f"  ... and {error_count - 5} more similar errors")
        
        if not features_list:
            print("Error: Could not calculate features for any molecules")
            return False
        
        print(f"Successfully calculated features for {len(features_list)} molecules, failed {error_count}")
        
        # Create feature DataFrame
        features_array = np.vstack(features_list)
        features_df = pd.DataFrame(
            features_array, 
            index=valid_indices,
            columns=ALL_FEATURES
        )
        
        # Filter valid rows
        df_valid = df.loc[valid_indices].copy()
        
        # Merge original data with features
        result_df = pd.concat([df_valid.reset_index(drop=True), 
                             features_df.reset_index(drop=True)], axis=1)
        
        # Ensure output directory exists
        output_dir = ensure_output_dir()
        
        # Generate output file path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_name = f"{base_name}-filtered.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        # Save results
        try:
            result_df.to_csv(output_file_path, index=False, encoding='utf-8')
            print(f"Successfully saved to: {output_file_path} (UTF-8 encoding)")
        except Exception as e:
            print(f"UTF-8 save failed, trying GB18030...")
            result_df.to_csv(output_file_path, index=False, encoding='GB18030')
            print(f"Successfully saved to: {output_file_path} (GB18030 encoding)")
        
        print(f"Final data shape: {result_df.shape}")
        print(f"Number of valid molecules: {len(features_list)}")
        print(f"Number of feature columns: {len(ALL_FEATURES)}")
        
        # Display preview
        print("\nPreview of first 3 rows:")
        try:
            preview_cols = [smiles_col, TARGET_COL, 'MolWt', 'logP']
            preview_cols = [col for col in preview_cols if col in result_df.columns]
            print(result_df[preview_cols].head(3))
        except Exception as e:
            print(f"Error displaying preview: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Starting batch processing of CSV files...")
    print("=" * 60)
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Check if data directory exists
    if not INPUT_RAW_DIR.exists():
        print(f"Error: Data directory '{INPUT_RAW_DIR}' does not exist")
        print("Creating data directory...")
        INPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created data directory: {INPUT_RAW_DIR}")
        print("Please place CSV files in this directory and run the script again.")
        return
    
    # Get all CSV files in data directory
    csv_files = [f.name for f in INPUT_RAW_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.csv']
    
    if not csv_files:
        print(f"No CSV files found in directory: {INPUT_RAW_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {INPUT_RAW_DIR}:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Statistics
    success_count = 0
    fail_count = 0
    processed_files = []
    output_files = []
    
    # Process each CSV file
    for file_name in csv_files:
        file_path = str(INPUT_RAW_DIR / file_name)
        print(file_path)
        
        if process_csv_file(file_path):
            success_count += 1
            processed_files.append(file_name)
            
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"{base_name}-filtered.csv"
            output_file_path = str(OUTPUT_PROCESSED_DIR / output_file_name)
            output_files.append(output_file_path)
        else:
            fail_count += 1
        
        print("-" * 60)
    
    # Output summary
    print("\n" + "=" * 60)
    print("Batch processing completed!")
    print("=" * 60)
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {fail_count} files")
    print(f"Output directory: {os.path.abspath(OUTPUT_PROCESSED_DIR)}")
    
    if success_count > 0:
        print("\nGenerated files:")
        for file, output_file in zip(processed_files, output_files):
            print(f"  {file} -> {output_file}")
        
        print("\nFeature column details:")
        print(f"1. Basic descriptors ({len(FEATURE_COLS)} columns):")
        print(f"   {', '.join(FEATURE_COLS)}")
        print(f"\n2. SMARTS fingerprints ({len(FP_COLS)} columns):")
        print(f"   {FP_COLS[0]} to {FP_COLS[-1]}")
        print(f"\n3. Morgan fingerprints ({len(MG_COLS)} columns):")
        print(f"   {MG_COLS[0]} to {MG_COLS[-1]}")
        print(f"\nTotal feature dimensions: {len(ALL_FEATURES)}")
        
        if output_files:
            first_output = output_files[0]
            if os.path.exists(first_output):
                try:
                    test_df = pd.read_csv(first_output, nrows=5)
                    print(f"\nFirst output file ({first_output}) column count: {len(test_df.columns)}")
                    print(f"First 8 column names: {list(test_df.columns[:8])}...")
                except Exception as e:
                    print(f"\nError reading output file: {e}")
    
    print("\nProgram execution completed!")

if __name__ == "__main__":
    main()