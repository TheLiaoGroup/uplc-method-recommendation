from pathlib import Path

# 当前文件: src/config/paths.py
# parents[0] = src/config
# parents[1] = src
# parents[2] = 项目根目录 uplc-method-recommendation
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "datas"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"



SMARTS_FILE = DATA_DIR / "SMARTS" / "priority_fgs_823_newnew.txt"

FEATURE_COLS = ['MolWt', 'logP', 'TPSA', 'H_bond_donors', 'H_bond_acceptors']
FP_COLS = [f'col{i}' for i in range(823)]
MG_COLS = [f'fp_{i}' for i in range(1024)]
ALL_FEATURES = FEATURE_COLS + FP_COLS + MG_COLS



FIXED_METHOD_ORDER = [
    'AM-I', 
    'AM-II', 
    'AM-III', 
    'AM-IV', 
    'AM-V', 
    'AM-VI'
]

# ---------- MODEL DIRECTORY MAPPING ----------
MODEL_DIR_MAP = {
    'AM-I': MODELS_DIR/ "ml" / "2-svr-models" / "AM-I-svr-model",
    'AM-II': MODELS_DIR / "ml" / "2-svr-models" / "AM-II-svr-model",
    'AM-III': MODELS_DIR / "ml" / "2-svr-models" / "AM-III-svr-model",
    'AM-IV': MODELS_DIR / "ml" / "2-svr-models" / "AM-IV-svr-model",
    'AM-V': MODELS_DIR / "ml" / "2-svr-models" / "AM-V-svr-model",
    'AM-VI': MODELS_DIR / "ml" / "2-svr-models" / "AM-VI-svr-model"
}

# ---------- METHOD EVALUATION RANGE CONFIGURATION ----------
METHOD_RANGE_CONFIG = {
    'AM-I': (30, 120),
    'AM-II': (30, 120),
    'AM-III': (30, 150),
    'AM-IV': (30, 120),
    'AM-V': (30, 150),
    'AM-VI': (30, 180)
}