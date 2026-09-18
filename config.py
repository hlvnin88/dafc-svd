"""Hyperparameter, đường dẫn, ngưỡng lọc. Không chứa kết quả."""
import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / 'data'
LOGS_DIR      = BASE_DIR / 'logs'
RESULTS_DIR   = BASE_DIR / 'results'
SELECTION_DIR = BASE_DIR / 'selection'

TABLES_DIR    = RESULTS_DIR / 'tables'
FIGURES_DIR   = RESULTS_DIR / 'figures'
MODELS_DIR    = RESULTS_DIR / 'models'
PRED_LOGS_DIR = RESULTS_DIR / 'prediction_logs'
ERR_VEC_DIR   = RESULTS_DIR / 'error_vectors'
COST_LOGS_DIR = RESULTS_DIR / 'cost_logs'
EXTRACT_LOGS  = LOGS_DIR / 'extraction'
RESOURCE_LOGS = LOGS_DIR / 'resource'

for _d in [DATA_DIR, LOGS_DIR, RESULTS_DIR, SELECTION_DIR,
           TABLES_DIR, FIGURES_DIR, MODELS_DIR, PRED_LOGS_DIR,
           ERR_VEC_DIR, COST_LOGS_DIR, EXTRACT_LOGS, RESOURCE_LOGS]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------- Schema ----------
SCHEMA_COLS = ['sample_id', 'source_code', 'label', 'dataset_name']
DATASET_DEVIGN = 'devign'
DATASET_JULIET = 'juliet'
DATASET_BIGVUL = 'bigvul'
DATASETS = [DATASET_DEVIGN, DATASET_JULIET, DATASET_BIGVUL]

# ---------- Split ----------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SPLIT_SEED  = 42
BIGVUL_TEST_RATIO = 0.15

# ---------- NGUỒN DỮ LIỆU ----------
DEVIGN_GDRIVE_ID = "1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF"

# Juliet: GitHub mirror của NIST SARD #112
JULIET_GITHUB_MIRRORS = [
    "https://github.com/arichardson/juliet-test-suite-c.git",
    "https://gitlab.com/samate/juliet-test-suite-1.3.git",
]
JULIET_NIST_ZIP_URL = (
    "https://samate.nist.gov/SARD/downloads/test-suites/"
    "2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip")

# Big-Vul: Google Drive — file code CSV của ZeoVan (MSR 2020)
BIGVUL_GDRIVE_ID = "1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X"

# Số raw mục tiêu — dùng để validate cache
TARGET_RAW_SIZES = {
    'devign': 27318,
    'juliet': 64099,
    'bigvul': 188636,
}

# Số clean mục tiêu — dùng để cắt deterministic
TARGET_CLEAN_SIZES = {
    'devign': 24754,
    'juliet': 58420,
    'bigvul': 151820,
}

# ---------- Features ----------
FEATURE_GROUPS = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']
FUSION_DIM = 256

# F1
F1_TFIDF_MAX_FEATURES = 5000       # Bảng 3 Update.docx
F1_NGRAM_RANGE        = (1, 2)

# F2 (BiLSTM)
BILSTM_EMBED_DIM  = 128
BILSTM_HIDDEN     = 128          # 2*hidden = 256
BILSTM_VOCAB_MAX  = 30000
BILSTM_MAX_LEN    = 400
BILSTM_LR         = 1e-3
BILSTM_EPOCHS     = 5

# F8/F9
CODEBERT_NAME      = 'microsoft/codebert-base'
GRAPHCODEBERT_NAME = 'microsoft/graphcodebert-base'
ENCODER_DIM        = 768           # Bảng 3 Update.docx
ENCODER_MAX_LEN    = 512

# F10: 32 static metrics
F10_METRIC_NAMES = [
    'num_lines','num_functions','num_params','cyclomatic_complexity',
    'nesting_depth','num_branches','num_loops','num_returns',
    'num_dangerous_apis','num_strcpy','num_strcat','num_sprintf',
    'num_gets','num_malloc','num_free','num_memcpy',
    'has_pointer','num_pointer_derefs','has_array','num_array_index',
    'has_ampersand','num_bit_ops','has_union','has_void_ptr',
    'has_bounds_check','has_null_check','has_size_check',
    'num_arithmetic_ops','num_assignments','num_comparisons',
    'num_casts','num_goto',
]

# GCN
GCN_HIDDEN = 128

# ---------- Filter ----------
THETA_COV  = 0.80
THETA_OVL  = 0.60
THETA_COST = 0.75
TOP_K = 7
MIN_COMBO_SIZE = 2
MAX_COMBO_SIZE = 4

ALPHA = 0.40
BETA  = 0.30
GAMMA = 0.20
ETA   = 0.10

COST_COMPONENT_WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)

# ---------- Individual classifier ----------
IND_HIDDEN       = 128
IND_DROPOUT      = 0.3
IND_LR           = 1e-4
IND_WEIGHT_DECAY = 1e-5
IND_BATCH_SIZE   = 32
IND_EPOCHS       = 30
IND_EARLY_STOP   = 5
IND_THRESHOLD    = 0.5

# ---------- MMAF ----------
MMAF_FUSION_DIM   = 256
MMAF_ATTN_HIDDEN  = 128
MMAF_CLS_HIDDEN   = 128
MMAF_DROPOUT      = 0.3
MMAF_LR           = 1e-4
MMAF_WEIGHT_DECAY = 1e-5
MMAF_BATCH_SIZE   = 32
MMAF_EPOCHS       = 30
MMAF_EARLY_STOP   = 5

# ---------- Seeds ----------
RANDOM_SEEDS = [42, 52, 62, 72, 82]
RANDOM_SEED  = 42

# ---------- Joern ----------
JOERN_PATH    = os.environ.get('JOERN_PATH', 'joern')
JOERN_TIMEOUT = 900
JOERN_SINKS   = ['strcpy','sprintf','gets','system','malloc','memcpy','strcat',
                 'scanf','sscanf','strncpy','strncat','vsprintf','popen','exec']

# ---------- Sensitivity ----------
THRESHOLD_CONFIGS = {
    'Conservative': {'cov': 0.83, 'ovl': 0.57, 'cost': 0.72},
    'Default':      {'cov': 0.80, 'ovl': 0.60, 'cost': 0.75},
    'Relaxed':      {'cov': 0.77, 'ovl': 0.63, 'cost': 0.78},
}
PRESCORE_CONFIGS = {
    'Default':         {'alpha': 0.40, 'beta': 0.30, 'gamma': 0.20, 'eta': 0.10},
    'BasePerf':        {'alpha': 0.45, 'beta': 0.25, 'gamma': 0.20, 'eta': 0.10},
    'Complementarity': {'alpha': 0.35, 'beta': 0.35, 'gamma': 0.20, 'eta': 0.10},
    'Overlap':         {'alpha': 0.35, 'beta': 0.30, 'gamma': 0.25, 'eta': 0.10},
    'Cost':            {'alpha': 0.35, 'beta': 0.30, 'gamma': 0.20, 'eta': 0.15},
}

RANDOM_SELECTION_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_config():
    print("="*70)
    print(" DAFC-SVD CONFIG ".center(70, '='))
    print("="*70)
    print(f"  DEVICE       : {DEVICE}")
    print(f"  THETA_COV    : {THETA_COV}")
    print(f"  THETA_OVL    : {THETA_OVL}")
    print(f"  THETA_COST   : {THETA_COST}")
    print(f"  TOP_K        : {TOP_K}")
    print(f"  (α,β,γ,η)   : ({ALPHA},{BETA},{GAMMA},{ETA})")
    print(f"  SEEDS        : {RANDOM_SEEDS}")
    print("="*70)