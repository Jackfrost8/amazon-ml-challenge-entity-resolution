import os


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current file:
# amazon-ml-challenge-entity-resolution/
#   code/
#     business_entity_resolution/
#       src/
#         config.py
#
# Repository root is 3 levels above src/
#
# src -> business_entity_resolution -> code -> repository root

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

# Dataset is directly inside repository root
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# BLOCKING / CANDIDATE GENERATION
# ============================================================

# Character-level TF-IDF / hashing
TFIDF_NGRAM_RANGE = (3, 3)

# Keep vocabulary/hash dimensions manageable for 24 GB RAM
TFIDF_MAX_FEATURES = 30000

# Number of candidates generated per Source 1 entity
KNN_NEIGHBORS = 15


# ============================================================
# SEMANTIC EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# XGBOOST MODEL
# ============================================================

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "learning_rate": 0.05,
    "max_depth": 6,
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1,
    "n_estimators": 800,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}


# ============================================================
# LIGHTGBM MODEL
# ============================================================

# IMPORTANT:
# Your installed LightGBM build does NOT have GPU support.
# Therefore use CPU here.

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 800,
    "n_jobs": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "device_type": "cpu"
}


# ============================================================
# MATCHING THRESHOLD
# ============================================================

MATCH_THRESHOLD = 0.88