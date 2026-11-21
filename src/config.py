import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ALGO_DATA_DIR", PROJECT_ROOT / "data"))
REPORT_DIR = Path(os.environ.get("ALGO_REPORT_DIR", PROJECT_ROOT / "reports"))

BHAV_DIR = DATA_DIR / "raw" / "bhav"
INDEX_DIR = DATA_DIR / "raw" / "indices"
REF_DIR = DATA_DIR / "raw" / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

BASE_PATH = PROCESSED_DIR / "base.parquet"
MARKET_PATH = PROCESSED_DIR / "market.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"
SECTOR_PATH = REF_DIR / "sectors.csv"
SYNTHETIC_MARKER = DATA_DIR / ".synthetic"

BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"

START_DATE = os.environ.get("ALGO_START_DATE", "2015-01-01")
HORIZON = int(os.environ.get("ALGO_HORIZON", "5"))
TARGET_THRESHOLD = float(os.environ.get("ALGO_TARGET_THRESHOLD", "0.03"))
TARGET_MODE = os.environ.get("ALGO_TARGET_MODE", "absolute")
TARGET_EXCESS_THRESHOLD = float(os.environ.get("ALGO_TARGET_EXCESS_THRESHOLD", "0.02"))

UNIVERSE_SIZE = int(os.environ.get("ALGO_UNIVERSE_SIZE", "700"))
LIQUIDITY_WINDOW = 60
MIN_PRICE = 20.0
INCREMENTAL_BUFFER_DAYS = 420
INCREMENTAL_RECOMPUTE_DAYS = 15

TRAIN_END = os.environ.get("ALGO_TRAIN_END", "2022-12-31")
VALID_END = os.environ.get("ALGO_VALID_END", "2023-12-31")
PURGE_DAYS = 10

COST_PER_SIDE = float(os.environ.get("ALGO_COST_PER_SIDE", "0.0015"))
RISK_FREE_RATE = 0.065
INITIAL_CAPITAL = 1_000_000.0
TOP_K_PER_DAY = 5
STOP_LOSS = 0.08
RETRAIN_AFTER_DAYS = 30

SYNTHETIC_SYMBOLS = int(os.environ.get("ALGO_SYNTHETIC_SYMBOLS", "150"))

RANDOM_STATE = 42
PREDICTION_COLUMNS = ["pred_dt", "pred_rf", "pred_xgb", "pred_ens"]
PRIMARY_PREDICTION = os.environ.get("ALGO_PRIMARY_PREDICTION", "pred_ens")
TARGETS = ["target_cls", "target_reg"]


def ensure_dirs():
    for path in (BHAV_DIR, INDEX_DIR, REF_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
