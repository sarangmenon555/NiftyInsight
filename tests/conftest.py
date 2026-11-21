import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORKDIR = tempfile.mkdtemp(prefix="algo_tests_")
os.environ["ALGO_DATA_DIR"] = os.path.join(WORKDIR, "data")
os.environ["ALGO_REPORT_DIR"] = os.path.join(WORKDIR, "reports")
os.environ["ALGO_START_DATE"] = "2018-01-01"
os.environ["ALGO_TRAIN_END"] = "2021-12-31"
os.environ["ALGO_VALID_END"] = "2023-06-30"
os.environ["ALGO_SYNTHETIC_SYMBOLS"] = "50"
