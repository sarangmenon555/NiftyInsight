set -euo pipefail
cd "$(dirname "$0")/.."
export ALGO_DATA_DIR="${ALGO_DATA_DIR:-$PWD/data_synthetic}"
export ALGO_REPORT_DIR="${ALGO_REPORT_DIR:-$PWD/reports_synthetic}"
export ALGO_START_DATE="${ALGO_START_DATE:-2017-01-01}"
export ALGO_TRAIN_END="${ALGO_TRAIN_END:-2021-12-31}"
export ALGO_VALID_END="${ALGO_VALID_END:-2023-06-30}"
python -m src.pipeline all --source synthetic --fast
