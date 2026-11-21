set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
fi
python -m src.pipeline run-daily --mode "${RUN_MODE:-download}" >> logs/run_daily.log 2>&1
