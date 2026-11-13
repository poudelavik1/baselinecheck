#!/bin/bash
# Evaluation + figures + tables after training.
#
#   sbatch slurm/evaluate.sbatch                                         # validation only
#   sbatch --dependency=afterok:<array_job_id> slurm/evaluate.sbatch     # chained after the array
#   sbatch --export=ALL,RUN_TEST=1 slurm/evaluate.sbatch                 # ONCE, at the very end: also the test set
#
# Model selection uses validation only. The test set is touched only with RUN_TEST=1, and
# evaluate.py refuses to recompute existing test results unless --overwrite is given.
#SBATCH --job-name=crackbench_eval
#SBATCH --partition=ASL-gpu          # <PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
source slurm/common.sh
COMMON=(--data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT")
STATUS=0

python -u evaluate.py --split val "${COMMON[@]}" || STATUS=1
python -u evaluate.py --split val --protocol original --no-efficiency "${COMMON[@]}" || STATUS=1
if [ "${RUN_TEST:-0}" = "1" ]; then
    python -u evaluate.py --split test --no-efficiency "${COMMON[@]}" || STATUS=1
    python -u evaluate.py --split test --protocol original --no-efficiency "${COMMON[@]}" || STATUS=1
fi
# PREVIOUS_RUNS (slurm/env.sh) adds the "previous run, not same protocol" reference rows.
python -u make_report.py --qualitative "${COMMON[@]}" || STATUS=1

echo "===== finished $(date) | exit $STATUS ====="
echo "Results: $RUNS_ROOT/figures  $RUNS_ROOT/tables"
exit $STATUS
