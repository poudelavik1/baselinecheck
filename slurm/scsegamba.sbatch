#!/bin/bash
# SCSegamba, in its own environment, under the same benchmark protocol (train.py).
#   bash scripts/install_scsegamba.sh                       # once, login node
#   srun ... python scripts/test_scsegamba.py               # must PASS first (see that file)
#   sbatch slurm/scsegamba.sbatch                           # seeds 0 1 2 as array tasks
# The benchmark's evaluate.sbatch cannot build this model (wrong env), so each task also
# evaluates its own run on validation here. For the test set, at the very end:
#   sbatch --export=ALL,RUN_TEST=1 slurm/scsegamba.sbatch
#SBATCH --job-name=crackbench_scsegamba
#SBATCH --partition=ASL-gpu          # <PARTITION>
#SBATCH --array=0-2%3                # task id = seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

set -uo pipefail
source slurm/env.sh
source slurm/common.sh "$SCSEGAMBA_ENV_DIR"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
COMMON=(--data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT")

python -u train.py --config configs/scsegamba.yaml --seed "$SEED" "${COMMON[@]}" ${TRAIN_ARGS:-} || exit 1
python -u evaluate.py --split val --models scsegamba --seeds "$SEED" "${COMMON[@]}" || exit 1
if [ "${RUN_TEST:-0}" = "1" ]; then
    python -u evaluate.py --split test --models scsegamba --seeds "$SEED" --no-efficiency "${COMMON[@]}" || exit 1
fi
echo "===== finished $(date) ====="
