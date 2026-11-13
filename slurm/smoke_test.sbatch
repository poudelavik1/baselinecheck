#!/bin/bash
# Short end-to-end check on the cluster: 1 epoch on ~20 real tiles for every model, with
# the pretrained weights read from the offline cache, through figures and tables.
#   sbatch slurm/smoke_test.sbatch
# Results go to $RUNS_ROOT/_smoke (deleted and rewritten each time), never to the real runs.
#SBATCH --job-name=crackbench_smoke
#SBATCH --partition=ASL-gpu          # <PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
source slurm/common.sh
python -u scripts/smoke_test.py --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT/_smoke" --pretrained --seeds 0 1 ${SMOKE_ARGS:-}
STATUS=$?
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
