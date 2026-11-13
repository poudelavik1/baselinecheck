# Sourced by every sbatch script: settings, environment, sanity checks.
# Usage inside a job:  source slurm/common.sh [env dir]

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR" || exit 1
if [ ! -f slurm/env.sh ]; then
    echo "slurm/env.sh not found. Run: cp slurm/env.example.sh slurm/env.sh  and edit it." >&2
    exit 1
fi
source slurm/env.sh

# The login node's conda is not on the compute nodes: put the environment on PATH instead
# of activating it (same approach as the earlier swinunet jobs).
ACTIVE_ENV_DIR="${1:-$ENV_DIR}"
if [ ! -x "$ACTIVE_ENV_DIR/bin/python" ]; then
    echo "Python environment not found: $ACTIVE_ENV_DIR (set ENV_DIR in slurm/env.sh)" >&2
    exit 1
fi
export PATH="$ACTIVE_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ACTIVE_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

export MPLBACKEND=Agg                 # no display on the server
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1              # the DataLoader workers do the parallel work
export TORCH_HOME="$WEIGHTS_CACHE/torch"
export HF_HOME="$WEIGHTS_CACHE/huggingface"
export OFFLINE=1                      # compute nodes: read pretrained weights from the cache only
export HF_HUB_OFFLINE=1
mkdir -p "$RUNS_ROOT" logs

echo "===== ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname) | $(date) ====="
echo "code   : $(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --format=%s 2>/dev/null)"
echo "python : $(command -v python)"
echo "data   : $DATA_ROOT"
echo "runs   : $RUNS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
[ -d "$DATA_ROOT" ] || { echo "DATA_ROOT not found: $DATA_ROOT" >&2; exit 1; }
