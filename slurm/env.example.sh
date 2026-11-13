# Site settings. Copy to slurm/env.sh (ignored by git) and edit:
#     cp slurm/env.example.sh slurm/env.sh && nano slurm/env.sh
# Every sbatch script and slurm/submit.sh reads slurm/env.sh.

export ENV_NAME="<ENV_NAME>"                          # conda env of the benchmark
export ENV_DIR="$HOME/.conda/envs/$ENV_NAME"          # check with: conda env list
export DATA_ROOT="$HOME/swinunet/split_80_10_10"      # dataset (never copied into the repo)
export RUNS_ROOT="$HOME/crack_benchmark_runs"         # checkpoints, CSVs, figures, tables
export WEIGHTS_CACHE="$HOME/weights_cache"            # filled by scripts/download_weights.py

export PARTITION="ASL-gpu"                            # <PARTITION>: check with sinfo
export MAX_PARALLEL=4                                 # array throttle (%K): tasks running at once

# Optional: earlier results shown as "previous run, not same protocol" (':'-separated folders)
export PREVIOUS_RUNS="$HOME/swinunet"

# Separate environments of the Tier 2 models (created by the install scripts)
export SCSEGAMBA_ENV_DIR="$HOME/.conda/envs/scsegamba"
export NNUNET_ENV_DIR="$HOME/.conda/envs/nnunet"
