#!/bin/bash
# Build the job list and submit the training array. Run on the login node, from the repo root.
#
#   bash slurm/submit.sh                               # every model of the main env x seeds 0 1 2
#   bash slurm/submit.sh --tiers core tier1            # a subset: by tier ...
#   bash slurm/submit.sh --families CNN                # ... by family ...
#   bash slurm/submit.sh --models unet_r34 deepcrack --seeds 0     # ... or by name / seed
#   bash slurm/submit.sh --missing                     # only runs without FINISHED (failed, timed out, new)
#   bash slurm/submit.sh --protocol original           # the original-recipe runs (separate table)
#   EVAL=1 bash slurm/submit.sh                        # also queue evaluation + report after the array
#   DRY_RUN=1 bash slurm/submit.sh --tiers tier1       # print the list, submit nothing
#
# Throttle: MAX_PARALLEL in slurm/env.sh (tasks running at once, the %K of --array).
# Extra sbatch options: SBATCH_ARGS="--time=24:00:00" bash slurm/submit.sh
# Extra train.py options: TRAIN_ARGS="--set train.batch_size=8 train.accum_steps=2" bash slurm/submit.sh --models unet_r50

set -euo pipefail
cd "$(dirname "$0")/.."
[ -f slurm/env.sh ] || { echo "Run: cp slurm/env.example.sh slurm/env.sh  and edit it." >&2; exit 1; }
source slurm/env.sh
export PATH="$ENV_DIR/bin:$PATH"
mkdir -p logs "$RUNS_ROOT/joblists"

JOBLIST="$RUNS_ROOT/joblists/joblist_$(date +%Y%m%d_%H%M%S).txt"
COUNT=$(python scripts/make_joblist.py --out "$JOBLIST" --runs-root "$RUNS_ROOT" "$@" | tail -n 1)
if [ "$COUNT" -eq 0 ]; then
    echo "Nothing to submit (all selected runs are finished, or the filter matched no model)."
    exit 0
fi
echo "Job list: $JOBLIST ($COUNT tasks, at most ${MAX_PARALLEL:-4} at once)"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

export JOBLIST TRAIN_ARGS="${TRAIN_ARGS:-}"     # reach the job through --export=ALL
JOB_ID=$(sbatch --parsable --partition="$PARTITION" --array="0-$((COUNT - 1))%${MAX_PARALLEL:-4}" \
    --export=ALL ${SBATCH_ARGS:-} slurm/train_array.sbatch)
JOB_ID="${JOB_ID%%;*}"
echo "Submitted array job $JOB_ID"
echo "  monitor : squeue -u $USER      |  sacct -j $JOB_ID --format=JobID%20,JobName,State,Elapsed,ExitCode"
echo "  logs    : tail -f logs/crackbench_${JOB_ID}_0.out"

if [ "${EVAL:-0}" = "1" ]; then
    # afterany: evaluate whatever finished even if one task failed (afterok would cancel the evaluation).
    EVAL_ID=$(sbatch --parsable --partition="$PARTITION" --dependency="afterany:$JOB_ID" slurm/evaluate.sbatch)
    echo "Queued evaluation job ${EVAL_ID%%;*} (runs after the array ends)"
else
    echo "  evaluate afterwards:"
    echo "    sbatch --partition=$PARTITION --dependency=afterok:$JOB_ID slurm/evaluate.sbatch     # only if every task succeeded"
    echo "    sbatch --partition=$PARTITION --dependency=afterany:$JOB_ID slurm/evaluate.sbatch    # whatever finished"
fi
