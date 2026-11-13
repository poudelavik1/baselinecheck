#!/bin/bash
# nnU-Net v2 (default 2D configuration) on the SAME split, scored by evaluate.py.
#   bash scripts/install_nnunet.sh          # once, on the login node
#   sbatch slurm/nnunet.sbatch              # resubmit to continue after a time limit
#   sbatch --export=ALL,RUN_TEST=1 slurm/nnunet.sbatch     # at the very end: also the test set
#   sbatch --export=ALL,NNUNET_TRAINER=nnUNetTrainer_250epochs slurm/nnunet.sbatch   # shorter schedule
#
# nnU-Net configures itself (patch size, batch size, network, 1000 epochs by default, its
# own augmentation and Dice+CE loss): that is the point of the "auto-configured" family.
# Only the split and the evaluation are forced to match the benchmark. It has no seed
# argument, so it is reported as a single run (seed 0).
#SBATCH --job-name=crackbench_nnunet
#SBATCH --partition=ASL-gpu          # <PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail
source slurm/env.sh
source slurm/common.sh "$NNUNET_ENV_DIR"

export nnUNet_raw="$RUNS_ROOT/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$RUNS_ROOT/nnunet/nnUNet_preprocessed"
export nnUNet_results="$RUNS_ROOT/nnunet/nnUNet_results"
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
ID=501; NAME=Dataset501_Cracks; TRAINER="${NNUNET_TRAINER:-nnUNetTrainer}"
PREDICTIONS="$RUNS_ROOT/nnunet/predictions"
set -e

if [ ! -f "$nnUNet_raw/$NAME/dataset.json" ]; then
    python -u scripts/nnunet/convert_to_nnunet.py --data-root "$DATA_ROOT"
fi
if [ ! -f "$nnUNet_preprocessed/$NAME/nnUNetPlans.json" ]; then
    nnUNetv2_plan_and_preprocess -d $ID -c 2d --verify_dataset_integrity -np "${SLURM_CPUS_PER_TASK:-8}"
fi
python -u scripts/nnunet/convert_to_nnunet.py --write-splits      # fold 0 = my train / val split

FOLD_DIR="$nnUNet_results/$NAME/${TRAINER}__nnUNetPlans__2d/fold_0"
if [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; then
    CONTINUE=""; [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && CONTINUE="--c"
    nnUNetv2_train $ID 2d 0 -tr "$TRAINER" $CONTINUE
fi

predict() {   # predict <images folder> <split>
    nnUNetv2_predict -i "$nnUNet_raw/$NAME/$1" -o "$PREDICTIONS/raw_$2" -d $ID -c 2d -f 0 -tr "$TRAINER" --save_probabilities
    python -u scripts/nnunet/export_predictions.py --input "$PREDICTIONS/raw_$2" --split "$2" --out "$PREDICTIONS"
}

# Score with the benchmark's own evaluate.py (the main env has the benchmark dependencies).
export PATH="$ENV_DIR/bin:$PATH"; export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
run_nnunet() { PATH="$NNUNET_ENV_DIR/bin:$PATH" LD_LIBRARY_PATH="$NNUNET_ENV_DIR/lib:${LD_LIBRARY_PATH:-}" "$@"; }

[ -d "$PREDICTIONS/val" ] || run_nnunet predict imagesVal val
python -u evaluate.py --split val --model nnunet --seed 0 --predictions-dir "$PREDICTIONS" \
    --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT"
if [ "${RUN_TEST:-0}" = "1" ]; then
    [ -d "$PREDICTIONS/test" ] || run_nnunet predict imagesTs test
    python -u evaluate.py --split test --model nnunet --seed 0 --predictions-dir "$PREDICTIONS" \
        --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT"
fi
echo "===== finished $(date) ====="
