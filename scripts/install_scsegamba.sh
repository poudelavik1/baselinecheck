#!/bin/bash
# Separate environment for SCSegamba (Mamba needs compiled CUDA kernels + mmcv-full).
# Run on the LOGIN node (needs internet), from the repo root:
#     bash scripts/install_scsegamba.sh
# then test on a GPU node BEFORE submitting anything:
#     srun -p ASL-gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:20:00 \
#          $HOME/.conda/envs/scsegamba/bin/python scripts/test_scsegamba.py
#
# The versions are the ones of the official README (torch 1.13.1 + cu116, mamba-ssm 1.2.0).
# mamba-ssm 1.2.0 ships prebuilt wheels for that combination; any other torch/CUDA pair
# compiles from source and needs gcc >= 7 and a CUDA toolkit with nvcc (see check_env.sh).
# GPUs newer than the RTX 30 series (RTX 40xx, Ada, L40, H100) need cu117 at least:
#     TORCH_SPEC="torch==1.13.1+cu117 torchvision==0.14.1+cu117" bash scripts/install_scsegamba.sh
# If the build fails, skip SCSegamba: nothing else depends on it.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV="${SCSEGAMBA_ENV_NAME:-scsegamba}"
TORCH_SPEC="${TORCH_SPEC:-torch==1.13.1+cu116 torchvision==0.14.1+cu116}"
command -v conda >/dev/null || { echo "conda not found on this node"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE "^$ENV\s" || conda create -y -n "$ENV" python=3.10
conda activate "$ENV"

pip install $TORCH_SPEC -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv-full
pip install mamba-ssm==1.2.0
pip install timm lmdb mmengine "numpy<2"
# What the benchmark code itself needs in this env (train.py, data.py, evaluate.py):
pip install "albumentations>=1.4,<2.1" opencv-python-headless pyyaml scipy scikit-image pandas matplotlib thop

bash scripts/install_third_party.sh scsegamba
python -c "import torch, mamba_ssm, mmcv; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| mmcv', mmcv.__version__, '| mamba-ssm ok')"
echo "Environment: $(conda info --base)/envs/$ENV  -> set SCSEGAMBA_ENV_DIR in slurm/env.sh if it differs"
echo "Next: run scripts/test_scsegamba.py in an srun session (command at the top of this file)."
