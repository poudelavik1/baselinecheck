#!/bin/bash
# Create the nnU-Net v2 environment. Run on the LOGIN node (needs internet):
#     bash scripts/install_nnunet.sh
# It gets its own env so that its pinned dependencies cannot disturb the benchmark env.
set -euo pipefail
ENV="${NNUNET_ENV_NAME:-nnunet}"
command -v conda >/dev/null || { echo "conda not found on this node"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE "^$ENV\s" || conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
# Take the same CUDA build of PyTorch as the benchmark env if you know it, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/install_nnunet.sh
pip install torch torchvision ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
pip install nnunetv2 opencv-python-headless
python -c "import nnunetv2, torch; print('nnunetv2 ok | torch', torch.__version__, '| cuda build', torch.version.cuda)"
echo "Environment: $(conda info --base)/envs/$ENV  -> set NNUNET_ENV_DIR in slurm/env.sh if it differs"
