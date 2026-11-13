#!/bin/bash
# Print everything needed to size the benchmark: Python, PyTorch, CUDA, GPU, library
# versions, Slurm partition limits, disk space and whether this node has internet.
#
# Run it twice and paste both outputs:
#   1) on the login node, inside your env:
#        bash scripts/check_env.sh 2>&1 | tee check_env_login.txt
#   2) on a GPU compute node (this is the one that shows the GPU):
#        srun -p <PARTITION> --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:10:00 \
#             bash scripts/check_env.sh 2>&1 | tee check_env_gpu.txt
#
# If conda cannot be activated on the compute node, point ENV_DIR at the env instead:
#        ENV_DIR=$HOME/.conda/envs/<ENV_NAME> bash scripts/check_env.sh
# The script never stops on an error: a missing tool is reported and skipped.

section() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

if [ -n "${ENV_DIR:-}" ]; then
    if [ -x "$ENV_DIR/bin/python" ]; then
        export PATH="$ENV_DIR/bin:$PATH"
        export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
    else
        echo "WARNING: ENV_DIR=$ENV_DIR has no bin/python; using whatever python is on PATH."
    fi
fi
PYTHON="${PYTHON:-python}"
have "$PYTHON" || PYTHON=python3

section "HOST"
echo "date        : $(date)"
echo "host        : $(hostname)"
echo "user        : $(whoami)"
echo "os          : $(grep -m1 PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"')"
echo "kernel      : $(uname -r)"
echo "glibc       : $(ldd --version 2>/dev/null | head -n1)"
echo "cpus (node) : $(nproc 2>/dev/null)"
echo "memory      : $(free -g 2>/dev/null | awk '/^Mem:/ {print $2 " GB total, " $7 " GB available"}')"
echo "slurm job   : ${SLURM_JOB_ID:-none (not inside a job)}"
echo "slurm cpus  : ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

section "PYTHON ENV"
echo "python      : $(command -v "$PYTHON" || echo 'NOT FOUND')"
"$PYTHON" --version 2>&1
echo "conda env   : ${CONDA_DEFAULT_ENV:-none active} (${CONDA_PREFIX:-no CONDA_PREFIX})"
echo "ENV_DIR     : ${ENV_DIR:-unset}"
if have conda; then
    echo "conda       : $(conda --version 2>&1) at $(command -v conda)"
    conda env list 2>/dev/null | sed 's/^/  /'
else
    echo "conda       : not on PATH on this node"
    ls -d "$HOME"/.conda/envs/* "$HOME"/miniconda3/envs/* "$HOME"/anaconda3/envs/* 2>/dev/null | sed 's/^/  env dir: /'
fi

section "GPU (nvidia-smi)"
if have nvidia-smi; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv 2>&1
else
    echo "nvidia-smi not found: this is expected on the login node. Run the srun command at the top of this script."
fi

section "PYTORCH"
"$PYTHON" - <<'PY' 2>&1
try:
    import torch
except Exception as exc:
    print("torch: NOT IMPORTABLE:", exc)
else:
    print("torch          :", torch.__version__)
    print("CUDA build     :", torch.version.cuda)
    print("cuDNN          :", torch.backends.cudnn.version())
    print("cuda available :", torch.cuda.is_available())
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index}          : {properties.name}, {properties.total_memory / 2**30:.1f} GiB, "
              f"capability {properties.major}.{properties.minor}")
    if torch.cuda.is_available():
        print("CUDA tensor op :", (torch.ones(1, device="cuda") + 1).item())
        print("AMP bf16 ok    :", torch.cuda.is_bf16_supported())
PY

section "LIBRARIES"
"$PYTHON" - <<'PY' 2>&1
from importlib import metadata

REQUIRED = ["torch", "torchvision", "segmentation-models-pytorch", "timm", "albumentations"]
OPTIONAL = ["numpy", "scipy", "pandas", "matplotlib", "opencv-python", "opencv-python-headless", "pillow",
            "scikit-image", "scikit-learn", "pyyaml", "tqdm", "thop", "fvcore", "huggingface-hub",
            "safetensors", "einops", "nnunetv2", "mamba-ssm", "triton"]
for title, names in (("required", REQUIRED), ("optional", OPTIONAL)):
    print(f"-- {title}")
    for name in names:
        try:
            print(f"{name:30s} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name:30s} NOT INSTALLED")
PY

section "BUILD TOOLS (only matter for SCSegamba / mamba-ssm)"
echo "gcc    : $(gcc --version 2>/dev/null | head -n1 || echo 'not found')"
echo "nvcc   : $(nvcc --version 2>/dev/null | tail -n1 || echo 'not found')"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
if type module >/dev/null 2>&1; then
    echo "module avail (cuda / gcc):"
    module avail 2>&1 | grep -i -E 'cuda|gcc|cudnn' | sed 's/^/  /' | head -n 20
else
    echo "environment modules: not available"
fi

section "SLURM"
if have sinfo; then
    echo "partition | avail | time limit | nodes | cpus | mem(MB) | gres | state"
    sinfo -o '%P | %a | %l | %D | %c | %m | %G | %T' 2>&1
    echo
    echo "my jobs:"
    squeue -u "$(whoami)" 2>&1 | head -n 15
    echo
    echo "limits (sacctmgr):"
    sacctmgr -nP show assoc user="$(whoami)" format=account,partition,qos,maxjobs,maxsubmit,grptres,maxwall 2>&1 | head -n 10
    echo "max array size: $(scontrol show config 2>/dev/null | grep -i MaxArraySize)"
else
    echo "sinfo not found"
fi

section "DISK"
df -h "$HOME" 2>&1
have quota && quota -s 2>&1 | tail -n 5
echo "DATA_ROOT : ${DATA_ROOT:-unset}"
echo "RUNS_ROOT : ${RUNS_ROOT:-unset}"
echo "TORCH_HOME: ${TORCH_HOME:-unset (default ~/.cache/torch)}"
echo "HF_HOME   : ${HF_HOME:-unset (default ~/.cache/huggingface)}"
for d in "${DATA_ROOT:-$HOME/dataset_20261113/split_80_10_10}" "$HOME/.cache/torch" "$HOME/.cache/huggingface"; do
    [ -e "$d" ] && echo "$(du -sh "$d" 2>/dev/null | cut -f1)  $d"
done

section "INTERNET FROM THIS NODE (weights must be pre-downloaded if these fail)"
for url in https://download.pytorch.org https://huggingface.co https://github.com https://pypi.org; do
    if have curl; then
        code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$url" 2>/dev/null)
        [ "$code" != "000" ] && echo "reachable   ($code) $url" || echo "UNREACHABLE       $url"
    else
        echo "curl not found; cannot test $url"
    fi
done

section "PREVIOUS RESULT FILES (for the 'previous run, not same protocol' table)"
find "$HOME" -maxdepth 6 \( -name training_history.csv -o -name val_threshold_sweep.csv -o -name test_metrics.json \) \
    -not -path '*/.conda/*' -not -path '*/site-packages/*' 2>/dev/null | head -n 40

echo
echo "Done."
