"""SCSegamba (Liu et al., CVPR 2025), from the official code.

https://github.com/Karl1109/SCSegamba (Apache-2.0). The repository needs mamba-ssm
(compiled CUDA kernels) and mmcv-full, so it lives in its own conda environment:
scripts/install_scsegamba.sh creates the env and clones the code into
third_party/SCSegamba, and scripts/test_scsegamba.py checks it in an srun session.

The network is built exactly as the official models/decoder.py does (SAVSS backbone,
arch='Crack', + MFS head) and is trained from scratch, as in the paper.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from .third_party import foreign_packages, third_party_dir


class SCSegamba(nn.Module):
    """Expects input normalized with mean 0.5 / std 0.5, as in the official loader."""

    def __init__(self, network: nn.Module, multiple: int = 32):
        super().__init__()
        self.network = network
        self.multiple = multiple

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % self.multiple, (-width) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        logits = self.network(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits[..., :height, :width]


def build_scsegamba() -> nn.Module:
    repository = third_party_dir("SCSegamba", "SCSEGAMBA_DIR")
    with foreign_packages(repository, ("models", "mmcls", "util", "datasets", "eval")):
        from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS  # type: ignore[import-not-found]
        from models.decoder import Decoder  # type: ignore[import-not-found]

        backbone = SAVSS(arch="Crack", out_indices=(0, 1, 2, 3), drop_path_rate=0.2, final_norm=True,
                         convert_syncbn=True)
        network = Decoder(backbone)
    return SCSegamba(network)
