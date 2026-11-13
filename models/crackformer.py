"""CrackFormer-II (Liu et al., ICCV 2021 / IEEE T-ITS 2023), from the official code.

The official repository (https://github.com/LouisNUST/CrackFormer-II) has no licence
file, so its code is not copied here: scripts/install_crackformer.sh clones it into
third_party/CrackFormer-II and this module wraps the official `crackformer` class,
unchanged. The network is trained from scratch, as in the paper (no pretrained weights).

Adaptation to the benchmark: the official forward returns five side outputs and the
fused output. The fused output is the prediction; the side outputs are returned as
auxiliary logits in train mode (unsupervised in the benchmark, loss.aux_weight = 0).
SegNet-style unpooling needs sides divisible by 32: other sizes are padded and cropped.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from .third_party import foreign_packages, third_party_dir


class CrackFormer(nn.Module):
    """Expects input normalized with mean 0.5 / std 0.5, as in the official loader."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % 32, (-width) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        *sides, fused = self.network(x)
        if pad_h or pad_w:
            fused = fused[..., :height, :width]
            sides = [side[..., :height, :width] for side in sides]
        if self.training:
            return fused, list(sides)
        return fused


def build_crackformer() -> nn.Module:
    repository = third_party_dir("CrackFormer-II", "CRACKFORMER_DIR")
    source = repository / "CrackFormer-II" if (repository / "CrackFormer-II" / "nets").is_dir() else repository
    with foreign_packages(source, ("utils", "nets")):
        from nets.crackformerII import crackformer  # type: ignore[import-not-found]

        network = crackformer()
    return CrackFormer(network)
