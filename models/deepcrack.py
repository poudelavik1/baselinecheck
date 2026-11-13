"""DeepCrack (Liu et al., Neurocomputing 2019), implemented from the paper.

"DeepCrack: A deep hierarchical feature learning architecture for crack segmentation".
The network is VGG-16 with batch normalization, without the fully connected layers and
the fifth pooling layer. A 1x1 convolution on the last feature map of each of the five
convolution stages gives a side output, which is upsampled to the input size; the five
side outputs are concatenated and fused by a 1x1 convolution (deeply supervised nets).

Differences from the paper, on purpose:
* The guided-filter / CRF refinement is a post-processing step outside the network and
  is not applied: every model in the benchmark is scored on its raw probability map.
* Side outputs are upsampled bilinearly (the paper's deconvolution, with fixed weights).
* In the benchmark only the fused output is supervised (loss.aux_weight = 0), like every
  other model; set loss.aux_weight > 0 for the paper's deep supervision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DeepCrack(nn.Module):
    """Expects ImageNet-normalized input; returns logits, or (logits, side logits) in train mode."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision

        features = torchvision.models.vgg16_bn(weights="DEFAULT" if pretrained else None).features
        # vgg16_bn.features: stage k ends with a max-pool at indices 6, 13, 23, 33, 43.
        self.stage1 = features[0:6]      # 64,  1/1
        self.stage2 = features[6:13]     # 128, 1/2 (starts with the pool of stage 1)
        self.stage3 = features[13:23]    # 256, 1/4
        self.stage4 = features[23:33]    # 512, 1/8
        self.stage5 = features[33:43]    # 512, 1/16 (the fifth pool is dropped)
        self.side = nn.ModuleList(nn.Conv2d(channels, 1, 1) for channels in (64, 128, 256, 512, 512))
        self.fuse = nn.Conv2d(5, 1, 1)
        nn.init.constant_(self.fuse.weight, 0.2)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor):
        size = x.shape[-2:]
        sides = []
        for stage, side in zip((self.stage1, self.stage2, self.stage3, self.stage4, self.stage5), self.side):
            x = stage(x)
            sides.append(F.interpolate(side(x), size=size, mode="bilinear", align_corners=False))
        fused = self.fuse(torch.cat(sides, 1))
        if self.training:
            return fused, sides
        return fused
