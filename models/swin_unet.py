"""HairlineSwinUNet, ported unchanged from train_swin_unet_hairline/src/swinunet/models.py.

As for HairlineUNet, the benchmark builds it with deep_supervision=False (the auxiliary
heads are not on the path from input to output).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import DecoderBlock, DoubleConv, ThinLineResponse


class PatchExpand(nn.Module):
    """Swin-Unet patch expanding, the inverse of patch merging. Tokens are (N, H, W, C)."""

    def __init__(self, input_channels: int, output_channels: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.expand = nn.Linear(input_channels, output_channels * scale * scale, bias=False)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x):
        batch, height, width, _ = x.shape
        scale = self.scale
        x = self.expand(x).reshape(batch, height, width, scale, scale, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height * scale, width * scale, -1)
        return self.norm(x)


class SwinDecoderStage(nn.Module):
    """Swin-Unet up-stage: patch expanding, skip concatenation, linear fusion, Swin blocks."""

    def __init__(self, input_channels: int, blocks: nn.Sequential):
        super().__init__()
        output_channels = input_channels // 2
        self.expand = PatchExpand(input_channels, output_channels)
        self.fuse = nn.Linear(2 * output_channels, output_channels)
        self.blocks = blocks

    def forward(self, x, skip):
        # Patch merging pads odd sizes, so the expanded map can be one token larger than the skip.
        x = self.expand(x)[:, :skip.shape[1], :skip.shape[2]]
        return self.blocks(self.fuse(torch.cat((x, skip), -1)))


class HairlineSwinUNet(nn.Module):
    """Swin-Unet (Cao et al., 2021) with a full-resolution head for 1-4 px cracks.

    Input: RGB in [0, 1]. Eval mode returns logits (N, 1, H, W). Train mode with deep
    supervision returns (logits, [auxiliary logits]).
    """

    def __init__(
        self, encoder_name: str = "swin_t", pretrained: bool = True, decoder_depth: int = 2,
        hairline_head: bool = True, head_channels=(48, 32), stem_channels: int = 32,
        line_prior: bool = True, deep_supervision: bool = True, prior_probability: float = 0.03,
    ):
        super().__init__()
        import torchvision

        self.encoder = torchvision.models.get_model(encoder_name, weights="DEFAULT" if pretrained else None)
        self.encoder.head = nn.Identity()
        # encoder.features = [patch embedding, stage 1, merge, stage 2, merge, stage 3, merge, stage 4]
        stages = self.encoder.features
        width = stages[0][0].out_channels  # C; the stages carry C, 2C, 4C, 8C channels

        def mirror(stage):
            # As in the paper, a decoder stage starts from the weights of the encoder stage it mirrors.
            return nn.Sequential(*copy.deepcopy(list(stage)[:decoder_depth]))

        self.up3 = SwinDecoderStage(8 * width, mirror(stages[5]))   # 1/16, 4C
        self.up2 = SwinDecoderStage(4 * width, mirror(stages[3]))   # 1/8,  2C
        self.up1 = SwinDecoderStage(2 * width, mirror(stages[1]))   # 1/4,  C
        self.decoder_norm = nn.LayerNorm(width)
        self.hairline_head = hairline_head
        self.thin_lines = ThinLineResponse() if hairline_head and line_prior else None
        if hairline_head:
            d1, d0 = head_channels
            stem_inputs = 3 + (self.thin_lines.output_channels if self.thin_lines is not None else 0)
            self.full_resolution = DoubleConv(stem_inputs, stem_channels)                                        # 1/1
            self.half_resolution = nn.Sequential(nn.MaxPool2d(2), DoubleConv(stem_channels, 2 * stem_channels))  # 1/2
            self.dec1 = DecoderBlock(width, 2 * stem_channels, d1)   # 1/2
            self.dec0 = DecoderBlock(d1, stem_channels, d0)          # 1/1
            self.head = nn.Conv2d(d0, 1, 1)
        else:
            self.final_expand = PatchExpand(width, width, scale=4)   # 1/4 -> 1/1
            self.head = nn.Conv2d(width, 1, 1)
        self.deep_supervision = deep_supervision
        # Auxiliary heads on the Swin decoder tokens at 1/8 and 1/4 scale.
        self.auxiliary_heads = (
            nn.ModuleList([nn.Linear(2 * width, 1), nn.Linear(width, 1)]) if deep_supervision else None
        )
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head, *(self.auxiliary_heads or [])]:
            nn.init.constant_(head.bias, bias)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        normalized = (x - self.image_mean) / self.image_std
        stages = self.encoder.features
        e1 = stages[1](stages[0](normalized))                    # 1/4, tokens (N, H/4, W/4, C)
        e2 = stages[3](stages[2](e1))                            # 1/8
        e3 = stages[5](stages[4](e2))                            # 1/16
        e4 = self.encoder.norm(stages[7](stages[6](e3)))         # 1/32, bottleneck
        d3 = self.up3(e4, e3)                                    # 1/16
        d2 = self.up2(d3, e2)                                    # 1/8
        d1 = self.decoder_norm(self.up1(d2, e1))                 # 1/4
        if self.hairline_head:
            stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
            full = self.full_resolution(stem_input)
            half = self.half_resolution(full)
            logits = self.head(self.dec0(self.dec1(d1.permute(0, 3, 1, 2), half), full))
        else:
            logits = self.head(self.final_expand(d1).permute(0, 3, 1, 2))
            if logits.shape[-2:] != size:  # input size not divisible by the patch size
                logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        if self.training and self.deep_supervision:
            auxiliary = [
                F.interpolate(head(tokens).permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False)
                for head, tokens in zip(self.auxiliary_heads, (d2, d1))
            ]
            return logits, auxiliary
        return logits
