"""HairlineUNet, ported unchanged from train_unet_hairline.py.

In the benchmark it is built with deep_supervision=False: the auxiliary heads are a
training device and are not on the path from input to output, so the network that
produces the prediction is identical. The original-recipe config turns them back on.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import DecoderBlock, DoubleConv, ThinLineResponse

ENCODER_CHANNELS = {
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
}


class HairlineUNet(nn.Module):
    """ResNet-encoder U-Net with a full-resolution skip for 1-4 px cracks.

    Input: RGB in [0, 1]. Eval mode returns logits (N, 1, H, W). Train mode with deep
    supervision returns (logits, [auxiliary logits]).
    """

    def __init__(
        self, encoder_name: str = "resnet34", pretrained: bool = True,
        decoder_channels=(256, 128, 64, 48, 32), stem_channels: int = 32,
        line_prior: bool = True, deep_supervision: bool = True, prior_probability: float = 0.03,
    ):
        super().__init__()
        import torchvision

        self.encoder = torchvision.models.get_model(encoder_name, weights="DEFAULT" if pretrained else None)
        self.encoder.fc = nn.Identity()
        c0, c1, c2, c3, c4 = ENCODER_CHANNELS[encoder_name]
        d4, d3, d2, d1, d0 = decoder_channels
        self.thin_lines = ThinLineResponse() if line_prior else None
        stem_inputs = 3 + (self.thin_lines.output_channels if line_prior else 0)
        self.full_resolution = DoubleConv(stem_inputs, stem_channels)
        self.dec4 = DecoderBlock(c4, c3, d4)              # 1/16
        self.dec3 = DecoderBlock(d4, c2, d3)              # 1/8
        self.dec2 = DecoderBlock(d3, c1, d2)              # 1/4
        self.dec1 = DecoderBlock(d2, c0, d1)              # 1/2
        self.dec0 = DecoderBlock(d1, stem_channels, d0)   # 1/1
        self.head = nn.Conv2d(d0, 1, 1)
        self.deep_supervision = deep_supervision
        self.auxiliary_heads = nn.ModuleList([nn.Conv2d(d2, 1, 1), nn.Conv2d(d1, 1, 1)]) if deep_supervision else None
        # Start near the crack prior so early training is not dominated by background.
        bias = -math.log((1 - prior_probability) / prior_probability)
        for head in [self.head, *(self.auxiliary_heads or [])]:
            nn.init.constant_(head.bias, bias)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        normalized = (x - self.image_mean) / self.image_std
        stem_input = normalized if self.thin_lines is None else torch.cat((normalized, self.thin_lines(x)), 1)
        full = self.full_resolution(stem_input)
        encoder = self.encoder
        e0 = encoder.relu(encoder.bn1(encoder.conv1(normalized)))   # 1/2
        e1 = encoder.layer1(encoder.maxpool(e0))                     # 1/4
        e2 = encoder.layer2(e1)                                      # 1/8
        e3 = encoder.layer3(e2)                                      # 1/16
        e4 = encoder.layer4(e3)                                      # 1/32
        d = self.dec3(self.dec4(e4, e3), e2)
        d2 = self.dec2(d, e1)
        d1 = self.dec1(d2, e0)
        logits = self.head(self.dec0(d1, full))
        if self.training and self.deep_supervision:
            auxiliary = [
                F.interpolate(head(features), size=size, mode="bilinear", align_corners=False)
                for head, features in zip(self.auxiliary_heads, (d2, d1))
            ]
            return logits, auxiliary
        return logits
