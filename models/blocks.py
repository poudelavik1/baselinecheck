"""Blocks shared by HairlineUNet and HairlineSwinUNet, copied unchanged from the original scripts
(train_unet_hairline.py and train_swin_unet_hairline/src/swinunet/models.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(ConvBNReLU(input_channels, output_channels), ConvBNReLU(output_channels, output_channels))


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.conv = DoubleConv(input_channels + skip_channels, output_channels)
        self.attention = SCSE(output_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.attention(self.conv(torch.cat((x, skip), 1)))


class ThinLineResponse(nn.Module):
    """Fixed morphological black-hat and white top-hat on grayscale input.

    Black-hat (closing - image) responds to dark structures narrower than the
    kernel; top-hat (image - opening) to bright ones. The module has no parameters.
    """

    def __init__(self, kernel_sizes=(7, 15)):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.output_channels = 2 * len(self.kernel_sizes)
        self.register_buffer("luma", torch.tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1))

    def forward(self, rgb):
        gray = (rgb * self.luma).sum(1, keepdim=True)
        responses = []
        for size in self.kernel_sizes:
            closed = -F.max_pool2d(-F.max_pool2d(gray, size, 1, size // 2), size, 1, size // 2)
            opened = F.max_pool2d(-F.max_pool2d(-gray, size, 1, size // 2), size, 1, size // 2)
            responses.extend((closed - gray, gray - opened))
        return torch.cat(responses, 1)
