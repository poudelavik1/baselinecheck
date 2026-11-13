"""Losses. The benchmark uses BCE + Dice for every model; `hairline` is the original recipe."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def split_outputs(outputs: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """(main logits, auxiliary logits) for models that return either a tensor or a tuple."""
    if isinstance(outputs, (tuple, list)):
        return outputs[0], list(outputs[1])
    return outputs, []


class BCEDiceLoss(nn.Module):
    """Binary cross-entropy plus soft Dice (computed over the batch) on logits."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, aux_weight: float = 0.0, smooth: float = 1.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.aux_weight, self.smooth = bce_weight, dice_weight, aux_weight, smooth

    def single(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits)
        intersection = (probability * target).sum()
        dice = (2 * intersection + self.smooth) / (probability.sum() + target.sum() + self.smooth)
        return self.bce_weight * bce + self.dice_weight * (1 - dice)

    def forward(self, outputs: Any, target: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        logits, auxiliary = split_outputs(outputs)
        loss = self.single(logits, target)
        if self.aux_weight > 0:
            for aux_logits in auxiliary:
                loss = loss + self.aux_weight * self.single(aux_logits, target)
        return loss


# ---- HairlineCrackLoss, ported unchanged from train_swin_unet_hairline (src/swinunet/losses.py) ---- #
def soft_erode(image: torch.Tensor) -> torch.Tensor:
    return torch.minimum(-F.max_pool2d(-image, (3, 1), 1, (1, 0)), -F.max_pool2d(-image, (1, 3), 1, (0, 1)))


def soft_skeleton(image: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability: torch.Tensor, target: torch.Tensor, iterations: int, smooth: float = 1.0) -> torch.Tensor:
    predicted_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    topology_precision = ((predicted_skeleton * target).sum() + smooth) / (predicted_skeleton.sum() + smooth)
    topology_sensitivity = ((target_skeleton * probability).sum() + smooth) / (target_skeleton.sum() + smooth)
    return 1 - 2 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)


class HairlineCrackLoss(nn.Module):
    """BCE + Tversky (region) + clDice (centerline continuity), with deep-supervision terms."""

    def __init__(
        self, bce_weight: float = 1.0, tversky_weight: float = 1.0, cldice_weight: float = 0.5,
        false_positive_weight: float = 0.4, false_negative_weight: float = 0.6,
        skeleton_iterations: int = 10, auxiliary_weights: tuple[float, ...] = (0.4, 0.2),
        label_tolerance: int = 0, cldice_warmup: float = 0.0,
    ):
        super().__init__()
        self.bce_weight, self.tversky_weight, self.cldice_weight = bce_weight, tversky_weight, cldice_weight
        self.fp_weight, self.fn_weight = false_positive_weight, false_negative_weight
        self.skeleton_iterations = skeleton_iterations
        self.auxiliary_weights = tuple(auxiliary_weights)
        self.label_tolerance = int(label_tolerance)
        self.cldice_warmup = cldice_warmup  # fraction of training over which clDice ramps in

    def region_loss(self, logits, target, valid):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        probability = torch.sigmoid(logits) * valid
        target = target * valid
        true_positive = (probability * target).sum()
        false_positive = (probability * (1 - target)).sum()
        false_negative = ((1 - probability) * target).sum()
        tversky = (true_positive + 1.0) / (
            true_positive + self.fp_weight * false_positive + self.fn_weight * false_negative + 1.0
        )
        return self.bce_weight * bce + self.tversky_weight * (1 - tversky), probability, target

    def forward(self, outputs: Any, target: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        logits, auxiliary = split_outputs(outputs)
        valid = torch.ones_like(target)
        if self.label_tolerance > 0:
            ring = F.max_pool2d(target, 2 * self.label_tolerance + 1, 1, self.label_tolerance) - target
            valid = valid * (1 - ring)
        loss, probability, target_valid = self.region_loss(logits, target, valid)
        scale = min(1.0, progress / self.cldice_warmup) if self.cldice_warmup > 0 else 1.0
        weight = self.cldice_weight * scale
        if weight > 0:
            loss = loss + weight * soft_cldice_loss(probability, target_valid, self.skeleton_iterations)
        for auxiliary_weight, auxiliary_logits in zip(self.auxiliary_weights, auxiliary):
            loss = loss + auxiliary_weight * self.region_loss(auxiliary_logits, target, valid)[0]
        return loss


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Loss from the `loss` section. `progress` (0..1 of training) is passed at call time."""
    options = dict(config["loss"])
    name = options.pop("name")
    if name == "bce_dice":
        return BCEDiceLoss(options.get("bce_weight", 1.0), options.get("dice_weight", 1.0), options.get("aux_weight", 0.0))
    if name == "hairline":
        epochs = max(1, int(config["train"]["epochs"]))
        warmup = float(config["recipe"].get("cldice_warmup_epochs", 0)) / epochs
        keys = ("bce_weight", "tversky_weight", "cldice_weight", "false_positive_weight", "false_negative_weight",
                "skeleton_iterations", "auxiliary_weights", "label_tolerance")
        return HairlineCrackLoss(cldice_warmup=warmup, **{k: options[k] for k in keys if k in options})
    raise ValueError(f"Unknown loss: {name}")
