"""Shared training/inference helpers: seeding, EMA, full-tile and sliding-window prediction."""

from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch; request deterministic cuDNN where feasible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def use_offline_weights() -> None:
    """Point torch / Hugging Face at the pre-downloaded cache (see scripts/download_weights.py).

    WEIGHTS_CACHE=<dir> sets TORCH_HOME and HF_HOME below it. OFFLINE=1 (set by the Slurm
    scripts) forbids network access so a missing weight fails at once instead of hanging.
    """
    cache = os.environ.get("WEIGHTS_CACHE")
    if cache:
        os.environ.setdefault("TORCH_HOME", str(Path(cache) / "torch"))
        os.environ.setdefault("HF_HOME", str(Path(cache) / "huggingface"))
    if os.environ.get("OFFLINE") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ModelEMA:
    """Exponential moving average of weights (and BN statistics). Original-recipe runs only."""

    def __init__(self, model: nn.Module, decay: float = 0.999, ramp_steps: int = 2000):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.ramp_steps, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.ramp_steps))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


# torch 1.13 (the SCSegamba environment) has neither torch.OutOfMemoryError nor torch.amp.GradScaler.
OutOfMemoryError = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def grad_scaler(device: torch.device, amp: bool):
    enabled = amp and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _forward_padded(model: nn.Module, images: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    """Logits for a batch whose sides need not be multiples of 32 (reflect-pad, then crop)."""
    height, width = images.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        images = F.pad(images, (0, pad_w, 0, pad_h), mode=mode)
    logits = model(images)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits[..., :height, :width].float()


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window, stride))
    return starts + [length - window]


def _forward_tta(model: nn.Module, images: torch.Tensor, tta: bool) -> torch.Tensor:
    """Probabilities; with TTA the mean over identity, horizontal and vertical flips."""
    probability = torch.sigmoid(_forward_padded(model, images))
    if not tta:
        return probability
    for dims in ((-1,), (-2,)):
        probability = probability + torch.sigmoid(_forward_padded(model, images.flip(dims))).flip(dims)
    return probability / 3.0


@torch.no_grad()
def predict(model: nn.Module, images: torch.Tensor, config: dict[str, Any], amp: bool = False) -> torch.Tensor:
    """Probabilities (N, 1, H, W) at the input size.

    Tiles up to data.eval_max_side are predicted whole; larger images with a sliding
    window (data.eval_window / data.eval_stride), averaging the overlaps.
    """
    data = config["data"]
    tta = bool(config.get("recipe", {}).get("tta", False))
    device = images.device
    height, width = images.shape[-2:]
    with autocast(device, amp):
        if max(height, width) <= int(data["eval_max_side"]):
            return _forward_tta(model, images, tta).float()
        window, stride = int(data["eval_window"]), int(data["eval_stride"])
        total = torch.zeros((images.shape[0], 1, height, width), device=device)
        weight = torch.zeros((1, 1, height, width), device=device)
        boxes = [(top, left) for top in _window_starts(height, window, stride)
                 for left in _window_starts(width, window, stride)]
        for start in range(0, len(boxes), 8):
            chunk = boxes[start:start + 8]
            crops = torch.cat([images[..., t:t + window, l:l + window] for t, l in chunk], 0)
            probabilities = _forward_tta(model, crops, tta).float()
            for index, (top, left) in enumerate(chunk):
                part = probabilities[index * images.shape[0]:(index + 1) * images.shape[0]]
                total[..., top:top + window, left:left + window] += part
                weight[..., top:top + window, left:left + window] += 1
        return total / weight


def atomic_save(obj: Any, path: Path) -> None:
    """Write a checkpoint so that a job killed mid-write never leaves a corrupt file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    import platform

    return f"CPU ({platform.processor() or platform.machine()})"
