#!/usr/bin/env python3
"""Measure training speed and GPU memory per model and estimate the cost of the benchmark.

Run it on a GPU node before submitting the array:
    srun -p ASL-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:40:00 \
        python scripts/estimate_compute.py --train-tiles 13603 --val-tiles 1700

For every model it runs a few real training steps (random tensors, benchmark crop and
batch size, AMP) and reports steps/s, peak memory and the hours of one run with the
configured number of epochs (an upper bound: early stopping usually ends sooner).
Data loading is not included; with 8 workers it is rarely the bottleneck at 256 px.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import config as cfg  # noqa: E402
import engine  # noqa: E402
from losses import build_loss  # noqa: E402
from models import build_model, select  # noqa: E402


def measure(name: str, config: dict, device: torch.device, steps: int) -> dict:
    batch, crop = int(config["train"]["batch_size"]), int(config["data"]["crop_size"])
    model = build_model(name, pretrained=False, config=config).to(device).train()
    loss_fn = build_loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = engine.grad_scaler(device, True)
    images = torch.rand(batch, 3, crop, crop, device=device)
    masks = (torch.rand(batch, 1, crop, crop, device=device) > 0.97).float()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps + 5):
        if step == 5:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
        with engine.autocast(device, True):
            outputs = model(images)
        loss = loss_fn(outputs, masks, 1.0)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_step = (time.perf_counter() - started) / steps
    model.eval()
    with torch.no_grad(), engine.autocast(device, True):
        for step in range(steps // 2 + 2):
            if step == 2:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
            model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_step = (time.perf_counter() - started) / (steps // 2)
    memory = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    return {"batch": batch, "train_step": train_step, "eval_step": eval_step, "memory": memory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-tiles", type=int, default=13603)
    parser.add_argument("--val-tiles", type=int, default=1700)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--env", default="main")
    parser.add_argument("--set", nargs="*", default=[], help="Config overrides, e.g. train.batch_size=32")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {engine.device_name(device)}"
          + (f", {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB" if device.type == "cuda" else ""))
    print(f"{'model':18s} {'tier':6s} {'batch':>5s} {'GiB':>6s} {'step/s':>7s} {'min/epoch':>10s} {'h/run (max)':>12s}")
    totals: dict[str, float] = {}
    for spec in select(models=args.models, env=args.env):
        config = cfg.load_config(REPO_ROOT / "configs" / f"{spec.name}.yaml", args.set)
        try:
            result = measure(spec.name, config, device, args.steps)
        except Exception as exc:
            print(f"{spec.name:18s} {spec.tier:6s} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        epoch_seconds = (args.train_tiles / result["batch"]) * result["train_step"] \
            + (args.val_tiles / result["batch"]) * result["eval_step"]
        hours = epoch_seconds * int(config["train"]["epochs"]) / 3600
        totals[spec.tier] = totals.get(spec.tier, 0.0) + hours * args.seeds
        print(f"{spec.name:18s} {spec.tier:6s} {result['batch']:5d} {result['memory']:6.1f} "
              f"{1 / result['train_step']:7.1f} {epoch_seconds / 60:10.1f} {hours:12.1f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nGPU-hours for {args.seeds} seeds, all epochs run (upper bound):")
    for tier, hours in totals.items():
        print(f"  {tier:6s} {hours:7.1f} h")
    print(f"  total  {sum(totals.values()):7.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
