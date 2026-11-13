#!/usr/bin/env python3
"""Check SCSegamba on a GPU node before submitting jobs (run with the scsegamba env's python).

    srun -p ASL-gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:20:00 \
         $HOME/.conda/envs/scsegamba/bin/python scripts/test_scsegamba.py

Passes if the Mamba CUDA kernels import, the official network builds, and a training
step and an evaluation forward run at the benchmark sizes. Prints the largest batch that
fits, to set train.batch_size / train.accum_steps in configs/scsegamba.yaml.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import torch

    print(f"torch {torch.__version__} | cuda build {torch.version.cuda} | available {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no GPU visible. Mamba's kernels are CUDA only; run this inside srun --gres=gpu:1.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm import Mamba

        layer = Mamba(d_model=16, d_state=8, d_conv=4, expand=2).cuda()
        layer(torch.rand(1, 32, 16, device="cuda")).sum().backward()
        print("[ok] mamba-ssm kernels run on this GPU")
    except Exception:
        print("FAIL: mamba-ssm does not work here:\n" + traceback.format_exc())
        return 1
    try:
        import engine
        from losses import BCEDiceLoss
        from models import build_model

        model = build_model("scsegamba", pretrained=False, config={}).cuda()
        print(f"[ok] SCSegamba built: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
        loss_fn = BCEDiceLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = engine.grad_scaler(torch.device("cuda"), True)
        largest = 0
        for batch in (1, 2, 4, 8, 16):
            try:
                torch.cuda.reset_peak_memory_stats()
                images = torch.rand(batch, 3, 256, 256, device="cuda")
                masks = (torch.rand(batch, 1, 256, 256, device="cuda") > 0.97).float()
                model.train()
                with engine.autocast(torch.device("cuda"), True):
                    logits = model(images)
                assert logits.shape == masks.shape, f"output {tuple(logits.shape)} != {tuple(masks.shape)}"
                loss = loss_fn(logits, masks)
                assert torch.isfinite(loss), "non-finite loss with AMP (set train.amp=false in configs/scsegamba.yaml)"
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                largest = batch
                print(f"[ok] train step, batch {batch:2d} @ 256 px: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            except engine.OutOfMemoryError:
                print(f"[--] batch {batch} @ 256 px does not fit")
                torch.cuda.empty_cache()
                break
        if not largest:
            print("FAIL: not even batch 1 fits")
            return 1
        model.eval()
        with torch.no_grad():
            for size in ((256, 256), (320, 320), (384, 544), (512, 512)):
                out = model(torch.rand(1, 3, *size, device="cuda"))
                assert out.shape[-2:] == size, f"eval output {tuple(out.shape)} for input {size}"
        print("[ok] evaluation forward at 256, 320, 384x544 and 512 px")
        accumulate = max(1, 16 // largest)
        print(f"\nPASS. Benchmark batch is 16: use train.batch_size={min(16, largest)} train.accum_steps={accumulate} "
              "in configs/scsegamba.yaml" + (" (already the default)." if largest >= 16 else "."))
        return 0
    except Exception:
        print("FAIL:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
