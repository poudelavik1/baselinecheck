#!/usr/bin/env python3
"""Pre-download every pretrained encoder on the LOGIN node (compute nodes may be offline).

    export WEIGHTS_CACHE=$HOME/weights_cache
    python scripts/download_weights.py                 # all models of the main env
    python scripts/download_weights.py --models unet_r34 segformer_b2

It simply builds each registered model with pretrained weights, with TORCH_HOME and
HF_HOME pointed below $WEIGHTS_CACHE, so the files land exactly where training looks for
them. Training jobs then run with OFFLINE=1 (set in slurm/common.sh). Afterwards it
verifies that every model can be built again with the network switched off.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=os.environ.get("WEIGHTS_CACHE"), help="Default: $WEIGHTS_CACHE")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--env", default="main")
    parser.add_argument("--verify-only", action="store_true", help="Only check that the cache works offline")
    args = parser.parse_args()
    if not args.cache:
        print("Set WEIGHTS_CACHE or pass --cache.", file=sys.stderr)
        return 2
    cache = Path(args.cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["WEIGHTS_CACHE"] = str(cache)
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HOME"] = str(cache / "huggingface")
    if args.verify_only:
        os.environ["OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    import engine
    from models import build_model, select

    engine.use_offline_weights()
    failed = []
    for spec in select(models=args.models, env=args.env):
        try:
            build_model(spec.name, pretrained=True, config={})
            print(f"[ok]   {spec.name}", flush=True)
        except Exception as exc:
            failed.append(spec.name)
            print(f"[FAIL] {spec.name}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 2**30
    print(f"Cache: {cache} ({size:.2f} GiB)")
    if failed:
        print(f"FAILED: {failed}")
        return 1
    if not args.verify_only:
        print("\nVerifying that the cache works with the network switched off ...", flush=True)
        command = [sys.executable, __file__, "--cache", str(cache), "--env", args.env, "--verify-only"]
        return subprocess.call(command + (["--models", *args.models] if args.models else []))
    print("All models load offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
