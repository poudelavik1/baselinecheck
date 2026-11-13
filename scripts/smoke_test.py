#!/usr/bin/env python3
"""End-to-end smoke test: 1 epoch on ~20 tiles for every model, through figures and tables.

    python scripts/smoke_test.py                       # synthetic data, random init (laptop)
    python scripts/smoke_test.py --data-root $DATA_ROOT --pretrained     # server: real data + weight cache
    python scripts/smoke_test.py --models unet_r34 deepcrack

One model failing never stops the others; the summary at the end lists what failed.
Results go to a throw-away folder (default: ./runs/_smoke), never to the real runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import select  # noqa: E402


def run(command: list[str]) -> tuple[bool, str]:
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    process = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env=environment)
    return process.returncode == 0, (process.stdout + process.stderr)


def drop_resume_checkpoints(runs: Path) -> None:
    """last.pt (weights + optimizer) is the large file and is not needed after a smoke run."""
    for path in runs.glob("*/*/seed_*/last.pt"):
        path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None, help="Default: a synthetic dataset is generated")
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs" / "_smoke"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tiers", nargs="*", default=None)
    parser.add_argument("--env", default="main", help="Models of this environment (main | scsegamba)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained weights (tests the weight cache)")
    parser.add_argument("--keep", action="store_true", help="Do not delete previous smoke results first")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    runs = Path(args.runs_root)
    if runs.exists() and not args.keep:
        shutil.rmtree(runs)
    runs.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    data_root = args.data_root
    if not data_root:
        data_root = str(runs / "synthetic_data")
        ok, log = run([python, "scripts/make_synthetic_dataset.py", "--out", data_root])
        if not ok:
            print(log)
            return 1
    common = ["--data-root", data_root, "--runs-root", str(runs)]

    specs = select(models=args.models, tiers=args.tiers, env=args.env)
    status: dict[str, str] = {}
    for spec in specs:
        for seed in args.seeds:
            started = time.time()
            command = [python, "train.py", "--config", f"configs/{spec.name}.yaml", "--seed", str(seed), "--smoke-test",
                       *common]
            if not args.pretrained:
                command.append("--no-pretrained")
            ok, log = run(command)
            drop_resume_checkpoints(runs)
            if not ok and "is not installed" in log:
                print(f"[skip] {spec.name}: third-party code not installed (see SERVER_GUIDE.md, Tier 2)", flush=True)
                break
            status[f"{spec.name}/seed_{seed}"] = "ok" if ok else "TRAIN FAILED"
            print(f"[{'ok' if ok else 'FAIL'}] train {spec.name} seed {seed} ({time.time() - started:.0f}s)", flush=True)
            if not ok:
                print("\n".join(log.strip().splitlines()[-12:]))

    # The original-recipe path (own loss, EMA, deep supervision, TTA) on one seed.
    for name in ("hairline_unet", "swin_unet"):
        config = REPO_ROOT / "configs" / "original" / f"{name}.yaml"
        if config.exists() and (not args.models or name in args.models) and args.env == "main":
            command = [python, "train.py", "--config", str(config), "--seed", "0", "--smoke-test", *common]
            if not args.pretrained:
                command.append("--no-pretrained")
            ok, log = run(command)
            status[f"original/{name}"] = "ok" if ok else "TRAIN FAILED"
            print(f"[{'ok' if ok else 'FAIL'}] train original-recipe {name}", flush=True)
            if not ok:
                print("\n".join(log.strip().splitlines()[-12:]))

    steps = [
        ("evaluate val", [python, "evaluate.py", "--split", "val", *common]),
        ("evaluate test", [python, "evaluate.py", "--split", "test", "--no-efficiency", *common]),
        ("evaluate original val", [python, "evaluate.py", "--split", "val", "--protocol", "original", "--no-efficiency", *common]),
        ("evaluate original test", [python, "evaluate.py", "--split", "test", "--protocol", "original", "--no-efficiency", *common]),
        ("report + figures", [python, "make_report.py", "--qualitative", "--runs-root", str(runs), "--data-root", data_root]),
    ]
    for title, command in steps:
        started = time.time()
        ok, log = run(command)
        status[title] = "ok" if ok else "FAILED"
        print(f"[{'ok' if ok else 'FAIL'}] {title} ({time.time() - started:.0f}s)", flush=True)
        tail = log.strip().splitlines()
        print("\n".join(tail[-40:] if title.startswith("report") or not ok else [l for l in tail if "FAIL" in l or "Error" in l][:20]))

    figures = sorted(p.name for p in (runs / "figures").glob("*.png"))
    tables = sorted(p.name for p in (runs / "tables").glob("*.csv"))
    print(f"\nFigures ({len(figures)}): {', '.join(figures)}")
    print(f"Tables  ({len(tables)}): {', '.join(tables)}")
    failed = {key: value for key, value in status.items() if value != "ok"}
    print(f"\nSMOKE TEST: {len(status) - len(failed)}/{len(status)} steps ok" + (f" | FAILED: {failed}" if failed else ""))
    expected = 9   # figures 1-8 (figure 2 is two files)
    if len(figures) < expected:
        print(f"WARNING: expected at least {expected} figures, found {len(figures)}")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
