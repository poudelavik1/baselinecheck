#!/usr/bin/env python3
"""Write the job list of the Slurm array: one line per (model, seed, config).

    python scripts/make_joblist.py --out joblist.txt                    # all models x seeds 0 1 2
    python scripts/make_joblist.py --tiers core tier1 --out joblist.txt
    python scripts/make_joblist.py --families CNN --seeds 0 --out joblist.txt
    python scripts/make_joblist.py --models unet_r34 deepcrack --out joblist.txt
    python scripts/make_joblist.py --missing --out joblist.txt          # only runs without FINISHED
    python scripts/make_joblist.py --protocol original --out joblist.txt

Line k (0-based) is the task with SLURM_ARRAY_TASK_ID = k. The array size follows the
registry automatically. Prints the number of tasks on the last line of stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config as cfg  # noqa: E402
from models import FAMILIES, TIERS, select  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tiers", nargs="*", default=None, choices=TIERS)
    parser.add_argument("--families", nargs="*", default=None, choices=FAMILIES)
    parser.add_argument("--env", default="main", help="main | scsegamba (models needing another env are left out)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--protocol", default="benchmark", choices=("benchmark", "original"))
    parser.add_argument("--missing", action="store_true", help="Leave out runs that already have FINISHED")
    parser.add_argument("--runs-root", default=None)
    args = parser.parse_args()

    runs = cfg.runs_root(args.runs_root)
    lines = []
    for spec in select(args.models, args.tiers, args.families, env=args.env):
        folder = "configs" if args.protocol == "benchmark" else "configs/original"
        config_path = REPO_ROOT / folder / f"{spec.name}.yaml"
        if not config_path.exists():
            if args.protocol == "benchmark":
                print(f"WARNING: {config_path} missing; run scripts/make_configs.py", file=sys.stderr)
            continue
        for seed in args.seeds:
            if args.missing and (cfg.run_dir(runs, args.protocol, spec.name, seed) / "FINISHED").exists():
                continue
            lines.append(f"{spec.name} {seed} {folder}/{spec.name}.yaml")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    for index, line in enumerate(lines):
        print(f"{index:3d}  {line}", file=sys.stderr)
    print(len(lines))


if __name__ == "__main__":
    main()
