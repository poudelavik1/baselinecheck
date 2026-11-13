#!/usr/bin/env python3
"""Turn nnU-Net probability files into the 8-bit probability PNGs evaluate.py reads.

    python scripts/nnunet/export_predictions.py --input <nnUNetv2_predict output> --split val --out <dir>

nnUNetv2_predict --save_probabilities writes <case>.npz with 'probabilities' of shape
(classes, 1, H, W). The crack-class probability is saved as <out>/<split>/<tile name>.png
so that nnU-Net is scored by exactly the same code as every other model.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DATASET = "Dataset501_Cracks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with (Path(os.environ["nnUNet_raw"]) / DATASET / "case_map.csv").open(newline="", encoding="utf-8") as handle:
        cases = {row["case"]: row["name"] for row in csv.DictReader(handle) if row["split"] == args.split}
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    missing = 0
    for case, name in cases.items():
        path = Path(args.input) / f"{case}.npz"
        if not path.exists():
            missing += 1
            continue
        probabilities = np.load(path)["probabilities"]
        crack = np.squeeze(probabilities[1]).astype(np.float32)
        cv2.imwrite(str(out / f"{name}.png"), np.clip(np.round(crack * 255.0), 0, 255).astype(np.uint8))
    print(f"{args.split}: wrote {len(cases) - missing} probability maps to {out}" + (f" ({missing} MISSING)" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
