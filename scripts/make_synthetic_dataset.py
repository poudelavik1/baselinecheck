#!/usr/bin/env python3
"""Generate a tiny synthetic crack dataset (random thin polylines on textured tiles).

Used by the smoke test so the whole pipeline can be checked without the real data:
    python scripts/make_synthetic_dataset.py --out /tmp/synthetic --train 24 --val 8 --test 8

It mimics the quirks of the real dataset: <source>_r####_c#### names, a few tiles of
other sizes, one JPEG mask and one 0/1 mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_tile(rng: np.random.Generator, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    base = rng.integers(110, 190)
    image = rng.normal(base, 12, (height, width, 3)).astype(np.float32)
    image += cv2.GaussianBlur(rng.normal(0, 40, (height, width)).astype(np.float32), (0, 0), 25)[..., None]
    mask = np.zeros((height, width), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        points = [np.array([rng.uniform(0, width), rng.uniform(0, height)])]
        direction = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(4, 10))):
            direction += rng.normal(0, 0.5)
            points.append(points[-1] + 30 * np.array([np.cos(direction), np.sin(direction)]))
        polyline = np.round(np.array(points)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [polyline], False, 255, thickness=int(rng.integers(1, 5)))
    darker = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.7)[..., None] / 255.0
    image = np.clip(image - 70 * darker, 0, 255).astype(np.uint8)
    return image, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
        for index in range(count):
            if index % 8 == 5:
                height, width, stem = 320, 320, f"{split}{index:03d}"
            elif index % 8 == 6:
                height, width, stem = 384, 544, f"IMG{index:02d}_{split}_{index}"
            else:
                height, width, stem = 256, 256, f"SYN{index % 2 + 1}_{split}_r{index:04d}_c{index * 3 % 17:04d}"
            image, mask = make_tile(rng, height, width)
            cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if index % 8 == 5:      # JPEG mask with the same name as the image, as in CRACK500
                cv2.imwrite(str(root / split / "masks" / f"{stem}.jpg"), mask)
            elif index % 8 == 6:    # 0/1 mask
                cv2.imwrite(str(root / split / "masks" / f"{stem}_mask.png"), (mask > 0).astype(np.uint8))
            else:
                cv2.imwrite(str(root / split / "masks" / f"{stem}_mask.png"), mask)
    print(f"Synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
