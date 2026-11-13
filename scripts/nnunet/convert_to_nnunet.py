#!/usr/bin/env python3
"""Convert the dataset to nnU-Net v2 format, keeping MY train/val/test split.

    python scripts/nnunet/convert_to_nnunet.py --data-root $DATA_ROOT            # writes $nnUNet_raw/Dataset501_Cracks
    python scripts/nnunet/convert_to_nnunet.py --write-splits                    # after plan_and_preprocess

Layout written (2D natural images: one RGB PNG per case, labels 0/1):
    Dataset501_Cracks/imagesTr, labelsTr     train + val tiles (nnU-Net trains on fold 0 = my split)
    Dataset501_Cracks/imagesVal, imagesTs    val and test tiles, for nnUNetv2_predict
    Dataset501_Cracks/case_map.csv           nnU-Net case id <-> tile name, split
--write-splits puts splits_final.json into $nnUNet_preprocessed so that fold 0 is exactly
train -> train, val -> val (no cross-validation, no leakage of my val tiles into training).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config as cfg  # noqa: E402
import data  # noqa: E402

DATASET = "Dataset501_Cracks"


def case_id(split: str, index: int, name: str) -> str:
    return f"{split}_{index:05d}_{re.sub(r'[^a-z0-9]+', '-', name)[:60]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--write-splits", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Tiles per split (smoke test)")
    args = parser.parse_args()
    raw = Path(os.environ["nnUNet_raw"]) / DATASET

    if args.write_splits:
        with (raw / "case_map.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        splits = [{"train": [r["case"] for r in rows if r["split"] == "train"],
                   "val": [r["case"] for r in rows if r["split"] == "val"]}]
        target = Path(os.environ["nnUNet_preprocessed"]) / DATASET
        target.mkdir(parents=True, exist_ok=True)
        (target / "splits_final.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
        print(f"splits_final.json: fold 0 = {len(splits[0]['train'])} train / {len(splits[0]['val'])} val cases")
        return 0

    root = cfg.data_root(args.data_root)
    base = cfg.load_config(cfg.CONFIG_DIR / "base.yaml")
    for folder in ("imagesTr", "labelsTr", "imagesVal", "imagesTs"):
        (raw / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "val", "test"):
        pairs = data.subsample(data.collect_pairs(root, split, base["data"].get("exclude_regex")), args.limit)
        for index, (name, image_path, mask_path) in enumerate(pairs):
            image, mask = data.read_pair(image_path, mask_path)   # same reading and binarization as the benchmark
            case = case_id(split, index, name)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if split in ("train", "val"):
                cv2.imwrite(str(raw / "imagesTr" / f"{case}_0000.png"), bgr)
                cv2.imwrite(str(raw / "labelsTr" / f"{case}.png"), mask)
            if split == "val":
                cv2.imwrite(str(raw / "imagesVal" / f"{case}_0000.png"), bgr)
            if split == "test":
                cv2.imwrite(str(raw / "imagesTs" / f"{case}_0000.png"), bgr)
            rows.append({"case": case, "name": name, "split": split})
        print(f"{split}: {len(pairs)} tiles", flush=True)
    with (raw / "case_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    training = sum(1 for r in rows if r["split"] in ("train", "val"))
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"}, "labels": {"background": 0, "crack": 1},
        "numTraining": training, "file_ending": ".png",
    }, indent=1), encoding="utf-8")
    print(f"Wrote {raw} ({training} training cases incl. val)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
