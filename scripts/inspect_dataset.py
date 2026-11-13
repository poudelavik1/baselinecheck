#!/usr/bin/env python3
"""Describe a crack-segmentation tile dataset. CPU only, safe on a login node.

Usage:
    python scripts/inspect_dataset.py --data-root ~/dataset_20261113/split_80_10_10 | tee inspect_dataset.txt
    DATA_ROOT=~/dataset_20261113/split_80_10_10 python scripts/inspect_dataset.py

Prints: folder layout, image/mask pairing and formats, tile sizes, mask value range,
sample counts and crack-pixel percentage per split, the source image every tile comes
from (inferred from the filename), and whether a source, a neighbouring tile or an
identical image appears in more than one split.

Needs only numpy and Pillow (or OpenCV). Nothing is written unless --save-csv is given.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except ImportError:  # pragma: no cover - fall back to OpenCV
    Image = None
    import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "valid", "validation", "test")
IMAGE_DIR_NAMES = ("images", "image", "imgs", "img")
MASK_DIR_NAMES = ("masks", "mask", "labels", "label", "annotations", "gt")

TILE_PATTERN = re.compile(
    r"^(?P<source>.*?)[_\-]r(?:ow)?[_\-]?(?P<row>\d+)[_\-]c(?:ol)?[_\-]?(?P<col>\d+)$", re.IGNORECASE
)
# CRACK500 crops: CRACK500_<date>_<time>_<x>_<y>; the photo is <date>_<time>.
CRACK500_PATTERN = re.compile(r"^(?P<source>crack500_\d{8}_\d{6})(?:_\d+)*$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #
def normalized_stem(path: Path) -> str:
    """Lower-case stem without role suffixes (_mask, _label, _image, _img)."""
    stem = path.stem.lower().strip()
    previous = None
    while stem != previous:
        previous = stem
        stem = re.sub(r"(?:[_\- ](?:mask|label|image|img|gt))$", "", stem)
    return stem


def parse_source(stem: str) -> tuple[str, int | None, int | None]:
    """Return (source image, tile row, tile column) inferred from a normalized stem."""
    match = TILE_PATTERN.match(stem)
    if match:
        return match.group("source") or "(no prefix)", int(match.group("row")), int(match.group("col"))
    match = CRACK500_PATTERN.match(stem)
    if match:
        return match.group("source"), None, None
    source = re.sub(r"(?:[_\-]\d+)+$", "", stem)
    return source or stem, None, None


def source_group(source: str, row: int | None) -> str:
    """Coarser group for the summary table: tiled sources stay, photo sets collapse (CRACK500_#_#)."""
    return source if row is not None else re.sub(r"\d+", "#", source)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def image_header(path: Path) -> tuple[int, int, str]:
    """(width, height, mode) without decoding the pixels when Pillow is available."""
    if Image is not None:
        with Image.open(path) as image:
            return image.size[0], image.size[1], image.mode
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    channels = 1 if array.ndim == 2 else array.shape[2]
    return array.shape[1], array.shape[0], f"{channels}ch-{array.dtype}"


def read_mask(path: Path) -> tuple[np.ndarray, str]:
    """(mask as stored, reduced to one channel; stored mode)."""
    if Image is not None:
        with Image.open(path) as image:
            mode = image.mode
            array = np.asarray(image if mode in ("L", "I;16", "I", "1", "P") else image.convert("L"))
        return array, mode
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    mode = f"{1 if array.ndim == 2 else array.shape[2]}ch-{array.dtype}"
    return (array if array.ndim == 2 else array[..., 0]), mode


def inspect_pair(job: tuple[str, str, Path, Path, bool]) -> dict[str, Any]:
    split, stem, image_path, mask_path, do_hash = job
    record: dict[str, Any] = {"split": split, "stem": stem, "image": image_path.name, "mask": mask_path.name}
    try:
        width, height, mode = image_header(image_path)
        mask, mask_mode = read_mask(mask_path)
        mask = mask.astype(np.int64, copy=False) if mask.dtype == bool else mask
        values = np.unique(mask)
        maximum = int(values.max())
        # Same rule as the earlier training scripts: 0/1 masks, else threshold at 128.
        threshold = 1 if maximum <= 1 else 128
        record.update(
            image_w=width, image_h=height, image_mode=mode, image_ext=image_path.suffix.lower(),
            mask_w=int(mask.shape[1]), mask_h=int(mask.shape[0]), mask_mode=mask_mode,
            mask_ext=mask_path.suffix.lower(), mask_dtype=str(mask.dtype),
            mask_min=int(values.min()), mask_max=maximum, mask_n_values=int(values.size),
            mask_values=" ".join(str(int(v)) for v in values[:6]) + (" ..." if values.size > 6 else ""),
            crack_pixels=int((mask >= threshold).sum()), pixels=int(mask.size),
            md5=hashlib.md5(image_path.read_bytes()).hexdigest() if do_hash else "",
            error="",
        )
    except Exception as exc:  # a broken file must not stop the report
        record["error"] = f"{type(exc).__name__}: {exc}"
    source, row, col = parse_source(stem)
    record.update(source=source, row=row, col=col, group=source_group(source, row))
    return record


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def first_existing(parent: Path, names: Iterable[str]) -> Path | None:
    children = {child.name.lower(): child for child in parent.iterdir() if child.is_dir()}
    return next((children[name] for name in names if name in children), None)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def print_layout(root: Path, depth: int = 3) -> None:
    def walk(folder: Path, level: int) -> None:
        entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        files = [p for p in entries if p.is_file()]
        directories = [p for p in entries if p.is_dir()]
        indent = "    " * level
        if files:
            extensions = Counter(p.suffix.lower() or "(none)" for p in files)
            summary = ", ".join(f"{count} x {ext}" for ext, count in extensions.most_common())
            examples = ", ".join(p.name for p in files[:2])
            print(f"{indent}[{len(files)} files: {summary}]  e.g. {examples}")
        for directory in directories:
            print(f"{indent}{directory.name}/")
            if level + 1 < depth:
                walk(directory, level + 1)

    print(f"{root}/")
    walk(root, 1)


def collect_pairs(split_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    """Pair images and masks of one split by normalized stem. Returns (pairs, problems)."""
    problems: list[str] = []
    image_dir = first_existing(split_dir, IMAGE_DIR_NAMES)
    mask_dir = first_existing(split_dir, MASK_DIR_NAMES)
    if image_dir is None or mask_dir is None:
        return [], [f"{split_dir.name}: no images/ + masks/ folders found"]
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for target, paths, role in ((images, list_images(image_dir), "image"), (masks, list_images(mask_dir), "mask")):
        for path in paths:
            key = normalized_stem(path)
            if key in target:
                problems.append(f"{split_dir.name}: duplicate {role} id {key}: {target[key].name}, {path.name}")
            else:
                target[key] = path
    only_images, only_masks = sorted(set(images) - set(masks)), sorted(set(masks) - set(images))
    if only_images:
        problems.append(f"{split_dir.name}: {len(only_images)} images without a mask, e.g. "
                        f"{[images[k].name for k in only_images[:3]]}")
    if only_masks:
        problems.append(f"{split_dir.name}: {len(only_masks)} masks without an image, e.g. "
                        f"{[masks[k].name for k in only_masks[:3]]}")
    return [(key, images[key], masks[key]) for key in sorted(set(images) & set(masks))], problems


# --------------------------------------------------------------------------- #
# Report helpers
# --------------------------------------------------------------------------- #
def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def table(header: list[str], rows: list[list[Any]]) -> None:
    cells = [[str(value) for value in row] for row in [header, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    for index, row in enumerate(cells):
        print("  " + "  ".join(
            value.ljust(width) if column == 0 else value.rjust(width)
            for column, (value, width) in enumerate(zip(row, widths))
        ))
        if index == 0:
            print("  " + "  ".join("-" * width for width in widths))


def percent(numerator: float, denominator: float, digits: int = 2) -> str:
    return f"{100.0 * numerator / denominator:.{digits}f}%" if denominator else "n/a"


def top(counter: Counter, limit: int = 6) -> str:
    text = ", ".join(f"{key} x{count}" for key, count in counter.most_common(limit))
    return text + (f", ... (+{len(counter) - limit} more)" if len(counter) > limit else "")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"),
                        help="Dataset folder holding train/ val/ test/ (default: $DATA_ROOT).")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Reader threads. Keep small on a login node (default: 4).")
    parser.add_argument("--max-per-split", type=int, default=0,
                        help="Read only the first N pairs of each split (0 = all).")
    parser.add_argument("--no-hash", action="store_true", help="Skip the exact-duplicate check (md5 of every image).")
    parser.add_argument("--max-sources", type=int, default=60, help="Rows shown in the per-source tables.")
    parser.add_argument("--save-csv", default="", help="Optional path for a per-tile CSV of everything measured.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root:
        print("Set DATA_ROOT or pass --data-root.", file=sys.stderr)
        return 2
    root = Path(args.data_root).expanduser()
    if not root.is_dir():
        print(f"Dataset folder not found: {root}", file=sys.stderr)
        return 2

    heading("1. FOLDER LAYOUT")
    print_layout(root)

    split_dirs = [child for name in SPLIT_NAMES for child in root.iterdir()
                  if child.is_dir() and child.name.lower() == name]
    if not split_dirs:
        print(f"\nNo split folders ({', '.join(SPLIT_NAMES)}) under {root}.", file=sys.stderr)
        return 1
    splits = [d.name for d in split_dirs]

    heading("2. IMAGE / MASK PAIRING")
    jobs: list[tuple[str, str, Path, Path, bool]] = []
    problems: list[str] = []
    for split_dir in split_dirs:
        pairs, split_problems = collect_pairs(split_dir)
        problems += split_problems
        if args.max_per_split:
            pairs = pairs[: args.max_per_split]
        jobs += [(split_dir.name, key, image, mask, not args.no_hash) for key, image, mask in pairs]
        print(f"  {split_dir.name}: {len(pairs)} image/mask pairs")
    print("  Pairing problems: " + ("none" if not problems else ""))
    for problem in problems:
        print(f"    ! {problem}")
    if args.max_per_split:
        print(f"  NOTE: --max-per-split {args.max_per_split}: the statistics below are for a subset.")

    print(f"\nReading {len(jobs)} pairs with {args.workers} threads ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        records = list(pool.map(inspect_pair, jobs, chunksize=64))
    failed = [r for r in records if r["error"]]
    records = [r for r in records if not r["error"]]
    if failed:
        print(f"  ! {len(failed)} unreadable pairs, e.g.")
        for record in failed[:5]:
            print(f"    {record['split']}/{record['image']}: {record['error']}")
    by_split = {split: [r for r in records if r["split"] == split] for split in splits}

    heading("3. FORMATS AND TILE SIZES")
    for split in splits:
        rows = by_split[split]
        print(f"  {split}:")
        print("    image ext : " + top(Counter(r["image_ext"] for r in rows)))
        print("    image mode: " + top(Counter(r["image_mode"] for r in rows)))
        print("    image size: " + top(Counter(f"{r['image_w']}x{r['image_h']}" for r in rows)))
        print("    mask ext  : " + top(Counter(r["mask_ext"] for r in rows)))
        print("    mask mode : " + top(Counter(f"{r['mask_mode']}/{r['mask_dtype']}" for r in rows)))
        print("    mask size : " + top(Counter(f"{r['mask_w']}x{r['mask_h']}" for r in rows)))
        mismatched = [r for r in rows if (r["image_w"], r["image_h"]) != (r["mask_w"], r["mask_h"])]
        print(f"    image/mask size mismatch: {len(mismatched)}"
              + (f", e.g. {mismatched[0]['image']}" if mismatched else ""))
    sizes = Counter((r["image_w"], r["image_h"]) for r in records)
    print(f"  Distinct tile sizes overall: {len(sizes)}"
          f" (min side {min(min(s) for s in sizes)}, max side {max(max(s) for s in sizes)})")
    not_32 = sum(count for (w, h), count in sizes.items() if w % 32 or h % 32)
    print(f"  Tiles whose width or height is not a multiple of 32: {not_32} ({percent(not_32, len(records))})")

    heading("4. MASK VALUES")
    print(f"  Global mask value range: [{min(r['mask_min'] for r in records)}, {max(r['mask_max'] for r in records)}]")

    def value_kind(record: dict[str, Any]) -> str:
        values = record["mask_values"]
        if values in ("0", "0 255", "255", "0 1", "1"):
            return {"0": "all zero (no crack)", "0 255": "binary {0,255}", "255": "all 255",
                    "0 1": "binary {0,1}", "1": "all 1"}[values]
        return "NOT binary (>2 grey levels, e.g. JPEG or anti-aliased)"

    for split in splits:
        print(f"  {split}: " + top(Counter(value_kind(r) for r in by_split[split]), 8))
    nonbinary = [r for r in records if value_kind(r).startswith("NOT")]
    if nonbinary:
        print(f"  Non-binary masks: {len(nonbinary)}; by mask extension: {top(Counter(r['mask_ext'] for r in nonbinary))}")
        for record in nonbinary[:3]:
            print(f"    e.g. {record['mask']}: {record['mask_n_values']} levels: {record['mask_values']}")
    print("  Binarization used below: mask >= 1 if the mask max is <= 1, else mask >= 128.")

    heading("5. SAMPLES AND CLASS BALANCE PER SPLIT")
    buckets = [(0.0, "0"), (0.005, "<0.5%"), (0.01, "0.5-1%"), (0.02, "1-2%"), (0.05, "2-5%"), (1.01, ">5%")]

    def bucket(fraction: float) -> str:
        if fraction == 0:
            return "0"
        return next(label for limit, label in buckets[1:] if fraction < limit)

    rows = []
    for split in [*splits, "ALL"]:
        subset = records if split == "ALL" else by_split[split]
        if not subset:
            continue
        fractions = np.array([r["crack_pixels"] / r["pixels"] for r in subset])
        positive = fractions[fractions > 0]
        crack, total = sum(r["crack_pixels"] for r in subset), sum(r["pixels"] for r in subset)
        rows.append([
            split, len(subset), percent(len(subset), len(records), 1),
            f"{len(subset) - len(positive)} ({percent(len(subset) - len(positive), len(subset), 1)})",
            percent(crack, total, 3), f"{100 * fractions.mean():.3f}%",
            f"{100 * np.median(positive):.3f}%" if positive.size else "n/a",
            f"1:{(total - crack) / crack:.0f}" if crack else "n/a",
        ])
    table(["split", "tiles", "share", "empty tiles", "crack px % (pooled)", "mean per tile",
           "median (crack tiles)", "crack:background"], rows)
    print("\n  Tiles by crack-pixel fraction:")
    labels = [label for _, label in buckets]
    table(["split", *labels], [
        [split, *[Counter(bucket(r["crack_pixels"] / r["pixels"]) for r in by_split[split])[label] for label in labels]]
        for split in splits
    ])

    heading("6. SOURCE IMAGES (inferred from filenames)")
    print("  Rule: <source>_r####_c#### -> <source>; CRACK500_<date>_<time>_<x>_<y> -> the photo;")
    print("  otherwise the stem without its trailing numbers. Check a few rows against what you know.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    rows = []
    for group, members in sorted(groups.items(), key=lambda item: -len(item[1])):
        counts = Counter(r["split"] for r in members)
        crack, total = sum(r["crack_pixels"] for r in members), sum(r["pixels"] for r in members)
        rows.append([group, len({r["source"] for r in members}), len(members),
                     *[counts.get(split, 0) for split in splits], percent(crack, total, 3),
                     top(Counter(f"{r['image_w']}x{r['image_h']}" for r in members), 1)])
    table(["source group", "sources", "tiles", *splits, "crack px %", "tile size"], rows[: args.max_sources])
    if len(rows) > args.max_sources:
        print(f"  ... {len(rows) - args.max_sources} more groups (raise --max-sources)")

    heading("7. LEAKAGE CHECK: THE SAME SOURCE IN MORE THAN ONE SPLIT")
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sources[record["source"]].append(record)
    shared = {s: m for s, m in sources.items() if len({r["split"] for r in m}) > 1}
    print(f"  Sources: {len(sources)} | in more than one split: {len(shared)} ({percent(len(shared), len(sources), 1)})")
    for split in splits:
        subset = by_split[split]
        in_shared = sum(1 for r in subset if r["source"] in shared)
        print(f"    {split}: {in_shared}/{len(subset)} tiles ({percent(in_shared, len(subset), 1)}) "
              "come from a source that is also in another split")
    if shared:
        rows = [[s, len(m), *[sum(1 for r in m if r["split"] == split) for split in splits]]
                for s, m in sorted(shared.items(), key=lambda item: -len(item[1]))]
        table(["shared source", "tiles", *splits], rows[: args.max_sources])
        if len(rows) > args.max_sources:
            print(f"  ... {len(rows) - args.max_sources} more shared sources")

    # Neighbouring tiles of one orthomosaic in different splits: they share image content if the
    # tiling overlapped, and near-identical texture/illumination even if it did not.
    train_name = next((s for s in splits if s.lower() == "train"), None)
    tiled = [r for r in records if r["row"] is not None]
    if train_name and tiled:
        print("\n  Grid neighbours (tiles with an r/c index only): a val/test tile counts when one of its")
        print("  8 neighbours (row +-1, col +-1) in the same source is a TRAIN tile.")
        train_cells = {(r["source"], r["row"], r["col"]) for r in tiled if r["split"] == train_name}
        rows = []
        for split in splits:
            if split == train_name:
                continue
            subset = [r for r in tiled if r["split"] == split]
            touching = sum(
                1 for r in subset
                if any((r["source"], r["row"] + dr, r["col"] + dc) in train_cells
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
            )
            rows.append([split, len(subset), touching, percent(touching, len(subset), 1)])
        table(["split", "tiled tiles", "with a train neighbour", "share"], rows)
        print("  Whether neighbours share pixels depends on the tiling stride (overlap), which the")
        print("  filenames do not record; say what stride/overlap was used when the tiles were cut.")

    if not args.no_hash:
        by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_hash[record["md5"]].append(record)
        duplicates = [m for m in by_hash.values() if len(m) > 1]
        across = [m for m in duplicates if len({r["split"] for r in m}) > 1]
        print(f"\n  Byte-identical images: {len(duplicates)} groups, {len(across)} of them across splits")
        for members in across[:5]:
            print("    " + " == ".join(f"{r['split']}/{r['image']}" for r in members[:3]))

    manifests = sorted(root.glob("*.csv"))
    if manifests:
        heading("8. SPLIT MANIFEST")
        for manifest in manifests:
            with manifest.open(newline="", encoding="utf-8", errors="replace") as handle:
                lines = list(csv.reader(handle))
            print(f"  {manifest.name}: {max(0, len(lines) - 1)} rows; columns: {lines[0] if lines else []}")
            if lines and "seed" in lines[0]:
                print(f"    seed(s): {sorted({row[lines[0].index('seed')] for row in lines[1:] if row})}")
            if lines and "split" in lines[0]:
                print(f"    rows per split: {dict(Counter(row[lines[0].index('split')] for row in lines[1:] if row))}")

    heading("SUMMARY")
    crack, total = sum(r["crack_pixels"] for r in records), sum(r["pixels"] for r in records)
    print(f"  {len(records)} tiles: " + ", ".join(f"{split} {len(by_split[split])}" for split in splits))
    print(f"  Tile sizes: {top(Counter({f'{w}x{h}': n for (w, h), n in sizes.items()}), 4)}")
    print(f"  Crack pixels: {percent(crack, total, 3)} of all pixels")
    print(f"  Non-binary masks: {len(nonbinary)} | pairing problems: {len(problems)} | unreadable: {len(failed)}")
    print(f"  Sources in more than one split: {len(shared)} of {len(sources)}"
          + ("  -> the split is by TILE, not by source image" if shared else "  -> split is by source"))

    if args.save_csv:
        fields = [k for k in records[0] if k != "error"]
        with open(args.save_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"  Per-tile CSV written to {args.save_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
