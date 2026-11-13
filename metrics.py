"""Segmentation metrics on numpy arrays. No torch import: usable on a laptop.

Probabilities are stored as uint8 (0..255). A threshold index k means ``prob_u8 >= k``,
i.e. probability >= k / 255. Dataset-level ("pooled") scores are computed from pixel
counts summed over all tiles; per-tile scores are saved separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

LEVELS = 256


def safe_divide(numerator: float, denominator: float, empty: float = 0.0) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else empty


def scores(tp: float, fp: float, fn: float) -> dict[str, float]:
    """IoU, F1 (= Dice), precision and recall from pixel counts."""
    return {
        "iou": safe_divide(tp, tp + fp + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
    }


def tile_scores(tp: float, fp: float, fn: float) -> dict[str, float]:
    """Per-tile scores; a tile with no crack and no prediction scores 1."""
    if tp + fp + fn == 0:
        return {"iou": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
    return scores(tp, fp, fn)


# --------------------------------------------------------------------------- #
# Threshold curves from histograms
# --------------------------------------------------------------------------- #
def histograms(prob_u8: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(histogram of crack pixels, histogram of background pixels) over the 256 levels."""
    gt = gt.astype(bool)
    positive = np.bincount(prob_u8[gt].ravel(), minlength=LEVELS).astype(np.int64)
    negative = np.bincount(prob_u8[~gt].ravel(), minlength=LEVELS).astype(np.int64)
    return positive, negative


def curves(positive: np.ndarray, negative: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and IoU at the 255 thresholds k/255, k = 1..255."""
    tp = np.cumsum(positive[::-1])[::-1][1:].astype(np.float64)   # prob_u8 >= k
    fp = np.cumsum(negative[::-1])[::-1][1:].astype(np.float64)
    fn = positive.sum() - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
    return {"threshold": np.arange(1, LEVELS) / (LEVELS - 1.0), "precision": precision, "recall": recall,
            "f1": f1, "iou": iou}


def average_precision(curve: dict[str, np.ndarray]) -> float:
    """Area under the precision-recall curve (step interpolation, as in scikit-learn)."""
    recall = np.concatenate([curve["recall"], [0.0]])      # recall falls as the threshold rises
    precision = np.concatenate([curve["precision"], [1.0]])
    return float(np.sum((recall[:-1] - recall[1:]) * precision[:-1]))


def best_threshold_index(curve: dict[str, np.ndarray]) -> int:
    """Threshold index k (1..255) with the highest pooled F1."""
    return int(np.argmax(curve["f1"])) + 1


# --------------------------------------------------------------------------- #
# Tolerance F1 and clDice
# --------------------------------------------------------------------------- #
def distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest True pixel (inf if there is none)."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return ndimage.distance_transform_edt(~mask).astype(np.float32)


def tolerance_counts(pred: np.ndarray, gt: np.ndarray, tolerances: list[int]) -> dict[str, int]:
    """Counts for tolerance precision/recall.

    A predicted pixel is correct if a labelled pixel lies within `t` px; a labelled pixel
    is found if a predicted pixel lies within `t` px. t = 0 gives the strict scores.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    to_gt = distance_to(gt)[pred]
    to_pred = distance_to(pred)[gt]
    counts: dict[str, int] = {"n_pred": int(pred.sum()), "n_gt": int(gt.sum())}
    for tolerance in tolerances:
        counts[f"pred_hit_{tolerance}"] = int((to_gt <= tolerance).sum())
        counts[f"gt_hit_{tolerance}"] = int((to_pred <= tolerance).sum())
    return counts


def tolerance_scores(counts: dict[str, float], tolerance: int) -> dict[str, float]:
    precision = safe_divide(counts[f"pred_hit_{tolerance}"], counts["n_pred"])
    recall = safe_divide(counts[f"gt_hit_{tolerance}"], counts["n_gt"])
    return {"precision": precision, "recall": recall, "f1": safe_divide(2 * precision * recall, precision + recall)}


def skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return skeletonize(mask.astype(bool)) if mask.any() else np.zeros(mask.shape, dtype=bool)


def cldice_counts(pred: np.ndarray, gt: np.ndarray, gt_skeleton: np.ndarray | None = None) -> dict[str, int]:
    """Counts for clDice (Shit et al., 2021): skeleton of one mask inside the other mask."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    pred_skeleton = skeleton(pred)
    gt_skeleton = skeleton(gt) if gt_skeleton is None else gt_skeleton
    return {
        "skel_pred_in_gt": int((pred_skeleton & gt).sum()), "skel_pred": int(pred_skeleton.sum()),
        "skel_gt_in_pred": int((gt_skeleton & pred).sum()), "skel_gt": int(gt_skeleton.sum()),
    }


def cldice_score(counts: dict[str, float]) -> float:
    topology_precision = safe_divide(counts["skel_pred_in_gt"], counts["skel_pred"])
    topology_sensitivity = safe_divide(counts["skel_gt_in_pred"], counts["skel_gt"])
    return safe_divide(2 * topology_precision * topology_sensitivity, topology_precision + topology_sensitivity)


# --------------------------------------------------------------------------- #
# One tile, all threshold-dependent metrics
# --------------------------------------------------------------------------- #
def tile_record(prob_u8: np.ndarray, gt: np.ndarray, threshold_index: int, tolerances: list[int],
                gt_skeleton: np.ndarray | None = None) -> dict[str, Any]:
    """Counts and scores of one tile at one threshold."""
    gt = gt.astype(bool)
    pred = prob_u8 >= threshold_index
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    record: dict[str, Any] = {"height": gt.shape[0], "width": gt.shape[1], "tp": tp, "fp": fp, "fn": fn}
    record.update(tile_scores(tp, fp, fn))
    record.update(tolerance_counts(pred, gt, tolerances))
    record.update(cldice_counts(pred, gt, gt_skeleton))
    return record


def pooled_metrics(records: list[dict[str, Any]], tolerances: list[int]) -> dict[str, float]:
    """Dataset-level metrics from summed per-tile counts, plus the mean of per-tile IoU/F1."""
    def total(key: str) -> float:
        return float(sum(record[key] for record in records))

    result = scores(total("tp"), total("fp"), total("fn"))
    result["tile_mean_iou"] = float(np.mean([record["iou"] for record in records]))
    result["tile_mean_f1"] = float(np.mean([record["f1"] for record in records]))
    keys = ["n_pred", "n_gt", *[f"pred_hit_{t}" for t in tolerances], *[f"gt_hit_{t}" for t in tolerances]]
    sums = {key: total(key) for key in keys}
    for tolerance in tolerances:
        tolerant = tolerance_scores(sums, tolerance)
        result[f"tol_f1_{tolerance}"] = tolerant["f1"]
        result[f"tol_precision_{tolerance}"] = tolerant["precision"]
        result[f"tol_recall_{tolerance}"] = tolerant["recall"]
    result["cldice"] = cldice_score({key: total(key) for key in ("skel_pred_in_gt", "skel_pred", "skel_gt_in_pred", "skel_gt")})
    return result
