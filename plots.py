"""Thesis figures. Figures 1-7 need only the result CSV/JSON files (no GPU, no dataset).

Every model keeps one colour and every family one marker and line style in all figures.
Each figure is saved as 300 dpi PNG and as PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

import results as res
from models.registry import FAMILIES, REGISTRY

# Colour-blind-safe colours (Paul Tol "muted" and "bright", Okabe-Ito). One per registered model,
# assigned in registry order so a model keeps its colour when others are added after it.
PALETTE = ["#332288", "#CC6677", "#117733", "#DDCC77", "#88CCEE", "#882255", "#44AA99", "#AA4499", "#999933",
           "#E69F00", "#0072B2", "#D55E00", "#000000", "#EE6677", "#228833", "#4477AA", "#AA3377", "#66CCEE",
           "#CCBB44", "#777777"]
FAMILY_MARKERS = dict(zip(FAMILIES, ["o", "s", "D", "^", "P", "*"]))
FAMILY_LINES = dict(zip(FAMILIES, ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]))


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 100, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def color(model: str) -> str:
    names = list(REGISTRY)
    index = names.index(model) if model in names else len(names) + sum(map(ord, model))
    return PALETTE[index % len(PALETTE)]


def style(model: str) -> dict[str, Any]:
    family = res.info(model)["family"]
    return {"color": color(model), "marker": FAMILY_MARKERS[family], "linestyle": FAMILY_LINES[family]}


def label(model: str) -> str:
    return res.info(model)["label"]


def save(figure: plt.Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / f"{name}.png", bbox_inches="tight")
    figure.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)
    return out / f"{name}.png"


def by_family(models: list[str]) -> list[str]:
    """Models ordered by family (FAMILIES order), registry order within a family."""
    return [m for family in FAMILIES for m in models if res.info(m)["family"] == family]


# --------------------------------------------------------------------------- #
def fig1_bars(summary: pd.DataFrame, out: Path, split: str = "val") -> Path:
    """Fig. 1: IoU and F1 per model, mean +- std over seeds, grouped by family."""
    order = by_family(list(summary["model"]))
    table = summary.set_index("model").loc[order]
    figure, axes = plt.subplots(2, 1, figsize=(max(7.0, 0.55 * len(order) + 2), 6.4), sharex=True)
    positions, ticks, position, previous = [], [], 0.0, None
    spans: list[tuple[str, float, float]] = []
    for model in order:
        family = res.info(model)["family"]
        if previous is not None and family != previous:
            position += 0.6
        if not spans or spans[-1][0] != family:
            spans.append((family, position, position))
        spans[-1] = (family, spans[-1][1], position)
        positions.append(position)
        ticks.append(label(model))
        position, previous = position + 1.0, family
    for axis, metric, title in zip(axes, ("iou", "f1"), ("IoU", "F1 (Dice)")):
        values, errors = table[metric].to_numpy(), table[f"{metric}_std"].to_numpy()
        axis.bar(positions, values, yerr=errors, capsize=3, color=[color(m) for m in order], edgecolor="black",
                 linewidth=0.5, error_kw={"linewidth": 0.9})
        for x, value, error in zip(positions, values, errors):
            axis.text(x, value + error + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        low = max(0.0, float(np.nanmin(values - errors)) - 0.08)
        axis.set_ylim(low, min(1.0, float(np.nanmax(values + errors)) + 0.08))
        axis.set_ylabel(f"{split} {title}")
        axis.grid(axis="x", visible=False)
    for family, start, end in spans:
        axes[0].text((start + end) / 2, 1.02, family, ha="center", va="bottom", fontsize=9, fontweight="bold",
                     transform=axes[0].get_xaxis_transform())
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(ticks, rotation=40, ha="right")
    return save(figure, out, f"fig1_{split}_iou_f1_bars")


def fig2_training(results: res.Results, out: Path) -> list[Path]:
    """Fig. 2: train/val loss and IoU against epoch, one panel per model, mean line + std band."""
    models = by_family(results.models)
    models = [m for m in models if res.history_band(results, m, "val_iou") is not None]
    if not models:
        return []
    paths = []
    for quantity, name in (("loss", "Loss"), ("iou", "IoU")):
        columns = min(4, len(models))
        rows = int(np.ceil(len(models) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.5 * rows), sharex=False, squeeze=False)
        for axis, model in zip(axes.ravel(), models):
            for part, line, alpha in (("train", "--", 0.15), ("val", "-", 0.25)):
                band = res.history_band(results, model, f"{part}_{quantity}")
                if band is None:
                    continue
                axis.plot(band.index, band["mean"], linestyle=line, color=color(model), linewidth=1.4, label=part)
                axis.fill_between(band.index, band["mean"] - band["std"], band["mean"] + band["std"],
                                  color=color(model), alpha=alpha, linewidth=0)
            axis.set_title(label(model), fontsize=9)
            axis.legend(frameon=False, fontsize=7)
        for axis in axes.ravel()[len(models):]:
            axis.set_visible(False)
        for axis in axes[-1]:
            axis.set_xlabel("Epoch")
        for axis in axes[:, 0]:
            axis.set_ylabel(name)
        figure.tight_layout()
        paths.append(save(figure, out, f"fig2_training_{quantity}"))
    return paths


def fig3_pr(results: res.Results, summary: pd.DataFrame, out: Path, split: str = "val") -> Path | None:
    """Fig. 3: precision-recall curves of all models, AP in the legend."""
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    ap = summary.set_index("model")["ap"]
    drawn = 0
    for model in sorted(results.models, key=lambda m: -ap.get(m, 0)):
        curve = res.mean_curve(results, model, split)
        if curve is None:
            continue
        axis.plot(curve["recall"], curve["precision"], color=color(model), linestyle=style(model)["linestyle"],
                  linewidth=1.5, label=f"{label(model)} (AP {ap.get(model, float('nan')):.3f})")
        drawn += 1
    if not drawn:
        plt.close(figure)
        return None
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    axis.legend(frameon=False, loc="lower left", ncol=1 if drawn <= 10 else 2)
    return save(figure, out, f"fig3_{split}_precision_recall")


def fig4_f1_threshold(results: res.Results, summary: pd.DataFrame, out: Path, split: str = "val") -> Path | None:
    """Fig. 4: F1 against threshold per model; the marker is the threshold chosen on validation."""
    figure, axis = plt.subplots(figsize=(6.8, 5.0))
    thresholds = summary.set_index("model")["threshold"]
    drawn = 0
    for model in by_family(results.models):
        curve = res.mean_curve(results, model, split)
        if curve is None:
            continue
        look = style(model)
        axis.plot(curve["threshold"], curve["f1"], color=look["color"], linestyle=look["linestyle"], linewidth=1.4,
                  label=label(model))
        chosen = float(thresholds.get(model, np.nan))
        if np.isfinite(chosen):
            axis.plot(chosen, np.interp(chosen, curve["threshold"], curve["f1"]), marker=look["marker"],
                      color=look["color"], markeredgecolor="black", markersize=8, linestyle="none")
        drawn += 1
    if not drawn:
        plt.close(figure)
        return None
    axis.set(xlabel="Probability threshold", ylabel=f"{split} F1", xlim=(0, 1))
    axis.legend(frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    return save(figure, out, f"fig4_{split}_f1_vs_threshold")


def fig5_efficiency(summary: pd.DataFrame, out: Path, split: str = "val") -> Path | None:
    """Fig. 5: IoU against inference time; bubble area = number of parameters."""
    table = summary.dropna(subset=["time_ms", "params_m"])
    if table.empty:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for _, row in table.iterrows():
        look = style(row["model"])
        axis.scatter(row["time_ms"], row["iou"], s=18 * row["params_m"], color=look["color"], marker=look["marker"],
                     alpha=0.75, edgecolor="black", linewidth=0.6)
        axis.errorbar(row["time_ms"], row["iou"], yerr=row["iou_std"], color="black", linewidth=0.7, capsize=2)
        axis.annotate(label(row["model"]), (row["time_ms"], row["iou"]), xytext=(6, 5), textcoords="offset points",
                      fontsize=7.5)
    for size in (5, 25, 50):
        axis.scatter([], [], s=18 * size, color="lightgray", edgecolor="black", linewidth=0.6, label=f"{size} M params")
    for family in FAMILIES:
        if (table["family"] == family).any():
            axis.scatter([], [], marker=FAMILY_MARKERS[family], color="white", edgecolor="black", s=50, label=family)
    device = next((d for d in table["device"] if d), "")
    axis.set(xlabel="Inference time per 512x512 tile (ms, log scale)", ylabel=f"{split} IoU")
    if device:
        axis.set_title(f"Timed on {device}", fontsize=8, loc="left")
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    axis.tick_params(axis="x", which="minor", labelsize=8)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), labelspacing=1.6, fontsize=7.5,
                borderpad=1.0)
    return save(figure, out, f"fig5_{split}_iou_vs_time")


def fig6_per_tile(per_tile: pd.DataFrame, out: Path, split: str = "val") -> Path | None:
    """Fig. 6: distribution of per-tile IoU per model (violin with the quartile box inside)."""
    if per_tile.empty:
        return None
    order = list(per_tile.median().sort_values(ascending=False).index)
    figure, axis = plt.subplots(figsize=(max(6.5, 0.6 * len(order) + 2), 4.6))
    values = [per_tile[m].to_numpy() for m in order]
    if len(per_tile) >= 5:
        parts = axis.violinplot(values, positions=range(len(order)), widths=0.85, showextrema=False)
        for body, model in zip(parts["bodies"], order):
            body.set_facecolor(color(model))
            body.set_edgecolor("black")
            body.set_alpha(0.55)
    boxes = axis.boxplot(values, positions=range(len(order)), widths=0.22, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "linewidth": 1.3})
    for patch, model in zip(boxes["boxes"], order):
        patch.set_facecolor(color(model))
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([label(m) for m in order], rotation=40, ha="right")
    axis.set(ylabel=f"Per-tile {split} IoU", ylim=(0, 1))
    axis.grid(axis="x", visible=False)
    return save(figure, out, f"fig6_{split}_per_tile_iou")


def fig7_tolerance(summary: pd.DataFrame, out: Path, split: str = "val") -> Path | None:
    """Fig. 7: F1 against the pixel tolerance, one line per model."""
    tolerances = sorted(int(c.split("_")[-1]) for c in summary.columns if c.startswith("tol_f1_") and not c.endswith("_std"))
    if not tolerances:
        return None
    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    for model in by_family(list(summary["model"])):
        row = summary.set_index("model").loc[model]
        look = style(model)
        axis.errorbar(tolerances, [row[f"tol_f1_{t}"] for t in tolerances], yerr=[row[f"tol_f1_{t}_std"] for t in tolerances],
                      color=look["color"], marker=look["marker"], linestyle=look["linestyle"], linewidth=1.4,
                      markersize=5, capsize=2, label=label(model))
    axis.set(xlabel="Tolerance (px)", ylabel=f"{split} F1", xticks=tolerances)
    axis.legend(frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    return save(figure, out, f"fig7_{split}_f1_vs_tolerance")


def fig8_qualitative(rows: list[dict[str, Any]], model_names: list[str], out: Path, split: str = "val") -> Path:
    """Fig. 8: rows = example tiles, columns = input | ground truth | each model.

    Each row is {"category", "name", "image" (RGB uint8), "gt" (bool), "preds": {model: bool mask}}.
    Overlay: true positive green, false positive red, false negative blue.
    """
    columns = 2 + len(model_names)
    figure, axes = plt.subplots(len(rows), columns, figsize=(1.9 * columns, 1.95 * len(rows) + 0.3), squeeze=False)
    green, red, blue = np.array([0, 158, 115]), np.array([213, 94, 0]), np.array([0, 114, 178])   # Okabe-Ito
    for row_axes, row in zip(axes, rows):
        image, gt = row["image"], row["gt"].astype(bool)
        gray = np.repeat((0.35 * image.mean(axis=2, keepdims=True) + 150).clip(0, 255), 3, axis=2).astype(np.uint8)
        row_axes[0].imshow(image)
        row_axes[0].set_ylabel(row["category"], fontsize=9)
        truth = gray.copy()
        truth[gt] = (0, 0, 0)
        row_axes[1].imshow(truth)
        for axis, model in zip(row_axes[2:], model_names):
            pred = row["preds"][model].astype(bool)
            overlay = gray.copy()
            overlay[pred & gt], overlay[pred & ~gt], overlay[~pred & gt] = green, red, blue
            axis.imshow(overlay)
            union = (pred | gt).sum()
            axis.text(0.03, 0.97, f"IoU {(pred & gt).sum() / union:.2f}" if union else "IoU -", fontsize=7,
                      transform=axis.transAxes, va="top", ha="left",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
        for axis in row_axes:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
    for axis, title in zip(axes[0], ["Input", "Ground truth", *[label(m) for m in model_names]]):
        axis.set_title(title, fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in (green, red, blue)]
    figure.legend(handles, ["True positive", "False positive", "False negative"], loc="lower center", ncol=3,
                  frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.2, h_pad=0.4)
    return save(figure, out, f"fig8_{split}_qualitative")


def make_figures(results: res.Results, out: Path, split: str = "val") -> list[Path]:
    """Figures 1-7 from result files. Missing inputs skip a figure instead of failing."""
    setup()
    summary = res.summary_table(results, split)
    paths: list[Path] = []
    if summary.empty:
        print(f"[figures] no {split} results yet")
        return paths
    steps = [
        lambda: fig1_bars(summary, out, split), lambda: fig2_training(results, out),
        lambda: fig3_pr(results, summary, out, split), lambda: fig4_f1_threshold(results, summary, out, split),
        lambda: fig5_efficiency(summary, out, split), lambda: fig6_per_tile(res.per_tile_iou(results, split), out, split),
        lambda: fig7_tolerance(summary, out, split),
    ]
    for number, step in enumerate(steps, start=1):
        try:
            made = step()
            paths += made if isinstance(made, list) else [made] if made else []
        except Exception as exc:
            print(f"[figures] figure {number} failed: {type(exc).__name__}: {exc}")
    return paths
