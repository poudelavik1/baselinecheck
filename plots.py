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
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Bbox
import pandas as pd

import results as res
from models.registry import FAMILIES, REGISTRY, family_label

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
    headers = [axes[0].text((start + end) / 2, 1.02, family_label(family), ha="center", va="bottom", fontsize=9,
                            fontweight="bold", transform=axes[0].get_xaxis_transform()) for family, start, end in spans]
    figure.canvas.draw()   # a header that would overlap an earlier one (narrow one-bar families) moves up a level
    renderer = figure.canvas.get_renderer()
    for i, header in enumerate(headers):
        for level in (1.02, 1.09, 1.16):
            header.set_y(level)
            box = header.get_window_extent(renderer)
            if not any(box.overlaps(h.get_window_extent(renderer)) for h in headers[:i]):
                break
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


BUBBLE_AREA = 10.0   # scatter marker area (pt^2) per million parameters


def _marker_radius(marker: str, area: float) -> float:
    """Radius (points) of the circle around a scatter marker of this area, edge line included."""
    shape = MarkerStyle(marker)
    vertices = shape.get_path().transformed(shape.get_transform()).vertices
    return float(np.hypot(vertices[:, 0], vertices[:, 1]).max() * np.sqrt(area)) + 0.5


def _log_ticks(low: float, high: float) -> list[float]:
    """Round tick values for a log axis between low and high, about 5-9 of them."""
    decades = range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
    for steps in ((1, 1.5, 2, 3, 4, 5, 6, 8), (1, 2, 3, 5), (1, 2, 5), (1,)):
        ticks = [s * 10.0 ** d for d in decades for s in steps if low <= s * 10.0 ** d <= high]
        if len(ticks) <= 9:
            return ticks
    return ticks


def _size_steps(params: pd.Series) -> list[float]:
    """Three round parameter counts spanning the models, for the bubble-size legend."""
    low, high = float(params.min()), float(params.max())
    step = 10 ** np.floor(np.log10(high)) / 2
    return sorted({max(step, round(v / step) * step) for v in np.linspace(low, high, 3)})


def _hits_circle(box: Bbox, cx: float, cy: float, radius: float) -> bool:
    nearest_x, nearest_y = np.clip(cx, box.x0, box.x1), np.clip(cy, box.y0, box.y1)
    return (nearest_x - cx) ** 2 + (nearest_y - cy) ** 2 < radius ** 2


def _place_labels(axis: plt.Axes, points: list[tuple[str, float, float, float, float]], fontsize: float = 7.5) -> None:
    """Label each point beside its marker without covering another label, a marker or an error bar.

    points: (text, x, y, yerr, marker radius in points) in data coordinates. Tries spots around the marker at
    twelve angles (right first) and growing distances (further out with a leader line), and takes the first free
    one; if none is free, the one with the fewest collisions. Axis limits must be final before this is called.
    """
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    px = figure.dpi / 72.0
    frame = axis.get_window_extent(renderer)
    centres = axis.transData.transform([(x, y) for _, x, y, _, _ in points])
    circles = [(cx, cy, radius * px) for (cx, cy), (*_, radius) in zip(centres, points)]
    bars = []
    for (cx, _), (_, x, y, err, _) in zip(centres, points):
        (_, low), (_, high) = axis.transData.transform([(x, y - err), (x, y + err)])
        bars.append(Bbox.from_extents(cx - 2.5 * px, low, cx + 2.5 * px, high))
    angles = np.radians([0, 30, -30, 60, -60, 180, 150, -150, 120, -120, 90, -90])
    crowding = [sum(np.hypot(*(c - o)) < 90 * px for o in centres) for c in centres]
    placed: list[Bbox] = []
    for i in sorted(range(len(points)), key=lambda i: -crowding[i]):
        text, x, y, _, radius = points[i]
        best = None
        for extra in (0, 10, 20, 30, 40):   # points beyond the marker edge
            for angle in angles:
                dx, dy = np.cos(angle), np.sin(angle)
                ha = "left" if dx > 0.3 else "right" if dx < -0.3 else "center"
                va = "bottom" if dy > 0.3 else "top" if dy < -0.3 else "center"
                offset = (dx * (radius + 2 + extra), dy * (radius + 2 + extra))
                trial = axis.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va,
                                      fontsize=fontsize)
                box = trial.get_window_extent(renderer).padded(1.5 * px)
                trial.remove()
                inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
                cost = 100 * (not inside) + sum(_hits_circle(box, *c) for c in circles) \
                    + sum(box.overlaps(b) for b in bars + placed)
                if best is None or cost < best[0]:
                    best = (cost, offset, ha, va, extra, box)
                if cost == 0:
                    break
            if best[0] == 0:
                break
        _, offset, ha, va, extra, box = best
        leader = None
        if extra > 0:
            anchor = ({"left": 0.0, "right": 1.0}.get(ha, 0.5), {"bottom": 0.0, "top": 1.0}.get(va, 0.5))
            leader = {"arrowstyle": "-", "color": "0.45", "linewidth": 0.6, "shrinkA": 1, "shrinkB": radius,
                      "relpos": anchor}
        axis.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                      arrowprops=leader)
        placed.append(box)


def fig5_efficiency(summary: pd.DataFrame, out: Path, split: str = "val") -> Path | None:
    """Fig. 5: IoU against inference time; bubble area = number of parameters."""
    table = summary.dropna(subset=["time_ms", "params_m"]).sort_values("params_m", ascending=False)
    if table.empty:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    points = []
    for _, row in table.iterrows():   # largest first, so small bubbles stay on top
        look = style(row["model"])
        area = BUBBLE_AREA * row["params_m"]
        axis.scatter(row["time_ms"], row["iou"], s=area, color=look["color"], marker=look["marker"], alpha=0.8,
                     edgecolor="black", linewidth=0.6, zorder=2)
        axis.errorbar(row["time_ms"], row["iou"], yerr=row["iou_std"], color="black", linewidth=0.7, capsize=2,
                      zorder=3)
        points.append((label(row["model"]), row["time_ms"], row["iou"], row["iou_std"],
                       _marker_radius(look["marker"], area)))

    device = next((d for d in table["device"] if d), "")
    axis.set(xlabel="Inference time per 512×512 tile (ms, log scale)", ylabel=f"{split} IoU")
    if device:
        axis.set_title(f"Timed on {device}", fontsize=8, loc="left")
    low, high = float(table["time_ms"].min()), float(table["time_ms"].max())
    axis.set_xscale("log")
    axis.set_xlim(low / 1.18, high * 1.18)
    axis.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(_log_ticks(*axis.get_xlim())))
    axis.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    bottom, top = float((table["iou"] - table["iou_std"]).min()), float((table["iou"] + table["iou_std"]).max())
    axis.set_ylim(bottom - 0.08 * (top - bottom), top + 0.08 * (top - bottom))

    families = [f for f in FAMILIES if (table["family"] == f).any()]
    shapes = [Line2D([], [], marker=FAMILY_MARKERS[f], linestyle="none", markersize=7, markerfacecolor="white",
                     markeredgecolor="black", markeredgewidth=0.8) for f in families]
    family_legend = axis.legend(shapes, [family_label(f) for f in families], title="Family", loc="upper left",
                                bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=7.5, title_fontsize=8, alignment="left", borderaxespad=0)
    axis.add_artist(family_legend)
    family_legend.set_clip_on(False)   # else bbox_inches="tight" treats it as inside the axes and crops it
    figure.canvas.draw()
    below = family_legend.get_window_extent().transformed(axis.transAxes.inverted()).y0 - 0.06
    sizes = _size_steps(table["params_m"])
    bubbles = [Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(BUBBLE_AREA * v),
                      markerfacecolor="lightgray", markeredgecolor="black", markeredgewidth=0.6) for v in sizes]
    axis.legend(bubbles, [f"{v:g} M" for v in sizes], title="Parameters", loc="upper left",
                bbox_to_anchor=(1.03, below), frameon=False, fontsize=7.5, title_fontsize=8, alignment="left",
                borderaxespad=0, labelspacing=1.3, handletextpad=1.0)

    _place_labels(axis, points)
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
