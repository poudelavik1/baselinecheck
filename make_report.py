"""Tables, ranking, significance tests and figures from the evaluated runs.

    python make_report.py                          # figures 1-7 + tables (no GPU, no dataset needed)
    python make_report.py --qualitative            # also figure 8 (needs the dataset and checkpoints)
    python make_report.py --previous ~/swinunet ~/old_runs    # add "previous run" reference rows

Model selection uses the VALIDATION results only. Test tables are written separately,
and only if test results exist. Output: $RUNS_ROOT/figures and $RUNS_ROOT/tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import config as cfg
import plots
import results as res

ALPHA = 0.05
PREVIOUS_LABEL = "previous run, not same protocol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--data-root", default=None, help="Only for --qualitative")
    parser.add_argument("--qualitative", action="store_true", help="Render figure 8 (dataset + checkpoints)")
    parser.add_argument("--qualitative-split", default="val", choices=("val", "test"))
    parser.add_argument("--top-k", type=int, default=4, help="Models shown in figure 8")
    parser.add_argument("--tiles", nargs="*", default=[], metavar="CATEGORY:NAME",
                        help="Choose the tiles of figure 8 by hand, e.g. hairline:eh104_r0003_c0012")
    parser.add_argument("--previous", nargs="*", default=None,
                        help="Folders searched for test_metrics.json of earlier runs (default: $PREVIOUS_RUNS)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def pm(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def formatted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """The benchmark table as it appears in the thesis: mean ± std over seeds."""
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "Model": row["label"], "Family": row["family"], "Tier": row["tier"], "Seeds": int(row["n_seeds"]),
            "Params (M)": "-" if pd.isna(row["params_m"]) else f"{row['params_m']:.1f}",
            "FLOPs (G)": "-" if pd.isna(row["flops_g"]) else f"{row['flops_g']:.1f}",
            "Time (ms)": "-" if pd.isna(row["time_ms"]) else f"{row['time_ms']:.1f}",
            "IoU": pm(row["iou"], row["iou_std"]), "F1": pm(row["f1"], row["f1_std"]),
            "Precision": pm(row["precision"], row["precision_std"]), "Recall": pm(row["recall"], row["recall_std"]),
            "AP": pm(row["ap"], row["ap_std"]), "clDice": pm(row["cldice"], row["cldice_std"]),
            "Tol-F1@2px": pm(row["tol_f1_2"], row["tol_f1_2_std"]),
        })
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, out: Path, name: str, caption: str) -> None:
    """CSV and LaTeX (pandas) versions of a table."""
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    latex_table = table.map(lambda v: v.replace("±", r"$\pm$").replace("_", r"\_") if isinstance(v, str) else v)
    latex_table.columns = [str(c).replace("_", r"\_") for c in latex_table.columns]
    try:
        latex = latex_table.to_latex(index=False, escape=False, caption=caption, label=f"tab:{name}", na_rep="-",
                                     float_format="%.4f")
    except ImportError:   # pandas needs jinja2 for to_latex
        lines = [" & ".join(map(str, latex_table.columns)) + r" \\ \hline"]
        lines += [" & ".join(map(str, row)) + r" \\" for row in latex_table.itertuples(index=False)]
        latex = ("\\begin{table}\n\\caption{%s}\n\\begin{tabular}{%s}\n%s\n\\end{tabular}\n\\end{table}\n"
                 % (caption, "l" * len(latex_table.columns), "\n".join(lines)))
    (out / f"{name}.tex").write_text(latex, encoding="utf-8")


def significance_table(per_tile: pd.DataFrame, best: str) -> pd.DataFrame:
    """Wilcoxon signed-rank test on per-tile IoU: the best model against each other model."""
    from scipy.stats import wilcoxon

    rows = []
    for model in per_tile.columns:
        if model == best:
            continue
        difference = per_tile[best] - per_tile[model]
        if len(difference) < 6 or np.allclose(difference, 0):
            p_value = float("nan")
        else:
            p_value = float(wilcoxon(per_tile[best], per_tile[model], zero_method="wilcox").pvalue)
        rows.append({
            "best": res.info(best)["label"], "other": res.info(model)["label"], "other_name": model,
            "n_tiles": len(difference), "median_diff_iou": float(difference.median()),
            "mean_diff_iou": float(difference.mean()), "p_value": p_value,
            "significant_p<0.05": bool(p_value < ALPHA) if np.isfinite(p_value) else False,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ranking and recommendation (validation only)
# --------------------------------------------------------------------------- #
def rank_and_recommend(summary: pd.DataFrame, per_tile: pd.DataFrame, tables: Path) -> list[str]:
    lines: list[str] = []
    ranked = summary.sort_values("iou", ascending=False).reset_index(drop=True)
    tolerance_rank = summary["tol_f1_2"].rank(ascending=False, method="min").astype(int)
    tolerance_rank.index = summary["model"]
    ranking = pd.DataFrame({
        "rank_iou": range(1, len(ranked) + 1), "model": ranked["label"], "family": ranked["family"],
        "val IoU": [pm(m, s) for m, s in zip(ranked["iou"], ranked["iou_std"])],
        "val Tol-F1@2px": [pm(m, s) for m, s in zip(ranked["tol_f1_2"], ranked["tol_f1_2_std"])],
        "rank_tol_f1_2": [int(tolerance_rank[m]) for m in ranked["model"]],
        "Time (ms)": ranked["time_ms"].round(1), "Params (M)": ranked["params_m"].round(1),
    })
    write_table(ranking, tables, "ranking_val", "Models ranked by mean validation IoU over seeds.")
    lines.append("RANKING BY MEAN VALIDATION IoU (selection uses validation only)")
    lines.append(ranking.to_string(index=False))

    best = ranked.loc[0, "model"]
    lines.append(f"\nWINNER (val IoU): {ranked.loc[0, 'label']}  IoU {pm(ranked.loc[0, 'iou'], ranked.loc[0, 'iou_std'])}")
    best_tolerance = summary.sort_values("tol_f1_2", ascending=False).iloc[0]
    if best_tolerance["model"] != best:
        lines.append(f"NOTE: by tolerance-F1 @ 2 px the best model is {best_tolerance['label']} "
                     f"({pm(best_tolerance['tol_f1_2'], best_tolerance['tol_f1_2_std'])}). For hairline cracks "
                     "labelled a pixel or two off, this metric is the fairer one; weigh both.")

    if len(per_tile.columns) >= 2 and best in per_tile.columns:
        significance = significance_table(per_tile, best)
        write_table(significance.drop(columns=["other_name"]), tables, "significance_val",
                    "Wilcoxon signed-rank test on per-tile validation IoU, best model against each other model.")
        lines.append("\nWILCOXON SIGNED-RANK ON PER-TILE VAL IoU (best vs each)")
        lines.append(significance.drop(columns=["other_name"]).to_string(index=False))
        if len(ranked) >= 2:
            second = ranked.loc[1, "model"]
            row = significance[significance["other_name"] == second]
            p_value = float(row["p_value"].iloc[0]) if len(row) else float("nan")
            gap = ranked.loc[0, "iou"] - ranked.loc[1, "iou"]
            seed_overlap = gap <= ranked.loc[0, "iou_std"] + ranked.loc[1, "iou_std"]
            if not np.isfinite(p_value) or p_value >= ALPHA or seed_overlap:
                pair = ranked.loc[[0, 1]]
                key = "time_ms" if pair["time_ms"].notna().all() else "params_m"
                choice = pair.sort_values(key).iloc[0] if pair[key].notna().all() else pair.iloc[0]
                reason = []
                if not np.isfinite(p_value) or p_value >= ALPHA:
                    reason.append(f"Wilcoxon p = {p_value:.3g} (not below {ALPHA})")
                if seed_overlap:
                    reason.append(f"IoU gap {gap:.4f} is within the seed-to-seed spread")
                lines.append(f"\nFLAG: the top two ({ranked.loc[0, 'label']} and {ranked.loc[1, 'label']}) are NOT "
                             f"clearly different: {'; '.join(reason)}.")
                lines.append(f"RECOMMENDATION: take the faster/smaller one -> {choice['label']} "
                             f"({choice['time_ms']:.1f} ms per tile, {choice['params_m']:.1f} M parameters).")
            else:
                lines.append(f"\nThe winner is significantly better than the runner-up ({ranked.loc[1, 'label']}): "
                             f"Wilcoxon p = {p_value:.3g}, IoU gap {gap:.4f}.")
    return lines


# --------------------------------------------------------------------------- #
# Previous runs (reference only)
# --------------------------------------------------------------------------- #
def previous_run_rows(folders: list[str]) -> pd.DataFrame:
    """Rows from test_metrics.json files of earlier training scripts (different protocol)."""
    rows = []
    for folder in folders:
        base = Path(folder).expanduser()
        if not base.exists():
            continue
        files = [base] if base.is_file() else sorted(base.rglob("test_metrics.json"))
        for path in files:
            if any(part in ("site-packages", ".git", "benchmark", "original") for part in path.parts):
                continue
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            history = path.parent / "training_history.csv"
            epochs = len(pd.read_csv(history)) if history.exists() else None
            sweep = (path.parent / "val_threshold_sweep.csv").exists()
            for variant in ("no_tta", "tta"):
                block = content.get(variant)
                if not isinstance(block, dict) or "iou" not in block:
                    continue
                rows.append({
                    "Run": path.parent.name, "Variant": variant, "Label": PREVIOUS_LABEL, "Split": "test",
                    "IoU": block.get("iou"), "F1": block.get("f1"), "Precision": block.get("precision"),
                    "Recall": block.get("recall"), f"Tol-F1@{content.get('tolerance_px', 2)}px": block.get("tol_f1"),
                    "Threshold": block.get("threshold"), "Best epoch": content.get("best_epoch"),
                    "Epochs run": epochs, "Selected on": content.get("select_metric"),
                    "Has val sweep": sweep, "Path": str(path.parent),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure 8
# --------------------------------------------------------------------------- #
def choose_tiles(results: res.Results, models: list[str], split: str, manual: list[str]) -> list[tuple[str, str]]:
    """(category, tile name) rows: easy / hairline / shadowed-or-stained, picked from per-tile results."""
    if manual:
        return [tuple(item.split(":", 1)) for item in manual]   # type: ignore[misc]
    run = next(r for r in results.of(models[0]) if split in r.per_tile)
    tiles = run.per_tile[split].set_index("name")
    iou = res.per_tile_iou(results, split)[models].mean(axis=1)
    tiles = tiles.loc[tiles.index.intersection(iou.index)]
    tiles = tiles[(tiles[["height", "width"]].max(axis=1) <= 640) & (tiles["gt_pixels"] > 0)]
    if tiles.empty:
        return []
    iou = iou.loc[tiles.index]
    chosen: list[tuple[str, str]] = []

    def take(category: str, pool: pd.Index, quantile: float) -> None:
        pool = pool.difference([name for _, name in chosen])
        if len(pool):
            ranked = iou.loc[pool].sort_values()
            chosen.append((category, ranked.index[int(round(quantile * (len(ranked) - 1)))]))

    wide = tiles.index[tiles["gt_width"] >= tiles["gt_width"].median()]
    take("Easy", wide, 0.9)
    thin = tiles["gt_width"].sort_values().index[: max(1, len(tiles) // 5)]
    take("Hairline", thin, 0.5)
    uneven = tiles["illumination_std"].sort_values(ascending=False).index[: max(1, len(tiles) // 10)]
    take("Shadowed / stained", uneven, 0.5)
    return chosen


def render_qualitative(results: res.Results, summary: pd.DataFrame, args: argparse.Namespace, figures: Path) -> None:
    import torch

    import data
    import engine
    from models import build_model

    split = args.qualitative_split
    ranked = [m for m in summary.sort_values("iou", ascending=False)["model"]
              if any((r.path / "best.pt").exists() and "val" in r.metrics for r in results.of(m))]
    models = ranked[: args.top_k]
    if not models:
        print("[figure 8] no checkpoints found (figure 8 is rendered on the server)")
        return
    root = cfg.data_root(args.data_root)
    wanted = choose_tiles(results, models, split, args.tiles)
    if not wanted:
        print("[figure 8] no suitable tiles")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    pairs: dict[str, tuple[Path, Path]] = {}
    for model in models:
        run = next(r for r in results.of(model) if (r.path / "best.pt").exists() and "val" in r.metrics)
        checkpoint = torch.load(run.path / "best.pt", map_location=device, weights_only=False)
        config = checkpoint["config"]
        if not pairs:
            pairs = {name: (image, mask) for name, image, mask in data.collect_pairs(root, split)}
            for category, name in wanted:
                image, mask = data.read_pair(*pairs[name])
                rows.append({"category": category, "name": name, "image": image, "gt": mask, "preds": {}})
        network = build_model(model, pretrained=False, config=config).to(device)
        network.load_state_dict(checkpoint["model"])
        network.eval()
        threshold = run.metrics["val"]["threshold"]
        for row in rows:
            tensor, _ = data.to_tensors(row["image"], row["gt"])
            probability = engine.predict(network, tensor.unsqueeze(0).to(device), config)
            row["preds"][model] = (probability[0, 0] >= threshold).cpu().numpy()
    plots.setup()
    path = plots.fig8_qualitative(rows, models, figures, split)
    print(f"[figure 8] {path} | tiles: " + ", ".join(f"{r['category']}={r['name']}" for r in rows))


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    runs = cfg.runs_root(args.runs_root)
    figures, tables = runs / "figures", runs / "tables"
    report: list[str] = []

    results = res.load_results(runs, "benchmark")
    unfinished = [f"{r.model}/seed_{r.seed}" for r in results.runs if not r.finished]
    if unfinished:
        report.append(f"WARNING: {len(unfinished)} unfinished runs are included where evaluated: {', '.join(unfinished)}")
    val_summary = res.summary_table(results, "val")
    if val_summary.empty:
        print(f"No validation results under {runs / 'benchmark'}. Run evaluate.py --split val first.")
        return 1

    write_table(formatted_summary(val_summary), tables, "benchmark_val",
                "Benchmark on the validation set (same protocol for all models; mean ± std over seeds).")
    val_summary.to_csv(tables / "benchmark_val_raw.csv", index=False)
    res.seed_table(results, "val").to_csv(tables / "benchmark_val_per_seed.csv", index=False)
    per_tile = res.per_tile_iou(results, "val")
    report += rank_and_recommend(val_summary, per_tile, tables)

    groups = sorted({g for r in results.runs if "val" in r.metrics for g in r.metrics["val"]["per_group"]})
    if groups:
        by_group = pd.DataFrame([
            {"Model": res.info(m)["label"],
             **{g: np.mean([r.metrics["val"]["per_group"][g]["iou"] for r in results.of(m, "val")
                            if g in r.metrics["val"]["per_group"]] or [np.nan]) for g in groups}}
            for m in results.models if results.of(m, "val")]).round(4)
        write_table(by_group, tables, "val_iou_by_source", "Validation IoU per source group (mean over seeds).")

    test_summary = res.summary_table(results, "test")
    if not test_summary.empty:
        write_table(formatted_summary(test_summary), tables, "benchmark_test",
                    "Benchmark on the test set, at the threshold chosen on validation (mean ± std over seeds).")
        test_summary.to_csv(tables / "benchmark_test_raw.csv", index=False)
        report.append("\nTEST RESULTS written to tables/benchmark_test.* (reporting only; not used for selection)")

    # Original recipes and earlier runs: a separate table, never mixed into the ranking above.
    original = res.load_results(runs, "original")
    separate = []
    for split in ("val", "test"):
        summary = res.summary_table(original, split)
        if not summary.empty:
            table = formatted_summary(summary).drop(columns=["Tier", "Params (M)", "FLOPs (G)", "Time (ms)"])
            table.insert(1, "Label", "original recipe (own loss / EMA / deep supervision / TTA)")
            table.insert(2, "Split", split)
            separate.append(table)
    if separate:
        write_table(pd.concat(separate), tables, "original_recipe",
                    "Original-recipe runs. Not the benchmark protocol: do not compare with the benchmark table.")
        report.append("\nORIGINAL-RECIPE RUNS (separate; not the benchmark protocol)")
        report.append(pd.concat(separate).to_string(index=False))
    folders = args.previous if args.previous is not None else os.environ.get("PREVIOUS_RUNS", "").split(os.pathsep)
    previous = previous_run_rows([f for f in folders if f])
    if not previous.empty:
        write_table(previous, tables, "previous_runs", f"Earlier runs, shown for reference only ({PREVIOUS_LABEL}).")
        report.append(f"\nPREVIOUS RUNS ({PREVIOUS_LABEL})")
        report.append(previous.drop(columns=["Path"]).to_string(index=False))

    paths = plots.make_figures(results, figures, "val")
    if not test_summary.empty:
        plots.fig1_bars(test_summary, figures, "test")
    report.append(f"\nFIGURES: {len(paths)} written to {figures}")
    if args.qualitative:
        try:
            render_qualitative(results, val_summary, args, figures)
        except Exception as exc:
            print(f"[figure 8] failed: {type(exc).__name__}: {exc}")
            raise

    text = "\n".join(report)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "report.txt").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")   # Windows consoles without UTF-8
    print(text)
    print(f"\nTables: {tables}\nFigures: {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
