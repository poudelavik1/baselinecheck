"""Load everything the report needs from $RUNS_ROOT. Pandas only: no torch, no dataset.

Works on a laptop from the rsynced results (JSON / CSV, no checkpoints).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.registry import REGISTRY

METRIC_COLUMNS = ["iou", "f1", "precision", "recall", "ap", "cldice", "tol_f1_2"]


@dataclass
class Run:
    model: str
    seed: int
    path: Path
    finished: bool
    history: pd.DataFrame | None = None
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)      # split -> metrics json
    curves: dict[str, pd.DataFrame] = field(default_factory=dict)         # split -> threshold curves
    per_tile: dict[str, pd.DataFrame] = field(default_factory=dict)       # split -> per-tile rows


@dataclass
class Results:
    protocol: str
    runs: list[Run]
    efficiency: dict[str, dict[str, Any]]

    @property
    def models(self) -> list[str]:
        """Models with at least one run, in registry order (unknown names last)."""
        present = {run.model for run in self.runs}
        ordered = [name for name in REGISTRY if name in present]
        return ordered + sorted(present - set(ordered))

    def of(self, model: str, split: str | None = None) -> list[Run]:
        return [r for r in self.runs if r.model == model and (split is None or split in r.metrics)]


def info(model: str) -> dict[str, str]:
    """Family, tier and figure label of a model (from the registry)."""
    spec = REGISTRY.get(model)
    if spec is None:
        return {"family": "CNN", "tier": "core", "label": model}
    return {"family": spec.family, "tier": spec.tier, "label": spec.label}


def load_results(runs_root: Path, protocol: str = "benchmark") -> Results:
    base = Path(runs_root) / protocol
    runs: list[Run] = []
    efficiency: dict[str, dict[str, Any]] = {}
    for model_dir in sorted(p for p in base.glob("*") if p.is_dir()) if base.is_dir() else []:
        if (model_dir / "efficiency.json").exists():
            efficiency[model_dir.name] = json.loads((model_dir / "efficiency.json").read_text(encoding="utf-8"))
        for run_dir in sorted(model_dir.glob("seed_*")):
            run = Run(model_dir.name, int(run_dir.name.split("_")[1]), run_dir, (run_dir / "FINISHED").exists())
            if (run_dir / "history.csv").exists():
                run.history = pd.read_csv(run_dir / "history.csv")
            for split in ("val", "test"):
                path = run_dir / "eval" / f"{split}_metrics.json"
                if not path.exists():
                    continue
                run.metrics[split] = json.loads(path.read_text(encoding="utf-8"))
                for store, suffix in ((run.curves, "curves"), (run.per_tile, "per_tile")):
                    csv_path = run_dir / "eval" / f"{split}_{suffix}.csv"
                    if csv_path.exists():
                        store[split] = pd.read_csv(csv_path)
            if run.history is not None or run.metrics:
                runs.append(run)
    return Results(protocol, runs, efficiency)


def flat_metrics(metrics_json: dict[str, Any]) -> dict[str, float]:
    """One flat row of numbers from a metrics JSON."""
    row = dict(metrics_json["metrics"])
    row["ap"] = metrics_json["ap"]
    row["threshold"] = metrics_json["threshold"]
    return row


def seed_table(results: Results, split: str) -> pd.DataFrame:
    """One row per (model, seed) with all pooled metrics."""
    rows = []
    for run in results.runs:
        if split in run.metrics:
            rows.append({"model": run.model, "seed": run.seed, **flat_metrics(run.metrics[split])})
    return pd.DataFrame(rows)


def summary_table(results: Results, split: str) -> pd.DataFrame:
    """One row per model: mean and std over seeds of every metric, plus efficiency."""
    seeds = seed_table(results, split)
    rows = []
    for model in results.models:
        subset = seeds[seeds["model"] == model] if len(seeds) else seeds
        if not len(subset):
            continue
        row: dict[str, Any] = {"model": model, **info(model), "n_seeds": len(subset)}
        numeric = subset.drop(columns=["model", "seed"])
        for column in numeric.columns:
            row[column] = numeric[column].mean()
            row[f"{column}_std"] = numeric[column].std(ddof=0) if len(subset) > 1 else 0.0
        efficiency = results.efficiency.get(model, {})
        row["params_m"] = efficiency.get("params_m", np.nan)
        row["flops_g"] = efficiency.get("flops_g", np.nan)
        row["time_ms"] = efficiency.get("time_ms", np.nan)
        row["device"] = efficiency.get("device", "")
        rows.append(row)
    return pd.DataFrame(rows)


def per_tile_iou(results: Results, split: str) -> pd.DataFrame:
    """Per-tile IoU averaged over seeds: rows = tiles, columns = models (tiles common to all)."""
    columns = {}
    for model in results.models:
        frames = [run.per_tile[split].set_index("name")["iou"] for run in results.of(model) if split in run.per_tile]
        if frames:
            columns[model] = pd.concat(frames, axis=1).mean(axis=1)
    return pd.DataFrame(columns).dropna() if columns else pd.DataFrame()


def mean_curve(results: Results, model: str, split: str) -> pd.DataFrame | None:
    """Threshold curves averaged over seeds (all runs share the same 255 thresholds)."""
    frames = [run.curves[split] for run in results.of(model) if split in run.curves]
    if not frames:
        return None
    return sum(frames) / len(frames)


def history_band(results: Results, model: str, column: str) -> pd.DataFrame | None:
    """Mean and std over seeds of one history column, per epoch."""
    series = [run.history.set_index("epoch")[column] for run in results.of(model)
              if run.history is not None and column in run.history]
    if not series:
        return None
    table = pd.concat(series, axis=1)
    return pd.DataFrame({"mean": table.mean(axis=1), "std": table.std(axis=1, ddof=0).fillna(0.0)})
