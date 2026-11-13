"""YAML configuration: one base file, a small per-model override, and CLI overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return `base` updated recursively with `override` (neither argument is modified)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a config. `_base_` (default base.yaml next to the file) is merged first.

    `overrides` are ``section.key=value`` strings; values are parsed as YAML.
    """
    path = Path(path)
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    raw = load_yaml(path)
    base_name = raw.pop("_base_", "base.yaml")
    config: dict[str, Any] = {}
    if base_name and path.name != base_name:
        base_path = path.parent / base_name
        if not base_path.exists():
            base_path = CONFIG_DIR / Path(base_name).name
        config = load_config(base_path)
    config = deep_merge(config, raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def runs_root(cli_value: str | None = None) -> Path:
    """Results folder: --runs-root, else $RUNS_ROOT, else ./runs."""
    return Path(cli_value or os.environ.get("RUNS_ROOT") or REPO_ROOT / "runs").expanduser()


def data_root(cli_value: str | None = None) -> Path:
    """Dataset folder: --data-root, else $DATA_ROOT. Never hardcoded."""
    value = cli_value or os.environ.get("DATA_ROOT")
    if not value:
        raise SystemExit("Set DATA_ROOT or pass --data-root (the folder holding train/ val/ test/).")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise SystemExit(f"Dataset folder not found: {path}")
    return path


def run_dir(root: Path, protocol: str, model: str, seed: int) -> Path:
    """$RUNS_ROOT/<protocol>/<model>/seed_<n>; protocol is 'benchmark' or 'original'."""
    return root / protocol / model / f"seed_{seed}"


def num_workers(default: int | None = None) -> int:
    """DataLoader workers: SLURM_CPUS_PER_TASK when inside a job, else a small default."""
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value:
        return max(1, int(value))
    if default is not None:
        return default
    return 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
