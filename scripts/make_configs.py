#!/usr/bin/env python3
"""Write configs/<model>.yaml for every registered model that has none yet.

Run it after adding a `register(...)` line in models/registry.py. Existing files are
never overwritten, so per-model edits (e.g. a smaller batch) are kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import REGISTRY  # noqa: E402


def main() -> None:
    for spec in REGISTRY.values():
        path = REPO_ROOT / "configs" / f"{spec.name}.yaml"
        if spec.external or path.exists():
            continue
        path.write_text(
            f"# {spec.label} | family: {spec.family} | {spec.tier}\n"
            "# Everything not set here comes from base.yaml (the shared benchmark protocol).\n"
            f"model: {spec.name}\n", encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
