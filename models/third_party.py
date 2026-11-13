"""Import networks from official repositories cloned into third_party/ (never committed).

The official CrackFormer and SCSegamba repositories use top-level packages named
`utils` and `models`. `models` is also the name of this package, so their modules are
imported with the repository first on sys.path and this package temporarily set aside.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent


def third_party_dir(name: str, env_var: str) -> Path:
    """Folder of a cloned repository: $<env_var>, else <repo>/third_party/<name>."""
    path = Path(os.environ.get(env_var) or REPO_ROOT / "third_party" / name)
    if not path.is_dir():
        raise RuntimeError(
            f"{name} is not installed: {path} does not exist. Run the install script in scripts/ "
            f"(see SERVER_GUIDE.md) or set {env_var}.")
    return path


@contextmanager
def foreign_packages(repository: Path, packages: tuple[str, ...]) -> Iterator[None]:
    """Import from `repository`, hiding this project's packages of the same name meanwhile."""
    def owned(name: str) -> bool:
        return any(name == package or name.startswith(package + ".") for package in packages)

    saved = {name: module for name, module in sys.modules.items() if owned(name)}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(repository))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path.remove(str(repository))
        for name in [name for name in sys.modules if owned(name)]:
            del sys.modules[name]   # the imported classes keep working; only the names are released
        sys.modules.update(saved)
        importlib.invalidate_caches()
