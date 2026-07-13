"""
Locate vendored pyrolite / burnman trees under ``petrology/`` and prepend them to
``sys.path`` so WSL (or in-tree) installs work without a separate pip checkout.

Prefer pip-installed packages when ``prefer_pip=True`` and import already succeeds.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

_PETROLOGY = Path(__file__).resolve().parent
_PYROLITE_ROOT = _PETROLOGY  # package dir: petrology/pyrolite/
_BURNMAN_ROOT = _PETROLOGY / "burnman"  # package dir: petrology/burnman/burnman/

VendoredName = Literal["pyrolite", "burnman"]


def _is_importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def vendored_path(name: VendoredName) -> Path:
    if name == "pyrolite":
        return _PYROLITE_ROOT
    if name == "burnman":
        return _BURNMAN_ROOT
    raise KeyError(name)


def ensure_vendored(*names: VendoredName, prefer_pip: bool = False) -> None:
    """Insert vendored roots at the front of ``sys.path`` if needed."""
    for name in names:
        if prefer_pip and _is_importable(name):
            continue
        root = vendored_path(name)
        if not root.is_dir():
            raise FileNotFoundError(f"vendored {name} not found at {root}")
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)


def pyrolite_available(*, prefer_pip: bool = True) -> bool:
    if prefer_pip and _is_importable("pyrolite"):
        return True
    if not (_PYROLITE_ROOT / "pyrolite" / "mineral" / "normative.py").is_file():
        return False
    ensure_vendored("pyrolite", prefer_pip=False)
    return _is_importable("pyrolite")


def burnman_available(*, prefer_pip: bool = True) -> bool:
    if prefer_pip and _is_importable("burnman"):
        return True
    if not (_BURNMAN_ROOT / "burnman" / "minerals" / "SLB_2022.py").is_file():
        return False
    ensure_vendored("burnman", prefer_pip=False)
    return _is_importable("burnman")
