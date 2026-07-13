"""Import path bootstrap for ``python -m pyAOBS.visualization.imodel_qt``."""

from __future__ import annotations

import sys
from pathlib import Path

# pyAOBS package root (directory containing ``petrology/``, ``visualization/``, …)
_PKG_ROOT = Path(__file__).resolve().parents[2]


def ensure_imodel_qt_import_paths() -> None:
    """
    Allow legacy ``import petrology`` used across pyAOBS GUI code.

    When launched as an installed submodule (``pyAOBS.visualization.imodel_qt``),
    only ``pyAOBS.*`` is on the path unless we prepend the in-tree package root.
    """
    root = str(_PKG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
