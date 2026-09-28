"""Import path bootstrap for ``python -m pyAOBS.visualization.imodel.gui``."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_imodel_qt_import_paths() -> None:
    """Ensure repo / editable-install roots are on ``sys.path``.

    When launched as an installed submodule (``pyAOBS.visualization.imodel``),
    this is usually a no-op. When run from a source checkout, add the package
    parents so sibling packages (``petrology``, ``utils``, …) resolve.
    """
    here = Path(__file__).resolve()
    # .../pyAOBS/visualization/imodel/gui/import_paths.py
    candidates = [
        here.parents[3],  # .../pyAOBS (package root containing visualization/)
        here.parents[4],  # repo root (if pyAOBS/ is nested)
    ]
    for root in candidates:
        s = str(root)
        if s and s not in sys.path:
            sys.path.insert(0, s)


# Back-compat alias
ensure_imodel_import_paths = ensure_imodel_qt_import_paths
