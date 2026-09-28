"""iphase Qt GUI — 震相分析工区。

启动：
  python -m pyAOBS.visualization.iphase.gui
"""

from __future__ import annotations

from .main_window import IPhaseMainWindow, run_iphase_app
from .project import IphaseProject

__all__ = ["IPhaseMainWindow", "IphaseProject", "run_iphase_app", "main"]


def main(argv: list[str] | None = None) -> int:
    return int(run_iphase_app(argv) or 0)
