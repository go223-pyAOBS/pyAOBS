"""``python -m pyAOBS.visualization.imodel`` → redirect to ``imodel.gui``.

Preferred entry (aligned with zplotpy / iphase)::

    python -m pyAOBS.visualization.imodel.gui
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "imodel GUI entry is: python -m pyAOBS.visualization.imodel.gui",
        file=sys.stderr,
    )
    from .gui.app import main as gui_main

    return int(gui_main(argv if argv is not None else sys.argv))


if __name__ == "__main__":
    raise SystemExit(main())
