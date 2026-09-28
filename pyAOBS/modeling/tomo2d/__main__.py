"""``python -m pyAOBS.modeling.tomo2d`` → Qt GUI。"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from pyAOBS.modeling.tomo2d.gui.app import main as gui_main

    return gui_main(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
