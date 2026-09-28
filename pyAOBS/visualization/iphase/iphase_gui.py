"""兼容旧模块名：转发到 Qt 工程化入口。

历史 Tk 实现已归档至 ``_archive/iphase_gui_tk.py``。
正式启动请使用::

    python -m pyAOBS.visualization.iphase.gui
"""

from __future__ import annotations


def main() -> int:
    from pyAOBS.visualization.iphase.gui import main as gui_main

    return int(gui_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
