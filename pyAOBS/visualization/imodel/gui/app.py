"""Qt 入口：延迟加载 PySide6 / Matplotlib，失败时给出安装提示。"""

from __future__ import annotations

import sys
from typing import Optional

from .import_paths import ensure_imodel_qt_import_paths


def main(argv: Optional[list[str]] = None) -> int:
    ensure_imodel_qt_import_paths()
    try:
        from .mainwindow import run_application
    except ImportError as exc:
        msg = str(exc).lower()
        if "pyside6" in msg or "matplotlib" in msg or "qt" in msg:
            hint = (
                "PySide6 或 Matplotlib/Qt 后端未就绪。请执行："
                " pip install 'pyAOBS[gui-qt]' 或 pip install PySide6 matplotlib"
            )
        elif "petrology" in msg:
            hint = (
                "无法导入 petrology 模块。请从仓库根目录执行："
                " pip install -e \".[gui-qt]\" ，"
                "或设置 PYTHONPATH 指向 pyAOBS 源码包目录（含 petrology/ 文件夹）。"
            )
        else:
            hint = f"imodel Qt 依赖未就绪：{exc}"
        raise RuntimeError(hint) from exc
    return run_application(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
