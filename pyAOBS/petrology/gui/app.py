"""PySide6 入口：延迟加载依赖，失败时给出安装提示。"""

from __future__ import annotations

import sys
from typing import Optional

# 全局 UI 字号（与 Workbench 可读性取向一致，略高于 Qt 默认）
UI_FONT_PT = 13
LOG_FONT_PT = 12


def _configure_app_font(app) -> None:
    from PySide6.QtGui import QFont

    from .mpl_cjk import pick_qt_font_family

    font = QFont(pick_qt_font_family())
    font.setPointSize(UI_FONT_PT)
    app.setFont(font)


def run_application(argv: Optional[list[str]] = None) -> int:
    import argparse

    raw = list(argv if argv is not None else sys.argv)
    parser = argparse.ArgumentParser(prog="petrology.gui", add_help=True)
    parser.add_argument(
        "--import-obs",
        dest="import_obs",
        default=None,
        help="启动后导入 CrustObservation JSON（imodel 导出）",
    )
    parser.add_argument(
        "--import-transect",
        dest="import_transect",
        default=None,
        help="启动后导入沿迹滑窗 windows CSV → Fig.15c 叠 Fig.12a",
    )
    args, qt_argv = parser.parse_known_args(raw[1:])
    qt_argv = [raw[0]] + qt_argv

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from .mainwindow import LipMainWindow
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 或 Matplotlib/Qt 后端未就绪。请执行："
            " pip install 'pyAOBS[gui-qt]' 或 pip install PySide6 matplotlib"
        ) from exc

    from petrology.thread_env import limit_native_parallelism

    app = QApplication(qt_argv)
    limit_native_parallelism()
    app.setApplicationName("LIP Petrology")
    _configure_app_font(app)
    win = LipMainWindow()
    win.show()
    if args.import_obs:
        obs_path = str(args.import_obs)

        def _do_import_obs() -> None:
            win.import_observation_file(obs_path)

        QTimer.singleShot(200, _do_import_obs)
    if args.import_transect:
        tran_path = str(args.import_transect)

        def _do_import_transect() -> None:
            win.import_transect_file(tran_path)

        QTimer.singleShot(350 if args.import_obs else 200, _do_import_transect)
    return app.exec()


def launch_lip_gui() -> int:
    return run_application()


def main(argv: Optional[list[str]] = None) -> int:
    return run_application(argv)
