# -*- coding: utf-8 -*-
"""
GUI 日志桥接：model_building / pyAOBS 的 INFO 进界面，控制台只留 WARNING+。

在 Qt 主窗口创建后调用::

    from pyAOBS.utils.gui_logging import configure_gui_logging
    configure_gui_logging(main_window.append_log)
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

# get_logger 在 GUI 模式下给新建 logger 用
_GUI_MODE = False
_GUI_HANDLER: Optional[logging.Handler] = None
_CONSOLE_LEVEL = logging.WARNING


def gui_logging_active() -> bool:
    return bool(_GUI_MODE)


def _is_library_logger(name: str) -> bool:
    n = str(name or "")
    if not n:
        return False
    if n == "pyAOBS" or n.startswith("pyAOBS."):
        return True
    if n == "model_building" or n.startswith("model_building."):
        return True
    return False


class GuiLogHandler(logging.Handler):
    """把 logging 记录转给 GUI 回调（勿在 emit 里做重活）。"""

    def __init__(self, emit_fn: Callable[[str], None]) -> None:
        super().__init__()
        self._emit_fn = emit_fn
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if record.levelno >= logging.WARNING:
                msg = "[%s] %s" % (record.levelname, msg)
            self._emit_fn(msg)
        except Exception:
            self.handleError(record)


def _quiet_stream_handlers(logger: logging.Logger, level: int) -> None:
    for h in list(logger.handlers):
        # FileHandler 是 StreamHandler 子类，勿动；只压控制台
        if isinstance(h, logging.FileHandler):
            continue
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)


def _attach_gui_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    if handler not in logger.handlers:
        logger.addHandler(handler)


def configure_gui_logging(
    emit_fn: Callable[[str], None],
    *,
    console_level: int = logging.WARNING,
    logger_level: int = logging.INFO,
) -> None:
    """
    安装 GUI 日志：库 INFO → ``emit_fn``；控制台 StreamHandler → WARNING+。

    可重复调用（会替换回调）。``emit_fn`` 宜线程安全（如 MainWindow.append_log）。
    """
    global _GUI_MODE, _GUI_HANDLER, _CONSOLE_LEVEL
    _GUI_MODE = True
    _CONSOLE_LEVEL = int(console_level)

    # 去掉旧 GuiHandler，避免重复刷屏
    if _GUI_HANDLER is not None:
        for name, obj in list(logging.Logger.manager.loggerDict.items()):
            if isinstance(obj, logging.Logger) and _GUI_HANDLER in obj.handlers:
                obj.removeHandler(_GUI_HANDLER)
        root = logging.getLogger()
        if _GUI_HANDLER in root.handlers:
            root.removeHandler(_GUI_HANDLER)

    handler = GuiLogHandler(emit_fn)
    handler.setLevel(int(logger_level))
    _GUI_HANDLER = handler

    # 已创建的库 logger
    for name, obj in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(obj, logging.Logger):
            continue
        if not _is_library_logger(str(name)):
            continue
        obj.setLevel(min(obj.level or logger_level, logger_level) or logger_level)
        if obj.level == logging.NOTSET:
            obj.setLevel(logger_level)
        _quiet_stream_handlers(obj, _CONSOLE_LEVEL)
        _attach_gui_handler(obj, handler)
        obj.propagate = False

    # 包根上也挂一份（尚未 get_logger 的子模块仍可能 propagate 到此）
    for root_name in ("pyAOBS", "model_building"):
        lg = logging.getLogger(root_name)
        if lg.level == logging.NOTSET:
            lg.setLevel(logger_level)
        _quiet_stream_handlers(lg, _CONSOLE_LEVEL)
        _attach_gui_handler(lg, handler)
        lg.propagate = False


def apply_gui_logging_to_new_logger(logger: logging.Logger) -> None:
    """供 ``get_logger`` 在新建 handler 后调用。"""
    if not _GUI_MODE:
        return
    _quiet_stream_handlers(logger, _CONSOLE_LEVEL)
    if _GUI_HANDLER is not None:
        _attach_gui_handler(logger, _GUI_HANDLER)
    logger.propagate = False
