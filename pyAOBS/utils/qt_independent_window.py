# -*- coding: utf-8 -*-
"""独立 Qt 子窗：非模态 + 可见外边框。

Windows 11 原生边框几乎不可见，子窗叠在主窗上会像同一块面板。
各 GUI 的 ``configure_independent_dialog`` / ``show_modeless_dialog`` 应走这里，
不要只依赖系统标题栏。
"""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

_FRAMED_PROP = "_pyaobsIndependentFramed"
_FRAME_ATTR = "pyaobsIndependentWindow"
_FRAME_OBJECT = "PyAobsIndependentFrame"

# 与 tomo2d 状态栏 / workbench COLOR_BORDER 同级的深色描边
FRAME_BORDER_COLOR = "#2b3340"
FRAME_BORDER_PX = 3

INDEPENDENT_WINDOW_FLAGS = (
    Qt.WindowType.Window
    | Qt.WindowType.WindowTitleHint
    | Qt.WindowType.WindowSystemMenuHint
    | Qt.WindowType.WindowMinMaxButtonsHint
    | Qt.WindowType.WindowCloseButtonHint
)

# 结果提示框不要 MinMax：Wayland 下最大化主窗再弹出带 MinMax 的独立
# QMessageBox，易触发 xdg_surface buffer 与 maximized state 尺寸不一致的协议崩溃。
MESSAGE_WINDOW_FLAGS = (
    Qt.WindowType.Window
    | Qt.WindowType.WindowTitleHint
    | Qt.WindowType.WindowSystemMenuHint
    | Qt.WindowType.WindowCloseButtonHint
)

_KNOWN_CHROME_FRAMES = frozenset(
    {
        _FRAME_OBJECT,
        "ImodelToolWindowFrame",
        "ImodelMainFrame",
        "IphaseToolWindowFrame",
        "IphaseMainFrame",
        "WorkbenchMainFrame",
        "ObsRtmMainFrame",
        "VeditMainFrame",
    }
)

_FRAME_QSS = (
    f'QWidget[{_FRAME_ATTR}="true"] {{'
    f"  border: {FRAME_BORDER_PX}px solid {FRAME_BORDER_COLOR};"
    "  background-color: palette(window);"
    "}"
)


def mark_chrome_frame(frame: QWidget) -> None:
    """QSS 边框必须 WA_StyledBackground，否则 Fusion/Win11 上经常不画。"""
    frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def configure_independent_window(win: QWidget) -> None:
    """脱离 parent、非模态、系统标题栏，并画一圈可见客户区边框。"""
    if hasattr(win, "setModal"):
        try:
            win.setModal(False)
        except Exception:
            pass
    try:
        win.setWindowModality(Qt.WindowModality.NonModal)
    except Exception:
        pass
    want_drops = bool(win.acceptDrops())
    win.setParent(None)
    flags = MESSAGE_WINDOW_FLAGS if isinstance(win, QMessageBox) else INDEPENDENT_WINDOW_FLAGS
    win.setWindowFlags(flags)
    if want_drops:
        win.setAcceptDrops(True)
        win.setProperty("_pyaobsWantDrops", True)
    apply_independent_window_frame(win)


def apply_independent_window_frame(win: QWidget) -> None:
    """给顶层窗画可见边框；可重复调用。"""
    if not isinstance(win, QWidget):
        return
    if isinstance(win, QMainWindow):
        _frame_main_window(win)
    elif not isinstance(win, QMessageBox):
        _frame_widget(win)

    def _native(w: QWidget = win) -> None:
        try:
            apply_native_window_border(w)
        except RuntimeError:
            pass

    QTimer.singleShot(0, _native)
    if bool(win.property("_pyaobsWantDrops")):
        win.setAcceptDrops(True)


def apply_native_window_border(win: QWidget) -> None:
    """Windows 11 DWM 边框色；其它平台忽略。"""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(win.winId())
    except Exception:
        return
    if hwnd == 0:
        return
    _set_dwm_border_color(hwnd)


def _frame_main_window(win: QMainWindow) -> None:
    central: Optional[QWidget] = win.centralWidget()
    if central is not None and central.objectName() in _KNOWN_CHROME_FRAMES:
        mark_chrome_frame(central)
        win.setProperty(_FRAMED_PROP, True)
        return
    if bool(win.property(_FRAMED_PROP)):
        if central is not None:
            mark_chrome_frame(central)
        return
    if central is None:
        _frame_widget(win)
        return
    frame = QWidget()
    frame.setObjectName(_FRAME_OBJECT)
    mark_chrome_frame(frame)
    frame.setStyleSheet(
        f"QWidget#{_FRAME_OBJECT} {{"
        f"  border: {FRAME_BORDER_PX}px solid {FRAME_BORDER_COLOR};"
        "  background-color: palette(window);"
        "}"
    )
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(0)
    win.takeCentralWidget()
    lay.addWidget(central)
    win.setCentralWidget(frame)
    win.setProperty(_FRAMED_PROP, True)


def _frame_widget(win: QWidget) -> None:
    win.setProperty(_FRAME_ATTR, True)
    win.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    if bool(win.property(_FRAMED_PROP)):
        style = win.style()
        if style is not None:
            style.unpolish(win)
            style.polish(win)
        win.update()
        return
    win.setProperty(_FRAMED_PROP, True)
    current = win.styleSheet() or ""
    if _FRAME_ATTR not in current:
        win.setStyleSheet((current + "\n" + _FRAME_QSS).strip())
    _ensure_border_margins(win, pad=FRAME_BORDER_PX + 1)
    style = win.style()
    if style is not None:
        style.unpolish(win)
        style.polish(win)
    win.update()


def _ensure_border_margins(win: QWidget, *, pad: int) -> None:
    lay = win.layout()
    if lay is None:
        return
    m = lay.contentsMargins()
    lay.setContentsMargins(
        max(m.left(), pad),
        max(m.top(), pad),
        max(m.right(), pad),
        max(m.bottom(), pad),
    )


def _set_dwm_border_color(hwnd: int) -> None:
    """DWMWA_BORDER_COLOR=34；COLORREF 为 0x00BBGGRR。"""
    try:
        import ctypes

        dwmapi = ctypes.WinDLL("dwmapi")
        color = ctypes.c_uint(0x0040332B)  # #2b3340
        dwmapi.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd),
            ctypes.c_uint(34),
            ctypes.byref(color),
            ctypes.sizeof(color),
        )
    except Exception:
        pass
