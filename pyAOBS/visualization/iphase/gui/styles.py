"""Qt styles for iphase GUI — palette and contrast aligned with Workbench shell_qt / imodel_qt."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from pyAOBS.workbench.shell_qt.styles import (
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_BORDER_LIGHT,
    COLOR_BORDER_MED,
    COLOR_PANEL,
    COLOR_PANEL_ALT,
    COLOR_SELECTION,
    COLOR_SHELL,
    COLOR_STATUS_BG,
    COLOR_SUCCESS,
    COLOR_SUCCESS_BG,
    COLOR_SUCCESS_BORDER,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_SECONDARY,
    COLOR_TOOLBAR,
    UI_FONT_PT,
    WORKBENCH_QSS,
)

PARAM_STRIP_FONT_PT = 10

# Reuse Workbench QSS selectors with Iphase* object names.
IPHASE_BASE_QSS = WORKBENCH_QSS.replace("Workbench", "Iphase")

IPHASE_EXTRA_QSS = f"""
QWidget#IphasePlotPanel {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
QWidget#IphaseToolWindowFrame {{
    background-color: {COLOR_PANEL};
    border: 3px solid {COLOR_BORDER};
    border-radius: 5px;
}}
QWidget#IphaseParamStrip {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    min-height: 32px;
}}
QWidget#IphaseParamStrip QLabel {{
    font-size: {PARAM_STRIP_FONT_PT}px;
}}
QWidget#IphaseParamStrip QPushButton {{
    min-height: 22px;
    max-height: 26px;
    padding: 1px 6px;
    font-size: {PARAM_STRIP_FONT_PT}px;
    font-weight: 600;
}}
QWidget#IphaseParamStrip QLineEdit,
QWidget#IphaseParamStrip QComboBox {{
    min-height: 22px;
    max-height: 24px;
    padding: 0 3px;
    font-size: {PARAM_STRIP_FONT_PT}px;
}}
QWidget#IphaseParamStrip QCheckBox {{
    font-size: {PARAM_STRIP_FONT_PT}px;
    spacing: 4px;
}}
QLabel#IphaseParamGroupTitle {{
    color: {COLOR_ACCENT};
    font-size: {PARAM_STRIP_FONT_PT}px;
    font-weight: 700;
    padding-right: 4px;
    min-width: 52px;
}}
QDialog {{
    background-color: {COLOR_SHELL};
    color: {COLOR_TEXT};
    border: 3px solid {COLOR_BORDER};
}}
QLabel#IphasePathBadgeIdle {{
    color: {COLOR_TEXT_MUTED};
    font-weight: 600;
    background-color: {COLOR_PANEL_ALT};
    border: 1px dashed {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    padding: 3px 8px;
}}
QLabel#IphasePathBadgeOpen {{
    color: {COLOR_SUCCESS};
    font-weight: bold;
    background-color: {COLOR_SUCCESS_BG};
    border: 1px solid {COLOR_SUCCESS_BORDER};
    border-left: 4px solid {COLOR_SUCCESS};
    border-radius: 3px;
    padding: 3px 8px;
}}
QLabel#IphaseCaption {{
    color: {COLOR_TEXT_MUTED};
    font-size: {UI_FONT_PT - 2}px;
}}
"""

IPHASE_QSS = IPHASE_BASE_QSS + IPHASE_EXTRA_QSS


def apply_iphase_font(app) -> None:
    font = QFont()
    font.setPointSize(UI_FONT_PT)
    app.setFont(font)


def apply_iphase_chrome(win: QMainWindow) -> None:
    """Main window: Workbench-like outer frame + global QSS."""
    win.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    win.setStyleSheet(IPHASE_QSS)

    from pyAOBS.utils.qt_independent_window import mark_chrome_frame

    central = win.centralWidget()
    if central is None:
        return
    if central.objectName() == "IphaseMainFrame":
        mark_chrome_frame(central)
        return
    frame = QWidget()
    frame.setObjectName("IphaseMainFrame")
    mark_chrome_frame(frame)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(0)
    win.takeCentralWidget()
    lay.addWidget(central)
    win.setCentralWidget(frame)


def apply_tool_window_chrome(win: QMainWindow) -> None:
    """Child tool windows (inversion / diagnostics)."""
    win.setWindowFlags(win.windowFlags() | Qt.WindowType.Window)
    win.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    win.setStyleSheet(IPHASE_QSS)

    from pyAOBS.utils.qt_independent_window import mark_chrome_frame

    central = win.centralWidget()
    if central is None:
        return
    if central.objectName() == "IphaseToolWindowFrame":
        mark_chrome_frame(central)
        return
    frame = QWidget()
    frame.setObjectName("IphaseToolWindowFrame")
    mark_chrome_frame(frame)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(0)
    win.takeCentralWidget()
    lay.addWidget(central)
    win.setCentralWidget(frame)


def apply_dialog_style(dialog) -> None:
    from pyAOBS.utils.qt_independent_window import apply_independent_window_frame

    dialog.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    dialog.setStyleSheet(IPHASE_QSS)
    apply_independent_window_frame(dialog)


def set_path_badge(label: QLabel, *, active: bool, text: str) -> None:
    label.setText(text)
    label.setObjectName("IphasePathBadgeOpen" if active else "IphasePathBadgeIdle")
    label.style().unpolish(label)
    label.style().polish(label)


def polish_status_bar_label(label: QLabel, level: str = "info") -> None:
    suffix = {
        "success": "Success",
        "failed": "Failed",
        "running": "Running",
        "warn": "Warn",
    }.get(level, "")
    label.setObjectName(
        "IphaseStatusBarMessage" + suffix if suffix else "IphaseStatusBarMessage"
    )
    label.style().unpolish(label)
    label.style().polish(label)


_tool_win_registry: list = []


def show_modeless_tool_window(win, *, activate: bool = True) -> None:
    """Show a QMainWindow / tool widget without blocking the main window."""
    if not isinstance(win, QWidget):
        raise TypeError("show_modeless_tool_window expects a QWidget")

    win.setWindowModality(Qt.WindowModality.NonModal)
    if hasattr(win, "setModal"):
        try:
            win.setModal(False)
        except Exception:
            pass
    win.setParent(None)
    from pyAOBS.utils.qt_independent_window import (
        INDEPENDENT_WINDOW_FLAGS,
        apply_independent_window_frame,
        apply_native_window_border,
    )

    win.setWindowFlags(INDEPENDENT_WINDOW_FLAGS)
    apply_independent_window_frame(win)
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

    if win not in _tool_win_registry:
        _tool_win_registry.append(win)

    def _on_destroyed(*_args) -> None:
        try:
            _tool_win_registry.remove(win)
        except ValueError:
            pass

    win.destroyed.connect(_on_destroyed)
    win.show()
    apply_native_window_border(win)
    if activate:
        win.raise_()
        win.activateWindow()
