"""Qt styles for imodel_qt — palette and contrast aligned with Workbench shell_qt."""

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
    LOG_FONT_PT,
    UI_FONT_PT,
    WORKBENCH_QSS,
)

SIDE_PANEL_FONT_PT = 10

# Reuse Workbench QSS selectors with Imodel* object names (same contrast / borders / buttons).
IMODEL_BASE_QSS = WORKBENCH_QSS.replace("Workbench", "Imodel")

IMODEL_EXTRA_QSS = f"""
QWidget#ImodelPlotPanel {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
QWidget#ImodelLogPanel {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
QWidget#ImodelToolWindowFrame {{
    background-color: {COLOR_PANEL};
    border: 3px solid {COLOR_BORDER};
    border-radius: 5px;
}}
QToolButton#ImodelParamsToggle {{
    background-color: {COLOR_TOOLBAR};
    border: 2px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    min-width: 18px;
    font-weight: 700;
}}
QToolButton#ImodelParamsToggle:hover {{
    background-color: {COLOR_SELECTION};
    border-color: {COLOR_ACCENT};
}}
QLabel#ImodelPathBadgeIdle {{
    color: {COLOR_TEXT_MUTED};
    font-weight: 600;
    background-color: {COLOR_PANEL_ALT};
    border: 1px dashed {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    padding: 3px 8px;
}}
QLabel#ImodelPathBadgeOpen {{
    color: {COLOR_SUCCESS};
    font-weight: bold;
    background-color: {COLOR_SUCCESS_BG};
    border: 1px solid {COLOR_SUCCESS_BORDER};
    border-left: 4px solid {COLOR_SUCCESS};
    border-radius: 3px;
    padding: 3px 8px;
}}
QLabel#ImodelCaption {{
    color: {COLOR_TEXT_MUTED};
    font-size: {UI_FONT_PT - 2}px;
}}
QDialog {{
    background-color: {COLOR_SHELL};
    color: {COLOR_TEXT};
}}
QWidget#ImodelSidePanel {{
    font-size: {SIDE_PANEL_FONT_PT}px;
}}
QWidget#ImodelToolsBar {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    min-height: 32px;
}}
QWidget#ImodelToolsBar QLabel#ImodelToolsCaption {{
    font-weight: bold;
    font-size: {SIDE_PANEL_FONT_PT}px;
    color: {COLOR_TEXT};
    padding: 0 4px 0 2px;
}}
QWidget#ImodelToolsBar QPushButton {{
    min-height: 22px;
    max-height: 26px;
    padding: 1px 6px;
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 600;
}}
QWidget#ImodelSidePanel QGroupBox {{
    font-size: {SIDE_PANEL_FONT_PT}px;
    margin-top: 6px;
    padding: 4px 3px 3px 3px;
}}
QWidget#ImodelSidePanel QGroupBox::title {{
    padding: 0 3px;
    subcontrol-origin: margin;
    subcontrol-position: top left;
}}
QWidget#ImodelSidePanel QPushButton {{
    min-height: 22px;
    max-height: 26px;
    padding: 1px 6px;
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 600;
}}
QWidget#ImodelSidePanel QPushButton#ImodelPrimaryButton {{
    min-height: 24px;
    padding: 2px 8px;
}}
QWidget#ImodelSidePanel QLineEdit,
QWidget#ImodelSidePanel QComboBox {{
    min-height: 22px;
    max-height: 24px;
    padding: 0 3px;
    font-size: {SIDE_PANEL_FONT_PT}px;
}}
QWidget#ImodelSidePanel QCheckBox {{
    font-size: {SIDE_PANEL_FONT_PT}px;
    spacing: 4px;
}}
QWidget#ImodelSidePanel QLabel {{
    font-size: {SIDE_PANEL_FONT_PT}px;
}}
QWidget#ImodelSidePanel QLabel#ImodelStatusPanelCompact {{
    font-family: Consolas, "Courier New", monospace;
    font-size: {SIDE_PANEL_FONT_PT - 1}px;
    color: {COLOR_TEXT_SECONDARY};
    background-color: {COLOR_STATUS_BG};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-left: 3px solid {COLOR_ACCENT};
    border-radius: 3px;
    padding: 2px 4px;
}}
QWidget#ImodelSidePanel QLabel#ImodelPathBadgeIdle,
QWidget#ImodelSidePanel QLabel#ImodelPathBadgeOpen {{
    padding: 2px 6px;
    font-size: {SIDE_PANEL_FONT_PT - 1}px;
}}
"""

IMODEL_QSS = IMODEL_BASE_QSS + IMODEL_EXTRA_QSS


def apply_imodel_font(app) -> None:
    font = QFont()
    font.setPointSize(UI_FONT_PT)
    app.setFont(font)


def apply_imodel_chrome(win: QMainWindow) -> None:
    """Main window: Workbench-like outer frame + global QSS."""
    win.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    win.setStyleSheet(IMODEL_QSS)

    central = win.centralWidget()
    if central is None or central.objectName() == "ImodelMainFrame":
        return
    frame = QWidget()
    frame.setObjectName("ImodelMainFrame")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(0)
    win.takeCentralWidget()
    lay.addWidget(central)
    win.setCentralWidget(frame)


def apply_tool_window_chrome(win: QMainWindow) -> None:
    """Child tool windows (gravity, scatter, property grid)."""
    from PySide6.QtCore import Qt as _Qt

    win.setWindowFlags(win.windowFlags() | _Qt.WindowType.Window)
    win.setAttribute(_Qt.WidgetAttribute.WA_StyledBackground, True)
    win.setStyleSheet(IMODEL_QSS)

    central = win.centralWidget()
    if central is None or central.objectName() == "ImodelToolWindowFrame":
        return
    frame = QWidget()
    frame.setObjectName("ImodelToolWindowFrame")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(0)
    win.takeCentralWidget()
    lay.addWidget(central)
    win.setCentralWidget(frame)


def apply_dialog_style(dialog) -> None:
    dialog.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    dialog.setStyleSheet(IMODEL_QSS)


def section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("ImodelSectionTitle")
    return lbl


def hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("ImodelHint")
    lbl.setWordWrap(True)
    return lbl


def compact_status_panel(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("ImodelStatusPanelCompact")
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl


def status_panel(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("ImodelStatusPanel")
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl


def caption_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("ImodelCaption")
    return lbl


def primary_button(text: str) -> "QPushButton":
    from PySide6.QtWidgets import QPushButton

    btn = QPushButton(text)
    btn.setObjectName("ImodelPrimaryButton")
    return btn


def set_path_badge(label: QLabel, *, active: bool, text: str) -> None:
    label.setText(text)
    label.setObjectName("ImodelPathBadgeOpen" if active else "ImodelPathBadgeIdle")
    label.style().unpolish(label)
    label.style().polish(label)


def status_bar_message_label(text: str, level: str = "info") -> QLabel:
    lbl = QLabel(text)
    base = "ImodelStatusBarMessage"
    suffix = {
        "success": "Success",
        "failed": "Failed",
        "running": "Running",
        "warn": "Warn",
    }.get(level, "")
    lbl.setObjectName(base + suffix if suffix else base)
    return lbl


def polish_status_bar_label(label: QLabel, level: str = "info") -> None:
    suffix = {
        "success": "Success",
        "failed": "Failed",
        "running": "Running",
        "warn": "Warn",
    }.get(level, "")
    label.setObjectName(
        "ImodelStatusBarMessage" + suffix if suffix else "ImodelStatusBarMessage"
    )
    label.style().unpolish(label)
    label.style().polish(label)


def show_modeless_dialog(dlg) -> None:
    """Show a tool/preview dialog without blocking the parent window."""
    from PySide6.QtWidgets import QDialog

    if not isinstance(dlg, QDialog):
        raise TypeError("show_modeless_dialog expects a QDialog")
    dlg.setModal(False)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    _registry = getattr(show_modeless_dialog, "_registry", None)
    if _registry is None:
        _registry = []
        setattr(show_modeless_dialog, "_registry", _registry)
    _registry.append(dlg)

    def _on_finished(_code: int) -> None:
        try:
            _registry.remove(dlg)
        except ValueError:
            pass

    dlg.finished.connect(_on_finished)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
