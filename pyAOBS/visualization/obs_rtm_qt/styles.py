# -*- coding: utf-8 -*-
"""OBS RTM Qt 样式：与 Workbench / imodel 同一套对比度（边框、字色、面板）。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.workbench.shell_qt.styles import (
    COLOR_ACCENT,
    COLOR_BORDER_LIGHT,
    COLOR_BORDER_MED,
    COLOR_PANEL,
    COLOR_PANEL_ALT,
    COLOR_SHELL,
    COLOR_TEXT,
    COLOR_TEXT_SECONDARY,
    UI_FONT_PT,
    WORKBENCH_QSS,
)

# 侧栏略小于正文，但仍需清晰可读（原 10pt 偏淡）
SIDE_PANEL_FONT_PT = 12

OBS_RTM_BASE_QSS = WORKBENCH_QSS.replace("Workbench", "ObsRtm")

OBS_RTM_EXTRA_QSS = f"""
QWidget#ObsRtmPlotPanel,
QWidget#ObsRtmPlotPanelDual {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
QWidget#ObsRtmPlotPanel QLabel,
QWidget#ObsRtmPlotPanelDual QLabel {{
    color: {COLOR_TEXT};
    font-weight: 700;
    font-size: {SIDE_PANEL_FONT_PT}px;
}}
QWidget#ObsRtmPlotPanel QCheckBox,
QWidget#ObsRtmPlotPanelDual QCheckBox {{
    color: {COLOR_TEXT};
    font-weight: 700;
    font-size: {SIDE_PANEL_FONT_PT}px;
}}
QWidget#ObsRtmLogPanel {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
QWidget#ObsRtmStageBar {{
    background-color: {COLOR_PANEL_ALT};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    padding: 2px;
}}
QTabBar::tab {{
    font-weight: 700;
    color: {COLOR_TEXT_SECONDARY};
}}
QTabBar::tab:selected {{
    font-weight: 800;
    color: {COLOR_TEXT};
}}
QLabel#ObsRtmSectionTitle {{
    color: {COLOR_TEXT};
    font-weight: 800;
    font-size: {UI_FONT_PT + 1}px;
    padding: 2px 0 4px 0;
    border-bottom: 3px solid {COLOR_ACCENT};
    margin-bottom: 2px;
}}
QLabel#ObsRtmHint {{
    color: {COLOR_TEXT_SECONDARY};
    font-weight: 600;
    font-size: {UI_FONT_PT - 1}px;
}}
QDialog {{
    background-color: {COLOR_SHELL};
    color: {COLOR_TEXT};
}}
QWidget#ObsRtmSidePanel {{
    font-size: {SIDE_PANEL_FONT_PT}px;
    color: {COLOR_TEXT};
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
QWidget#ObsRtmSidePanel QGroupBox {{
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 800;
    color: {COLOR_TEXT};
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
}}
QWidget#ObsRtmSidePanel QGroupBox::title {{
    padding: 0 6px;
    subcontrol-origin: margin;
    subcontrol-position: top left;
    color: {COLOR_TEXT};
    font-weight: 800;
}}
QWidget#ObsRtmSidePanel QPushButton {{
    min-height: 24px;
    max-height: 30px;
    padding: 2px 8px;
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 700;
    color: {COLOR_TEXT};
}}
QWidget#ObsRtmSidePanel QPushButton#ObsRtmPrimaryButton {{
    min-height: 28px;
    max-height: 34px;
    padding: 4px 10px;
    color: #ffffff;
    font-weight: 800;
}}
QWidget#ObsRtmSidePanel QLineEdit {{
    min-height: 24px;
    max-height: 28px;
    padding: 0 4px;
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 600;
    color: {COLOR_TEXT};
}}
/* SpinBox 勿设 max-height：Windows 上会压扁内部 lineEdit，导致粘贴/选中异常 */
QWidget#ObsRtmSidePanel QSpinBox,
QWidget#ObsRtmSidePanel QDoubleSpinBox {{
    min-height: 24px;
    padding: 0 4px;
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 600;
    color: {COLOR_TEXT};
}}
/* 勿给 QComboBox 设 max-height：Windows 上会压扁/锁死弹层，选一次后再点打不开 */
QWidget#ObsRtmSidePanel QComboBox {{
    min-height: 26px;
    padding: 0 4px;
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 600;
    color: {COLOR_TEXT};
    combobox-popup: 0;
}}
/* 勿给 ItemView 设过大 min-height，否则弹层顶部会出现空白带 */
QWidget#ObsRtmSidePanel QComboBox QAbstractItemView {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER_MED};
    selection-background-color: {COLOR_ACCENT};
    selection-color: #ffffff;
    outline: 0;
    padding: 0px;
    margin: 0px;
}}
QWidget#ObsRtmSidePanel QComboBox QAbstractItemView::item {{
    min-height: 22px;
    max-height: 28px;
    padding: 1px 6px;
    margin: 0px;
    color: {COLOR_TEXT};
}}
QWidget#ObsRtmSidePanel QCheckBox {{
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 700;
    color: {COLOR_TEXT};
    spacing: 6px;
}}
QWidget#ObsRtmSidePanel QLabel {{
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 700;
    color: {COLOR_TEXT};
}}
QWidget#ObsRtmSidePanel QLabel#ObsRtmHint {{
    font-size: {SIDE_PANEL_FONT_PT}px;
    font-weight: 600;
    color: {COLOR_TEXT_SECONDARY};
}}
"""

OBS_RTM_QSS = OBS_RTM_BASE_QSS + OBS_RTM_EXTRA_QSS


def apply_obs_rtm_font(app) -> None:
    font = QFont()
    font.setPointSize(UI_FONT_PT)
    app.setFont(font)


def apply_obs_rtm_chrome(win: QMainWindow) -> None:
    """主窗口：Workbench 风格外框 + 全局 QSS。"""
    win.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    win.setStyleSheet(OBS_RTM_QSS)

    from pyAOBS.utils.qt_independent_window import mark_chrome_frame

    central = win.centralWidget()
    if central is None:
        return
    if central.objectName() == "ObsRtmMainFrame":
        mark_chrome_frame(central)
        return
    frame = QWidget()
    frame.setObjectName("ObsRtmMainFrame")
    mark_chrome_frame(frame)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(0)
    win.takeCentralWidget()
    lay.addWidget(central)
    win.setCentralWidget(frame)


def section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("ObsRtmSectionTitle")
    return lbl


def hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("ObsRtmHint")
    lbl.setWordWrap(True)
    return lbl


def primary_button(text: str):
    from PySide6.QtWidgets import QPushButton

    btn = QPushButton(text)
    btn.setObjectName("ObsRtmPrimaryButton")
    return btn


def compact_form(form: QFormLayout) -> QFormLayout:
    """侧栏统一表单：右对齐标签列 + 紧凑间距，避免各面板参差。"""
    form.setContentsMargins(4, 4, 4, 4)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(4)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setLabelAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    form.setFormAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
    )
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
    return form


def side_panel_layout(panel: QWidget) -> QVBoxLayout:
    """ObsRtmSidePanel 统一外边距/间距。"""
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(6)
    return lay

