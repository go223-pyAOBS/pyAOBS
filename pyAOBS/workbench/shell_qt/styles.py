"""Qt styles for Workbench — aligned with imodel_qt contrast (panels, text, chrome)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QLabel, QMainWindow, QTableWidgetItem, QVBoxLayout, QWidget

UI_FONT_PT = 13
LOG_FONT_PT = 12

# Palette (imodel-like: light shell + white panels + dark borders)
COLOR_SHELL = "#e0e4ea"
COLOR_PANEL = "#ffffff"
COLOR_PANEL_ALT = "#f0f3f7"
COLOR_BORDER = "#404040"
COLOR_BORDER_MED = "#606060"
COLOR_BORDER_LIGHT = "#a8adb8"
COLOR_TEXT = "#121212"
COLOR_TEXT_SECONDARY = "#2a2a2a"
COLOR_TEXT_MUTED = "#4a4a4a"
COLOR_TOOLBAR = "#cfd5de"
COLOR_TOOLBAR_BORDER = "#606060"
COLOR_ACCENT = "#2563a8"
COLOR_ACCENT_DARK = "#1a4a7a"
COLOR_STATUS_BG = "#e8edf4"
COLOR_TABLE_HEADER = "#c8d2de"
COLOR_SELECTION = "#b8d4f0"

# Run status semantics
COLOR_SUCCESS = "#166534"
COLOR_SUCCESS_BG = "#dcfce7"
COLOR_SUCCESS_BORDER = "#86efac"
COLOR_FAILED = "#991b1b"
COLOR_FAILED_BG = "#fee2e2"
COLOR_FAILED_BORDER = "#fca5a5"
COLOR_CANCELLED = "#92400e"
COLOR_CANCELLED_BG = "#fef3c7"
COLOR_CANCELLED_BORDER = "#fcd34d"
COLOR_RUNNING = "#1e40af"
COLOR_RUNNING_BG = "#dbeafe"
COLOR_RUNNING_BORDER = "#93c5fd"
COLOR_INFO = "#1e5080"
COLOR_INFO_BG = "#e0f2fe"

WORKBENCH_QSS = f"""
QMainWindow {{
    background-color: {COLOR_SHELL};
    color: {COLOR_TEXT};
    font-size: {UI_FONT_PT}px;
}}
QWidget {{
    color: {COLOR_TEXT};
    font-size: {UI_FONT_PT}px;
}}
QWidget#WorkbenchMainFrame {{
    background-color: {COLOR_SHELL};
    border: 3px solid {COLOR_BORDER};
    border-radius: 5px;
}}
QWidget#WorkbenchSidePanel, QWidget#WorkbenchTabPanel {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
QLabel#WorkbenchAppTitle {{
    color: {COLOR_TEXT};
    font-size: {UI_FONT_PT + 2}px;
    font-weight: bold;
    padding: 0 12px 0 4px;
    border-right: 1px solid {COLOR_BORDER_LIGHT};
    margin-right: 8px;
}}
QLabel#WorkbenchSectionTitle {{
    color: {COLOR_TEXT};
    font-weight: bold;
    font-size: {UI_FONT_PT + 2}px;
    padding: 4px 0 8px 0;
    border-bottom: 3px solid {COLOR_ACCENT};
    margin-bottom: 6px;
}}
QLabel#WorkbenchProjectBadge {{
    color: {COLOR_TEXT};
    font-weight: bold;
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
    padding: 5px 12px;
}}
QLabel#WorkbenchProjectBadgeOpen {{
    color: {COLOR_SUCCESS};
    font-weight: bold;
    background-color: {COLOR_SUCCESS_BG};
    border: 2px solid {COLOR_SUCCESS_BORDER};
    border-left: 5px solid {COLOR_SUCCESS};
    border-radius: 4px;
    padding: 5px 12px;
}}
QLabel#WorkbenchProjectBadgeIdle {{
    color: {COLOR_TEXT_MUTED};
    font-weight: bold;
    background-color: {COLOR_PANEL_ALT};
    border: 2px dashed {COLOR_BORDER_LIGHT};
    border-radius: 4px;
    padding: 5px 12px;
}}
QLabel#WorkbenchRunSummary {{
    font-weight: 600;
    font-size: {UI_FONT_PT - 1}px;
    padding: 6px 10px;
    border-radius: 4px;
    border: 1px solid {COLOR_BORDER_LIGHT};
    background-color: {COLOR_PANEL_ALT};
}}
QLabel#WorkbenchStatusSuccess {{
    color: {COLOR_SUCCESS};
    font-weight: bold;
    background-color: {COLOR_SUCCESS_BG};
    border: 1px solid {COLOR_SUCCESS_BORDER};
    border-radius: 3px;
    padding: 2px 8px;
}}
QLabel#WorkbenchStatusFailed {{
    color: {COLOR_FAILED};
    font-weight: bold;
    background-color: {COLOR_FAILED_BG};
    border: 1px solid {COLOR_FAILED_BORDER};
    border-radius: 3px;
    padding: 2px 8px;
}}
QLabel#WorkbenchStatusCancelled {{
    color: {COLOR_CANCELLED};
    font-weight: bold;
    background-color: {COLOR_CANCELLED_BG};
    border: 1px solid {COLOR_CANCELLED_BORDER};
    border-radius: 3px;
    padding: 2px 8px;
}}
QLabel#WorkbenchStatusRunning {{
    color: {COLOR_RUNNING};
    font-weight: bold;
    background-color: {COLOR_RUNNING_BG};
    border: 1px solid {COLOR_RUNNING_BORDER};
    border-radius: 3px;
    padding: 2px 8px;
}}
QLabel#WorkbenchStatusInfo {{
    color: {COLOR_INFO};
    font-weight: 600;
    background-color: {COLOR_INFO_BG};
    border: 1px solid #7dd3fc;
    border-radius: 3px;
    padding: 2px 8px;
}}
QLabel#WorkbenchHint {{
    color: {COLOR_TEXT_MUTED};
    font-size: {UI_FONT_PT - 1}px;
}}
QLabel#WorkbenchStatusPanel {{
    font-family: Consolas, "Courier New", monospace;
    font-size: {LOG_FONT_PT}px;
    color: {COLOR_TEXT_SECONDARY};
    background-color: {COLOR_STATUS_BG};
    border: 2px solid {COLOR_BORDER_LIGHT};
    border-left: 4px solid {COLOR_ACCENT};
    border-radius: 3px;
    padding: 10px;
}}
QGroupBox {{
    font-weight: bold;
    font-size: {UI_FONT_PT}px;
    color: {COLOR_TEXT};
    background-color: {COLOR_PANEL_ALT};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
}}
QGroupBox#WorkbenchCommandGroup {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_ACCENT};
    border-left: 5px solid {COLOR_ACCENT};
}}
QGroupBox#WorkbenchBridgeGroup {{
    background-color: {COLOR_INFO_BG};
    border: 2px solid #7dd3fc;
    border-left: 5px solid {COLOR_ACCENT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 8px;
    color: {COLOR_TEXT};
}}
QToolBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #dce2ea, stop:1 {COLOR_TOOLBAR});
    border-bottom: 3px solid {COLOR_TOOLBAR_BORDER};
    spacing: 8px;
    padding: 8px 10px;
}}
QToolBar QToolButton {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_LIGHT};
    border-radius: 4px;
    padding: 7px 14px;
    font-weight: 700;
    min-height: 28px;
}}
QToolBar QToolButton:hover {{
    background-color: #dce8f8;
    border-color: {COLOR_ACCENT};
    color: {COLOR_ACCENT_DARK};
}}
QToolBar QToolButton:pressed {{
    background-color: {COLOR_SELECTION};
    border-color: {COLOR_ACCENT_DARK};
}}
QTabWidget::pane {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_MED};
    border-radius: 0 0 4px 4px;
    top: -1px;
}}
QTabBar::tab {{
    background: {COLOR_TOOLBAR};
    color: {COLOR_TEXT_SECONDARY};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 2px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background: {COLOR_PANEL};
    color: {COLOR_ACCENT_DARK};
    border-color: {COLOR_BORDER_MED};
    border-bottom: 3px solid {COLOR_ACCENT};
    font-weight: 700;
}}
QTabBar::tab:hover:!selected {{
    background: #eceff3;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    min-height: 30px;
    padding: 3px 8px;
    selection-background-color: {COLOR_SELECTION};
}}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 2px solid {COLOR_ACCENT};
    background-color: #fafcff;
}}
QComboBox::drop-down {{
    border-left: 1px solid {COLOR_BORDER_LIGHT};
    width: 22px;
}}
QPlainTextEdit, QTextEdit {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    font-family: Consolas, "Courier New", monospace;
    font-size: {LOG_FONT_PT}px;
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    padding: 4px;
    selection-background-color: {COLOR_SELECTION};
}}
QTreeWidget {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    alternate-background-color: {COLOR_PANEL_ALT};
    outline: none;
}}
QTreeWidget::item {{
    padding: 3px 2px;
}}
QTreeWidget::item:selected {{
    background-color: {COLOR_SELECTION};
    color: {COLOR_TEXT};
}}
QTreeWidget::item:hover {{
    background-color: #e8eef5;
}}
QTableWidget {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    gridline-color: {COLOR_BORDER_LIGHT};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 3px;
    alternate-background-color: {COLOR_PANEL_ALT};
    selection-background-color: {COLOR_SELECTION};
    selection-color: {COLOR_TEXT};
}}
QHeaderView::section {{
    background-color: {COLOR_TABLE_HEADER};
    color: {COLOR_TEXT};
    font-weight: bold;
    font-size: {UI_FONT_PT}px;
    padding: 8px 10px;
    border: none;
    border-right: 1px solid {COLOR_BORDER_LIGHT};
    border-bottom: 3px solid {COLOR_BORDER_MED};
}}
QPushButton {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 700;
    min-height: 30px;
}}
QPushButton:hover {{
    background-color: #e8f0fa;
    border-color: {COLOR_ACCENT};
}}
QPushButton:pressed {{
    background-color: {COLOR_SELECTION};
}}
QPushButton:disabled {{
    color: #888888;
    background-color: #ececec;
    border-color: #cccccc;
}}
QPushButton#WorkbenchPrimaryButton {{
    background-color: {COLOR_ACCENT};
    color: #ffffff;
    border: 2px solid {COLOR_ACCENT_DARK};
    font-weight: bold;
    min-height: 34px;
    padding: 8px 18px;
}}
QPushButton#WorkbenchPrimaryButton:disabled {{
    background-color: #8aaec8;
    color: #eeeeee;
    border-color: #6a8aa8;
}}
QPushButton#WorkbenchPrimaryButton:hover {{
    background-color: #2f7bc4;
}}
QPushButton#WorkbenchDangerButton {{
    background-color: {COLOR_FAILED_BG};
    color: {COLOR_FAILED};
    border: 2px solid {COLOR_FAILED_BORDER};
}}
QPushButton#WorkbenchDangerButton:hover {{
    background-color: #fecaca;
    border-color: {COLOR_FAILED};
}}
QPushButton#WorkbenchWarnButton {{
    background-color: {COLOR_CANCELLED_BG};
    color: {COLOR_CANCELLED};
    border: 2px solid {COLOR_CANCELLED_BORDER};
}}
QPushButton#WorkbenchWarnButton:hover {{
    background-color: #fde68a;
}}
QStatusBar {{
    background: {COLOR_TOOLBAR};
    color: {COLOR_TEXT_SECONDARY};
    border-top: 3px solid {COLOR_TOOLBAR_BORDER};
    font-size: {UI_FONT_PT}px;
    font-weight: 600;
    padding: 4px 8px;
}}
QLabel#WorkbenchStatusBarMessage {{
    font-weight: 600;
    padding: 2px 6px;
}}
QLabel#WorkbenchStatusBarMessageSuccess {{
    color: {COLOR_SUCCESS};
    font-weight: bold;
}}
QLabel#WorkbenchStatusBarMessageFailed {{
    color: {COLOR_FAILED};
    font-weight: bold;
}}
QLabel#WorkbenchStatusBarMessageRunning {{
    color: {COLOR_RUNNING};
    font-weight: bold;
}}
QLabel#WorkbenchStatusBarMessageWarn {{
    color: {COLOR_CANCELLED};
    font-weight: bold;
}}
QScrollBar:vertical {{
    background: {COLOR_PANEL_ALT};
    width: 12px;
    border: 1px solid {COLOR_BORDER_LIGHT};
}}
QScrollBar::handle:vertical {{
    background: {COLOR_BORDER_LIGHT};
    min-height: 24px;
    border-radius: 4px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLOR_ACCENT};
}}
QCheckBox {{
    spacing: 6px;
    color: {COLOR_TEXT};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLOR_BORDER_MED};
    border-radius: 2px;
    background: {COLOR_PANEL};
}}
QCheckBox::indicator:checked {{
    background: {COLOR_ACCENT};
    border-color: #1e5080;
}}
QSplitter::handle {{
    background-color: {COLOR_BORDER_MED};
    width: 5px;
    height: 5px;
}}
QSplitter::handle:hover {{
    background-color: {COLOR_ACCENT};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QStatusBar QLabel {{
    color: {COLOR_TEXT_SECONDARY};
}}
QFormLayout QLabel {{
    color: {COLOR_TEXT_SECONDARY};
    font-weight: 700;
}}
"""

_STATUS_OBJECT_NAMES = {
    "success": "WorkbenchStatusSuccess",
    "failed": "WorkbenchStatusFailed",
    "cancelled": "WorkbenchStatusCancelled",
    "running": "WorkbenchStatusRunning",
}


def normalize_run_status(status: str) -> str:
    s = str(status or "").strip().lower()
    if s in ("success", "ok", "completed"):
        return "success"
    if s in ("failed", "error", "failure"):
        return "failed"
    if s in ("cancelled", "canceled"):
        return "cancelled"
    if s in ("running", "started", "active"):
        return "running"
    return s or "unknown"


def run_status_fg_bg(status: str) -> tuple[QColor, QColor]:
    key = normalize_run_status(status)
    if key == "success":
        return QColor(COLOR_SUCCESS), QColor(COLOR_SUCCESS_BG)
    if key == "failed":
        return QColor(COLOR_FAILED), QColor(COLOR_FAILED_BG)
    if key == "cancelled":
        return QColor(COLOR_CANCELLED), QColor(COLOR_CANCELLED_BG)
    if key == "running":
        return QColor(COLOR_RUNNING), QColor(COLOR_RUNNING_BG)
    return QColor(COLOR_TEXT), QColor(COLOR_PANEL_ALT)


def style_table_item_for_run_status(item: QTableWidgetItem, status: str, *, status_column: bool = False) -> None:
    fg, bg = run_status_fg_bg(status)
    item.setForeground(QBrush(fg))
    if status_column:
        item.setBackground(QBrush(bg))
        f = item.font()
        f.setBold(True)
        item.setFont(f)


def style_table_row_for_run_status(table, row: int, status: str, status_col: int = 2) -> None:
    fg, bg = run_status_fg_bg(status)
    key = normalize_run_status(status)
    for col in range(table.columnCount()):
        it = table.item(row, col)
        if it is None:
            continue
        it.setForeground(QBrush(fg if col == status_col else QColor(COLOR_TEXT)))
        if key in _STATUS_OBJECT_NAMES:
            it.setBackground(QBrush(bg if col == status_col else QColor(COLOR_PANEL if row % 2 == 0 else COLOR_PANEL_ALT)))


def run_summary_text(records: list[dict]) -> str:
    counts = {"success": 0, "failed": 0, "cancelled": 0, "other": 0}
    for rec in records:
        key = normalize_run_status(str(rec.get("status", "")))
        if key in counts:
            counts[key] += 1
        else:
            counts["other"] += 1
    parts = [
        f"共 {len(records)} 条",
        f"成功 {counts['success']}",
        f"失败 {counts['failed']}",
        f"取消 {counts['cancelled']}",
    ]
    if counts["other"]:
        parts.append(f"其他 {counts['other']}")
    return "  ·  ".join(parts)


def run_summary_label(records: list[dict]) -> QLabel:
    lbl = QLabel(run_summary_text(records))
    lbl.setObjectName("WorkbenchRunSummary")
    failed = sum(1 for r in records if normalize_run_status(str(r.get("status", ""))) == "failed")
    if failed:
        lbl.setStyleSheet(
            f"color: {COLOR_FAILED}; background-color: {COLOR_FAILED_BG}; "
            f"border: 1px solid {COLOR_FAILED_BORDER};"
        )
    return lbl


def status_badge(status: str) -> QLabel:
    key = normalize_run_status(status)
    lbl = QLabel(str(status or "—"))
    lbl.setObjectName(_STATUS_OBJECT_NAMES.get(key, "WorkbenchStatusInfo"))
    return lbl


def status_bar_message_label(text: str, level: str = "info") -> QLabel:
    """level: info | success | failed | running | warn"""
    lbl = QLabel(text)
    base = "WorkbenchStatusBarMessage"
    suffix = {
        "success": "Success",
        "failed": "Failed",
        "running": "Running",
        "warn": "Warn",
    }.get(level, "")
    lbl.setObjectName(base + suffix if suffix else base)
    return lbl


def set_project_badge_style(label: QLabel, *, project_open: bool, path_text: str = "") -> None:
    if project_open and path_text:
        label.setObjectName("WorkbenchProjectBadgeOpen")
        label.setText(f"项目: {path_text}")
    elif project_open:
        label.setObjectName("WorkbenchProjectBadgeOpen")
    else:
        label.setObjectName("WorkbenchProjectBadgeIdle")
        label.setText("当前项目：未打开")
    label.style().unpolish(label)
    label.style().polish(label)


def apply_workbench_font(app) -> None:
    font = QFont()
    font.setPointSize(UI_FONT_PT)
    app.setFont(font)


def apply_workbench_chrome(win: QMainWindow) -> None:
    """Main window chrome: imodel-style outer frame around central content."""
    win.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    win.setStyleSheet(WORKBENCH_QSS)

    central = win.centralWidget()
    if central is None or central.objectName() == "WorkbenchMainFrame":
        return
    frame = QWidget()
    frame.setObjectName("WorkbenchMainFrame")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(0)
    win.takeCentralWidget()
    lay.addWidget(central)
    win.setCentralWidget(frame)


def section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("WorkbenchSectionTitle")
    return lbl


def hint_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("WorkbenchHint")
    lbl.setWordWrap(True)
    return lbl


def status_panel(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("WorkbenchStatusPanel")
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl
