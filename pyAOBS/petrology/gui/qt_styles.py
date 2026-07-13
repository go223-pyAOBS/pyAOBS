"""Qt dialog styling — re-export imodel_qt styles or local fallback."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton

try:
    from visualization.imodel_qt.styles import (
        COLOR_ACCENT,
        COLOR_BORDER_LIGHT,
        COLOR_BORDER_MED,
        COLOR_PANEL,
        COLOR_PANEL_ALT,
        COLOR_STATUS_BG,
        COLOR_TEXT,
        COLOR_TEXT_MUTED,
        COLOR_TEXT_SECONDARY,
        apply_dialog_style,
        hint_label,
        primary_button,
    )
except ImportError:
    COLOR_SHELL = "#e0e4ea"
    COLOR_ACCENT = "#2563a8"
    COLOR_BORDER_LIGHT = "#a8adb8"
    COLOR_BORDER_MED = "#606060"
    COLOR_PANEL = "#ffffff"
    COLOR_PANEL_ALT = "#f0f3f7"
    COLOR_STATUS_BG = "#e8edf4"
    COLOR_TEXT = "#121212"
    COLOR_TEXT_MUTED = "#4a4a4a"
    COLOR_TEXT_SECONDARY = "#2a2a2a"

    def apply_dialog_style(dialog) -> None:
        dialog.setStyleSheet(
            f"QDialog {{ background-color: {COLOR_SHELL}; color: {COLOR_TEXT}; }}"
        )

    def hint_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        return lbl

    def primary_button(text: str) -> QPushButton:
        return QPushButton(text)

__all__ = [
    "COLOR_ACCENT",
    "COLOR_BORDER_LIGHT",
    "COLOR_BORDER_MED",
    "COLOR_PANEL",
    "COLOR_PANEL_ALT",
    "COLOR_STATUS_BG",
    "COLOR_TEXT",
    "COLOR_TEXT_MUTED",
    "COLOR_TEXT_SECONDARY",
    "apply_dialog_style",
    "hint_label",
    "primary_button",
]
