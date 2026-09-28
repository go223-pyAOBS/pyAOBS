# -*- coding: utf-8 -*-
from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..styles import hint_label, section_title


class PlaceholderPanel(QWidget):
    def __init__(self, title: str, body: str, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.addWidget(section_title(title))
        lay.addWidget(hint_label(body))
        lay.addStretch(1)
