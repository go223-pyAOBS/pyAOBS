"""Dialog: Modern REEBOX vs H–Vp (eq.(1)) computed track."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from petrology.hvp.track_compare import compare_tracks_at_grid, format_track_compare_table
from petrology.hvp.track_registry import TRACK_REGISTRY

from .app import LOG_FONT_PT, UI_FONT_PT


class TrackCompareDialog(QDialog):
    def __init__(
        self,
        *,
        scan_result,
    ) -> None:
        super().__init__(None)
        self.setModal(False)
        self.setWindowTitle("轨对比 — Modern vs H–Vp 图")
        self.resize(780, 520)

        root = QVBoxLayout(self)
        spec_lines = []
        for tid, spec in TRACK_REGISTRY.items():
            spec_lines.append(f"• {spec.label}: {spec.melting} | Vp: {spec.vp_model}")
        hint = QLabel("\n".join(spec_lines))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: {UI_FONT_PT - 1}pt; color: #444;")
        root.addWidget(hint)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFontFamily("Consolas")
        self._text.setStyleSheet(f"font-size: {LOG_FONT_PT}pt;")
        root.addWidget(self._text, stretch=1)

        rows = compare_tracks_at_grid(scan_result.points, vp_bias_km_s=0.0)
        self._text.setPlainText(format_track_compare_table(rows))

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)
