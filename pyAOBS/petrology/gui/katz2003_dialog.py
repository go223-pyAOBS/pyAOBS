"""Katz et al. (2003) Fig. 1–9 preview — LIP validation workflow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox

from petrology.validation.katz2003_workflow import (
    FIG_DIR,
    KATZ2003_PANELS,
    build_katz2003_figure,
    export_katz2003_figures,
)

from .app import UI_FONT_PT
from .tabbed_figure_dialog import TabbedFigureDialog


class Katz2003Dialog(TabbedFigureDialog):
    """Tabbed Katz (2003) Fig. 1–9 preview for LIP melting validation."""

    def __init__(
        self,
        *,
        tp_c: float | None = None,
        per_h2o_wt: float | None = None,
    ) -> None:
        self._tp_c = tp_c
        self._per_h2o_wt = per_h2o_wt
        super().__init__(
            window_title="Katz (2003) 图件 — LIP 熔融验证",
            error_title="Katz (2003)",
            fig_dir=FIG_DIR,
        )

    def _build_hint(self) -> QLabel:
        ctx_parts = [
            "Katz et al. (2003) Table 2 参数化 — 对应 LIP 橄榄石端元 (katz_lherzolite)。",
            "CLI: petrology/validation/katz2003_workflow.py",
        ]
        if self._tp_c is not None:
            ctx_parts.append(f"当前 Tp = {self._tp_c:g} °C")
        if self._per_h2o_wt is not None:
            ctx_parts.append(f"橄榄石 bulk H₂O = {self._per_h2o_wt:g} wt%")
        hint = QLabel("  ".join(ctx_parts))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: #555; font-size: {UI_FONT_PT - 1}pt;")
        return hint

    def _panel_items(self):
        return [(p.key, p.tab_title) for p in KATZ2003_PANELS]

    def _build_figure(self, key: str):
        return build_katz2003_figure(key)

    def _default_save_path(self, idx: int) -> str:
        return str(FIG_DIR / KATZ2003_PANELS[idx].filename)

    def _save_all_button_text(self) -> str | None:
        return "导出 Fig.1–9…"

    def _on_save_all(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", str(FIG_DIR))
        if not out_dir:
            return
        paths = export_katz2003_figures(Path(out_dir))
        QMessageBox.information(
            self,
            self._error_title,
            f"已导出 {len(paths)} 张图至:\n{out_dir}",
        )
