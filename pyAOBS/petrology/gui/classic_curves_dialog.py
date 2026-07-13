"""Non-modal preview for classic melting theory curves (no H / Vp)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QMessageBox

from petrology.validation.plot_classic_melting_curves import (
    CLASSIC_CURVE_PANELS,
    FIG_DIR,
    build_classic_curve_figure,
)

from .app import UI_FONT_PT
from .tabbed_figure_dialog import TabbedFigureDialog

_CLASSIC_EXPORT_NAMES = {
    "katz_fig1": "00_katz2003_fig1.png",
    "katz_fig2": "00_katz2003_fig2.png",
    "katz_fig3": "00_katz2003_fig3.png",
    "katz_fig4": "00_katz2003_fig4.png",
    "katz_fig5": "00_katz2003_fig5.png",
    "katz_fig6": "00_katz2003_fig6.png",
    "katz_fig7": "00_katz2003_fig7.png",
    "katz_fig8": "00_katz2003_fig8.png",
    "tp": "01_tp_solidus_adiabat.png",
    "fdp": "02_f_vs_p_decompression.png",
    "ft": "03_f_vs_t_isobar.png",
    "solidus": "04_lithology_solidus_catalog.png",
    "dfdp": "05_dfdp_adiabat.png",
}


class ClassicCurvesDialog(TabbedFigureDialog):
    """Tabbed matplotlib preview of classic melting panels."""

    def __init__(self, *, tp_c: float = 1350.0) -> None:
        self._tp_edit = QLineEdit(f"{tp_c:g}")
        super().__init__(
            window_title="经典熔融理论曲线",
            error_title="经典曲线",
            fig_dir=FIG_DIR,
        )

    def _extend_toolbar(self, bar: QHBoxLayout) -> None:
        bar.addWidget(QLabel("Tp (°C)"))
        self._tp_edit.setFixedWidth(90)
        self._tp_edit.setToolTip("用于 F(P) 与 dF/dP 面板；T–P / F(T) / 固相线目录与 Tp 无关")
        bar.addWidget(self._tp_edit)

    def _build_hint(self) -> QLabel:
        hint = QLabel(
            "Katz/G2 固相线、绝热线、等熵减压与 batch melting — 不含 H、V_LC。"
            f"  CLI: petrology/validation/plot_classic_melting_curves.py"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: #555; font-size: {UI_FONT_PT - 1}pt;")
        return hint

    def _panel_items(self):
        return CLASSIC_CURVE_PANELS

    def _build_figure(self, key: str):
        return build_classic_curve_figure(key, tp_c=self._read_tp())

    def _read_tp(self) -> float:
        return float(self._tp_edit.text().strip())

    def _on_refresh(self) -> None:
        try:
            self._read_tp()
        except ValueError:
            QMessageBox.warning(self, self._error_title, "Tp 须为有效数字。")
            return
        super()._on_refresh()

    def _default_save_path(self, idx: int) -> str:
        key = self._tab_keys[idx]
        return str(FIG_DIR / f"classic_{key}.png")

    def _save_all_button_text(self) -> str | None:
        return "导出全部 PNG…"

    def _on_save_all(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", str(FIG_DIR))
        if not out_dir:
            return
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        for key, fig in zip(self._tab_keys, self._figures):
            fig.savefig(base / _CLASSIC_EXPORT_NAMES.get(key, f"{key}.png"), dpi=150, bbox_inches="tight")
        QMessageBox.information(self, self._error_title, f"已导出 {len(self._figures)} 张图至:\n{base}")
