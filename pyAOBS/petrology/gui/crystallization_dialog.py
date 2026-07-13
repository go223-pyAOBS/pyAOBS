"""KKHS02 Step 3 结晶图件预览：Fig.2 路径 + Fig.5 ΔVp(F)。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .app import UI_FONT_PT
from .dialog_utils import show_modeless_dialog

_FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

_PANELS: tuple[tuple[str, str, Path, str], ...] = (
    (
        "fig2",
        "Fig.2 结晶路径",
        _FIG_DIR / "reproduce_fig2_wl1990.png",
        "单条初始熔体的分离结晶路径：堆晶 / 增量固相 / 残余熔体 Vp 与矿物比例（W&L 1990）。"
        "对应公式查阅 → Step 3「分离结晶 ΔVp(F)」的路径示意。",
    ),
    (
        "fig5",
        "Fig.5 ΔVp(F)",
        _FIG_DIR / "reproduce_fig5_wl1990.png",
        "原文四联图：横轴 V_LC / V_UC，纵轴相对 V_bulk 的偏差；细线=各熔体 FC 路径；"
        "空心/实心点 = F_xl=0.5 / 0.8（P_fc=100、400 MPa；报告态 600 MPa / 400 °C）。"
        "裸算 ΔVp_LC @ F≈0.7–0.8 ≈0.13–0.15 → GUI「ΔVp」与 Step-2 竖段。",
    ),
)

_CAPTION = (
    "Step 3 — 分离结晶（KKHS02 Fig.2 / Fig.5）。"
    "熔融柱给出假想 bulk Vp（方程 (1)）后，结晶使可观测下地壳相对变慢；"
    "读图时用 [V_LC−ΔVp, V_LC] 竖段。"
    "图件来自 validation/reproduce_fig2.py 与 reproduce_fig5.py（已缓存 PNG；重新生成可能较慢）。"
)


class CrystallizationDialog(QDialog):
    """Non-modal preview of KKHS02 crystallization figures (Fig.2 + Fig.5)."""

    def __init__(self) -> None:
        super().__init__(None)
        self.setModal(False)
        self.setWindowTitle("分离结晶图件 — Fig.2 / Fig.5")
        self.resize(980, 780)
        self.setMinimumSize(720, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        hint = QLabel(_CAPTION)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: {UI_FONT_PT - 1}pt; color: #444;")
        root.addWidget(hint)

        self._tabs = QTabWidget()
        self._img_labels: dict[str, QLabel] = {}
        self._panel_by_key = {k: (title, path, blurb) for k, title, path, blurb in _PANELS}
        for key, title, path, blurb in _PANELS:
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(4, 4, 4, 4)
            note = QLabel(blurb)
            note.setWordWrap(True)
            note.setStyleSheet(f"font-size: {UI_FONT_PT - 1}pt; color: #555;")
            lay.addWidget(note)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img = QLabel()
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll.setWidget(img)
            lay.addWidget(scroll, stretch=1)
            self._img_labels[key] = img
            self._tabs.addTab(page, title)
            self._load_panel(key, path)
        root.addWidget(self._tabs, stretch=1)

        bar = QHBoxLayout()
        btn_regen = QPushButton("重新生成当前页…")
        btn_regen.setToolTip("调用对应 reproduce_fig*.py 覆盖 figures/ 中的 PNG（Fig.5 可能较慢）")
        btn_regen.clicked.connect(self._regenerate_current)
        bar.addWidget(btn_regen)
        bar.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        bar.addWidget(btn_close)
        root.addLayout(bar)

    def _load_panel(self, key: str, path: Path) -> None:
        img = self._img_labels[key]
        if not path.is_file():
            img.setText(
                f"尚未生成图像：\n{path.name}\n\n"
                f"点击「重新生成当前页…」或运行：\n"
                f"python petrology/validation/reproduce_{key}.py"
            )
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            img.setText(f"无法读取：\n{path}")
            return
        img.setPixmap(pix)
        img.setToolTip(str(path))

    def _regenerate_current(self) -> None:
        idx = self._tabs.currentIndex()
        if idx < 0 or idx >= len(_PANELS):
            return
        key, title, path, _blurb = _PANELS[idx]
        try:
            if key == "fig2":
                from petrology.validation.reproduce_fig2 import run as run_fig2

                run_fig2(
                    engine="wl1990",
                    mineral_backend="fig2",
                    cipw_backend="fallback",
                    save_figure=path,
                    show=False,
                )
            else:
                from petrology.validation.reproduce_fig5 import run as run_fig5

                run_fig5(
                    mineral_backend="sb1994_fig2ol",
                    kd_engine="langmuir",
                    save_figure=path,
                    show=False,
                )
        except Exception as exc:
            QMessageBox.critical(self, "分离结晶", f"重新生成 {title} 失败：\n{exc}")
            return
        self._load_panel(key, path)
        QMessageBox.information(self, "分离结晶", f"已更新：\n{path}")


def show_crystallization_dialog() -> None:
    show_modeless_dialog(CrystallizationDialog(), singleton=True)
