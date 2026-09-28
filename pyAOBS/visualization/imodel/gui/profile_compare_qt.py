"""多条 1D Vp–depth 曲线同窗对比（非模态）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .plot_nav import PyqtgraphStyleNav
from .profile_dialog_qt import _AspectRatioHost, _PROFILE_ASPECT
from .profile_ops import DATUM_LABELS
from .styles import (
    apply_dialog_style,
    hint_label,
    show_modeless_dialog,
    show_modeless_message,
    status_panel,
)

_COLORS = (
    "#e74c3c",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#d35400",
    "#16a085",
    "#c0392b",
    "#2c3e50",
    "#7f8c8d",
    "#f39c12",
)


def _series_arrays(series: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(series.get("depth"), dtype=float).ravel()
    vp = np.asarray(series.get("vp"), dtype=float).ravel()
    n = min(depth.size, vp.size)
    return depth[:n], vp[:n]


class ProfileCompareDialog(QDialog):
    """叠绘多条已抽取的垂直剖面（纵横比与单剖面窗一致）。"""

    def __init__(
        self,
        parent: QWidget | None,
        series_list: Sequence[dict[str, Any]],
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self._series = [dict(s) for s in series_list]
        n = len(self._series)
        self.setWindowTitle(f"1D 对比 — {n} 条")
        # 与 VerticalProfileDialog 一致：竖长横窄
        self.resize(400, 620)
        self.setMinimumSize(280, 420)

        datums = {str(s.get("datum") or "z0") for s in self._series}
        mixed = len(datums) > 1
        title = f"Vp–depth compare (n={n})"
        if mixed:
            title += " ⚠ mixed Datum"

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        lines = [f"已叠绘 {n} 条剖面"]
        if mixed:
            labels = ", ".join(DATUM_LABELS.get(d, d) for d in sorted(datums))  # type: ignore[arg-type]
            lines.append(f"警告：Datum 混用（{labels}），深度零点不一致")
        else:
            d0 = next(iter(datums))
            lines.append(f"Datum: {DATUM_LABELS.get(d0, d0)}")  # type: ignore[arg-type]
        for i, s in enumerate(self._series[:10]):
            lines.append(f"  · {s.get('label') or f'#{i + 1}'}")
        if n > 10:
            lines.append(f"  · …共 {n} 条")
        root.addWidget(status_panel("\n".join(lines)))

        fig = Figure(figsize=(3.2, 5.0), dpi=100)
        fig.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.08)
        ax = fig.add_subplot(111)
        for i, s in enumerate(self._series):
            depth, vp = _series_arrays(s)
            if depth.size == 0:
                continue
            color = _COLORS[i % len(_COLORS)]
            lab = str(s.get("label") or f"#{i + 1}")
            ax.plot(vp, depth, color=color, lw=1.4, label=lab)
        ax.set_xlabel("Vp (km/s)")
        ax.invert_yaxis()
        ax.set_ylabel("Depth (km)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if any(str(s.get("label") or "") for s in self._series):
            ax.legend(loc="best", fontsize=7, framealpha=0.85)

        canvas = FigureCanvasQTAgg(fig)
        nav_row = QHBoxLayout()
        btn_reset = QPushButton("Reset View")
        btn_reset.setToolTip("双击图也可复位；滚轮缩放，左拖平移，右拖连续缩放")
        nav_row.addWidget(btn_reset)
        btn_save = QPushButton("Save Figure")
        btn_save.clicked.connect(self._save_figure)
        nav_row.addWidget(btn_save)
        btn_export = QPushButton("Export txt…")
        btn_export.setToolTip("将各系列分别导出为 depth/vp 文本")
        btn_export.clicked.connect(self._export_txt)
        nav_row.addWidget(btn_export)
        nav_row.addStretch(1)
        root.addLayout(nav_row)

        host = _AspectRatioHost(canvas, width_over_height=_PROFILE_ASPECT)
        root.addWidget(host, stretch=1)

        nav = PyqtgraphStyleNav(canvas)
        btn_reset.clicked.connect(nav.reset_view)
        canvas.draw()
        nav.remember_home_views()
        nav.remember_current_views()

        root.addWidget(
            hint_label(
                "At X / Average 成功后自动加入对比袋；侧栏「Clr 对比」清空。"
                "混用 Datum 时深度轴不可直接横向对比。"
            )
        )

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        apply_dialog_style(self)
        self._fig = fig

    def _save_figure(self) -> None:
        from .file_dialogs_qt import save_matplotlib_figure

        save_matplotlib_figure(
            self,
            self._fig,
            caption="Save Figure",
            default_stem="profile_compare",
        )

    def _export_txt(self) -> None:
        if not self._series:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "导出对比系列到目录")
        if not out_dir:
            return
        root = Path(out_dir)
        written: list[str] = []
        for i, s in enumerate(self._series):
            depth, vp = _series_arrays(s)
            if depth.size == 0:
                continue
            lab = str(s.get("label") or f"series_{i + 1}")
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in lab)[:80]
            path = root / f"V_1D_compare_{i + 1:02d}_{safe}.txt"
            with path.open("w", encoding="utf-8") as f:
                f.write("# depth_km  vp_km_s\n")
                for d, v in zip(depth, vp):
                    if np.isfinite(d) and np.isfinite(v):
                        f.write(f"{float(d):.6f}  {float(v):.6f}\n")
            written.append(path.name)
        show_modeless_message(
            "Export",
            f"已写入 {len(written)} 个文件到\n{root}",
        )


def show_profile_compare_dialog(
    parent: QWidget | None,
    series_list: Sequence[dict[str, Any]],
) -> None:
    if not series_list:
        show_modeless_message("1D 对比", "对比袋为空。请先用 At X / Average 抽取剖面。")
        return
    show_modeless_dialog(ProfileCompareDialog(parent, series_list))
