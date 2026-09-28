"""一维垂直剖面预览窗：平均线 + 范围包络，保存 V_1D_from_{sf|bm|z0}_….txt。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PySide6.QtCore import QSize
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .plot_nav import PyqtgraphStyleNav
from .profile_ops import (
    DepthDatum,
    DATUM_LABELS,
    VerticalProfileBundle,
    format_v1d_filename,
    write_v1d_envelope_txt,
    write_v1d_txt,
)
from .styles import show_modeless_dialog, show_modeless_message

# 竖长横窄：width / height（与 figsize 3.2×5.0 一致）
_PROFILE_ASPECT = 3.2 / 5.0


class _AspectRatioHost(QWidget):
    """在可用区域内居中放置子控件，并保持固定宽高比（最大化时同步放大）。"""

    def __init__(self, child: QWidget, *, width_over_height: float = _PROFILE_ASPECT) -> None:
        super().__init__()
        self._child = child
        self._ratio = float(width_over_height)
        child.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        child.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def sizeHint(self) -> QSize:
        return QSize(320, int(320 / self._ratio))

    def minimumSizeHint(self) -> QSize:
        return QSize(200, int(200 / self._ratio))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        aw = max(self.width(), 1)
        ah = max(self.height(), 1)
        if aw / ah > self._ratio:
            h = ah
            w = max(1, int(round(h * self._ratio)))
        else:
            w = aw
            h = max(1, int(round(w / self._ratio)))
        x = (aw - w) // 2
        y = (ah - h) // 2
        self._child.setGeometry(x, y, w, h)


class VerticalProfileDialog(QDialog):
    """显示垂直剖面 Vp–深度；范围模式绘制包络；可导出 depth/vp（及包络）txt。"""

    def __init__(
        self,
        parent: QWidget,
        profile: pd.DataFrame,
        *,
        depth_ylabel: str = "Depth (km)",
        plot_title: str = "Vertical profile",
        window_title: str = "垂直剖面",
        datum: DepthDatum = "z0",
        x_km: Optional[float] = None,
        x0_km: Optional[float] = None,
        x1_km: Optional[float] = None,
        individuals: Optional[list[tuple[float, pd.DataFrame]]] = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle(window_title)
        # 竖长横窄；最大化时画布按纵横比同步放大
        self.resize(400, 620)
        self.setMinimumSize(280, 420)

        self._profile = profile.copy()
        self._datum: DepthDatum = datum
        self._x_km = x_km
        self._x0_km = x0_km
        self._x1_km = x1_km
        self._individuals = list(individuals or [])
        self._depth_ylabel = depth_ylabel
        self._plot_title = plot_title

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        fig = Figure(figsize=(3.2, 5.0), dpi=100)
        fig.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.08)
        canvas = FigureCanvasQTAgg(fig)
        nav_row = QHBoxLayout()
        btn_reset = QPushButton("Reset View")
        btn_reset.setToolTip("双击图也可复位；滚轮缩放，左拖平移，右拖连续缩放")
        nav_row.addWidget(btn_reset)
        btn_save_fig = QPushButton("Save Figure")
        btn_save_fig.setToolTip("导出当前剖面图（png/jpg/pdf/ps 等；未写后缀时按所选类型补全）")
        btn_save_fig.clicked.connect(self._save_figure)
        nav_row.addWidget(btn_save_fig)
        self._chk_indiv = QCheckBox("Show sample traces")
        self._chk_indiv.setChecked(False)
        self._chk_indiv.setEnabled(len(self._individuals) > 1)
        self._chk_indiv.setToolTip("叠加各采样 X 的剖面（半透明）")
        self._chk_indiv.toggled.connect(self._redraw)
        nav_row.addWidget(self._chk_indiv)
        nav_row.addStretch(1)
        lay.addLayout(nav_row)

        host = _AspectRatioHost(canvas, width_over_height=_PROFILE_ASPECT)
        lay.addWidget(host, stretch=1)

        self._fig = fig
        self._canvas = canvas
        self._ax = fig.add_subplot(111)
        self._nav = PyqtgraphStyleNav(canvas)
        btn_reset.clicked.connect(self._nav.reset_view)

        self._redraw()
        self._canvas.draw()
        self._nav.remember_home_views()
        self._nav.remember_current_views()

        bbox = QDialogButtonBox()
        btn_save = bbox.addButton("Save V_1D…", QDialogButtonBox.ButtonRole.ActionRole)
        btn_save.setToolTip(
            "保存平均/单点：depth vp；多点时另存 *_envelope.txt（depth vp_min vp_max）"
        )
        btn_save.clicked.connect(self._save_mean_txt)
        if len(self._individuals) > 1:
            btn_all = bbox.addButton("Save all samples…", QDialogButtonBox.ButtonRole.ActionRole)
            btn_all.setToolTip(
                "目录内写入：平均 + 包络 + 每个采样 X 的剖面 txt"
            )
            btn_all.clicked.connect(self._save_all_samples)
        bbox.addButton(QDialogButtonBox.StandardButton.Close)
        bbox.rejected.connect(self.reject)
        lay.addWidget(bbox)

    def _has_envelope(self) -> bool:
        prof = self._profile
        return (
            len(self._individuals) > 1
            and "vp_min" in prof.columns
            and "vp_max" in prof.columns
        )

    def _save_figure(self) -> None:
        from .file_dialogs_qt import save_matplotlib_figure

        save_matplotlib_figure(
            self,
            self._fig,
            caption="Save Figure",
            default_stem="vertical_profile",
        )

    def _depth_label_hint(self) -> str:
        return f"Depth beneath {DATUM_LABELS.get(self._datum, self._datum)} (km)"

    def _redraw(self) -> None:
        ax = self._ax
        ax.clear()
        prof = self._profile
        depth = np.asarray(prof["depth"].values, dtype=float)
        vp = np.asarray(prof["vp"].values, dtype=float)

        has_env = self._has_envelope()
        if has_env:
            vmin = np.asarray(prof["vp_min"].values, dtype=float)
            vmax = np.asarray(prof["vp_max"].values, dtype=float)
            ax.fill_betweenx(
                depth,
                vmin,
                vmax,
                color="#4C78A8",
                alpha=0.28,
                linewidth=0,
                label="Range",
            )
            ax.plot(vmin, depth, color="#4C78A8", lw=0.8, alpha=0.7)
            ax.plot(vmax, depth, color="#4C78A8", lw=0.8, alpha=0.7)

        if self._chk_indiv.isChecked() and len(self._individuals) > 1:
            n = len(self._individuals)
            colors = cm.viridis(np.linspace(0, 1, n))
            for i, (x, pdf) in enumerate(self._individuals):
                label = f"x={x:.2f}" if i < 2 or i == n - 1 else ""
                ax.plot(
                    pdf["vp"].values,
                    pdf["depth"].values,
                    color=colors[i],
                    lw=0.9,
                    alpha=0.45,
                    label=label,
                )

        mean_label = "Average" if len(self._individuals) > 1 else "Profile"
        ax.plot(vp, depth, "r-", lw=2.0, alpha=0.95, label=mean_label)
        ax.invert_yaxis()
        ax.set_xlabel("Vp (km/s)")
        ax.set_ylabel(self._depth_ylabel or self._depth_label_hint())
        ax.set_title(self._plot_title, fontsize=10)
        ax.grid(True, alpha=0.3)

        # 横向收紧：按数据范围留约 4% 边距
        vp_all = [vp]
        if has_env:
            vp_all.extend([vmin, vmax])
        if self._chk_indiv.isChecked():
            for _x, pdf in self._individuals:
                vp_all.append(np.asarray(pdf["vp"].values, dtype=float))
        stacked = np.concatenate([np.asarray(a, dtype=float).ravel() for a in vp_all])
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size:
            lo, hi = float(np.min(stacked)), float(np.max(stacked))
            pad = max(0.04 * (hi - lo), 0.05)
            ax.set_xlim(lo - pad, hi + pad)

        _h, labels = ax.get_legend_handles_labels()
        if any(labels):
            ax.legend(loc="best", fontsize=7, framealpha=0.85)
        self._canvas.draw_idle()
        self._canvas.draw()
        self._nav.remember_home_views()
        self._nav.remember_current_views()

    def _default_mean_name(self) -> str:
        if self._x0_km is not None and self._x1_km is not None and len(self._individuals) > 1:
            return format_v1d_filename(
                datum=self._datum, x0_km=self._x0_km, x1_km=self._x1_km, kind="avg"
            )
        x = self._x_km
        if x is None and self._individuals:
            x = self._individuals[0][0]
        if x is None:
            x = 0.0
        return format_v1d_filename(datum=self._datum, x_km=float(x), kind="single")

    def _default_envelope_name(self) -> str:
        if self._x0_km is not None and self._x1_km is not None:
            return format_v1d_filename(
                datum=self._datum, x0_km=self._x0_km, x1_km=self._x1_km, kind="envelope"
            )
        return self._default_mean_name().replace(".txt", "_envelope.txt")

    def _write_mean_and_envelope(self, mean_path: Path) -> list[str]:
        """写入平均模型；有包络时同目录再写 *_envelope.txt。返回已写路径列表。"""
        written: list[str] = []
        write_v1d_txt(mean_path, self._profile["depth"].values, self._profile["vp"].values)
        written.append(str(mean_path))
        if self._has_envelope():
            env_path = mean_path.with_name(self._default_envelope_name())
            # 若用户另存为自定义名，包络跟平均同目录、用标准 envelope 名
            if mean_path.name != self._default_mean_name():
                env_path = mean_path.with_name(mean_path.stem + "_envelope.txt")
            write_v1d_envelope_txt(
                env_path,
                self._profile["depth"].values,
                self._profile["vp_min"].values,
                self._profile["vp_max"].values,
            )
            written.append(str(env_path))
        return written

    def _save_mean_txt(self) -> None:
        suggested = str(Path.cwd() / self._default_mean_name())
        path, _fl = QFileDialog.getSaveFileName(
            self,
            "保存 1D 速度模型",
            suggested,
            "Text (*.txt);;All (*.*)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".txt"):
            path = path + ".txt"
        try:
            written = self._write_mean_and_envelope(Path(path))
            msg = "已保存:\n" + "\n".join(written)
            if len(written) > 1:
                msg += "\n\n包络格式：depth  vp_min  vp_max"
            show_modeless_message("完成", msg, icon=QMessageBox.Icon.Information)
        except Exception as e:
            show_modeless_message("失败", str(e), icon=QMessageBox.Icon.Critical)

    def _save_all_samples(self) -> None:
        if not self._individuals:
            return
        directory = QFileDialog.getExistingDirectory(self, "选择保存目录", str(Path.cwd()))
        if not directory:
            return
        out = Path(directory)
        try:
            mean_name = self._default_mean_name()
            written = self._write_mean_and_envelope(out / mean_name)
            for x, pdf in self._individuals:
                name = format_v1d_filename(datum=self._datum, x_km=float(x), kind="single")
                write_v1d_txt(out / name, pdf["depth"].values, pdf["vp"].values)
                written.append(str(out / name))
            env_note = ""
            if self._has_envelope():
                env_note = f"\n包络: {Path(written[1]).name}（depth vp_min vp_max）"
            show_modeless_message("完成", f"已写入目录:\n{out}\n"
                f"平均: {mean_name}{env_note}\n"
                f"采样点: {len(self._individuals)} 个", icon=QMessageBox.Icon.Information)
        except Exception as e:
            show_modeless_message("失败", str(e), icon=QMessageBox.Icon.Critical)


def show_vertical_profile_dialog(
    parent: QWidget,
    *,
    profile: pd.DataFrame,
    depth_ylabel: str,
    plot_title: str,
    window_title: str,
    datum: DepthDatum = "z0",
    x_km: Optional[float] = None,
    x0_km: Optional[float] = None,
    x1_km: Optional[float] = None,
    individuals: Optional[list[tuple[float, pd.DataFrame]]] = None,
    bundle: Optional[VerticalProfileBundle] = None,
) -> VerticalProfileDialog:
    if bundle is not None:
        profile = bundle.mean
        individuals = bundle.individuals
        datum = bundle.datum
        x0_km = bundle.x_min
        x1_km = bundle.x_max
    dlg = VerticalProfileDialog(
        parent,
        profile,
        depth_ylabel=depth_ylabel,
        plot_title=plot_title,
        window_title=window_title,
        datum=datum,
        x_km=x_km,
        x0_km=x0_km,
        x1_km=x1_km,
        individuals=individuals,
    )
    show_modeless_dialog(dlg)
    return dlg
