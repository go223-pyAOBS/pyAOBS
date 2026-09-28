"""蒙特卡洛预览：非模态三幅图（上扰动、中第1次实现、下基础模型）。"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.figure import Figure

from ..dialog_utils import show_modeless_dialog, show_modeless_message
from ..plots.inv_monitor_model import MonitorModelWidget
from ..plots.mpl_figure_window import MplNavCanvas
from ..state.form_state import FormState
from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import ensure_matplotlib_cjk_font


class Layered1dEnsembleCanvas(QWidget):
    """N 条分段 1D 叠在同一张 Vp–深度图上。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        ensure_matplotlib_cjk_font()
        fig = Figure(figsize=(4.6, 6.6), facecolor="w", layout="constrained")
        self.ax = fig.add_subplot(111)
        self.ax.set_title("分段随机 1D（全部实现）")
        self._host = MplNavCanvas(fig, self)
        self._host.hint.setText("红粗线=第1次实现 · 虚线=Moho")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._host)
        self.fig = fig

    def set_profiles(
        self,
        profiles,
        *,
        z_max: float,
        title: str,
        highlight: int = 0,
        ylabel: str | None = None,
    ) -> None:
        from ..services.mc_init_models import draw_layered_1d_ensemble
        from ..services.mc_vin_layers import Vin1dSample, draw_vin_1d_ensemble
        from ..plots.inv_monitor_model import finish_figure_layout

        if profiles and isinstance(profiles[0], Vin1dSample):
            draw_vin_1d_ensemble(
                self.ax,
                profiles,
                z_max=z_max,
                title=title,
                highlight=highlight,
                ylabel=ylabel,
            )
            self._host.hint.setText("红粗线=第1次实现 · 线型=选层界面")
        else:
            draw_layered_1d_ensemble(
                self.ax, profiles, z_max=z_max, title=title, highlight=highlight
            )
            self._host.hint.setText("红粗线=第1次实现 · 虚线=Moho")
        finish_figure_layout(self.fig)
        self._host.canvas.draw_idle()
        self._host.remember_home_views()

    def show_empty(self, message: str) -> None:
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            color="#64748b",
            wrap=True,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._host.canvas.draw_idle()


class MonteCarloPreviewWindow(QWidget):
    """独立绘图窗：等值线 / DWS 勾选与监视、对比共用表单偏好。"""

    def __init__(
        self,
        state: FormState,
        *,
        pull: Callable[[], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._pull = pull
        self.setWindowTitle("蒙特卡洛预览")
        self.resize(1280, 980)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        self.ck_contours = QCheckBox("叠加等值线")
        from ..plots.velocity_contours import contours_enabled

        self.ck_contours.setChecked(contours_enabled(self.state))
        self.ck_contours.setToolTip(
            "勾选：中图第1次实现、下图基础模型按当前色标叠对应等值线"
            "（vp / vs / vpvs）。上图 ΔV 不叠速度等值线。"
        )
        self.ck_dws = QCheckBox("DWS 遮罩")
        from ..services.dws_plot import dws_mask_enabled
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self.refresh(reset_home=False)
        )
        self.ck_dws.setChecked(dws_mask_enabled(self.state))
        self.ck_dws.setToolTip(
            "勾选：按基础 mesh 就近找 DWS（同目录 → 该次运行包 outputs/dws/ → "
            "表单 inv.dws_file）。无覆盖留白；有覆盖按 log(DWS) 透明。"
            "三幅图共用同一遮罩。找不到文件则整幅实色。"
        )
        btn_refresh = QPushButton("刷新")
        btn_refresh.setToolTip("按页签当前起始方式 / 种子 / 基础 mesh 重算第 1 次实现（不写盘）")
        btn_refresh.clicked.connect(lambda: self.refresh(reset_home=True))
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(self.ck_contours)
        bar.addWidget(self.cmap_combo)
        bar.addWidget(self.ck_dws)
        bar.addWidget(btn_refresh)
        bar.addStretch(1)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        self.plot = MonitorModelWidget(self, compare_bar=False)
        self.plot.show_empty_stack("等待计算蒙特卡洛初始场…")
        self.plot.set_interaction_hint(
            "上：ΔV · 中：第1次实现 · 下：基础 · 右：红粗线=第1次 1D · 右键写入表单"
        )
        self.profile = Layered1dEnsembleCanvas(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.plot)
        split.addWidget(self.profile)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([760, 480])
        root.addWidget(split, stretch=1)

        self.ck_contours.toggled.connect(self._on_contours_toggled)
        self.ck_dws.toggled.connect(self._on_dws_toggled)

    def _on_contours_toggled(self, on: bool) -> None:
        from ..plots.velocity_contours import set_contours_enabled

        set_contours_enabled(self.state, bool(on))
        self.refresh(reset_home=False)

    def _on_dws_toggled(self, on: bool) -> None:
        from ..services.dws_plot import set_dws_mask_enabled

        set_dws_mask_enabled(self.state, bool(on))
        self.refresh(reset_home=False)

    def refresh(self, *, reset_home: bool = True) -> None:
        from ..services.paths import resolve_work_dir
        from ..services.qc_workflows import paint_monte_carlo_preview

        if callable(self._pull):
            self._pull()
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            paint_monte_carlo_preview(
                self.plot,
                self.state,
                work,
                reset_home=reset_home,
                profile_plot=self.profile,
            )
        except Exception as e:
            show_modeless_message("蒙特卡洛预览", str(e))


def open_monte_carlo_preview(
    state: FormState,
    *,
    pull: Callable[[], None] | None = None,
    existing: MonteCarloPreviewWindow | None = None,
) -> MonteCarloPreviewWindow | None:
    """打开或刷新非模态预览窗。参数无效时提示并返回 None。"""
    from ..services.paths import resolve_work_dir
    from ..services.qc_workflows import monte_carlo_preview_params

    if callable(pull):
        pull()
    try:
        work = resolve_work_dir(state.get_str("work_dir"))
    except Exception as e:
        show_modeless_message("蒙特卡洛预览", str(e))
        return existing
    _mesh, mesh_abs, init_mode, amp, seed = monte_carlo_preview_params(state, work)
    if mesh_abs is None or not mesh_abs.is_file():
        show_modeless_message(
            "蒙特卡洛预览",
            "请先指定存在的初始/背景 mesh（空则回退 inv.mesh）。",
        )
        return existing
    if init_mode == "perturb" and amp is None:
        show_modeless_message("蒙特卡洛预览", "扰动 smesh 时幅度须为数值。")
        return existing
    if seed is None:
        show_modeless_message("蒙特卡洛预览", "随机种子须为整数。")
        return existing

    win = existing
    try:
        if win is not None and win.isVisible():
            win.refresh(reset_home=True)
            win.raise_()
            win.activateWindow()
            return win
    except RuntimeError:
        win = None

    win = MonteCarloPreviewWindow(state, pull=pull)
    try:
        win.refresh(reset_home=True)
    except Exception as e:
        show_modeless_message("蒙特卡洛预览", str(e))
        win.deleteLater()
        return None
    show_modeless_dialog(win, activate=True)
    return win
