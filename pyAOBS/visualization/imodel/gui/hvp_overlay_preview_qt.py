"""Unified Fig.12a background + model (H, V_LC) overlay preview."""

from __future__ import annotations

import matplotlib

for _b in ("qtagg", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(_b)
        break
    except ValueError:
        continue

from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from petrology.hvp.fig15_composite import FIG15_PANEL_HEIGHT_RATIOS, plot_fig15_composite_on_axes
from petrology.hvp.fig15c_plot import plot_fig12a_model_overlays
from petrology.imodel_bridge.export_contract import CrustObservation
from petrology.imodel_bridge.imodel_adapter import format_crust_observation_summary
from petrology.seismic.transect import TransectWindow

from .styles import apply_dialog_style, hint_label, show_modeless_dialog, status_panel
from .plot_nav import install_plot_nav_bar, notify_plot_updated


class HvpOverlayPreviewDialog(QDialog):
    """Fig.12a 标准底图 + 单点/多段观测 / 沿迹滑窗投点（可叠加）。"""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        observation: CrustObservation | None = None,
        observations: list[CrustObservation] | None = None,
        observation_labels: list[str] | None = None,
        windows: list[TransectWindow] | None = None,
        delta_vp_max_km_s: float = 0.15,
        thick_crust_h_min_km: float = 15.0,
        transect_label: str = "profile",
        window_half_width_km: float = 10.0,
        distance_step_km: float = 10.0,
        n_mc: int = 100,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        obs = observation
        obs_list = list(observations or [])
        if not obs_list and obs is not None:
            obs_list = [obs]
        labels = list(observation_labels or [])
        wins = list(windows or [])
        parts: list[str] = []
        if obs_list:
            parts.append(f"{len(obs_list)} 点" if len(obs_list) > 1 else "单点")
        if wins:
            parts.append(f"{len(wins)} 窗")
        title_suffix = " + ".join(parts) if parts else "无底图数据"
        self.setWindowTitle(f"H–Vp 投图 (Fig.15) — {title_suffix}")
        self.resize(860, 920 if wins else 580)
        self.setMinimumSize(640, 480)

        root = QVBoxLayout(self)
        summary_lines: list[str] = []
        if len(obs_list) == 1:
            summary_lines.append(format_crust_observation_summary(obs_list[0]))
        elif obs_list:
            summary_lines.append(f"累加观测点: {len(obs_list)}")
            for i, o in enumerate(obs_list[:8]):
                lab = labels[i] if i < len(labels) and labels[i] else (
                    f"x={o.x_km:g} km" if o.x_km is not None else f"#{i + 1}"
                )
                summary_lines.append(
                    f"  · {lab}: H={o.h_whole_km:.2f} km, V_LC={o.v_lc_km_s:.3f} km/s"
                )
            if len(obs_list) > 8:
                summary_lines.append(f"  · …共 {len(obs_list)} 点")
        if wins:
            thick = sum(1 for w in wins if w.thick_crust)
            summary_lines.append(
                f"沿迹滑窗: {len(wins)} 窗（厚壳 {thick}，薄壳 {len(wins) - thick}）"
            )
        if not summary_lines:
            summary_lines.append("请先导出单点观测或沿迹滑窗。")
        root.addWidget(status_panel("\n".join(summary_lines)))

        if wins:
            fig = Figure(figsize=(8.0, 10.6), dpi=100)
            gs = fig.add_gridspec(
                3, 1, height_ratios=list(FIG15_PANEL_HEIGHT_RATIOS), hspace=0.36
            )
            ax_a = fig.add_subplot(gs[0])
            ax_b = fig.add_subplot(gs[1], sharex=ax_a)
            ax_c = fig.add_subplot(gs[2])
            import matplotlib.pyplot as plt

            plt.setp(ax_a.get_xticklabels(), visible=False)
            plot_fig15_composite_on_axes(
                (ax_a, ax_b, ax_c),
                windows=wins,
                observations=obs_list or None,
                observation_labels=labels or None,
                transect_label=transect_label,
                window_half_width_km=window_half_width_km,
                distance_step_km=distance_step_km,
                n_mc=n_mc,
                delta_vp_max_km_s=delta_vp_max_km_s,
                h_min_km=thick_crust_h_min_km,
            )
            fig.subplots_adjust(top=0.97, bottom=0.06, left=0.12, right=0.88)
            hint = (
                "(a) 沿迹 V_LC ± MC；(b) 全壳/下地壳厚度（左轴 km）+ 厚度占比（右轴断续线）；"
                "(c) Fig.12a 投点（可多段累加）。"
            )
        else:
            fig = Figure(figsize=(7.8, 5.2), dpi=100)
            ax_c = fig.add_subplot(111)
            plot_fig12a_model_overlays(
                ax_c,
                observations=obs_list or None,
                observation_labels=labels or None,
                windows=None,
                delta_vp_max_km_s=delta_vp_max_km_s,
                h_min_km=thick_crust_h_min_km,
            )
            ax_c.set_xlabel("Igneous crustal thickness H (km)")
            ax_c.set_ylabel(r"Mean $V_{\mathrm{p}}$ (km/s)")
            ax_c.set_title("(c)", loc="left", fontsize=10, fontweight="bold")
            ax_c.grid(True, ls=":", lw=0.35, alpha=0.35)
            if obs_list:
                ax_c.legend(fontsize=7, loc="upper left", framealpha=0.92)
            fig.tight_layout()
            hint = (
                "底图为 digitized Fig.12a；多段累加点以不同颜色/标记区分。"
                "单点时竖段为 Step-2 读图带宽（V_bulk ≤ V_LC）。"
            )

        canvas = FigureCanvasQTAgg(fig)
        install_plot_nav_bar(root, canvas, parent=self)
        root.addWidget(canvas, stretch=1)
        canvas.draw()
        notify_plot_updated(canvas)
        root.addWidget(hint_label(hint))

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save Figure")
        btn_save.setToolTip("导出当前投图（png/jpg/pdf/ps 等；未写后缀时按所选类型补全）")
        btn_save.clicked.connect(self._save_figure)
        btn_row.addWidget(btn_save)
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
            default_stem="hvp_overlay",
        )


def show_hvp_overlay_preview(
    parent: QWidget | None,
    *,
    observation: CrustObservation | None = None,
    observations: list[CrustObservation] | None = None,
    observation_labels: list[str] | None = None,
    windows: list[TransectWindow] | None = None,
    delta_vp_max_km_s: float = 0.15,
    thick_crust_h_min_km: float = 15.0,
    transect_label: str = "profile",
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    n_mc: int = 100,
) -> None:
    show_modeless_dialog(
        HvpOverlayPreviewDialog(
            parent,
            observation=observation,
            observations=observations,
            observation_labels=observation_labels,
            windows=windows,
            delta_vp_max_km_s=delta_vp_max_km_s,
            thick_crust_h_min_km=thick_crust_h_min_km,
            transect_label=transect_label,
            window_half_width_km=window_half_width_km,
            distance_step_km=distance_step_km,
            n_mc=n_mc,
        )
    )
