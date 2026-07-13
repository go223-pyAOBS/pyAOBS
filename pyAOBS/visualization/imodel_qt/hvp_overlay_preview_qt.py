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
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from petrology.hvp.fig15_composite import FIG15_PANEL_HEIGHT_RATIOS, plot_fig15_composite_on_axes
from petrology.hvp.fig15c_plot import plot_fig12a_model_overlays
from petrology.imodel_bridge.export_contract import CrustObservation
from petrology.imodel_bridge.imodel_adapter import format_crust_observation_summary
from petrology.seismic.transect import TransectWindow

from .styles import apply_dialog_style, hint_label, show_modeless_dialog, status_panel


class HvpOverlayPreviewDialog(QDialog):
    """Fig.12a 标准底图 + 单点观测 / 沿迹滑窗投点（可叠加）。"""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        observation: CrustObservation | None = None,
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
        wins = list(windows or [])
        parts: list[str] = []
        if obs is not None:
            parts.append("单点")
        if wins:
            parts.append(f"{len(wins)} 窗")
        title_suffix = " + ".join(parts) if parts else "无底图数据"
        self.setWindowTitle(f"H–Vp 投图 (Fig.15) — {title_suffix}")
        self.resize(860, 920 if wins else 580)
        self.setMinimumSize(640, 480)

        root = QVBoxLayout(self)
        summary_lines: list[str] = []
        if obs is not None:
            summary_lines.append(format_crust_observation_summary(obs))
        if wins:
            thick = sum(1 for w in wins if w.thick_crust)
            summary_lines.append(
                f"沿迹滑窗: {len(wins)} 窗（厚壳 {thick}，薄壳 {len(wins) - thick}）"
            )
        if not summary_lines:
            summary_lines.append("请先导出单点观测或沿迹滑窗。")
        root.addWidget(status_panel("\n\n".join(summary_lines)))

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
                observation=obs,
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
                "(c) Fig.12a 投点。竖段为 Step-2 读图带宽（仅厚壳）。"
            )
        else:
            fig = Figure(figsize=(7.8, 5.2), dpi=100)
            ax_c = fig.add_subplot(111)
            plot_fig12a_model_overlays(
                ax_c,
                observation=obs,
                windows=None,
                delta_vp_max_km_s=delta_vp_max_km_s,
                h_min_km=thick_crust_h_min_km,
            )
            ax_c.set_xlabel("Igneous crustal thickness H (km)")
            ax_c.set_ylabel(r"Mean $V_{\mathrm{p}}$ (km/s)")
            ax_c.set_title("(c)", loc="left", fontsize=10, fontweight="bold")
            ax_c.grid(True, ls=":", lw=0.35, alpha=0.35)
            if obs is not None:
                ax_c.legend(fontsize=7, loc="upper left", framealpha=0.92)
            fig.tight_layout()
            hint = (
                "底图为 digitized Fig.12a；★/空心圆为单点观测。"
                "竖段为 Step-2 读图带宽（V_bulk ≤ V_LC，非平移）。"
            )

        canvas = FigureCanvasQTAgg(fig)
        tb_row = QWidget()
        tb_lay = QHBoxLayout(tb_row)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(NavigationToolbar2QT(canvas, tb_row))
        root.addWidget(tb_row)
        root.addWidget(canvas, stretch=1)
        root.addWidget(hint_label(hint))

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        apply_dialog_style(self)


def show_hvp_overlay_preview(
    parent: QWidget | None,
    *,
    observation: CrustObservation | None = None,
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
            windows=windows,
            delta_vp_max_km_s=delta_vp_max_km_s,
            thick_crust_h_min_km=thick_crust_h_min_km,
            transect_label=transect_label,
            window_half_width_km=window_half_width_km,
            distance_step_km=distance_step_km,
            n_mc=n_mc,
        )
    )
