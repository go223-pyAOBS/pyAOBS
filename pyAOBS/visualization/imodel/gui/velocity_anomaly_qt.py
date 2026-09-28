"""Velocity anomaly (ΔV / dv/v) window for imodel Qt.

Shows original model, reference model, and anomaly in three stacked panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils.qt_combo import defer_after_combo_popup
from pyAOBS.visualization.imodel.velocity_anomaly import (
    depthwise_horizontal_mean_velocity_anomaly,
    external_reference_velocity_anomaly,
    layer_average_velocity_anomaly,
)

from .model_load import load_velocity_grid
from .plot_nav import install_plot_nav_bar, notify_plot_updated
from .styles import show_modeless_tool_window, show_modeless_message
from .ui_chrome import apply_tool_window_chrome
from .zelt_iface_qt import plot_zelt_interfaces_on_axes

REF_DEPTH_MEAN = "depth-wise horizontal mean"
REF_LAYER_AVG = "layer average (v.in)"
REF_EXTERNAL = "external reference grid"
DISPLAY_DELTA = "ΔV (km/s)"
DISPLAY_RATIO = "dv/v (%)"

_REF_TOOLTIPS = {
    REF_LAYER_AVG: (
        "推荐：按 v.in 层界面做层内平均参考，跟随起伏的沉积/基底界面，"
        "更适合有结构的海洋/地壳剖面。"
    ),
    REF_DEPTH_MEAN: (
        "粗览用：按海底以下深度做横向平均（并排除海水）。"
        "基底与层界面非水平时会混叠不同岩性，不宜作为主解释依据。"
    ),
    REF_EXTERNAL: "加载外部 .grd/.nc 参考速度场，与当前模型同网格对比。",
}


def open_velocity_anomaly_window(
    parent: QWidget,
    grid_data: Any,
    *,
    velocity_var: str,
    x_coord: str,
    z_coord: str,
    title_prefix: str = "Velocity Anomaly",
    zelt_model: Any = None,
    basement_interface_idx: Optional[int] = None,
    seafloor_interface_idx: Optional[int] = None,
    moho_interface_idx: Optional[int] = None,
    seafloor_depths_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
    fig = Figure(figsize=(11, 9.2))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.0, 0.028],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.08,
        right=0.94,
        bottom=0.055,
        top=0.96,
        wspace=0.06,
        hspace=0.28,
    )
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_ref = fig.add_subplot(gs[1, 0], sharex=ax_orig)
    ax_anom = fig.add_subplot(gs[2, 0], sharex=ax_orig)
    axes = (ax_orig, ax_ref, ax_anom)
    caxes = (
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 1]),
    )
    # 不调用 Colorbar.remove()：部分 Matplotlib 会把 cax 从 figure 摘掉。
    cbar_holder: dict[str, Any] = {"cbars": [None, None, None]}

    # 独立顶层工具窗：不挂 parent，避免阻塞主窗口及其他子窗口。
    win = QMainWindow()
    win.setWindowTitle(f"{title_prefix} (V / Vref / ΔV)")
    win.resize(1120, 920)
    cw = QWidget()
    win.setCentralWidget(cw)
    lay = QVBoxLayout(cw)
    row = QHBoxLayout()
    btn_save = QPushButton("Save Figure")
    row.addWidget(btn_save)
    row.addWidget(QLabel("Reference:"))
    cmb_ref = QComboBox()
    # 有 v.in 时默认 layer average；否则退回 depth-wise（仅粗览）
    if zelt_model is not None:
        cmb_ref.addItems([REF_LAYER_AVG, REF_DEPTH_MEAN, REF_EXTERNAL])
        cmb_ref.setCurrentText(REF_LAYER_AVG)
    else:
        # 无 v.in 时不提供 layer average，避免一点选就报错
        cmb_ref.addItems([REF_DEPTH_MEAN, REF_EXTERNAL])
        cmb_ref.setCurrentText(REF_DEPTH_MEAN)
    cmb_ref.setToolTip(
        "\n".join(f"• {k}: {v}" for k, v in _REF_TOOLTIPS.items())
    )
    row.addWidget(cmb_ref)
    row.addWidget(QLabel("Anomaly Display:"))
    cmb_disp = QComboBox()
    cmb_disp.addItems([DISPLAY_DELTA, DISPLAY_RATIO])
    cmb_disp.setCurrentText(DISPLAY_DELTA)
    row.addWidget(cmb_disp)
    btn_ref = QPushButton("Load External Ref...")
    btn_ref.setEnabled(False)
    row.addWidget(btn_ref)
    chk_iface: QCheckBox | None = None
    if zelt_model is not None:
        chk_iface = QCheckBox("Show Interfaces (v.in layers)")
        chk_iface.setChecked(True)
        row.addWidget(chk_iface)
    row.addStretch(1)
    lay.addLayout(row)
    if zelt_model is not None:
        hint_text = (
            "推荐 Reference：layer average（跟随界面）。"
            "depth-wise 仅粗览——基底/层界面非水平时会混叠沉积与地壳。"
        )
    else:
        hint_text = (
            "当前无 v.in：仅提供 depth-wise / 外部参考。"
            "加载 v.in 后可用 layer average（推荐，跟随起伏界面）。"
        )
    hint = QLabel(hint_text)
    hint.setWordWrap(True)
    hint.setStyleSheet("color: #555; font-size: 11px;")
    lay.addWidget(hint)
    canvas = FigureCanvasQTAgg(fig)
    install_plot_nav_bar(lay, canvas, parent=cw)
    lay.addWidget(canvas)

    state: dict[str, Any] = {"ref_grid": None, "ref_path": ""}

    def _draw_interfaces(ax) -> None:
        if zelt_model is None:
            return
        if chk_iface is not None and not chk_iface.isChecked():
            return
        plot_zelt_interfaces_on_axes(
            ax,
            zelt_model,
            basement_interface_idx=basement_interface_idx,
            seafloor_interface_idx=seafloor_interface_idx,
            moho_interface_idx=moho_interface_idx,
        )

    def _compute_delta() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str
    ]:
        mode = cmb_ref.currentText()
        if mode == REF_DEPTH_MEAN:
            sf = None
            if seafloor_depths_fn is not None:
                x_probe = np.asarray(grid_data.coords[x_coord].values, dtype=float)
                try:
                    sf = np.asarray(seafloor_depths_fn(x_probe), dtype=float).reshape(-1)
                except Exception:
                    sf = None
            x_vals, z_vals, v, delta_v, ref_v = depthwise_horizontal_mean_velocity_anomaly(
                grid_data,
                velocity_var=velocity_var,
                x_coord=x_coord,
                z_coord=z_coord,
                seafloor_depths=sf,
            )
            if sf is not None and np.any(np.isfinite(sf)):
                desc = f"{REF_DEPTH_MEAN} (below seafloor, excl. water)"
            else:
                desc = f"{REF_DEPTH_MEAN} (excl. water Vp≤1.6)"
            return x_vals, z_vals, v, delta_v, ref_v, desc
        if mode == REF_LAYER_AVG:
            x_vals, z_vals, v, delta_v, ref_v = layer_average_velocity_anomaly(
                grid_data,
                velocity_var=velocity_var,
                x_coord=x_coord,
                z_coord=z_coord,
                zelt_model=zelt_model,
            )
            return x_vals, z_vals, v, delta_v, ref_v, REF_LAYER_AVG
        if state["ref_grid"] is None:
            raise ValueError("Please load external reference grid first.")
        x_vals, z_vals, v, delta_v, ref_v = external_reference_velocity_anomaly(
            grid_data,
            velocity_var=velocity_var,
            x_coord=x_coord,
            z_coord=z_coord,
            reference_grid=state["ref_grid"],
        )
        src = Path(state["ref_path"]).name if state["ref_path"] else "external grid"
        return x_vals, z_vals, v, delta_v, ref_v, f"{REF_EXTERNAL}: {src}"

    def _style_panel(ax, *, title: str, xlabel: bool) -> None:
        ax.set_ylabel("Depth (km)")
        ax.set_title(title, fontsize=10)
        if xlabel:
            ax.set_xlabel("Distance (km)")
        else:
            ax.tick_params(labelbottom=False)
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.3)

    def _redraw() -> None:
        nav = getattr(canvas, "_mpl_plot_nav", None)
        if nav is not None:
            nav.schedule_home_refresh()
        try:
            x_vals, z_vals, v_orig, delta_v, ref_v, ref_desc = _compute_delta()
        except Exception as e:
            for ax in axes:
                ax.clear()
                ax.set_axis_off()
            axes[1].text(
                0.5,
                0.5,
                str(e),
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                wrap=True,
            )
            for cax in caxes:
                cax.cla()
                cax.set_axis_off()
            cbar_holder["cbars"] = [None, None, None]
            canvas.draw()
            notify_plot_updated(canvas)
            return

        v_orig = np.asarray(v_orig, dtype=float)
        v_ref = np.asarray(ref_v, dtype=float)
        if v_ref.ndim == 1:
            v_ref = np.broadcast_to(v_ref.reshape(-1, 1), v_orig.shape).copy()
        elif v_ref.shape != v_orig.shape:
            v_ref = np.broadcast_to(v_ref, v_orig.shape).copy()

        disp_mode = cmb_disp.currentText()
        if disp_mode == DISPLAY_RATIO:
            # 仅在固体地球且 |Vref| 足够大时解释 dv/v，避免被近零参考放大
            safe = np.where(np.abs(v_ref) > 0.2, v_ref, np.nan)
            anom = 100.0 * np.asarray(delta_v, dtype=float) / safe
            anom_label = "dv/v (%)"
        else:
            anom = np.asarray(delta_v, dtype=float)
            anom_label = "ΔV (km/s)"

        finite_v = np.concatenate(
            [
                v_orig[np.isfinite(v_orig)].ravel(),
                v_ref[np.isfinite(v_ref)].ravel(),
            ]
        )
        if finite_v.size:
            vmin_v = float(np.nanmin(finite_v))
            vmax_v = float(np.nanmax(finite_v))
        else:
            vmin_v, vmax_v = 0.0, 1.0
        if not np.isfinite(vmin_v) or not np.isfinite(vmax_v) or vmax_v <= vmin_v:
            vmin_v, vmax_v = 0.0, 1.0

        anom_abs = float(np.nanmax(np.abs(anom))) if np.any(np.isfinite(anom)) else 0.0
        if anom_abs <= 0 or not np.isfinite(anom_abs):
            anom_abs = 0.1

        panels = (
            (ax_orig, caxes[0], v_orig, "viridis", vmin_v, vmax_v, "V (km/s)", "Original model"),
            (ax_ref, caxes[1], v_ref, "viridis", vmin_v, vmax_v, "Vref (km/s)", f"Reference ({ref_desc})"),
            (ax_anom, caxes[2], anom, "seismic", -anom_abs, anom_abs, anom_label, f"Anomaly — {disp_mode}"),
        )

        for i, (ax, cax, data, cmap, vmin, vmax, cbar_label, title) in enumerate(panels):
            ax.clear()
            ax.set_axis_on()
            im = ax.pcolormesh(
                x_vals,
                z_vals,
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
            _draw_interfaces(ax)
            _style_panel(ax, title=title, xlabel=(i == len(panels) - 1))
            cax.cla()
            cax.set_axis_on()
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(cbar_label)
            cbar_holder["cbars"][i] = cbar

        canvas.draw()
        notify_plot_updated(canvas)

    def _on_ref_mode_changed(_text: str) -> None:
        btn_ref.setEnabled(cmb_ref.currentText() == REF_EXTERNAL)
        # 先收回下拉，再重绘，避免同步 matplotlib 卡住导致菜单挂起
        defer_after_combo_popup(_redraw, cmb_ref)

    def _on_disp_changed(_text: str) -> None:
        defer_after_combo_popup(_redraw, cmb_disp)

    def _on_iface_toggled(_state: int) -> None:
        defer_after_combo_popup(_redraw)

    def _load_external_ref() -> None:
        path, _ = QFileDialog.getOpenFileName(
            win,
            "Load External Reference Grid",
            str(Path.cwd()),
            "Grid (*.grd *.nc);;NetCDF (*.nc);;All (*.*)",
        )
        if not path:
            return
        try:
            ds_ref, zelt_ref = load_velocity_grid(path)
            if zelt_ref is not None:
                raise ValueError("External reference should be a grid (.grd/.nc), not v.in.")
            state["ref_grid"] = ds_ref
            state["ref_path"] = path
            cmb_ref.setCurrentText(REF_EXTERNAL)
            _redraw()
        except Exception as ex:
            show_modeless_message("External Reference", str(ex), icon=QMessageBox.Icon.Critical)

    def _save() -> None:
        from .file_dialogs_qt import save_matplotlib_figure

        save_matplotlib_figure(
            win,
            fig,
            caption="Save Velocity Anomaly Figure",
            default_stem="velocity_anomaly",
        )

    btn_save.clicked.connect(_save)
    btn_ref.clicked.connect(_load_external_ref)
    cmb_ref.currentTextChanged.connect(_on_ref_mode_changed)
    cmb_disp.currentTextChanged.connect(_on_disp_changed)
    if chk_iface is not None:
        chk_iface.stateChanged.connect(_on_iface_toggled)
    apply_tool_window_chrome(win)
    _redraw()
    show_modeless_tool_window(win)
