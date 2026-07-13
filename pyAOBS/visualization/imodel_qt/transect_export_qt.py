"""Along-transect multi-window export dialog (Fig.15c) for imodel Qt."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from petrology.hvp.fig15c_plot import save_fig15c_figure
from petrology.imodel_bridge.transect_from_grid import imodel_grid_to_transect_windows
from petrology.seismic.transect import TransectWindow, export_transect_windows_csv

from .styles import apply_dialog_style, hint_label, primary_button, status_panel


class TransectExportDialog(QDialog):
    """Build sliding windows from velocity grid → Fig.15c on Fig.12a."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        get_ctx: Callable[[], dict[str, Any]],
        log_fn: Callable[[str], None],
        on_windows_saved: Callable[[list[TransectWindow], Path, Path | None], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("沿迹滑窗导出 (Fig.15 a–c)")
        self.resize(540, 480)
        self._get_ctx = get_ctx
        self._log = log_fn
        self._on_windows_saved = on_windows_saved
        self._windows: list[TransectWindow] = []
        self._rows: list[dict[str, float]] = []

        root = QVBoxLayout(self)
        root.addWidget(
            hint_label(
                "H = |M − B|；多边形模式：h_lc=|顶边(x)−M|，f_lower=h_lc/H；\n"
                "X 计算范围 = 多边形与剖面 X 范围的交集（最窄）。\n"
                "沿迹：深度样点 → 滑窗 + MC → (H, V_LC)。"
            )
        )

        prof = QGroupBox("剖面范围（与 Profile Extraction 一致）")
        pf = QFormLayout(prof)
        self._x_min = QDoubleSpinBox()
        self._x_max = QDoubleSpinBox()
        self._dx = QDoubleSpinBox()
        for w in (self._x_min, self._x_max):
            w.setRange(-1e6, 1e6)
            w.setDecimals(2)
            w.setSuffix(" km")
        self._dx.setRange(0.1, 500.0)
        self._dx.setValue(5.0)
        self._dx.setSuffix(" km")
        pf.addRow("X min:", self._x_min)
        pf.addRow("X max:", self._x_max)
        pf.addRow("采样 dx:", self._dx)
        root.addWidget(prof)

        lc = QGroupBox("下地壳顶界")
        lc_lay = QVBoxLayout(lc)
        self._rb_lc_interface = QRadioButton("v.in 界面（上下地壳分界）")
        self._rb_lc_polygon = QRadioButton("手画多边形（Polygon Selection）")
        self._rb_lc_interface.setChecked(True)
        lc_iface_row = QHBoxLayout()
        lc_iface_row.addWidget(self._rb_lc_interface)
        self._lc_top = QComboBox()
        lc_iface_row.addWidget(self._lc_top, stretch=1)
        lc_lay.addLayout(lc_iface_row)
        lc_lay.addWidget(self._rb_lc_polygon)
        for rb in (self._rb_lc_interface, self._rb_lc_polygon):
            rb.toggled.connect(lambda *_a: self._sync_lc_mode_widgets())
        root.addWidget(lc)

        agg = QGroupBox("滑窗 / MC（Fig.15a）")
        agg.setObjectName("ImodelBridgeGroup")
        af = QFormLayout(agg)
        self._win_half = QDoubleSpinBox()
        self._win_half.setRange(1.0, 200.0)
        self._win_half.setValue(10.0)
        self._win_half.setSuffix(" km")
        self._win_step = QDoubleSpinBox()
        self._win_step.setRange(1.0, 200.0)
        self._win_step.setValue(10.0)
        self._win_step.setSuffix(" km")
        self._h_min = QDoubleSpinBox()
        self._h_min.setRange(0.0, 100.0)
        self._h_min.setValue(15.0)
        self._h_min.setSuffix(" km")
        self._n_mc = QSpinBox()
        self._n_mc.setRange(10, 500)
        self._n_mc.setValue(100)
        self._vp_pick_sigma = QDoubleSpinBox()
        self._vp_pick_sigma.setRange(0.0, 0.5)
        self._vp_pick_sigma.setDecimals(3)
        self._vp_pick_sigma.setSingleStep(0.01)
        self._vp_pick_sigma.setValue(0.0)
        self._vp_pick_sigma.setSuffix(" km/s")
        self._vp_pick_sigma.setToolTip(
            "可选：叠加地震 Vp 拾取噪声。主不确定度来自下地壳深度域稀疏抽样（默认 0）"
        )
        af.addRow("窗半宽:", self._win_half)
        af.addRow("窗步长:", self._win_step)
        af.addRow("厚壳 H>", self._h_min)
        af.addRow("MC 次数:", self._n_mc)
        af.addRow("Vp 拾取 σ:", self._vp_pick_sigma)
        root.addWidget(agg)

        self._launch_lip = QCheckBox("保存后启动 LIP Petrology（--import-transect）")
        self._launch_lip.setChecked(True)
        root.addWidget(self._launch_lip)

        self._summary = status_panel("(尚未计算)")
        root.addWidget(self._summary)

        btn_row = QHBoxLayout()
        btn_calc = primary_button("计算滑窗")
        btn_calc.clicked.connect(self._on_compute)
        btn_prev = QPushButton("H–Vp 预览")
        btn_prev.clicked.connect(self._on_preview)
        btn_save = QPushButton("保存 CSV/PNG…")
        btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(btn_calc)
        btn_row.addWidget(btn_prev)
        btn_row.addWidget(btn_save)
        btn_row.addStretch()
        root.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        apply_dialog_style(self)
        self._sync_lc_mode_widgets()

    def _lc_mode(self) -> str:
        if self._rb_lc_polygon.isChecked():
            return "polygon"
        return "interface"

    def _sync_lc_mode_widgets(self) -> None:
        self._lc_top.setEnabled(self._lc_mode() == "interface")

    def _refresh_lc_top_combo(self) -> None:
        current = self._lc_top.currentData()
        self._lc_top.clear()
        opts = self._get_ctx().get("lc_top_interface_options") or []
        for label, fn in opts:
            self._lc_top.addItem(str(label), fn)
        if not opts:
            self._lc_top.addItem("(无 v.in 界面)", None)
        idx = self._lc_top.findData(current)
        if idx >= 0:
            self._lc_top.setCurrentIndex(idx)
        elif opts:
            self._lc_top.setCurrentIndex(0)

    @property
    def windows(self) -> list[TransectWindow]:
        return list(self._windows)

    def _sync_x_from_ctx(self) -> None:
        ctx = self._get_ctx()
        try:
            self._x_min.setValue(float(ctx.get("x_min_km", 0)))
            self._x_max.setValue(float(ctx.get("x_max_km", 0)))
            self._dx.setValue(float(ctx.get("dx_km", 5.0)))
        except Exception:
            pass

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_x_from_ctx()
        self._refresh_lc_top_combo()
        self._sync_lc_mode_widgets()

    def _compute(self) -> None:
        ctx = self._get_ctx()
        if ctx.get("grid_data") is None:
            raise RuntimeError("请先加载 Vp 模型")
        sample = ctx.get("sample_primary_velocity")
        moho = ctx.get("moho_depth_at_x")
        top = ctx.get("igneous_top_depth_at_x") or ctx.get("basement_depth_at_x")
        if sample is None or moho is None or top is None:
            raise RuntimeError("缺少速度采样或 B/M 界面")
        if not ctx.get("has_basement_interface") or not ctx.get("has_moho_interface"):
            raise RuntimeError("请先在 Profile 中设置 B（Basement）与 M（Moho）")

        mode = self._lc_mode()
        lc_fn = None
        polygon = None
        if mode == "interface":
            lc_fn = self._lc_top.currentData()
            if not callable(lc_fn):
                raise RuntimeError("请选择下地壳顶界界面")
        elif mode == "polygon":
            polys = ctx.get("selected_polygons") or []
            if callable(polys):
                polys = polys()
            if not polys:
                raise RuntimeError("请先用 Polygon Selection 圈定下地壳")
            polygon = [(float(p[0]), float(p[1])) for p in polys[-1]]

        rows, _samples, windows = imodel_grid_to_transect_windows(
            x_min_km=float(self._x_min.value()),
            x_max_km=float(self._x_max.value()),
            dx_km=float(self._dx.value()),
            grid_vp_fn=lambda x, z: float(sample(float(x), float(z))),
            moho_z_fn=moho,
            igneous_top_z_fn=top,
            lower_crust_mode=mode,
            lower_crust_top_z_fn=lc_fn,
            polygon_xz=polygon,
            window_half_width_km=float(self._win_half.value()),
            distance_step_km=float(self._win_step.value()),
            h_min_km=float(self._h_min.value()),
            n_mc=int(self._n_mc.value()),
            vp_pick_sigma_km_s=float(self._vp_pick_sigma.value()),
        )
        self._rows = rows
        self._windows = windows
        thick = sum(1 for w in windows if w.thick_crust)
        self._summary.setText(
            f"样点 {len(rows)} → 窗 {len(windows)}（厚壳 {thick}，薄壳 {len(windows) - thick}）"
        )
        self._log(
            f"Transect: {len(rows)} samples, {len(windows)} windows "
            f"(thick={thick}, thin={len(windows) - thick})"
        )

    def _on_compute(self) -> None:
        try:
            self._compute()
        except Exception as exc:
            QMessageBox.critical(self, "沿迹导出", str(exc))

    def _default_base_path(self) -> Path:
        ctx = self._get_ctx()
        model = str(ctx.get("current_model_file") or "").strip()
        if model:
            return Path(model).with_suffix("")
        return Path.cwd() / "imodel_transect"

    def _fig15_plot_kwargs(self) -> dict[str, float | int | str]:
        ctx = self._get_ctx()
        model = str(ctx.get("current_model_file") or "").strip()
        label = Path(model).stem if model else "profile"
        return {
            "transect_label": label,
            "window_half_width_km": float(self._win_half.value()),
            "distance_step_km": float(self._win_step.value()),
            "n_mc": int(self._n_mc.value()),
            "h_min_km": float(self._h_min.value()),
        }

    def _on_preview(self) -> None:
        if not self._windows:
            self._on_compute()
        if not self._windows:
            return
        from .hvp_overlay_preview_qt import show_hvp_overlay_preview

        kw = self._fig15_plot_kwargs()
        h_min = float(kw.pop("h_min_km"))
        show_hvp_overlay_preview(
            self,
            windows=self._windows,
            thick_crust_h_min_km=h_min,
            **kw,
        )

    def _on_save(self) -> None:
        if not self._windows:
            try:
                self._compute()
            except Exception as exc:
                QMessageBox.critical(self, "沿迹导出", str(exc))
                return
        if not self._windows:
            return
        base = self._default_base_path()
        win_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 windows CSV",
            str(base) + "_windows.csv",
            "CSV (*.csv)",
        )
        if not win_path:
            return
        win_path = Path(win_path)
        export_transect_windows_csv(win_path, self._windows)

        rows_path = win_path.with_name(win_path.stem + "_samples.csv")
        if self._rows:
            with rows_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(
                    fh,
                    fieldnames=["distance_km", "depth_km", "vp_insitu_km_s", "h_whole_km"],
                )
                w.writeheader()
                w.writerows(self._rows)

        fig_path = win_path.with_suffix(".png")
        save_fig15c_figure(self._windows, fig_path, **self._fig15_plot_kwargs())
        self._log(f"已保存 windows → {win_path}\nFig.15 (a–c) → {fig_path}")

        if self._on_windows_saved:
            self._on_windows_saved(self._windows, win_path, fig_path)

        if self._launch_lip.isChecked():
            launch_lip_with_transect(win_path)

    def accept_and_export(self) -> list[TransectWindow]:
        self._compute()
        if not self._windows:
            return []
        base = self._default_base_path()
        win_path = Path(str(base) + "_windows.csv")
        export_transect_windows_csv(win_path, self._windows)
        fig_path = win_path.with_suffix(".png")
        save_fig15c_figure(self._windows, fig_path, **self._fig15_plot_kwargs())
        if self._on_windows_saved:
            self._on_windows_saved(self._windows, win_path, fig_path)
        if self._launch_lip.isChecked():
            launch_lip_with_transect(win_path)
        return self._windows


def launch_lip_with_transect(windows_csv: Path | str) -> None:
    cmd = [sys.executable, "-m", "petrology.gui", "--import-transect", str(windows_csv)]
    try:
        subprocess.Popen(cmd)
    except Exception as exc:
        QMessageBox.warning(None, "LIP Petrology", f"无法启动:\n{exc}")
