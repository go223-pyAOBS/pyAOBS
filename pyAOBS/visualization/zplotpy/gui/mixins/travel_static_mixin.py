# -*- coding: utf-8 -*-
"""Traveltime / static / water correction mixed into QtFastViewer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
except Exception:
    pg = None  # type: ignore

try:
    from ...core.theoretical_traveltime import TheoreticalTravelTimeCalculator
except ImportError:  # pragma: no cover
    from pyAOBS.visualization.zplotpy.core.theoretical_traveltime import (
        TheoreticalTravelTimeCalculator,
    )


class TravelStaticMixin:
    """理论走时、静校正、水层校正与减速度时移。"""

    def _calculate_theoretical_traveltime_dialog(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("请先加载数据")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("计算理论走时")
        dialog.resize(640, 360)
        layout = QtWidgets.QFormLayout(dialog)

        model_edit = QtWidgets.QLineEdit(self._theory_model_file or "", dialog)
        browse_btn = QtWidgets.QPushButton("浏览", dialog)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(model_edit)
        model_row.addWidget(browse_btn)
        model_wrap = QtWidgets.QWidget(dialog)
        model_wrap.setLayout(model_row)
        layout.addRow("模型文件(v.in)", model_wrap)

        shot_x = QtWidgets.QDoubleSpinBox(dialog)
        shot_x.setRange(-1e6, 1e6)
        shot_x.setDecimals(3)
        shot_x.setValue(0.0)
        shot_z = QtWidgets.QDoubleSpinBox(dialog)
        shot_z.setRange(-1e6, 1e6)
        shot_z.setDecimals(3)
        shot_z.setValue(0.0)
        shot_auto = QtWidgets.QCheckBox("自动从数据提取炮点", dialog)
        shot_auto.setChecked(True)
        shot_row = QtWidgets.QHBoxLayout()
        shot_row.addWidget(QtWidgets.QLabel("X"))
        shot_row.addWidget(shot_x)
        shot_row.addWidget(QtWidgets.QLabel("Z"))
        shot_row.addWidget(shot_z)
        shot_row.addWidget(shot_auto)
        shot_wrap = QtWidgets.QWidget(dialog)
        shot_wrap.setLayout(shot_row)
        layout.addRow("炮点位置(km)", shot_wrap)

        ray_edit = QtWidgets.QLineEdit("1.2", dialog)
        nray_spin = QtWidgets.QSpinBox(dialog)
        nray_spin.setRange(1, 2000)
        nray_spin.setValue(10)
        xmin_spin = QtWidgets.QDoubleSpinBox(dialog)
        xmax_spin = QtWidgets.QDoubleSpinBox(dialog)
        zmin_spin = QtWidgets.QDoubleSpinBox(dialog)
        zmax_spin = QtWidgets.QDoubleSpinBox(dialog)
        for s in (xmin_spin, xmax_spin, zmin_spin, zmax_spin):
            s.setRange(-1e6, 1e6)
            s.setDecimals(3)
        range_auto = QtWidgets.QCheckBox("自动使用模型范围", dialog)
        range_auto.setChecked(True)
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("xmin"))
        range_row.addWidget(xmin_spin)
        range_row.addWidget(QtWidgets.QLabel("xmax"))
        range_row.addWidget(xmax_spin)
        range_row.addWidget(QtWidgets.QLabel("zmin"))
        range_row.addWidget(zmin_spin)
        range_row.addWidget(QtWidgets.QLabel("zmax"))
        range_row.addWidget(zmax_spin)
        range_row.addWidget(range_auto)
        range_wrap = QtWidgets.QWidget(dialog)
        range_wrap.setLayout(range_row)
        layout.addRow("ray参数", ray_edit)
        layout.addRow("nray", nray_spin)
        layout.addRow("范围(km)", range_wrap)

        use_picks = QtWidgets.QCheckBox("使用观测拾取生成 tx.in", dialog)
        use_picks.setChecked(False)
        layout.addRow(use_picks)

        def _browse_model():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog, "选择速度模型文件", "", "v.in files (*.in *.vin);;All files (*)", options=self._file_dialog_options()
            )
            if path:
                model_edit.setText(path)

        browse_btn.clicked.connect(_browse_model)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return

        model_file = model_edit.text().strip()
        if not model_file or not Path(model_file).exists():
            QtWidgets.QMessageBox.warning(self, "参数错误", "请提供有效的模型文件路径。")
            return
        self._theory_model_file = model_file

        shot_position = None if shot_auto.isChecked() else (float(shot_x.value()), float(shot_z.value()))
        try:
            ray_values = [float(x.strip()) for x in ray_edit.text().split(",") if x.strip()]
            if not ray_values:
                ray_values = [1.2]
        except Exception:
            QtWidgets.QMessageBox.warning(self, "参数错误", "ray 参数格式错误，请输入如 1.2 或 1.1,2.1")
            return

        ray_params: Dict[str, object] = {
            "ray": ray_values,
            "nray": int(nray_spin.value()),
        }
        if not range_auto.isChecked():
            ray_params.update(
                {
                    "xmin": float(xmin_spin.value()),
                    "xmax": float(xmax_spin.value()),
                    "zmin": float(zmin_spin.value()),
                    "zmax": float(zmax_spin.value()),
                }
            )

        self._calculate_theoretical_traveltime(
            model_file=model_file,
            shot_position=shot_position,
            ray_params=ray_params,
            use_observed_picks=bool(use_picks.isChecked()),
        )


    def _calculate_theoretical_traveltime(
        self,
        model_file: str,
        shot_position: Optional[tuple[float, float]] = None,
        ray_params: Optional[Dict[str, object]] = None,
        use_observed_picks: bool = False,
    ) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        try:
            self.lbl_status.setText("正在计算理论走时...")
            calc = TheoreticalTravelTimeCalculator(
                model_file_path=model_file,
                data_loader=self.loader,
                pick_manager=self.pick_manager,
            )
            model_info = calc.get_model_info()
            if not model_info.get("has_model"):
                self.lbl_status.setText("理论走时失败：模型加载失败")
                return
            if model_info.get("model_type") != "vin":
                self.lbl_status.setText("理论走时失败：仅支持 v.in 模型")
                return
            success = calc.calculate_travel_times(
                auto_generate_inputs=True,
                shot_position=shot_position,
                ray_params=ray_params,
                use_observed_picks=use_observed_picks,
                pick_word=int(self.spin_apick.value()),
            )
            if not success:
                self.lbl_status.setText("理论走时失败：RAYINVR 计算失败")
                return
            self.theoretical_traveltime_calculator = calc
            self.show_theoretical_times = True
            self.theoretical_times_data = None
            self.show_water_layer_correction = False
            self.water_layer_corrected_times = None
            self.request_render(immediate=True)
            self.lbl_status.setText("理论走时计算完成")
        except Exception as exc:
            self.lbl_status.setText(f"理论走时失败: {exc}")


    def _clear_theoretical_traveltime(self) -> None:
        self.theoretical_traveltime_calculator = None
        self.theoretical_times_data = None
        self.show_theoretical_times = False
        self.water_layer_corrections = {}
        self.water_layer_corrected_times = None
        self.show_water_layer_correction = False
        self._clear_theoretical_item()
        self._clear_water_corr_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除理论走时")


    def _calculate_water_layer_correction_dialog(self) -> None:
        if self.theoretical_traveltime_calculator is None:
            self.lbl_status.setText("请先计算理论走时")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("计算水层校正")
        dialog.resize(420, 220)
        layout = QtWidgets.QFormLayout(dialog)
        water_depth = self.theoretical_traveltime_calculator.get_water_layer_depth()
        if isinstance(water_depth, (int, float)):
            depth_text = f"{float(water_depth):.3f} km"
        elif isinstance(water_depth, tuple) and len(water_depth) == 2:
            depth_vals = np.asarray(water_depth[1], dtype=float)
            depth_text = f"[{float(np.min(depth_vals)):.3f}, {float(np.max(depth_vals)):.3f}] km"
        else:
            depth_text = "无法自动提取"
        layout.addRow("水层深度", QtWidgets.QLabel(depth_text, dialog))

        vwater = QtWidgets.QDoubleSpinBox(dialog)
        vwater.setRange(0.1, 10.0)
        vwater.setDecimals(3)
        vwater.setValue(1.5)
        vrepl = QtWidgets.QDoubleSpinBox(dialog)
        vrepl.setRange(0.0, 10.0)
        vrepl.setDecimals(3)
        vrepl.setValue(0.0)
        layout.addRow("水层速度(km/s)", vwater)
        layout.addRow("替换速度(km/s,0=自动)", vrepl)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        v_replacement = float(vrepl.value()) if float(vrepl.value()) > 0 else None
        self._calculate_water_layer_correction(v_water=float(vwater.value()), v_replacement=v_replacement)


    def _calculate_water_layer_correction(
        self, v_water: float = 1.5, v_replacement: Optional[float] = None
    ) -> None:
        calc = self.theoretical_traveltime_calculator
        if calc is None:
            self.lbl_status.setText("请先计算理论走时")
            return
        try:
            rays = calc.get_all_rays(max_rays=1000)
            if not rays:
                self.lbl_status.setText("水层校正失败：未获取射线")
                return
            corrections = calc.calculate_water_layer_correction(
                rays=rays,
                water_depth=None,
                v_water=float(v_water),
                v_replacement=v_replacement,
                return_by_distance=False,
            )
            if not corrections:
                self.lbl_status.setText("水层校正失败：未得到有效校正量")
                return
            avg_correction = float(np.mean(np.asarray(list(corrections.values()), dtype=float)))
            self.water_layer_corrections = {int(k): float(v) for k, v in corrections.items()}
            # 按射线终点距离建立“距离->校正量”映射，用于逐点校正理论走时曲线
            dist_vals: List[float] = []
            corr_vals: List[float] = []
            for ray_idx, corr in self.water_layer_corrections.items():
                if ray_idx < 0 or ray_idx >= len(rays):
                    continue
                ray = rays[ray_idx]
                xarr = np.asarray(ray.get("x", []), dtype=float)
                if xarr.size == 0:
                    continue
                dist_vals.append(float(xarr[-1]))
                corr_vals.append(float(corr))

            distance_grid = np.array([], dtype=float)
            correction_grid = np.array([], dtype=float)
            if dist_vals:
                d = np.asarray(dist_vals, dtype=float)
                c = np.asarray(corr_vals, dtype=float)
                order = np.argsort(d)
                d = d[order]
                c = c[order]
                # 合并重复距离（取均值），保证后续插值单调
                uniq_d, inv = np.unique(d, return_inverse=True)
                uniq_c = np.zeros_like(uniq_d, dtype=float)
                cnt = np.zeros_like(uniq_d, dtype=float)
                for i, gid in enumerate(inv):
                    uniq_c[gid] += c[i]
                    cnt[gid] += 1.0
                cnt[cnt <= 0] = 1.0
                uniq_c = uniq_c / cnt
                distance_grid = uniq_d
                correction_grid = uniq_c

            self.show_water_layer_correction = True
            self.water_layer_corrected_times = {
                "avg_correction": np.array([avg_correction], dtype=float),
                "distances": distance_grid,
                "corrections": correction_grid,
            }
            self.request_render(immediate=True)
            self.lbl_status.setText(
                f"水层校正完成：{len(corrections)}条射线, 平均校正 {avg_correction:.6f}s, 距离映射点 {int(distance_grid.size)}"
            )
        except Exception as exc:
            self.lbl_status.setText(f"水层校正失败: {exc}")


    def _clear_water_layer_correction(self) -> None:
        self.water_layer_corrections = {}
        self.water_layer_corrected_times = None
        self.show_water_layer_correction = False
        self._clear_water_corr_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除水层校正")


    def _show_water_correction_curve(self) -> None:
        if not self.water_layer_corrected_times:
            self.lbl_status.setText("请先计算水层校正")
            return
        dmap = np.asarray(self.water_layer_corrected_times.get("distances", []), dtype=float)
        cmap = np.asarray(self.water_layer_corrected_times.get("corrections", []), dtype=float)
        if dmap.size == 0 or cmap.size == 0 or dmap.size != cmap.size:
            self.lbl_status.setText("水层校正曲线不可用：缺少有效距离映射点")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("水层校正曲线")
        dlg.resize(760, 420)
        lay = QtWidgets.QVBoxLayout(dlg)

        plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        plot.showGrid(x=True, y=True, alpha=float(self._theme_color("plot_grid_alpha", "0.15")))
        plot.getPlotItem().setLabels(left="Correction (s)", bottom="Distance (km)")
        axis_pen = pg.mkPen(self._theme_color("plot_axis", "#1f2937"), width=1)
        for axis_name in ("left", "bottom"):
            axis = plot.getPlotItem().getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
        curve = pg.PlotDataItem(
            x=dmap,
            y=cmap,
            pen=pg.mkPen(self._theme_color("water_pen", "#1ea0d2"), width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(self._theme_color("water_pen", "#1ea0d2")),
            symbolPen=pg.mkPen(self._theme_color("water_pen", "#1ea0d2"), width=1),
        )
        plot.addItem(curve)
        lay.addWidget(plot, stretch=1)

        stats = (
            f"点数={dmap.size}  距离范围=[{float(np.min(dmap)):.3f}, {float(np.max(dmap)):.3f}] km  "
            f"校正范围=[{float(np.min(cmap)):.6f}, {float(np.max(cmap)):.6f}] s"
        )
        lay.addWidget(QtWidgets.QLabel(stats, dlg))

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)

        dlg.exec()


    def _compute_display_tshift(self, trace_idx: int, x_offset: Optional[float] = None) -> float:
        """计算某道在当前显示坐标下的时间平移（含对齐/静校正/折合速度）。

        姿态预览开启时：
          - 叠加全局走时 final (= prior+corr)，使主图反映走时校正（Z 样点仍可不旋）。
          - 折合仍用**原始**偏移，避免 dx/dy 改 offset 后折合把波形纵向错开；
            横轴仍可用预览后的偏移（见渲染路径）。
        """
        tshift = float(self._alignment_offsets.get(int(trace_idx), 0.0))
        if self.static_correction_enabled:
            tshift += float(self.static_corrector.get_correction(int(trace_idx)))
        # 姿态预览：折合用原始 offset，勿用预览改正后的 x
        if bool(self._orientation_preview_enabled) and self.loaded is not None:
            offsets_raw = np.asarray(self.loaded.get("offsets", []), dtype=float)
            x_red = None
            if 0 <= int(trace_idx) < offsets_raw.size:
                x_red = float(offsets_raw[int(trace_idx)])
            tshift += self._compute_reduction_tshift(trace_idx, x_red)
            tshift += float(self._orientation_preview_solution.get("time_shift_sec", 0.0) or 0.0)
        else:
            tshift += self._compute_reduction_tshift(trace_idx, x_offset)
        return tshift


    def _compute_reduction_tshift(self, trace_idx: int, x_offset: Optional[float] = None) -> float:
        """仅计算折合速度对应的时间平移（不含对齐/静校正）。"""
        if self.params.vred <= 0:
            return 0.0
        x = float(x_offset) if x_offset is not None else 0.0
        if x_offset is None and self.loaded is not None:
            offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
            if 0 <= int(trace_idx) < offsets_all.size:
                x = float(offsets_all[int(trace_idx)])
        rvred = 1.0 / float(self.params.vred)
        rvredf = 0.0
        if self.loaded is not None:
            header = self.loaded.get("header")
            vredf = float(getattr(header, "vredf", 0.0) or 0.0) if header is not None else 0.0
            if vredf > 0.0:
                rvredf = 1.0 / vredf
        # Fortran 对齐：时间窗与绘制基于 (rvred - rvredf)
        return -abs(x) * (rvred - rvredf)

    def _build_reduced_pick_times(self, trace_indices: List[int], pick_word: int) -> Dict[int, float]:
        """构建带折合速度修正的拾取时间（不含静校正/临时对齐）。"""
        if self.pick_manager is None:
            return {}
        by_word = self.pick_manager.get_picks_by_word(int(pick_word))
        if not by_word:
            return {}
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float) if self.loaded is not None else np.array([], dtype=float)
        out: Dict[int, float] = {}
        for ig in trace_indices:
            trace_idx = int(ig)
            tpk = by_word.get(trace_idx)
            if tpk is None or float(tpk) <= 0.0:
                continue
            x = float(offsets_all[trace_idx]) if 0 <= trace_idx < offsets_all.size else 0.0
            out[trace_idx] = float(tpk) + self._compute_reduction_tshift(trace_idx, x)
        return out


    def _ensure_static_preview_item(self) -> None:
        if self._static_preview_item is None:
            self._static_preview_item = pg.PlotDataItem(
                pen=pg.mkPen(
                    self._theme_color("static_preview_pen", "#c828a0"),
                    width=1.8,
                    style=QtCore.Qt.PenStyle.DashLine,
                )
            )
            self._static_preview_item.setZValue(35)
            self.plot.addItem(self._static_preview_item)


    def _clear_static_preview_item(self) -> None:
        if self._static_preview_item is not None:
            self._static_preview_item.setData([], [])


    def _ensure_theoretical_item(self) -> None:
        if self._theoretical_item is None:
            self._theoretical_item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("theory_pen", "#f07814"), width=2)
            )
            self._theoretical_item.setZValue(34)
            self.plot.addItem(self._theoretical_item)


    def _clear_theoretical_item(self) -> None:
        if self._theoretical_item is not None:
            self._theoretical_item.setData([], [])


    def _calculate_static_correction_dialog(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("提示：请先加载数据并拾取")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("计算静校正")
        dialog.resize(420, 180)
        layout = QtWidgets.QFormLayout(dialog)
        sigma_spin = QtWidgets.QDoubleSpinBox(dialog)
        sigma_spin.setRange(0.1, 100.0)
        sigma_spin.setDecimals(2)
        sigma_spin.setValue(3.0)
        smooth_spin = QtWidgets.QDoubleSpinBox(dialog)
        smooth_spin.setRange(0.0, 1.0)
        smooth_spin.setDecimals(3)
        smooth_spin.setSingleStep(0.01)
        smooth_spin.setValue(0.1)
        layout.addRow("sigma (km)", sigma_spin)
        layout.addRow("smoothness", smooth_spin)
        info = QtWidgets.QLabel("基于当前拾取字提取短波长静校正（需至少3个有效拾取）。", dialog)
        info.setWordWrap(True)
        layout.addRow(info)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        self._calculate_static_correction(
            sigma=float(sigma_spin.value()),
            smoothness=float(smooth_spin.value()),
        )


    def _calculate_static_correction(self, sigma: float, smoothness: float) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        pick_word = int(self.spin_apick.value())
        all_picks = self.pick_manager.get_all_picks()
        if not all_picks:
            self.lbl_status.setText("静校正失败：无拾取数据")
            return
        idx = self._extract_indices()
        if idx.size < 3:
            self.lbl_status.setText("静校正失败：当前过滤后道数不足")
            return
        by_word = self.pick_manager.get_picks_by_word(pick_word)
        picked_idx = np.array(
            [int(i) for i in idx if float(by_word.get(int(i), 0.0) or 0.0) > 0.0],
            dtype=int,
        )
        if picked_idx.size < 3:
            self.lbl_status.setText("静校正失败：当前拾取字有效拾取道不足（至少3道）")
            return
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        x_coords = offsets[picked_idx] if offsets.size else np.arange(picked_idx.size, dtype=float)
        trace_indices = [int(i) for i in picked_idx]
        # 在折合显示模式下，静校正应基于当前显示时间（t' = t - |x|/vred + |x|/vredf）提取。
        display_times = self._build_reduced_pick_times(trace_indices, pick_word)
        corrections = self.static_corrector.extract_short_wavelength_gaussian(
            picks=all_picks,
            trace_indices=trace_indices,
            x_coords=np.asarray(x_coords, dtype=float),
            pick_word=pick_word,
            sigma=float(sigma),
            min_picks=3,
            display_times=display_times if display_times else None,
            smoothness=float(smoothness),
        )
        if not corrections:
            self.lbl_status.setText("静校正失败：有效拾取不足或计算失败")
            return
        # 严格模式：仅对当前拾取字有有效拾取的道生效，不外推到其它道。
        strict_trace_set = set(trace_indices)
        corrections = {int(k): float(v) for k, v in corrections.items() if int(k) in strict_trace_set}
        if not corrections:
            self.lbl_status.setText("静校正失败：未生成可应用的校正道")
            return
        self.static_corrector.set_corrections(corrections)
        vals = np.asarray(list(corrections.values()), dtype=float)
        min_corr = float(np.min(vals))
        max_corr = float(np.max(vals))
        mean_corr = float(np.mean(vals))

        # 先进入预览态并立刻绘制拟合曲线，供用户评估后再决定是否应用
        self.static_correction_enabled = False
        self.static_preview_mode = True
        self.request_render(immediate=True)

        self._show_static_correction_decision(
            correction_count=len(corrections),
            min_corr=min_corr,
            max_corr=max_corr,
            mean_corr=mean_corr,
        )


    def _clear_static_correction(self) -> None:
        if self._static_decision_box is not None:
            try:
                self._static_decision_box.close()
            except Exception:
                pass
            self._static_decision_box = None
        self.static_corrector.clear_corrections()
        self.static_correction_enabled = False
        self.static_preview_mode = False
        self._clear_static_preview_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除静校正")


    def _show_static_correction_decision(
        self,
        correction_count: int,
        min_corr: float,
        max_corr: float,
        mean_corr: float,
    ) -> None:
        if self._static_decision_box is not None:
            try:
                self._static_decision_box.close()
            except Exception:
                pass
            self._static_decision_box = None

        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle("静校正拟合预览")
        msg.setText(
            f"静校正已计算完成：{correction_count} 道\n"
            f"范围 [{min_corr:.4f}, {max_corr:.4f}] s\n"
            f"平均 {mean_corr:.4f} s\n\n"
            "拟合曲线（虚线预览）已显示。\n"
            "可在主窗口缩放/平移检查后再决定是否应用。"
        )
        btn_apply = msg.addButton("应用静校正", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_preview = msg.addButton("仅保留预览", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        btn_cancel = msg.addButton("取消并清除", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_apply)
        msg.setModal(False)
        msg.setWindowModality(QtCore.Qt.WindowModality.NonModal)

        def _on_clicked(button: QtWidgets.QAbstractButton) -> None:
            self._static_decision_box = None
            if button == btn_cancel:
                self.static_corrector.clear_corrections()
                self.static_correction_enabled = False
                self.static_preview_mode = False
                self._clear_static_preview_item()
                self.lbl_status.setText("已取消静校正")
                self.request_render(delay_ms=10)
                return
            if button == btn_apply:
                self.static_correction_enabled = True
                self.static_preview_mode = False
                self.lbl_status.setText(
                    f"静校正已应用：{correction_count}道, 范围[{min_corr:.4f},{max_corr:.4f}]s"
                )
            else:
                self.static_correction_enabled = False
                self.static_preview_mode = True
                self.lbl_status.setText(
                    f"静校正预览：{correction_count}道, 范围[{min_corr:.4f},{max_corr:.4f}]s（虚线）"
                )
            self.request_render(delay_ms=10)

        msg.buttonClicked.connect(_on_clicked)
        self._static_decision_box = msg
        msg.show()
        QtCore.QTimer.singleShot(0, lambda: self._position_static_decision_box(msg))


    def _position_static_decision_box(self, box: QtWidgets.QWidget) -> None:
        """将静校正决策框放置到主窗口右上角，并限制在屏幕可视区内。"""
        try:
            margin = 12
            box.adjustSize()
            parent_geo = self.frameGeometry()
            box_geo = box.frameGeometry()

            x = int(parent_geo.right() - box_geo.width() - margin)
            y = int(parent_geo.top() + margin)

            screen = self.screen() or QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                min_x = int(avail.left() + margin)
                max_x = int(avail.right() - box_geo.width() - margin)
                min_y = int(avail.top() + margin)
                max_y = int(avail.bottom() - box_geo.height() - margin)
                if max_x < min_x:
                    max_x = min_x
                if max_y < min_y:
                    max_y = min_y
                x = max(min_x, min(x, max_x))
                y = max(min_y, min(y, max_y))

            box.move(x, y)
        except Exception:
            # 定位失败不影响主流程
            pass

    def _ensure_water_corr_item(self) -> None:
        if self._water_corr_item is None:
            self._water_corr_item = pg.PlotDataItem(
                pen=pg.mkPen(
                    self._theme_color("water_pen", "#1ea0d2"),
                    width=2,
                    style=QtCore.Qt.PenStyle.DashLine,
                )
            )
            self._water_corr_item.setZValue(34)
            self.plot.addItem(self._water_corr_item)


    def _clear_water_corr_item(self) -> None:
        if self._water_corr_item is not None:
            self._water_corr_item.setData([], [])

