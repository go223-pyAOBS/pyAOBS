# -*- coding: utf-8 -*-
"""Location map UI / cursor / jump mixed into QtFastViewer."""

from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
    import pyqtgraph.exporters as pg_exporters
except Exception:
    pg = None  # type: ignore
    pg_exporters = None  # type: ignore


class LocationMapMixin:
    """位置 Map 对话框、色标、光标联动与选道跳转。"""

    def _show_location_map(self) -> None:
        if self.loaded is None:
            self._show_themed_info("位置Map", "当前尚未加载数据。")
            return
        trace_headers = self.loaded.get("trace_headers", []) or []
        if not trace_headers:
            self._show_themed_info("位置Map", "当前数据缺少道头，无法绘制位置分布。")
            return
        # 工程输入/姿态校正已指定地形时预加载（真正叠加在 plot/mode 就绪后）
        try:
            self.ensure_shared_terrain_loaded(self._resolve_shared_terrain_path() or None)
        except Exception:
            pass

        def _valid_xy(x: float, y: float) -> bool:
            return math.isfinite(x) and math.isfinite(y) and (abs(x) > 1e-9 or abs(y) > 1e-9)

        src_x: List[float] = []
        src_y: List[float] = []
        rec_x: List[float] = []
        rec_y: List[float] = []
        mode_utm_count = 0
        mode_geo_count = 0
        trace_points: List[Tuple[float, float]] = []
        trace_indices: List[int] = []

        use_utm = False
        for th in trace_headers:
            sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
            syutm = float(getattr(th, "syutm", 0.0) or 0.0)
            rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
            ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
            if _valid_xy(sxutm, syutm) or _valid_xy(rxutm, ryutm):
                use_utm = True
                break

        # 自动判别“哪组坐标代表接收点(随道变化)”：
        # - 若 sx/sy 变化更大，则接收点来自 sx/sy（用户当前数据场景）
        # - 否则接收点来自 rx/ry（常见场景）
        sx_series: List[Tuple[float, float]] = []
        rx_series: List[Tuple[float, float]] = []
        for th in trace_headers:
            if use_utm:
                sxv = float(getattr(th, "sxutm", 0.0) or 0.0)
                syv = float(getattr(th, "syutm", 0.0) or 0.0)
                rxv = float(getattr(th, "rxutm", 0.0) or 0.0)
                ryv = float(getattr(th, "ryutm", 0.0) or 0.0)
            else:
                sxv = float(getattr(th, "slong", 0.0) or 0.0)
                syv = float(getattr(th, "slat", 0.0) or 0.0)
                rxv = float(getattr(th, "rlong", 0.0) or 0.0)
                ryv = float(getattr(th, "rlat", 0.0) or 0.0)
            if _valid_xy(sxv, syv):
                sx_series.append((sxv, syv))
            if _valid_xy(rxv, ryv):
                rx_series.append((rxv, ryv))

        def _var_count(series: List[Tuple[float, float]], decimals: int) -> int:
            if not series:
                return 0
            uniq = {(round(float(x), decimals), round(float(y), decimals)) for x, y in series}
            return len(uniq)

        # 与姿态校正 / RTM --geom obs 一致：默认 OBS 在道头 s* 槽
        try:
            from pyAOBS.geometry_roles import orientation_rec_role_label, resolve_geom

            geom = resolve_geom(self._orientation_geom_mode(), trace_headers)  # type: ignore[arg-type]
            self._location_map_rec_role = orientation_rec_role_label(geom)
        except Exception:
            dec = 2 if use_utm else 6
            s_var = _var_count(sx_series, dec)
            r_var = _var_count(rx_series, dec)
            self._location_map_rec_role = "sx" if s_var <= r_var else "rx"

        for trace_idx, th in enumerate(trace_headers):
            px, py = self._trace_position_for_map(trace_idx, mode="utm" if use_utm else "geo")
            if _valid_xy(px, py):
                trace_points.append((px, py))
                trace_indices.append(int(trace_idx))
                rec_x.append(px)
                rec_y.append(py)
            if use_utm:
                if self._location_map_rec_role == "sx":
                    sx = float(getattr(th, "rxutm", 0.0) or 0.0)
                    sy = float(getattr(th, "ryutm", 0.0) or 0.0)
                else:
                    sx = float(getattr(th, "sxutm", 0.0) or 0.0)
                    sy = float(getattr(th, "syutm", 0.0) or 0.0)
                if _valid_xy(sx, sy):
                    src_x.append(sx)
                    src_y.append(sy)
                    mode_utm_count += 1
            else:
                if self._location_map_rec_role == "sx":
                    sx = float(getattr(th, "rlong", 0.0) or 0.0)
                    sy = float(getattr(th, "rlat", 0.0) or 0.0)
                else:
                    sx = float(getattr(th, "slong", 0.0) or 0.0)
                    sy = float(getattr(th, "slat", 0.0) or 0.0)
                if _valid_xy(sx, sy):
                    src_x.append(sx)
                    src_y.append(sy)
                    mode_geo_count += 1

        sx = np.asarray(src_x, dtype=float)
        sy = np.asarray(src_y, dtype=float)
        rx = np.asarray(rec_x, dtype=float)
        ry = np.asarray(rec_y, dtype=float)
        smode = "utm" if use_utm and mode_utm_count > 0 else ("geo" if (not use_utm and mode_geo_count > 0) else "none")
        rmode = "utm" if use_utm else "geo"

        if sx.size == 0 and rx.size == 0:
            self._show_themed_info("位置Map", "未检测到有效震源/接收点坐标。")
            return

        if self._location_map_dialog is not None:
            try:
                self._location_map_dialog.close()
            except Exception:
                pass
            self._location_map_dialog = None

        # 使用独立窗口（无 parent）确保绝不阻塞主 GUI
        dialog = QtWidgets.QDialog(None)
        self._location_map_dialog = dialog
        dialog.setWindowTitle("位置Map（震源/接收点）")
        dialog.resize(920, 640)
        dialog.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        dialog.setModal(False)
        dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        def _on_destroyed(*_):
            self._location_map_dialog = None
            self._location_map_cursor_item = None
            self._location_map_selected_item = None
            self._location_map_plot_item = None
            self._location_map_plot_widget = None
            self._location_map_base_bounds = None
            self._location_map_terrain_item = None
            self._location_map_terrain_cache_key = None
            self._location_map_colorbar_gradient = None
            self._location_map_colorbar_min_label = None
            self._location_map_colorbar_max_label = None
            self._location_map_terrain_proj_label = None
            self._location_map_trace_points = np.empty((0, 2), dtype=float)
            self._location_map_trace_indices = np.empty((0,), dtype=int)
            self._location_map_mode = "none"
            self._location_map_rec_role = "auto"
            self._map_link_trace_idx = None
            if self.loaded is not None:
                self.request_render(delay_ms=10)
        dialog.destroyed.connect(_on_destroyed)
        lay = QtWidgets.QVBoxLayout(dialog)

        note = QtWidgets.QLabel(
            f"震源点: {int(sx.size)}（{smode}）  |  接收点: {int(rx.size)}（{rmode}）",
            dialog,
        )
        lay.addWidget(note)
        terrain_row = QtWidgets.QHBoxLayout()
        btn_load_terrain = QtWidgets.QPushButton("更换地形…", dialog)
        btn_load_terrain.setToolTip("工程输入页已指定地形时会自动共用；此处仅在需要更换时再选文件")
        btn_clear_terrain = QtWidgets.QPushButton("清除地形", dialog)
        btn_dump_terrain = QtWidgets.QPushButton("查看转换样本", dialog)
        btn_save_map = QtWidgets.QPushButton("保存图像", dialog)
        chk_force_geo = QtWidgets.QCheckBox("强制按经纬度解释", dialog)
        chk_force_geo.setChecked(bool(getattr(self, "_location_map_terrain_force_geo", False)))
        _terr_path = ""
        try:
            _terr_path = str((self._location_map_terrain_meta or {}).get("path", "") or "")
            if not _terr_path:
                _terr_path = str(getattr(self, "_orientation_terrain_path", "") or "")
        except Exception:
            _terr_path = ""
        if _terr_path:
            terrain_hint = QtWidgets.QLabel(f"已共用：{Path(_terr_path).name}", dialog)
        else:
            terrain_hint = QtWidgets.QLabel("支持 .grd/.nc/.xyz/.txt", dialog)
        terrain_hint.setStyleSheet("color:#666;")
        terrain_row.addWidget(btn_load_terrain)
        terrain_row.addWidget(btn_clear_terrain)
        terrain_row.addWidget(btn_dump_terrain)
        terrain_row.addWidget(btn_save_map)
        terrain_row.addWidget(chk_force_geo)
        terrain_row.addWidget(terrain_hint)
        terrain_row.addStretch(1)
        lay.addLayout(terrain_row)
        proj_row = QtWidgets.QHBoxLayout()
        chk_manual_zone = QtWidgets.QCheckBox("手动UTM分带", dialog)
        chk_manual_zone.setChecked(bool(getattr(self, "_location_map_terrain_manual_zone_enabled", False)))
        proj_row.addWidget(chk_manual_zone)
        proj_row.addWidget(QtWidgets.QLabel("Zone:", dialog))
        spin_zone = QtWidgets.QSpinBox(dialog)
        spin_zone.setRange(1, 60)
        spin_zone.setValue(int(getattr(self, "_location_map_terrain_manual_zone_value", 50)))
        proj_row.addWidget(spin_zone)
        proj_row.addWidget(QtWidgets.QLabel("半球:", dialog))
        combo_hemi = QtWidgets.QComboBox(dialog)
        combo_hemi.addItem("自动", "auto")
        combo_hemi.addItem("北半球", "north")
        combo_hemi.addItem("南半球", "south")
        hemi_val = str(getattr(self, "_location_map_terrain_manual_hemi", "auto")).lower()
        hemi_idx = max(0, combo_hemi.findData(hemi_val))
        combo_hemi.setCurrentIndex(hemi_idx)
        proj_row.addWidget(combo_hemi)
        proj_row.addSpacing(8)
        chk_sac2y_tm = QtWidgets.QCheckBox("SAC2Y兼容TM", dialog)
        chk_sac2y_tm.setChecked(bool(getattr(self, "_location_map_terrain_use_sac2y_tm", False)))
        proj_row.addWidget(chk_sac2y_tm)
        proj_row.addWidget(QtWidgets.QLabel("lon0:", dialog))
        spin_tm_lon0 = QtWidgets.QDoubleSpinBox(dialog)
        spin_tm_lon0.setDecimals(6)
        spin_tm_lon0.setRange(-180.0, 360.0)
        spin_tm_lon0.setSingleStep(0.1)
        spin_tm_lon0.setValue(float(getattr(self, "_location_map_terrain_tm_lon0", 120.0)))
        proj_row.addWidget(spin_tm_lon0)
        chk_tm_wrap = QtWidgets.QCheckBox("lon<0加360", dialog)
        chk_tm_wrap.setChecked(bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True)))
        proj_row.addWidget(chk_tm_wrap)
        chk_swap_lonlat = QtWidgets.QCheckBox("经纬互换", dialog)
        chk_swap_lonlat.setChecked(bool(getattr(self, "_location_map_terrain_swap_lonlat", False)))
        proj_row.addWidget(chk_swap_lonlat)
        proj_row.addStretch(1)
        lay.addLayout(proj_row)
        vis_row = QtWidgets.QHBoxLayout()
        vis_row.addWidget(QtWidgets.QLabel("色带:", dialog))
        combo_palette = QtWidgets.QComboBox(dialog)
        combo_palette.addItem("地形", "terrain")
        combo_palette.addItem("灰度", "gray")
        combo_palette.addItem("GMT风格", "gmt")
        combo_palette.addItem("反色地形", "terrain_r")
        combo_palette.addItem("自选CPT", "custom_cpt")
        pal_val = str(getattr(self, "_location_map_terrain_palette", "terrain")).lower()
        pal_idx = max(0, combo_palette.findData(pal_val))
        combo_palette.setCurrentIndex(pal_idx)
        vis_row.addWidget(combo_palette)
        btn_pick_cpt = QtWidgets.QPushButton("选择CPT", dialog)
        vis_row.addWidget(btn_pick_cpt)
        vis_row.addWidget(QtWidgets.QLabel("光照:", dialog))
        spin_shade = QtWidgets.QDoubleSpinBox(dialog)
        spin_shade.setRange(0.0, 1.0)
        spin_shade.setSingleStep(0.05)
        spin_shade.setDecimals(2)
        spin_shade.setValue(float(getattr(self, "_location_map_terrain_shade_strength", 0.75)))
        vis_row.addWidget(spin_shade)
        vis_row.addWidget(QtWidgets.QLabel("高:", dialog))
        spin_light_alt = QtWidgets.QDoubleSpinBox(dialog)
        spin_light_alt.setRange(1.0, 89.0)
        spin_light_alt.setSingleStep(1.0)
        spin_light_alt.setDecimals(1)
        spin_light_alt.setValue(float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0)))
        vis_row.addWidget(spin_light_alt)
        vis_row.addWidget(QtWidgets.QLabel("方位:", dialog))
        spin_light_az = QtWidgets.QDoubleSpinBox(dialog)
        spin_light_az.setRange(0.0, 360.0)
        spin_light_az.setSingleStep(5.0)
        spin_light_az.setDecimals(1)
        spin_light_az.setValue(float(getattr(self, "_location_map_terrain_light_az_deg", 315.0)))
        vis_row.addWidget(spin_light_az)
        chk_coast = QtWidgets.QCheckBox("海陆分界增强", dialog)
        chk_coast.setChecked(bool(getattr(self, "_location_map_terrain_coast_enhance", True)))
        vis_row.addWidget(chk_coast)
        vis_row.addStretch(1)
        lay.addLayout(vis_row)
        spin_tm_lon0.setEnabled(chk_sac2y_tm.isChecked())
        chk_tm_wrap.setEnabled(chk_sac2y_tm.isChecked())
        spin_zone.setEnabled(chk_manual_zone.isChecked() and (not chk_sac2y_tm.isChecked()))
        combo_hemi.setEnabled(chk_manual_zone.isChecked() and (not chk_sac2y_tm.isChecked()))
        proj_info = QtWidgets.QLabel(dialog)
        proj_info.setStyleSheet("color:#4b5563;")
        proj_info.setWordWrap(True)
        proj_info.setText(self._location_map_terrain_proj_text or "投影参数：未进行经纬度到UTM转换")
        self._location_map_terrain_proj_label = proj_info
        lay.addWidget(proj_info)

        pw = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.enableAutoRange(False)
        plot_item = pw.getPlotItem()
        self._location_map_plot_widget = pw
        self._location_map_plot_item = plot_item
        # 必须先于地形叠加设置：否则 geo 地形不会投影到 UTM，看起来像“没加载”
        self._location_map_mode = "utm" if use_utm else "geo"
        vb = plot_item.getViewBox()
        if vb is not None:
            vb.setAspectLocked(True, ratio=1.0)
        plot_item.setMenuEnabled(False)
        if use_utm:
            plot_item.setLabels(left="Y (m)", bottom="X (m)")
        else:
            plot_item.setLabels(left="Latitude (deg)", bottom="Longitude (deg)")

        legend = plot_item.addLegend(offset=(8, 8))
        if sx.size > 0:
            src_item = pg.ScatterPlotItem(
                x=sx,
                y=sy,
                size=16,
                pen=pg.mkPen("#7f1d1d", width=1.0),
                brush=pg.mkBrush("#ef4444"),
                symbol="t",
            )
            src_item.setZValue(120)
            plot_item.addItem(src_item)
            try:
                legend.addItem(src_item, "震源点")
            except Exception:
                pass
        if rx.size > 0:
            rec_item = pg.ScatterPlotItem(
                x=rx,
                y=ry,
                size=8,
                pen=pg.mkPen("#6b7280", width=1.0),
                brush=pg.mkBrush("#9ca3af"),
                symbol="o",
            )
            rec_item.setZValue(80)
            plot_item.addItem(rec_item)
            try:
                legend.addItem(rec_item, "接收点")
            except Exception:
                pass

        # 当前道实时标记（由主剖面鼠标移动驱动）
        self._location_map_cursor_item = pg.ScatterPlotItem(
            size=14,
            pen=pg.mkPen("#ffffff", width=1.6),
            brush=pg.mkBrush("#facc15"),
            symbol="o",
            pxMode=True,
        )
        self._location_map_cursor_item.setZValue(1000)
        plot_item.addItem(self._location_map_cursor_item, ignoreBounds=True)
        self._location_map_selected_item = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen("#f3f4f6", width=2.0),
            brush=pg.mkBrush("#facc15"),
            symbol="o",
            pxMode=True,
        )
        self._location_map_selected_item.setZValue(999)
        plot_item.addItem(self._location_map_selected_item, ignoreBounds=True)

        # Map 点击联动：跳到最近道
        pw.scene().sigMouseClicked.connect(lambda ev, _pw=pw: self._on_location_map_mouse_clicked(ev, _pw))
        chk_force_geo.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_force_geo", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_manual_zone.toggled.connect(
            lambda v, _spin=spin_zone, _combo=combo_hemi, _tm=chk_sac2y_tm: (
                setattr(self, "_location_map_terrain_manual_zone_enabled", bool(v)),
                _spin.setEnabled(bool(v) and (not _tm.isChecked())),
                _combo.setEnabled(bool(v) and (not _tm.isChecked())),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_zone.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_manual_zone_value", int(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        combo_hemi.currentIndexChanged.connect(
            lambda _i, _c=combo_hemi: setattr(
                self,
                "_location_map_terrain_manual_hemi",
                str(_c.currentData() or "auto"),
            )
        )
        combo_hemi.currentIndexChanged.connect(
            lambda _i, _c=combo_hemi: defer_after_combo_popup(
                self._refresh_location_map_terrain_overlay, _c
            )
        )
        chk_sac2y_tm.toggled.connect(
            lambda v, _spin=spin_tm_lon0, _wrap=chk_tm_wrap, _m=chk_manual_zone, _z=spin_zone, _h=combo_hemi: (
                setattr(self, "_location_map_terrain_use_sac2y_tm", bool(v)),
                _spin.setEnabled(bool(v)),
                _wrap.setEnabled(bool(v)),
                _z.setEnabled(_m.isChecked() and (not bool(v))),
                _h.setEnabled(_m.isChecked() and (not bool(v))),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_tm_lon0.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_tm_lon0", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_tm_wrap.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_tm_lon_wrap360", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_swap_lonlat.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_swap_lonlat", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        combo_palette.currentIndexChanged.connect(
            lambda _i, _c=combo_palette: (
                setattr(self, "_location_map_terrain_palette", str(_c.currentData() or "terrain")),
                defer_after_combo_popup(self._refresh_location_map_terrain_overlay, _c),
            )
        )
        btn_pick_cpt.clicked.connect(self._pick_location_map_cpt_file)
        spin_shade.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_shade_strength", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_light_alt.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_light_alt_deg", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_light_az.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_light_az_deg", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_coast.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_coast_enhance", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        btn_load_terrain.clicked.connect(self._load_location_map_terrain_from_dialog)
        btn_clear_terrain.clicked.connect(self._clear_location_map_terrain)
        btn_dump_terrain.clicked.connect(self._show_location_map_terrain_sample_dialog)
        btn_save_map.clicked.connect(self._save_location_map_figure)
        # 对话框建成且 map_mode 已设定后再叠地形
        try:
            terr_path = self._resolve_shared_terrain_path()
            self.ensure_shared_terrain_loaded(terr_path or None, force_reload=False)
        except Exception as exc:
            self._debug_log("TERRAIN", f"location map ensure failed: {exc}")
        if self._location_map_terrain_meta is not None:
            try:
                self._location_map_terrain_cache_key = None  # 模式刚设定，强制重建
                self._apply_location_map_terrain_overlay()
                _name = Path(str((self._location_map_terrain_meta or {}).get("path", "") or "")).name
                if _name:
                    terrain_hint.setText(f"已共用：{_name}")
                    self._set_status_text(f"位置Map：已自动叠加地形 {_name}", hold_ms=1800)
            except Exception as exc:
                terrain_hint.setText(f"地形叠加失败：{exc}")
                self._debug_log("TERRAIN", f"location map auto overlay failed: {exc}")
        else:
            _rp = self._resolve_shared_terrain_path()
            if _rp:
                terrain_hint.setText(f"地形未加载：{Path(_rp).name}（检查路径/格式）")
            else:
                terrain_hint.setText("未指定地形（工程输入页设置后将自动共用）")

        # 自动缩放到全部点
        x_all = np.concatenate([a for a in (sx, rx) if a.size > 0])
        y_all = np.concatenate([a for a in (sy, ry) if a.size > 0])
        if x_all.size > 0 and y_all.size > 0:
            x_fin = np.asarray(x_all[np.isfinite(x_all)], dtype=float)
            y_fin = np.asarray(y_all[np.isfinite(y_all)], dtype=float)
            if x_fin.size > 0 and y_fin.size > 0:
                if x_fin.size > 200:
                    xmin = float(np.nanpercentile(x_fin, 1.0))
                    xmax = float(np.nanpercentile(x_fin, 99.0))
                else:
                    xmin = float(np.min(x_fin))
                    xmax = float(np.max(x_fin))
                if y_fin.size > 200:
                    ymin = float(np.nanpercentile(y_fin, 1.0))
                    ymax = float(np.nanpercentile(y_fin, 99.0))
                else:
                    ymin = float(np.min(y_fin))
                    ymax = float(np.max(y_fin))
                if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
                    xmin, xmax = float(np.min(x_fin)), float(np.max(x_fin))
                    ymin, ymax = float(np.min(y_fin)), float(np.max(y_fin))
                if abs(xmax - xmin) < 1e-9:
                    xmax = xmin + 1.0
                if abs(ymax - ymin) < 1e-9:
                    ymax = ymin + 1.0
                self._location_map_base_bounds = (xmin, xmax, ymin, ymax)
                self._apply_location_map_view_range()

        map_row = QtWidgets.QHBoxLayout()
        map_row.addWidget(pw, stretch=1)
        cbar_col = QtWidgets.QVBoxLayout()
        lbl_max = QtWidgets.QLabel("", dialog)
        lbl_min = QtWidgets.QLabel("", dialog)
        lbl_max.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        lbl_min.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        grad = pg.GradientWidget(orientation="right")
        grad.setMinimumWidth(26)
        grad.setMaximumWidth(26)
        cbar_col.addWidget(lbl_max)
        cbar_col.addWidget(grad, stretch=1)
        cbar_col.addWidget(lbl_min)
        cbar_wrap = QtWidgets.QWidget(dialog)
        cbar_wrap.setLayout(cbar_col)
        cbar_wrap.setVisible(False)
        map_row.addWidget(cbar_wrap)
        self._location_map_colorbar_gradient = grad
        self._location_map_colorbar_min_label = lbl_min
        self._location_map_colorbar_max_label = lbl_max
        lay.addLayout(map_row, stretch=1)

        close_btn = QtWidgets.QPushButton("关闭", dialog)
        close_btn.clicked.connect(dialog.close)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        lay.addLayout(row)
        if trace_points:
            self._location_map_trace_points = np.asarray(trace_points, dtype=float)
            self._location_map_trace_indices = np.asarray(trace_indices, dtype=int)
            self._location_map_mode = "utm" if use_utm else "geo"
        else:
            self._location_map_trace_points = np.empty((0, 2), dtype=float)
            self._location_map_trace_indices = np.empty((0,), dtype=int)
            self._location_map_mode = "none"
            self._location_map_rec_role = "auto"
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        # 打开 map 后立即根据当前鼠标位置同步一次标记
        if self._last_render_trace_indices.size > 0 and self._last_render_offsets.size > 0 and self.mouse_x is not None:
            try:
                nearest_i = int(np.argmin(np.abs(self._last_render_offsets - float(self.mouse_x))))
                trace_idx = int(self._last_render_trace_indices[nearest_i])
                self._update_location_map_cursor_for_trace(trace_idx)
            except Exception:
                pass
        if self._map_link_trace_idx is not None:
            try:
                self._update_location_map_selected_for_trace(int(self._map_link_trace_idx))
            except Exception:
                pass


    def _trace_position_for_map(self, trace_idx: int, mode: str) -> Tuple[float, float]:
        if self.loaded is None:
            return (0.0, 0.0)
        trace_headers = self.loaded.get("trace_headers", []) or []
        if trace_idx < 0 or trace_idx >= len(trace_headers):
            return (0.0, 0.0)
        th = trace_headers[trace_idx]

        def _valid_xy(x: float, y: float) -> bool:
            return math.isfinite(x) and math.isfinite(y) and (abs(x) > 1e-9 or abs(y) > 1e-9)

        if str(mode).lower() == "utm":
            rec_role = str(getattr(self, "_location_map_rec_role", "rx")).lower()
            if rec_role == "sx":
                rx = float(getattr(th, "sxutm", 0.0) or 0.0)
                ry = float(getattr(th, "syutm", 0.0) or 0.0)
                sx = float(getattr(th, "rxutm", 0.0) or 0.0)
                sy = float(getattr(th, "ryutm", 0.0) or 0.0)
            else:
                rx = float(getattr(th, "rxutm", 0.0) or 0.0)
                ry = float(getattr(th, "ryutm", 0.0) or 0.0)
                sx = float(getattr(th, "sxutm", 0.0) or 0.0)
                sy = float(getattr(th, "syutm", 0.0) or 0.0)
            if _valid_xy(rx, ry):
                return (rx, ry)
            if _valid_xy(sx, sy):
                off_km = float(getattr(th, "offsti", 0.0) or 0.0)
                azi_deg = float(getattr(th, "azi", 0.0) or 0.0)
                if math.isfinite(off_km) and abs(off_km) > 1e-9 and math.isfinite(azi_deg):
                    dist_m = abs(off_km) * 1000.0
                    ang = math.radians(azi_deg)
                    ex = sx + dist_m * math.sin(ang)
                    ey = sy + dist_m * math.cos(ang)
                    if _valid_xy(ex, ey):
                        return (ex, ey)
                return (sx, sy)
            return (0.0, 0.0)

        rec_role = str(getattr(self, "_location_map_rec_role", "rx")).lower()
        if rec_role == "sx":
            lon = float(getattr(th, "slong", 0.0) or 0.0)
            lat = float(getattr(th, "slat", 0.0) or 0.0)
            slon = float(getattr(th, "rlong", 0.0) or 0.0)
            slat = float(getattr(th, "rlat", 0.0) or 0.0)
        else:
            lon = float(getattr(th, "rlong", 0.0) or 0.0)
            lat = float(getattr(th, "rlat", 0.0) or 0.0)
            slon = float(getattr(th, "slong", 0.0) or 0.0)
            slat = float(getattr(th, "slat", 0.0) or 0.0)
        if _valid_xy(lon, lat):
            return (lon, lat)
        if _valid_xy(slon, slat):
            return (slon, slat)
        return (0.0, 0.0)


    def _save_location_map_figure(self) -> None:
        if self._location_map_plot_item is None:
            self._show_themed_info("保存位置图", "位置Map尚未打开。")
            return
        out, selected = self._get_save_file_name(
            "保存位置Map图像",
            "location_map.png",
            "PNG Image (*.png);;SVG Vector (*.svg)",
            default_suffix=".png",
        )
        if not out:
            return
        try:
            use_svg = "svg" in (selected or "").lower() or out.lower().endswith(".svg")
            if use_svg:
                exporter = pg_exporters.SVGExporter(self._location_map_plot_item)
                exporter.export(out)
            else:
                exporter = pg_exporters.ImageExporter(self._location_map_plot_item)
                exporter.parameters()["width"] = 1800
                exporter.export(out)
            self._set_status_text(f"位置图已保存：{Path(out).name}", hold_ms=1800)
        except Exception as exc:
            self._show_themed_info("保存位置图失败", str(exc))


    def _build_location_map_colormap_and_levels(
        self, palette: str, zmin: float, zmax: float
    ) -> Tuple[pg.ColorMap, float, float]:
        pal = str(palette or "terrain").lower()
        if pal == "custom_cpt":
            cpt = self._get_custom_cpt_colormap()
            if cpt is not None:
                z_arr, rgb_arr = cpt
                cmin = float(np.min(z_arr))
                cmax = float(np.max(z_arr))
                cspan = max(1e-12, cmax - cmin)
                pos = np.clip((z_arr - cmin) / cspan, 0.0, 1.0)
                cmap = pg.ColorMap(pos, np.clip(rgb_arr, 0, 255).astype(np.ubyte))
                return cmap, cmin, cmax
        vals = np.linspace(0.0, 1.0, 256, dtype=float)
        rgb = self._terrain_colormap_rgb(vals, palette=pal).reshape(256, 3)
        cmap = pg.ColorMap(vals, np.clip(rgb, 0, 255).astype(np.ubyte))
        return cmap, float(zmin), float(zmax)


    def _update_location_map_colorbar(self, zmin: Optional[float], zmax: Optional[float]) -> None:
        grad = self._location_map_colorbar_gradient
        lbl_min = self._location_map_colorbar_min_label
        lbl_max = self._location_map_colorbar_max_label
        if grad is None or lbl_min is None or lbl_max is None:
            return
        if zmin is None or zmax is None or (not np.isfinite(zmin)) or (not np.isfinite(zmax)):
            grad.parentWidget().setVisible(False)
            return
        pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
        cmap, lv_min, lv_max = self._build_location_map_colormap_and_levels(pal, float(zmin), float(zmax))
        if hasattr(grad, "setColorMap"):
            try:
                grad.setColorMap(cmap)
            except Exception:
                pass
        if hasattr(grad, "item"):
            try:
                pos, cols = cmap.getStops(mode="byte")
                ticks = []
                for i in range(len(pos)):
                    c = cols[i]
                    ticks.append((float(pos[i]), (int(c[0]), int(c[1]), int(c[2]), 255)))
                grad.item.restoreState({"mode": "rgb", "ticks": ticks})
            except Exception:
                pass
        lbl_min.setText(f"{lv_min:.0f}")
        lbl_max.setText(f"{lv_max:.0f}")
        grad.parentWidget().setVisible(True)


    def _pick_location_map_cpt_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择CPT文件",
            "",
            "CPT (*.cpt *.txt);;All files (*)",
            options=self._file_dialog_options(),
        )
        if not path:
            return
        self._location_map_terrain_cpt_path = str(path)
        self._location_map_terrain_cpt_cache_key = None
        self._location_map_terrain_cpt_cache_data = None
        self._location_map_terrain_palette = "custom_cpt"
        self._location_map_terrain_cache_key = None
        self._set_status_text(f"CPT已选择：{Path(path).name}", hold_ms=1800)
        self._refresh_location_map_terrain_overlay()


    def _get_custom_cpt_colormap(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        path = str(getattr(self, "_location_map_terrain_cpt_path", "") or "").strip()
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        mtime = float(p.stat().st_mtime)
        key = (str(p), mtime)
        if self._location_map_terrain_cpt_cache_key == key and self._location_map_terrain_cpt_cache_data is not None:
            return self._location_map_terrain_cpt_cache_data
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return None
        color_model = "RGB"
        pts: List[Tuple[float, float, float, float]] = []

        def _to_rgb_triplet(c1: float, c2: float, c3: float) -> Tuple[float, float, float]:
            if color_model == "HSV":
                h = float(c1) % 360.0
                s = float(c2)
                v = float(c3)
                if s > 1.0:
                    s /= 100.0
                if v > 1.0:
                    v /= 100.0
                s = max(0.0, min(1.0, s))
                v = max(0.0, min(1.0, v))
                rr, gg, bb = colorsys.hsv_to_rgb(h / 360.0, s, v)
                return (rr * 255.0, gg * 255.0, bb * 255.0)
            return (float(c1), float(c2), float(c3))

        for raw in lines:
            s = raw.strip()
            if not s:
                continue
            if "COLOR_MODEL" in s.upper():
                su = s.upper().replace(" ", "")
                if "HSV" in su:
                    color_model = "HSV"
                elif "RGB" in su:
                    color_model = "RGB"
            if s.startswith("#"):
                continue
            toks = s.split()
            if not toks:
                continue
            if toks[0].upper() in ("B", "F", "N"):
                continue
            nums: List[float] = []
            ok = True
            for t in toks[:8]:
                try:
                    nums.append(float(t))
                except Exception:
                    ok = False
                    break
            if not ok or len(nums) < 4:
                continue
            if len(nums) >= 8:
                z1, c11, c12, c13, z2, c21, c22, c23 = nums[:8]
                r1, g1, b1 = _to_rgb_triplet(c11, c12, c13)
                r2, g2, b2 = _to_rgb_triplet(c21, c22, c23)
                pts.append((z1, r1, g1, b1))
                pts.append((z2, r2, g2, b2))
            else:
                z, c1, c2, c3 = nums[:4]
                r, g, b = _to_rgb_triplet(c1, c2, c3)
                pts.append((z, r, g, b))
        if len(pts) < 2:
            return None
        pts.sort(key=lambda x: x[0])
        z_vals = np.asarray([p0 for p0, _, _, _ in pts], dtype=float)
        rgb_vals = np.asarray([[r, g, b] for _, r, g, b in pts], dtype=float)
        # 去重
        uniq_z: List[float] = []
        uniq_rgb: List[np.ndarray] = []
        for i in range(z_vals.size):
            z = float(z_vals[i])
            if len(uniq_z) == 0 or abs(z - uniq_z[-1]) > 1e-12:
                uniq_z.append(z)
                uniq_rgb.append(rgb_vals[i])
            else:
                uniq_rgb[-1] = rgb_vals[i]
        z_arr = np.asarray(uniq_z, dtype=float)
        rgb_arr = np.asarray(uniq_rgb, dtype=float)
        colors = np.clip(rgb_arr, 0.0, 255.0)
        self._location_map_terrain_cpt_cache_key = key
        self._location_map_terrain_cpt_cache_data = (z_arr, colors)
        return self._location_map_terrain_cpt_cache_data


    def _apply_location_map_view_range(
        self, terrain_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        if self._location_map_plot_item is None:
            return
        vb = self._location_map_plot_item.getViewBox()
        if vb is None:
            return
        bounds = self._location_map_base_bounds
        if bounds is None and terrain_bounds is None:
            return
        if bounds is None:
            xmin, xmax, ymin, ymax = terrain_bounds  # type: ignore
        elif terrain_bounds is None:
            xmin, xmax, ymin, ymax = bounds
        else:
            xmin = float(min(bounds[0], terrain_bounds[0]))
            xmax = float(max(bounds[1], terrain_bounds[1]))
            ymin = float(min(bounds[2], terrain_bounds[2]))
            ymax = float(max(bounds[3], terrain_bounds[3]))
        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            return
        dx = max(1.0, abs(xmax - xmin))
        dy = max(1.0, abs(ymax - ymin))
        xpad = 0.03 * dx
        ypad = 0.03 * dy
        vb.setRange(
            xRange=(xmin - xpad, xmax + xpad),
            yRange=(ymin - ypad, ymax + ypad),
            padding=0.0,
        )


    def _update_location_map_cursor_for_trace(self, trace_idx: int) -> None:
        if self._location_map_cursor_item is None:
            return
        if self._location_map_mode not in ("utm", "geo"):
            self._location_map_cursor_item.setData([], [])
            return
        x, y = self._trace_position_for_map(int(trace_idx), self._location_map_mode)
        if not (math.isfinite(x) and math.isfinite(y)) or (abs(x) < 1e-9 and abs(y) < 1e-9):
            self._location_map_cursor_item.setData([], [])
            return
        self._location_map_cursor_item.setData([float(x)], [float(y)])


    def _update_location_map_selected_for_trace(self, trace_idx: int) -> None:
        if self._location_map_selected_item is None:
            return
        if self._location_map_mode not in ("utm", "geo"):
            self._location_map_selected_item.setData([], [])
            return
        x, y = self._trace_position_for_map(int(trace_idx), self._location_map_mode)
        if not (math.isfinite(x) and math.isfinite(y)) or (abs(x) < 1e-9 and abs(y) < 1e-9):
            self._location_map_selected_item.setData([], [])
            return
        self._location_map_selected_item.setData([float(x)], [float(y)])


    def _on_location_map_mouse_clicked(self, ev, map_widget: pg.PlotWidget) -> None:
        if ev is None or map_widget is None:
            return
        if self._location_map_trace_points.size == 0 or self._location_map_trace_indices.size == 0:
            return
        vb = map_widget.getViewBox()
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_pt = vb.mapSceneToView(pos)
        mx = float(mouse_pt.x())
        my = float(mouse_pt.y())
        pts = self._location_map_trace_points
        idxs = self._location_map_trace_indices
        try:
            idx_filtered = self._extract_indices()
            if idx_filtered.size > 0:
                keep = np.isin(idxs, idx_filtered.astype(int))
                if np.any(keep):
                    pts = pts[keep]
                    idxs = idxs[keep]
        except Exception:
            pass
        dx = pts[:, 0] - mx
        dy = pts[:, 1] - my
        d2 = dx * dx + dy * dy
        if d2.size == 0:
            return
        nearest = int(np.argmin(d2))
        trace_idx = int(idxs[nearest])
        self._jump_to_trace_from_map(trace_idx)


    def _jump_to_trace_from_map(self, trace_idx: int) -> None:
        if self.loaded is None:
            return
        trace_headers = self.loaded.get("trace_headers", []) or []
        if trace_idx < 0 or trace_idx >= len(trace_headers):
            return
        self._map_link_trace_idx = int(trace_idx)
        self.request_render(delay_ms=10)
        th = trace_headers[trace_idx]
        shot = int(getattr(th, "ishoti", 0) or 0)
        try:
            if shot > 0 and int(self.spin_irec.value()) != shot:
                self.spin_irec.setValue(shot)
        except Exception:
            pass
        self.request_render(immediate=True)

        def _center_on_trace():
            if self.loaded is None:
                return
            offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
            if trace_idx < 0 or trace_idx >= offsets.size:
                return
            x0 = float(offsets[trace_idx])
            vb = self.plot.getViewBox()
            xr, yr = vb.viewRange()
            xspan = max(1e-9, float(abs(xr[1] - xr[0])))
            vb.setXRange(x0 - 0.5 * xspan, x0 + 0.5 * xspan, padding=0.0)
            vb.setYRange(float(yr[0]), float(yr[1]), padding=0.0)
            self._sync_window_controls_from_view()
            self._update_location_map_cursor_for_trace(trace_idx)
            self._update_location_map_selected_for_trace(trace_idx)
            self._set_status_text(f"Map定位：已跳转到道 {trace_idx}（shot={shot if shot > 0 else 'N/A'}）", hold_ms=1800)

        QtCore.QTimer.singleShot(30, _center_on_trace)


