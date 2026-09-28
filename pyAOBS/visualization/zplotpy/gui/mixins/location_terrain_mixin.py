# -*- coding: utf-8 -*-
"""Location-map terrain / shared bathymetry mixed into QtFastViewer."""

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
except Exception:
    pg = None  # type: ignore


class LocationTerrainMixin:
    """地形叠加、坐标转换与 ensure_shared_terrain_loaded。"""

    def _load_location_map_terrain_from_dialog(self) -> None:
        if self._location_map_dialog is None or self._location_map_plot_item is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择地形文件",
            "",
            "Terrain (*.grd *.nc *.xyz *.txt);;NetCDF (*.grd *.nc);;XYZ (*.xyz *.txt);;All files (*)",
            options=self._file_dialog_options(),
        )
        if not path:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self._set_status_text("位置Map：正在加载并转换地形...", hold_ms=2000)
            QtWidgets.QApplication.processEvents()
            force_geo = bool(getattr(self, "_location_map_terrain_force_geo", False))
            meta = self._load_terrain_meta(path, force_geo=force_geo)
            self._location_map_terrain_meta = meta
            self._location_map_terrain_cache_key = None
            # 同步共享地形缓存（姿态工区 / 位置 Map 共用路径）
            try:
                self._orientation_terrain_path = str(path)
                self._orientation_terrain_meta_raw = dict(meta)
                meta_utm = self._convert_terrain_meta_to_utm(meta)
                if meta_utm is not None:
                    self._orientation_terrain_meta_utm = meta_utm
            except Exception:
                pass
            self._apply_location_map_terrain_overlay()
            self._set_status_text(f"地形已加载：{Path(path).name}", hold_ms=1800)
        except Exception as exc:
            self._show_themed_info("地形加载失败", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()



    def _clear_location_map_terrain(self) -> None:
        self._location_map_terrain_meta = None
        self._location_map_terrain_cache_key = None
        self._location_map_terrain_proj_text = ""
        if self._location_map_terrain_proj_label is not None:
            self._location_map_terrain_proj_label.setText("投影参数：未进行经纬度到UTM转换")
        if self._location_map_plot_item is not None and self._location_map_terrain_item is not None:
            try:
                self._location_map_plot_item.removeItem(self._location_map_terrain_item)
            except Exception:
                pass
        self._location_map_terrain_item = None
        self._update_location_map_colorbar(None, None)
        self._set_status_text("地形叠加已清除", hold_ms=1200)



    def _refresh_location_map_terrain_overlay(self) -> None:
        if self._location_map_terrain_meta is None:
            return
        try:
            self._apply_location_map_terrain_overlay()
        except Exception as exc:
            self._show_themed_info("地形刷新失败", str(exc))



    def _build_location_map_terrain_sample_text(self, max_rows: int = 80) -> str:
        meta = self._location_map_terrain_meta
        if meta is None:
            return "尚未加载地形数据。"
        mode = str(meta.get("mode", ""))
        coord_kind = str(meta.get("coord_kind", "unknown")).lower()
        map_mode = str(getattr(self, "_location_map_mode", "none")).lower()
        use_sac2y_tm = bool(getattr(self, "_location_map_terrain_use_sac2y_tm", False))

        if mode == "points":
            x_raw = np.asarray(meta.get("x", []), dtype=float)
            y_raw = np.asarray(meta.get("y", []), dtype=float)
            z_raw = np.asarray(meta.get("z", []), dtype=float)
        elif mode == "grid":
            x = np.asarray(meta.get("x", []), dtype=float)
            y = np.asarray(meta.get("y", []), dtype=float)
            z = np.asarray(meta.get("z", []), dtype=float)
            if x.size == 0 or y.size == 0 or z.size == 0:
                return "地形网格为空。"
            xx, yy = np.meshgrid(x, y, indexing="xy")
            x_raw = xx.reshape(-1)
            y_raw = yy.reshape(-1)
            z_raw = np.asarray(z, dtype=float).reshape(-1)
        else:
            return f"不支持的地形模式: {mode}"

        raw_valid = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(z_raw)
        if not np.any(raw_valid):
            return "原始地形坐标无有效点。"
        x_raw = np.asarray(x_raw[raw_valid], dtype=float)
        y_raw = np.asarray(y_raw[raw_valid], dtype=float)
        z_raw = np.asarray(z_raw[raw_valid], dtype=float)

        lon_geo = np.full(x_raw.shape, np.nan, dtype=float)
        lat_geo = np.full(y_raw.shape, np.nan, dtype=float)
        x_proj = np.asarray(x_raw, dtype=float)
        y_proj = np.asarray(y_raw, dtype=float)
        geo_order = "n/a"
        proj_text = "原样坐标（未做经纬度->UTM）"

        if map_mode == "utm" and coord_kind == "geo":
            lon_geo, lat_geo, geo_order = self._normalize_geo_lonlat_order(x_raw, y_raw)
            if use_sac2y_tm:
                lon0_tm = float(getattr(self, "_location_map_terrain_tm_lon0", 120.0))
                wrap_tm = bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True))
                x_proj, y_proj, _info = self._convert_lonlat_to_sac2y_tm_arrays(
                    lon_geo, lat_geo, lon0=lon0_tm, wrap_lon360=wrap_tm
                )
                proj_text = f"SAC2Y TM(lon0={lon0_tm:.6f}, wrap360={'on' if wrap_tm else 'off'})"
            else:
                zone_override, hemi_override = self._get_manual_utm_override()
                x_proj, y_proj, _info = self._convert_lonlat_to_utm_arrays(
                    lon_geo,
                    lat_geo,
                    override_zone=zone_override,
                    override_hemisphere=hemi_override,
                )
                proj_text = "UTM(EPSG)模式"

        proj_valid = np.isfinite(x_proj) & np.isfinite(y_proj)
        if not np.any(proj_valid):
            return "投影后无有效点。请检查投影参数。"
        x_keep = x_raw[proj_valid]
        y_keep = y_raw[proj_valid]
        z_keep = z_raw[proj_valid]
        lon_keep = lon_geo[proj_valid]
        lat_keep = lat_geo[proj_valid]
        xp_keep = np.asarray(x_proj[proj_valid], dtype=float)
        yp_keep = np.asarray(y_proj[proj_valid], dtype=float)

        n = int(xp_keep.size)
        rows = max(1, min(int(max_rows), n))
        if n <= rows:
            idx = np.arange(n, dtype=int)
        else:
            idx = np.linspace(0, n - 1, rows, dtype=int)
            idx = np.unique(idx)

        lines: List[str] = []
        lines.append(f"path: {meta.get('path', '')}")
        lines.append(f"mode={mode}, coord_kind={coord_kind}, map_mode={map_mode}")
        lines.append(f"projection={proj_text}, geo_order={geo_order}")
        lines.append(f"valid_points={n}, shown={int(idx.size)}")
        lines.append("idx, raw_x, raw_y, raw_z, lon_used, lat_used, proj_x, proj_y")
        for ii in idx:
            lines.append(
                f"{int(ii):6d}, "
                f"{x_keep[ii]:.8f}, {y_keep[ii]:.8f}, {z_keep[ii]:.3f}, "
                f"{lon_keep[ii]:.8f}, {lat_keep[ii]:.8f}, "
                f"{xp_keep[ii]:.3f}, {yp_keep[ii]:.3f}"
            )
        return "\n".join(lines)



    def _show_location_map_terrain_sample_dialog(self) -> None:
        if self._location_map_dialog is None:
            self._show_themed_info("转换样本", "请先打开位置Map。")
            return
        try:
            text = self._build_location_map_terrain_sample_text(max_rows=80)
        except Exception as exc:
            self._show_themed_info("转换样本生成失败", str(exc))
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("地形转换样本")
        dlg.resize(980, 620)
        dlg.setModal(False)
        lay = QtWidgets.QVBoxLayout(dlg)
        editor = QtWidgets.QPlainTextEdit(dlg)
        editor.setReadOnly(True)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        editor.setFont(font)
        editor.setPlainText(text)
        lay.addWidget(editor, stretch=1)
        row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("复制", dlg)
        btn_close = QtWidgets.QPushButton("关闭", dlg)
        btn_copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(editor.toPlainText()))
        btn_close.clicked.connect(dlg.close)
        row.addStretch(1)
        row.addWidget(btn_copy)
        row.addWidget(btn_close)
        lay.addLayout(row)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()



    def _load_terrain_meta(self, path: str, force_geo: bool = False) -> Dict[str, object]:
        def _guess_coord_kind(xv: np.ndarray, yv: np.ndarray, xname: str = "", yname: str = "") -> str:
            xn = str(xname).lower()
            yn = str(yname).lower()
            if ("lon" in xn and "lat" in yn) or ("lon" in yn and "lat" in xn):
                return "geo"
            if ("x" == xn and "y" == yn) or ("easting" in xn or "northing" in yn):
                # 仅命名提示，不强制判定
                pass
            if xv.size == 0 or yv.size == 0:
                return "unknown"
            x_abs = float(np.nanmax(np.abs(xv[np.isfinite(xv)]))) if np.any(np.isfinite(xv)) else 0.0
            y_abs = float(np.nanmax(np.abs(yv[np.isfinite(yv)]))) if np.any(np.isfinite(yv)) else 0.0
            xmin = float(np.nanmin(xv[np.isfinite(xv)])) if np.any(np.isfinite(xv)) else 0.0
            xmax = float(np.nanmax(xv[np.isfinite(xv)])) if np.any(np.isfinite(xv)) else 0.0
            ymin = float(np.nanmin(yv[np.isfinite(yv)])) if np.any(np.isfinite(yv)) else 0.0
            ymax = float(np.nanmax(yv[np.isfinite(yv)])) if np.any(np.isfinite(yv)) else 0.0
            if -180.5 <= xmin <= 180.5 and -180.5 <= xmax <= 180.5 and -90.5 <= ymin <= 90.5 and -90.5 <= ymax <= 90.5:
                return "geo"
            if x_abs > 1000.0 and y_abs > 1000.0:
                return "utm"
            return "unknown"

        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in (".xyz", ".txt"):
            arr = np.loadtxt(str(p), comments="#", dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape((1, -1))
            if arr.shape[1] < 3:
                raise ValueError("xyz文本至少需要三列：x y z")
            x = np.asarray(arr[:, 0], dtype=float)
            y = np.asarray(arr[:, 1], dtype=float)
            z = np.asarray(arr[:, 2], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[valid], y[valid], z[valid]
            if x.size == 0:
                raise ValueError("xyz文本没有有效数据点")
            coord_kind = "geo" if force_geo else _guess_coord_kind(x, y)
            return {"mode": "points", "x": x, "y": y, "z": z, "path": str(p), "coord_kind": coord_kind}

        if suffix in (".nc", ".grd"):
            try:
                import xarray as xr  # type: ignore
            except Exception:
                raise RuntimeError("读取 .nc/.grd 需要 xarray，请先安装：pip install xarray netCDF4")
            try:
                from ..xarray_nc import open_netcdf_like_dataset  # type: ignore
            except ImportError:  # script/非包内运行等
                from pyAOBS.visualization.xarray_nc import open_netcdf_like_dataset  # type: ignore
            ds = open_netcdf_like_dataset(p)
            try:
                data_var = None
                for name, var in ds.data_vars.items():
                    if getattr(var, "ndim", 0) >= 2:
                        data_var = name
                        break
                if data_var is None:
                    for name in ds.variables:
                        try:
                            var = ds[name]
                        except Exception:
                            continue
                        if getattr(var, "ndim", 0) >= 2:
                            data_var = name
                            break
                if data_var is None:
                    try:
                        try:
                            from ..gravity_obs_grid import gmt_nf_coards_flat_to_dataarray
                        except ImportError:
                            from pyAOBS.visualization.gravity_obs_grid import gmt_nf_coards_flat_to_dataarray

                        da_pad = gmt_nf_coards_flat_to_dataarray(ds)
                        x = np.asarray(da_pad.coords["lon"].values, dtype=float)
                        y = np.asarray(da_pad.coords["lat"].values, dtype=float)
                        z = np.asarray(da_pad.values, dtype=float)
                        coord_kind = "geo" if force_geo else _guess_coord_kind(x, y, "lon", "lat")
                        return {
                            "mode": "grid",
                            "x": x,
                            "y": y,
                            "z": z,
                            "path": str(p),
                            "coord_kind": coord_kind,
                            "x_name": "lon",
                            "y_name": "lat",
                        }
                    except Exception:
                        raise ValueError("nc/grd中未找到二维地形变量")
                da = ds[data_var].squeeze()
                if da.ndim < 2:
                    raise ValueError("地形变量维度不足（需要二维）")
                dims = list(da.dims)
                ydim, xdim = dims[-2], dims[-1]
                x = np.asarray(ds[xdim].values, dtype=float)
                y = np.asarray(ds[ydim].values, dtype=float)
                z = np.asarray(da.values, dtype=float)
                if z.ndim > 2:
                    z = z.reshape(z.shape[-2], z.shape[-1])
                if z.shape[0] != y.size or z.shape[1] != x.size:
                    # 尝试转置
                    zt = z.T
                    if zt.shape[0] == y.size and zt.shape[1] == x.size:
                        z = zt
                    else:
                        raise ValueError("nc/grd网格维度与坐标长度不一致")
                coord_kind = "geo" if force_geo else _guess_coord_kind(x, y, xdim, ydim)
                return {
                    "mode": "grid",
                    "x": x,
                    "y": y,
                    "z": z,
                    "path": str(p),
                    "coord_kind": coord_kind,
                    "x_name": str(xdim),
                    "y_name": str(ydim),
                }
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
        raise ValueError("不支持的地形格式，请使用 .grd/.nc/.xyz/.txt")



    def _get_manual_utm_override(self) -> Tuple[Optional[int], Optional[str]]:
        if not bool(getattr(self, "_location_map_terrain_manual_zone_enabled", False)):
            return (None, None)
        zone = int(getattr(self, "_location_map_terrain_manual_zone_value", 0) or 0)
        if zone < 1 or zone > 60:
            return (None, None)
        hemi = str(getattr(self, "_location_map_terrain_manual_hemi", "auto") or "auto").lower()
        if hemi not in ("auto", "north", "south"):
            hemi = "auto"
        return (zone, hemi)



    def _convert_lonlat_to_utm_arrays(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        override_zone: Optional[int] = None,
        override_hemisphere: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(valid):
            raise ValueError("经纬度数据无有效点，无法转换为UTM")
        lon_valid = lon[valid]
        lat_valid = lat[valid]
        lon0 = float(np.nanmean(lon_valid))
        lat0 = float(np.nanmean(lat_valid))
        zone_auto = int(np.floor((lon0 + 180.0) / 6.0) + 1)
        zone_auto = max(1, min(60, zone_auto))
        zone = int(override_zone) if (override_zone is not None) else int(zone_auto)
        zone = max(1, min(60, zone))
        hemi_hint = str(override_hemisphere or "auto").lower()
        if hemi_hint == "north":
            lat_sign = 1.0
            hemi_src = "manual"
        elif hemi_hint == "south":
            lat_sign = -1.0
            hemi_src = "manual"
        else:
            lat_sign = 1.0 if lat0 >= 0.0 else -1.0
            hemi_src = "auto"
        epsg = (32600 + zone) if lat_sign >= 0.0 else (32700 + zone)
        central_meridian = float(-183.0 + 6.0 * float(zone))
        hemisphere = 1.0 if lat_sign >= 0.0 else -1.0
        try:
            from pyproj import Transformer  # type: ignore
        except Exception:
            raise RuntimeError("经纬度地形叠加到UTM坐标需要 pyproj，请安装：pip install pyproj")
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x_out = np.full(lon.shape, np.nan, dtype=float)
        y_out = np.full(lat.shape, np.nan, dtype=float)
        xx, yy = transformer.transform(lon_valid, lat_valid)
        x_out[valid] = np.asarray(xx, dtype=float)
        y_out[valid] = np.asarray(yy, dtype=float)
        info: Dict[str, object] = {
            "zone": float(zone),
            "zone_auto": float(zone_auto),
            "epsg": float(epsg),
            "central_meridian": float(central_meridian),
            "lon0": float(lon0),
            "lat0": float(lat0),
            "hemisphere": float(hemisphere),
            "zone_mode": "manual" if (override_zone is not None) else "auto",
            "hemi_mode": str(hemi_src),
        }
        return x_out, y_out, info



    def _convert_lonlat_to_sac2y_tm_arrays(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        lon0: float,
        wrap_lon360: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(valid):
            raise ValueError("经纬度数据无有效点，无法转换为SAC2Y TM")
        lon_valid = lon[valid]
        lat_valid = lat[valid]
        if bool(wrap_lon360):
            lon_use = np.where(lon_valid < 0.0, lon_valid + 360.0, lon_valid)
        else:
            lon_use = lon_valid
        try:
            from pyproj import Transformer  # type: ignore
        except Exception:
            raise RuntimeError("SAC2Y兼容TM坐标转换需要 pyproj，请安装：pip install pyproj")
        proj_string = f"+proj=tmerc +lon_0={float(lon0):.10f} +datum=WGS84 +units=m +k_0=0.9996 +ellps=WGS84"
        transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
        xx, yy = transformer.transform(lon_use, lat_valid)
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        # 与 sac2y/format_utils 保持一致：x+500000，南半球 y+10000000（不加 ex/ey）
        xx = xx + 500000.0
        yy = np.where(lat_valid < 0.0, yy + 10000000.0, yy)
        x_out = np.full(lon.shape, np.nan, dtype=float)
        y_out = np.full(lat.shape, np.nan, dtype=float)
        x_out[valid] = xx
        y_out[valid] = yy
        info: Dict[str, object] = {
            "method": "sac2y_tm",
            "lon0_tm": float(lon0),
            "wrap_lon360": bool(wrap_lon360),
            "lon_mean": float(np.nanmean(lon_valid)),
            "lat_mean": float(np.nanmean(lat_valid)),
            "central_meridian": float(lon0),
            "x_false_easting": 500000.0,
            "south_y_shift": 10000000.0,
            "note": "SAC2Y兼容：不使用ex/ey",
        }
        return x_out, y_out, info



    def _normalize_geo_lonlat_order(
        self, x_geo: np.ndarray, y_geo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        x_geo = np.asarray(x_geo, dtype=float)
        y_geo = np.asarray(y_geo, dtype=float)
        if bool(getattr(self, "_location_map_terrain_swap_lonlat", False)):
            return y_geo, x_geo, "manual_swapped"

        xv = x_geo[np.isfinite(x_geo)]
        yv = y_geo[np.isfinite(y_geo)]
        if xv.size == 0 or yv.size == 0:
            return x_geo, y_geo, "auto_default"
        x_min, x_max = float(np.min(xv)), float(np.max(xv))
        y_min, y_max = float(np.min(yv)), float(np.max(yv))
        x_is_lon = (-180.5 <= x_min <= 360.5) and (-180.5 <= x_max <= 360.5)
        x_is_lat = (-90.5 <= x_min <= 90.5) and (-90.5 <= x_max <= 90.5)
        y_is_lon = (-180.5 <= y_min <= 360.5) and (-180.5 <= y_max <= 360.5)
        y_is_lat = (-90.5 <= y_min <= 90.5) and (-90.5 <= y_max <= 90.5)
        # 典型“列顺序反了”场景：x像纬度，y像经度
        if x_is_lat and y_is_lon and (not x_is_lon or not y_is_lat):
            return y_geo, x_geo, "auto_swapped"
        return x_geo, y_geo, "auto_default"



    def _terrain_colormap_rgb(self, norm: np.ndarray, palette: str = "terrain") -> np.ndarray:
        n = np.clip(np.asarray(norm, dtype=float), 0.0, 1.0)
        pal = str(palette or "terrain").lower()
        if pal == "gray":
            stops = np.array([0.0, 1.0], dtype=float)
            colors = np.array([[30.0, 30.0, 30.0], [240.0, 240.0, 240.0]], dtype=float)
        elif pal == "custom_cpt":
            cpt = self._get_custom_cpt_colormap()
            if cpt is not None:
                z_arr, colors = cpt
                zmin = float(np.min(z_arr))
                zmax = float(np.max(z_arr))
                span = max(1e-12, zmax - zmin)
                stops = np.clip((z_arr - zmin) / span, 0.0, 1.0)
            else:
                stops = np.array([0.0, 0.35, 0.5, 0.78, 1.0], dtype=float)
                colors = np.array(
                    [
                        [26.0, 58.0, 118.0],
                        [55.0, 126.0, 184.0],
                        [102.0, 166.0, 86.0],
                        [158.0, 122.0, 76.0],
                        [240.0, 240.0, 240.0],
                    ],
                    dtype=float,
                )
        elif pal == "gmt":
            stops = np.array([0.0, 0.18, 0.35, 0.5, 0.68, 0.85, 1.0], dtype=float)
            colors = np.array(
                [
                    [20.0, 46.0, 115.0],
                    [48.0, 98.0, 168.0],
                    [94.0, 159.0, 201.0],
                    [121.0, 173.0, 98.0],
                    [171.0, 146.0, 95.0],
                    [205.0, 190.0, 160.0],
                    [248.0, 248.0, 248.0],
                ],
                dtype=float,
            )
        else:
            # 深海蓝 -> 浅海青 -> 陆地绿 -> 棕色 -> 近白高地
            stops = np.array([0.0, 0.35, 0.5, 0.78, 1.0], dtype=float)
            colors = np.array(
                [
                    [26.0, 58.0, 118.0],
                    [55.0, 126.0, 184.0],
                    [102.0, 166.0, 86.0],
                    [158.0, 122.0, 76.0],
                    [240.0, 240.0, 240.0],
                ],
                dtype=float,
            )
            if pal == "terrain_r":
                colors = colors[::-1].copy()
        rgb = np.empty((n.size, 3), dtype=float)
        flat = n.reshape(-1)
        for c in range(3):
            rgb[:, c] = np.interp(flat, stops, colors[:, c])
        return rgb.reshape(n.shape + (3,))



    def _custom_cpt_rgb_from_values(self, values: np.ndarray) -> Optional[np.ndarray]:
        cpt = self._get_custom_cpt_colormap()
        if cpt is None:
            return None
        z_arr, rgb_arr = cpt
        vals = np.asarray(values, dtype=float)
        flat = vals.reshape(-1)
        rgb = np.empty((flat.size, 3), dtype=float)
        for c in range(3):
            rgb[:, c] = np.interp(flat, z_arr, rgb_arr[:, c], left=rgb_arr[0, c], right=rgb_arr[-1, c])
        return rgb.reshape(vals.shape + (3,))



    def _terrain_rgba_from_grid(
        self,
        z_grid: np.ndarray,
        palette: str = "terrain",
        shade_strength: float = 0.75,
        coast_enhance: bool = True,
        light_alt_deg: float = 45.0,
        light_az_deg: float = 315.0,
    ) -> np.ndarray:
        zf = np.asarray(z_grid, dtype=float)
        zmin = float(np.nanpercentile(zf, 2.0))
        zmax = float(np.nanpercentile(zf, 98.0))
        span = max(1e-12, zmax - zmin)
        norm = np.clip((zf - zmin) / span, 0.0, 1.0)
        pal = str(palette or "terrain").lower()
        if pal == "custom_cpt":
            rgb_custom = self._custom_cpt_rgb_from_values(zf)
            if rgb_custom is None:
                rgb = self._terrain_colormap_rgb(norm, palette="terrain")
            else:
                rgb = rgb_custom
        elif pal == "gmt":
            # GMT风格：海陆分别拉伸色带，海岸更自然
            sea = zf < 0.0
            land = ~sea
            rgb = np.zeros(zf.shape + (3,), dtype=float)
            if np.any(sea):
                z_sea = zf[sea]
                smin = float(np.nanpercentile(z_sea, 2.0))
                smax = float(np.nanpercentile(z_sea, 98.0))
                sspan = max(1e-12, smax - smin)
                s_norm = np.clip((z_sea - smin) / sspan, 0.0, 1.0)
                s_rgb = self._terrain_colormap_rgb(s_norm, palette="gmt")
                rgb[sea] = s_rgb
            if np.any(land):
                z_land = zf[land]
                lmin = float(np.nanpercentile(z_land, 2.0))
                lmax = float(np.nanpercentile(z_land, 98.0))
                lspan = max(1e-12, lmax - lmin)
                l_norm = np.clip((z_land - lmin) / lspan, 0.0, 1.0)
                l_rgb = self._terrain_colormap_rgb(l_norm, palette="terrain")
                rgb[land] = l_rgb
        else:
            rgb = self._terrain_colormap_rgb(norm, palette=pal)
        # 改进光照：近似GMT hillshade
        gy, gx = np.gradient(zf)
        slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        az = np.radians(float(light_az_deg))
        alt = np.radians(float(light_alt_deg))
        hill = (
            np.sin(alt) * np.sin(slope)
            + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        )
        hill = np.clip((hill + 1.0) * 0.5, 0.0, 1.0)
        strength = float(np.clip(shade_strength, 0.0, 1.0))
        shade = (1.0 - 0.4 * strength) + (0.8 * strength) * hill
        rgb = np.clip(rgb * shade[..., None], 0.0, 255.0)
        if bool(coast_enhance):
            land = zf >= 0.0
            edge = np.zeros_like(land, dtype=bool)
            edge[1:, :] |= land[1:, :] != land[:-1, :]
            edge[:, 1:] |= land[:, 1:] != land[:, :-1]
            # 简单膨胀一圈，让海岸线在缩放时更明显
            edge2 = edge.copy()
            edge2[:-1, :] |= edge[1:, :]
            edge2[1:, :] |= edge[:-1, :]
            edge2[:, :-1] |= edge[:, 1:]
            edge2[:, 1:] |= edge[:, :-1]
            coast_color = np.array([248.0, 242.0, 150.0], dtype=float)
            blend = 0.78
            rgb[edge2] = (1.0 - blend) * rgb[edge2] + blend * coast_color
        rgba = np.zeros(zf.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = rgb.astype(np.uint8)
        rgba[..., 3] = 185
        if bool(coast_enhance):
            rgba[..., 3] = np.where(edge2, 235, rgba[..., 3])
        return rgba



    def _apply_location_map_terrain_overlay(self) -> None:
        if self._location_map_plot_item is None:
            return
        meta = self._location_map_terrain_meta
        if meta is None:
            return

        # 缓存：参数未变化时不重复重建地形底图
        mode = str(meta.get("mode", ""))
        coord_kind = str(meta.get("coord_kind", "unknown")).lower()
        map_mode = str(getattr(self, "_location_map_mode", "none")).lower()
        use_sac2y_tm = bool(getattr(self, "_location_map_terrain_use_sac2y_tm", False))
        pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
        shade = float(getattr(self, "_location_map_terrain_shade_strength", 0.75))
        light_alt = float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0))
        light_az = float(getattr(self, "_location_map_terrain_light_az_deg", 315.0))
        coast = bool(getattr(self, "_location_map_terrain_coast_enhance", True))
        swap_ll = bool(getattr(self, "_location_map_terrain_swap_lonlat", False))
        force_geo = bool(getattr(self, "_location_map_terrain_force_geo", False))
        cpt_path = str(getattr(self, "_location_map_terrain_cpt_path", "") or "")
        cpt_mtime = 0.0
        if cpt_path:
            try:
                cpt_mtime = float(Path(cpt_path).stat().st_mtime)
            except Exception:
                cpt_mtime = 0.0
        cache_key: Tuple[object, ...] = (
            "terrain",
            str(meta.get("path", "")),
            mode,
            coord_kind,
            map_mode,
            use_sac2y_tm,
            float(getattr(self, "_location_map_terrain_tm_lon0", 120.0)),
            bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True)),
            bool(getattr(self, "_location_map_terrain_manual_zone_enabled", False)),
            int(getattr(self, "_location_map_terrain_manual_zone_value", 0) or 0),
            str(getattr(self, "_location_map_terrain_manual_hemi", "auto")),
            pal,
            float(shade),
            float(light_alt),
            float(light_az),
            bool(coast),
            bool(swap_ll),
            bool(force_geo),
            cpt_path,
            cpt_mtime,
        )
        if (
            self._location_map_terrain_item is not None
            and self._location_map_terrain_cache_key == cache_key
        ):
            return

        self._location_map_terrain_cache_key = cache_key
        if self._location_map_terrain_item is not None:
            try:
                self._location_map_plot_item.removeItem(self._location_map_terrain_item)
            except Exception:
                pass
            self._location_map_terrain_item = None

        def _set_proj_info_text(text: str) -> None:
            self._location_map_terrain_proj_text = str(text)
            if self._location_map_terrain_proj_label is not None:
                self._location_map_terrain_proj_label.setText(self._location_map_terrain_proj_text)

        def _format_proj_info(info: Dict[str, object]) -> str:
            method = str(info.get("method", "") or "").lower()
            geo_order = str(info.get("geo_order", ""))
            order_text = ""
            if geo_order == "manual_swapped":
                order_text = "，经纬顺序=手动互换"
            elif geo_order == "auto_swapped":
                order_text = "，经纬顺序=自动互换"
            elif geo_order == "auto_default":
                order_text = "，经纬顺序=默认(x=lon,y=lat)"
            if method == "sac2y_tm":
                lon0_tm = float(info.get("lon0_tm", np.nan))
                wrap = bool(info.get("wrap_lon360", True))
                lon_mean = float(info.get("lon_mean", np.nan))
                lat_mean = float(info.get("lat_mean", np.nan))
                return (
                    "投影参数："
                    f"模式=SAC2Y兼容TM，lon0={lon0_tm:.6f}°（中央经线），"
                    f"lon<0加360={'开' if wrap else '关'}，"
                    f"参考点(均值) lon={lon_mean:.6f}°, lat={lat_mean:.6f}°，"
                    f"偏移规则=x+500000，南半球y+10000000（ex/ey未使用）{order_text}"
                )
            zone = int(round(float(info.get("zone", 0.0))))
            zone_auto = int(round(float(info.get("zone_auto", float(zone)))))
            epsg = int(round(float(info.get("epsg", 0.0))))
            lon0 = float(info.get("lon0", np.nan))
            lat0 = float(info.get("lat0", np.nan))
            cm = float(info.get("central_meridian", np.nan))
            hemi = "N" if float(info.get("hemisphere", 1.0)) >= 0.0 else "S"
            zone_mode = str(info.get("zone_mode", "auto"))
            hemi_mode = str(info.get("hemi_mode", "auto"))
            mode_text = "手动" if zone_mode == "manual" else "自动"
            extra = ""
            if zone_mode == "manual" and zone_auto != zone:
                extra = f"（自动建议Zone={zone_auto}）"
            if hemi_mode == "manual":
                extra += "（半球手动）"
            return (
                "投影参数："
                f"EPSG={epsg}，UTM Zone={zone}{hemi}，中央经线={cm:.3f}°，"
                f"参考点(均值) lon={lon0:.6f}°, lat={lat0:.6f}°，"
                f"分带模式={mode_text}{extra}{order_text}"
            )

        if mode == "grid":
            x = np.asarray(meta.get("x", []), dtype=float)
            y = np.asarray(meta.get("y", []), dtype=float)
            z = np.asarray(meta.get("z", []), dtype=float)
            if x.size < 2 or y.size < 2 or z.size == 0:
                raise ValueError("地形网格数据不足")
            if map_mode == "utm" and coord_kind == "geo":
                # 经纬度网格投影到UTM后不再是规则矩形，改为点叠加显示。
                xx, yy = np.meshgrid(x, y, indexing="xy")
                x_flat = xx.reshape(-1)
                y_flat = yy.reshape(-1)
                z_flat = np.asarray(z, dtype=float).reshape(-1)
                valid = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(z_flat)
                if not np.any(valid):
                    raise ValueError("地形网格无有效点")
                x_flat = x_flat[valid]
                y_flat = y_flat[valid]
                z_flat = z_flat[valid]
                lon_geo, lat_geo, geo_order = self._normalize_geo_lonlat_order(x_flat, y_flat)
                if use_sac2y_tm:
                    lon0_tm = float(getattr(self, "_location_map_terrain_tm_lon0", 120.0))
                    wrap_tm = bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True))
                    x_utm, y_utm, proj_info = self._convert_lonlat_to_sac2y_tm_arrays(
                        lon_geo,
                        lat_geo,
                        lon0=lon0_tm,
                        wrap_lon360=wrap_tm,
                    )
                else:
                    zone_override, hemi_override = self._get_manual_utm_override()
                    x_utm, y_utm, proj_info = self._convert_lonlat_to_utm_arrays(
                        lon_geo,
                        lat_geo,
                        override_zone=zone_override,
                        override_hemisphere=hemi_override,
                    )
                proj_info["geo_order"] = geo_order
                zmin = float(np.nanmin(z_flat))
                zmax = float(np.nanmax(z_flat))
                span = max(1e-12, zmax - zmin)
                norm = np.clip((z_flat - zmin) / span, 0.0, 1.0)
                valid_xy = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(norm)
                if not np.any(valid_xy):
                    raise ValueError("地形投影后无有效点（可能坐标顺序或投影参数不匹配）")
                x_utm = np.asarray(x_utm[valid_xy], dtype=float)
                y_utm = np.asarray(y_utm[valid_xy], dtype=float)
                z_plot = np.asarray(z_flat[valid_xy], dtype=float)
                xmin, xmax = float(np.min(x_utm)), float(np.max(x_utm))
                ymin, ymax = float(np.min(y_utm)), float(np.max(y_utm))
                if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
                    raise ValueError("地形投影范围无效")
                # 将投影后的散点重采样成规则栅格，避免“离散点”观感
                nx = 700
                ny = max(220, int(round(nx * max(1e-9, (ymax - ymin)) / max(1e-9, (xmax - xmin)))))
                ny = min(ny, 900)
                x_edges = np.linspace(xmin, xmax, nx + 1)
                y_edges = np.linspace(ymin, ymax, ny + 1)
                sum_z, _, _ = np.histogram2d(y_utm, x_utm, bins=[y_edges, x_edges], weights=z_plot)
                cnt, _, _ = np.histogram2d(y_utm, x_utm, bins=[y_edges, x_edges])
                with np.errstate(invalid="ignore", divide="ignore"):
                    z_grid = sum_z / cnt
                finite_grid = np.isfinite(z_grid)
                if not np.any(finite_grid):
                    raise ValueError("地形重采样后无有效栅格")
                fill_value = float(np.nanmedian(z_plot))
                z_grid = np.where(np.isfinite(z_grid), z_grid, fill_value)
                pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
                shade = float(getattr(self, "_location_map_terrain_shade_strength", 0.75))
                coast = bool(getattr(self, "_location_map_terrain_coast_enhance", True))
                alt_deg = float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0))
                az_deg = float(getattr(self, "_location_map_terrain_light_az_deg", 315.0))
                rgba = self._terrain_rgba_from_grid(
                    z_grid,
                    palette=pal,
                    shade_strength=shade,
                    coast_enhance=coast,
                    light_alt_deg=alt_deg,
                    light_az_deg=az_deg,
                )
                # 直接使用栅格顺序，保持与坐标轴方向一致
                img = pg.ImageItem(rgba, axisOrder="row-major")
                img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
                img.setZValue(-96)
                self._location_map_plot_item.addItem(img)
                self._location_map_terrain_item = img
                t_bounds = (
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )
                self._apply_location_map_view_range(t_bounds)
                self._update_location_map_colorbar(float(np.nanmin(z_plot)), float(np.nanmax(z_plot)))
                text = _format_proj_info(proj_info)
                _set_proj_info_text(text)
                self._set_status_text(text, hold_ms=2800)
                return
            # 与漂移图/姿态预览同一色标（地形色带 + hillshade）
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
            shade = float(getattr(self, "_location_map_terrain_shade_strength", 0.75))
            coast = bool(getattr(self, "_location_map_terrain_coast_enhance", True))
            alt_deg = float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0))
            az_deg = float(getattr(self, "_location_map_terrain_light_az_deg", 315.0))
            rgba = self._terrain_rgba_from_grid(
                np.asarray(z, dtype=float),
                palette=pal,
                shade_strength=shade,
                coast_enhance=coast,
                light_alt_deg=alt_deg,
                light_az_deg=az_deg,
            )
            img = pg.ImageItem(rgba, axisOrder="row-major")
            img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
            self._location_map_plot_item.addItem(img)
            img.setZValue(-100)
            self._location_map_terrain_item = img
            t_bounds = (xmin, xmax, ymin, ymax)
            self._apply_location_map_view_range(t_bounds)
            self._update_location_map_colorbar(float(np.nanmin(z)), float(np.nanmax(z)))
            _set_proj_info_text("投影参数：当前地形坐标按UTM/投影坐标直接叠加（未执行经纬度转换）")
            return

        if mode == "points":
            x = np.asarray(meta.get("x", []), dtype=float)
            y = np.asarray(meta.get("y", []), dtype=float)
            z = np.asarray(meta.get("z", []), dtype=float)
            if x.size == 0:
                raise ValueError("地形点数据为空")
            if map_mode == "utm" and coord_kind == "geo":
                lon_geo, lat_geo, geo_order = self._normalize_geo_lonlat_order(x, y)
                if use_sac2y_tm:
                    lon0_tm = float(getattr(self, "_location_map_terrain_tm_lon0", 120.0))
                    wrap_tm = bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True))
                    x, y, proj_info = self._convert_lonlat_to_sac2y_tm_arrays(
                        lon_geo,
                        lat_geo,
                        lon0=lon0_tm,
                        wrap_lon360=wrap_tm,
                    )
                else:
                    zone_override, hemi_override = self._get_manual_utm_override()
                    x, y, proj_info = self._convert_lonlat_to_utm_arrays(
                        lon_geo,
                        lat_geo,
                        override_zone=zone_override,
                        override_hemisphere=hemi_override,
                    )
                proj_info["geo_order"] = geo_order
                text = _format_proj_info(proj_info)
                _set_proj_info_text(text)
                self._set_status_text(text, hold_ms=2800)
            else:
                _set_proj_info_text("投影参数：当前地形坐标按UTM/投影坐标直接叠加（未执行经纬度转换）")
            zf = np.asarray(z, dtype=float)
            zmin = float(np.nanmin(zf)) if zf.size > 0 else 0.0
            zmax = float(np.nanmax(zf)) if zf.size > 0 else 1.0
            span = max(1e-12, zmax - zmin)
            norm = np.clip((zf - zmin) / span, 0.0, 1.0)
            valid_xy = np.isfinite(x) & np.isfinite(y) & np.isfinite(norm)
            if not np.any(valid_xy):
                raise ValueError("地形点在当前投影下无有效坐标")
            x = np.asarray(x[valid_xy], dtype=float)
            y = np.asarray(y[valid_xy], dtype=float)
            norm = np.asarray(norm[valid_xy], dtype=float)
            zf = np.asarray(zf[valid_xy], dtype=float)
            # 使用全量点构建规则栅格预览，避免海量散点导致界面卡顿。
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            nx = 700
            ny = max(220, int(round(nx * max(1e-9, (ymax - ymin)) / max(1e-9, (xmax - xmin)))))
            ny = min(ny, 900)
            x_edges = np.linspace(xmin, xmax, nx + 1)
            y_edges = np.linspace(ymin, ymax, ny + 1)
            sum_z, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=zf)
            cnt, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
            with np.errstate(invalid="ignore", divide="ignore"):
                z_grid = sum_z / cnt
            if not np.any(np.isfinite(z_grid)):
                raise ValueError("地形点栅格化失败：无有效网格")
            fill_value = float(np.nanmedian(zf))
            z_grid = np.where(np.isfinite(z_grid), z_grid, fill_value)
            pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
            shade = float(getattr(self, "_location_map_terrain_shade_strength", 0.75))
            coast = bool(getattr(self, "_location_map_terrain_coast_enhance", True))
            alt_deg = float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0))
            az_deg = float(getattr(self, "_location_map_terrain_light_az_deg", 315.0))
            rgba = self._terrain_rgba_from_grid(
                z_grid,
                palette=pal,
                shade_strength=shade,
                coast_enhance=coast,
                light_alt_deg=alt_deg,
                light_az_deg=az_deg,
            )
            img = pg.ImageItem(rgba, axisOrder="row-major")
            img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
            img.setZValue(-90)
            self._location_map_plot_item.addItem(img)
            self._location_map_terrain_item = img
            t_bounds = (xmin, xmax, ymin, ymax)
            self._apply_location_map_view_range(t_bounds)
            self._update_location_map_colorbar(float(np.nanmin(zf)), float(np.nanmax(zf)))
            return



    def _normalize_depth_to_km(v: float) -> float:
        d = abs(float(v))
        return d / 1000.0 if d > 20.0 else d



    def _convert_xy_to_utm_guess(self, x: float, y: float) -> Tuple[float, float]:
        if -180.5 <= float(x) <= 360.5 and -90.5 <= float(y) <= 90.5:
            lon = np.asarray([float(x)], dtype=float)
            lat = np.asarray([float(y)], dtype=float)
            zone_override, hemi_override = self._get_manual_utm_override()
            xu, yu, _ = self._convert_lonlat_to_utm_arrays(
                lon,
                lat,
                override_zone=zone_override,
                override_hemisphere=hemi_override,
            )
            if np.isfinite(xu[0]) and np.isfinite(yu[0]):
                return float(xu[0]), float(yu[0])
        return float(x), float(y)



    def _resolve_shared_terrain_path(self) -> str:
        """解析共用地形路径：优先工程 provider，再本地已缓存路径。"""
        provider = getattr(self, "_shared_terrain_path_provider", None)
        if callable(provider):
            try:
                p = str(provider() or "").strip()
                if p:
                    return p
            except Exception:
                pass
        p = str(getattr(self, "_orientation_terrain_path", "") or "").strip()
        if p:
            return p
        try:
            return str((self._location_map_terrain_meta or {}).get("path", "") or "").strip()
        except Exception:
            return ""



    def ensure_shared_terrain_loaded(
        self,
        path: Optional[str] = None,
        *,
        force_reload: bool = False,
    ) -> bool:
        """加载水深/地形一次，供姿态校正与位置 Map 共用。

        工程输入页已指定路径时调用本方法，后续对话框无需再手动加载。
        UTM 转换失败时仍尽量把原始地形交给位置 Map（叠加层可自行投影）。
        """
        p = str(path or self._resolve_shared_terrain_path() or "").strip()
        if not p:
            return False
        try:
            if not Path(p).is_file():
                return False
        except Exception:
            return False

        cur = str(getattr(self, "_orientation_terrain_path", "") or "")
        orient_ok = (
            (not force_reload)
            and self._orientation_terrain_meta_utm is not None
            and cur == p
        )
        map_ok = (
            (not force_reload)
            and isinstance(getattr(self, "_location_map_terrain_meta", None), dict)
            and str((self._location_map_terrain_meta or {}).get("path", "")) == p
        )
        if orient_ok and map_ok:
            return True

        map_ready = bool(map_ok)
        orient_ready = bool(orient_ok)
        try:
            meta_raw: Optional[Dict[str, object]] = None
            if not orient_ok or not map_ok:
                # 需要刷新时重新读文件；若仅补 Map 且已有 raw 可复用
                if (
                    (not force_reload)
                    and isinstance(self._orientation_terrain_meta_raw, dict)
                    and cur == p
                ):
                    meta_raw = dict(self._orientation_terrain_meta_raw)
                else:
                    meta_raw = self._load_terrain_meta(p, force_geo=False)

            if meta_raw is not None:
                self._orientation_terrain_path = p
                self._orientation_terrain_meta_raw = dict(meta_raw)
                # 位置 Map：始终同步原始 meta（不依赖 UTM 是否成功）
                if not map_ok:
                    meta_map = dict(meta_raw)
                    meta_map["path"] = p
                    self._location_map_terrain_meta = meta_map
                    self._location_map_terrain_cache_key = None
                    map_ready = True
                    if self._location_map_plot_item is not None:
                        try:
                            self._apply_location_map_terrain_overlay()
                        except Exception:
                            pass
                if not orient_ok:
                    meta_utm = self._convert_terrain_meta_to_utm(meta_raw)
                    if meta_utm is not None:
                        self._orientation_terrain_meta_utm = meta_utm
                        orient_ready = True
                    else:
                        try:
                            self._debug_log(
                                "TERRAIN",
                                "shared terrain: map ok, but UTM convert failed",
                            )
                        except Exception:
                            pass
            return bool(map_ready or orient_ready)
        except Exception as exc:
            try:
                self._debug_log("TERRAIN", f"ensure_shared_terrain_loaded failed: {exc}")
            except Exception:
                pass
            return False



    def _convert_terrain_meta_to_utm(
        self, terrain_meta: Dict[str, object]
    ) -> Optional[Dict[str, object]]:
        """将地形 meta 转到 UTM（供位置 Map / 共享缓存；姿态实现已迁出）。"""
        try:
            from pyAOBS.processors.relocation.services.terrain_io import (
                convert_terrain_to_utm,
            )
            return convert_terrain_to_utm(dict(terrain_meta))
        except Exception:
            return None

