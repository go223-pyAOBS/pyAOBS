"""重力异常平面图：经纬度栅格 + 模型剖面轨迹与观测站位。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.visualization.gravity_obs_grid import (
    decimate_gravity_plan_map_arrays,
    estimate_plan_map_load_bbox,
    imshow_extent_for_1d_lon_lat,
    load_gravity_plan_map_arrays,
    profile_lon_lat_for_x_obs_km,
    read_gravity_grid_domain,
)
from .ui_chrome import apply_tool_window_chrome


class GravityPlanMapWindow(QMainWindow):
    """在世界重力 lon-lat 格网上叠加密度模型水平剖面的经纬轨迹与等价观测站位。"""

    def __init__(
        self,
        parent: QWidget,
        *,
        get_ctx: Callable[[], dict[str, Any]],
        get_gravity_filepath: Callable[[], Path],
        get_profile_endpoint_line: Callable[[], str],
        log_fn: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gravity anomaly plan map (lon–lat)")
        self.resize(740, 680)
        self._get_ctx = get_ctx
        self._get_fp = get_gravity_filepath
        self._get_ep = get_profile_endpoint_line
        self._log_fn = log_fn
        self._gravity_cache_key: tuple[str, float, str] | None = None
        self._gravity_cache_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, str, str
        ] | None = None
        self._display_max_dim = 1600

        cw = QWidget()
        self.setCentralWidget(cw)
        vl = QVBoxLayout(cw)
        bar = QHBoxLayout()
        self._chk_profile = QCheckBox("Dense profile trace")
        self._chk_profile.setChecked(True)
        self._chk_stations = QCheckBox("Observation stations (≈Full Model)")
        self._chk_stations.setChecked(True)
        self._chk_zoom = QCheckBox("Zoom to profile / stations")
        self._chk_zoom.setChecked(True)
        rb = QPushButton("Redraw")
        rb.clicked.connect(self.redraw_plan_map)
        bar.addWidget(self._chk_profile)
        bar.addWidget(self._chk_stations)
        bar.addWidget(self._chk_zoom)
        bar.addStretch(1)
        bar.addWidget(rb)
        vl.addLayout(bar)

        ext_row = QHBoxLayout()
        ext_row.addWidget(QLabel("绘图范围 (°):"))
        self._lon_min_e = QLineEdit()
        self._lon_max_e = QLineEdit()
        self._lat_min_e = QLineEdit()
        self._lat_max_e = QLineEdit()
        for w, hint in (
            (self._lon_min_e, "Lon min"),
            (self._lon_max_e, "Lon max"),
            (self._lat_min_e, "Lat min"),
            (self._lat_max_e, "Lat max"),
        ):
            w.setPlaceholderText(hint)
            w.setMinimumWidth(72)
            w.setToolTip(
                "四格都填数字则按此矩形范围制图（会与格网域求交裁剪）。"
                "留空则用下方「Zoom to profile」自动范围或不限制。"
            )
        ext_row.addWidget(QLabel("经度"))
        ext_row.addWidget(self._lon_min_e)
        ext_row.addWidget(self._lon_max_e)
        ext_row.addWidget(QLabel("纬度"))
        ext_row.addWidget(self._lat_min_e)
        ext_row.addWidget(self._lat_max_e)
        ext_row.addStretch(1)
        vl.addLayout(ext_row)

        self._fig = Figure(figsize=(8.0, 6.4))
        self._canvas = FigureCanvasQTAgg(self._fig)
        tl = NavigationToolbar2QT(self._canvas, cw)
        vl.addWidget(tl)
        vl.addWidget(self._canvas, stretch=1)
        lbl = QLabel(
            "底图为 Data directory × Grid filename 的重力场；轨迹与站位与 Full Model 使用同一经纬逻辑。"
            "若模型 x 轴无 LON/LAT，则在 Profile Lon/Lat 填 "
            '\'lon0 lat0 lon1 lat1\'（剖面两端点）或 \'lon_min lon_max lat_min lat_max\' '
            '（东经北纬范围例：110–120 & 21–23 → 空格分隔写 110 120 21 23）。'
            '勾选 Zoom 时视图在剖面/站位外包络基础上每侧至少外延 5°；'
            '或在上方四格填入经/纬最小最大值强制绘图窗口（四格须同时有效）。观测站数=min(50, nx)。'
        )
        lbl.setWordWrap(True)
        vl.addWidget(lbl)
        apply_tool_window_chrome(self)

    def _manual_plot_extent_deg(self) -> Optional[tuple[float, float, float, float]]:
        """四格均填且可解析为浮点则返回 (lon_lo, lon_hi, lat_lo, lat_hi)；全空则 None。"""
        eds = (self._lon_min_e, self._lon_max_e, self._lat_min_e, self._lat_max_e)
        parts = [e.text().strip() for e in eds]
        if not any(parts):
            return None
        if not all(parts):
            QMessageBox.warning(
                self,
                "绘图范围",
                "手动范围须在四个框内都填入数值，或全部留空以使用自动范围。",
            )
            return None
        try:
            lon_a, lon_b, lat_a, lat_b = (float(p) for p in parts)
        except ValueError:
            QMessageBox.warning(self, "绘图范围", "经纬度范围须为数字（可用小数或科学计数法）。")
            return None
        lo_x, hi_x = sorted((lon_a, lon_b))
        lo_y, hi_y = sorted((lat_a, lat_b))
        return (lo_x, hi_x, lo_y, hi_y)

    @staticmethod
    def _bbox_cache_token(bbox: Optional[tuple[float, float, float, float]]) -> str:
        if bbox is None:
            return "full"
        return ",".join(f"{float(x):.4f}" for x in bbox)

    def _load_gravity_cached(
        self,
        fp: Path,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, bool]:
        """返回 (lon, lat, z, name, units, from_cache)。"""
        mtime = float(fp.stat().st_mtime)
        token = self._bbox_cache_token(bbox)
        key = (str(fp.resolve()), mtime, token)
        if self._gravity_cache_key == key and self._gravity_cache_data is not None:
            return (*self._gravity_cache_data, True)

        dlg = QProgressDialog(
            "正在读取重力格网（按需区域）…" if bbox else "正在读取全球重力格网…",
            None,
            0,
            0,
            self,
        )
        dlg.setWindowTitle("平面重力图")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.show()
        QApplication.processEvents()

        try:
            data = load_gravity_plan_map_arrays(fp, bbox=bbox)
        finally:
            dlg.close()

        self._gravity_cache_key = key
        self._gravity_cache_data = data
        return (*data, False)

    def _plan_load_bbox_and_note(
        self,
        *,
        lon_p: np.ndarray,
        lat_p: np.ndarray,
        lon_s: Optional[np.ndarray],
        lat_s: Optional[np.ndarray],
        grid_domain: tuple[float, float, float, float],
    ) -> tuple[Optional[tuple[float, float, float, float]], str]:
        manual = self._manual_plot_extent_deg()
        if manual is not None:
            eps = np.finfo(np.float64).resolution * 4000.0
            g_lon0, g_lon1, g_la0, g_la1 = grid_domain
            cx0 = max(manual[0], g_lon0)
            cx1 = min(manual[1], g_lon1)
            cy0 = max(manual[2], g_la0)
            cy1 = min(manual[3], g_la1)
            if cx0 >= cx1 - eps or cy0 >= cy1 - eps:
                QMessageBox.warning(
                    self,
                    "绘图范围",
                    f"手动范围与格网域 [{g_lon0:.2f},{g_lon1:.2f}]×[{g_la0:.2f},{g_la1:.2f}] 无有效交集，"
                    "将尝试按 Zoom 自动范围读取。",
                )
                manual = None
            else:
                note = f"extent=manual(lon[{cx0:.4f},{cx1:.4f}],lat[{cy0:.4f},{cy1:.4f}]∩grid)"
                return (cx0, cx1, cy0, cy1), note

        load_bbox = estimate_plan_map_load_bbox(
            lon_p=lon_p,
            lat_p=lat_p,
            lon_s=lon_s,
            lat_s=lat_s,
            manual_bbox=manual,
            zoom_to_profile=self._chk_zoom.isChecked(),
            grid_domain=grid_domain,
        )
        if load_bbox is not None:
            if manual is not None:
                return load_bbox, "extent=manual∩grid"
            if self._chk_zoom.isChecked():
                return load_bbox, "extent=auto(profile+padded≥5°, subset-load)"
            return load_bbox, "extent=subset"
        return None, "extent=full grid"

    def redraw_plan_map(self) -> None:
        c = self._get_ctx()
        gd = c.get("grid_data")
        pe = c.get("profile_extractor")
        if gd is None or pe is None:
            QMessageBox.warning(self, "Warning", "需要先加载模型。")
            return
        fp = self._get_fp()
        if not fp.is_file():
            QMessageBox.warning(self, "Warning", f"未找到重力格网文件:\n{fp}")
            return
        x_coord = pe.x_coord
        try:
            x_prof_km = np.asarray(gd.coords[x_coord].values, dtype=float)
        except KeyError:
            x_prof_km = np.asarray(gd[x_coord].values, dtype=float)

        ep = self._get_ep().strip()
        lon_p, lat_p, geo_note_p = profile_lon_lat_for_x_obs_km(
            x_obs_km=x_prof_km,
            x_grid_km=x_prof_km,
            grid_data=gd,
            x_coord=x_coord,
            endpoint_line=ep or None,
        )
        if lon_p is None or lat_p is None:
            QMessageBox.warning(self, "Warning", geo_note_p or "cannot resolve lon/lat for profile.")
            return

        x_min_km = float(np.nanmin(x_prof_km))
        x_max_km = float(np.nanmax(x_prof_km))
        n_obs = min(50, max(2, int(len(x_prof_km))))
        edges = np.linspace(x_min_km, x_max_km, n_obs + 1)
        x_station = 0.5 * (edges[:-1] + edges[1:])
        lon_s: Optional[np.ndarray]
        lat_s: Optional[np.ndarray]
        geo_note_s: str = ""
        lon_s, lat_s, geo_note_s = profile_lon_lat_for_x_obs_km(
            x_obs_km=x_station,
            x_grid_km=x_prof_km,
            grid_data=gd,
            x_coord=x_coord,
            endpoint_line=ep or None,
        )

        try:
            grid_domain = read_gravity_grid_domain(fp)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        load_bbox, extent_note = self._plan_load_bbox_and_note(
            lon_p=lon_p,
            lat_p=lat_p,
            lon_s=lon_s,
            lat_s=lat_s,
            grid_domain=grid_domain,
        )
        extent_box = load_bbox

        try:
            t0 = time.perf_counter()
            lon_1d, lat_1d, z_2d, gname, units, from_cache = self._load_gravity_cached(
                fp, load_bbox
            )
            t_load = time.perf_counter() - t0
        except ImportError as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        z_plot = z_2d
        lon_plot = lon_1d
        lat_plot = lat_1d
        crop_note = "subset-read" if load_bbox is not None else "full-read"

        lon_plot, lat_plot, z_plot = decimate_gravity_plan_map_arrays(
            lon_plot, lat_plot, z_plot, max_dim=self._display_max_dim
        )
        dec_note = f"display={z_plot.shape[1]}×{z_plot.shape[0]}"

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        extent_im = imshow_extent_for_1d_lon_lat(lon_plot, lat_plot)
        pcm = ax.imshow(
            z_plot,
            extent=extent_im,
            origin="lower",
            aspect="auto",
            cmap="turbo_r",
            interpolation="nearest",
        )
        cb = self._fig.colorbar(pcm, ax=ax)
        cb_label_parts = []
        if gname:
            cb_label_parts.append(str(gname))
        if units:
            cb_label_parts.append(f"[{units}]")
        cb.set_label(" ".join(cb_label_parts) if cb_label_parts else "value")

        if self._chk_profile.isChecked():
            ax.plot(lon_p, lat_p, color="white", linestyle="-", linewidth=2.8, alpha=1.0, label="profile")
            ax.plot(lon_p, lat_p, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        if (
            self._chk_stations.isChecked()
            and lon_s is not None
            and lat_s is not None
            and np.any(np.isfinite(lon_s))
            and np.any(np.isfinite(lat_s))
        ):
            ax.scatter(
                lon_s,
                lat_s,
                c="#ff3366",
                s=42,
                marker="s",
                zorder=6,
                linewidths=0.6,
                edgecolors="white",
                label=f"stations (n={len(lon_s)})",
            )

        if extent_box is not None:
            cx0, cx1, cy0, cy1 = extent_box
            ax.set_xlim(cx0, cx1)
            ax.set_ylim(cy0, cy1)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(f"{Path(fp).name}\n{geo_note_p}", fontsize=10)
        ax.grid(True, alpha=0.35, linestyle=":", linewidth=0.6)
        ax.set_aspect("auto")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
        try:
            self._fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

        log_tail = ""
        if extent_note:
            log_tail = f"; {extent_note}"
        cached = "cache-hit" if from_cache else f"load={t_load:.2f}s"
        loaded_shape = f"{lon_1d.size}×{lat_1d.size}"
        self._log_fn(
            f"\nPlan map: source={fp} field={gname!r} loaded={loaded_shape} ({cached}, {crop_note}); "
            f"{dec_note}; profile: {geo_note_p}; stations: {geo_note_s}{log_tail}"
        )
