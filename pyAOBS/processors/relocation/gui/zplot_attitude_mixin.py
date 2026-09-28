# -*- coding: utf-8 -*-
"""从 zplotpy QtFastViewer 迁出的姿态校正实现（仅 RelocationViewer 混入）。

独立 zplotpy 不再包含本模块逻辑；姿态入口只跳转本工区。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

try:
    from pyAOBS.processors.relocation import (
        OrientationCorrectionInput,
        OrientationObservation,
        build_bathymetry_sampler,
        run_orientation_correction,
    )
    from pyAOBS.processors.relocation.polarization_features import (
        extract_polarization_features,
    )
except ImportError:  # pragma: no cover
    from .. import (  # type: ignore
        OrientationCorrectionInput,
        OrientationObservation,
        build_bathymetry_sampler,
        run_orientation_correction,
    )
    from ..polarization_features import extract_polarization_features  # type: ignore


class ZplotAttitudeMixin:
    """姿态联合校正 / 主图预览 / 结果窗 —— 挂在 RelocationViewer 上。"""

    def _init_attitude_mixin_state(self) -> None:
        """在 RelocationViewer.__init__ 中于 super() 之后调用。"""
        self._relocation_host_mode = True
        defaults = {
            "wave_pre": 0.30,
            "wave_post": 0.70,
            "att_iter": 4.0,
            "att_wtt": 0.15,
            "att_wpol": 1.0,
            "att_wsym": 0.0,
            "prior_tt_shift_sec": 0.0,
            "correct_tilt": 0.0,
            "use_rmean": 1.0,
            "use_rtrend": 1.0,
            "use_bandpass": 1.0,
            "freqlo": 3.0,
            "freqhi": 15.0,
            "npoles": 8.0,
            "izerop": 1.0,
        }
        ui = getattr(self, "_orientation_ui_params", None)
        if not isinstance(ui, dict) or not ui:
            self._orientation_ui_params = dict(defaults)
        else:
            for k, v in defaults.items():
                self._orientation_ui_params.setdefault(k, v)
        if not getattr(self, "_orientation_current_solution", None):
            self._orientation_current_solution = {
                "azimuth_deg": 0.0,
                "tilt_deg": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "prior_tt_shift_sec": 0.0,
                "tt_corr_sec": 0.0,
                "time_shift_sec": 0.0,
            }
        if not hasattr(self, "_orientation_last_applied_solution"):
            self._orientation_last_applied_solution = {}
        if not hasattr(self, "_orientation_preview_enabled"):
            self._orientation_preview_enabled = False
        if not hasattr(self, "_orientation_preview_solution"):
            self._orientation_preview_solution = {}
        if not hasattr(self, "_orientation_preview_cache"):
            self._orientation_preview_cache = None
        if not hasattr(self, "_orientation_solution_persist_cb"):
            self._orientation_solution_persist_cb = None
        if not hasattr(self, "_orientation_ui_persist_cb"):
            self._orientation_ui_persist_cb = None
        chk = getattr(self, "chk_orientation_preview_toggle", None)
        if chk is not None:
            # 仅当先前已对本控件 connect 过时再 disconnect。
            # PySide6 对「未连接的槽」调用 disconnect 会打 RuntimeWarning（异常捕获也压不住）。
            hooked = getattr(self, "_orientation_preview_toggle_hooked", None)
            if hooked is not chk:
                if hooked is not None:
                    try:
                        hooked.toggled.disconnect(self._on_orientation_preview_toggle)
                    except (TypeError, RuntimeError):
                        pass
                chk.toggled.connect(self._on_orientation_preview_toggle)
                self._orientation_preview_toggle_hooked = chk
        try:
            self.enable_relocation_host_mode(True)
        except Exception:
            pass

    @staticmethod
    def _orientation_len_to_km(v: float) -> float:
        d = abs(float(v))
        return d / 1000.0 if d > 50.0 else d

    def _invalidate_orientation_preview_cache(self) -> None:
        self._orientation_preview_cache = None

    @staticmethod
    def _orientation_solution_is_meaningful(sol: Optional[Dict[str, float]]) -> bool:
        """非全零解才值得做主图预览。"""
        if not isinstance(sol, dict) or not sol:
            return False
        for k in (
            "azimuth_deg",
            "tilt_deg",
            "dx",
            "dy",
            "dz",
            "prior_tt_shift_sec",
            "tt_corr_sec",
            "time_shift_sec",
        ):
            try:
                if abs(float(sol.get(k, 0.0) or 0.0)) > 1e-9:
                    return True
            except Exception:
                continue
        return False


    def apply_saved_orientation_preview(
        self,
        solution: Optional[Dict[str, float]] = None,
        *,
        enabled: bool = True,
    ) -> bool:
        """用已保存/当前姿态解开启主图预览（需已加载数据）。

        返回是否成功开启预览。数据未加载时仅写入当前解，不开启。
        """
        sol = dict(solution if isinstance(solution, dict) else (self._orientation_current_solution or {}))
        # 同步到当前解（作下次反演初值）
        try:
            for k in (
                "azimuth_deg",
                "tilt_deg",
                "dx",
                "dy",
                "dz",
                "prior_tt_shift_sec",
                "tt_corr_sec",
                "time_shift_sec",
            ):
                if k in sol:
                    self._orientation_current_solution[k] = float(sol[k])
        except Exception:
            pass
        if not enabled:
            self._set_orientation_main_preview(False, keep_cache=False)
            return False
        if not self._orientation_solution_is_meaningful(sol):
            return False
        if self.loaded is None:
            return False
        _tilt_pv = float(sol.get("tilt_deg", 0.0))
        if not bool(float(self._orientation_ui_params.get("correct_tilt", 0.0))):
            _tilt_pv = 0.0
        preview = {
            "azimuth_deg": float(sol.get("azimuth_deg", 0.0)),
            "tilt_deg": float(_tilt_pv),
            "dx": float(sol.get("dx", 0.0)),
            "dy": float(sol.get("dy", 0.0)),
            "dz": float(sol.get("dz", 0.0)),
            "prior_tt_shift_sec": float(sol.get("prior_tt_shift_sec", 0.0)),
            "tt_corr_sec": float(sol.get("tt_corr_sec", 0.0)),
            "time_shift_sec": float(sol.get("time_shift_sec", 0.0)),
        }
        self._set_orientation_main_preview(True, solution=preview, rebuild_if_needed=True, keep_cache=True)
        n_grp = 0
        cache = self._orientation_preview_cache if isinstance(self._orientation_preview_cache, dict) else None
        if cache is not None:
            n_grp = int(cache.get("n_groups", 0) or 0)
        try:
            self.lbl_status.setText(
                f"已加载姿态解并开启预览：az={preview['azimuth_deg']:.2f}°, "
                f"tilt={preview['tilt_deg']:.2f}°, "
                f"prior={preview['prior_tt_shift_sec']:.3f}s, "
                f"corr={preview['tt_corr_sec']:.3f}s, "
                f"final={preview['time_shift_sec']:.3f}s"
                f"（三分量组 {n_grp}）| 可用状态栏「快速切换」开关"
            )
        except Exception:
            pass
        return True


    def _update_orientation_preview_badge(self) -> None:
        lbl = getattr(self, "lbl_orientation_preview", None)
        chk = getattr(self, "chk_orientation_preview_toggle", None)
        if lbl is None:
            return
        has_cache = isinstance(self._orientation_preview_cache, dict) and ("traces" in self._orientation_preview_cache)
        if chk is not None:
            chk.blockSignals(True)
            chk.setEnabled(bool(has_cache))
            chk.setChecked(bool(self._orientation_preview_enabled))
            chk.blockSignals(False)
        if bool(self._orientation_preview_enabled):
            lbl.setText("姿态校正预览: ON")
            lbl.setStyleSheet("color:#059669; font-weight:700;")
        else:
            lbl.setText("姿态校正预览: OFF")
            lbl.setStyleSheet("color:#64748b; font-weight:600;")


    def _on_orientation_preview_toggle(self, checked: bool) -> None:
        if bool(checked):
            cache_ok = isinstance(self._orientation_preview_cache, dict) and ("traces" in self._orientation_preview_cache)
            if (not cache_ok) or (not self._orientation_preview_solution):
                self._set_status_text("尚无可用姿态校正预览缓存，请先在结果窗口点击“应用到主图预览（全道）”。", hold_ms=2000)
                chk = getattr(self, "chk_orientation_preview_toggle", None)
                if chk is not None:
                    chk.blockSignals(True)
                    chk.setChecked(False)
                    chk.blockSignals(False)
                return
        self._set_orientation_main_preview(bool(checked), rebuild_if_needed=False, keep_cache=True)


    def _ask_pick_save_mode(self, title: str) -> bool:
        """Return True to save orientation-corrected picks, False for original picks."""
        has_preview = bool(self._orientation_preview_solution)
        if not has_preview:
            return False
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle(title)
        msg.setText("保存拾取走时时，选择保存模式：")
        msg.setInformativeText("“校正后走时”仅应用姿态校正的全局 dt 偏移，不写入折合显示时移。")
        btn_raw = msg.addButton("保存原始走时（默认）", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_corr = msg.addButton("保存校正后走时", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        msg.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_raw)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_corr:
            return True
        return False


    def _build_pick_snapshot(
        self,
        use_orientation_corrected: bool = False,
    ) -> Dict[int, Dict[int, float]]:
        if self.pick_manager is None:
            return {}
        base_raw = self.pick_manager.get_all_picks()
        base: Dict[int, Dict[int, float]] = {
            int(trace_idx): {int(word): float(tpk) for word, tpk in by_word.items()}
            for trace_idx, by_word in base_raw.items()
        }
        out = dict(base)
        dt_shift = float(self._orientation_preview_solution.get("time_shift_sec", 0.0))
        if not use_orientation_corrected:
            return out
        return {
            int(trace_idx): {int(word): float(tpk) + dt_shift for word, tpk in by_word.items() if float(tpk) > 0.0}
            for trace_idx, by_word in out.items()
        }


    def _set_orientation_main_preview(
        self,
        enabled: bool,
        solution: Optional[Dict[str, float]] = None,
        rebuild_if_needed: bool = True,
        keep_cache: bool = True,
    ) -> None:
        old_sol = dict(self._orientation_preview_solution)
        if solution is not None:
            self._orientation_preview_solution = dict(solution)
            if old_sol != self._orientation_preview_solution:
                self._invalidate_orientation_preview_cache()
        self._orientation_preview_enabled = bool(enabled)
        if (not self._orientation_preview_enabled) and (not bool(keep_cache)):
            self._orientation_preview_solution = {}
            self._invalidate_orientation_preview_cache()
        if self._orientation_preview_enabled and self.loaded is not None and bool(rebuild_if_needed):
            cache_ok = isinstance(self._orientation_preview_cache, dict) and ("traces" in self._orientation_preview_cache)
            if not cache_ok:
                try:
                    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                    self._apply_orientation_solution_to_all_traces(
                        traces=self.loaded.get("traces", []),
                        offsets=np.asarray(self.loaded.get("offsets", []), dtype=float),
                        force_rebuild=True,
                    )
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
        self._update_orientation_preview_badge()
        self.request_render(delay_ms=10)


    def _apply_orientation_solution_to_all_traces(
        self,
        traces: List[np.ndarray],
        offsets: np.ndarray,
        force_rebuild: bool = False,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        if (not self._orientation_preview_enabled) or self.loaded is None:
            return traces, offsets
        sol = dict(self._orientation_preview_solution or {})
        az = float(sol.get("azimuth_deg", 0.0))
        tilt = float(sol.get("tilt_deg", 0.0))
        dx = float(sol.get("dx", 0.0))
        dy = float(sol.get("dy", 0.0))
        cache_key = (
            id(self.loaded),
            len(traces),
            float(az),
            float(tilt),
            float(dx),
            float(dy),
            float(sol.get("dz", 0.0)),
            float(sol.get("time_shift_sec", 0.0)),
            str(self._orientation_geom_mode()),
        )
        if (not force_rebuild) and isinstance(self._orientation_preview_cache, dict) and self._orientation_preview_cache.get("key") == cache_key:
            tr_cached = self._orientation_preview_cache.get("traces", traces)
            off_cached = self._orientation_preview_cache.get("offsets", offsets)
            return list(tr_cached), np.asarray(off_cached, dtype=float)

        headers = self.loaded.get("trace_headers", []) or []
        if len(headers) != len(traces):
            return traces, offsets
        try:
            from pyAOBS.processors.relocation.services.preview_apply import apply_orientation_to_gather
            from pyAOBS.geometry_roles import infer_use_utm

            out_traces, out_offsets, n_rot = apply_orientation_to_gather(
                traces,
                np.asarray(offsets, dtype=float),
                headers,
                sol,
                prefer_utm=bool(infer_use_utm(headers)),
                geom=self._orientation_geom_mode(),  # type: ignore[arg-type]
            )
        except Exception as exc:
            self._debug_log("ORIENT_PREVIEW", f"apply failed: {exc}")
            return traces, offsets

        self._orientation_preview_cache = {
            "key": cache_key,
            "traces": out_traces,
            "offsets": out_offsets,
            "n_groups": int(n_rot),
        }
        if n_rot <= 0:
            self._debug_log("ORIENT_PREVIEW", "no complete 3C groups rotated")
        return out_traces, np.asarray(out_offsets, dtype=float)


    def _convert_orientation_terrain_to_utm(self, terrain_meta: Dict[str, object]) -> Optional[Dict[str, object]]:
        mode = str((terrain_meta or {}).get("mode", "")).lower()
        coord_kind = str((terrain_meta or {}).get("coord_kind", "unknown")).lower()
        if mode not in ("grid", "points"):
            return None
        if coord_kind == "utm":
            out = dict(terrain_meta)
            out["coord_kind"] = "utm"
            return out

        if coord_kind != "geo":
            # unknown：交给 relocation.terrain_io 再猜一次（像 UTM 则直接用）
            try:
                from pyAOBS.processors.relocation.services.terrain_io import convert_terrain_to_utm

                alt = convert_terrain_to_utm(dict(terrain_meta))
                if alt is not None:
                    return alt
            except Exception:
                pass
            return None

        try:
            if mode == "grid":
                x = np.asarray(terrain_meta.get("x", []), dtype=float)
                y = np.asarray(terrain_meta.get("y", []), dtype=float)
                z = np.asarray(terrain_meta.get("z", []), dtype=float)
                if x.size < 2 or y.size < 2 or z.size == 0:
                    return None
                xx, yy = np.meshgrid(x, y, indexing="xy")
                lon, lat, _ = self._normalize_geo_lonlat_order(xx.reshape(-1), yy.reshape(-1))
                zone_override, hemi_override = self._get_manual_utm_override()
                x_utm, y_utm, _info = self._convert_lonlat_to_utm_arrays(
                    lon,
                    lat,
                    override_zone=zone_override,
                    override_hemisphere=hemi_override,
                )
                z_flat = np.asarray(z, dtype=float).reshape(-1)
                valid = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(z_flat)
                if not np.any(valid):
                    return None
                x_utm = np.asarray(x_utm[valid], dtype=float)
                y_utm = np.asarray(y_utm[valid], dtype=float)
                z_flat = np.asarray(z_flat[valid], dtype=float)
                return {
                    "mode": "points",
                    "coord_kind": "utm",
                    "x": x_utm,
                    "y": y_utm,
                    "z": z_flat,
                    "path": str(terrain_meta.get("path", "")),
                }

            x = np.asarray(terrain_meta.get("x", []), dtype=float).reshape(-1)
            y = np.asarray(terrain_meta.get("y", []), dtype=float).reshape(-1)
            z = np.asarray(terrain_meta.get("z", []), dtype=float).reshape(-1)
            lon, lat, _ = self._normalize_geo_lonlat_order(x, y)
            zone_override, hemi_override = self._get_manual_utm_override()
            x_utm, y_utm, _info = self._convert_lonlat_to_utm_arrays(
                lon,
                lat,
                override_zone=zone_override,
                override_hemisphere=hemi_override,
            )
            valid = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(z)
            if not np.any(valid):
                return None
            return {
                "mode": "points",
                "coord_kind": "utm",
                "x": np.asarray(x_utm[valid], dtype=float),
                "y": np.asarray(y_utm[valid], dtype=float),
                "z": np.asarray(z[valid], dtype=float),
                "path": str(terrain_meta.get("path", "")),
            }
        except Exception:
            return None


    def _collect_orientation_selected_points_utm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect source/receiver UTM points for V-selected traces only."""
        src_pts: List[Tuple[float, float]] = []
        rec_pts: List[Tuple[float, float]] = []
        selections = self._current_apick_waveform_selections()

        for sel in selections:
            trace_idx = int(sel.get("trace_idx", -1))
            src, rec = self._extract_trace_xyz(trace_idx)
            if np.isfinite(src[:2]).all() and (abs(float(src[0])) > 1e-9 or abs(float(src[1])) > 1e-9):
                sx, sy = self._convert_xy_to_utm_guess(float(src[0]), float(src[1]))
                src_pts.append((sx, sy))
            if np.isfinite(rec[:2]).all() and (abs(float(rec[0])) > 1e-9 or abs(float(rec[1])) > 1e-9):
                rx, ry = self._convert_xy_to_utm_guess(float(rec[0]), float(rec[1]))
                rec_pts.append((rx, ry))
        src_arr = np.asarray(src_pts, dtype=float) if src_pts else np.empty((0, 2), dtype=float)
        rec_arr = np.asarray(rec_pts, dtype=float) if rec_pts else np.empty((0, 2), dtype=float)
        return src_arr, rec_arr


    def _render_orientation_terrain_preview(
        self, plot: pg.PlotWidget, terrain_meta_utm: Optional[Dict[str, object]]
    ) -> None:
        pi = plot.getPlotItem()
        pi.clear()
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setLabels(left="Y (m)", bottom="X (m)")
        terrain_bounds: Optional[Tuple[float, float, float, float]] = None
        if terrain_meta_utm is not None:
            try:
                from pyAOBS.processors.relocation.gui.orientation_result_plots import (
                    draw_terrain_utm_underlay,
                )

                draw_terrain_utm_underlay(
                    plot,
                    dict(terrain_meta_utm),
                    palette=str(getattr(self, "_location_map_terrain_palette", "terrain") or "terrain"),
                    shade_strength=float(getattr(self, "_location_map_terrain_shade_strength", 0.75)),
                    coast_enhance=bool(getattr(self, "_location_map_terrain_coast_enhance", True)),
                )
            except Exception:
                pass
            try:
                x = np.asarray(terrain_meta_utm.get("x", []), dtype=float).reshape(-1)
                y = np.asarray(terrain_meta_utm.get("y", []), dtype=float).reshape(-1)
                valid = np.isfinite(x) & np.isfinite(y)
                if np.any(valid):
                    terrain_bounds = (
                        float(np.min(x[valid])),
                        float(np.max(x[valid])),
                        float(np.min(y[valid])),
                        float(np.max(y[valid])),
                    )
            except Exception:
                terrain_bounds = None

        src_arr, rec_arr = self._collect_orientation_selected_points_utm()
        if src_arr.size > 0:
            src_item = pg.ScatterPlotItem(
                x=src_arr[:, 0], y=src_arr[:, 1],
                size=13, pen=pg.mkPen("#7f1d1d", width=1.1), brush=pg.mkBrush("#ef4444"), symbol="t"
            )
            src_item.setZValue(20)
            pi.addItem(src_item)
        if rec_arr.size > 0:
            rec_item = pg.ScatterPlotItem(
                x=rec_arr[:, 0], y=rec_arr[:, 1],
                size=8, pen=pg.mkPen("#334155", width=1.0), brush=pg.mkBrush("#94a3b8"), symbol="o"
            )
            rec_item.setZValue(18)
            pi.addItem(rec_item)

        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        if terrain_bounds is not None:
            all_x.append(np.asarray([terrain_bounds[0], terrain_bounds[1]], dtype=float))
            all_y.append(np.asarray([terrain_bounds[2], terrain_bounds[3]], dtype=float))
        if src_arr.size > 0:
            all_x.append(src_arr[:, 0])
            all_y.append(src_arr[:, 1])
        if rec_arr.size > 0:
            all_x.append(rec_arr[:, 0])
            all_y.append(rec_arr[:, 1])
        if all_x and all_y:
            x_all = np.concatenate(all_x)
            y_all = np.concatenate(all_y)
            if x_all.size > 0 and y_all.size > 0:
                xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
                ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                dx = max(1.0, xmax - xmin)
                dy = max(1.0, ymax - ymin)
                px = 0.04 * dx
                py = 0.04 * dy
                plot.setXRange(xmin - px, xmax + px, padding=0.0)
                plot.setYRange(ymin - py, ymax + py, padding=0.0)


    def _terrain_depth_sampler_from_observations(
        self, observations: List[OrientationObservation]
    ) -> Optional[Callable[[float, float], Optional[float]]]:
        terrain_meta = getattr(self, "_orientation_terrain_meta_utm", None)
        terrain_sampler = build_bathymetry_sampler(terrain_meta)
        if terrain_sampler is None:
            return None
        coord_kind = str((terrain_meta or {}).get("coord_kind", "unknown")).lower()
        geo_vals = [o.source_xy_geo for o in observations if o.source_xy_geo is not None]
        utm_vals = [o.source_xy_utm for o in observations if o.source_xy_utm is not None]
        fallback_candidates: List[Tuple[float, float]] = []
        if coord_kind == "utm":
            if utm_vals:
                u = np.asarray(utm_vals, dtype=float)
                fallback_candidates.append((float(np.median(u[:, 0])), float(np.median(u[:, 1]))))
            if geo_vals:
                g = np.asarray(geo_vals, dtype=float)
                gx, gy = float(np.median(g[:, 0])), float(np.median(g[:, 1]))
                fallback_candidates.append(self._convert_xy_to_utm_guess(gx, gy))
        else:
            if geo_vals:
                g = np.asarray(geo_vals, dtype=float)
                fallback_candidates.append((float(np.median(g[:, 0])), float(np.median(g[:, 1]))))
            if utm_vals:
                u = np.asarray(utm_vals, dtype=float)
                fallback_candidates.append((float(np.median(u[:, 0])), float(np.median(u[:, 1]))))

        def _normalize_depth_km(v: float) -> float:
            return self._normalize_depth_to_km(float(v))

        def _try_sample(cx: float, cy: float) -> Optional[float]:
            v = terrain_sampler(float(cx), float(cy))
            if v is not None and np.isfinite(float(v)):
                return float(_normalize_depth_km(float(v)))
            return None

        def _depth_sampler(x: float, y: float) -> Optional[float]:
            xx, yy = float(x), float(y)
            if coord_kind == "utm":
                # If caller passes lon/lat while terrain is UTM, convert first.
                xx, yy = self._convert_xy_to_utm_guess(xx, yy)
            tries: List[Tuple[float, float]] = [(xx, yy)]
            if coord_kind == "geo":
                tries.append((float(y), float(x)))  # 经纬顺序兜底
            if coord_kind == "unknown":
                tries.append((float(y), float(x)))
            for cx, cy in fallback_candidates:
                tries.append((cx, cy))
                if coord_kind in ("geo", "unknown"):
                    tries.append((cy, cx))
            for cx, cy in tries:
                out = _try_sample(cx, cy)
                if out is not None:
                    return out
            return None

        return _depth_sampler


    def _sample_initial_depth_km(
        self, observations: List[OrientationObservation], depth_sampler: Callable[[float, float], Optional[float]]
    ) -> Optional[float]:
        if not observations:
            return None
        # 优先在 OBS（receiver）采样水深；炮点坐标仅作回退。
        for o in observations:
            rec = np.asarray(o.receiver_xyz[:2], dtype=float)
            if np.isfinite(rec).all() and (abs(float(rec[0])) + abs(float(rec[1]))) > 1e-12:
                d = depth_sampler(float(rec[0]), float(rec[1]))
                if d is not None and np.isfinite(float(d)) and float(d) > 0.0:
                    return float(d)
        for o in observations:
            for xy in (o.source_xy_utm, o.source_xy_geo, o.source_xyz[:2]):
                if xy is None:
                    continue
                d = depth_sampler(float(xy[0]), float(xy[1]))
                if d is not None and np.isfinite(float(d)) and float(d) > 0.0:
                    return float(d)
        return None


    def _show_orientation_input_preview(
        self, observations: List[OrientationObservation], depth_km: Optional[float], max_rows: int = 5
    ) -> bool:
        water_v = 1.5
        depth_ok = depth_km is not None and np.isfinite(float(depth_km)) and float(depth_km) > 0.0
        depth_text = f"{float(depth_km):.4f} km" if depth_ok else "无效/未采样到"
        # 预览优先展示偏移距最小的道，便于先核查近偏移数据质量。
        preview_obs = sorted(
            list(observations),
            key=lambda o: (abs(float(o.offset_km)), int(o.trace_idx)),
        )
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("姿态校正输入预览")
        dlg.resize(980, 460)
        lay = QtWidgets.QVBoxLayout(dlg)
        summary = QtWidgets.QLabel(
            f"水深: {depth_text}    海水波速: {water_v:.3f} km/s    "
            f"观测数量: {len(observations)}（按偏移距从小到大预览前 {min(len(preview_obs), max_rows)} 条）",
            dlg,
        )
        summary.setStyleSheet("font-weight:600;")
        lay.addWidget(summary)

        table = QtWidgets.QTableWidget(dlg)
        table.setColumnCount(9)
        table.setHorizontalHeaderLabels(
            [
                "道号",
                "震源点坐标(x,y,z)",
                "接收点坐标(x,y,z)",
                "偏移距(km)",
                "斜线距离(km)",
                "预测走时(s)",
                "基准折合走时(s)",
                "基准真实走时(s)",
                "残差=建议观测校正(s)",
            ]
        )
        table.setRowCount(min(len(preview_obs), max_rows))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)

        n = min(len(preview_obs), max_rows)
        for i in range(n):
            o = preview_obs[i]
            off = abs(float(o.offset_km))
            t_true_obs = float(o.t0)
            x_off = float(off)
            if self.loaded is not None:
                offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
                tidx = int(o.trace_idx)
                if offsets_all.size > tidx >= 0:
                    x_off = float(offsets_all[tidx])
            t_fold_obs = t_true_obs + float(self._compute_display_tshift(int(o.trace_idx), x_off))
            if depth_ok:
                slant = float(np.sqrt(off * off + float(depth_km) * float(depth_km)))
                t_pred = slant / water_v
                residual = t_pred - t_true_obs
                slant_text = f"{slant:.4f}"
                tpred_text = f"{t_pred:.4f}"
                res_text = f"{residual:.4f}"
            else:
                slant_text = "N/A"
                tpred_text = "N/A"
                res_text = "N/A"
            s = o.source_xyz
            r = o.receiver_xyz
            vals = [
                str(int(o.trace_idx)),
                f"({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})",
                f"({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})",
                f"{off:.4f}",
                slant_text,
                tpred_text,
                f"{t_fold_obs:.4f}",
                f"{t_true_obs:.4f}",
                res_text,
            ]
            for c, v in enumerate(vals):
                table.setItem(i, c, QtWidgets.QTableWidgetItem(v))

        table.horizontalHeader().setStretchLastSection(True)
        table.resizeColumnsToContents()
        lay.addWidget(table, stretch=1)

        if depth_ok:
            hint = QtWidgets.QLabel("确认后将按上述参数执行姿态校正。", dlg)
        else:
            hint = QtWidgets.QLabel("当前水深无效：可查看预览，但无法开始校正。请先在姿态校正窗口加载水深文件。", dlg)
        hint.setStyleSheet("color:#4b5563;")
        lay.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("取消", dlg)
        btn_ok = QtWidgets.QPushButton("确认并开始", dlg)
        btn_ok.setEnabled(bool(depth_ok))
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(dlg.accept)
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        lay.addLayout(row)
        return dlg.exec() == int(QtWidgets.QDialog.DialogCode.Accepted)


    def _find_3c_group_for_trace(self, trace_idx: int) -> Tuple[Optional[Dict[int, int]], str]:
        if self.loaded is None:
            return None, "未加载数据"
        headers = self.loaded.get("trace_headers", []) or []
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if trace_idx < 0 or trace_idx >= len(headers):
            return None, f"无效道号 {trace_idx}"

        base = headers[trace_idx]
        shot = int(getattr(base, "ishoti", 0) or 0)
        rec = int(getattr(base, "ireci", 0) or 0)
        comp_map: Dict[int, int] = {}
        for i, th in enumerate(headers):
            if int(getattr(th, "ishoti", 0) or 0) != shot:
                continue
            if int(getattr(th, "ireci", 0) or 0) != rec:
                continue
            c = int(getattr(th, "itypei", 0) or 0)
            if c in (1, 2, 3):
                comp_map[c] = i

        # Fallback: same shot + nearest offset for missing components.
        if len(comp_map) < 3 and offsets.size == len(headers):
            x0 = float(offsets[trace_idx])
            for c in (1, 2, 3):
                if c in comp_map:
                    continue
                best_i = -1
                best_dx = float("inf")
                for i, th in enumerate(headers):
                    if int(getattr(th, "ishoti", 0) or 0) != shot:
                        continue
                    if int(getattr(th, "itypei", 0) or 0) != c:
                        continue
                    dx = abs(float(offsets[i]) - x0)
                    if dx < best_dx:
                        best_dx = dx
                        best_i = i
                if best_i >= 0:
                    comp_map[c] = best_i

        missing = [name for code, name in ((1, "垂直"), (2, "径向"), (3, "切向")) if code not in comp_map]
        if missing:
            return None, "缺少三分量: " + ", ".join(missing)
        return comp_map, ""


    def _build_orientation_observations(self) -> Tuple[Optional[List[OrientationObservation]], str]:
        if self.loaded is None:
            return None, "请先加载数据"
        # 姿态校正汇总全部 V 段：apick=1 直达 + 其它次生；走时/倾角由算法只取直达
        selections = list(getattr(self, "waveform_selections", []) or [])
        if not selections:
            return None, "请先使用 V 键选择至少1段波形（apick=1=直达水波，其它=折射/反射）"

        traces = self.loaded.get("traces", [])
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if len(traces) == 0 or times.size < 4:
            return None, "当前数据无效"
        dt = float(times[1] - times[0])
        if dt <= 0:
            return None, "时间采样间隔异常"

        pre_sec = float(self._orientation_ui_params.get("wave_pre", 0.30))
        post_sec = float(self._orientation_ui_params.get("wave_post", 0.70))
        obs_list: List[OrientationObservation] = []
        for sel in selections:
            trace_idx = int(sel.get("trace_idx", -1))
            group, err = self._find_3c_group_for_trace(trace_idx)
            if group is None:
                return None, f"道 {trace_idx} 无法构造三分量：{err}"

            # 缺省按直达字 1，避免旧数据无 pick_word 时被当前 spin 误标成次生相
            try:
                pick_word = int(sel.get("pick_word", 1) or 1)
            except Exception:
                pick_word = 1
            if pick_word < 1:
                pick_word = 1
            key = (int(trace_idx), int(pick_word))
            t_ref = float(self._waveop_corrected_ttrue.get(key, float(sel.get("t_true", 0.0))))
            t0 = t_ref - pre_sec
            t1 = t_ref + post_sec
            tau = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
            if tau.size < 8:
                return None, f"道 {trace_idx} 截窗采样点不足"

            ztr = np.asarray(traces[int(group[1])], dtype=float).reshape(-1)
            rtr = np.asarray(traces[int(group[2])], dtype=float).reshape(-1)
            ttr = np.asarray(traces[int(group[3])], dtype=float).reshape(-1)
            z_win = np.interp(tau, times, ztr, left=0.0, right=0.0)
            r_win = np.interp(tau, times, rtr, left=0.0, right=0.0)
            t_win = np.interp(tau, times, ttr, left=0.0, right=0.0)
            # 轻度预处理进反演：rmean/rtrend/可选带通（无增益）
            try:
                from pyAOBS.processors.relocation.services.models import AttitudeUiParams
                from pyAOBS.processors.relocation.services.waveform_preprocess import (
                    preprocess_zrt,
                )

                ui = AttitudeUiParams.from_dict(self._orientation_ui_params)
                sr = 1.0 / float(dt) if float(dt) > 0 else 0.0
                z_win, r_win, t_win = preprocess_zrt(
                    z_win, r_win, t_win, ui, sampling_rate=sr
                )
            except Exception:
                pass

            def _rms(a: np.ndarray) -> float:
                x = np.asarray(a, dtype=float).reshape(-1)
                return float(np.sqrt(np.mean(x * x))) if x.size else 0.0

            rz, rr, rt = _rms(z_win), _rms(r_win), _rms(t_win)
            if rz > 1e-12 and max(rr, rt) < 1e-8 * rz:
                return None, (
                    f"道 {trace_idx} 的 R/T 截窗能量接近 0（相对 Z）。"
                    f"请确认 itypei=2/3 存在且与 Z 同炮检；"
                    f"RMS Z/R/T={rz:.3g}/{rr:.3g}/{rt:.3g}"
                )

            src, rec = self._extract_trace_xyz(int(group[1]))
            src_geo, src_utm = self._extract_trace_source_coords(int(group[1]))
            off_km = float(offsets[int(group[1])]) if offsets.size > int(group[1]) else 0.0

            obs_list.append(
                OrientationObservation(
                    trace_idx=int(trace_idx),
                    pick_word=int(pick_word),
                    t0=float(t_ref),
                    dt=float(dt),
                    z=np.asarray(z_win, dtype=float),
                    r=np.asarray(r_win, dtype=float),
                    t=np.asarray(t_win, dtype=float),
                    source_xyz=np.asarray(src, dtype=float),
                    receiver_xyz=np.asarray(rec, dtype=float),
                    offset_km=float(off_km),
                    source_xy_geo=np.asarray(src_geo, dtype=float) if src_geo is not None else None,
                    source_xy_utm=np.asarray(src_utm, dtype=float) if src_utm is not None else None,
                )
            )
        return obs_list, ""


    def _open_attitude_correction_dialog(self) -> None:
        # 工程输入/位置 Map 已指定地形时自动共用，无需再手动加载
        try:
            self.ensure_shared_terrain_loaded(self._resolve_shared_terrain_path() or None)
        except Exception:
            pass

        # 打开时同步主图带通参数（校正用轻度预处理，不含增益）
        try:
            self._orientation_ui_params["freqlo"] = float(self.spin_freqlo.value())
            self._orientation_ui_params["freqhi"] = float(self.spin_freqhi.value())
            self._orientation_ui_params["npoles"] = float(self.spin_npoles.value())
            self._orientation_ui_params["izerop"] = 1.0 if self.chk_zerop.isChecked() else 0.0
            self._orientation_ui_params["use_bandpass"] = (
                1.0 if bool(self.chk_filter.isChecked()) else 0.0
            )
        except Exception:
            pass

        dlg = QtWidgets.QDialog(None)
        dlg.setWindowTitle("姿态校正参数")
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dlg.resize(980, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        spin_wave_pre = QtWidgets.QDoubleSpinBox(dlg)
        spin_wave_pre.setRange(0.05, 2.5)
        spin_wave_pre.setDecimals(3)
        spin_wave_pre.setSingleStep(0.05)
        spin_wave_pre.setValue(float(self._orientation_ui_params.get("wave_pre", 0.30)))
        spin_wave_post = QtWidgets.QDoubleSpinBox(dlg)
        spin_wave_post.setRange(0.05, 3.5)
        spin_wave_post.setDecimals(3)
        spin_wave_post.setSingleStep(0.05)
        spin_wave_post.setValue(float(self._orientation_ui_params.get("wave_post", 0.70)))
        spin_iter = QtWidgets.QSpinBox(dlg)
        spin_iter.setRange(1, 20)
        spin_iter.setValue(int(round(float(self._orientation_ui_params.get("att_iter", 4.0)))))
        spin_prior_tt = QtWidgets.QDoubleSpinBox(dlg)
        spin_prior_tt.setRange(-2.0, 2.0)
        spin_prior_tt.setDecimals(3)
        spin_prior_tt.setSingleStep(0.01)
        spin_prior_tt.setValue(float(self._orientation_ui_params.get("prior_tt_shift_sec", 0.0)))
        spin_prior_tt.setToolTip(
            "观测侧全局走时 shift：正=加走时(变晚)，负=减走时(变早)。"
            "反演最优值约等于残差(预测−观测)；例如残差−1s 则校正约−1s"
        )
        spin_wtt = QtWidgets.QDoubleSpinBox(dlg)
        spin_wtt.setRange(0.0, 10.0)
        spin_wtt.setDecimals(2)
        spin_wtt.setSingleStep(0.05)
        spin_wtt.setValue(float(self._orientation_ui_params.get("att_wtt", 0.15)))
        spin_wtt.setToolTip(
            "走时项仅作用于 apick=1 直达水波；次生相不进 J_tt。"
            "默认较低：校正主要看方位/位置与波形项"
        )
        spin_wpol = QtWidgets.QDoubleSpinBox(dlg)
        spin_wpol.setRange(0.0, 10.0)
        spin_wpol.setDecimals(2)
        spin_wpol.setSingleStep(0.1)
        spin_wpol.setValue(float(self._orientation_ui_params.get("att_wpol", 1.0)))
        spin_wpol.setToolTip(
            "ppol 侧：多炮 ORI 圆一致性 + T≈0；勾选校正倾角时含入射角残差"
        )
        spin_wsym = QtWidgets.QDoubleSpinBox(dlg)
        spin_wsym.setRange(0.0, 10.0)
        spin_wsym.setDecimals(2)
        spin_wsym.setSingleStep(0.1)
        spin_wsym.setValue(float(self._orientation_ui_params.get("att_wsym", 0.0)))
        spin_wsym.setToolTip(
            "非 ppol 窗对称（默认 0）。主约束为多炮 ORI 一致性（计入 w_pol）"
        )
        form.addRow("窗前 (s)", spin_wave_pre)
        form.addRow("窗后 (s)", spin_wave_post)
        form.addRow("迭代次数", spin_iter)
        form.addRow("预置走时 shift (s)", spin_prior_tt)
        form.addRow("走时权重 wtt", spin_wtt)
        form.addRow("极化权重 wpol", spin_wpol)
        form.addRow("对称权重 wsym", spin_wsym)

        chk_correct_tilt = QtWidgets.QCheckBox("校正倾角 tilt", dlg)
        chk_correct_tilt.setChecked(bool(float(self._orientation_ui_params.get("correct_tilt", 0.0))))
        chk_correct_tilt.setToolTip(
            "仅 apick=1 直达用水深几何 INC_th=atan(x/h)；次生相不参与倾角项；"
            "若 V 段全是次生相则算法强制关闭。"
            "默认关：tilt=0；勾选后搜 ±15°（非零 tilt 会混合 Z/R）。"
        )
        form.addRow("倾角", chk_correct_tilt)

        lbl_phase_mode = QtWidgets.QLabel(
            "震相：apick=1 直达（走时+姿态）；其它字次生（默认仅姿态）。校正汇总全部 V 段。",
            dlg,
        )
        lbl_phase_mode.setWordWrap(True)
        lbl_phase_mode.setStyleSheet("color:#0f766e; font-weight:600;")
        form.addRow("震相策略", lbl_phase_mode)

        chk_rmean = QtWidgets.QCheckBox("rmean", dlg)
        chk_rmean.setChecked(bool(float(self._orientation_ui_params.get("use_rmean", 1.0))))
        chk_rmean.setToolTip("去平均（进反演）")
        chk_rtrend = QtWidgets.QCheckBox("rtrend", dlg)
        chk_rtrend.setChecked(bool(float(self._orientation_ui_params.get("use_rtrend", 1.0))))
        chk_rtrend.setToolTip("去趋势（进反演）")
        chk_bandpass = QtWidgets.QCheckBox("带通(同主图)", dlg)
        chk_bandpass.setChecked(bool(float(self._orientation_ui_params.get("use_bandpass", 1.0))))
        chk_bandpass.setToolTip(
            f"使用主图带通 fL–fH="
            f"{float(self._orientation_ui_params.get('freqlo', 3.0)):.2f}–"
            f"{float(self._orientation_ui_params.get('freqhi', 15.0)):.2f} Hz；不含增益"
        )
        prep_row = QtWidgets.QHBoxLayout()
        prep_row.addWidget(chk_rmean)
        prep_row.addWidget(chk_rtrend)
        prep_row.addWidget(chk_bandpass)
        prep_row.addStretch(1)
        prep_wrap = QtWidgets.QWidget(dlg)
        prep_wrap.setLayout(prep_row)
        form.addRow("校正波形预处理", prep_wrap)

        lbl_depth_preview = QtWidgets.QLabel("未计算", dlg)
        lbl_depth_preview.setStyleSheet("color:#0f172a; font-weight:600;")
        form.addRow("当前采样水深(km)", lbl_depth_preview)
        lay.addLayout(form)

        terrain_row = QtWidgets.QHBoxLayout()
        btn_load_terrain = QtWidgets.QPushButton("更换水深…", dlg)
        btn_load_terrain.setToolTip("工程输入页已指定地形时会自动共用；此处仅在需要更换时再选文件")
        btn_clear_terrain = QtWidgets.QPushButton("清除水深文件", dlg)
        lbl_terrain = QtWidgets.QLabel(dlg)
        lbl_terrain.setStyleSheet("color:#334155;")
        lbl_terrain.setWordWrap(True)
        terrain_row.addWidget(btn_load_terrain)
        terrain_row.addWidget(btn_clear_terrain)
        terrain_row.addWidget(lbl_terrain, stretch=1)
        lay.addLayout(terrain_row)

        plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        lay.addWidget(plot, stretch=1)
        click_info_label = QtWidgets.QLabel("点击预览中的震源点/接收点可查看该道采样与走时信息。", dlg)
        click_info_label.setStyleSheet("color:#1f2937;")
        click_info_label.setWordWrap(True)
        lay.addWidget(click_info_label)

        def _refresh_terrain_label():
            if self._orientation_terrain_meta_utm is not None:
                p = str(self._orientation_terrain_path or self._orientation_terrain_meta_utm.get("path", ""))
                lbl_terrain.setText(
                    f"已共用水深: {Path(p).name if p else '(未知)'}（UTM，与工程输入/位置 Map 共用）"
                )
            else:
                lbl_terrain.setText("未加载水深文件（可在工程输入页指定）")

        def _refresh_preview():
            self._render_orientation_terrain_preview(plot, self._orientation_terrain_meta_utm)
            # Overlay clickable points with per-trace metadata.
            try:
                observations, _err = self._build_orientation_observations()
            except Exception:
                observations = None
            if not observations:
                return
            depth_sampler = self._terrain_depth_sampler_from_observations(observations)
            spots = []
            for o in observations:
                src_xy = self._convert_xy_to_utm_guess(float(o.source_xyz[0]), float(o.source_xyz[1]))
                rec_xy = self._convert_xy_to_utm_guess(float(o.receiver_xyz[0]), float(o.receiver_xyz[1]))
                for role, xy in (("source", src_xy), ("receiver", rec_xy)):
                    if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                        continue
                    data = {
                        "role": role,
                        "trace_idx": int(o.trace_idx),
                        "offset_km": float(abs(o.offset_km)),
                        "t_obs": float(o.t0),
                        "src": np.asarray(o.source_xyz, dtype=float),
                        "rec": np.asarray(o.receiver_xyz, dtype=float),
                    }
                    if depth_sampler is not None:
                        d = depth_sampler(float(xy[0]), float(xy[1]))
                        if d is not None and np.isfinite(float(d)):
                            depth_km = self._normalize_depth_to_km(float(d))
                            slant = float(np.sqrt(float(data["offset_km"]) ** 2 + depth_km ** 2))
                            t_pred = slant / 1.5
                            data["depth_km"] = float(depth_km)
                            data["slant_km"] = float(slant)
                            data["t_pred"] = float(t_pred)
                            data["residual"] = float(t_pred - float(data["t_obs"]))
                        else:
                            data["depth_km"] = float("nan")
                    else:
                        data["depth_km"] = float("nan")

                    spots.append(
                        {
                            "pos": (float(xy[0]), float(xy[1])),
                            "size": 11.0 if role == "source" else 9.0,
                            "brush": pg.mkBrush(255, 255, 255, 0),
                            "pen": pg.mkPen(255, 255, 255, 0),
                            "symbol": "t" if role == "source" else "o",
                            "data": data,
                        }
                    )
            if not spots:
                return
            pick_item = pg.ScatterPlotItem(pxMode=True)
            pick_item.setData(spots=spots)
            pick_item.setZValue(50)
            plot.addItem(pick_item)

            def _on_pick(_item, points):
                if not points:
                    return
                data = points[0].data()
                if not isinstance(data, dict):
                    return
                role = "震源" if str(data.get("role", "")) == "source" else "接收"
                tr = int(data.get("trace_idx", -1))
                off = float(data.get("offset_km", np.nan))
                t_obs = float(data.get("t_obs", np.nan))
                src = np.asarray(data.get("src", [np.nan, np.nan, np.nan]), dtype=float)
                rec = np.asarray(data.get("rec", [np.nan, np.nan, np.nan]), dtype=float)
                depth_km = float(data.get("depth_km", np.nan))
                if np.isfinite(depth_km) and depth_km > 0.0:
                    slant = float(data.get("slant_km", np.nan))
                    t_pred = float(data.get("t_pred", np.nan))
                    residual = float(data.get("residual", np.nan))
                    click_info_label.setText(
                        f"{role}点 | 道{tr} | 水深={depth_km:.4f} km | 偏移距={off:.4f} km | "
                        f"斜距={slant:.4f} km | 预测走时={t_pred:.4f} s | 观测走时={t_obs:.4f} s | "
                        f"残差(预测−观测)={residual:.4f} s → 建议观测校正≈{residual:.4f} s\n"
                        f"S=({src[0]:.3f},{src[1]:.3f},{src[2]:.3f})  R=({rec[0]:.3f},{rec[1]:.3f},{rec[2]:.3f})"
                    )
                else:
                    click_info_label.setText(
                        f"{role}点 | 道{tr} | 未采样到有效水深 | 偏移距={off:.4f} km | 观测走时={t_obs:.4f} s\n"
                        f"S=({src[0]:.3f},{src[1]:.3f},{src[2]:.3f})  R=({rec[0]:.3f},{rec[1]:.3f},{rec[2]:.3f})"
                    )

            pick_item.sigClicked.connect(_on_pick)

        def _refresh_phase_mode_label(observations=None) -> None:
            try:
                from pyAOBS.processors.relocation.orientation_correction import (
                    resolve_phase_policy,
                )

                obs = observations
                if obs is None:
                    obs, _err = self._build_orientation_observations()
                if not obs:
                    lbl_phase_mode.setText(
                        "震相：尚无 V 段。约定 apick=1=直达（走时+姿态），其它=次生（仅姿态）。"
                    )
                    return
                pol = resolve_phase_policy(
                    obs,
                    w_tt=float(spin_wtt.value()),
                    correct_tilt=bool(chk_correct_tilt.isChecked()),
                )
                lbl_phase_mode.setText(pol.message)
            except Exception:
                lbl_phase_mode.setText(
                    "震相：apick=1 直达（走时+姿态）；其它字次生（默认仅姿态）。"
                )

        def _refresh_depth_preview():
            try:
                observations, _err = self._build_orientation_observations()
            except Exception:
                observations = None
            _refresh_phase_mode_label(observations)
            if not observations:
                lbl_depth_preview.setText("无V段观测")
                return
            depth_sampler = self._terrain_depth_sampler_from_observations(observations)
            if depth_sampler is None:
                lbl_depth_preview.setText("未加载水深")
                return
            depth0 = self._sample_initial_depth_km(observations, depth_sampler)
            if depth0 is None or (not np.isfinite(float(depth0))) or float(depth0) <= 0.0:
                lbl_depth_preview.setText("采样失败")
                return
            lbl_depth_preview.setText(f"{float(depth0):.4f}")

        def _load_terrain():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择水深/地形文件",
                "",
                "Terrain (*.grd *.nc *.xyz *.txt);;NetCDF (*.grd *.nc);;XYZ (*.xyz *.txt);;All files (*)",
                options=self._file_dialog_options(),
            )
            if not path:
                return
            try:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                self._set_status_text("正在加载并转换水深文件...", hold_ms=2000)
                QtWidgets.QApplication.processEvents()
                if not self.ensure_shared_terrain_loaded(path, force_reload=True):
                    raise ValueError("该水深文件无法加载或转换为UTM坐标")
                _refresh_terrain_label()
                _refresh_preview()
                _refresh_depth_preview()
                self._set_status_text(f"水深文件已加载并转换UTM：{Path(path).name}", hold_ms=1800)
            except Exception as exc:
                self._show_themed_info("加载水深失败", str(exc))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        def _clear_terrain():
            self._orientation_terrain_path = ""
            self._orientation_terrain_meta_raw = None
            self._orientation_terrain_meta_utm = None
            _refresh_terrain_label()
            _refresh_preview()
            _refresh_depth_preview()

        btn_load_terrain.clicked.connect(_load_terrain)
        btn_clear_terrain.clicked.connect(_clear_terrain)
        spin_wtt.valueChanged.connect(lambda *_: _refresh_phase_mode_label())
        chk_correct_tilt.toggled.connect(lambda *_: _refresh_phase_mode_label())
        _refresh_terrain_label()
        _refresh_preview()
        _refresh_depth_preview()

        tip = QtWidgets.QLabel(
            "说明：校正汇总全部 V 段。apick=1 直达→走时/位置+姿态；"
            "其它 apick 次生→默认仅姿态（可与直达一起估方位）。"
            "输入 = 原始截窗 + rmean/rtrend +（可选）主图带通，不含增益。",
            dlg,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#4b5563;")
        lay.addWidget(tip)
        lbl_run_progress = QtWidgets.QLabel("校正进度：未开始", dlg)
        lbl_run_progress.setStyleSheet("color:#334155;")
        bar_run_progress = QtWidgets.QProgressBar(dlg)
        bar_run_progress.setRange(0, 1)
        bar_run_progress.setValue(0)
        bar_run_progress.setFormat("%v/%m")
        lay.addWidget(lbl_run_progress)
        lay.addWidget(bar_run_progress)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("取消", dlg)
        btn_preview = QtWidgets.QPushButton("姿态参数预览", dlg)
        btn_run = QtWidgets.QPushButton("开始校正", dlg)
        btn_cancel.clicked.connect(dlg.close)

        def _store_ui_params():
            self._orientation_ui_params["wave_pre"] = float(spin_wave_pre.value())
            self._orientation_ui_params["wave_post"] = float(spin_wave_post.value())
            self._orientation_ui_params["att_iter"] = float(spin_iter.value())
            self._orientation_ui_params["prior_tt_shift_sec"] = float(spin_prior_tt.value())
            self._orientation_ui_params["att_wtt"] = float(spin_wtt.value())
            self._orientation_ui_params["att_wpol"] = float(spin_wpol.value())
            self._orientation_ui_params["att_wsym"] = float(spin_wsym.value())
            self._orientation_ui_params["correct_tilt"] = 1.0 if chk_correct_tilt.isChecked() else 0.0
            self._orientation_ui_params["use_rmean"] = 1.0 if chk_rmean.isChecked() else 0.0
            self._orientation_ui_params["use_rtrend"] = 1.0 if chk_rtrend.isChecked() else 0.0
            self._orientation_ui_params["use_bandpass"] = 1.0 if chk_bandpass.isChecked() else 0.0
            try:
                self._orientation_ui_params["freqlo"] = float(self.spin_freqlo.value())
                self._orientation_ui_params["freqhi"] = float(self.spin_freqhi.value())
                self._orientation_ui_params["npoles"] = float(self.spin_npoles.value())
                self._orientation_ui_params["izerop"] = 1.0 if self.chk_zerop.isChecked() else 0.0
            except Exception:
                pass
            cb = getattr(self, "_orientation_ui_persist_cb", None)
            if callable(cb):
                try:
                    cb(dict(self._orientation_ui_params))
                except Exception:
                    pass

        def _preview():
            _store_ui_params()
            _refresh_depth_preview()
            self._preview_orientation_input_from_current_params()
        def _run():
            _store_ui_params()
            btn_run.setEnabled(False)
            try:
                self._run_attitude_correction_with_current_params(
                    parent_dialog=dlg,
                    progress_bar=bar_run_progress,
                    progress_label=lbl_run_progress,
                )
            finally:
                btn_run.setEnabled(True)
        btn_preview.clicked.connect(_preview)
        btn_run.clicked.connect(_run)
        # 关闭对话框时落盘到内存/工区，避免未点预览就保存工区时 prior 仍为 0
        dlg.finished.connect(lambda *_: _store_ui_params())
        row.addWidget(btn_cancel)
        row.addWidget(btn_preview)
        row.addWidget(btn_run)
        lay.addLayout(row)
        self._register_floating_dialog(dlg)
        dlg.show()


    def _store_orientation_solution_memory(self, result) -> Dict[str, float]:
        """仅把解写入内存当前解（工区初值/存盘），不改波形与道头。"""
        dx, dy, dz = result.position_correction
        details = getattr(result, "details", {}) or {}
        t_prior = float(details.get("prior_time_shift_sec", 0.0))
        t_final = float(details.get("time_shift_sec", 0.0))
        t_corr = float(details.get("tt_corr_sec", t_final - t_prior))
        self._orientation_current_solution = {
            "azimuth_deg": float(result.azimuth_deg),
            "tilt_deg": float(result.tilt_deg),
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "prior_tt_shift_sec": float(t_prior),
            "tt_corr_sec": float(t_corr),
            "time_shift_sec": float(t_final),
            "initial_azimuth_deg": float(details.get("initial_azimuth_deg", 0.0)),
            "objective": float(getattr(result, "objective", float("nan"))),
            "accepted": 0.0,
        }
        return dict(self._orientation_current_solution)


    def _accept_orientation_solution(self, result) -> None:
        """接受为当前修正：旋转三分量波形 + 更新 OBS 道头几何/偏移，并可写回 .z/.hdr。"""
        if self.loaded is None:
            self._show_themed_info("接受为当前修正", "请先加载数据。")
            return
        sol = self._store_orientation_solution_memory(result)
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg.setWindowTitle("接受为当前修正")
        msg.setText(
            "将把姿态解写入当前内存数据：\n"
            "• 旋转全部完整 Z/R/T 三分量波形\n"
            "• 更新 OBS 道头几何 (dx,dy,dz) 与偏移距\n"
            "• 可选：按最终走时 final (= prior+corr) 平移拾取时间\n\n"
            "此操作会改变后续显示与保存内容；建议先另存备份。"
        )
        btn_apply = msg.addButton("应用并另存 .z", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_ow = msg.addButton("应用并覆盖原 .z", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        btn_mem = msg.addButton("仅应用内存（稍后手动保存）", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        msg.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is None or clicked not in (btn_apply, btn_ow, btn_mem):
            self.lbl_status.setText("已取消接受为当前修正")
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            from pyAOBS.processors.relocation.services.preview_apply import (
                commit_orientation_to_loaded,
            )
            from pyAOBS.geometry_roles import infer_use_utm

            headers = self.loaded.get("trace_headers", []) or []
            stats = commit_orientation_to_loaded(
                self.loaded,
                sol,
                prefer_utm=bool(infer_use_utm(headers)),
                geom=self._orientation_geom_mode(),  # type: ignore[arg-type]
            )
            # 拾取：按全局 dt 平移（与「保存校正后走时」一致）
            dt = float(sol.get("time_shift_sec", 0.0))
            n_pick = 0
            if abs(dt) > 1e-12 and self.pick_manager is not None:
                try:
                    self._push_pick_undo("姿态校正接受：拾取时间平移")
                except Exception:
                    pass
                # picks: trace_idx -> {pick_word: time}
                for tr, by_word in list(getattr(self.pick_manager, "picks", {}) or {}).items():
                    if not isinstance(by_word, dict):
                        continue
                    for pw, t0 in list(by_word.items()):
                        try:
                            self.pick_manager.add_pick(int(tr), float(t0) + dt, int(pw))
                            n_pick += 1
                        except Exception:
                            continue
                try:
                    self._sync_picks_into_trace_headers(use_orientation_corrected=False)
                except Exception:
                    pass

            # 已写入数据后，关闭主图预览，避免重复旋转
            self._set_orientation_main_preview(False, keep_cache=False)
            # 解已落到数据上：下次反演初值归零，但保留一份 last_applied 供工区存盘
            self._orientation_last_applied_solution = dict(sol)
            self._orientation_last_applied_solution["accepted"] = 1.0
            self._orientation_current_solution = {
                "azimuth_deg": 0.0,
                "tilt_deg": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "prior_tt_shift_sec": 0.0,
                "tt_corr_sec": 0.0,
                "time_shift_sec": 0.0,
                "objective": float("nan"),
                "accepted": 1.0,
            }
            # 工区仍保存「已应用的解」
            cb = getattr(self, "_orientation_solution_persist_cb", None)
            if callable(cb):
                try:
                    cb(dict(self._orientation_last_applied_solution))
                except Exception:
                    pass

            if hasattr(self, "processor") and hasattr(self.processor, "clear_cache"):
                try:
                    self.processor.clear_cache()
                except Exception:
                    pass
            self.request_render(delay_ms=20)
            summary = (
                f"已接受为当前修正：旋转 {stats.get('n_groups_rotated', 0)} 组三分量，"
                f"更新道头几何 {stats.get('n_headers_geom', 0)} 道，"
                f"偏移 {stats.get('n_headers_offset', 0)} 道，拾取平移 {n_pick} 个。"
            )
            self.lbl_status.setText(summary)

            if clicked is btn_mem:
                self._show_themed_info(
                    "接受为当前修正",
                    summary + "\n\n仅写入内存。请用「保存.z / 写入HDR」落盘。",
                )
                return

            # 写回 .z
            dfile = str(getattr(self, "_dfile", "") or "")
            if clicked is btn_ow and dfile and Path(dfile).is_file():
                out = dfile
            else:
                out, _ = self._get_save_file_name(
                    "另存校正后 .z",
                    Path(dfile).name if dfile else "corrected.z",
                    "Z files (*.z);;All (*)",
                    default_suffix=".z",
                )
            if out:
                try:
                    self._sync_picks_into_trace_headers(use_orientation_corrected=False)
                except Exception:
                    pass
                ok = False
                try:
                    ok = bool(self.loader.save_z_format(out, hfile=None, write_picks_to_data=True))
                    if ok and out == dfile:
                        # 覆盖后刷新懒加载
                        try:
                            self.loader.traces = self.loaded.get("traces")
                        except Exception:
                            pass
                except Exception as exc:
                    self._show_themed_info("保存 .z 失败", str(exc))
                    ok = False
                if ok:
                    summary += f"\n已写入 .z：{Path(out).name}"
                    # 可选写 hdr
                    hfile = str(getattr(self, "_hfile", "") or "")
                    if hfile and Path(hfile).is_file() and self.pick_manager is not None:
                        ans = QtWidgets.QMessageBox.question(
                            self,
                            "写入 HDR",
                            f"是否同步写入道头文件？\n{hfile}",
                            QtWidgets.QMessageBox.StandardButton.Yes
                            | QtWidgets.QMessageBox.StandardButton.No,
                            QtWidgets.QMessageBox.StandardButton.No,
                        )
                        if ans == QtWidgets.QMessageBox.StandardButton.Yes:
                            try:
                                if self.pick_manager.save_to_header_file(hfile, headers):
                                    summary += f"\n已写入 .hdr：{Path(hfile).name}"
                            except Exception as exc:
                                summary += f"\n写入 HDR 失败：{exc}"
                else:
                    summary += "\n保存 .z 失败或已取消。"
            self._show_themed_info("接受为当前修正", summary)
        except Exception as exc:
            self._show_themed_info("接受为当前修正失败", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()


    def _persist_orientation_solution(self, result=None) -> None:
        """保存姿态结果到工区 JSON（不旋转波形、不改道头）。"""
        if result is not None:
            self._store_orientation_solution_memory(result)
        sol = dict(self._orientation_current_solution or {})
        # 若刚接受过，优先存 last_applied
        last = getattr(self, "_orientation_last_applied_solution", None)
        if isinstance(last, dict) and self._orientation_solution_is_meaningful(last):
            if not self._orientation_solution_is_meaningful(sol) or float(sol.get("accepted", 0)) >= 1.0 and abs(float(sol.get("azimuth_deg", 0))) < 1e-12:
                sol = dict(last)
        if not self._orientation_solution_is_meaningful(sol):
            self._show_themed_info("保存姿态结果", "当前无有效姿态解可保存。")
            return
        cb = getattr(self, "_orientation_solution_persist_cb", None)
        if callable(cb):
            try:
                cb(sol)
                self.lbl_status.setText("姿态结果已写入工区（未修改 .z/.hdr 波形）")
                return
            except Exception as exc:
                self._show_themed_info("保存姿态结果失败", str(exc))
                return
        out, _ = self._get_save_file_name(
            "保存姿态结果",
            "attitude_solution.json",
            "JSON (*.json)",
            default_suffix=".json",
        )
        if not out:
            return
        payload = {
            "attitude_solution": {
                k: float(sol.get(k, 0.0))
                for k in (
                    "azimuth_deg",
                    "tilt_deg",
                    "dx",
                    "dy",
                    "dz",
                    "prior_tt_shift_sec",
                    "tt_corr_sec",
                    "time_shift_sec",
                )
            },
            "attitude_ui": dict(self._orientation_ui_params),
        }
        Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.lbl_status.setText(f"姿态结果已保存：{Path(out).name}")


    def _solution_dict_to_result_proxy(self, sol: Optional[Dict[str, float]] = None):
        """用已保存解构造结果可视化所需的轻量对象。"""
        d = dict(sol if isinstance(sol, dict) else (self._orientation_current_solution or {}))

        class _Proxy:
            pass

        p = _Proxy()
        p.azimuth_deg = float(d.get("azimuth_deg", 0.0))
        p.tilt_deg = float(d.get("tilt_deg", 0.0))
        p.position_correction = (
            float(d.get("dx", 0.0)),
            float(d.get("dy", 0.0)),
            float(d.get("dz", 0.0)),
        )
        p.objective = float(d.get("objective", float("nan")))
        prior = float(d.get("prior_tt_shift_sec", 0.0))
        final = float(d.get("time_shift_sec", 0.0))
        corr = float(d.get("tt_corr_sec", final - prior))
        p.details = {
            "prior_time_shift_sec": prior,
            "tt_corr_sec": corr,
            "time_shift_sec": final,
            "initial_azimuth_deg": float(d.get("initial_azimuth_deg", 0.0)),
        }
        p.success = True
        p.message = ""
        p.iteration_history = []
        p.source_depth_history = []
        return p


    def show_orientation_solution_result_figures(
        self,
        solution: Optional[Dict[str, float]] = None,
        *,
        apply_main_section_preview: bool = False,
    ) -> bool:
        """打开校正结果图（三分量/极化等），默认不套整剖面主图预览。"""
        sol = dict(solution if isinstance(solution, dict) else (self._orientation_current_solution or {}))
        if not self._orientation_solution_is_meaningful(sol):
            return False
        observations, err = self._build_orientation_observations()
        if not observations:
            self._show_themed_info("预览姿态解", f"无法生成观测：{err or '无 V 段'}")
            return False
        proxy = self._solution_dict_to_result_proxy(sol)
        self._show_orientation_corrected_visuals(
            observations,
            proxy,
            auto_apply_main_preview=bool(apply_main_section_preview),
        )
        return True


    def _show_orientation_diagnostics(self, result) -> None:
        """兼容旧入口：统一打开页签结果窗（含迭代诊断），不再单独弹诊断窗。"""
        observations, err = self._build_orientation_observations()
        if not observations:
            try:
                from pyAOBS.processors.relocation.gui.orientation_result_plots import (
                    build_diagnostics_page,
                )

                page = build_diagnostics_page(result, parent=None)
                if page is None:
                    return
                dlg = QtWidgets.QDialog(None)
                dlg.setWindowTitle("姿态校正迭代诊断")
                dlg.setModal(False)
                lay = QtWidgets.QVBoxLayout(dlg)
                lay.addWidget(page)
                self._register_floating_dialog(dlg)
                dlg.show()
            except Exception:
                return
            return
        self._show_orientation_corrected_visuals(
            observations, result, auto_apply_main_preview=False
        )


    def _show_orientation_corrected_visuals(
        self,
        observations: List[OrientationObservation],
        result,
        *,
        auto_apply_main_preview: bool = False,
    ) -> None:
        if not observations:
            return

        az = float(getattr(result, "azimuth_deg", 0.0))
        tilt = float(getattr(result, "tilt_deg", 0.0))
        # 与「应用到主图预览」一致：未勾选校正倾角时强制 tilt=0，避免结果窗 Z 被混叠
        if not bool(float(self._orientation_ui_params.get("correct_tilt", 0.0))):
            tilt = 0.0
        pos = np.asarray(getattr(result, "position_correction", (0.0, 0.0, 0.0)), dtype=float)
        if pos.size != 3:
            pos = np.zeros(3, dtype=float)
        details = getattr(result, "details", {}) or {}
        t_prior = float(details.get("prior_time_shift_sec", 0.0))
        dt_shift = float(details.get("time_shift_sec", 0.0))
        t_corr = float(details.get("tt_corr_sec", dt_shift - t_prior))
        tt_txt = f"prior={t_prior:.3f}s, corr={t_corr:.3f}s, final={dt_shift:.3f}s"

        def _len_to_km(v: float) -> float:
            d = abs(float(v))
            return d / 1000.0 if d > 50.0 else d

        def _rotate_components_local(r: np.ndarray, t: np.ndarray, z: np.ndarray, az_deg: float, tilt_deg: float):
            from pyAOBS.processors.relocation.orientation_correction import rotate_components

            return rotate_components(r, t, z, az_deg, tilt_deg)

        pre_sec = float(self._orientation_ui_params.get("wave_pre", 0.30))
        post_sec = float(self._orientation_ui_params.get("wave_post", 0.70))
        disp_xy = np.asarray(pos[:2], dtype=float)

        conv_vals: List[float] = []
        dir_vecs: List[Optional[np.ndarray]] = []
        for o in observations:
            src_xy0 = np.asarray(o.source_xyz[:2], dtype=float)
            rec_xy0 = np.asarray(o.receiver_xyz[:2], dtype=float)
            v0 = rec_xy0 - src_xy0
            g0 = float(np.linalg.norm(v0))
            if np.isfinite(g0) and g0 > 1e-9:
                dir_vecs.append(np.asarray(v0, dtype=float))
                if np.isfinite(float(o.offset_km)) and abs(float(o.offset_km)) > 1e-9:
                    conv_vals.append(abs(float(o.offset_km)) / g0)
            else:
                dir_vecs.append(None)
        conv_km_per_coord = float(np.median(np.asarray(conv_vals, dtype=float))) if conv_vals else 0.0
        if not np.isfinite(conv_km_per_coord) or conv_km_per_coord <= 0.0:
            conv_km_per_coord = float(_len_to_km(np.linalg.norm(disp_xy))) / max(float(np.linalg.norm(disp_xy)), 1e-9)

        recs: List[Dict[str, object]] = []
        for iobs, o in enumerate(observations):
            zb = np.asarray(o.z, dtype=float).reshape(-1)
            rb = np.asarray(o.r, dtype=float).reshape(-1)
            tb = np.asarray(o.t, dtype=float).reshape(-1)
            n = int(min(zb.size, rb.size, tb.size))
            if n < 8:
                continue
            # tilt≈0：Z 样点恒等拷贝（与主图 preview_apply 一致）；R/T 仍按 az 旋转
            if abs(float(tilt)) < 1e-6:
                r2, t2, _z_unused = _rotate_components_local(rb[:n], tb[:n], zb[:n], az, 0.0)
                z2 = zb[:n].copy()
            else:
                r2, t2, z2 = _rotate_components_local(rb[:n], tb[:n], zb[:n], az, tilt)
            off_km = float(o.offset_km)
            v0 = dir_vecs[iobs] if iobs < len(dir_vecs) else None
            if v0 is not None and np.isfinite(off_km):
                nv = float(np.linalg.norm(v0))
                if np.isfinite(nv) and nv > 1e-9:
                    u0 = np.asarray(v0, dtype=float) / nv
                    delta_km = float(np.dot(disp_xy, u0)) * float(conv_km_per_coord)
                    disp_km = float(_len_to_km(np.linalg.norm(disp_xy)))
                    delta_km = float(np.clip(delta_km, -disp_km, disp_km))
                    off_km = float(off_km + delta_km)
            # 叠绘对比轴：相对拾取中心 + 折合（用更新后偏移），去掉全局 dt。
            # 全局 dt 是整 Trace 整体平移，留在对比轴上只会错开灰/彩，无法看旋转波形差异。
            tau_rel = np.linspace(-float(pre_sec), float(post_sec), n, dtype=float)
            tau_fold = tau_rel + float(self._compute_reduction_tshift(-1, float(off_km)))
            # 绝对真实时（含观测侧 dt）仅作标注/主图预览用，不参与灰彩叠绘
            tau_true = (
                np.linspace(float(o.t0) - pre_sec, float(o.t0) + post_sec, n, dtype=float)
                + float(dt_shift)
            )
            recs.append(
                {
                    "trace_idx": int(o.trace_idx),
                    "offset_km": float(off_km),
                    "z": np.asarray(z2, dtype=float),
                    "r": np.asarray(r2, dtype=float),
                    "t": np.asarray(t2, dtype=float),
                    "z0": np.asarray(zb[:n], dtype=float),
                    "r0": np.asarray(rb[:n], dtype=float),
                    "t0": np.asarray(tb[:n], dtype=float),
                    "tau_rel": np.asarray(tau_rel, dtype=float),
                    "tau_true": np.asarray(tau_true, dtype=float),
                    "tau_fold": np.asarray(tau_fold, dtype=float),
                }
            )
        if not recs:
            return
        # 注意：不要把主图增益/静校正/mute 套到短窗上——iscale=0 会把弱 R/T 直接置零，看起来像“空分量”
        recs.sort(key=lambda d: float(d.get("offset_km", 0.0)))

        def _rms(arr: np.ndarray) -> float:
            x = np.asarray(arr, dtype=float).reshape(-1)
            if x.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(x * x)))

        def _pctl_max(arrs: List[np.ndarray]) -> float:
            vals: List[float] = []
            for a in arrs:
                x = np.abs(np.asarray(a, dtype=float).reshape(-1))
                if x.size:
                    vals.append(float(np.percentile(x, 98)))
            return max(1e-9, max(vals) if vals else 1e-9)

        rms_before = {k: float(np.mean([_rms(np.asarray(r[k + "0"], dtype=float)) for r in recs])) for k in ("z", "r", "t")}
        rms_after = {k: float(np.mean([_rms(np.asarray(r[k], dtype=float)) for r in recs])) for k in ("z", "r", "t")}
        z_resid = float(
            max(
                (
                    float(np.max(np.abs(np.asarray(r["z"], dtype=float) - np.asarray(r["z0"], dtype=float))))
                    for r in recs
                ),
                default=0.0,
            )
        )

        dlg = QtWidgets.QDialog(None)
        dlg.setWindowTitle("姿态校正后结果可视化")
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dlg.resize(1400, 720)
        lay = QtWidgets.QVBoxLayout(dlg)
        tab = QtWidgets.QTabWidget(dlg)
        lay.addWidget(tab)

        # Tab0: 迭代诊断（刚跑完校正时有 history；预览已保存解可能无）
        try:
            from pyAOBS.processors.relocation.gui.orientation_result_plots import (
                build_diagnostics_page,
            )

            diag_page = build_diagnostics_page(result, parent=dlg)
            if diag_page is not None:
                tab.addTab(diag_page, "迭代诊断")
        except Exception as _exc_diag:
            self._debug_log("ORIENT_RESULT", f"diagnostics tab failed: {_exc_diag}")

        # Tab: corrected 3C V-window waveforms with updated offsets and folded times.
        wave_page = QtWidgets.QWidget(dlg)
        wave_lay = QtWidgets.QVBoxLayout(wave_page)
        tilt_note = (
            f"tilt={tilt:.4g}°：tilt=0 时 Z'=Z（|ΔZ|_max={z_resid:.3g}）。"
            if abs(float(tilt)) < 1e-9
            else f"|ΔZ|_max={z_resid:.3g}（倾角非零时 Z 与 R 混合）。"
        )
        wave_info = QtWidgets.QLabel(
            f"V 截窗三分量对比（反演同预处理；无增益）。"
            f" az={az:.2f}°, tilt={tilt:.2f}°, {tt_txt}。"
            f" 纵轴=相对拾取中心+折合（已去掉全局 dt，灰/彩对波形）；"
            f" 灰虚线=旋转前，彩实线=旋转后。"
            f" tilt=0 时仅 R/T 随方位变、Z 灰彩应重合；主图预览另叠加全局走时平移。"
            f" RMS前 Z/R/T={rms_before['z']:.3g}/{rms_before['r']:.3g}/{rms_before['t']:.3g}；"
            f" 后 Z/R/T={rms_after['z']:.3g}/{rms_after['r']:.3g}/{rms_after['t']:.3g}。"
            f" {tilt_note}"
            + (
                " ⚠ 校正前 R/T 能量接近 0：请检查三分量道头 itypei 与 V 窗是否对齐。"
                if max(rms_before["r"], rms_before["t"]) < 1e-6 * max(rms_before["z"], 1e-12)
                else ""
            ),
            wave_page,
        )
        wave_info.setWordWrap(True)
        wave_info.setStyleSheet("color:#475569;")
        wave_lay.addWidget(wave_info)

        amp_row = QtWidgets.QHBoxLayout()
        amp_row.addWidget(QtWidgets.QLabel("显示振幅", wave_page))
        spin_wave_amp = QtWidgets.QDoubleSpinBox(wave_page)
        spin_wave_amp.setRange(0.0, 2.0)
        spin_wave_amp.setDecimals(2)
        spin_wave_amp.setSingleStep(0.05)
        spin_wave_amp.setValue(1.0)
        spin_wave_amp.setToolTip("调节 wiggle 横向振幅（0–2，相对默认比例）；不影响反演数据")
        amp_row.addWidget(spin_wave_amp)
        amp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, wave_page)
        amp_slider.setRange(0, 200)  # 0.00–2.00 ×100
        amp_slider.setValue(100)
        amp_slider.setToolTip("拖动调节显示振幅（0–2）")
        amp_row.addWidget(amp_slider, stretch=1)
        amp_row.addStretch(0)
        wave_lay.addLayout(amp_row)

        wave_fig = QtWidgets.QWidget(wave_page)
        comps_row = QtWidgets.QHBoxLayout(wave_fig)
        comps_row.setContentsMargins(0, 0, 0, 0)
        wave_lay.addWidget(wave_fig, stretch=1)

        comp_labels = [("z", "垂直 Z"), ("r", "径向 R"), ("t", "切向 T")]
        all_max = {
            k: _pctl_max(
                [np.asarray(r[k], dtype=float) for r in recs]
                + [np.asarray(r[k + "0"], dtype=float) for r in recs]
            )
            for k, _ in comp_labels
        }
        off_span = abs(float(recs[-1]["offset_km"]) - float(recs[0]["offset_km"]))
        wig_scale0 = 0.08 * max(0.05, off_span if off_span > 1e-6 else 1.0)
        wave_plots: List[Tuple[str, str, pg.PlotWidget]] = []
        for k, cname in comp_labels:
            pw = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
            pw.setMinimumWidth(280)
            pw.setMinimumHeight(320)
            pw.showGrid(x=True, y=True, alpha=0.14)
            pw.setLabel("left", "相对拾取中心+折合 (s)，已去全局dt")
            pw.setLabel("bottom", "更新后偏移距 (km)")
            pw.invertY(True)
            comps_row.addWidget(pw, stretch=1)
            wave_plots.append((k, cname, pw))

        def _redraw_wave_amp(amp_factor: float = 1.0) -> None:
            amp = max(0.0, min(2.0, float(amp_factor)))
            wig = float(wig_scale0) * amp
            for k, cname, pw in wave_plots:
                pw.clear()
                color = {"z": "#1d4ed8", "r": "#b45309", "t": "#6d28d9"}.get(k, "#334155")
                for recd in recs:
                    x0 = float(recd["offset_km"])
                    y = np.asarray(recd["tau_fold"], dtype=float)
                    a0 = np.asarray(recd[k + "0"], dtype=float) / float(all_max[k])
                    a1 = np.asarray(recd[k], dtype=float) / float(all_max[k])
                    pw.plot(
                        x0 + wig * a0,
                        y,
                        pen=pg.mkPen("#94a3b8", width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                    )
                    pw.plot(x0 + wig * a1, y, pen=pg.mkPen(color, width=1.2))
                pw.setTitle(
                    f"{cname}（灰=旋转前 / 彩=旋转后；RMS {rms_before[k]:.3g}→{rms_after[k]:.3g}；"
                    f"振幅×{amp:.2f}；纵轴已去全局dt；98%归一）"
                )

        _amp_sync = {"busy": False}

        def _on_spin_amp(v: float) -> None:
            if _amp_sync["busy"]:
                return
            _amp_sync["busy"] = True
            try:
                amp_slider.blockSignals(True)
                amp_slider.setValue(int(round(float(v) * 100.0)))
                amp_slider.blockSignals(False)
                _redraw_wave_amp(float(v))
            finally:
                _amp_sync["busy"] = False

        def _on_slider_amp(iv: int) -> None:
            if _amp_sync["busy"]:
                return
            _amp_sync["busy"] = True
            try:
                amp = float(iv) / 100.0
                spin_wave_amp.blockSignals(True)
                spin_wave_amp.setValue(amp)
                spin_wave_amp.blockSignals(False)
                _redraw_wave_amp(amp)
            finally:
                _amp_sync["busy"] = False

        spin_wave_amp.valueChanged.connect(_on_spin_amp)
        amp_slider.valueChanged.connect(_on_slider_amp)
        _redraw_wave_amp(1.0)
        try:
            from pyAOBS.processors.relocation.gui.orientation_result_plots import (
                attach_save_figures_button,
            )

            attach_save_figures_button(
                wave_lay, wave_fig, default_stem="orientation_waveforms_3c", parent=wave_page
            )
        except Exception:
            pass
        tab.addTab(wave_page, "校正后三分量波形")

        # Tab2: final polarization particle-motion + principal direction.
        pol_page = QtWidgets.QWidget(dlg)
        pol_lay = QtWidgets.QVBoxLayout(pol_page)
        pol_info = QtWidgets.QLabel(
            "最终极化图：叠加波形（R/T/Z）质点轨迹与主极化方向。"
            "蓝实线=校正前轨迹，紫虚线=校正前主方向；"
            "绿实线=校正后轨迹，红虚线=校正后主方向。",
            pol_page,
        )
        pol_info.setWordWrap(True)
        pol_info.setStyleSheet("color:#475569;")
        pol_lay.addWidget(pol_info)

        nmin_after = int(min(len(np.asarray(r["r"], dtype=float)) for r in recs))
        r_stack_after = np.mean(np.asarray([np.asarray(r["r"], dtype=float)[:nmin_after] for r in recs], dtype=float), axis=0)
        t_stack_after = np.mean(np.asarray([np.asarray(r["t"], dtype=float)[:nmin_after] for r in recs], dtype=float), axis=0)
        z_stack_after = np.mean(np.asarray([np.asarray(r["z"], dtype=float)[:nmin_after] for r in recs], dtype=float), axis=0)
        feat_after = extract_polarization_features(z_stack_after, r_stack_after, t_stack_after)
        pv_after = np.asarray(feat_after.principal_vector, dtype=float)  # [R, T, Z]

        nmin_before = int(min(len(np.asarray(o.r, dtype=float)) for o in observations))
        r_stack_before = np.mean(np.asarray([np.asarray(o.r, dtype=float)[:nmin_before] for o in observations], dtype=float), axis=0)
        t_stack_before = np.mean(np.asarray([np.asarray(o.t, dtype=float)[:nmin_before] for o in observations], dtype=float), axis=0)
        z_stack_before = np.mean(np.asarray([np.asarray(o.z, dtype=float)[:nmin_before] for o in observations], dtype=float), axis=0)
        feat_before = extract_polarization_features(z_stack_before, r_stack_before, t_stack_before)
        pv_before = np.asarray(feat_before.principal_vector, dtype=float)  # [R, T, Z]

        pol_fig = QtWidgets.QWidget(pol_page)
        grid = QtWidgets.QGridLayout(pol_fig)
        grid.setContentsMargins(0, 0, 0, 0)
        pol_lay.addWidget(pol_fig, stretch=1)

        def _pm_plot(
            xa_before: np.ndarray,
            ya_before: np.ndarray,
            vx_before: float,
            vy_before: float,
            xa_after: np.ndarray,
            ya_after: np.ndarray,
            vx_after: float,
            vy_after: float,
            xlabel: str,
            ylabel: str,
            title: str,
            row: int,
            col: int,
        ) -> None:
            pw = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
            pw.showGrid(x=True, y=True, alpha=0.16)
            pw.setLabel("left", ylabel)
            pw.setLabel("bottom", xlabel)
            pw.setTitle(title)
            pw.addLegend(offset=(8, 8))
            # 校正前用高对比蓝系（避免浅灰在白底上发虚）；校正后绿/红区分轨迹与主方向
            pw.plot(xa_before, ya_before, pen=pg.mkPen("#1d4ed8", width=2.0), name="校正前轨迹")
            pw.plot(xa_after, ya_after, pen=pg.mkPen("#059669", width=2.0), name="校正后轨迹")
            rr = float(
                max(
                    np.max(np.abs(xa_before)),
                    np.max(np.abs(ya_before)),
                    np.max(np.abs(xa_after)),
                    np.max(np.abs(ya_after)),
                    1e-6,
                )
            )
            line_x_b = np.asarray([-rr, rr], dtype=float) * float(vx_before)
            line_y_b = np.asarray([-rr, rr], dtype=float) * float(vy_before)
            pw.plot(
                line_x_b,
                line_y_b,
                pen=pg.mkPen("#7c3aed", width=2.4, style=QtCore.Qt.PenStyle.DashLine),
                name="校正前主方向",
            )
            line_x_a = np.asarray([-rr, rr], dtype=float) * float(vx_after)
            line_y_a = np.asarray([-rr, rr], dtype=float) * float(vy_after)
            pw.plot(
                line_x_a,
                line_y_a,
                pen=pg.mkPen("#dc2626", width=2.4, style=QtCore.Qt.PenStyle.DashLine),
                name="校正后主方向",
            )
            grid.addWidget(pw, row, col)

        _pm_plot(
            r_stack_before, z_stack_before, float(pv_before[0]), float(pv_before[2]),
            r_stack_after, z_stack_after, float(pv_after[0]), float(pv_after[2]),
            "R", "Z", "质点运动 R-Z（前后对比）", 0, 0
        )
        _pm_plot(
            r_stack_before, t_stack_before, float(pv_before[0]), float(pv_before[1]),
            r_stack_after, t_stack_after, float(pv_after[0]), float(pv_after[1]),
            "R", "T", "质点运动 R-T（前后对比）", 0, 1
        )
        _pm_plot(
            t_stack_before, z_stack_before, float(pv_before[1]), float(pv_before[2]),
            t_stack_after, z_stack_after, float(pv_after[1]), float(pv_after[2]),
            "T", "Z", "质点运动 T-Z（前后对比）", 1, 0
        )

        summary = QtWidgets.QPlainTextEdit(pol_page)
        summary.setReadOnly(True)
        summary.setPlainText(
            "\n".join(
                [
                    "最终极化指标（叠加波形）",
                    f"校正前主方向 [R,T,Z] = ({float(pv_before[0]):.4f}, {float(pv_before[1]):.4f}, {float(pv_before[2]):.4f})",
                    f"校正后主方向 [R,T,Z] = ({float(pv_after[0]):.4f}, {float(pv_after[1]):.4f}, {float(pv_after[2]):.4f})",
                    f"校正前: 线性度={float(feat_before.linearity):.4f}, 矩形度={float(feat_before.rectilinearity):.4f}, 主导占比={float(feat_before.dominant_energy_ratio):.4f}",
                    f"校正后: 线性度={float(feat_after.linearity):.4f}, 矩形度={float(feat_after.rectilinearity):.4f}, 主导占比={float(feat_after.dominant_energy_ratio):.4f}",
                ]
            )
        )
        grid.addWidget(summary, 1, 1)
        try:
            from pyAOBS.processors.relocation.gui.orientation_result_plots import (
                attach_save_figures_button,
            )

            attach_save_figures_button(
                pol_lay, pol_fig, default_stem="orientation_polarization", parent=pol_page
            )
        except Exception:
            pass
        tab.addTab(pol_page, "最终极化图")

        # Tab3–5: OBS 漂移 / 方位对比 / 每道 ppol 分布统计
        try:
            from pyAOBS.processors.relocation.gui.orientation_result_plots import (
                build_azimuth_compare_page,
                build_obs_drift_page,
                build_ppol_distribution_page,
            )
            from pyAOBS.processors.relocation.orientation_correction import (
                compute_ppol_trace_results,
            )

            pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
            try:
                self.ensure_shared_terrain_loaded(self._resolve_shared_terrain_path() or None)
            except Exception:
                pass
            drift_page = build_obs_drift_page(
                observations,
                pos_tuple,
                parent=dlg,
                terrain_meta_utm=getattr(self, "_orientation_terrain_meta_utm", None),
                terrain_path=str(getattr(self, "_orientation_terrain_path", "") or ""),
                terrain_palette=str(getattr(self, "_location_map_terrain_palette", "terrain") or "terrain"),
                terrain_shade_strength=float(getattr(self, "_location_map_terrain_shade_strength", 0.75)),
                terrain_coast_enhance=bool(getattr(self, "_location_map_terrain_coast_enhance", True)),
            )
            tab.addTab(drift_page, "OBS 漂移图")
            # 原方位：未校正默认 0°；若 details 带初值则用之
            az_before = float(details.get("initial_azimuth_deg", 0.0) or 0.0)
            ppol_summary = compute_ppol_trace_results(
                observations,
                azimuth_deg=float(az),
                tilt_deg=float(tilt),
                position_correction=pos_tuple,
            )
            az_page = build_azimuth_compare_page(
                az_before,
                float(az),
                tilt_deg=float(tilt),
                ori_circ_mean_deg=float(ppol_summary.ori_circ_mean_deg),
                ori_mean_aligned_to_az_deg=float(ppol_summary.ori_mean_aligned_to_az_deg),
                ori_circ_std_deg=float(ppol_summary.ori_circ_std_deg),
                az_minus_ori_amb180_deg=float(ppol_summary.az_minus_ori_amb180_deg),
                parent=dlg,
            )
            tab.addTab(az_page, "方位对比")
            ppol_page = build_ppol_distribution_page(
                observations,
                azimuth_deg=float(az),
                tilt_deg=float(tilt),
                position_correction=pos_tuple,
                parent=dlg,
            )
            tab.addTab(ppol_page, "ppol 每道分布")
        except Exception as _exc_plots:
            self._debug_log("ORIENT_RESULT", f"drift/azimuth/ppol tabs failed: {_exc_plots}")

        # tilt 已在上方按 correct_tilt 规范化；主图预览同步该值
        preview_solution = {
            "azimuth_deg": float(az),
            "tilt_deg": float(tilt),
            "dx": float(pos[0]),
            "dy": float(pos[1]),
            "dz": float(pos[2]),
            "prior_tt_shift_sec": float(t_prior),
            "tt_corr_sec": float(t_corr),
            "time_shift_sec": float(dt_shift),
        }

        row = QtWidgets.QHBoxLayout()
        btn_save_tab = QtWidgets.QPushButton("保存当前页图…", dlg)
        btn_save_tab.setToolTip("导出当前页签整页内容为 PNG/JPEG（各页另有「保存本组图」仅截图区）")
        btn_save = QtWidgets.QPushButton("保存姿态结果", dlg)
        btn_save.setToolTip("写入工区 JSON / 当前解；不旋转波形、不改道头")
        btn_accept = QtWidgets.QPushButton("接受为当前修正", dlg)
        btn_accept.setToolTip("旋转 Z/R/T、更新 OBS 道头几何与偏移；可选写回 .z/.hdr")
        btn_apply = QtWidgets.QPushButton("应用到主图预览（全道）", dlg)
        btn_apply.setToolTip(
            "把 az/tilt 旋转、偏移改正与全局走时 final 套到整剖面显示；不写磁盘。"
            "tilt≈0 时 Z 样点不旋（仅整体走时平移）；折合仍用原始偏移。"
        )
        btn_clear = QtWidgets.QPushButton("清除主图预览", dlg)

        def _save_current_tab_figure() -> None:
            try:
                from pyAOBS.processors.relocation.gui.orientation_result_plots import (
                    save_widget_figure,
                )

                w = tab.currentWidget()
                if w is None:
                    return
                title = str(tab.tabText(tab.currentIndex()) or "orientation_tab")
                stem = "orientation_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)
                out = save_widget_figure(
                    w,
                    parent=dlg,
                    default_stem=stem.strip("_") or "orientation_tab",
                    caption=f"保存当前页图（{title}）",
                )
                if out:
                    self.lbl_status.setText(f"已保存结果图：{out}")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, "保存失败", str(exc))

        btn_save_tab.clicked.connect(_save_current_tab_figure)
        def _apply_preview():
            self._set_orientation_main_preview(True, solution=preview_solution)
            n_grp = 0
            cache = self._orientation_preview_cache if isinstance(self._orientation_preview_cache, dict) else None
            if cache is not None:
                n_grp = int(cache.get("n_groups", 0) or 0)
            z_note = "Z样点不旋" if abs(float(tilt)) < 1e-9 else "Z已按tilt混叠"
            msg = (
                f"已应用姿态预览：旋转 {n_grp} 组三分量；"
                f"az={az:.2f}°, tilt={tilt:.2f}°（{z_note}）；"
                f"{tt_txt}（主图已叠加全局走时平移）。"
                f"切换类型→径向/横向查看 R/T。"
            )
            if n_grp <= 0:
                msg += " ⚠ 未找到完整 Z/R/T 组，主图不会旋转水平分量。"
            self.lbl_status.setText(msg)
        def _clear_preview():
            self._set_orientation_main_preview(False)
            self.lbl_status.setText("已清除姿态校正主图预览")
        def _save_sol():
            self._persist_orientation_solution(result)
        def _accept_sol():
            self._accept_orientation_solution(result)
        btn_save.clicked.connect(_save_sol)
        btn_accept.clicked.connect(_accept_sol)
        btn_apply.clicked.connect(_apply_preview)
        btn_clear.clicked.connect(_clear_preview)
        row.addWidget(btn_save_tab)
        row.addWidget(btn_save)
        row.addWidget(btn_accept)
        row.addWidget(btn_apply)
        row.addWidget(btn_clear)
        row.addStretch(1)
        btn = QtWidgets.QPushButton("关闭", dlg)
        btn.clicked.connect(dlg.close)
        row.addWidget(btn)
        lay.addLayout(row)
        self._register_floating_dialog(dlg)
        dlg.show()
        # 默认只展示结果图，不自动套整剖面；需用户点「应用到主图预览」
        if auto_apply_main_preview:
            _apply_preview()


    def enable_relocation_host_mode(self, enabled: bool = True) -> None:
        """由 RelocationViewer / 姿态工区调用：恢复内嵌姿态校正入口。"""
        self._relocation_host_mode = bool(enabled)
        self._apply_zplot_relocation_entry_ui()


    def _apply_zplot_relocation_entry_ui(self) -> None:
        """嵌入 relocation 时显示内嵌「姿态」；独立 zplotpy 隐藏该按钮。"""
        host = bool(getattr(self, "_relocation_host_mode", False))
        btn = getattr(self, "btn_waveop_att", None)
        lbl = getattr(self, "lbl_orientation_preview", None)
        chk = getattr(self, "chk_orientation_preview_toggle", None)
        if btn is not None:
            btn.setVisible(host)
            if host:
                btn.setText("姿态")
                btn.setToolTip(
                    "姿态校正：基于 V 段三分量+走时联合反演方位/倾角/位置/走时。"
                    "请先用 V 选波；水深优先用工程输入页。"
                )
                btn.setEnabled(self.loaded is not None)
            else:
                btn.setEnabled(False)
        if lbl is not None:
            lbl.setVisible(host)
        if chk is not None:
            chk.setVisible(host)
        if not host:
            try:
                self._set_orientation_main_preview(False, keep_cache=False)
            except Exception:
                self._orientation_preview_enabled = False
                self._orientation_preview_solution = {}
                self._orientation_preview_cache = None


    def _on_waveop_att_clicked(self) -> None:
        if bool(getattr(self, "_relocation_host_mode", False)):
            self._run_attitude_correction_placeholder()


    def _run_attitude_correction_placeholder(self) -> None:
        self._open_attitude_correction_dialog()


    def _preview_orientation_input_from_current_params(self) -> None:
        observations, err = self._build_orientation_observations()
        if observations is None:
            self._show_themed_info("姿态参数预览", f"无法生成预览：{err}")
            return
        depth_sampler = self._terrain_depth_sampler_from_observations(observations)
        depth0 = self._sample_initial_depth_km(observations, depth_sampler) if depth_sampler is not None else None
        self._show_orientation_input_preview(observations, depth0)


    def _run_attitude_correction_with_current_params(
        self,
        parent_dialog: Optional[QtWidgets.QDialog] = None,
        progress_bar: Optional[QtWidgets.QProgressBar] = None,
        progress_label: Optional[QtWidgets.QLabel] = None,
    ) -> None:
        observations, err = self._build_orientation_observations()
        if observations is None:
            self.lbl_status.setText(f"姿态校正失败：{err}")
            self._show_themed_info("姿态校正", f"姿态校正失败：{err}")
            return

        depth_sampler = self._terrain_depth_sampler_from_observations(observations)
        depth0 = self._sample_initial_depth_km(observations, depth_sampler) if depth_sampler is not None else None
        if depth_sampler is None:
            msg = "未检测到可用水深采样（请先在姿态校正窗口加载水深文件），无法执行姿态校正。"
            self.lbl_status.setText(f"姿态校正失败：{msg}")
            self._show_themed_info("姿态校正失败", msg)
            return
        if depth0 is None or not np.isfinite(depth0) or depth0 <= 0.0:
            msg = "当前地形无法采样有效水深，无法执行姿态校正。"
            self.lbl_status.setText(f"姿态校正失败：{msg}")
            self._show_themed_info("姿态校正失败", msg)
            return

        max_iters = max(1, int(round(float(self._orientation_ui_params.get("att_iter", 4.0)))))
        if progress_bar is not None:
            progress_bar.setRange(0, max_iters)
            progress_bar.setValue(0)
        if progress_label is not None:
            progress_label.setText(f"校正进度：0/{max_iters}（准备开始）")
        QtWidgets.QApplication.processEvents()

        def _on_progress(cur_iter: int, total_iter: int, stage: str) -> None:
            total_safe = max(1, int(total_iter))
            cur_safe = max(0, min(int(cur_iter), total_safe))
            if progress_bar is not None:
                progress_bar.setMaximum(total_safe)
                progress_bar.setValue(cur_safe)
            if progress_label is not None:
                progress_label.setText(f"校正进度：{cur_safe}/{total_safe}，{stage}")
            QtWidgets.QApplication.processEvents()

        try:
            _do_tilt = bool(float(self._orientation_ui_params.get("correct_tilt", 0.0)))
            _tilt0 = (
                float(self._orientation_current_solution.get("tilt_deg", 0.0)) if _do_tilt else 0.0
            )
            result = run_orientation_correction(
                OrientationCorrectionInput(
                    observations=observations,
                    initial_azimuth_deg=float(self._orientation_current_solution.get("azimuth_deg", 0.0)),
                    initial_tilt_deg=float(_tilt0),
                    initial_position_correction=(
                        float(self._orientation_current_solution.get("dx", 0.0)),
                        float(self._orientation_current_solution.get("dy", 0.0)),
                        float(self._orientation_current_solution.get("dz", 0.0)),
                    ),
                    initial_time_shift_sec=float(
                        self._orientation_ui_params.get("prior_tt_shift_sec", 0.0)
                    ),
                    depth_sampler=depth_sampler,
                    max_iterations=max_iters,
                    w_tt=max(0.0, float(self._orientation_ui_params.get("att_wtt", 0.15))),
                    w_pol=max(0.0, float(self._orientation_ui_params.get("att_wpol", 1.0))),
                    w_sym=max(0.0, float(self._orientation_ui_params.get("att_wsym", 0.0))),
                    correct_tilt=bool(_do_tilt),
                    progress_callback=_on_progress,
                )
            )
        finally:
            if progress_bar is not None:
                progress_bar.setValue(max_iters)
            QtWidgets.QApplication.processEvents()
        if not result.success:
            if progress_label is not None:
                progress_label.setText("校正进度：失败")
            self.lbl_status.setText(f"姿态校正失败：{result.message}")
            self._show_themed_info("姿态校正失败", result.message)
            return

        dx, dy, dz = result.position_correction
        j_tt = float(result.details.get("J_tt", float("nan")))
        j_pol = float(result.details.get("J_pol", float("nan")))
        j_sym = float(result.details.get("J_sym", float("nan")))
        t_prior = float(result.details.get("prior_time_shift_sec", 0.0))
        t_corr = float(result.details.get("tt_corr_sec", 0.0))
        t_final = float(result.details.get("time_shift_sec", 0.0))
        depth_info = "未使用地形采样"
        if result.source_depth_history:
            depth_info = (
                f"水深采样轮次={len(result.source_depth_history)}，"
                f"范围[{min(result.source_depth_history):.3f}, {max(result.source_depth_history):.3f}]"
            )

        msg = (
            f"方位修正: {result.azimuth_deg:.2f}°\n"
            f"倾斜修正: {result.tilt_deg:.2f}°\n"
            f"位置修正: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}\n"
            f"走时预置 prior: {t_prior:.3f} s\n"
            f"走时校正 corr:  {t_corr:.3f} s\n"
            f"走时最终 final: {t_final:.3f} s  (= prior + corr)\n"
            f"目标函数: J={result.objective:.4f} (Jtt={j_tt:.4f}, Jpol={j_pol:.4f}, Jsym={j_sym:.4f})\n"
            f"{depth_info}"
        )
        self._show_themed_info("姿态校正结果", msg)
        # 跑完只记入内存解，便于「保存姿态结果」；真正改数据需点「接受为当前修正」
        # 结果图统一进一个页签窗（诊断/波形/极化/漂移/方位/ppol），不再多窗弹出
        self._store_orientation_solution_memory(result)
        self._show_orientation_corrected_visuals(
            observations, result, auto_apply_main_preview=False
        )
        if progress_label is not None:
            progress_label.setText(f"校正进度：完成（{max_iters}/{max_iters}）")
        self.lbl_status.setText(
            f"姿态校正完成：az={result.azimuth_deg:.2f}°, tilt={result.tilt_deg:.2f}°, "
            f"dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}。"
            f"可「保存姿态结果」或「接受为当前修正」（写回波形/道头）"
        )


