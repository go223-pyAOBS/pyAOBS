# -*- coding: utf-8 -*-
"""Waveform render loop + plot item helpers mixed into QtFastViewer."""

from __future__ import annotations

import math
import time
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

from pyAOBS.utils.qt_combo import defer_after_combo_popup, hide_combo_popup


class RenderCoreMixin:
    """request_render / _render_now、曲线池/阴影/密度图项与可见道索引。"""

    def _on_view_range_changed(self, *_args) -> None:
        if self.loaded is None:
            return
        self._sync_window_controls_from_view()
        self._viewport_interacting = True
        self.request_render(delay_ms=20)
        self._interaction_end_timer.start(220)


    def _on_interaction_end(self) -> None:
        self._viewport_interacting = False
        self._sync_window_controls_from_view()
        self.request_render(delay_ms=30)


    def request_render(self, delay_ms: int = 60, immediate: bool = False) -> None:
        if self.loaded is None:
            return
        if immediate:
            self._render_timer.stop()
            self._render_now()
            return
        self._render_timer.start(max(0, int(delay_ms)))


    def _on_combo_request_render(self, *_args) -> None:
        """Combo 选中后先收回下拉，再排队重绘，避免弹层卡住。"""
        sender = self.sender()
        combo = sender if isinstance(sender, QtWidgets.QComboBox) else None
        defer_after_combo_popup(lambda: self.request_render(), combo)


    def _extract_indices(self, include_removed: bool = False) -> np.ndarray:
        traces = self.loaded.get("traces", [])
        ntr = len(traces)
        if ntr == 0:
            return np.empty((0,), dtype=int)

        idx = np.arange(ntr, dtype=int)
        headers = self.loaded.get("trace_headers", [])

        irec = int(self.spin_irec.value())
        if irec > 0 and headers:
            mask = np.array(
                [int(getattr(headers[i], "ishoti", 0) or 0) == irec for i in idx],
                dtype=bool,
            )
            idx = idx[mask]

        itype = int(self.combo_itype.currentIndex())  # 0..4
        if itype > 0 and headers and idx.size > 0:
            mask = np.array(
                [int(getattr(headers[i], "itypei", 0) or 0) == itype for i in idx],
                dtype=bool,
            )
            idx = idx[mask]

        nskip = int(self.spin_nskip.value())
        if nskip > 0 and idx.size > 0:
            step = nskip + 1
            idx = idx[::step]
        # 姿态预览开启时必须用校正后偏移做 xmin/xmax 过滤，否则 R/T 可能被滤成“空”
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if bool(self._orientation_preview_enabled):
            cache = self._orientation_preview_cache if isinstance(self._orientation_preview_cache, dict) else None
            if cache is not None and "offsets" in cache:
                offsets = np.asarray(cache.get("offsets", offsets), dtype=float)
        if offsets.size > 0 and idx.size > 0:
            xmin = float(min(self.spin_xmin.value(), self.spin_xmax.value()))
            xmax = float(max(self.spin_xmin.value(), self.spin_xmax.value()))
            if xmin < xmax:
                mask = np.array(
                    [xmin <= float(offsets[int(i)]) <= xmax for i in idx],
                    dtype=bool,
                )
                idx = idx[mask]
        if (not include_removed) and self._removed_traces and idx.size > 0:
            removed = self._removed_traces
            idx = np.array([int(i) for i in idx if int(i) not in removed], dtype=int)
        return idx


    def _register_floating_dialog(self, dlg: QtWidgets.QDialog) -> None:
        self._floating_dialogs.append(dlg)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._floating_dialogs.remove(d) if d in self._floating_dialogs else None)


    def _render_now(self) -> None:
        if self.loaded is None:
            return
        # Show progress immediately for pre-denoise preparation stage.
        dn_render_active = bool(self._denoise_run_armed) and bool(self._denoise_params.get("enabled", False))
        dn_perf_diag = bool(self._denoise_params.get("perf_diag", False)) and bool(dn_render_active)
        perf_marks: Dict[str, float] = {}
        if dn_perf_diag:
            perf_marks["t0"] = time.perf_counter()
        if dn_render_active:
            self._set_denoise_progress(0, 100, "预处理中")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass

        t0 = time.perf_counter()
        traces = self.loaded.get("traces", [])
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if bool(self._orientation_preview_enabled):
            cache = self._orientation_preview_cache if isinstance(self._orientation_preview_cache, dict) else None
            if cache is not None and "traces" in cache and "offsets" in cache:
                traces = cache.get("traces", traces)
                offsets_all = np.asarray(cache.get("offsets", offsets_all), dtype=float)
            else:
                traces, offsets_all = self._apply_orientation_solution_to_all_traces(
                    traces=traces,
                    offsets=offsets_all,
                    force_rebuild=True,
                )
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size == 0:
            if dn_render_active:
                self._set_denoise_progress(0, -1)
            if dn_perf_diag:
                t1 = time.perf_counter()
                self._debug_log("DENOISE_PERF", f"render_now early=no_data total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f}")
            return

        if dn_render_active:
            self._set_denoise_progress(5, 100, "索引提取")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if dn_perf_diag:
            perf_marks["before_extract"] = time.perf_counter()
        idx = self._extract_indices()
        if dn_perf_diag:
            perf_marks["after_extract"] = time.perf_counter()
        if idx.size == 0:
            if dn_render_active:
                self._set_denoise_progress(0, -1)
            self._last_denoise_scope_count = 0
            self._ensure_curve_pool(0)
            self._clear_shade_item()
            self._clear_density_item()
            self._clear_stack_item()
            self._clear_static_preview_item()
            self._clear_theoretical_item()
            self._clear_txin_item()
            self._clear_txin_map_preview_item()
            self._clear_water_corr_item()
            self._clear_wave_select_item()
            self._clear_wave_select_marker_item()
            self._clear_waveop_stack_item()
            self.lbl_status.setText("当前过滤条件下无可显示道")
            if dn_perf_diag:
                t1 = time.perf_counter()
                ext_ms = (perf_marks["after_extract"] - perf_marks["before_extract"]) * 1000.0
                self._debug_log("DENOISE_PERF", f"render_now early=no_idx extract_ms={ext_ms:.1f} total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f}")
            return

        # 仅保留视窗内道，再渲染预算抽稀
        if dn_render_active:
            self._set_denoise_progress(15, 100, "视窗筛选")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if dn_perf_diag:
            perf_marks["before_visible"] = time.perf_counter()
        xcoords = offsets_all[idx]
        vis_mask = self._visible_mask(xcoords)
        idx_vis = idx[vis_mask]
        if dn_perf_diag:
            perf_marks["after_visible"] = time.perf_counter()
        if idx_vis.size == 0:
            if dn_render_active:
                self._set_denoise_progress(0, -1)
            self._last_denoise_scope_count = 0
            self._ensure_curve_pool(0)
            self._clear_shade_item()
            self._clear_density_item()
            self._clear_stack_item()
            self._clear_static_preview_item()
            self._clear_theoretical_item()
            self._clear_txin_item()
            self._clear_txin_map_preview_item()
            self._clear_water_corr_item()
            self._clear_wave_select_item()
            self._clear_wave_select_marker_item()
            self._clear_waveop_stack_item()
            self.lbl_status.setText("当前视窗内无道")
            if dn_perf_diag:
                t1 = time.perf_counter()
                vis_ms = (perf_marks["after_visible"] - perf_marks["before_visible"]) * 1000.0
                self._debug_log("DENOISE_PERF", f"render_now early=no_visible visible_ms={vis_ms:.1f} total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f}")
            return

        # 渲染预算：交互中更激进
        axis_w_px = max(1.0, float(self.plot.getViewBox().width()))
        target = int(max(250, min(1400, axis_w_px * (0.9 if self._viewport_interacting else 2.0))))
        stride = int(np.ceil(idx_vis.size / target)) if idx_vis.size > target else 1
        idx_render = idx_vis[::stride] if stride > 1 else idx_vis
        if idx_render[-1] != idx_vis[-1]:
            idx_render = np.append(idx_render, idx_vis[-1])
        # 强制渲染全部 V 选波相关道，避免抽稀后选段丢失
        all_wave_selections = list(getattr(self, "waveform_selections", []) or [])
        if all_wave_selections:
            forced = np.asarray(
                [int(s.get("trace_idx", -1)) for s in all_wave_selections],
                dtype=int,
            )
            if forced.size > 0:
                forced = forced[(forced >= 0)]
                if forced.size > 0:
                    vis_set = set(int(v) for v in idx_vis)
                    forced = np.asarray([int(v) for v in forced if int(v) in vis_set], dtype=int)
                    if forced.size > 0:
                        idx_render = np.unique(np.concatenate([idx_render, forced]))

        render_offsets = offsets_all[idx_render]
        # 必须 copy：预览缓存不能被后续 mute/处理链原地改写
        raw_traces = [np.asarray(traces[int(i)], dtype=np.float64).copy() for i in idx_render]
        # 增益/mute 用原始偏移，避免 dx/dy 改正后 iscale=1 把 Z 振幅拉歪（波形“不对”）
        offsets_raw_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
        process_offsets = np.asarray(
            [
                float(offsets_raw_all[int(i)])
                if 0 <= int(i) < offsets_raw_all.size
                else float(render_offsets[j])
                for j, i in enumerate(idx_render)
            ],
            dtype=float,
        ) if bool(self._orientation_preview_enabled) else np.asarray(render_offsets, dtype=float)
        trace_headers_all = self.loaded.get("trace_headers", [])
        render_gains = np.ones(len(idx_render), dtype=float)
        if trace_headers_all:
            for ii, gidx in enumerate(idx_render):
                ig = int(gidx)
                if 0 <= ig < len(trace_headers_all):
                    render_gains[ii] = float(max(1, int(getattr(trace_headers_all[ig], "igaini", 1) or 1)))
        self._last_render_trace_indices = np.asarray(idx_render, dtype=int)
        self._last_render_offsets = np.asarray(render_offsets, dtype=float)

        self.params.irec = int(self.spin_irec.value())
        self.params.itype = int(self.combo_itype.currentIndex())
        self.params.nskip = int(self.spin_nskip.value())
        self.params.ndecim = int(self.spin_ndecim.value())
        self.params.vred = float(self.spin_vred.value())
        self.params.xmin = float(self.spin_xmin.value())
        self.params.xmax = float(self.spin_xmax.value())
        self.params.tmin = float(self.spin_tmin.value())
        self.params.tmax = float(self.spin_tmax.value())
        self.params.amp = float(self.spin_amp.value())
        self.params.iscale = int(self.combo_iscale.currentIndex())
        self.params.rcor = float(self.spin_rcor.value())
        self.params.sf = float(self.spin_sf.value())
        self.params.tvg = float(self.spin_tvg.value())
        self.params.pvg = float(self.spin_pvg.value())
        self.params.clip = float(self.spin_clip.value())
        self.params.ibndps = 1 if self.chk_filter.isChecked() else 0
        self.params.freqlo = float(self.spin_freqlo.value())
        self.params.freqhi = float(self.spin_freqhi.value())
        self.params.npoles = int(self.spin_npoles.value())
        self.params.izerop = 1 if self.chk_zerop.isChecked() else 0
        self.params.rmean = 1 if self.chk_rmean.isChecked() else 0
        self.params.rtrend = 1 if self.chk_rtrend.isChecked() else 0
        self.params.iout = 0 if self.params.rmean else 2
        self._sync_denoise_params_from_ui()
        self.params.tcrcor = float(self.spin_tcrcor.value())
        self.params.tlag = float(self.spin_tlag.value())
        self.params.hilbratio = float(self.spin_hilbratio.value())
        mode_idx = int(self.combo_mode.currentIndex())
        if mode_idx == 1:
            self.params.ishade = 1
        elif mode_idx == 2:
            self.params.ishade = -1
        else:
            self.params.ishade = 0

        proc_params = self._build_processing_params()

        # 可选多边形 mute：多边形外部临时置零，再进入显示处理链。
        if dn_render_active:
            self._set_denoise_progress(30, 100, "构建输入")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        traces_for_processing = self._apply_mute_to_raw_traces(
            raw_traces=raw_traces,
            trace_indices=np.asarray(idx_render, dtype=int),
            render_offsets=np.asarray(render_offsets, dtype=float),
            times=np.asarray(times, dtype=float),
        )
        denoise_indices = self._resolve_denoise_indices(
            idx_all=np.asarray(idx, dtype=int),
            idx_visible=np.asarray(idx_vis, dtype=int),
            idx_render=np.asarray(idx_render, dtype=int),
        )
        self._last_denoise_scope_count = int(np.asarray(denoise_indices, dtype=int).size)

        # 处理链复用现有内核桥（realtime=True 时会自动降级带通）
        if times.size > 1:
            sr = 1.0 / float(times[1] - times[0])
        else:
            sr = None
        # iscale=1 且 sf=0 时，按稳定参考道估计 sf，避免随视窗变化导致噪声跳变
        user_sf = float(proc_params.sf)
        sf_override: Optional[float] = None
        if bool(self.chk_gain.isChecked()) and int(proc_params.iscale) == 1 and user_sf <= 0.0 and idx.size > 0:
            ref_gidx = int(idx[0])
            ref_gain = 1.0
            if trace_headers_all and 0 <= ref_gidx < len(trace_headers_all):
                ref_gain = float(max(1, int(getattr(trace_headers_all[ref_gidx], "igaini", 1) or 1)))
            sf_override = self._estimate_auto_sf(ref_gidx, sr, ref_gain)
            if sf_override > 0.0:
                proc_params.sf = float(sf_override)
        if dn_render_active:
            self._set_denoise_progress(45, 100, "处理链计算")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if dn_perf_diag:
            perf_marks["before_process"] = time.perf_counter()
        try:
            processed = self.processor.process_traces(
                traces=traces_for_processing,
                times=times,
                offsets=process_offsets,
                params=proc_params,
                gains=render_gains,
                sampling_rate=sr,
                realtime_interaction=self._viewport_interacting,
            )
        except Exception as exc:
            # 处理链异常时仍保留上游 mute 结果，避免渲染链回退到原始波形
            processed = traces_for_processing
            self._denoise_backend_stage = f"{self._denoise_backend_stage} | 后处理回退:{type(exc).__name__}"
        if dn_perf_diag:
            perf_marks["after_process"] = time.perf_counter()
        processed_before_denoise = [np.asarray(tr, dtype=np.float64).copy() for tr in processed]
        denoise_use_cache = (
            bool(self._denoise_params.get("enabled", False))
            and bool(self._denoise_run_armed)
        )
        denoise_cache_hit = False
        if denoise_use_cache:
            if dn_render_active:
                self._set_denoise_progress(65, 100, "去噪准备")
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
            cache_key = self._denoise_cache_key_of(
                traces_before_denoise=processed_before_denoise,
                render_trace_indices=np.asarray(idx_render, dtype=int),
                denoise_trace_indices=np.asarray(denoise_indices, dtype=int),
            )
            cache_entry = self._denoise_cache_entries.get(cache_key)
            cache_out = cache_entry.get("output") if isinstance(cache_entry, dict) else None
            if isinstance(cache_out, list) and len(cache_out) == len(processed_before_denoise):
                processed = [np.asarray(tr, dtype=np.float64).copy() for tr in cache_out]
                self._denoise_backend_stage = str(cache_entry.get("stage", self._denoise_backend_stage))
                self._denoise_last_applied_count = int(cache_entry.get("applied_count", 0))
                self._denoise_last_delta_mean_abs = float(cache_entry.get("delta_mean_abs", 0.0))
                self._denoise_last_delta_max_abs = float(cache_entry.get("delta_max_abs", 0.0))
                denoise_cache_hit = True
                # LRU: 命中后提升为最新
                self._denoise_cache_entries.pop(cache_key, None)
                self._denoise_cache_entries[cache_key] = cache_entry
                self._debug_log("DENOISE_CACHE", f"hit size={len(self._denoise_cache_entries)}")
            else:
                self._debug_log("DENOISE_CACHE", f"miss size={len(self._denoise_cache_entries)}")
        # 交互期间仅复用缓存，不做新的去噪重算，避免缩放/拖拽卡顿。
        if self._viewport_interacting and denoise_use_cache and (not denoise_cache_hit):
            self._denoise_backend_stage = "交互中: 缓存未命中，暂停去噪重算"
            self._debug_log("DENOISE_CACHE", "interacting-miss: skip recompute")
        elif not denoise_cache_hit:
            # 去噪放在显示链后段：对“当前操作结果（含已启用 mute/滤波/增益）”执行。
            if dn_render_active:
                self._set_denoise_progress(0, 0, "去噪计算")
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
            if dn_perf_diag:
                perf_marks["before_denoise"] = time.perf_counter()
            processed = self._apply_denoise_to_render_traces(
                traces_in=processed,
                times=np.asarray(times, dtype=float),
                render_trace_indices=np.asarray(idx_render, dtype=int),
                denoise_trace_indices=np.asarray(denoise_indices, dtype=int),
            )
            if dn_perf_diag:
                perf_marks["after_denoise"] = time.perf_counter()
            if denoise_use_cache:
                cache_key = self._denoise_cache_key_of(
                    traces_before_denoise=processed_before_denoise,
                    render_trace_indices=np.asarray(idx_render, dtype=int),
                    denoise_trace_indices=np.asarray(denoise_indices, dtype=int),
                )
                self._denoise_cache_entries[cache_key] = {
                    "output": [np.asarray(tr, dtype=np.float64).copy() for tr in processed],
                    "stage": str(self._denoise_backend_stage),
                    "applied_count": int(self._denoise_last_applied_count),
                    "delta_mean_abs": float(self._denoise_last_delta_mean_abs),
                    "delta_max_abs": float(self._denoise_last_delta_max_abs),
                }
                self._denoise_cache_entries.move_to_end(cache_key)
                while len(self._denoise_cache_entries) > int(self._denoise_cache_limit):
                    self._denoise_cache_entries.popitem(last=False)
                self._debug_log("DENOISE_CACHE", f"store size={len(self._denoise_cache_entries)}")
        if dn_render_active:
            self._set_denoise_progress(90, 100, "结果整理")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if (
            bool(self._denoise_params.get("show_diff", False))
            and int(getattr(self, "_denoise_last_applied_count", 0)) > 0
            and len(processed_before_denoise) == len(processed)
        ):
            active_set = set(int(i) for i in np.asarray(denoise_indices, dtype=int).tolist())
            diff_gain = float(self._denoise_params.get("diff_gain", 1.0))
            if (not np.isfinite(diff_gain)) or diff_gain <= 0.0:
                diff_gain = 1.0
            diff_traces: List[np.ndarray] = []
            for i, tr_now in enumerate(processed):
                base = np.asarray(processed_before_denoise[i], dtype=np.float64)
                now = np.asarray(tr_now, dtype=np.float64)
                if base.shape != now.shape:
                    now = np.resize(now, base.shape)
                gidx = int(idx_render[i]) if i < len(idx_render) else -1
                if gidx in active_set:
                    diff_traces.append((now - base) * diff_gain)
                else:
                    # 差值模式下隐藏未参与去噪的道，避免“零线看起来像原波形未变化”
                    diff_traces.append(np.full_like(base, np.nan))
            processed = diff_traces
            self._denoise_backend_stage = f"{self._denoise_backend_stage} | 差值显示(仅目标道)"
            self._debug_log(
                "DENOISE_DIFF",
                f"enabled=1 gain={diff_gain:g} applied={int(self._denoise_last_applied_count)} active={len(active_set)} render={len(diff_traces)}",
            )
        elif bool(self._denoise_params.get("ab_raw", False)):
            # A/B 仅控制显示：A=原始(当前显示链去噪前)，不参与“是否去噪计算”的决策
            processed = [np.asarray(tr, dtype=np.float64).copy() for tr in processed_before_denoise]
            self._denoise_backend_stage = "A/B显示: 原始(A)"
            self._debug_log("DENOISE_AB", "display=A(raw)")
        self._update_denoise_hint()

        # 时间抽样：用户 ndecim + 动态 LOD
        step_user = max(1, int(self.params.ndecim))
        step_lod = 1
        if times.size > 5000:
            step_lod = 2 if self._viewport_interacting else 1
        step = max(step_user, step_lod)
        t_plot = times[::step]

        # 显示缩放（Fortran 贴近）：不对每帧再按全局振幅归一，避免抵消增益参数效果
        if render_offsets.size > 1:
            spacing = float(np.median(np.diff(np.sort(render_offsets))))
            if spacing <= 0:
                spacing = 1.0
        else:
            spacing = 1.0
        scale = 0.45 * spacing * float(self.spin_dscale.value())

        # 仅用于叠加道显示幅度估计，不参与主道二次归一
        max_amp = 0.0
        for tr in processed:
            if len(tr) == 0:
                continue
            local = float(np.percentile(np.abs(tr), 98))
            if local > max_amp:
                max_amp = local
        if max_amp <= 1e-12:
            max_amp = 1.0

        self._clear_stack_item()
        use_density = self._display_mode_is_density()
        if use_density:
            self._render_density_image(
                processed=processed,
                render_offsets=np.asarray(render_offsets, dtype=float),
                idx_render=np.asarray(idx_render, dtype=int),
                t_plot=np.asarray(t_plot, dtype=float),
                times=np.asarray(times, dtype=float),
                step=int(step),
                scale_dscale=float(self.spin_dscale.value()),
            )
        else:
            self._clear_density_item()
            self._ensure_curve_pool(len(processed))
            self._clear_shade_item()
            shade_segment_count = 0
            can_render_shade = (
                self.params.ishade != 0
                and self.shade_kernel is not None
                and (not self._viewport_interacting or self.chk_rt_shade.isChecked())
            )
            shade_x_parts: List[np.ndarray] = []
            shade_y_parts: List[np.ndarray] = []
            fill_positive = self.params.ishade > 0
            mute_polygon_arr: Optional[np.ndarray] = None
            if self._mute_enabled and len(self._mute_polygon_points) >= 3:
                mute_polygon_arr = np.asarray(self._mute_polygon_points, dtype=float)
            row_step = 1
            if self._viewport_interacting:
                row_step = 3
            elif len(idx_render) > 1200:
                row_step = 2
            for i, tr in enumerate(processed):
                trd = np.asarray(tr)[::step]
                tshift = self._compute_display_tshift(int(idx_render[i]), float(render_offsets[i]))
                t_trace = t_plot + tshift
                if mute_polygon_arr is not None and trd.size == t_trace.size:
                    keep_mask = self._build_mute_inside_mask(
                        float(render_offsets[i]),
                        np.asarray(t_trace, dtype=float),
                        mute_polygon_arr,
                    )
                    if self._mute_invert:
                        keep_mask = ~keep_mask
                    # 对保留区做一次可见域增益归一，避免“裁剪后仍沿用全道增益”导致显示发灰。
                    if int(proc_params.iscale) in (0, 2):
                        valid_gain = keep_mask & np.isfinite(trd)
                        if np.any(valid_gain):
                            local_max = float(np.max(np.abs(trd[valid_gain])))
                            if local_max > 1e-12:
                                trd = np.asarray(trd, dtype=float).copy()
                                trd[valid_gain] = trd[valid_gain] * (float(proc_params.amp) / local_max)
                                if float(proc_params.clip) > 0.0:
                                    c = float(proc_params.clip)
                                    trd[valid_gain] = np.clip(trd[valid_gain], -c, c)
                    trd = np.asarray(trd, dtype=float).copy()
                    trd[~keep_mask] = np.nan
                xw = render_offsets[i] + trd * scale
                if self._map_link_trace_idx is not None and int(idx_render[i]) == int(self._map_link_trace_idx):
                    self._curve_items[i].setPen(pg.mkPen("#f59e0b", width=2.0))
                elif int(idx_render[i]) in self._denoise_selected_traces:
                    self._curve_items[i].setPen(pg.mkPen("#8b5cf6", width=1.5))
                else:
                    self._curve_items[i].setPen(pg.mkPen(self._theme_color("wave_pen", "#0a0a0a"), width=1))
                self._curve_items[i].setData(xw, t_trace, connect="finite")
                if can_render_shade and xw.size > 0:
                    try:
                        segs = self.shade_kernel.build_segments(
                            x_values=np.asarray(xw, dtype=np.float64),
                            t_values=np.asarray(t_trace, dtype=np.float64),
                            baseline_x=float(render_offsets[i]),
                            fill_positive=fill_positive,
                            row_step=row_step,
                        )
                        if segs.shape[0] > 0:
                            shade_x_parts.append(segs[:, :, 0].reshape(-1))
                            shade_y_parts.append(segs[:, :, 1].reshape(-1))
                            shade_segment_count += int(segs.shape[0])
                    except Exception:
                        # 阴影失败不影响主波形渲染
                        pass

            if can_render_shade and shade_segment_count > 0:
                self._ensure_shade_item()
                try:
                    sx = np.concatenate(shade_x_parts)
                    sy = np.concatenate(shade_y_parts)
                    self._shade_item.setData(sx, sy, connect="pairs")
                except Exception:
                    self._clear_shade_item()

        # V 选波窗口高亮：显示全部 apick 的段，按拾取字分色（当前字更粗）
        all_wave_selections = list(getattr(self, "waveform_selections", []) or [])
        active_apick = int(self.spin_apick.value())
        if all_wave_selections:
            render_map = {int(gidx): i for i, gidx in enumerate(np.asarray(idx_render, dtype=int))}
            segs_by_word: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
            marker_spots = []
            for sel in all_wave_selections:
                gidx = int(sel.get("trace_idx", -1))
                row = render_map.get(gidx)
                if row is None:
                    continue
                pw = int(sel.get("pick_word", active_apick))
                color = self._waveop_apick_display_color(pw)
                is_active = pw == active_apick
                t_center = float(sel.get("t_display", 0.0))
                x0 = float(render_offsets[row])
                trd = np.asarray(processed[row])[::step]
                if trd.size != t_plot.size:
                    continue
                tshift = self._compute_display_tshift(gidx, x0)
                t_trace = t_plot + tshift
                # Density：用道中心竖线段标窗，避免叠加 wiggle 横向摆动
                if use_density:
                    x_trace = np.full(trd.shape, x0, dtype=float)
                else:
                    x_trace = x0 + trd * scale
                mask = (t_trace >= (t_center - 0.3)) & (t_trace <= (t_center + 0.7))
                if np.any(mask):
                    seg_x = np.asarray(x_trace[mask], dtype=float)
                    seg_y = np.asarray(t_trace[mask], dtype=float)
                    bucket = segs_by_word.setdefault(pw, ([], []))
                    bucket[0].append(np.concatenate([seg_x, np.asarray([np.nan])]))
                    bucket[1].append(np.concatenate([seg_y, np.asarray([np.nan])]))
                marker_spots.append(
                    {
                        "pos": (x0, t_center),
                        "brush": pg.mkBrush(color),
                        "pen": pg.mkPen("#ffffff" if is_active else "#505050", width=1.2 if is_active else 0.8),
                        "size": 9.0 if is_active else 7.0,
                    }
                )
            # 先清空未画出的字，再按字写入
            drawn_words = set(int(k) for k in segs_by_word.keys())
            for pw, item in list(self._wave_select_items.items()):
                if int(pw) not in drawn_words:
                    try:
                        item.setData([], [])
                    except Exception:
                        pass
            for pw, (xs, ys) in segs_by_word.items():
                item = self._ensure_wave_select_item_for(pw)
                width = 2.2 if int(pw) == active_apick else 1.5
                item.setPen(pg.mkPen(self._waveop_apick_display_color(pw), width=width))
                item.setData(np.concatenate(xs), np.concatenate(ys), connect="finite")
            if not segs_by_word:
                self._clear_wave_select_item()
            if marker_spots:
                self._ensure_wave_select_marker_item()
                self._wave_select_marker_item.setData(spots=marker_spots)
            else:
                self._clear_wave_select_marker_item()
        else:
            self._clear_wave_select_item()
            self._clear_wave_select_marker_item()

        # Density 下无 wiggle 笔色时，用竖线标出选中道（去噪选道 / 位置图联动）
        self._update_density_trace_highlights(
            render_offsets=np.asarray(render_offsets, dtype=float),
            idx_render=np.asarray(idx_render, dtype=int),
            enabled=bool(use_density),
        )

        pick_count = self._render_picks(offsets_all, allowed_trace_indices=idx)
        # Mute 多边形始终叠在密度图之上
        if self._mute_polygon_points:
            self._refresh_mute_polygon_overlay(update_labels=False)
        # 静校正曲线：当前拾取字校正后时间（虚线预览）
        if self.static_corrector.has_corrections() and self.pick_manager is not None:
            apick = int(self.spin_apick.value())
            by_word = self.pick_manager.get_picks_by_word(apick)
            px: List[float] = []
            py: List[float] = []
            for gidx in idx:
                ig = int(gidx)
                tpk = by_word.get(ig)
                if tpk is None or float(tpk) <= 0:
                    continue
                if ig < 0 or ig >= offsets_all.size:
                    continue
                corr = float(self.static_corrector.get_correction(ig))
                x_val = float(offsets_all[ig])
                tshift_base = float(self._alignment_offsets.get(ig, 0.0)) + self._compute_reduction_tshift(ig, x_val)
                px.append(float(offsets_all[ig]))
                py.append(float(tpk) + corr + tshift_base)
            if len(px) >= 2:
                order = np.argsort(np.asarray(px, dtype=float))
                xarr = np.asarray(px, dtype=float)[order]
                yarr = np.asarray(py, dtype=float)[order]
                self._ensure_static_preview_item()
                self._static_preview_item.setData(xarr, yarr)
            else:
                self._clear_static_preview_item()
        else:
            self._clear_static_preview_item()
        theory_points = 0
        txin_points = 0
        txin_preview_points = 0
        water_points = 0
        if self.show_theoretical_times and self.theoretical_traveltime_calculator is not None:
            try:
                th_data = self.theoretical_traveltime_calculator.get_theoretical_times(xcoords)
                if th_data is not None and "distance" in th_data and "time" in th_data:
                    tx = np.asarray(th_data["distance"], dtype=float)
                    tt = np.asarray(th_data["time"], dtype=float)
                    t_reduction = np.asarray(
                        [self._compute_reduction_tshift(-1, float(xv)) for xv in tx],
                        dtype=float,
                    )
                    tt_disp = tt + t_reduction
                    self.theoretical_times_data = {"distances": tx, "times": tt}
                    self._ensure_theoretical_item()
                    self._theoretical_item.setData(tx, tt_disp)
                    theory_points = int(tx.size)
                    if self.show_water_layer_correction:
                        avg_corr = 0.0
                        corr_interp = None
                        if self.water_layer_corrected_times:
                            if "avg_correction" in self.water_layer_corrected_times:
                                arr = np.asarray(self.water_layer_corrected_times["avg_correction"], dtype=float)
                                if arr.size > 0:
                                    avg_corr = float(arr[0])
                            dmap = np.asarray(self.water_layer_corrected_times.get("distances", []), dtype=float)
                            cmap = np.asarray(self.water_layer_corrected_times.get("corrections", []), dtype=float)
                            if dmap.size >= 2 and cmap.size == dmap.size:
                                # 按距离逐点插值校正；超出范围使用边界值
                                corr_interp = np.interp(tx, dmap, cmap, left=float(cmap[0]), right=float(cmap[-1]))
                            elif dmap.size == 1 and cmap.size == 1:
                                corr_interp = np.full_like(tx, float(cmap[0]), dtype=float)
                        if corr_interp is None:
                            corr_interp = np.full_like(tx, avg_corr, dtype=float)
                        wt = (tt - corr_interp) + t_reduction
                        self._ensure_water_corr_item()
                        self._water_corr_item.setData(tx, wt)
                        water_points = int(tx.size)
                    else:
                        self._clear_water_corr_item()
                else:
                    self._clear_theoretical_item()
                    self._clear_water_corr_item()
            except Exception:
                self._clear_theoretical_item()
                self._clear_water_corr_item()
        else:
            self._clear_theoretical_item()
            self._clear_water_corr_item()

        if self.show_txin_overlay and self.txin_overlay_data is not None:
            try:
                tx_off = np.asarray(self.txin_overlay_data.get("offsets", []), dtype=float)
                tx_t = np.asarray(self.txin_overlay_data.get("times", []), dtype=float)
                tx_pw = np.asarray(self.txin_overlay_data.get("pick_words", []), dtype=int)
                if tx_off.size > 0 and tx_t.size == tx_off.size and tx_pw.size == tx_off.size:
                    tx_red = np.asarray(
                        [self._compute_reduction_tshift(-1, float(xv)) for xv in tx_off],
                        dtype=float,
                    )
                    tx_disp = tx_t + tx_red
                    self._ensure_txin_item()
                    spots = []
                    for i in range(int(tx_off.size)):
                        pw = int(tx_pw[i])
                        color = pg.intColor(max(1, pw), hues=48, values=1, alpha=220)
                        spots.append(
                            {
                                "pos": (float(tx_off[i]), float(tx_disp[i])),
                                "brush": pg.mkBrush(color),
                                "pen": pg.mkPen(self._theme_color("pick_active_edge", "#ffffff"), width=0.8),
                                "size": 7.0,
                                "data": {"pick_word": pw},
                            }
                        )
                    self._txin_item.setData(spots=spots)
                    txin_points = int(tx_off.size)
                else:
                    self._clear_txin_item()
            except Exception:
                self._clear_txin_item()
        else:
            self._clear_txin_item()

        if self.txin_map_preview_data is not None:
            try:
                p_idx = np.asarray(self.txin_map_preview_data.get("trace_indices", []), dtype=int)
                p_t = np.asarray(self.txin_map_preview_data.get("times", []), dtype=float)
                p_pw = np.asarray(self.txin_map_preview_data.get("pick_words", []), dtype=int)
                if p_idx.size > 0 and p_t.size == p_idx.size and p_pw.size == p_idx.size:
                    spots = []
                    for i in range(int(p_idx.size)):
                        gidx = int(p_idx[i])
                        if gidx < 0 or gidx >= offsets_all.size:
                            continue
                        x = float(offsets_all[gidx])
                        y = float(p_t[i]) + self._compute_display_tshift(gidx, x)
                        color = pg.intColor(max(1, int(p_pw[i])), hues=48, values=1, alpha=220)
                        spots.append(
                            {
                                "pos": (x, y),
                                "brush": pg.mkBrush(0, 0, 0, 0),
                                "pen": pg.mkPen(color, width=1.4),
                                "symbol": "x",
                                "size": 10.0,
                                "data": {"pick_word": int(p_pw[i])},
                            }
                        )
                    if spots:
                        self._ensure_txin_map_preview_item()
                        self._txin_map_preview_item.setData(spots=spots)
                        txin_preview_points = len(spots)
                    else:
                        self._clear_txin_map_preview_item()
                else:
                    self._clear_txin_map_preview_item()
            except Exception:
                self._clear_txin_map_preview_item()
        else:
            self._clear_txin_map_preview_item()

        # 波形操作叠加：显示在当前视图中心右侧，避免遮挡主剖面
        if self.waveop_stack_result is not None:
            try:
                tau = np.asarray(self.waveop_stack_result.get("tau", []), dtype=float)
                stack = np.asarray(self.waveop_stack_result.get("stack", []), dtype=float)
                centers = np.asarray(self.waveop_stack_result.get("centers", []), dtype=float)
                if tau.size >= 8 and stack.size == tau.size:
                    x_view, y_view = self.plot.getViewBox().viewRange()
                    x_center = 0.5 * float(x_view[0] + x_view[1])
                    y_anchor = float(np.median(centers)) if centers.size > 0 else 0.5 * float(y_view[0] + y_view[1])
                    x_span = max(1e-6, float(abs(x_view[1] - x_view[0])))
                    x_anchor = x_center + 0.30 * x_span
                    x_max = float(max(x_view[0], x_view[1]))
                    x_anchor = min(x_anchor, x_max - 0.04 * x_span)
                    amp = float(np.percentile(np.abs(stack), 98)) if stack.size > 0 else 0.0
                    if amp <= 1e-12:
                        amp = 1.0
                    x_scale = 0.10 * x_span / amp
                    x_stack = x_anchor + stack * x_scale
                    y_stack = y_anchor + tau
                    self._ensure_waveop_stack_item()
                    self._waveop_stack_item.setData(x_stack, y_stack)
                else:
                    self._clear_waveop_stack_item()
            except Exception:
                self._clear_waveop_stack_item()
        else:
            self._clear_waveop_stack_item()

        # 叠加显示：当前活动拾取字，按“拾取时刻 ±0.5s”窗口对齐叠加
        stack_auto_flag = 0
        stack_seed_count = 0
        if self.chk_show_stack.isChecked() and len(processed) >= 2:
            try:
                apick = int(self.spin_apick.value())
                by_word = self.pick_manager.get_picks_by_word(apick) if self.pick_manager is not None else {}
                if by_word:
                    dt_plot = float(t_plot[1] - t_plot[0]) if t_plot.size > 1 else 0.001
                    half_window_sec = 0.5
                    tau = np.arange(
                        -half_window_sec,
                        half_window_sec + 0.5 * dt_plot,
                        dt_plot,
                        dtype=np.float64,
                    )
                    if tau.size >= 8:
                        stack_acc = np.zeros_like(tau, dtype=np.float64)
                        n_stack = 0
                        idx_render_arr = np.asarray(idx_render, dtype=int)
                        for i, tr in enumerate(processed):
                            gidx = int(idx_render_arr[i])
                            tpk = by_word.get(gidx)
                            if tpk is None or float(tpk) <= 0:
                                continue
                            trd = np.asarray(tr)[::step]
                            if trd.size != t_plot.size:
                                continue
                            # 叠加使用“当前显示状态”的时间基准（包含静校正），不依赖 A/F 的波形平移状态
                            pick_eff = float(tpk)
                            t_trace = t_plot
                            if self.static_correction_enabled:
                                corr = float(self.static_corrector.get_correction(gidx))
                                pick_eff += corr
                                t_trace = t_plot + corr
                            seg = np.interp(pick_eff + tau, t_trace, trd, left=0.0, right=0.0)
                            stack_acc += seg
                            n_stack += 1

                        if n_stack > 0:
                            stack_seed_count = int(n_stack)
                            stack_auto_flag = 1
                            stack = stack_acc / float(n_stack)

                            # 叠加道始终显示在当前画面中心右侧
                            x_view, y_view = self.plot.getViewBox().viewRange()
                            x_center = 0.5 * float(x_view[0] + x_view[1])
                            y_center = 0.5 * float(y_view[0] + y_view[1])
                            x_span = max(1e-6, float(abs(x_view[1] - x_view[0])))
                            x_anchor = x_center + 0.30 * x_span
                            x_max = float(max(x_view[0], x_view[1]))
                            x_anchor = min(x_anchor, x_max - 0.04 * x_span)

                            stack_amp = float(np.percentile(np.abs(stack), 98)) if stack.size > 0 else 0.0
                            if stack_amp <= 1e-12:
                                stack_amp = 1.0
                            x_scale = 0.10 * x_span / stack_amp
                            x_stack = x_anchor + stack * x_scale
                            y_stack = y_center + tau

                            self._ensure_stack_item()
                            self._stack_item.setData(x_stack, y_stack)
            except Exception:
                self._clear_stack_item()

        # 首次渲染时拟合一次；之后范围由用户窗口控制（xmin/xmax/tmin/tmax）
        if not self._viewport_interacting and (not self._did_initial_view_fit):
            xmin, xmax = float(np.min(offsets_all)), float(np.max(offsets_all))
            ymin, ymax = float(np.min(times)), float(np.max(times))
            self.plot.setXRange(xmin, xmax, padding=0.02)
            self.plot.setYRange(ymax, ymin, padding=0.02)  # 反转Y已开启，传入大->小
            vb = self.plot.getViewBox()
            vb.enableAutoRange(axis=vb.XAxis, enable=False)
            vb.enableAutoRange(axis=vb.YAxis, enable=False)
            self._did_initial_view_fit = True

        header = self.loaded.get("header")
        vredf = float(getattr(header, "vredf", 0.0) or 0.0) if header is not None else 0.0
        vred = float(self.params.vred)
        rvred = (1.0 / vred) if vred > 0 else 0.0
        rvredf = (1.0 / vredf) if vredf > 0 else 0.0
        d_rvred = rvred - rvredf

        _elapsed = (time.perf_counter() - t0) * 1000.0
        if dn_perf_diag:
            t1 = time.perf_counter()
            ext_ms = (perf_marks.get("after_extract", perf_marks["t0"]) - perf_marks.get("before_extract", perf_marks["t0"])) * 1000.0
            vis_ms = (perf_marks.get("after_visible", perf_marks["t0"]) - perf_marks.get("before_visible", perf_marks["t0"])) * 1000.0
            proc_ms = (perf_marks.get("after_process", perf_marks["t0"]) - perf_marks.get("before_process", perf_marks["t0"])) * 1000.0
            den_ms = (perf_marks.get("after_denoise", perf_marks["t0"]) - perf_marks.get("before_denoise", perf_marks["t0"])) * 1000.0
            self._debug_log(
                "DENOISE_PERF",
                f"render_now total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f} extract_ms={ext_ms:.1f} "
                f"visible_ms={vis_ms:.1f} process_ms={proc_ms:.1f} denoise_ms={den_ms:.1f} "
                f"scope={int(getattr(self, '_last_denoise_scope_count', 0))} render={len(idx_render)}",
            )
        if dn_render_active:
            self._set_denoise_progress(0, -1)

    def _visible_mask(self, xcoords: np.ndarray) -> np.ndarray:
        if xcoords.size == 0:
            return np.zeros((0,), dtype=bool)
        x_range = self.plot.getViewBox().viewRange()[0]
        xmin, xmax = float(min(x_range)), float(max(x_range))
        # 加一点边缘缓冲，避免拖动时闪烁
        pad = max(1e-6, 0.02 * (xmax - xmin))
        return (xcoords >= xmin - pad) & (xcoords <= xmax + pad)


    def _ensure_curve_pool(self, n: int) -> None:
        while len(self._curve_items) < n:
            item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("wave_pen", "#0a0a0a"), width=1)
            )
            self.plot.addItem(item)
            self._curve_items.append(item)
        for i in range(n, len(self._curve_items)):
            self._curve_items[i].setData([], [])


    def _ensure_shade_item(self) -> None:
        if self._shade_item is None:
            self._shade_item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("shade_pen", "#30343b"), width=1),
                connect="pairs",
            )
            self.plot.addItem(self._shade_item)


    def _clear_shade_item(self) -> None:
        if self._shade_item is not None:
            self._shade_item.setData([], [])

    @staticmethod

    def _gray_lut() -> np.ndarray:
        x = np.linspace(0, 255, 256, dtype=np.uint8)
        return np.column_stack([x, x, x])


    def _ensure_density_item(self) -> None:
        if self._density_item is None:
            # row-major: 数组 shape=(nt, ntr) 与 setRect(x,t,w,h) 一致（同 obs_rtm）
            try:
                self._density_item = pg.ImageItem(axisOrder="row-major")
            except TypeError:
                self._density_item = pg.ImageItem()
            self._density_item.setLookupTable(self._gray_lut())
            # 置于底层且不抢鼠标，保证拾取 / Mute / 缩放平移不受影响
            self._density_item.setZValue(-5)
            try:
                self._density_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            except Exception:
                pass
            self.plot.addItem(self._density_item)


    def _clear_density_item(self) -> None:
        if self._density_item is not None:
            try:
                self._density_item.clear()
            except Exception:
                self._density_item.setImage(np.zeros((1, 1), dtype=np.float32))
        self._clear_density_trace_highlights()


    def _ensure_density_hl_item(self) -> None:
        if self._density_hl_item is None:
            self._density_hl_item = pg.PlotDataItem(
                pen=pg.mkPen("#f59e0b", width=1.6),
                connect="pairs",
            )
            self._density_hl_item.setZValue(25)
            try:
                self._density_hl_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            except Exception:
                pass
            self.plot.addItem(self._density_hl_item)


    def _clear_density_trace_highlights(self) -> None:
        if self._density_hl_item is not None:
            self._density_hl_item.setData([], [])


    def _update_density_trace_highlights(
        self,
        *,
        render_offsets: np.ndarray,
        idx_render: np.ndarray,
        enabled: bool,
    ) -> None:
        """Density 模式：用竖线标去噪选中道 / 位置图联动道（不依赖 wiggle 笔色）。"""
        if not enabled:
            self._clear_density_trace_highlights()
            return
        xs: List[float] = []
        try:
            _, y_range = self.plot.getViewBox().viewRange()
            y0, y1 = float(y_range[0]), float(y_range[1])
        except Exception:
            y0, y1 = 0.0, 1.0
        want = set(int(i) for i in self._denoise_selected_traces)
        if self._map_link_trace_idx is not None:
            want.add(int(self._map_link_trace_idx))
        if not want:
            self._clear_density_trace_highlights()
            return
        for i, gidx in enumerate(np.asarray(idx_render, dtype=int)):
            if int(gidx) not in want:
                continue
            if i >= render_offsets.size:
                continue
            x = float(render_offsets[i])
            xs.extend([x, x])
        if not xs:
            self._clear_density_trace_highlights()
            return
        ys = np.tile(np.asarray([y0, y1], dtype=float), len(xs) // 2)
        self._ensure_density_hl_item()
        assert self._density_hl_item is not None
        self._density_hl_item.setData(np.asarray(xs, dtype=float), ys, connect="pairs")


    def _display_mode_is_density(self) -> bool:
        return int(self.combo_mode.currentIndex()) == 3

    @staticmethod

    def _density_raster_dx(xs: np.ndarray) -> float:
        """估计均匀栅格 dx：用相邻道间距的中位数，忽略大空隙。"""
        xs = np.asarray(xs, dtype=float)
        if xs.size <= 1:
            return 1.0
        d = np.diff(xs)
        d = d[np.isfinite(d) & (d > 1e-12)]
        if d.size == 0:
            return 1.0
        # 大空隙不参与中位数，避免把整段空白当成“道宽”
        med = float(np.median(d))
        small = d[d <= max(med * 3.0, 1e-9)]
        if small.size:
            return float(np.median(small))
        return med


    def _render_density_image(
        self,
        processed: List[np.ndarray],
        render_offsets: np.ndarray,
        idx_render: np.ndarray,
        t_plot: np.ndarray,
        times: np.ndarray,
        step: int,
        scale_dscale: float,
    ) -> None:
        """
        变密度：按真实 offset 栅格化到均匀网格。

        不可把 n 道直接均分铺满 [xmin,xmax]——空隙段会被邻道横向拉宽，
        看起来像“空白偏移距出现了别的道的波形”。
        """
        ntr = len(processed)
        if ntr == 0 or t_plot.size == 0:
            self._clear_density_item()
            return
        nt = int(t_plot.size)
        order = np.argsort(np.asarray(render_offsets, dtype=float))
        xs = np.asarray(render_offsets, dtype=float)[order]
        dt = float(times[step] - times[0]) if times.size > step else (
            float(times[1] - times[0]) if times.size > 1 else 0.004
        )
        if not np.isfinite(dt) or abs(dt) < 1e-12:
            dt = 0.004
        o1 = float(t_plot[0]) if t_plot.size else float(times[0]) if times.size else 0.0

        cols = np.zeros((nt, ntr), dtype=np.float32)
        for k, src_i in enumerate(order):
            ii = int(src_i)
            trd = np.asarray(processed[ii], dtype=np.float64)[::step]
            if trd.size < nt:
                buf = np.zeros(nt, dtype=np.float64)
                buf[: trd.size] = trd
                trd = buf
            elif trd.size > nt:
                trd = trd[:nt]
            tshift = self._compute_display_tshift(int(idx_render[ii]), float(render_offsets[ii]))
            sh = int(round(-float(tshift) / dt)) if abs(dt) > 1e-12 else 0
            col = np.nan_to_num(trd.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if sh:
                col = np.roll(col, -sh)
                if sh > 0:
                    col[-sh:] = 0
                else:
                    col[:-sh] = 0
            cols[:, k] = col

        dx_med = self._density_raster_dx(xs)
        if ntr == 1:
            x0 = float(xs[0]) - 0.5 * dx_med
            x1 = float(xs[0]) + 0.5 * dx_med
            gather = cols
        else:
            # ImageItem 列 j 对应 [x0+j*dx, x0+(j+1)*dx]，空隙 bin 保持 0
            x0 = float(xs[0]) - 0.5 * dx_med
            x1 = float(xs[-1]) + 0.5 * dx_med
            span = max(x1 - x0, dx_med)
            nx = int(min(max(int(np.ceil(span / dx_med)), ntr), 8000))
            dx = span / float(nx)
            gather = np.zeros((nt, nx), dtype=np.float32)
            # 同 bin 多道时取能量更大者，避免抽稀后丢强能量
            bin_amp = np.zeros(nx, dtype=np.float32)
            for k in range(ntr):
                j = int(np.floor((float(xs[k]) - x0) / dx))
                j = int(min(max(j, 0), nx - 1))
                amp = float(np.max(np.abs(cols[:, k])))
                if amp >= float(bin_amp[j]):
                    gather[:, j] = cols[:, k]
                    bin_amp[j] = amp

        # 色标只用有道的列，避免空白列把百分位拉歪
        occupied = np.any(np.abs(gather) > 0, axis=0)
        flat = np.abs(gather[:, occupied]) if np.any(occupied) else np.abs(gather)
        if flat.size > 250_000:
            sample = flat.ravel()[:: max(1, flat.size // 200_000)]
        else:
            sample = flat.ravel()
        clim = float(np.percentile(sample, 98.0)) if sample.size else 1.0
        if clim < 1e-20:
            clim = float(np.max(flat)) if flat.size else 1.0
        if clim < 1e-20:
            clim = 1.0
        clim = clim / max(0.05, float(scale_dscale))

        w = float(x1 - x0)
        h = max(nt, 1) * abs(dt)
        self._ensure_density_item()
        assert self._density_item is not None
        self._density_item.setImage(gather, autoLevels=False)
        self._density_item.setLevels((-clim, clim))
        self._density_item.setRect(QtCore.QRectF(x0, o1, w, h))
        # 隐藏 wiggle / 填充
        self._ensure_curve_pool(0)
        self._clear_shade_item()
