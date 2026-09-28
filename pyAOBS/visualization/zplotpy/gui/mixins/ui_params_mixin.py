# -*- coding: utf-8 -*-
"""UI parameter collect/apply + gain presets mixed into QtFastViewer."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    from ...core.parameters import ZPlotParameters
except ImportError:  # pragma: no cover
    from pyAOBS.visualization.zplotpy.core.parameters import ZPlotParameters


class UiParamsMixin:
    """窗口/处理参数同步、自动比例与增益预设。"""

    def _loaded_times_dt_seconds(self) -> Optional[float]:
        """当前已加载剖面的采样间隔 dt（秒）；无有效时间轴时为 None。"""
        loaded = getattr(self, "loaded", None)
        if not isinstance(loaded, dict):
            return None
        times = np.asarray(loaded.get("times", []), dtype=float)
        if times.size < 2:
            return None
        dt = float(abs(times[1] - times[0]))
        if not np.isfinite(dt) or dt <= 0:
            return None
        return dt


    def _update_y_axis_label(self) -> None:
        vred = float(self.spin_vred.value())
        if vred > 0:
            self.plot.getPlotItem().setLabels(left=f"t-x/{vred:.3g}", bottom="Offset (km)")
        else:
            self.plot.getPlotItem().setLabels(left="Time (s)", bottom="Offset (km)")


    def _apply_window_from_controls(self) -> None:
        """将 xmin/xmax/tmin/tmax 直接应用到当前视图窗口。"""
        if self.loaded is None:
            return
        if self._syncing_window_controls:
            return
        xmin = float(min(self.spin_xmin.value(), self.spin_xmax.value()))
        xmax = float(max(self.spin_xmin.value(), self.spin_xmax.value()))
        tmin = float(min(self.spin_tmin.value(), self.spin_tmax.value()))
        tmax = float(max(self.spin_tmin.value(), self.spin_tmax.value()))
        if xmin < xmax:
            self.plot.setXRange(xmin, xmax, padding=0.0)
        if tmin < tmax:
            self.plot.setYRange(tmax, tmin, padding=0.0)
        self._did_initial_view_fit = True
        self.request_render(delay_ms=10)


    def _sync_window_controls_from_view(self) -> None:
        """将当前视图范围回写到 xmin/xmax/tmin/tmax 控件。"""
        if self.loaded is None:
            return
        vb = self.plot.getViewBox()
        xr, yr = vb.viewRange()
        xmin, xmax = float(min(xr)), float(max(xr))
        tmin, tmax = float(min(yr)), float(max(yr))
        self._syncing_window_controls = True
        try:
            for w in (self.spin_xmin, self.spin_xmax, self.spin_tmin, self.spin_tmax):
                w.blockSignals(True)
            self.spin_xmin.setValue(xmin)
            self.spin_xmax.setValue(xmax)
            self.spin_tmin.setValue(tmin)
            self.spin_tmax.setValue(tmax)
        finally:
            for w in (self.spin_xmin, self.spin_xmax, self.spin_tmin, self.spin_tmax):
                w.blockSignals(False)
            self._syncing_window_controls = False


    def _estimate_auto_sf(self, trace_idx: int, sampling_rate: Optional[float], trace_gain: float) -> float:
        """为 iscale=1 且 sf=0 估计稳定 sf（避免随视窗跳变）。"""
        if self.loaded is None:
            return 0.0
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if trace_idx < 0 or trace_idx >= len(traces) or trace_idx >= offsets.size:
            return 0.0
        tr = np.asarray(traces[int(trace_idx)], dtype=np.float64).copy()
        if tr.size == 0:
            return 0.0
        try:
            do_rmean = bool(int(getattr(self.params, "rmean", 1) or 0))
            do_rtrend = bool(int(getattr(self.params, "rtrend", 0) or 0))
        except Exception:
            do_rmean = int(getattr(self.params, "iout", 0) or 0) != 2
            do_rtrend = False
        if do_rmean or do_rtrend:
            tr = self.processor.apply_rmean_rtrend(tr, do_rmean=do_rmean, do_rtrend=do_rtrend)
        if int(self.params.ibndps) != 0 and sampling_rate is not None and sampling_rate > 0:
            try:
                tr = self.processor.apply_bandpass_filter(
                    tr,
                    self.params.freqlo,
                    self.params.freqhi,
                    self.params.npoles,
                    self.params.izerop,
                    float(sampling_rate),
                ).astype(np.float64, copy=False)
            except Exception:
                pass
        ampmax_ref = float(np.max(np.abs(tr))) if tr.size > 0 else 0.0
        if ampmax_ref <= 1e-20:
            return 0.0
        off = abs(float(offsets[int(trace_idx)]))
        off_for_pow = max(off * 10.0, 1e-6)
        denom = ampmax_ref * max(1.0, float(trace_gain)) * (off_for_pow ** max(0.0, float(self.params.rcor)))
        if denom <= 1e-20:
            return 0.0
        return float(self.params.amp) / denom


    def _build_processing_params(self) -> ZPlotParameters:
        """按当前开关生成处理参数（滤波/增益可独立关闭）。"""
        proc_params = ZPlotParameters()
        try:
            proc_params.from_dict(self.params.to_dict())
        except Exception:
            # 兜底：保留默认参数
            pass

        if not bool(self.chk_filter.isChecked()):
            proc_params.ibndps = 0

        # 增益总开关：关闭时跳过增益/裁剪。勿把 iscale 置 0——iscale=0 是自动增益。
        proc_params.gain_on = 1 if bool(self.chk_gain.isChecked()) else 0

        # rmean / rtrend：写入处理链；并同步 legacy iout（2=不去直流）
        proc_params.rmean = 1 if bool(self.chk_rmean.isChecked()) else 0
        proc_params.rtrend = 1 if bool(self.chk_rtrend.isChecked()) else 0
        proc_params.iout = 0 if proc_params.rmean else 2

        return proc_params


    def _collect_ui_parameters(self) -> Dict[str, object]:
        return {
            "irec": int(self.spin_irec.value()),
            "itype": int(self.combo_itype.currentIndex()),
            "nskip": int(self.spin_nskip.value()),
            "ndecim": int(self.spin_ndecim.value()),
            "vred": float(self.spin_vred.value()),
            "xmin": float(self.spin_xmin.value()),
            "xmax": float(self.spin_xmax.value()),
            "tmin": float(self.spin_tmin.value()),
            "tmax": float(self.spin_tmax.value()),
            "amp": float(self.spin_amp.value()),
            "iscale": int(self.combo_iscale.currentIndex()),
            "rcor": float(self.spin_rcor.value()),
            "sf": float(self.spin_sf.value()),
            "tvg": float(self.spin_tvg.value()),
            "pvg": float(self.spin_pvg.value()),
            "clip": float(self.spin_clip.value()),
            "dscale": float(self.spin_dscale.value()),
            "filter_on": bool(self.chk_filter.isChecked()),
            "gain_on": bool(self.chk_gain.isChecked()),
            "rmean": bool(self.chk_rmean.isChecked()),
            "rtrend": bool(self.chk_rtrend.isChecked()),
            "freqlo": float(self.spin_freqlo.value()),
            "freqhi": float(self.spin_freqhi.value()),
            "npoles": int(self.spin_npoles.value()),
            "zerop": bool(self.chk_zerop.isChecked()),
            "denoise_enabled": bool(self.chk_denoise_enabled.isChecked()),
            "denoise_ab_raw": (not bool(self.chk_denoise_ab_raw.isChecked())),
            "denoise_show_diff": bool(self.chk_denoise_show_diff.isChecked()),
            "denoise_diff_gain": float(self.combo_denoise_diff_gain.currentData() or 1.0),
            "denoise_scope": str(self.combo_denoise_scope.currentData() or "rendered"),
            "denoise_select_mode": str(self.combo_denoise_scope.currentData() or "rendered") == "selected",
            "denoise_selected_traces": sorted(int(i) for i in self._denoise_selected_traces),
            "denoise_f_s": float(self.spin_denoise_f_s.value()),
            "denoise_f_e": float(self.spin_denoise_f_e.value()),
            "denoise_strength": float(self.spin_denoise_strength.value()),
            "denoise_bwconn": int(self.combo_denoise_bwconn.currentText()),
            "denoise_workers": int(self.spin_denoise_workers.value()),
            "denoise_coh_win": int(self._denoise_params.get("coh_win", 11)),
            "denoise_coh_lag": int(self._denoise_params.get("coh_lag", 2)),
            "denoise_coh_thr": float(self._denoise_params.get("coh_thr", 0.55)),
            "denoise_coh_blend": float(self._denoise_params.get("coh_blend", 0.35)),
            "denoise_coh_penalty": float(self._denoise_params.get("coh_penalty", 0.08)),
            "denoise_perf_diag": bool(self._denoise_params.get("perf_diag", False)),
            "denoise_morph_enable": bool(self._denoise_params.get("morph_enable", True)),
            "denoise_morph_preset": str(self._denoise_params.get("morph_preset", "balanced")),
            "denoise_morph_quantile": float(self._denoise_params.get("morph_quantile", 0.70)),
            "denoise_morph_min_area": int(self._denoise_params.get("morph_min_area", 24)),
            "denoise_morph_expand": int(self._denoise_params.get("morph_expand", 1)),
            "denoise_morph_floor_ratio": float(self._denoise_params.get("morph_floor_ratio", 0.03)),
            "denoise_morph_keep_strong_q": float(self._denoise_params.get("morph_keep_strong_q", 0.95)),
            "denoise_pick_guidance": bool(self.chk_denoise_pick_guidance.isChecked()),
            "denoise_pick_wavelet_length": float(self.spin_denoise_pick_hw.value()),
            "denoise_pick_floor": float(self.spin_denoise_pick_floor.value()),
            "denoise_return_debug": False,
            "denoise_return_result": False,
            "mode": int(self.combo_mode.currentIndex()),
            "rt_shade": bool(self.chk_rt_shade.isChecked()),
            "pick_mode": bool(self.chk_pick_mode.isChecked()),
            "apick": int(self.spin_apick.value()),
            "pick_size": int(self.spin_pick_size.value()),
            "tcrcor": float(self.spin_tcrcor.value()),
            "tlag": float(self.spin_tlag.value()),
            "hilbratio": float(self.spin_hilbratio.value()),
            "show_stack": bool(self.chk_show_stack.isChecked()),
            "orientation_ui_params": dict(self._orientation_ui_params),
            "orientation_current_solution": dict(self._orientation_current_solution),
        }


    def _apply_ui_parameters(self, conf: Dict[str, object]) -> None:
        if "irec" in conf:
            self.spin_irec.setValue(max(0, int(conf["irec"])))
        if "itype" in conf:
            self.combo_itype.setCurrentIndex(max(0, min(4, int(conf["itype"]))))
        if "nskip" in conf:
            self.spin_nskip.setValue(max(0, int(conf["nskip"])))
        if "ndecim" in conf:
            self.spin_ndecim.setValue(max(1, int(conf["ndecim"])))
        if "vred" in conf:
            self.spin_vred.setValue(max(0.0, float(conf["vred"])))
        if "xmin" in conf:
            self.spin_xmin.setValue(float(conf["xmin"]))
        if "xmax" in conf:
            self.spin_xmax.setValue(float(conf["xmax"]))
        if "tmin" in conf:
            self.spin_tmin.setValue(float(conf["tmin"]))
        if "tmax" in conf:
            self.spin_tmax.setValue(float(conf["tmax"]))
        if "amp" in conf:
            self.spin_amp.setValue(float(conf["amp"]))
        if "iscale" in conf:
            self.combo_iscale.setCurrentIndex(max(0, min(2, int(conf["iscale"]))))
        if "rcor" in conf:
            self.spin_rcor.setValue(float(conf["rcor"]))
        if "sf" in conf:
            self.spin_sf.setValue(max(0.0, float(conf["sf"])))
        if "tvg" in conf:
            self.spin_tvg.setValue(float(conf["tvg"]))
        if "pvg" in conf:
            self.spin_pvg.setValue(float(conf["pvg"]))
        if "clip" in conf:
            self.spin_clip.setValue(max(0.0, float(conf["clip"])))
        if "dscale" in conf:
            self.spin_dscale.setValue(max(0.2, min(5.0, float(conf["dscale"]))))
        if "filter_on" in conf:
            self.chk_filter.setChecked(bool(conf["filter_on"]))
        if "gain_on" in conf:
            self.chk_gain.setChecked(bool(conf["gain_on"]))
        if "rmean" in conf:
            self.chk_rmean.setChecked(bool(conf["rmean"]))
        elif "iout" in conf:
            # 兼容旧参数：iout=2 表示不去直流
            self.chk_rmean.setChecked(int(conf.get("iout", 0) or 0) != 2)
        if "rtrend" in conf:
            self.chk_rtrend.setChecked(bool(conf["rtrend"]))
        if "freqlo" in conf:
            self.spin_freqlo.setValue(float(conf["freqlo"]))
        if "freqhi" in conf:
            self.spin_freqhi.setValue(float(conf["freqhi"]))
        if "npoles" in conf:
            self.spin_npoles.setValue(max(1, int(conf["npoles"])))
        if "zerop" in conf:
            self.chk_zerop.setChecked(bool(conf["zerop"]))
        if "denoise_enabled" in conf:
            self.chk_denoise_enabled.setChecked(bool(conf["denoise_enabled"]))
        if "denoise_ab_raw" in conf:
            # 兼容字段语义：denoise_ab_raw=True 表示原始(A)
            self.chk_denoise_ab_raw.setChecked(not bool(conf["denoise_ab_raw"]))
        if "denoise_show_diff" in conf:
            self.chk_denoise_show_diff.setChecked(bool(conf["denoise_show_diff"]))
        if "denoise_diff_gain" in conf:
            try:
                gain_v = float(conf["denoise_diff_gain"])
            except Exception:
                gain_v = 1.0
            idx_gain = self.combo_denoise_diff_gain.findData(gain_v)
            if idx_gain < 0:
                idx_gain = 0
            self.combo_denoise_diff_gain.setCurrentIndex(idx_gain)
        if "denoise_scope" in conf:
            scope = str(conf["denoise_scope"]).strip().lower()
            idx_scope = self.combo_denoise_scope.findData(scope)
            if idx_scope >= 0:
                self.combo_denoise_scope.setCurrentIndex(idx_scope)
        # 兼容旧字段 denoise_select_mode：当前由 scope=selected 决定选道模式，忽略该字段
        if "denoise_selected_traces" in conf:
            val_sel = conf.get("denoise_selected_traces", [])
            if isinstance(val_sel, (list, tuple)):
                self._denoise_selected_traces = {int(v) for v in val_sel if isinstance(v, (int, float))}
        if "denoise_f_s" in conf:
            self.spin_denoise_f_s.setValue(max(0.0, float(conf["denoise_f_s"])))
        if "denoise_f_e" in conf:
            self.spin_denoise_f_e.setValue(max(0.1, float(conf["denoise_f_e"])))
        if "denoise_strength" in conf:
            self.spin_denoise_strength.setValue(max(0.01, float(conf["denoise_strength"])))
        if "denoise_bwconn" in conf:
            bw = int(conf["denoise_bwconn"])
            self.combo_denoise_bwconn.setCurrentText("4" if bw == 4 else "8")
        if "denoise_workers" in conf:
            self.spin_denoise_workers.setValue(max(1, int(conf["denoise_workers"])))
        if "denoise_coh_win" in conf:
            self._denoise_params["coh_win"] = int(max(5, int(conf["denoise_coh_win"])))
        if "denoise_coh_lag" in conf:
            self._denoise_params["coh_lag"] = int(max(0, int(conf["denoise_coh_lag"])))
        if "denoise_coh_thr" in conf:
            self._denoise_params["coh_thr"] = float(min(0.95, max(0.05, float(conf["denoise_coh_thr"]))))
        if "denoise_coh_blend" in conf:
            self._denoise_params["coh_blend"] = float(min(0.90, max(0.0, float(conf["denoise_coh_blend"]))))
        if "denoise_coh_penalty" in conf:
            self._denoise_params["coh_penalty"] = float(min(0.50, max(0.0, float(conf["denoise_coh_penalty"]))))
        if "denoise_perf_diag" in conf:
            self._denoise_params["perf_diag"] = bool(conf["denoise_perf_diag"])
        if "denoise_morph_enable" in conf:
            self._denoise_params["morph_enable"] = bool(conf["denoise_morph_enable"])
        if "denoise_morph_preset" in conf:
            v = str(conf["denoise_morph_preset"]).strip().lower()
            if v in ("conservative", "balanced", "strong"):
                self._denoise_params["morph_preset"] = v
        if "denoise_morph_quantile" in conf:
            self._denoise_params["morph_quantile"] = float(min(0.98, max(0.05, float(conf["denoise_morph_quantile"]))))
        if "denoise_morph_min_area" in conf:
            self._denoise_params["morph_min_area"] = int(max(1, int(conf["denoise_morph_min_area"])))
        if "denoise_morph_expand" in conf:
            self._denoise_params["morph_expand"] = int(min(6, max(0, int(conf["denoise_morph_expand"]))))
        if "denoise_morph_floor_ratio" in conf:
            self._denoise_params["morph_floor_ratio"] = float(min(0.50, max(0.0, float(conf["denoise_morph_floor_ratio"]))))
        if "denoise_morph_keep_strong_q" in conf:
            self._denoise_params["morph_keep_strong_q"] = float(min(0.999, max(0.05, float(conf["denoise_morph_keep_strong_q"]))))
        if "denoise_pick_guidance" in conf:
            self.chk_denoise_pick_guidance.setChecked(bool(conf["denoise_pick_guidance"]))
        if "denoise_pick_wavelet_length" in conf:
            self.spin_denoise_pick_hw.setValue(float(max(0.02, min(5.0, float(conf["denoise_pick_wavelet_length"])))))
        elif "denoise_pick_half_width" in conf:
            sig_old = float(conf["denoise_pick_half_width"])
            twl = float(sig_old) * (2.0 * math.sqrt(2.0 * math.log(2.0)))
            self.spin_denoise_pick_hw.setValue(float(max(0.02, min(5.0, twl))))
        if "denoise_pick_floor" in conf:
            self.spin_denoise_pick_floor.setValue(float(min(0.95, max(0.0, float(conf["denoise_pick_floor"])))))
        # 兼容旧配置字段：保留读取但忽略 debug/result UI（已移除）
        self._sync_denoise_params_from_ui()
        if "mode" in conf:
            self.combo_mode.setCurrentIndex(
                max(0, min(self.combo_mode.count() - 1, int(conf["mode"])))
            )
        if "rt_shade" in conf:
            self.chk_rt_shade.setChecked(bool(conf["rt_shade"]))
        if "pick_mode" in conf:
            self.chk_pick_mode.setChecked(bool(conf["pick_mode"]))
        if "apick" in conf:
            apick = int(conf["apick"])
            apick = max(1, min(int(self.spin_apick.maximum()), apick))
            self.spin_apick.setValue(apick)
        if "pick_size" in conf:
            psize = max(2, min(40, int(conf["pick_size"])))
            self.spin_pick_size.setValue(psize)
        if "tcrcor" in conf:
            self.spin_tcrcor.setValue(float(conf["tcrcor"]))
        if "tlag" in conf:
            self.spin_tlag.setValue(float(conf["tlag"]))
        if "hilbratio" in conf:
            self.spin_hilbratio.setValue(float(conf["hilbratio"]))
        if "show_stack" in conf:
            self.chk_show_stack.setChecked(bool(conf["show_stack"]))
        if "orientation_ui_params" in conf:
            val = conf["orientation_ui_params"]
            if isinstance(val, dict):
                for k in (
                    "wave_pre",
                    "wave_post",
                    "att_iter",
                    "att_wtt",
                    "att_wpol",
                    "att_wsym",
                    "prior_tt_shift_sec",
                    "correct_tilt",
                    "use_rmean",
                    "use_rtrend",
                    "use_bandpass",
                ):
                    if k in val:
                        try:
                            self._orientation_ui_params[k] = float(val[k])
                        except Exception:
                            pass
        if "orientation_current_solution" in conf:
            val = conf["orientation_current_solution"]
            if isinstance(val, dict):
                for k in (
                    "azimuth_deg",
                    "tilt_deg",
                    "dx",
                    "dy",
                    "dz",
                    "prior_tt_shift_sec",
                    "tt_corr_sec",
                    "time_shift_sec",
                    "objective",
                    "accepted",
                ):
                    if k in val:
                        try:
                            self._orientation_current_solution[k] = float(val[k])
                        except Exception:
                            pass
        # V段与叠加校正基准由“波形操作”面板独立保存/加载，不与参数文件混用。

    def _apply_far_offset_boost(self) -> None:
        """一键增强远偏移弱能量可见性（Fortran 风格友好参数）。"""
        # 远偏移增强：固定比例 + 距离补偿为主，避免过度放大噪声
        self.combo_iscale.setCurrentIndex(1)
        self.spin_rcor.setValue(0.8)
        self.spin_amp.setValue(1.6)
        self.spin_tvg.setValue(0.8)
        self.spin_pvg.setValue(1.2)
        self.chk_filter.setChecked(True)
        self.spin_clip.setValue(2.5)
        self.spin_dscale.setValue(1.25)
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已应用远偏移增强预设：iscale=1, rcor=0.8, amp=1.6, tvg=0.8, pvg=1.2, dscale=1.25")


    def _apply_gain_preset_balanced(self) -> None:
        """平衡显示预设：兼顾近偏移与远偏移可见性。"""
        self.combo_iscale.setCurrentIndex(0)
        self.spin_rcor.setValue(0.3)
        self.spin_amp.setValue(1.2)
        self.spin_tvg.setValue(1.0)
        self.spin_pvg.setValue(1.0)
        self.chk_filter.setChecked(True)
        self.spin_clip.setValue(0.0)
        self.spin_dscale.setValue(1.0)
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已应用平衡显示预设：iscale=0, rcor=0.3, amp=1.2, tvg=1.0, pvg=1.0, dscale=1.0")


    def _apply_gain_preset_strong(self) -> None:
        """强增强预设：弱能量优先，允许更高噪声。"""
        self.combo_iscale.setCurrentIndex(1)
        self.spin_rcor.setValue(1.2)
        self.spin_amp.setValue(2.2)
        self.spin_tvg.setValue(0.6)
        self.spin_pvg.setValue(1.5)
        self.chk_filter.setChecked(True)
        self.spin_clip.setValue(2.5)
        self.spin_dscale.setValue(1.6)
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已应用强增强预设：iscale=1, rcor=1.2, amp=2.2, tvg=0.6, pvg=1.5, dscale=1.6")


    def _update_gain_effect_hint(self) -> None:
        iscale = int(self.combo_iscale.currentIndex())
        clip = float(self.spin_clip.value())
        dscale = float(self.spin_dscale.value())
        if iscale == 2:
            txt = "iscale=2: 主要生效 amp/tvg/pvg/clip；sf/rcor 对显示影响较弱"
        elif iscale == 1:
            txt = "iscale=1: 主要生效 sf/rcor/amp/clip（最接近 Fortran 固定比例）"
        else:
            txt = "iscale=0: 主要生效 amp/clip；sf/rcor 不参与主缩放"
        if clip > 0:
            txt += f"；当前 clip={clip:.3g} 已开启裁剪"
        txt += f"；dscale={dscale:.3g}"
        if hasattr(self, "lbl_gain_hint") and self.lbl_gain_hint is not None:
            self.lbl_gain_hint.setText(txt)


    def _on_rmean_rtrend_toggled(self, which: str = "rmean") -> None:
        """勾选 rmean/rtrend 时重绘，并提示为何带通开启时外观常不变。"""
        try:
            self.processor.clear_cache()
        except Exception:
            pass
        on = bool(self.chk_rmean.isChecked()) if which == "rmean" else bool(self.chk_rtrend.isChecked())
        filt_on = bool(self.chk_filter.isChecked())
        flo = float(self.spin_freqlo.value()) if hasattr(self, "spin_freqlo") else 0.0
        name = "rmean" if which == "rmean" else "rtrend"
        if filt_on and flo > 1e-9:
            self._set_status_text(
                f"{name}={'ON' if on else 'OFF'}（处理链已生效）。"
                f"当前带通 fL={flo:g}Hz 已去掉直流/缓变，波形外观通常几乎不变；"
                f"请先取消勾选「滤波」再对比。",
                hold_ms=4200,
            )
        else:
            self._set_status_text(
                f"{name}={'ON' if on else 'OFF'}："
                f"{'去平均' if which == 'rmean' else '去趋势'}已"
                f"{'开启' if on else '关闭'}（滤波前；自动增益下若直流很小也可能不明显）",
                hold_ms=2800,
            )
        self.request_render(delay_ms=10)
