# -*- coding: utf-8 -*-
"""Denoise / coherence / select logic mixed into QtFastViewer."""

from __future__ import annotations

import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyAOBS.utils.qt_combo import hide_combo_popup

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

from ...core.denoise_coh_hints import (
    COHERENCE_DIP_SLOPE_PRESET_ITEMS,
    CUSTOM_DIP_SLOPE_COMBO_MARKER,
    estimate_moveout_slope_s_per_km_from_picks,
    suggest_coherence_cb_cl_start,
)

# Helpers used by denoise apply path
try:
    from pyAOBS.processors.denoise import denoise_trace, denoise_section
except ImportError:
    denoise_trace = None  # type: ignore
    denoise_section = None  # type: ignore


class DenoiseMixin:
    """去噪参数、选道、相干门控与应用到渲染道。"""

    def _sync_denoise_params_from_ui(self, *_args) -> None:
        """同步去噪参数到内存态与参数对象。"""
        bwconn = 8
        try:
            bwconn = int(self.combo_denoise_bwconn.currentText().strip())
        except Exception:
            bwconn = 8
        if bwconn not in (4, 8):
            bwconn = 8
        f_s = float(self.spin_denoise_f_s.value())
        f_e = float(self.spin_denoise_f_e.value())
        if f_e <= f_s:
            f_e = max(f_s + 0.1, 0.1)
            self.spin_denoise_f_e.blockSignals(True)
            self.spin_denoise_f_e.setValue(f_e)
            self.spin_denoise_f_e.blockSignals(False)
        scope_value = str(self.combo_denoise_scope.currentData() or "rendered").strip().lower()
        if scope_value not in ("rendered", "visible", "record", "selected"):
            scope_value = "rendered"
        diff_gain = float(self.combo_denoise_diff_gain.currentData() or 1.0)
        if (not np.isfinite(diff_gain)) or diff_gain <= 0.0:
            diff_gain = 1.0
        coh_win = int(self._denoise_params.get("coh_win", 11))
        if coh_win < 5:
            coh_win = 5
        if (coh_win % 2) == 0:
            coh_win += 1
        coh_lag = max(0, int(self._denoise_params.get("coh_lag", 2)))
        coh_thr = float(self._denoise_params.get("coh_thr", 0.55))
        coh_thr = float(min(0.95, max(0.05, coh_thr)))
        coh_blend = float(self._denoise_params.get("coh_blend", 0.35))
        coh_blend = float(min(0.90, max(0.0, coh_blend)))
        coh_penalty = float(self._denoise_params.get("coh_penalty", 0.08))
        coh_penalty = float(min(0.50, max(0.0, coh_penalty)))
        perf_diag = bool(self._denoise_params.get("perf_diag", False))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_preset = str(self._denoise_params.get("morph_preset", "balanced")).strip().lower()
        if morph_preset not in ("conservative", "balanced", "strong"):
            morph_preset = "balanced"
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_quantile = float(min(0.98, max(0.05, morph_quantile)))
        morph_min_area = int(max(1, int(self._denoise_params.get("morph_min_area", 24))))
        morph_expand = int(min(6, max(0, int(self._denoise_params.get("morph_expand", 1)))))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_floor_ratio = float(min(0.50, max(0.0, morph_floor_ratio)))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        morph_keep_strong_q = float(min(0.999, max(morph_quantile, morph_keep_strong_q)))
        self._denoise_params = {
            "enabled": bool(self.chk_denoise_enabled.isChecked()),
            # UI 语义与内部参数反向映射：勾选(B)->ab_raw=False，取消(A)->ab_raw=True
            "ab_raw": (not bool(self.chk_denoise_ab_raw.isChecked())),
            "show_diff": bool(getattr(self, "chk_denoise_show_diff", None) and self.chk_denoise_show_diff.isChecked()),
            "diff_gain": diff_gain,
            "scope": scope_value,
            "f_s": f_s,
            "f_e": f_e,
            "bwconn": bwconn,
            "strength": float(self.spin_denoise_strength.value()),
            "workers": max(1, int(self.spin_denoise_workers.value())),
            "coh_win": coh_win,
            "coh_lag": coh_lag,
            "coh_thr": coh_thr,
            "coh_blend": coh_blend,
            "coh_penalty": coh_penalty,
            "perf_diag": perf_diag,
            "morph_enable": morph_enable,
            "morph_preset": morph_preset,
            "morph_quantile": morph_quantile,
            "morph_min_area": morph_min_area,
            "morph_expand": morph_expand,
            "morph_floor_ratio": morph_floor_ratio,
            "morph_keep_strong_q": morph_keep_strong_q,
            "pick_guidance": bool(self.chk_denoise_pick_guidance.isChecked()),
            "pick_wavelet_length_sec": float(self.spin_denoise_pick_hw.value()),
            "pick_guidance_floor": float(self.spin_denoise_pick_floor.value()),
            "return_debug": False,
            "return_result": False,
        }
        # 与统一参数对象对齐，保证保存/加载可复现
        self.params.denoise_enabled = 1 if self._denoise_params["enabled"] else 0
        self.params.denoise_ab_raw = 1 if self._denoise_params["ab_raw"] else 0
        self.params.denoise_scope = str(self._denoise_params["scope"])
        self.params.denoise_f_s = float(self._denoise_params["f_s"])
        self.params.denoise_f_e = float(self._denoise_params["f_e"])
        self.params.denoise_bwconn = int(self._denoise_params["bwconn"])
        self.params.denoise_strength = float(self._denoise_params["strength"])
        self.params.denoise_workers = int(self._denoise_params["workers"])
        self.params.denoise_coh_win = int(self._denoise_params["coh_win"])
        self.params.denoise_coh_lag = int(self._denoise_params["coh_lag"])
        self.params.denoise_coh_thr = float(self._denoise_params["coh_thr"])
        self.params.denoise_coh_blend = float(self._denoise_params["coh_blend"])
        self.params.denoise_coh_penalty = float(self._denoise_params["coh_penalty"])
        self.params.denoise_perf_diag = 1 if bool(self._denoise_params["perf_diag"]) else 0
        self.params.denoise_morph_enable = 1 if bool(self._denoise_params["morph_enable"]) else 0
        self.params.denoise_morph_preset = str(self._denoise_params["morph_preset"])
        self.params.denoise_morph_quantile = float(self._denoise_params["morph_quantile"])
        self.params.denoise_morph_min_area = int(self._denoise_params["morph_min_area"])
        self.params.denoise_morph_expand = int(self._denoise_params["morph_expand"])
        self.params.denoise_morph_floor_ratio = float(self._denoise_params["morph_floor_ratio"])
        self.params.denoise_morph_keep_strong_q = float(self._denoise_params["morph_keep_strong_q"])
        self.params.denoise_pick_guidance = 1 if bool(self._denoise_params["pick_guidance"]) else 0
        self.params.denoise_pick_wavelet_length = float(self._denoise_params["pick_wavelet_length_sec"])
        self.params.denoise_pick_floor = float(self._denoise_params["pick_guidance_floor"])
        self.params.denoise_return_debug = 0
        self.params.denoise_return_result = 0
        self._update_denoise_hint()


    def _open_denoise_coh_dialog(self) -> None:
        """弹出相干参数窗口，避免占用主面板高度。"""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("相干参数")
        dlg.setModal(True)

        lay = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()

        spin_win = QtWidgets.QSpinBox(dlg)
        spin_win.setRange(5, 101)
        spin_win.setSingleStep(2)
        spin_win.setValue(int(self._denoise_params.get("coh_win", 11)))
        spin_win.setToolTip("cw（单位：样点）：局部 semblance 窗长（奇数）；越大越稳、越平滑。")
        form.addRow("窗长(cw, 样点)", spin_win)

        spin_lag = QtWidgets.QSpinBox(dlg)
        spin_lag.setRange(0, 8)
        spin_lag.setSingleStep(1)
        spin_lag.setValue(int(self._denoise_params.get("coh_lag", 2)))
        spin_lag.setToolTip("cl（单位：样点）：邻道时移搜索范围；越大越能跟踪倾斜同相轴。")
        form.addRow("Lag(cl, 样点)", spin_lag)

        spin_thr = QtWidgets.QDoubleSpinBox(dlg)
        spin_thr.setRange(0.05, 0.95)
        spin_thr.setDecimals(2)
        spin_thr.setSingleStep(0.01)
        spin_thr.setValue(float(self._denoise_params.get("coh_thr", 0.55)))
        spin_thr.setToolTip("ct（单位：无量纲）：相干门控阈值；越高越保守（只保护更高相干区域）。")
        form.addRow("阈值(ct, 无量纲)", spin_thr)

        spin_blend = QtWidgets.QDoubleSpinBox(dlg)
        spin_blend.setRange(0.0, 0.90)
        spin_blend.setDecimals(2)
        spin_blend.setSingleStep(0.01)
        spin_blend.setValue(float(self._denoise_params.get("coh_blend", 0.35)))
        spin_blend.setToolTip("cb（单位：无量纲）：原始信号回混上限；越大越保真，过大可能降低抑噪。")
        form.addRow("回混(cb, 无量纲)", spin_blend)

        spin_pen = QtWidgets.QDoubleSpinBox(dlg)
        spin_pen.setRange(0.0, 0.50)
        spin_pen.setDecimals(3)
        spin_pen.setSingleStep(0.01)
        spin_pen.setValue(float(self._denoise_params.get("coh_penalty", 0.08)))
        spin_pen.setToolTip("cp（单位：无量纲）：lag 轨迹平滑惩罚；越大越连续，过大可能忽略快速变化。")
        form.addRow("平滑惩罚(cp, 无量纲)", spin_pen)

        chk_morph = QtWidgets.QCheckBox("形态学约束", dlg)
        chk_morph.setChecked(bool(self._denoise_params.get("morph_enable", True)))
        chk_morph.setToolTip("开启后在GCV后增加连通域掩膜，抑制孤立噪点")
        form.addRow("Morph", chk_morph)

        combo_morph_preset = QtWidgets.QComboBox(dlg)
        combo_morph_preset.addItem("保守(保信号)", "conservative")
        combo_morph_preset.addItem("平衡", "balanced")
        combo_morph_preset.addItem("强抑噪", "strong")
        idx_preset = combo_morph_preset.findData(str(self._denoise_params.get("morph_preset", "balanced")))
        combo_morph_preset.setCurrentIndex(idx_preset if idx_preset >= 0 else 1)
        combo_morph_preset.setToolTip("形态学约束预设档位")
        form.addRow("Morph预设", combo_morph_preset)

        lay.addLayout(form)

        lbl_coh_dt_equiv = QtWidgets.QLabel(dlg)
        lbl_coh_dt_equiv.setWordWrap(True)
        lbl_coh_dt_equiv.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#246;",
        )

        def _coh_eff_win_samples(raw: int) -> int:
            w = int(max(5, raw))
            if (w % 2) == 0:
                w += 1
            return w

        def _update_coh_dt_equiv_hint() -> None:
            dt_s = self._loaded_times_dt_seconds()
            if dt_s is None:
                lbl_coh_dt_equiv.setText(
                    "等效尺度：未加载剖面或时间轴无效，无法按 dt 换算毫秒（可先加载数据后再打开此窗口核对）。",
                )
                lbl_coh_dt_equiv.setStyleSheet(
                    "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#888;",
                )
                return
            cw_raw = int(spin_win.value())
            w_eff = _coh_eff_win_samples(cw_raw)
            cl = int(spin_lag.value())
            dt_ms = float(dt_s * 1000.0)
            cw_ms = float(w_eff * dt_s * 1000.0)
            lag_half_ms = float(cl * dt_s * 1000.0)
            sr = 1.0 / dt_s
            fnyq = 0.5 * sr
            lbl_coh_dt_equiv.setStyleSheet(
                "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#246;",
            )
            w_note = ""
            if w_eff != cw_raw:
                w_note = f"（实际盒式相干窗会使用奇数 cw={w_eff}）"
            lbl_coh_dt_equiv.setText(
                f"等效尺度（由当前剖面 dt 推算）：dt≈{dt_ms:.6g} ms，采样率≈{sr:.6g} Hz，Nyquist≈{fnyq:.6g} Hz。\n"
                f"cw：盒式局部统计窗长约 {cw_ms:.6g} ms（{cw_raw}→{w_eff} 样点）{w_note}。\n"
                f"cl：邻道时移搜索为整数样点偏移 ±{cl}，单侧最大约 ±{lag_half_ms:.6g} ms。"
            )

        spin_win.valueChanged.connect(lambda _v: _update_coh_dt_equiv_hint())
        spin_lag.valueChanged.connect(lambda _v: _update_coh_dt_equiv_hint())
        _update_coh_dt_equiv_hint()
        lay.addWidget(lbl_coh_dt_equiv)

        tip = QtWidgets.QLabel(
            "参数说明：cw/cl 单位为样点；ct/cb/cp 为无量纲。\n"
            "调参方向：弱事件被削弱时可适当降 ct / 升 cb；噪声残留偏多时可升 ct / 降 cb。",
            dlg,
        )
        tip.setStyleSheet("color:#666;")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        tip2 = QtWidgets.QLabel(
            "固定建议初值：cw=11, cl=2, ct=0.55, cb=0.35, cp=0.08。"
            "可用「拾取→估梯度」按某一拾取字的同相轴走时估 ms/km，再点「按 dt 与偏移距推荐 cl/cb」。",
            dlg,
        )
        tip2.setStyleSheet("color:#666;")
        tip2.setWordWrap(True)
        lay.addWidget(tip2)

        slope_form = QtWidgets.QFormLayout()
        combo_dip_slope = QtWidgets.QComboBox(dlg)
        for _lbl, slope_spkm in COHERENCE_DIP_SLOPE_PRESET_ITEMS:
            combo_dip_slope.addItem(_lbl, float(slope_spkm))
        combo_dip_slope.addItem("自定义（下方 ms/km）", CUSTOM_DIP_SLOPE_COMBO_MARKER)
        combo_dip_slope.setCurrentIndex(1)
        combo_dip_slope.setToolTip(
            "用于「推荐 cl」：假定相邻道在记录时间采样轴（times）上沿偏移的时差梯度；**未自动含折合速度**。\n"
            "折合仅在绘图纵轴平移波形、不重采样；屏上看平≠样点域已对齐，大 moveout 仍宜高档或自定义更大 ms/km。",
        )
        spin_dip_slope_custom_ms = QtWidgets.QDoubleSpinBox(dlg)
        spin_dip_slope_custom_ms.setRange(0.5, 200.0)
        spin_dip_slope_custom_ms.setDecimals(2)
        spin_dip_slope_custom_ms.setSingleStep(1.0)
        spin_dip_slope_custom_ms.setValue(18.0)
        spin_dip_slope_custom_ms.setSuffix(" ms/km")
        spin_dip_slope_custom_ms.setToolTip(
            "仅在选中「自定义」时生效：梯度=该值÷1000（s/km），指记录时间域；与折合显示 t−|x|(1/v−1/vhdr) 的纵轴画法无关。"
        )

        def _dip_slope_combo_is_custom() -> bool:
            return combo_dip_slope.currentData() == CUSTOM_DIP_SLOPE_COMBO_MARKER

        def _current_dip_slope_s_per_km() -> float:
            d = combo_dip_slope.currentData()
            if d == CUSTOM_DIP_SLOPE_COMBO_MARKER:
                return float(max(1e-6, spin_dip_slope_custom_ms.value() / 1000.0))
            try:
                return float(d)
            except (TypeError, ValueError):
                return 0.012

        def _on_dip_slope_preset_changed(_idx: int = 0) -> None:
            hide_combo_popup(combo_dip_slope)
            spin_dip_slope_custom_ms.setEnabled(_dip_slope_combo_is_custom())

        combo_dip_slope.currentIndexChanged.connect(_on_dip_slope_preset_changed)
        _on_dip_slope_preset_changed()
        slope_form.addRow("推荐 cl：梯度档位", combo_dip_slope)
        slope_form.addRow("自定义梯度", spin_dip_slope_custom_ms)
        lay.addLayout(slope_form)

        pick_row_w = QtWidgets.QWidget(dlg)
        pick_row_l = QtWidgets.QHBoxLayout(pick_row_w)
        pick_row_l.setContentsMargins(0, 0, 0, 0)
        spin_pw_for_slope = QtWidgets.QSpinBox(pick_row_w)
        npick_cap = 40
        try:
            if self.pick_manager is not None:
                npick_cap = int(max(1, min(80, int(getattr(self.pick_manager, "npick", 40) or 40))))
        except Exception:
            npick_cap = 40
        spin_pw_for_slope.setRange(1, int(npick_cap))
        try:
            spin_pw_for_slope.setValue(int(max(1, min(npick_cap, int(self.spin_apick.value())))))
        except Exception:
            spin_pw_for_slope.setValue(1)
        spin_pw_for_slope.setToolTip("仅使用该拾取字的点估计 |Δt|/|Δh|；换震相请换拾取字。")
        btn_pick_to_slope = QtWidgets.QPushButton("拾取→估梯度(ms/km)", pick_row_w)
        btn_pick_to_slope.setToolTip(
            "按该拾取字将各道走时与 offsets(km) 排序，取相邻拾取段 |Δt|/|Δh| 的 median 填入自定义梯度，"
            "并在说明里给出 p75 供保守加大 lag。再走「按 dt…推荐 cl/cb」。"
        )
        pick_row_l.addWidget(QtWidgets.QLabel("拾取字"))
        pick_row_l.addWidget(spin_pw_for_slope)
        pick_row_l.addStretch(1)
        pick_row_l.addWidget(btn_pick_to_slope)
        lay.addWidget(pick_row_w)

        def _infer_slope_from_picks() -> None:
            if self.pick_manager is None:
                lbl_cb_cl_rec.setText("拾取管理器不可用。")
                lbl_cb_cl_rec.setStyleSheet("color:#a44; font-size:11px;")
                return
            pw_i = int(spin_pw_for_slope.value())
            by_w = self.pick_manager.get_picks_by_word(pw_i)
            offs_a = None
            ld = getattr(self, "loaded", None)
            if isinstance(ld, dict):
                offs_a = ld.get("offsets")
            med, p75, msg = estimate_moveout_slope_s_per_km_from_picks(by_w, offs_a)
            if med is None or (not math.isfinite(med)) or med <= 0.0:
                lbl_cb_cl_rec.setText(msg)
                lbl_cb_cl_rec.setStyleSheet("color:#a44; font-size:11px;")
                return
            ic = combo_dip_slope.findData(CUSTOM_DIP_SLOPE_COMBO_MARKER)
            if ic >= 0:
                combo_dip_slope.setCurrentIndex(int(ic))
            spin_dip_slope_custom_ms.setValue(float(min(200.0, max(0.5, med * 1000.0))))
            extra = ""
            if p75 is not None and math.isfinite(p75) and med > 0.0 and p75 > med * 1.02:
                extra = f" 保守侧 p75≈{p75 * 1000.0:.2f} ms/km（可手改自定义接近此值以略增 cl）。"
            lbl_cb_cl_rec.setText(msg + extra)
            lbl_cb_cl_rec.setStyleSheet("color:#246; font-size:11px;")
            _on_dip_slope_preset_changed()
            _update_coh_dt_equiv_hint()

        btn_pick_to_slope.clicked.connect(_infer_slope_from_picks)

        lbl_cb_cl_rec = QtWidgets.QLabel("", dlg)
        lbl_cb_cl_rec.setWordWrap(True)
        lbl_cb_cl_rec.setStyleSheet("color:#444; font-size:11px;")

        chk_perf_diag = QtWidgets.QCheckBox("性能诊断日志", dlg)
        chk_perf_diag.setChecked(bool(self._denoise_params.get("perf_diag", False)))
        chk_perf_diag.setToolTip("开启后记录去噪各阶段耗时（写入调试日志）")

        lay.addWidget(lbl_cb_cl_rec)
        lay.addWidget(chk_perf_diag)

        btn_restore = QtWidgets.QPushButton("恢复固定初值", dlg)
        btn_restore.setToolTip("cw=11, cl=2, ct=0.55, cb=0.35, cp=0.08（与文档固定建议一致）")
        btn_suggest_cb_cl = QtWidgets.QPushButton("按 dt 与偏移距推荐 cl/cb", dlg)
        btn_suggest_cb_cl.setToolTip(
            "用当前剖面的 dt 与 offsets(km) 邻道间距中位数估算 lag(cl)，并按采样率启发回混(cb)。\n"
            "梯度取自上方档位或自定义 ms/km（乘以median Δh 与安全系数得到靶邻道时差）。",
        )

        def _apply_dt_spacing_cb_cl_hint() -> None:
            dt_s = self._loaded_times_dt_seconds()
            if dt_s is None:
                lbl_cb_cl_rec.setText("无法推荐：请先加载剖面（有效时间轴）。")
                lbl_cb_cl_rec.setStyleSheet("color:#a44; font-size:11px;")
                return
            offs = None
            ld = getattr(self, "loaded", None)
            if isinstance(ld, dict):
                offs = ld.get("offsets")
            cl_r, cb_r, detail = suggest_coherence_cb_cl_start(
                dt_s,
                offs,
                lag_max=int(spin_lag.maximum()),
                dip_slope_s_per_km=float(_current_dip_slope_s_per_km()),
            )
            spin_lag.setValue(int(cl_r))
            spin_blend.setValue(float(min(float(spin_blend.maximum()), max(float(spin_blend.minimum()), cb_r))))
            lbl_cb_cl_rec.setText(detail)
            lbl_cb_cl_rec.setStyleSheet("color:#444; font-size:11px;")
            _update_coh_dt_equiv_hint()

        btn_suggest_cb_cl.clicked.connect(_apply_dt_spacing_cb_cl_hint)

        def _restore_defaults() -> None:
            spin_win.setValue(11)
            spin_lag.setValue(2)
            spin_thr.setValue(0.55)
            spin_blend.setValue(0.35)
            spin_pen.setValue(0.08)
            chk_morph.setChecked(True)
            combo_morph_preset.setCurrentIndex(combo_morph_preset.findData("balanced"))
            lbl_cb_cl_rec.clear()
            _update_coh_dt_equiv_hint()
        btn_restore.clicked.connect(_restore_defaults)

        btn_row_preset = QtWidgets.QHBoxLayout()
        btn_row_preset.addWidget(btn_restore)
        btn_row_preset.addWidget(btn_suggest_cb_cl)
        lay.addLayout(btn_row_preset)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return

        coh_win = int(spin_win.value())
        if (coh_win % 2) == 0:
            coh_win += 1
        self._denoise_params["coh_win"] = int(max(5, coh_win))
        self._denoise_params["coh_lag"] = int(max(0, spin_lag.value()))
        self._denoise_params["coh_thr"] = float(min(0.95, max(0.05, spin_thr.value())))
        self._denoise_params["coh_blend"] = float(min(0.90, max(0.0, spin_blend.value())))
        self._denoise_params["coh_penalty"] = float(min(0.50, max(0.0, spin_pen.value())))
        self._denoise_params["perf_diag"] = bool(chk_perf_diag.isChecked())
        self._denoise_params["morph_enable"] = bool(chk_morph.isChecked())
        morph_preset = str(combo_morph_preset.currentData() or "balanced").strip().lower()
        if morph_preset not in ("conservative", "balanced", "strong"):
            morph_preset = "balanced"
        self._denoise_params["morph_preset"] = morph_preset
        if morph_preset == "conservative":
            self._denoise_params["morph_quantile"] = 0.64
            self._denoise_params["morph_min_area"] = 12
            self._denoise_params["morph_expand"] = 1
            self._denoise_params["morph_floor_ratio"] = 0.02
            self._denoise_params["morph_keep_strong_q"] = 0.94
        elif morph_preset == "strong":
            self._denoise_params["morph_quantile"] = 0.82
            self._denoise_params["morph_min_area"] = 48
            self._denoise_params["morph_expand"] = 0
            self._denoise_params["morph_floor_ratio"] = 0.05
            self._denoise_params["morph_keep_strong_q"] = 0.97
        else:
            self._denoise_params["morph_quantile"] = 0.70
            self._denoise_params["morph_min_area"] = 24
            self._denoise_params["morph_expand"] = 1
            self._denoise_params["morph_floor_ratio"] = 0.03
            self._denoise_params["morph_keep_strong_q"] = 0.95

        self._on_denoise_ui_changed()


    def _on_denoise_ui_changed(self, *_args) -> None:
        """去噪控件变更：同步参数并触发重绘。"""
        hide_combo_popup(self.sender() if isinstance(self.sender(), QtWidgets.QComboBox) else None)
        self._denoise_run_armed = False
        self._clear_denoise_cache()
        self._sync_denoise_params_from_ui()
        self._sync_plot_pan_lock_state()
        self.request_render(delay_ms=10)


    def _on_denoise_view_mode_changed(self, *_args) -> None:
        """仅显示模式变更（A/B/差值）不解除已启动态。"""
        hide_combo_popup(self.sender() if isinstance(self.sender(), QtWidgets.QComboBox) else None)
        self._sync_denoise_params_from_ui()
        self._sync_plot_pan_lock_state()
        self.request_render(delay_ms=10)


    def _clear_denoise_selected_traces(self) -> None:
        """清空手动选道集合。"""
        self._denoise_run_armed = False
        self._clear_denoise_cache()
        self._denoise_selected_traces.clear()
        self._update_denoise_hint()
        self._sync_plot_pan_lock_state()
        self.request_render(delay_ms=10)


    def _is_denoise_select_mode_active(self) -> bool:
        scope_selected = str(self._denoise_params.get("scope", "rendered")).strip().lower() == "selected"
        return (
            scope_selected
            and bool(self.chk_denoise_enabled.isChecked())
            and self.loaded is not None
        )


    def _is_denoise_select_modifier_active(self) -> bool:
        """手动选道手势修饰键：Shift/Ctrl/Alt 任一按下。"""
        mods = QtWidgets.QApplication.keyboardModifiers()
        return bool(
            (mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
            or (mods & QtCore.Qt.KeyboardModifier.ControlModifier)
            or (mods & QtCore.Qt.KeyboardModifier.AltModifier)
        )


    def _clear_denoise_cache(self) -> None:
        self._denoise_cache_entries.clear()
        self._denoise_frozen_ready = False
        self._denoise_frozen_trace_set = set()
        self._denoise_frozen_by_trace = {}
        self._denoise_frozen_original_by_trace = {}
        self._denoise_frozen_delta_mean_abs = 0.0
        self._denoise_frozen_delta_max_abs = 0.0


    def _toggle_denoise_selected_trace(self, trace_idx: int, remove_only: bool = False) -> bool:
        """切换/移除单道选中状态，返回是否有变化。"""
        trace_idx = int(trace_idx)
        changed = False
        if remove_only:
            if trace_idx in self._denoise_selected_traces:
                self._denoise_selected_traces.remove(trace_idx)
                changed = True
        else:
            if trace_idx in self._denoise_selected_traces:
                self._denoise_selected_traces.remove(trace_idx)
            else:
                self._denoise_selected_traces.add(trace_idx)
            changed = True
        if changed:
            self._update_denoise_hint()
            self._set_status_text(
                f"选道：道 {trace_idx} -> {'选中' if trace_idx in self._denoise_selected_traces else '取消'}",
                hold_ms=900,
                force=True,
            )
            self._debug_log(
                "DENOISE_PICK",
                f"trace={int(trace_idx)} remove_only={int(bool(remove_only))} selected_count={len(self._denoise_selected_traces)}",
            )
            self.request_render(delay_ms=10)
        return changed


    def _toggle_denoise_selected_trace_by_scene_pos(self, scene_pos, remove_only: bool = False) -> bool:
        """根据 scene 坐标定位最近渲染道并执行切换。"""
        if self._last_render_trace_indices.size == 0 or self._last_render_offsets.size == 0:
            return False
        vb = self.plot.getViewBox()
        try:
            mouse_pt = vb.mapSceneToView(scene_pos)
        except Exception:
            return False
        x = float(mouse_pt.x())
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x)))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        return self._toggle_denoise_selected_trace(trace_idx, remove_only=remove_only)


    def _denoise_indices_signature(self, indices: np.ndarray) -> Tuple[int, int, int, int]:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return (0, 0, 0, 0)
        return (
            int(arr.size),
            int(arr.min()),
            int(arr.max()),
            int(np.sum(arr, dtype=np.int64)),
        )


    def _denoise_traces_signature(self, traces: List[np.ndarray]) -> Tuple[int, int, float, float, float, float]:
        n_trace = int(len(traces))
        n_sample = 0
        acc_abs_mean = 0.0
        acc_mean = 0.0
        acc_pow = 0.0
        acc_anchor = 0.0
        for tr in traces:
            x = np.asarray(tr, dtype=np.float64).reshape(-1)
            if x.size <= 0:
                continue
            n_sample += int(x.size)
            acc_abs_mean += float(np.mean(np.abs(x)))
            acc_mean += float(np.mean(x))
            acc_pow += float(np.mean(x * x))
            mid = int(x.size // 2)
            acc_anchor += float(x[0] + x[mid] + x[-1])
        return (
            n_trace,
            int(n_sample),
            round(acc_abs_mean, 6),
            round(acc_mean, 6),
            round(acc_pow, 6),
            round(acc_anchor, 6),
        )


    def _trace_linear_corr_abs(self, a: np.ndarray, b: np.ndarray) -> float:
        """Estimate absolute linear correlation between two traces."""
        x = np.asarray(a, dtype=np.float64).reshape(-1)
        y = np.asarray(b, dtype=np.float64).reshape(-1)
        n = int(min(x.size, y.size))
        if n < 16:
            return 0.0
        x = x[:n]
        y = y[:n]
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        sx = float(np.sqrt(np.mean(x * x)))
        sy = float(np.sqrt(np.mean(y * y)))
        if sx <= 1e-12 or sy <= 1e-12:
            return 0.0
        c = float(np.mean((x / sx) * (y / sy)))
        if not np.isfinite(c):
            return 0.0
        return float(min(1.0, max(0.0, abs(c))))


    def _shift_trace_samples(self, x: np.ndarray, shift: int) -> np.ndarray:
        """Shift trace by integer samples with zero padding."""
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        n = int(arr.size)
        if n <= 0 or int(shift) == 0:
            return arr.copy()
        y = np.zeros_like(arr)
        s = int(shift)
        if s > 0:
            y[s:] = arr[: n - s]
        else:
            k = -s
            y[: n - k] = arr[k:]
        return y


    def _coh_conv_kernel(self, win: int) -> np.ndarray:
        """Cached box kernel for local semblance window."""
        w = int(max(5, int(win)))
        if (w % 2) == 0:
            w += 1
        ker = self._coh_gate_kernel_cache.get(int(w))
        if ker is None:
            ker = np.ones((w,), dtype=np.float64)
            self._coh_gate_kernel_cache[int(w)] = ker
        return ker


    def _coh_lag_values(self, max_lag: int) -> np.ndarray:
        """Cached lag values array."""
        mlag = int(max(0, int(max_lag)))
        cached = self._coh_gate_lags_cache.get(mlag)
        if cached is not None:
            return cached
        lags_arr = np.arange(-mlag, mlag + 1, dtype=np.float64)
        if lags_arr.size <= 0:
            lags_arr = np.zeros((1,), dtype=np.float64)
        if len(self._coh_gate_lags_cache) >= 16:
            first_key = next(iter(self._coh_gate_lags_cache.keys()))
            self._coh_gate_lags_cache.pop(first_key, None)
        self._coh_gate_lags_cache[mlag] = lags_arr
        return lags_arr


    def _dp_best_prev_l1(
        self,
        prev_score: np.ndarray,
        pen: float,
        scratch: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each state j, solve max_i(prev[i]-pen*|i-j|) in O(k)."""
        p = np.asarray(prev_score, dtype=np.float64).reshape(-1)
        k = int(p.size)
        if k <= 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int32)
        if pen <= 0.0:
            idx = np.arange(k, dtype=np.int32)
            return p.copy(), idx

        if scratch is not None:
            left_val = scratch["left_val"]
            right_val = scratch["right_val"]
            left_idx = scratch["left_idx"]
            right_idx = scratch["right_idx"]
            best_val = scratch["best_val"]
            best_idx = scratch["best_idx"]
        else:
            left_val = np.empty((k,), dtype=np.float64)
            right_val = np.empty((k,), dtype=np.float64)
            left_idx = np.empty((k,), dtype=np.int32)
            right_idx = np.empty((k,), dtype=np.int32)
            best_val = np.empty((k,), dtype=np.float64)
            best_idx = np.empty((k,), dtype=np.int32)

        left_val[0] = p[0]
        left_idx[0] = 0
        for j in range(1, k):
            cand = left_val[j - 1] - pen
            if p[j] >= cand:
                left_val[j] = p[j]
                left_idx[j] = j
            else:
                left_val[j] = cand
                left_idx[j] = left_idx[j - 1]

        right_val[k - 1] = p[k - 1]
        right_idx[k - 1] = k - 1
        for j in range(k - 2, -1, -1):
            cand = right_val[j + 1] - pen
            if p[j] >= cand:
                right_val[j] = p[j]
                right_idx[j] = j
            else:
                right_val[j] = cand
                right_idx[j] = right_idx[j + 1]

        sel = left_val >= right_val
        np.copyto(best_val, right_val)
        np.copyto(best_idx, right_idx)
        best_val[sel] = left_val[sel]
        best_idx[sel] = left_idx[sel]
        return best_val, best_idx.astype(np.int32, copy=False)


    def _coh_get_thread_cache(self) -> Dict[str, Dict[Tuple[int, ...], Dict[str, np.ndarray]]]:
        cache = getattr(self._coh_thread_local, "coh_cache", None)
        if cache is None:
            cache = {"work": {}, "dp": {}}
            self._coh_thread_local.coh_cache = cache
        return cache


    def _coh_get_work_buffers(self, k: int, n: int) -> Dict[str, np.ndarray]:
        cache = self._coh_get_thread_cache()["work"]
        key = (int(k), int(n))
        buf = cache.get(key)
        if buf is None:
            kk = int(max(1, k))
            nn = int(max(1, n))
            buf = {
                "a_stack": np.empty((kk, nn), dtype=np.float64),
                "c_stack": np.empty((kk, nn), dtype=np.float64),
                "s_stack": np.empty((kk, nn), dtype=np.float64),
                "e_stack": np.empty((kk, nn), dtype=np.float64),
                "num": np.empty((kk, nn), dtype=np.float64),
                "den": np.empty((kk, nn), dtype=np.float64),
                "sem_stack": np.empty((kk, nn), dtype=np.float64),
                "score": np.empty((kk, nn), dtype=np.float64),
                "parent": np.empty((kk, nn), dtype=np.int32),
                "lag_idx": np.empty((nn,), dtype=np.int32),
            }
            if len(cache) >= 4:
                first_key = next(iter(cache.keys()))
                cache.pop(first_key, None)
            cache[key] = buf
        return buf


    def _coh_get_dp_scratch(self, k: int) -> Dict[str, np.ndarray]:
        cache = self._coh_get_thread_cache()["dp"]
        key = (int(k),)
        buf = cache.get(key)
        if buf is None:
            kk = int(max(1, k))
            buf = {
                "left_val": np.empty((kk,), dtype=np.float64),
                "right_val": np.empty((kk,), dtype=np.float64),
                "left_idx": np.empty((kk,), dtype=np.int32),
                "right_idx": np.empty((kk,), dtype=np.int32),
                "best_val": np.empty((kk,), dtype=np.float64),
                "best_idx": np.empty((kk,), dtype=np.int32),
            }
            if len(cache) >= 8:
                first_key = next(iter(cache.keys()))
                cache.pop(first_key, None)
            cache[key] = buf
        return buf


    def _moving_sum_2d_same(self, x2d: np.ndarray, win: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized moving-sum on axis=1 with zero padding ('same')."""
        x = np.asarray(x2d, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("_moving_sum_2d_same expects 2D array")
        n = int(x.shape[1])
        if n <= 0:
            return np.zeros_like(x)
        w = int(max(1, int(win)))
        if (w % 2) == 0:
            w += 1
        h = int(w // 2)
        xp = np.pad(x, ((0, 0), (h, h)), mode="constant")
        cs = np.cumsum(np.pad(xp, ((0, 0), (1, 0)), mode="constant"), axis=1)
        out_arr = cs[:, w:] - cs[:, :-w]
        out_arr = out_arr.astype(np.float64, copy=False)
        if out is not None and out.shape == out_arr.shape:
            np.copyto(out, out_arr)
            return out
        return out_arr


    def _local_semblance_gate(
        self,
        tr_prev: np.ndarray,
        tr_cur: np.ndarray,
        tr_next: np.ndarray,
        *,
        max_lag: int = 2,
        win: int = 11,
        coh_thr: float = 0.55,
        lag_penalty: float = 0.08,
        prev_shift_cache: Optional[Dict[int, np.ndarray]] = None,
        next_shift_cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Estimate per-sample coherence gate by local semblance and lag-path tracking."""
        a = np.asarray(tr_prev, dtype=np.float64).reshape(-1)
        b = np.asarray(tr_cur, dtype=np.float64).reshape(-1)
        c = np.asarray(tr_next, dtype=np.float64).reshape(-1)
        n = int(min(a.size, b.size, c.size))
        if n <= 16:
            return np.zeros((max(0, n),), dtype=np.float64)
        a = a[:n]
        b = b[:n]
        c = c[:n]
        ker = self._coh_conv_kernel(int(win))
        w = int(ker.size)
        eps = 1e-12
        lags_arr = self._coh_lag_values(int(max_lag))
        lags = lags_arr.astype(np.int32, copy=False)
        if lags.size <= 0:
            return np.zeros((n,), dtype=np.float64)
        k = int(lags.size)
        buf = self._coh_get_work_buffers(k, n)
        a_stack = buf["a_stack"]
        c_stack = buf["c_stack"]
        s_stack = buf["s_stack"]
        e_stack = buf["e_stack"]
        num = buf["num"]
        den = buf["den"]
        sem_stack = buf["sem_stack"]
        for li, lag in enumerate(lags):
            lag_i = int(lag)
            if prev_shift_cache is not None:
                a_al = prev_shift_cache.get(-lag_i)
                if a_al is None:
                    a_al = self._shift_trace_samples(a, -lag_i)
            else:
                a_al = self._shift_trace_samples(a, -lag_i)
            if next_shift_cache is not None:
                c_al = next_shift_cache.get(lag_i)
                if c_al is None:
                    c_al = self._shift_trace_samples(c, lag_i)
            else:
                c_al = self._shift_trace_samples(c, lag_i)
            a_stack[li, :] = np.asarray(a_al, dtype=np.float64).reshape(-1)[:n]
            c_stack[li, :] = np.asarray(c_al, dtype=np.float64).reshape(-1)[:n]

        b_row = b.reshape((1, n))
        np.add(a_stack, c_stack, out=s_stack)
        s_stack += b_row
        np.multiply(a_stack, a_stack, out=e_stack)
        e_stack += (b_row * b_row)
        e_stack += (c_stack * c_stack)
        num = self._moving_sum_2d_same(s_stack, w, out=num)
        num = num * num
        den = self._moving_sum_2d_same(e_stack, w, out=den)
        den *= 3.0
        den += eps
        np.divide(num, den, out=sem_stack)
        np.clip(sem_stack, 0.0, 1.0, out=sem_stack)

        pen = float(max(0.0, float(lag_penalty)))
        if pen <= 1e-12:
            # Fast path when no lag smoothness penalty.
            best = np.max(sem_stack, axis=0)
            thr = float(min(0.95, max(0.05, coh_thr)))
            gate = np.clip((best - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0)
            gate = np.convolve(gate, self._coh_gate_smooth_kernel, mode="same")
            return np.clip(gate, 0.0, 1.0)

        # Lag-path tracking DP with O(k*n) transition step for L1 penalty.
        score = buf["score"]
        parent = buf["parent"]
        lag_idx = buf["lag_idx"]
        score.fill(-1e18)
        parent.fill(0)
        score[:, 0] = sem_stack[:, 0]
        dp_scratch = self._coh_get_dp_scratch(k)
        for t in range(1, n):
            best_prev_val, best_prev_idx = self._dp_best_prev_l1(score[:, t - 1], pen, scratch=dp_scratch)
            score[:, t] = sem_stack[:, t] + best_prev_val
            parent[:, t] = best_prev_idx.astype(np.int32)
        lag_idx.fill(0)
        lag_idx[-1] = int(np.argmax(score[:, -1]))
        for t in range(n - 1, 0, -1):
            lag_idx[t - 1] = int(parent[int(lag_idx[t]), t])
        best = sem_stack[lag_idx, np.arange(n)]

        thr = float(min(0.95, max(0.05, coh_thr)))
        gate = np.clip((best - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0)
        # light smoothing to avoid flicker-like sample spikes
        gate = np.convolve(gate, self._coh_gate_smooth_kernel, mode="same")
        return np.clip(gate, 0.0, 1.0)


    def _denoise_pick_times_for_global_idx(self, gidx: int) -> List[float]:
        """Collect positive pick times (seconds) for one global trace index."""
        if self.pick_manager is None or gidx < 0:
            return []
        try:
            d = self.pick_manager.get_picks_for_trace(int(gidx))
        except Exception:
            return []
        out: List[float] = []
        for _pw, tv in d.items():
            try:
                v = float(tv)
            except Exception:
                continue
            if np.isfinite(v) and v > 0.0:
                out.append(v)
        return out


    def _denoise_pick_template_times_for_globals(self, gidx_list: List[int]) -> List[float]:
        """将若干全局道上的拾取合并为一套时间模板（去噪范围内共用，含未拾取道）。"""
        if self.pick_manager is None or not gidx_list:
            return []
        seen: set[float] = set()
        merged: List[float] = []
        for gi in gidx_list:
            for t in self._denoise_pick_times_for_global_idx(int(gi)):
                key = round(float(t), 4)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(float(t))
        merged.sort()
        return merged


    def _denoise_pick_guidance_cache_stamp(self) -> object:
        """Invalidate denoise cache when pick-guided mode depends on pick sets."""
        if not bool(self._denoise_params.get("pick_guidance", False)):
            return 0
        pm = self.pick_manager
        if pm is None:
            return 1
        try:
            parts: List[Tuple[int, int, float]] = []
            for tidx in sorted(pm.picks.keys()):
                row = pm.picks[tidx]
                for pw in sorted(row.keys()):
                    parts.append((int(tidx), int(pw), round(float(row[pw]), 6)))
            return hash(tuple(parts))
        except Exception:
            return 2


    def _denoise_cache_key_of(
        self,
        traces_before_denoise: List[np.ndarray],
        render_trace_indices: np.ndarray,
        denoise_trace_indices: np.ndarray,
    ) -> Tuple[object, ...]:
        return (
            self._denoise_traces_signature(traces_before_denoise),
            self._denoise_indices_signature(np.asarray(render_trace_indices, dtype=int)),
            self._denoise_indices_signature(np.asarray(denoise_trace_indices, dtype=int)),
            round(float(self._denoise_params.get("f_s", 3.0)), 6),
            round(float(self._denoise_params.get("f_e", 20.0)), 6),
            int(self._denoise_params.get("bwconn", 8)),
            round(float(self._denoise_params.get("strength", 3.0)), 6),
            int(self._denoise_params.get("coh_win", 11)),
            int(self._denoise_params.get("coh_lag", 2)),
            round(float(self._denoise_params.get("coh_thr", 0.55)), 6),
            round(float(self._denoise_params.get("coh_blend", 0.35)), 6),
            round(float(self._denoise_params.get("coh_penalty", 0.08)), 6),
            int(bool(self._denoise_params.get("morph_enable", True))),
            str(self._denoise_params.get("morph_preset", "balanced")),
            round(float(self._denoise_params.get("morph_quantile", 0.70)), 6),
            int(self._denoise_params.get("morph_min_area", 24)),
            int(self._denoise_params.get("morph_expand", 1)),
            round(float(self._denoise_params.get("morph_floor_ratio", 0.03)), 6),
            round(float(self._denoise_params.get("morph_keep_strong_q", 0.95)), 6),
            int(bool(self._denoise_params.get("pick_guidance", False))),
            round(float(self._denoise_params.get("pick_wavelet_length_sec", 0.19)), 6),
            round(float(self._denoise_params.get("pick_guidance_floor", 0.12)), 6),
            self._denoise_pick_guidance_cache_stamp(),
            int(bool(self._denoise_params.get("return_debug", False))),
            int(bool(self._denoise_params.get("return_result", False))),
        )


    def _start_denoise_now(self) -> None:
        """显式启动去噪执行。"""
        if not bool(self.chk_denoise_enabled.isChecked()):
            self._set_status_text("请先勾选去噪启用，再开始去噪", hold_ms=1400)
            return
        self._sync_denoise_params_from_ui()
        self._clear_denoise_cache()
        self._denoise_run_armed = True
        self._set_denoise_progress(0, 0)
        # Ensure progress bar paints before entering heavy render path.
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        self._set_status_text("去噪已启动（本次结果将缓存）", hold_ms=1400)
        self.request_render(immediate=True)


    def _set_denoise_progress(self, done: int, total: int, phase: Optional[str] = None) -> None:
        """更新去噪进度条；total<0 隐藏，total<=0 显示不定进度。"""
        if not hasattr(self, "progress_denoise") or self.progress_denoise is None:
            return
        ptxt = str(phase).strip() if phase is not None else ""
        last_ptxt = str(getattr(self, "_denoise_progress_phase", "") or "")
        if ptxt and ptxt != last_ptxt:
            self._denoise_progress_phase = ptxt
            self._set_status_text(f"DN: {ptxt}", force=True)
        d = int(max(0, done))
        t_raw = int(total)
        if t_raw < 0:
            self._denoise_progress_phase = ""
            self.progress_denoise.setVisible(False)
            return
        t = int(max(0, t_raw))
        self.progress_denoise.setVisible(True)
        if t <= 0:
            self.progress_denoise.setRange(0, 0)
            self.progress_denoise.setFormat(f"DN {ptxt}" if ptxt else "DN 计算中...")
            return
        if d > t:
            d = t
        self.progress_denoise.setRange(0, t)
        self.progress_denoise.setValue(d)
        if ptxt:
            self.progress_denoise.setFormat(f"DN {ptxt} {d}/{t} (%p%)")
        else:
            self.progress_denoise.setFormat(f"DN {d}/{t} (%p%)")


    def _finish_denoise_drag_select(self) -> bool:
        """结束框选并批量加入范围内道；返回是否执行了批量框选。"""
        if (not self._denoise_select_drag_active) or self._last_render_trace_indices.size == 0:
            self._denoise_select_drag_active = False
            self._denoise_select_drag_start_x = None
            self._denoise_select_drag_last_x = None
            self._denoise_select_drag_mode = "add"
            return False
        x0 = self._denoise_select_drag_start_x
        x1 = self._denoise_select_drag_last_x
        mode = str(self._denoise_select_drag_mode or "add")
        self._denoise_select_drag_active = False
        self._denoise_select_drag_start_x = None
        self._denoise_select_drag_last_x = None
        self._denoise_select_drag_mode = "add"
        if x0 is None or x1 is None:
            return False
        xmin = float(min(x0, x1))
        xmax = float(max(x0, x1))
        # 拖拽宽度太小视为点击，不走批量框选
        xr, _ = self.plot.getViewBox().viewRange()
        x_span = max(1e-9, abs(float(xr[1]) - float(xr[0])))
        if abs(xmax - xmin) < 0.005 * x_span:
            return False
        sel_mask = (self._last_render_offsets >= xmin) & (self._last_render_offsets <= xmax)
        picked = self._last_render_trace_indices[sel_mask]
        picked_set = {int(v) for v in picked.tolist()} if picked.size > 0 else set()
        if mode == "replace":
            self._denoise_selected_traces = set(picked_set)
        elif mode == "remove":
            if picked_set:
                self._denoise_selected_traces.difference_update(picked_set)
        else:
            if picked_set:
                self._denoise_selected_traces.update(picked_set)
        self._denoise_select_drag_just_finished = True
        self._update_denoise_hint()
        op_cn = "替换"
        if mode == "remove":
            op_cn = "移除"
        elif mode == "add":
            op_cn = "追加"
        self._set_status_text(
            f"选道框选[{op_cn}]：框内 {len(picked_set)} 道，当前已选 {len(self._denoise_selected_traces)} 道",
            hold_ms=1400,
        )
        self.request_render(delay_ms=10)
        return True


    def _update_denoise_hint(self) -> None:
        if not hasattr(self, "lbl_denoise_hint") or self.lbl_denoise_hint is None:
            return
        scope_count = int(max(0, int(getattr(self, "_last_denoise_scope_count", 0))))
        dmean = float(getattr(self, "_denoise_last_delta_mean_abs", 0.0))
        hint_text = f"道数：{scope_count} | Δmean={dmean:.3e}"
        if bool(self._denoise_params.get("show_diff", False)):
            dg = float(self._denoise_params.get("diff_gain", 1.0))
            if math.isfinite(dg) and dg > 0.0:
                hint_text += f" | 差值已×增益{dg:g}(Δmean 为增益前波形域均值)"
            else:
                hint_text += " | 差值模式(Δmean 为增益前)"
        self.lbl_denoise_hint.setText(hint_text)
        self._debug_log("DENOISE_HINT", hint_text)


    def _apply_denoise_to_render_traces(
        self,
        traces_in: List[np.ndarray],
        times: np.ndarray,
        render_trace_indices: np.ndarray,
        denoise_trace_indices: np.ndarray,
    ) -> List[np.ndarray]:
        """对当前显示链最终道执行 trace 级去噪。"""
        self._denoise_last_applied_count = 0
        self._denoise_last_delta_mean_abs = 0.0
        self._denoise_last_delta_max_abs = 0.0
        if not bool(self._denoise_params.get("enabled", False)):
            self._denoise_backend_stage = "关闭"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:enabled=0")
            return traces_in
        if not bool(self._denoise_run_armed):
            if self._denoise_frozen_ready and len(self._denoise_frozen_by_trace) > 0:
                out_cached: List[np.ndarray] = []
                hit = 0
                render_ids = np.asarray(render_trace_indices, dtype=int)
                frozen_ids = self._denoise_frozen_trace_set
                for i, tr in enumerate(traces_in):
                    gidx = int(render_ids[i]) if i < render_ids.size else -1
                    if gidx in frozen_ids and gidx in self._denoise_frozen_by_trace:
                        y = np.asarray(self._denoise_frozen_by_trace[gidx], dtype=np.float64)
                        x = np.asarray(tr, dtype=np.float64)
                        if y.shape != x.shape:
                            y = np.resize(y, x.shape)
                        out_cached.append(y)
                        hit += 1
                    else:
                        out_cached.append(np.asarray(tr, dtype=np.float64))
                self._denoise_last_applied_count = int(hit)
                self._denoise_last_delta_mean_abs = float(getattr(self, "_denoise_frozen_delta_mean_abs", 0.0))
                self._denoise_last_delta_max_abs = float(getattr(self, "_denoise_frozen_delta_max_abs", 0.0))
                self._denoise_backend_stage = f"缓存回放 | {hit}/{len(out_cached)}道"
                self._set_denoise_progress(0, -1)
                self._debug_log("DENOISE_RUN", f"replay:frozen hit={hit}/{len(out_cached)}")
                return out_cached
            self._denoise_backend_stage = "待开始(点击开始去噪)"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:run_armed=0")
            return traces_in
        if times.size < 2:
            self._denoise_backend_stage = "跳过: 时间轴不足"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:times<2")
            return traces_in

        dt = float(abs(times[1] - times[0]))
        if (not np.isfinite(dt)) or dt <= 0.0:
            self._denoise_backend_stage = "跳过: dt无效"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", f"skip:dt_invalid dt={dt}")
            return traces_in

        # 交互期间道数较大时先保持流畅，静止后自动回到去噪渲染
        if self._viewport_interacting and len(traces_in) > 300:
            self._denoise_backend_stage = f"交互中跳过({len(traces_in)}道)"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", f"skip:interacting traces={len(traces_in)}")
            return traces_in

        f_s = float(self._denoise_params.get("f_s", 3.0))
        f_e = float(self._denoise_params.get("f_e", 20.0))
        bwconn = int(self._denoise_params.get("bwconn", 8))
        strength = float(self._denoise_params.get("strength", 3.0))
        coh_win = int(self._denoise_params.get("coh_win", 11))
        coh_lag = int(self._denoise_params.get("coh_lag", 2))
        coh_thr = float(self._denoise_params.get("coh_thr", 0.55))
        coh_blend = float(self._denoise_params.get("coh_blend", 0.35))
        coh_penalty = float(self._denoise_params.get("coh_penalty", 0.08))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_min_area = int(self._denoise_params.get("morph_min_area", 24))
        morph_expand = int(self._denoise_params.get("morph_expand", 1))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        pg_enable = bool(self._denoise_params.get("pick_guidance", False))
        pick_wl = float(self._denoise_params.get("pick_wavelet_length_sec", 0.19))
        pick_fl = float(self._denoise_params.get("pick_guidance_floor", 0.12))
        times_full = np.asarray(times, dtype=np.float64).reshape(-1)
        t0_fb = float(times_full[0]) if times_full.size > 0 else 0.0
        return_debug = bool(self._denoise_params.get("return_debug", False))
        return_result = bool(self._denoise_params.get("return_result", False))
        perf_diag = bool(self._denoise_params.get("perf_diag", False))
        active_set = set(int(i) for i in np.asarray(denoise_trace_indices, dtype=int).tolist())
        if len(active_set) == 0:
            self._denoise_backend_stage = "范围内无可去噪道"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:active_set=0")
            return traces_in
        progress_total = int(len(active_set))
        progress_done = 0
        progress_tick = 0
        self._set_denoise_progress(progress_done, progress_total, "去噪计算")
        self._debug_log(
            "DENOISE_RUN",
            f"start(post_ops) render={len(traces_in)} active={len(active_set)} dt={dt:.6f} "
            f"f=({f_s:.3f},{f_e:.3f}) str={strength:.3f} "
            f"coh=(w{coh_win},l{coh_lag},t{coh_thr:.2f},b{coh_blend:.2f},p{coh_penalty:.3f})",
        )

        denoised: List[np.ndarray] = []
        stage_name = "P1-trace"
        denoised_count = 0
        coh_blend_count = 0
        coh_gate_mean_list: List[float] = []
        weak_floor_hit_count = 0
        weak_floor_mean_list: List[float] = []
        delta_mean_list: List[float] = []
        delta_max = 0.0
        render_ids = np.asarray(render_trace_indices, dtype=int)
        active_positions = [i for i, gi in enumerate(render_ids.tolist()) if int(gi) in active_set]
        gidx_in_scope = [int(render_ids[int(p)]) for p in active_positions]
        pg_template_times: List[float] = []
        pick_rows_pg: Optional[List[List[float]]] = None
        if pg_enable:
            pg_template_times = self._denoise_pick_template_times_for_globals(gidx_in_scope)
            if pg_template_times:
                pick_rows_pg = [list(pg_template_times) for _ in active_positions]
        neigh_pos: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        for k, p in enumerate(active_positions):
            p_prev = active_positions[k - 1] if k > 0 else None
            p_next = active_positions[k + 1] if (k + 1) < len(active_positions) else None
            neigh_pos[int(p)] = (p_prev, p_next)
        denoise_base_by_pos: Dict[int, np.ndarray] = {}
        denoise_success_by_pos: Dict[int, bool] = {}
        stage_success_name = stage_name
        fail_type_name: Optional[str] = None
        worker_n = int(max(1, int(self._denoise_params.get("workers", 1))))
        # Adaptive parallel guard:
        # Thread parallelism can be slower on small/medium workloads due to
        # scheduling overhead and potential GIL contention.
        avg_samples = 0.0
        if active_positions:
            try:
                avg_samples = float(
                    np.mean(
                        np.asarray(
                            [np.asarray(traces_in[int(p)], dtype=np.float64).size for p in active_positions],
                            dtype=np.float64,
                        )
                    )
                )
            except Exception:
                avg_samples = 0.0
        allow_thread_parallel = (
            worker_n > 1
            and len(active_positions) >= 512
            and avg_samples >= 2048.0
            and (not self._viewport_interacting)
        )

        def _run_denoise_one(pos: int) -> Tuple[int, np.ndarray, str, bool, Optional[str]]:
            x_local = np.asarray(traces_in[int(pos)], dtype=np.float64)
            ns_loc = int(x_local.size)
            if times_full.size >= ns_loc:
                ta_loc = times_full[:ns_loc].copy()
            else:
                ta_loc = np.arange(ns_loc, dtype=np.float64) * float(dt) + float(t0_fb)
            pt_loc = list(pg_template_times) if pg_enable else []
            try:
                res_local = denoise_trace(
                    x_local,
                    dt=dt,
                    f_s=f_s,
                    f_e=f_e,
                    bwconn=bwconn,
                    strength=strength,
                    morph_enable=morph_enable,
                    morph_quantile=morph_quantile,
                    morph_min_area=morph_min_area,
                    morph_expand=morph_expand,
                    morph_floor_ratio=morph_floor_ratio,
                    morph_keep_strong_quantile=morph_keep_strong_q,
                    morph_bwconn=bwconn,
                    return_debug=return_debug,
                    return_result=return_result,
                    pick_guidance_enable=bool(pg_enable),
                    pick_times=pt_loc,
                    pick_times_axis=ta_loc,
                    pick_t0_fallback=float(t0_fb),
                    pick_wavelet_length_sec=float(pick_wl),
                    pick_guidance_floor=float(pick_fl),
                )
                if hasattr(res_local, "data"):
                    y_local = np.asarray(getattr(res_local, "data"), dtype=np.float64)
                    meta_local = getattr(res_local, "meta", {}) or {}
                    stage_local = str(meta_local.get("stage", "P1-trace"))
                else:
                    y_local = np.asarray(res_local, dtype=np.float64)
                    stage_local = "P1-trace"
                if y_local.shape != x_local.shape:
                    y_local = np.resize(y_local, x_local.shape)
                return int(pos), y_local, stage_local, True, None
            except Exception as exc_local:
                return int(pos), x_local, "", False, type(exc_local).__name__

        section_batch_used = False
        # Structural acceleration path: batch denoise in processors layer.
        if len(active_positions) > 1 and (not return_debug):
            t_batch0 = time.perf_counter() if perf_diag else 0.0
            try:
                def _section_progress(done_i: int, total_i: int) -> None:
                    self._set_denoise_progress(int(done_i), int(total_i), "去噪计算")
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass

                block_in = np.asarray([np.asarray(traces_in[int(p)], dtype=np.float64) for p in active_positions], dtype=np.float64)
                self._set_denoise_progress(0, int(len(active_positions)), "去噪计算")
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
                block_out = denoise_section(
                    block_in,
                    dt=dt,
                    f_s=f_s,
                    f_e=f_e,
                    bwconn=bwconn,
                    strength=strength,
                    workers=worker_n,
                    morph_enable=morph_enable,
                    morph_quantile=morph_quantile,
                    morph_min_area=morph_min_area,
                    morph_expand=morph_expand,
                    morph_floor_ratio=morph_floor_ratio,
                    morph_keep_strong_quantile=morph_keep_strong_q,
                    morph_bwconn=bwconn,
                    return_debug=False,
                    progress_callback=_section_progress,
                    pick_guidance_enable=bool(pg_enable),
                    pick_times_per_row=pick_rows_pg,
                    pick_times_axis=times_full,
                    pick_t0_fallback=float(t0_fb),
                    pick_wavelet_length_sec=float(pick_wl),
                    pick_guidance_floor=float(pick_fl),
                )
                block_out = np.asarray(block_out, dtype=np.float64)
                if block_out.shape == block_in.shape:
                    for k, p in enumerate(active_positions):
                        denoise_base_by_pos[int(p)] = np.asarray(block_out[int(k)], dtype=np.float64)
                        denoise_success_by_pos[int(p)] = True
                    denoised_count = int(len(active_positions))
                    stage_success_name = "P2-section-batch"
                    progress_done = progress_total
                    self._set_denoise_progress(progress_done, progress_total, "去噪计算")
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
                    section_batch_used = True
                    self._debug_log(
                        "DENOISE_RUN",
                        f"section-batch ok active={len(active_positions)} wk={worker_n}",
                    )
                    if perf_diag:
                        self._debug_log(
                            "DENOISE_PERF",
                            f"section_batch_ms={(time.perf_counter() - t_batch0) * 1000.0:.1f} active={len(active_positions)} wk={worker_n}",
                        )
                else:
                    self._debug_log(
                        "DENOISE_RUN",
                        f"section-batch shape-mismatch in={block_in.shape} out={block_out.shape}, fallback trace-loop",
                    )
            except Exception as exc:
                self._debug_log("DENOISE_RUN", f"section-batch-fallback:{type(exc).__name__}")
                if perf_diag:
                    self._debug_log(
                        "DENOISE_PERF",
                        f"section_batch_fail_ms={(time.perf_counter() - t_batch0) * 1000.0:.1f} err={type(exc).__name__}",
                    )

        if (not section_batch_used) and ((not allow_thread_parallel) or len(active_positions) <= 1):
            if worker_n > 1 and len(active_positions) > 1:
                self._debug_log(
                    "DENOISE_RUN",
                    f"parallel->serial adaptive active={len(active_positions)} avg_samples={avg_samples:.1f} wk={worker_n}",
                )
            for pos in active_positions:
                p, y, stg, ok, err_name = _run_denoise_one(int(pos))
                denoise_base_by_pos[p] = np.asarray(y, dtype=np.float64)
                denoise_success_by_pos[p] = bool(ok)
                if ok:
                    denoised_count += 1
                    if stg:
                        stage_success_name = stg
                elif err_name:
                    fail_type_name = str(err_name)
                progress_done += 1
                progress_tick += 1
                if progress_tick >= 8 or progress_done >= progress_total:
                    progress_tick = 0
                    self._set_denoise_progress(progress_done, progress_total, "去噪计算")
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
        elif not section_batch_used:
            max_workers = int(min(worker_n, max(1, len(active_positions))))
            self._debug_log("DENOISE_RUN", f"parallel workers={max_workers} active={len(active_positions)}")
            # Chunked parallelism: reduce per-trace future overhead and context switching.
            # Use smaller chunks for better load balance, avoiding long tail at 75%/80% etc.
            chunk_size = int(math.ceil(len(active_positions) / max(1, max_workers * 4)))
            chunk_size = int(min(16, max(4, chunk_size)))
            chunks: List[List[int]] = [
                [int(v) for v in active_positions[k : k + chunk_size]]
                for k in range(0, len(active_positions), chunk_size)
            ]

            def _run_denoise_chunk(pos_list: List[int]) -> List[Tuple[int, np.ndarray, str, bool, Optional[str]]]:
                out_chunk: List[Tuple[int, np.ndarray, str, bool, Optional[str]]] = []
                for p_local in pos_list:
                    out_chunk.append(_run_denoise_one(int(p_local)))
                return out_chunk

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_denoise_chunk, ch): tuple(ch) for ch in chunks}
                for fut in as_completed(futures):
                    for p, y, stg, ok, err_name in fut.result():
                        denoise_base_by_pos[int(p)] = np.asarray(y, dtype=np.float64)
                        denoise_success_by_pos[int(p)] = bool(ok)
                        if ok:
                            denoised_count += 1
                            if stg:
                                stage_success_name = stg
                        elif err_name:
                            fail_type_name = str(err_name)
                        progress_done += 1
                        progress_tick += 1
                        if progress_tick >= 8 or progress_done >= progress_total:
                            progress_tick = 0
                            self._set_denoise_progress(progress_done, progress_total, "去噪计算")
                            try:
                                QtWidgets.QApplication.processEvents()
                            except Exception:
                                pass

        stage_name = stage_success_name
        if fail_type_name is not None:
            if denoised_count > 0:
                stage_name = f"{stage_name}+部分失败:{fail_type_name}"
            else:
                stage_name = f"部分失败:{fail_type_name}"

        # Layer-4 optimization: precompute shifted traces for active positions and lag set,
        # then reuse in local semblance gating to avoid repeated shift work.
        trace_shift_cache: Dict[int, Dict[int, np.ndarray]] = {}
        if int(coh_lag) > 0 and len(active_positions) > 1:
            lag_vals = list(range(-int(coh_lag), int(coh_lag) + 1))
            est_bytes = int(max(1, len(active_positions))) * int(max(1.0, avg_samples)) * int(len(lag_vals)) * 8
            # Guard memory usage: skip cache when estimated memory is too large.
            if est_bytes <= (256 * 1024 * 1024):
                for p in active_positions:
                    trp = np.asarray(traces_in[int(p)], dtype=np.float64)
                    one_cache: Dict[int, np.ndarray] = {0: trp}
                    for sv in lag_vals:
                        if sv == 0:
                            continue
                        one_cache[int(sv)] = self._shift_trace_samples(trp, int(sv))
                    trace_shift_cache[int(p)] = one_cache
                self._debug_log(
                    "DENOISE_COH",
                    f"shift-cache on active={len(active_positions)} lags={len(lag_vals)} estMB={est_bytes/1048576.0:.1f}",
                )
            else:
                self._debug_log(
                    "DENOISE_COH",
                    f"shift-cache skip active={len(active_positions)} estMB={est_bytes/1048576.0:.1f}",
                )

        def _post_blend_one(pos: int) -> Tuple[int, np.ndarray, bool, float, int, float, Optional[float], float]:
            x_local = np.asarray(traces_in[int(pos)], dtype=np.float64)
            y_local = np.asarray(denoise_base_by_pos.get(int(pos), x_local), dtype=np.float64)
            if y_local.shape != x_local.shape:
                y_local = np.resize(y_local, x_local.shape)
            local_coh_applied = False
            local_gate_mean = 0.0
            local_weak_floor_hit = 0
            local_weak_floor_mean = 0.0
            prev_i, next_i = neigh_pos.get(int(pos), (None, None))
            if prev_i is not None and next_i is not None:
                tr_prev = np.asarray(traces_in[int(prev_i)], dtype=np.float64)
                tr_next = np.asarray(traces_in[int(next_i)], dtype=np.float64)
                prev_cache = trace_shift_cache.get(int(prev_i))
                next_cache = trace_shift_cache.get(int(next_i))
                gate = self._local_semblance_gate(
                    tr_prev=tr_prev,
                    tr_cur=x_local,
                    tr_next=tr_next,
                    max_lag=coh_lag,
                    win=coh_win,
                    coh_thr=coh_thr,
                    lag_penalty=coh_penalty,
                    prev_shift_cache=prev_cache,
                    next_shift_cache=next_cache,
                )
                if gate.size > 0:
                    n_gate = int(min(gate.size, x_local.size, y_local.size))
                    if n_gate > 0:
                        max_blend = float(min(0.90, max(0.0, coh_blend)))
                        alpha = np.clip(gate[:n_gate], 0.0, 1.0) * float(max_blend)
                        # Weak-event protection floor:
                        # For coherent but locally weak samples that are over-suppressed,
                        # raise alpha floor adaptively to preserve valid weak arrivals.
                        y_head = np.asarray(y_local[:n_gate], dtype=np.float64)
                        x_head = np.asarray(x_local[:n_gate], dtype=np.float64)
                        w_e = int(max(5, coh_win))
                        if (w_e % 2) == 0:
                            w_e += 1
                        ker_e = np.ones((w_e,), dtype=np.float64) / float(w_e)
                        ex = np.convolve(x_head * x_head, ker_e, mode="same")
                        ey = np.convolve(y_head * y_head, ker_e, mode="same")
                        rx = np.sqrt(np.maximum(ex, 0.0))
                        ry = np.sqrt(np.maximum(ey, 0.0))
                        eps_e = 1e-12
                        # Low-energy score: lower local amplitude -> higher protection weight.
                        ref = float(np.percentile(rx, 70.0)) if rx.size > 0 else 0.0
                        if ref <= eps_e:
                            ref = float(np.mean(rx) + eps_e)
                        weak_score = np.clip(1.0 - (rx / max(ref, eps_e)), 0.0, 1.0)
                        # Suppression score: more attenuation after denoise -> stronger floor.
                        sup_score = np.clip((rx - ry) / np.maximum(rx, eps_e), 0.0, 1.0)
                        weak_floor_cap = min(0.25, max_blend * 0.80)
                        alpha_floor = weak_floor_cap * np.clip(gate[:n_gate], 0.0, 1.0) * weak_score * sup_score
                        alpha = np.maximum(alpha, alpha_floor)
                        if float(np.max(alpha)) > 1e-6:
                            y_local[:n_gate] = (1.0 - alpha) * y_head + alpha * x_head
                            local_coh_applied = True
                            local_gate_mean = float(np.mean(gate[:n_gate]))
                            local_weak_floor_hit = int(np.sum(alpha_floor > 1e-6))
                            local_weak_floor_mean = float(np.mean(alpha_floor))
            delta_mean_local: Optional[float] = None
            delta_max_local = 0.0
            if bool(denoise_success_by_pos.get(int(pos), False)):
                d_local = np.asarray(y_local - x_local, dtype=np.float64)
                if d_local.size > 0:
                    delta_mean_local = float(np.mean(np.abs(d_local)))
                    try:
                        delta_max_local = float(np.max(np.abs(d_local)))
                    except Exception:
                        delta_max_local = 0.0
            return (
                int(pos),
                y_local,
                bool(local_coh_applied),
                float(local_gate_mean),
                int(local_weak_floor_hit),
                float(local_weak_floor_mean),
                delta_mean_local,
                float(delta_max_local),
            )

        blend_result_by_pos: Dict[int, Tuple[np.ndarray, bool, float, int, float, Optional[float], float]] = {}
        t_blend0 = time.perf_counter() if perf_diag else 0.0
        blend_parallel = (
            worker_n > 1
            and len(active_positions) >= 64
            and (not self._viewport_interacting)
        )
        if blend_parallel:
            blend_workers = int(min(worker_n, max(1, len(active_positions))))
            with ThreadPoolExecutor(max_workers=blend_workers) as pool:
                futures = {pool.submit(_post_blend_one, int(p)): int(p) for p in active_positions}
                for fut in as_completed(futures):
                    pos, y_local, coh_applied, gate_mean, weak_hit, weak_mean, dmean_local, dmax_local = fut.result()
                    blend_result_by_pos[int(pos)] = (
                        np.asarray(y_local, dtype=np.float64),
                        bool(coh_applied),
                        float(gate_mean),
                        int(weak_hit),
                        float(weak_mean),
                        dmean_local,
                        float(dmax_local),
                    )
        else:
            for p in active_positions:
                pos, y_local, coh_applied, gate_mean, weak_hit, weak_mean, dmean_local, dmax_local = _post_blend_one(int(p))
                blend_result_by_pos[int(pos)] = (
                    np.asarray(y_local, dtype=np.float64),
                    bool(coh_applied),
                    float(gate_mean),
                    int(weak_hit),
                    float(weak_mean),
                    dmean_local,
                    float(dmax_local),
                )
        if perf_diag:
            self._debug_log(
                "DENOISE_PERF",
                f"post_blend_ms={(time.perf_counter() - t_blend0) * 1000.0:.1f} active={len(active_positions)} parallel={int(blend_parallel)}",
            )

        for i, tr in enumerate(traces_in):
            x = np.asarray(tr, dtype=np.float64)
            gidx = int(render_ids[i]) if i < render_ids.size else -1
            if gidx not in active_set:
                denoised.append(x)
                continue
            y, coh_applied, gate_mean, weak_hit, weak_mean, dmean_local, dmax_local = blend_result_by_pos.get(
                int(i),
                (np.asarray(denoise_base_by_pos.get(int(i), x), dtype=np.float64), False, 0.0, 0, 0.0, None, 0.0),
            )
            if y.shape != x.shape:
                y = np.resize(y, x.shape)
            if bool(coh_applied):
                coh_blend_count += 1
                coh_gate_mean_list.append(float(gate_mean))
                weak_floor_hit_count += int(weak_hit)
                weak_floor_mean_list.append(float(weak_mean))
            if dmean_local is not None:
                delta_mean_list.append(float(dmean_local))
            if float(dmax_local) > delta_max:
                delta_max = float(dmax_local)
            denoised.append(y)

        if coh_blend_count > 0:
            stage_name = f"{stage_name}+coh"
            gmean = float(np.mean(np.asarray(coh_gate_mean_list, dtype=np.float64))) if coh_gate_mean_list else 0.0
            wmean = float(np.mean(np.asarray(weak_floor_mean_list, dtype=np.float64))) if weak_floor_mean_list else 0.0
            self._debug_log(
                "DENOISE_COH",
                f"coh_blend_count={coh_blend_count}/{denoised_count} gate_mean={gmean:.3f} "
                f"weak_floor_hit={weak_floor_hit_count} weak_floor_mean={wmean:.4f}",
            )
        self._denoise_backend_stage = f"{stage_name} | {denoised_count}/{len(denoised)}道"
        self._denoise_last_applied_count = int(denoised_count)
        if delta_mean_list:
            self._denoise_last_delta_mean_abs = float(np.mean(np.asarray(delta_mean_list, dtype=np.float64)))
        self._denoise_last_delta_max_abs = float(delta_max)
        # 一次性计算完成后冻结结果：后续不重算，直接回放缓存，直到用户再次点击“开始去噪”
        self._denoise_frozen_ready = True
        self._denoise_frozen_trace_set = set(int(i) for i in active_set)
        frozen_map: Dict[int, np.ndarray] = {}
        original_map: Dict[int, np.ndarray] = {}
        render_ids = np.asarray(render_trace_indices, dtype=int)
        for i, tr_raw in enumerate(traces_in):
            gidx_o = int(render_ids[i]) if i < render_ids.size else -1
            if gidx_o >= 0 and gidx_o in active_set:
                original_map[gidx_o] = np.asarray(tr_raw, dtype=np.float64).copy()
        for i, y in enumerate(denoised):
            gidx = int(render_ids[i]) if i < render_ids.size else -1
            if gidx in self._denoise_frozen_trace_set:
                frozen_map[gidx] = np.asarray(y, dtype=np.float64).copy()
        self._denoise_frozen_by_trace = frozen_map
        self._denoise_frozen_original_by_trace = original_map
        self._denoise_run_armed = False
        self._denoise_frozen_delta_mean_abs = float(self._denoise_last_delta_mean_abs)
        self._denoise_frozen_delta_max_abs = float(self._denoise_last_delta_max_abs)
        self._denoise_backend_stage = f"{self._denoise_backend_stage} | 已缓存"
        self._set_denoise_progress(0, -1)
        self._debug_log("DENOISE_RUN", f"done stage={self._denoise_backend_stage}")
        self._debug_log(
            "DENOISE_DELTA",
            f"mean_abs={self._denoise_last_delta_mean_abs:.6e} max_abs={self._denoise_last_delta_max_abs:.6e}",
        )
        return denoised


    def _resolve_denoise_indices(
        self,
        idx_all: np.ndarray,
        idx_visible: np.ndarray,
        idx_render: np.ndarray,
    ) -> np.ndarray:
        """根据去噪范围配置，返回应执行去噪的全局道号集合。"""
        scope = str(self._denoise_params.get("scope", "rendered")).strip().lower()
        if scope == "record":
            out = np.asarray(idx_all, dtype=int)
            self._debug_log("DENOISE_SCOPE", f"scope=record count={int(out.size)}")
            return out
        if scope == "visible":
            out = np.asarray(idx_visible, dtype=int)
            self._debug_log("DENOISE_SCOPE", f"scope=visible count={int(out.size)}")
            return out
        if scope == "selected":
            selected = np.asarray(sorted(int(i) for i in self._denoise_selected_traces), dtype=int)
            if selected.size == 0:
                self._debug_log("DENOISE_SCOPE", "scope=selected count=0")
                return np.asarray([], dtype=int)
            all_set = set(int(i) for i in np.asarray(idx_all, dtype=int).tolist())
            selected = np.asarray([int(i) for i in selected if int(i) in all_set], dtype=int)
            self._debug_log("DENOISE_SCOPE", f"scope=selected count={int(selected.size)}")
            return selected
        out = np.asarray(idx_render, dtype=int)
        self._debug_log("DENOISE_SCOPE", f"scope=rendered count={int(out.size)}")
        return out

