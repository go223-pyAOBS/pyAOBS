"""从 V 选波 + 加载数据组装 OrientationObservation。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..orientation_correction import OrientationObservation
from .geometry import GeometryResolver, ITYPE_R, ITYPE_T, ITYPE_Z
from .models import AttitudeUiParams, WaveformSelection
from .waveform_preprocess import preprocess_zrt
from .waveform_selection import WaveformSelectionStore


class OrientationObservationBuilder:
    """无 Qt：loaded + V 段 → OrientationObservation 列表。"""

    def __init__(
        self,
        loaded: Dict[str, Any],
        selection_store: WaveformSelectionStore,
        ui_params: Optional[AttitudeUiParams] = None,
        geometry: Optional[GeometryResolver] = None,
    ):
        self.loaded = loaded
        self.store = selection_store
        self.ui_params = ui_params or AttitudeUiParams()
        self.geometry = geometry or GeometryResolver(loaded)

    def build(
        self,
        apick: Optional[int] = None,
        *,
        all_apicks: bool = True,
    ) -> Tuple[Optional[List[OrientationObservation]], str]:
        """组装观测。

        默认 ``all_apicks=True``：汇总全部 V 段（apick=1 直达 + 其它次生），
        由 ``run_orientation_correction`` 按震相策略分流走时/姿态。
        """
        traces = self.loaded.get("traces", [])
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size < 4:
            return None, "当前数据无效"

        dt = float(times[1] - times[0])
        if dt <= 0:
            return None, "时间采样间隔异常"

        if all_apicks:
            selections = list(self.store.selections)
        else:
            selections = self.store.current_apick_selections(apick)
        if not selections:
            return None, "请先使用 V 键选择至少1段波形（apick=1 直达，其它为次生相）"

        pre_sec = float(self.ui_params.wave_pre)
        post_sec = float(self.ui_params.wave_post)
        obs_list: List[OrientationObservation] = []

        for sel in selections:
            obs, err = self._build_one(sel, traces, times, dt, pre_sec, post_sec)
            if obs is None:
                return None, err
            obs_list.append(obs)
        return obs_list, ""

    def _build_one(
        self,
        sel: WaveformSelection,
        traces: List[Any],
        times: np.ndarray,
        dt: float,
        pre_sec: float,
        post_sec: float,
    ) -> Tuple[Optional[OrientationObservation], str]:
        trace_idx = int(sel.trace_idx)
        group, err = self.geometry.find_3c_group(trace_idx)
        if group is None:
            return None, f"道 {trace_idx} 无法构造三分量：{err}"

        t_ref = float(self.store.t_ref_for(sel))
        t0 = t_ref - pre_sec
        t1 = t_ref + post_sec
        tau = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
        if tau.size < 8:
            return None, f"道 {trace_idx} 截窗采样点不足"

        ztr = np.asarray(traces[int(group[ITYPE_Z])], dtype=float).reshape(-1)
        rtr = np.asarray(traces[int(group[ITYPE_R])], dtype=float).reshape(-1)
        ttr = np.asarray(traces[int(group[ITYPE_T])], dtype=float).reshape(-1)
        z_win = np.interp(tau, times, ztr, left=0.0, right=0.0)
        r_win = np.interp(tau, times, rtr, left=0.0, right=0.0)
        t_win = np.interp(tau, times, ttr, left=0.0, right=0.0)
        # 轻度预处理进反演：rmean/rtrend/可选带通（无增益）
        sr = 1.0 / float(dt) if float(dt) > 0 else 0.0
        z_win, r_win, t_win = preprocess_zrt(
            z_win, r_win, t_win, self.ui_params, sampling_rate=sr
        )

        # 防止配错死道：Z 有能量而 R/T 全零时直接报错，避免校正后看起来“R/T 空”
        def _rms(a: np.ndarray) -> float:
            x = np.asarray(a, dtype=float).reshape(-1)
            return float(np.sqrt(np.mean(x * x))) if x.size else 0.0

        rz, rr, rt = _rms(z_win), _rms(r_win), _rms(t_win)
        if rz > 1e-12 and max(rr, rt) < 1e-8 * rz:
            return (
                None,
                (
                    f"道 {trace_idx} 的 R/T 截窗能量接近 0（相对 Z）。"
                    f"请确认 itypei=2/3 道存在且与 Z 同炮检；"
                    f"RMS Z/R/T={rz:.3g}/{rr:.3g}/{rt:.3g}"
                ),
            )

        z_idx = int(group[ITYPE_Z])
        src, rec = self.geometry.extract_xyz(z_idx)
        src_geo, src_utm = self.geometry.extract_source_coords(z_idx)
        off_km = float(self.geometry.offset_km(z_idx))

        return (
            OrientationObservation(
                trace_idx=int(trace_idx),
                pick_word=int(sel.pick_word),
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
            ),
            "",
        )
