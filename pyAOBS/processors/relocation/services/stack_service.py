"""V 段自适应叠加：薄封装 zplotpy.AdaptiveStacker。

语义对齐 qt_fast_viewer._run_waveop_stack_from_selections：
- 用 V 中心作 initial_picks
- align_traces 更新 V 基准 t_true（不写回 PickManager）
- 截窗归一化后均值叠加
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import StackResult, WaveformSelection
from .waveform_selection import DEFAULT_POST_SEC, DEFAULT_PRE_SEC, WaveformSelectionStore


def _get_stacker_cls():
    from pyAOBS.visualization.zplotpy.core.adaptive_stack import AdaptiveStacker

    return AdaptiveStacker


class WaveformStackService:
    def __init__(self, selection_store: WaveformSelectionStore):
        self.store = selection_store

    def run(
        self,
        loaded: Dict[str, Any],
        *,
        apick: Optional[int] = None,
        pre_sec: float = DEFAULT_PRE_SEC,
        post_sec: float = DEFAULT_POST_SEC,
        display_tshift_fn=None,
        reduction_tshift_fn=None,
        processed_traces: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Optional[StackResult], str]:
        """
        Args:
            display_tshift_fn: (trace_idx, offset) -> display 时间偏移；None 则 0
            reduction_tshift_fn: (trace_idx, offset) -> 折合时间偏移；None 则 0
            processed_traces: 可选预处理后全道波形；None 用 raw traces
        """
        selections = self.store.current_apick_selections(apick)
        if len(selections) < 2:
            return None, "波形叠加失败：当前拾取字下请先用 V 至少标注2个波形段"

        traces_raw = loaded.get("traces", [])
        times = np.asarray(loaded.get("times", []), dtype=float)
        offsets_all = np.asarray(loaded.get("offsets", []), dtype=float)
        if len(traces_raw) == 0 or times.size < 4 or offsets_all.size == 0:
            return None, "波形叠加失败：当前数据无效"

        dt = float(times[1] - times[0]) if times.size > 1 else 0.001
        t0 = float(times[0])
        tau = np.arange(-float(pre_sec), float(post_sec) + 0.5 * dt, dt, dtype=np.float64)
        if tau.size < 8:
            return None, "波形叠加失败：时间采样不足"

        if display_tshift_fn is None:
            display_tshift_fn = lambda _i, _o: 0.0
        if reduction_tshift_fn is None:
            reduction_tshift_fn = lambda _i, _o: 0.0

        source_traces = processed_traces if processed_traces is not None else traces_raw
        trace_indices: List[int] = []
        sel_refs: List[WaveformSelection] = []
        for sel in selections:
            ig = int(sel.trace_idx)
            if ig < 0 or ig >= len(source_traces) or ig >= offsets_all.size:
                continue
            trace_indices.append(ig)
            sel_refs.append(sel)
        if len(trace_indices) < 2:
            return None, "波形叠加失败：有效 V 段不足"

        selected_traces: List[np.ndarray] = []
        reduction_shifts: List[float] = []
        for li, ig in enumerate(trace_indices):
            red_shift = float(reduction_tshift_fn(int(ig), float(offsets_all[int(ig)])))
            reduction_shifts.append(red_shift)
            tr = np.asarray(source_traces[ig], dtype=float)
            tr_disp = np.interp(times - red_shift, times, tr, left=0.0, right=0.0)
            selected_traces.append(tr_disp)

        initial_picks: List[int] = []
        for li, sel in enumerate(sel_refs):
            t_true = float(self.store.t_ref_for(sel))
            t_display_reduced = t_true + float(reduction_shifts[li])
            ip = int(round((t_display_reduced - t0) / dt))
            initial_picks.append(ip if 0 <= ip < times.size else -1)

        AdaptiveStacker = _get_stacker_cls()
        try:
            result = AdaptiveStacker().align_traces(
                traces=selected_traces,
                times=times,
                initial_picks=initial_picks,
            )
        except Exception as exc:
            return None, f"V段自适应更新失败: {exc}"

        shifts = list(result.get("time_shifts", []))
        if not shifts:
            return None, "V段自适应更新失败：未返回有效偏移"

        updated_count = 0
        segments: List[np.ndarray] = []
        centers: List[float] = []
        for li, ig in enumerate(trace_indices):
            if li >= len(shifts) or initial_picks[li] < 0:
                continue
            old_true = float(sel_refs[li].t_true)
            new_true = float(np.clip(old_true + float(shifts[li]), t0, float(times[-1])))
            new_disp = new_true + float(
                display_tshift_fn(int(ig), float(offsets_all[int(ig)]))
            )
            self.store.update_t_true(sel_refs[li], new_true, t_display=new_disp)
            updated_count += 1

            t_center_display_reduced = new_true + float(reduction_shifts[li])
            seg = np.interp(
                t_center_display_reduced + tau,
                times,
                np.asarray(selected_traces[li], dtype=float),
                left=0.0,
                right=0.0,
            )
            amp = float(np.percentile(np.abs(seg), 98)) if seg.size > 0 else 0.0
            if amp > 1e-12:
                seg = seg / amp
            segments.append(np.asarray(seg, dtype=np.float64))
            centers.append(float(sel_refs[li].t_display))

        if updated_count < 2 or len(segments) < 2:
            return None, "波形叠加失败：可更新/可叠加的 V 段不足"

        stack = np.mean(np.asarray(segments, dtype=np.float64), axis=0)
        out = StackResult(
            tau=tau.astype(float).tolist(),
            stack=np.asarray(stack, dtype=float).tolist(),
            centers=list(centers),
        )
        return out, f"V段流程完成：更新 {updated_count} 条V基准并完成叠加（不写入拾取）"
