# -*- coding: utf-8 -*-
"""Stack display / evaluation mixed into QtFastViewer."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
except Exception:
    pg = None  # type: ignore


class StackMixin:
    """叠加显示、自动时移与叠加评价。"""

    def _on_show_stack_changed(self, _state: int) -> None:
        self.request_render()
        if self.chk_show_stack.isChecked():
            self.lbl_status.setText("叠加道已开启：显示在主图右侧蓝色曲线")
        else:
            self.lbl_status.setText("叠加道已关闭")


    def _compute_stack_auto_shifts(self, trace_indices: np.ndarray) -> Dict[int, float]:
        """为叠加显示计算“自动按拾取对齐”的时移（不改变主波形显示）。"""
        if self.pick_manager is None or trace_indices.size == 0:
            return {}
        apick = int(self.spin_apick.value())
        by_word = self.pick_manager.get_picks_by_word(apick)
        valid_times: List[float] = []
        valid_indices: List[int] = []
        for gidx in trace_indices:
            ig = int(gidx)
            tpk = by_word.get(ig)
            if tpk is None or float(tpk) <= 0:
                continue
            tt = float(tpk)
            if self.static_correction_enabled:
                tt += float(self.static_corrector.get_correction(ig))
            valid_indices.append(ig)
            valid_times.append(tt)
        if len(valid_times) < 2:
            return {}
        ref_time = float(np.median(np.asarray(valid_times, dtype=float)))
        return {ig: (ref_time - tt) for ig, tt in zip(valid_indices, valid_times)}


    def _ensure_stack_item(self) -> None:
        if self._stack_item is None:
            self._stack_item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("stack_pen", "#1478dc"), width=1.5)
            )
            self._stack_item.setZValue(28)
            self.plot.addItem(self._stack_item)


    def _clear_stack_item(self) -> None:
        if self._stack_item is not None:
            self._stack_item.setData([], [])


    def _show_stacking_evaluation(self) -> None:
        if not self.last_stacking_result:
            self.lbl_status.setText("提示：请先执行 F（自适应拾取更新）")
            return
        data = self.last_stacking_result
        original_picks = data.get("original_picks", {})
        updated_picks = data.get("updated_picks", {})
        time_shifts = data.get("time_shifts_by_trace", data.get("time_shifts", []))
        errors = data.get("errors_by_trace", data.get("errors", []))
        traces = data.get("traces", [])
        times = data.get("times", np.array([]))
        offsets = np.asarray(data.get("offsets", []), dtype=float)
        if not isinstance(original_picks, dict) or not isinstance(updated_picks, dict):
            self.lbl_status.setText("叠加评价失败：结果数据无效")
            return
        if len(updated_picks) == 0:
            self.lbl_status.setText("叠加评价失败：无有效更新拾取")
            return
        try:
            eval_result = self.stacking_evaluator.evaluate(
                original_picks=original_picks,
                updated_picks=updated_picks,
                time_shifts=time_shifts,
                errors=errors,
                quality_metric=float(data.get("quality_metric", 0.0) or 0.0),
                traces=list(traces),
                times=np.asarray(times, dtype=float),
            )
            trace_indices = sorted(updated_picks.keys())
            self.stacking_evaluator.create_comparison_plot(eval_result, trace_indices=trace_indices)
            self.stacking_evaluator.create_shift_visualization(eval_result, offsets, trace_indices=trace_indices)
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception as exc:
                self.lbl_status.setText(f"叠加评价图显示失败: {exc}")
                return
            self.lbl_status.setText("叠加评价可视化已显示")
        except Exception as exc:
            self.lbl_status.setText(f"叠加评价失败: {exc}")

