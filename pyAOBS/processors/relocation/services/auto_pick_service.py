"""自动拾取服务：薄封装 zplotpy.AutoPicker + PickManager。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import AutoPickParams


def _get_auto_picker_cls():
    from pyAOBS.visualization.zplotpy.core.auto_picker import AutoPicker

    return AutoPicker


class AutoPickService:
    """对 loaded 道集执行能量比自动拾取，写入 PickManager。"""

    def __init__(self, pick_manager: Any):
        """
        Args:
            pick_manager: zplotpy.PickManager 实例
        """
        self.pick_manager = pick_manager

    def run(
        self,
        loaded: Dict[str, Any],
        params: Optional[AutoPickParams] = None,
    ) -> Tuple[int, str]:
        """
        Returns:
            (成功拾取条数, 状态消息)
        """
        params = params or AutoPickParams()
        traces = loaded.get("traces", [])
        times = np.asarray(loaded.get("times", []), dtype=float)
        offsets = np.asarray(loaded.get("offsets", []), dtype=float)
        if len(traces) == 0 or times.size < 4:
            return 0, "自动拾取失败：数据无效"

        AutoPicker = _get_auto_picker_cls()
        picker = AutoPicker(
            window_length=float(params.window_length),
            min_energy_ratio=float(params.min_energy_ratio),
            search_start=params.search_start,
            search_end=params.search_end,
            vred=float(params.vred),
        )

        indices = params.trace_indices
        if indices is None:
            indices = list(range(len(traces)))

        ok = 0
        pick_word = int(params.pick_word)
        for ig in indices:
            ig = int(ig)
            if ig < 0 or ig >= len(traces):
                continue
            off = float(offsets[ig]) if ig < offsets.size else None
            try:
                result = picker.pick_trace(
                    np.asarray(traces[ig], dtype=float),
                    times,
                    offset=off,
                )
            except Exception:
                continue
            if not result or result.get("pick_time") is None:
                continue
            t = float(result["pick_time"])
            if self.pick_manager.add_pick(ig, t, pick_word):
                ok += 1

        if ok == 0:
            return 0, "自动拾取完成：未找到有效拾取点"
        return ok, f"自动拾取完成：写入 {ok} 个拾取点（apick={pick_word}）"
