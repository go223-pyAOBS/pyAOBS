# -*- coding: utf-8 -*-
"""精简剖面：wiggle 渲染 + P 拾取 + V 选波（PyQtGraph）。"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..services.models import AttitudeSolution
from ..services.pick_helpers import get_shared_pick, remove_shared_pick, set_shared_pick
from ..services.preview_apply import apply_orientation_to_gather, solution_cache_key
from ..services.waveform_selection import DEFAULT_POST_SEC, DEFAULT_PRE_SEC, WaveformSelectionStore


class SectionCanvas(QWidget):
    """全宽剖面画布。"""

    status_message = Signal(str)
    selection_changed = Signal()
    picks_changed = Signal()
    preview_changed = Signal(bool)  # 旋转预览开关变化

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.loaded: Optional[dict] = None
        self.pick_manager: Any = None
        self.sel_store: Optional[WaveformSelectionStore] = None

        self.apick = 1
        self.vred = 0.0
        self.dscale = 1.0
        self.pick_mode = False
        self.max_traces = 400  # 抽稀上限，保证交互流畅
        self.time_step = 1

        self.mouse_x: Optional[float] = None
        self.mouse_y: Optional[float] = None
        self._last_render_offsets = np.asarray([], dtype=float)
        self._last_render_trace_indices = np.asarray([], dtype=int)
        self._last_processed: List[np.ndarray] = []
        self._last_t_plot = np.asarray([], dtype=float)
        self._last_scale = 1.0

        # 姿态旋转预览（不改写 loaded 原始 traces）
        self._preview_enabled = False
        self._preview_solution: Optional[AttitudeSolution] = None
        self._preview_cache: Optional[dict] = None

        self._curve_items: List[pg.PlotDataItem] = []
        self._pick_item: Optional[pg.ScatterPlotItem] = None
        self._wave_item: Optional[pg.PlotDataItem] = None
        self._wave_marker: Optional[pg.ScatterPlotItem] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "Offset", units="km")
        self.plot.setLabel("left", "Time", units="s")
        self.plot.invertY(True)
        lay.addWidget(self.plot)

        vb = self.plot.getPlotItem().getViewBox()
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # 快捷键挂在 canvas 上（主窗也会再绑一层以保证焦点）
        QShortcut(QKeySequence("P"), self, activated=self.toggle_pick_mode)
        QShortcut(QKeySequence("V"), self, activated=self.add_waveform_selection_at_cursor)
        QShortcut(QKeySequence("Shift+V"), self, activated=self.remove_last_waveform_selection)

        _ = vb  # keep for clarity

    # ---- public API ----
    def set_session(
        self,
        loaded: Optional[dict],
        pick_manager: Any,
        sel_store: WaveformSelectionStore,
    ) -> None:
        self.loaded = loaded
        self.pick_manager = pick_manager
        self.sel_store = sel_store
        self.clear_orientation_preview(keep_solution=False)
        self.render()

    def set_apick(self, apick: int) -> None:
        self.apick = max(1, int(apick))
        if self.sel_store is not None:
            self.sel_store.state.current_apick = self.apick
        self.render()

    def set_vred(self, vred: float) -> None:
        self.vred = float(vred)
        self._update_y_label()
        self.render()

    def set_dscale(self, dscale: float) -> None:
        self.dscale = max(0.05, float(dscale))
        self.render()

    def toggle_pick_mode(self) -> None:
        self.pick_mode = not self.pick_mode
        msg = "拾取模式已开启（左键加/改，右键删）" if self.pick_mode else "拾取模式已关闭"
        self.status_message.emit(msg)

    # ---- orientation preview ----
    @property
    def preview_enabled(self) -> bool:
        return bool(self._preview_enabled)

    @property
    def has_preview_solution(self) -> bool:
        return self._preview_solution is not None

    def clear_orientation_preview(self, keep_solution: bool = False) -> None:
        self._preview_enabled = False
        self._preview_cache = None
        if not keep_solution:
            self._preview_solution = None
        self.preview_changed.emit(False)

    def set_orientation_preview(
        self,
        enabled: bool,
        solution: Optional[AttitudeSolution] = None,
        *,
        rebuild: bool = True,
    ) -> Tuple[bool, str]:
        """
        开启/关闭主图三分量旋转预览。

        Returns:
            (ok, message)
        """
        if solution is not None:
            if self._preview_solution is None or solution.to_dict() != self._preview_solution.to_dict():
                self._preview_cache = None
            self._preview_solution = AttitudeSolution(
                azimuth_deg=float(solution.azimuth_deg),
                tilt_deg=float(solution.tilt_deg),
                dx=float(solution.dx),
                dy=float(solution.dy),
                dz=float(solution.dz),
                time_shift_sec=float(solution.time_shift_sec),
            )

        if enabled:
            if self.loaded is None:
                return False, "请先加载数据"
            if self._preview_solution is None:
                return False, "尚无姿态解，请先运行姿态校正"
            self._preview_enabled = True
            if rebuild or self._preview_cache is None:
                ok, msg = self._rebuild_preview_cache()
                if not ok:
                    self._preview_enabled = False
                    self.preview_changed.emit(False)
                    return False, msg
            self.preview_changed.emit(True)
            self.render()
            sol = self._preview_solution
            return True, (
                f"旋转预览 ON：az={sol.azimuth_deg:.2f}°, tilt={sol.tilt_deg:.2f}°, "
                f"prior={sol.prior_tt_shift_sec:.3f}s, corr={sol.tt_corr_sec:.3f}s, "
                f"final={sol.time_shift_sec:.3f}s"
            )

        self._preview_enabled = False
        self.preview_changed.emit(False)
        self.render()
        return True, "旋转预览 OFF"

    def _rebuild_preview_cache(self) -> Tuple[bool, str]:
        if self.loaded is None or self._preview_solution is None:
            return False, "无法构建预览缓存"
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        headers = list(self.loaded.get("trace_headers") or [])
        if len(traces) == 0 or offsets.size == 0:
            return False, "数据无效"
        key = solution_cache_key(id(self.loaded), len(traces), self._preview_solution)
        if isinstance(self._preview_cache, dict) and self._preview_cache.get("key") == key:
            return True, "使用缓存"
        out_tr, out_off, n_rot = apply_orientation_to_gather(
            traces,
            offsets,
            headers,
            self._preview_solution,
            prefer_utm=True,
            geom="obs",
        )
        if n_rot <= 0:
            return False, "未找到完整三分量组，无法旋转预览"
        self._preview_cache = {
            "key": key,
            "traces": out_tr,
            "offsets": out_off,
            "n_groups": int(n_rot),
        }
        return True, f"已旋转 {n_rot} 组三分量"

    def _active_gather(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """返回当前渲染用 traces/offsets（预览开则用缓存）。"""
        assert self.loaded is not None
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if (
            self._preview_enabled
            and isinstance(self._preview_cache, dict)
            and "traces" in self._preview_cache
        ):
            return list(self._preview_cache["traces"]), np.asarray(
                self._preview_cache["offsets"], dtype=float
            )
        return traces, offsets

    def render(self) -> None:
        if self.loaded is None:
            self._clear_all()
            return
        traces, offsets = self._active_gather()
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size < 2 or offsets.size == 0:
            self._clear_all()
            return

        n = min(len(traces), offsets.size)
        idx = np.arange(n, dtype=int)
        if n > self.max_traces:
            stride = int(np.ceil(n / self.max_traces))
            idx = idx[::stride]

        # 时间抽稀
        step = max(1, int(self.time_step))
        if times.size > 4000:
            step = max(step, int(np.ceil(times.size / 2000)))
        t_plot = times[::step]
        self._last_t_plot = t_plot

        processed: List[np.ndarray] = []
        render_offsets: List[float] = []
        for ig in idx:
            tr = np.asarray(traces[int(ig)], dtype=float)[::step]
            if tr.size != t_plot.size:
                m = min(tr.size, t_plot.size)
                tr = tr[:m]
            amp = float(np.percentile(np.abs(tr), 98)) if tr.size else 0.0
            if amp > 1e-12:
                tr = tr / amp
            processed.append(tr)
            render_offsets.append(float(offsets[int(ig)]))

        render_offsets_arr = np.asarray(render_offsets, dtype=float)
        self._last_render_offsets = render_offsets_arr
        self._last_render_trace_indices = np.asarray(idx, dtype=int)
        self._last_processed = processed

        if render_offsets_arr.size > 1:
            spacing = float(np.median(np.diff(np.sort(render_offsets_arr))))
            if spacing <= 0:
                spacing = 1.0
        else:
            spacing = 1.0
        scale = 0.45 * spacing * float(self.dscale)
        self._last_scale = scale

        wave_pen = "#059669" if self._preview_enabled else "#111827"
        self._ensure_curve_pool(len(processed))
        for i, trd in enumerate(processed):
            gidx = int(idx[i])
            x0 = float(render_offsets_arr[i])
            tshift = self.compute_display_tshift(gidx, x0)
            t_trace = t_plot[: trd.size] + tshift
            xw = x0 + trd * scale
            self._curve_items[i].setPen(pg.mkPen(wave_pen, width=1.0))
            self._curve_items[i].setData(xw, t_trace, connect="finite")
        for j in range(len(processed), len(self._curve_items)):
            self._curve_items[j].setData([], [])

        self._render_wave_selections(processed, render_offsets_arr, idx, t_plot, scale)
        self._render_picks(offsets)
        self._update_y_label()

    # ---- geometry / time ----
    def compute_reduction_tshift(self, trace_idx: int, x_offset: Optional[float] = None) -> float:
        if self.vred <= 0:
            return 0.0
        x = float(x_offset) if x_offset is not None else 0.0
        if x_offset is None and self.loaded is not None:
            offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
            if 0 <= int(trace_idx) < offsets.size:
                x = float(offsets[int(trace_idx)])
        rvred = 1.0 / float(self.vred)
        rvredf = 0.0
        header = (self.loaded or {}).get("header")
        vredf = float(getattr(header, "vredf", 0.0) or 0.0) if header is not None else 0.0
        if vredf > 0.0:
            rvredf = 1.0 / vredf
        return -abs(x) * (rvred - rvredf)

    def compute_display_tshift(self, trace_idx: int, x_offset: Optional[float] = None) -> float:
        tshift = self.compute_reduction_tshift(trace_idx, x_offset)
        if self._preview_enabled and self._preview_solution is not None:
            tshift += float(self._preview_solution.time_shift_sec)
        return tshift

    def nearest_trace(self) -> Optional[Tuple[int, float]]:
        if self.mouse_x is None or self._last_render_offsets.size == 0:
            return None
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - float(self.mouse_x))))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        x_trace = float(self._last_render_offsets[nearest_i])
        return trace_idx, x_trace

    def _headers(self) -> list:
        return list((self.loaded or {}).get("trace_headers") or [])

    # ---- mouse / pick / V ----
    def _on_mouse_moved(self, pos) -> None:
        if self.plot.sceneBoundingRect().contains(pos):
            mouse_pt = self.plot.getPlotItem().vb.mapSceneToView(pos)
            self.mouse_x = float(mouse_pt.x())
            self.mouse_y = float(mouse_pt.y())

    def _on_mouse_clicked(self, event) -> None:
        if not self.pick_mode or self.loaded is None or self.pick_manager is None:
            return
        try:
            pos = event.scenePos()
            if not self.plot.sceneBoundingRect().contains(pos):
                return
            mouse_pt = self.plot.getPlotItem().vb.mapSceneToView(pos)
            self.mouse_x = float(mouse_pt.x())
            self.mouse_y = float(mouse_pt.y())
        except Exception:
            if self.mouse_x is None or self.mouse_y is None:
                return
        hit = self.nearest_trace()
        if hit is None:
            return
        trace_idx, x_trace = hit
        tshift = self.compute_display_tshift(trace_idx, x_trace)
        t_true = float(self.mouse_y) - tshift
        try:
            btn = event.button()
        except Exception:
            btn = Qt.MouseButton.LeftButton
        headers = self._headers()
        if btn == Qt.MouseButton.RightButton:
            ok = remove_shared_pick(self.pick_manager, headers, trace_idx, self.apick)
            self.status_message.emit(
                f"已删除拾取：道 {trace_idx} apick={self.apick}" if ok else "删除失败：无拾取"
            )
        else:
            ok = set_shared_pick(self.pick_manager, headers, trace_idx, self.apick, t_true)
            self.status_message.emit(
                f"拾取：道 {trace_idx} t={t_true:.3f}s apick={self.apick}" if ok else "拾取失败"
            )
        self.picks_changed.emit()
        self.render()

    def add_waveform_selection_at_cursor(self) -> None:
        if self.loaded is None or self.sel_store is None:
            self.status_message.emit("V选波失败：请先加载数据")
            return
        if self.mouse_x is None or self.mouse_y is None:
            self.status_message.emit("V选波失败：请先将鼠标移到剖面")
            return
        hit = self.nearest_trace()
        if hit is None:
            self.status_message.emit("V选波失败：当前无可选道")
            return
        trace_idx, x_trace = hit
        tshift = self.compute_display_tshift(trace_idx, x_trace)
        y_center = float(self.mouse_y)
        center_from_pick = False
        headers = self._headers()
        if self.pick_manager is not None:
            picked = get_shared_pick(self.pick_manager, headers, trace_idx, self.apick)
            if picked is not None and float(picked) > 0.0:
                y_center = float(picked) + tshift
                center_from_pick = True
        t_true = float(y_center - tshift)
        _, replaced = self.sel_store.upsert(
            trace_idx=trace_idx,
            offset=x_trace,
            t_display=y_center,
            t_true=t_true,
            pick_word=self.apick,
        )
        src = "拾取点" if center_from_pick else "鼠标"
        action = "更新" if replaced else "添加"
        t0, t1 = y_center - DEFAULT_PRE_SEC, y_center + DEFAULT_POST_SEC
        self.status_message.emit(
            f"V选波已{action}：道 {trace_idx}，中心={y_center:.3f}s({src})，窗=[{t0:.3f},{t1:.3f}]s"
        )
        self.selection_changed.emit()
        self.render()

    def remove_last_waveform_selection(self) -> None:
        if self.sel_store is None:
            return
        removed = self.sel_store.remove_last_for_apick(self.apick)
        if removed is None:
            self.status_message.emit("Shift+V：当前拾取字下没有可删除的 V 段")
            return
        self.status_message.emit(
            f"已删除最近 V 段：道 {removed.trace_idx}，中心={removed.t_display:.3f}s"
        )
        self.selection_changed.emit()
        self.render()

    def clear_waveform_selections(self) -> None:
        if self.sel_store is None:
            return
        self.sel_store.clear()
        self.selection_changed.emit()
        self.render()
        self.status_message.emit("已清除所有 V 选波窗口")

    # ---- overlays ----
    def _render_picks(self, offsets_all: np.ndarray) -> None:
        self._ensure_pick_item()
        if self.pick_manager is None or self._last_render_trace_indices.size == 0:
            self._pick_item.setData(spots=[])
            return
        headers = self._headers()
        spots = []
        visible = set(int(i) for i in self._last_render_trace_indices.tolist())
        for gidx in list(visible):
            t_raw = get_shared_pick(self.pick_manager, headers, gidx, self.apick)
            if t_raw is None or float(t_raw) <= 0:
                continue
            if gidx < 0 or gidx >= offsets_all.size:
                continue
            x = float(offsets_all[gidx])
            t_display = float(t_raw) + self.compute_display_tshift(gidx, x)
            spots.append(
                {
                    "pos": (x, t_display),
                    "brush": pg.mkBrush(220, 38, 38, 220),
                    "pen": pg.mkPen("#ffffff", width=0.8),
                    "size": 8.0,
                }
            )
        self._pick_item.setData(spots=spots)

    def _render_wave_selections(
        self,
        processed: List[np.ndarray],
        render_offsets: np.ndarray,
        idx_render: np.ndarray,
        t_plot: np.ndarray,
        scale: float,
    ) -> None:
        self._ensure_wave_items()
        if self.sel_store is None:
            self._wave_item.setData([], [])
            self._wave_marker.setData(spots=[])
            return
        sels = self.sel_store.current_apick_selections(self.apick)
        if not sels:
            self._wave_item.setData([], [])
            self._wave_marker.setData(spots=[])
            return

        render_map = {int(g): i for i, g in enumerate(np.asarray(idx_render, dtype=int))}
        seg_x_parts: List[np.ndarray] = []
        seg_y_parts: List[np.ndarray] = []
        markers = []
        for sel in sels:
            gidx = int(sel.trace_idx)
            row = render_map.get(gidx)
            if row is None:
                continue
            t_center = float(sel.t_display)
            x0 = float(render_offsets[row])
            trd = np.asarray(processed[row], dtype=float)
            m = min(trd.size, t_plot.size)
            tshift = self.compute_display_tshift(gidx, x0)
            t_trace = t_plot[:m] + tshift
            x_trace = x0 + trd[:m] * scale
            mask = (t_trace >= (t_center - DEFAULT_PRE_SEC)) & (t_trace <= (t_center + DEFAULT_POST_SEC))
            if np.any(mask):
                seg_x_parts.append(np.concatenate([x_trace[mask], [np.nan]]))
                seg_y_parts.append(np.concatenate([t_trace[mask], [np.nan]]))
            markers.append(
                {
                    "pos": (x0, t_center),
                    "brush": pg.mkBrush("#ef4444"),
                    "pen": pg.mkPen("#ffffff", width=1.0),
                    "size": 8.0,
                }
            )
        if seg_x_parts:
            self._wave_item.setPen(pg.mkPen("#22c55e", width=1.8))
            self._wave_item.setData(np.concatenate(seg_x_parts), np.concatenate(seg_y_parts), connect="finite")
        else:
            self._wave_item.setData([], [])
        self._wave_marker.setData(spots=markers)

    # ---- items ----
    def _ensure_curve_pool(self, n: int) -> None:
        while len(self._curve_items) < n:
            item = pg.PlotDataItem(pen=pg.mkPen("#111827", width=1.0))
            item.setZValue(10)
            self.plot.addItem(item)
            self._curve_items.append(item)

    def _ensure_pick_item(self) -> None:
        if self._pick_item is None:
            self._pick_item = pg.ScatterPlotItem(pxMode=True)
            self._pick_item.setZValue(50)
            self.plot.addItem(self._pick_item)

    def _ensure_wave_items(self) -> None:
        if self._wave_item is None:
            self._wave_item = pg.PlotDataItem()
            self._wave_item.setZValue(30)
            self.plot.addItem(self._wave_item)
        if self._wave_marker is None:
            self._wave_marker = pg.ScatterPlotItem(pxMode=True)
            self._wave_marker.setZValue(31)
            self.plot.addItem(self._wave_marker)

    def _clear_all(self) -> None:
        for c in self._curve_items:
            c.setData([], [])
        if self._pick_item is not None:
            self._pick_item.setData(spots=[])
        if self._wave_item is not None:
            self._wave_item.setData([], [])
        if self._wave_marker is not None:
            self._wave_marker.setData(spots=[])
        self._last_render_offsets = np.asarray([], dtype=float)
        self._last_render_trace_indices = np.asarray([], dtype=int)

    def _update_y_label(self) -> None:
        if self.vred > 0:
            self.plot.setLabel("left", f"t − x/{self.vred:g}", units="s")
        else:
            self.plot.setLabel("left", "Time", units="s")
