# -*- coding: utf-8 -*-
"""Mute polygon logic mixed into QtFastViewer."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
except Exception:
    pg = None  # type: ignore


class MuteMixin:
    """多边形 mute：编辑、叠加显示与应用到道数据。"""

    def _ensure_mute_polygon_item(self) -> None:
        if self._mute_polygon_item is None:
            self._mute_polygon_item = pg.PlotDataItem(
                pen=pg.mkPen("#0ea5e9", width=2.4, style=QtCore.Qt.PenStyle.DashLine),
            )
            self._mute_polygon_item.setZValue(40)
            self.plot.addItem(self._mute_polygon_item)


    def _ensure_mute_vertex_item(self) -> None:
        if self._mute_vertex_item is None:
            self._mute_vertex_item = pg.ScatterPlotItem(pxMode=True)
            self._mute_vertex_item.setZValue(41)
            self.plot.addItem(self._mute_vertex_item)


    def _clear_mute_polygon_item(self) -> None:
        if self._mute_polygon_item is not None:
            self._mute_polygon_item.setData([], [])
        if self._mute_vertex_item is not None:
            self._mute_vertex_item.setData([], [])


    def _refresh_mute_polygon_overlay(self, update_labels: bool = True) -> None:
        if not self._mute_polygon_points:
            self._clear_mute_polygon_item()
            return
        self._ensure_mute_polygon_item()
        self._ensure_mute_vertex_item()
        pts = np.asarray(self._mute_polygon_points, dtype=float)
        if pts.shape[0] == 1:
            self._mute_polygon_item.setData([pts[0, 0]], [pts[0, 1]])
        elif self._mute_enabled and pts.shape[0] >= 3:
            closed = np.vstack([pts, pts[0]])
            self._mute_polygon_item.setData(closed[:, 0], closed[:, 1])
        else:
            self._mute_polygon_item.setData(pts[:, 0], pts[:, 1])
        # 顶点把手：常态青色，当前选中顶点高亮橙色，便于定位/拖拽
        spots = []
        for i in range(pts.shape[0]):
            selected = (self._mute_selected_vertex_idx is not None and int(self._mute_selected_vertex_idx) == i)
            spots.append(
                {
                    "pos": (float(pts[i, 0]), float(pts[i, 1])),
                    "size": 11.0 if selected else 8.5,
                    "pen": pg.mkPen("#ffffff", width=1.5),
                    "brush": pg.mkBrush("#f59e0b" if selected else "#06b6d4"),
                    "symbol": "o",
                }
            )
        self._mute_vertex_item.setData(spots=spots)


    def _update_mute_status_button(self) -> None:
        if not hasattr(self, "btn_mute_status"):
            return
        if self._mute_edit_mode:
            txt = f"Mute: DRAW({len(self._mute_polygon_points)})"
            color = "#f59e0b"
        elif self._mute_enabled:
            txt = "Mute: ON(反选)" if self._mute_invert else "Mute: ON"
            color = "#22c55e"
        else:
            txt = "Mute: OFF"
            color = "#64748b"
        self.btn_mute_status.setText(txt)
        self.btn_mute_status.setStyleSheet(
            f"QPushButton{{border:1px solid {color}; color:{color}; font-weight:600;}}"
        )
        if hasattr(self, "btn_clear_mute"):
            self.btn_clear_mute.setEnabled(
                len(self._mute_polygon_points) > 0 or self._mute_enabled or self._mute_edit_mode
            )
        if hasattr(self, "chk_mute_invert"):
            self.chk_mute_invert.blockSignals(True)
            try:
                self.chk_mute_invert.setEnabled(bool(self._mute_enabled))
                self.chk_mute_invert.setChecked(bool(self._mute_invert) and bool(self._mute_enabled))
            finally:
                self.chk_mute_invert.blockSignals(False)


    def _on_mute_invert_toggled(self, checked: bool) -> None:
        """Mute 旁「反选」勾选框（与 Shift+M 同源）。"""
        if not self._mute_enabled:
            if hasattr(self, "chk_mute_invert"):
                self.chk_mute_invert.blockSignals(True)
                try:
                    self.chk_mute_invert.setChecked(False)
                    self.chk_mute_invert.setEnabled(False)
                finally:
                    self.chk_mute_invert.blockSignals(False)
            self.lbl_status.setText("反选失败：请先完成 Mute 闭合并应用")
            return
        self._mute_invert = bool(checked)
        self._update_mute_status_button()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            "Mute反选已开启（保留多边形外部）" if self._mute_invert else "Mute反选已关闭（保留多边形内部）"
        )


    def _clear_mute_all(self) -> None:
        """关闭Mute按钮：清空所有 mute（包含顶点）。"""
        had_effect = self._mute_enabled or len(self._mute_polygon_points) > 0 or self._mute_edit_mode
        self._mute_edit_mode = False
        self._mute_enabled = False
        self._mute_invert = False
        self._mute_polygon_points = []
        self._mute_drag_vertex_idx = None
        self._mute_selected_vertex_idx = None
        self._mute_drag_active = False
        self._sync_plot_pan_lock_state()
        self._clear_mute_polygon_item()
        self._update_mute_status_button()
        if had_effect:
            self.request_render(immediate=True)
        self.lbl_status.setText("Mute已关闭并清空（再次按M可重新绘制）")


    def _find_near_mute_vertex(self, x: float, y: float) -> Optional[int]:
        if not self._mute_polygon_points:
            return None
        xr, yr = self.plot.getViewBox().viewRange()
        x_span = max(1e-6, abs(float(xr[1]) - float(xr[0])))
        y_span = max(1e-6, abs(float(yr[1]) - float(yr[0])))
        tol_x = max(1e-6, 0.03 * x_span)
        tol_y = max(1e-6, 0.03 * y_span)
        best_idx: Optional[int] = None
        best_score = float("inf")
        for i, (px, py) in enumerate(self._mute_polygon_points):
            dx = (float(x) - float(px)) / tol_x
            dy = (float(y) - float(py)) / tol_y
            score = dx * dx + dy * dy
            if score < best_score:
                best_score = score
                best_idx = int(i)
        return best_idx if best_score <= 1.0 else None


    def _toggle_mute_polygon_mode(self) -> None:
        """M 键：无多边形时进入编辑；有多边形时切换 mute 开关。"""
        if self.loaded is None:
            self.lbl_status.setText("Mute失败：请先加载数据")
            return
        if self._mute_edit_mode:
            self._mute_edit_mode = False
            self._mute_drag_vertex_idx = None
            self._sync_plot_pan_lock_state()
            self._update_mute_status_button()
            if len(self._mute_polygon_points) >= 3:
                self.lbl_status.setText("Mute编辑已退出")
            else:
                self.lbl_status.setText("Mute编辑已退出（顶点不足3，未应用）")
            return
        # 有已闭合多边形时，M 直接启用/取消 mute 效果（保留多边形）
        if len(self._mute_polygon_points) >= 3:
            self._mute_enabled = not self._mute_enabled
            self._mute_drag_vertex_idx = None
            self._mute_selected_vertex_idx = None
            self._sync_plot_pan_lock_state()
            self._update_mute_status_button()
            self.request_render(immediate=True)
            self.lbl_status.setText("Mute已启用（按M可取消）" if self._mute_enabled else "Mute已取消（按M可恢复）")
            return
        self._mute_edit_mode = True
        self._set_plot_pan_enabled(False)
        self._mute_drag_vertex_idx = None
        # 若尚未形成有效多边形，编辑态默认禁用 mute 效果
        if len(self._mute_polygon_points) < 3:
            self._mute_enabled = False
            self._mute_selected_vertex_idx = None
        self._update_mute_status_button()
        self.lbl_status.setText("Mute编辑已开启：左键选中/拖拽或加点，右键点顶点删除，右键空白闭合应用")


    def _toggle_mute_invert(self) -> None:
        """Shift+M：切换 mute 内外反选。"""
        if not self._mute_enabled:
            self.lbl_status.setText("反选失败：请先完成 Mute 闭合并应用")
            return
        self._on_mute_invert_toggled(not bool(self._mute_invert))


    def _finalize_mute_polygon(self) -> None:
        if len(self._mute_polygon_points) < 3:
            self.lbl_status.setText("Mute闭合失败：至少需要3个顶点")
            return
        self._mute_edit_mode = False
        self._sync_plot_pan_lock_state()
        self._mute_enabled = True
        self._mute_drag_vertex_idx = None
        self._mute_selected_vertex_idx = None
        self._refresh_mute_polygon_overlay()
        self._update_mute_status_button()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            f"Mute已应用：{len(self._mute_polygon_points)} 个顶点（外部不绘制，保留区按当前参数滤波/增益）"
        )


    def _delete_selected_mute_vertex(self) -> None:
        """Del：删除当前选中（或光标附近）顶点，仅在 Mute 绘制编辑态生效。"""
        if not self._mute_edit_mode or not self._mute_polygon_points:
            return
        idx = self._mute_selected_vertex_idx
        if idx is None and self.mouse_x is not None and self.mouse_y is not None:
            idx = self._find_near_mute_vertex(float(self.mouse_x), float(self.mouse_y))
        if idx is None:
            self.lbl_status.setText("Mute删除失败：请先点选一个顶点")
            return
        i = int(idx)
        if i < 0 or i >= len(self._mute_polygon_points):
            return
        self._mute_polygon_points.pop(i)
        self._mute_drag_vertex_idx = None
        if len(self._mute_polygon_points) < 3:
            self._mute_enabled = False
        if not self._mute_polygon_points:
            self._mute_selected_vertex_idx = None
        else:
            self._mute_selected_vertex_idx = min(i, len(self._mute_polygon_points) - 1)
        self._refresh_mute_polygon_overlay()
        self._update_mute_status_button()
        # 右键删点后立即重绘波形，保证 mute 效果及时更新
        self.request_render(immediate=True)
        self.lbl_status.setText(f"Mute编辑：已删除顶点 #{i + 1}，剩余 {len(self._mute_polygon_points)} 个")


    def _build_mute_inside_mask(
        self,
        x_trace: float,
        t_display: np.ndarray,
        polygon: np.ndarray,
    ) -> np.ndarray:
        """计算纵向采样点是否位于 mute 多边形内部（显示坐标系）。"""
        n = int(polygon.shape[0])
        if n < 3 or t_display.size == 0:
            return np.zeros_like(t_display, dtype=bool)
        y_hits: List[float] = []
        for i in range(n):
            x1, y1 = float(polygon[i, 0]), float(polygon[i, 1])
            x2, y2 = float(polygon[(i + 1) % n, 0]), float(polygon[(i + 1) % n, 1])
            if abs(x2 - x1) < 1e-12:
                if abs(x_trace - x1) < 1e-9:
                    y_hits.extend([y1, y2])
                continue
            xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
            if x_trace < xmin or x_trace >= xmax:
                continue
            ratio = (x_trace - x1) / (x2 - x1)
            if 0.0 <= ratio <= 1.0:
                y_hits.append(y1 + ratio * (y2 - y1))
        if len(y_hits) < 2:
            return np.zeros_like(t_display, dtype=bool)
        y_hits.sort()
        mask = np.zeros_like(t_display, dtype=bool)
        for k in range(0, len(y_hits) - 1, 2):
            y0 = float(y_hits[k])
            y1 = float(y_hits[k + 1])
            lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
            mask |= (t_display >= lo) & (t_display <= hi)
        return mask


    def _apply_mute_to_raw_traces(
        self,
        raw_traces: List[np.ndarray],
        trace_indices: np.ndarray,
        render_offsets: np.ndarray,
        times: np.ndarray,
    ) -> List[np.ndarray]:
        """将多边形外部样点临时置零（仅渲染链路）。"""
        if not self._mute_enabled or len(self._mute_polygon_points) < 3:
            return raw_traces
        polygon = np.asarray(self._mute_polygon_points, dtype=float)
        out: List[np.ndarray] = []
        for li, tr in enumerate(raw_traces):
            tr_arr = np.asarray(tr, dtype=float).copy()
            ns = int(min(tr_arr.size, times.size))
            if ns <= 0:
                out.append(tr_arr)
                continue
            trace_idx = int(trace_indices[li])
            x_trace = float(render_offsets[li])
            tshift = float(self._compute_display_tshift(trace_idx, x_trace))
            t_display = np.asarray(times[:ns], dtype=float) + tshift
            inside = self._build_mute_inside_mask(x_trace, t_display, polygon)
            keep_mask = (~inside) if self._mute_invert else inside
            tr_arr[:ns][~keep_mask] = 0.0
            out.append(tr_arr)
        return out

