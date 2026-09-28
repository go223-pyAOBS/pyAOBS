# -*- coding: utf-8 -*-
"""
道集预览：PySide6 + pyqtgraph（对齐 zplotpy）。

多边形 mute 操作（同 qt_fast_viewer）:
  - 进入绘制后：左键加点/选顶点拖拽，右键点顶点删除，右键空白闭合
  - 中键撤销一点；闭合后左键点顶点可再进编辑
  - 编辑时禁用拖拽平移（避免与加点冲突），滚轮仍可缩放
  - 反选：保留多边形外部

快捷键（对齐 zplotpy）:
  - I：显示鼠标最近道信息
  - Shift+P / 「脉冲成像」：进入脉冲拾取模式，左键点选
  - Esc：退出脉冲拾取
  - ←/→：浏览当前炮（琥珀临时选）
  - Shift+←/→ / Shift+拖动：多选加入临时集（琥珀，不追加）
  - 右键：把临时所选追加到手选（红波形）

手选（非 mute 编辑；mute/预处理页关闭叠画）:
  - 左键：单选浏览（琥珀；保留已追加红波形）
  - 右键：仅追加已左键/Shift 临时选（红）；落点不参与选取
  - 中键：取消当前选择（清临时选；若在已追加中则移出）
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore as pgQtCore
except ImportError as exc:  # pragma: no cover
    pg = None  # type: ignore
    pgQtCore = None  # type: ignore
    _PG_ERR = exc
else:
    _PG_ERR = None

Point = Tuple[float, float]


class GatherCanvas(QWidget):
    status = Signal(str)
    mute_changed = Signal()
    # 非 mute：道下标 + browse|pending_add|append|cancel
    trace_picked = Signal(int, str)
    # ←/→：delta；shift_multi=是否加入临时多选（不追加）
    navigate_shot = Signal(int, bool)
    # 脉冲拾取模式：左键点选后发出取样 dict
    impulse_point_picked = Signal(object)
    # 进入/退出脉冲模式时通知（便于按钮文案）
    impulse_mode_changed = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._data: Optional[np.ndarray] = None
        self._d1 = 0.004
        self._o1 = 0.0
        self._pclip = 98.0
        self._dscale = 1.0
        self._display_vred = 0.0  # 显示折合 km/s；0=关
        self._display_mode = "wiggle"  # wiggle | fill+ | fill- | density
        self._title = ""
        self._x_coords: Optional[np.ndarray] = None  # 每道 x（trace 或 model_x km）
        self._x_label = "Trace"
        # 折合/|x| 参考原点（OBS x）；横轴为 model distance 时用
        self._x_reduce_origin: float = 0.0
        # 全炮典型道间距；稀疏手选时 wiggle/density 定宽用它，避免按大空隙拉宽
        self._nominal_dx: Optional[float] = None
        # 强制图窗 X（如定稿预览用全炮 model x 范围）；None=按当前道自适应
        self._view_x_range: Optional[Tuple[float, float]] = None
        self._highlight_idx: Optional[int] = None
        self._selected_shot_ids: List[int] = []  # 统一炮集：剖面多炮高亮
        self._trace_labels: List[str] = []  # 每道标签（如 shot 路径）
        self._wiggle_item = None
        self._hl_curve = None
        self._sel_curve = None
        self._shade_item = None
        self._last_apply_mute = True
        self._mute_tp = 0.15  # 多边形边缘过渡(s)，与速度 mute tp 共用
        self._auto_range = True
        self._view_initialized = False
        self._saved_view = None  # ((x0,x1),(y0,y1))
        self.mouse_x: Optional[float] = None
        self.mouse_y: Optional[float] = None
        self._trace_info_dlg: Optional[QDialog] = None
        self._hand_pick_enabled = True
        self._overlays_enabled = True  # mute/预处理时关：不画红/琥珀
        self._last_shift_shot: Optional[int] = None
        self._pending_shot_ids: List[int] = []  # Shift/左键临时选（琥珀）

        self._mute_points: List[Point] = []
        self._mute_edit = False
        self._mute_enabled = False
        self._mute_invert = False
        self._drag_idx: Optional[int] = None
        self._sel_idx: Optional[int] = None
        self._dragging = False
        self._impulse_mode = False
        self._impulse_pick: Optional[dict] = None
        self._impulse_marker = None

        if pg is None:
            self.plot = None
            self._img = None
            self._poly_item = None
            self._vtx_item = None
            self._hl_region = None
            self._hl_line = None
            self._wiggle_item = None
            self._hl_curve = None
            self._shade_item = None
            self._impulse_marker = None
            self._hint = QLabel(
                "需要 pyqtgraph（与 zplotpy 相同）:\n  pip install pyqtgraph\n%s"
                % (_PG_ERR,)
            )
            self._hint.setWordWrap(True)
            lay.addWidget(self._hint)
            return

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
        self.plot = pg.PlotWidget(background="w")
        # 关闭 pyqtgraph 右键菜单，避免与 mute 右键闭合冲突
        self.plot.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.plot.getPlotItem().setMenuEnabled(False)
        self.plot.getViewBox().setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Time", units="s")
        self.plot.invertY(True)
        self._img = pg.ImageItem()
        self._img.setLookupTable(self._gray_lut())
        self.plot.addItem(self._img)
        # 全部 wiggle 合并为 1 条曲线（NaN 分隔），显著快于数百个 PlotDataItem
        self._wiggle_item = pg.PlotDataItem(
            pen=pg.mkPen("#111111", width=1), connect="finite"
        )
        self._wiggle_item.setZValue(6)
        self.plot.addItem(self._wiggle_item)
        # 临时选（左键/Shift）：琥珀波形
        self._hl_curve = pg.PlotDataItem(
            pen=pg.mkPen("#f59e0b", width=2.0), connect="finite"
        )
        self._hl_curve.setZValue(7)
        self.plot.addItem(self._hl_curve)
        self._shade_item = pg.PlotDataItem(
            pen=pg.mkPen("#404040", width=1),
            connect="pairs",
        )
        self._shade_item.setZValue(5)
        self.plot.addItem(self._shade_item)
        # 选中炮高亮：半透明色带 + 竖线
        self._hl_region = pg.LinearRegionItem(
            values=(0, 0),
            orientation="vertical",
            brush=pg.mkBrush(245, 158, 11, 55),
            pen=pg.mkPen("#f59e0b", width=0),
            movable=False,
        )
        self._hl_region.setZValue(8)
        self._hl_region.setVisible(False)
        self.plot.addItem(self._hl_region)
        self._hl_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("#f59e0b", width=1.8),
            movable=False,
        )
        self._hl_line.setZValue(9)
        self._hl_line.setVisible(False)
        self.plot.addItem(self._hl_line)
        # 已追加手选炮：红色波形高亮（与当前浏览琥珀波形区分）
        self._sel_curve = pg.PlotDataItem(
            pen=pg.mkPen("#dc2626", width=1.8), connect="finite"
        )
        self._sel_curve.setZValue(8.5)
        self.plot.addItem(self._sel_curve)
        self._poly_item = pg.PlotDataItem(
            pen=pg.mkPen("#0ea5e9", width=2.4, style=Qt.PenStyle.DashLine)
        )
        self._poly_item.setZValue(20)
        self._vtx_item = pg.ScatterPlotItem(pxMode=True)
        self._vtx_item.setZValue(21)
        self.plot.addItem(self._poly_item)
        self.plot.addItem(self._vtx_item)
        self._impulse_marker = pg.ScatterPlotItem(
            pxMode=True,
            size=16,
            symbol="+",
            pen=pg.mkPen("#e11d48", width=2.2),
            brush=pg.mkBrush("#e11d48"),
        )
        self._impulse_marker.setZValue(25)
        self.plot.addItem(self._impulse_marker)
        lay.addWidget(self.plot)

        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_moved)
        # 释放鼠标；viewport 上接滚轮（mute 编辑时 ViewBox 鼠标关闭仍可缩放）
        self.plot.scene().installEventFilter(self)
        self.plot.viewport().installEventFilter(self)

        sc_m = QShortcut(QKeySequence("M"), self)
        sc_m.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_m.activated.connect(self._on_m_shortcut)
        sc_inv = QShortcut(QKeySequence("Shift+M"), self)
        sc_inv.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_inv.activated.connect(self._on_shift_m_shortcut)
        sc_del = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        sc_del.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_del.activated.connect(self._on_del_shortcut)
        sc_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        sc_esc.activated.connect(self._on_escape)
        sc_i = QShortcut(QKeySequence("I"), self)
        # 主窗口激活即可（不必点进画布再聚焦）
        sc_i.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_i.activated.connect(self._on_i_shortcut)
        sc_imp = QShortcut(QKeySequence("Shift+P"), self)
        sc_imp.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_imp.activated.connect(self._on_impulse_shortcut)
        for seq, delta, shift_multi in (
            ("Left", -1, False),
            ("Right", 1, False),
            ("Shift+Left", -1, True),
            ("Shift+Right", 1, True),
        ):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(
                lambda d=delta, s=shift_multi: self._emit_nav_if_idle(int(d), bool(s))
            )

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "左键/Shift 琥珀选道；右键仅追加已选（不按落点）；中键取消；"
            "I 道信息；脉冲成像按钮/Shift+P 进入拾取，左键点选，Esc 取消"
        )

    def set_hand_pick_enabled(self, on: bool) -> None:
        self._hand_pick_enabled = bool(on)
        if not self._hand_pick_enabled:
            self._last_shift_shot = None
            self._pending_shot_ids = []
            self._refresh_pending_marks()

    def hand_pick_enabled(self) -> bool:
        return bool(self._hand_pick_enabled)

    def set_overlays_enabled(self, on: bool) -> None:
        """mute 编辑 / 预处理页关闭红琥珀叠画。"""
        self._overlays_enabled = bool(on)
        if not self._overlays_enabled:
            if self._hl_curve is not None:
                self._hl_curve.setData([], [])
            if self._sel_curve is not None:
                self._sel_curve.setData([], [])
            if self._hl_line is not None:
                self._hl_line.setVisible(False)
        else:
            self._update_highlight_wiggle()
            self._refresh_highlight()

    def overlays_enabled(self) -> bool:
        """mute 编辑/已启用时不用红琥珀高亮波形。"""
        return (
            bool(self._overlays_enabled)
            and not bool(self._mute_edit)
            and not self.mute_enabled()
        )

    def set_pending_shot_ids(self, ids: Optional[Sequence[int]]) -> None:
        out: List[int] = []
        seen = set()
        for i in ids or []:
            try:
                v = int(i)
            except (TypeError, ValueError):
                continue
            if v not in seen:
                seen.add(v)
                out.append(v)
        self._pending_shot_ids = out
        self._refresh_pending_marks()

    def add_pending_shot_id(self, sid: int) -> None:
        v = int(sid)
        if v not in self._pending_shot_ids:
            self._pending_shot_ids.append(v)
        self._refresh_pending_marks()

    def pending_shot_ids(self) -> List[int]:
        return list(self._pending_shot_ids)

    def clear_pending_shot_ids(self) -> None:
        self._pending_shot_ids = []
        self._refresh_pending_marks()

    @staticmethod
    def _gray_lut() -> np.ndarray:
        x = np.linspace(0, 255, 256, dtype=np.uint8)
        return np.column_stack([x, x, x])

    # ----- mute state API -----
    def mute_points(self) -> List[Point]:
        return list(self._mute_points)

    def set_mute_points(self, pts: List[Point], *, enabled: bool = False, invert: bool = False) -> None:
        self._mute_points = [(float(a), float(b)) for a, b in pts]
        self._mute_enabled = bool(enabled) and len(self._mute_points) >= 3
        self._mute_invert = bool(invert)
        self._refresh_overlay()
        self.mute_changed.emit()

    def mute_enabled(self) -> bool:
        return self._mute_enabled and len(self._mute_points) >= 3

    def set_mute_tp(self, tp: float) -> None:
        """多边形 mute 边缘过渡（秒），与速度 mute tp 一致。"""
        try:
            self._mute_tp = max(0.0, float(tp))
        except (TypeError, ValueError):
            self._mute_tp = 0.15

    def set_mute_enabled(self, enabled: bool) -> None:
        """开关 mute（内存保留顶点）。关闭：恢复剖面且不画顶点；开启：重绘顶点+mute。"""
        want = bool(enabled) and len(self._mute_points) >= 3
        self._mute_edit = False
        self._drag_idx = None
        self._sel_idx = None
        self._set_pan(True)
        self._mute_enabled = want
        self._last_apply_mute = bool(want)
        self._refresh_overlay()
        if self._data is not None:
            self._redraw_image(apply_mute=want)
        self.mute_changed.emit()

    def mute_invert(self) -> bool:
        return self._mute_invert

    def clear_mute(self) -> None:
        self._mute_edit = False
        self._mute_enabled = False
        self._mute_invert = False
        self._mute_points = []
        self._drag_idx = None
        self._sel_idx = None
        self._set_pan(True)
        self._refresh_overlay()
        self._redraw_image()
        self.status.emit("Mute 已清空")
        self.mute_changed.emit()

    def toggle_mute_mode(self) -> None:
        """M：无多边形进绘制；有多边形则开/关 mute；绘制中再按 M 退出编辑。"""
        if self.plot is None or self._data is None:
            self.status.emit("Mute：请先加载道集")
            return
        if self._mute_edit:
            self.exit_edit_mode()
            return
        if len(self._mute_points) >= 3:
            self._mute_enabled = not self._mute_enabled
            self._drag_idx = None
            self._sel_idx = None
            self._set_pan(True)
            self._refresh_overlay()
            self._redraw_image()
            if self._mute_enabled and (
                self._impulse_mode or self._impulse_pick is not None
            ):
                self.exit_impulse_mode(clear_marker=True)
            self.status.emit(
                "Mute 已启用（M 取消）" if self._mute_enabled else "Mute 已取消（M 恢复）"
            )
            self.mute_changed.emit()
            return
        # 进入 mute 绘制：与脉冲拾取互斥
        if self._impulse_mode or self._impulse_pick is not None:
            self.exit_impulse_mode(clear_marker=True)
        self._mute_edit = True
        self._mute_enabled = False
        self._set_pan(False)
        self._refresh_overlay()
        # mute 时关掉手选红/琥珀叠画
        if self._hl_curve is not None:
            self._hl_curve.setData([], [])
        if self._sel_curve is not None:
            self._sel_curve.setData([], [])
        self.status.emit(
            "Mute 绘制：左键加点/拖顶点，右键删点或闭合，中键撤销，滚轮缩放"
        )
        self.mute_changed.emit()

    def exit_edit_mode(self) -> None:
        if not self._mute_edit:
            return
        self._mute_edit = False
        self._drag_idx = None
        self._set_pan(True)
        self._refresh_overlay()
        self._refresh_highlight()
        if len(self._mute_points) >= 3:
            self.status.emit("Mute 编辑已退出")
        else:
            self.status.emit("Mute 编辑已退出（顶点不足 3）")
        self.mute_changed.emit()

    def _on_escape(self) -> None:
        if self._impulse_mode:
            self.exit_impulse_mode(clear_marker=True)
            return
        self.exit_edit_mode()

    def impulse_mode(self) -> bool:
        return bool(self._impulse_mode)

    def enter_impulse_mode(self) -> None:
        """进入脉冲拾取：左键点选 (道,时间)，Esc 取消。"""
        if self._data is None:
            self.status.emit("脉冲成像：请先加载道集")
            return
        if self._mute_edit:
            self.exit_edit_mode()
        self._impulse_mode = True
        self._set_pan(True)
        self.status.emit("脉冲拾取：左键点击道集上一点；Esc 取消")
        self.impulse_mode_changed.emit(True)

    def exit_impulse_mode(self, *, clear_marker: bool = False) -> None:
        was = bool(self._impulse_mode)
        self._impulse_mode = False
        if clear_marker:
            self._impulse_pick = None
            self._set_impulse_marker(None, None)
        if was:
            self.status.emit("已退出脉冲拾取")
            self.impulse_mode_changed.emit(False)

    def _set_impulse_marker(self, x: Optional[float], t_disp: Optional[float]) -> None:
        if self._impulse_marker is None:
            return
        if x is None or t_disp is None:
            self._impulse_marker.setData([], [])
            return
        self._impulse_marker.setData([float(x)], [float(t_disp)])

    def toggle_invert(self) -> None:
        if not self.mute_enabled():
            self.status.emit("反选失败：请先闭合并启用 Mute")
            return
        self._mute_invert = not self._mute_invert
        self._redraw_image()
        self.status.emit(
            "Mute 反选 ON（保留外部）" if self._mute_invert else "Mute 反选 OFF（保留内部）"
        )
        self.mute_changed.emit()

    def delete_selected_vertex(self) -> None:
        if not self._mute_edit or not self._mute_points:
            return
        idx = self._sel_idx
        if idx is None:
            self.status.emit("请先点选一个顶点")
            return
        i = int(idx)
        if i < 0 or i >= len(self._mute_points):
            return
        self._mute_points.pop(i)
        self._drag_idx = None
        if len(self._mute_points) < 3:
            self._mute_enabled = False
        self._sel_idx = None if not self._mute_points else min(i, len(self._mute_points) - 1)
        self._refresh_overlay()
        # 编辑中不 mute
        self._redraw_image(apply_mute=False)
        self.status.emit("已删顶点，剩余 %d" % len(self._mute_points))
        self.mute_changed.emit()

    def finalize_polygon(self) -> None:
        if len(self._mute_points) < 3:
            self.status.emit("闭合失败：至少 3 个顶点")
            return
        if self._impulse_mode or self._impulse_pick is not None:
            self.exit_impulse_mode(clear_marker=True)
        self._mute_edit = False
        self._mute_enabled = True
        self._drag_idx = None
        self._sel_idx = None
        self._set_pan(True)
        self._refresh_overlay()
        # 本地叠多边形 mute；_data 保持未 bake 多边形的底图
        self._last_apply_mute = True
        self._redraw_image(apply_mute=True)
        # mute 启用后不叠手选高亮波形
        self._refresh_highlight()
        self.status.emit("Mute 已应用：%d 顶点（Shift+M 反选）" % len(self._mute_points))
        self.mute_changed.emit()

    # ----- display -----
    def clear(self, msg: str = "", *, reset_view: bool = True) -> None:
        if self.plot is None:
            return
        if not reset_view and self._view_initialized:
            self._capture_view()
        self._data = None
        self._highlight_idx = None
        self._trace_labels = []
        if reset_view:
            self._auto_range = True
            self._view_initialized = False
            self._saved_view = None
        self._img.clear()
        if self._wiggle_item is not None:
            self._wiggle_item.setData([], [])
        if self._hl_curve is not None:
            self._hl_curve.setData([], [])
        if self._sel_curve is not None:
            self._sel_curve.setData([], [])
        if self._shade_item is not None:
            self._shade_item.setData([], [])
        if self._hl_region is not None:
            self._hl_region.setVisible(False)
        if self._hl_line is not None:
            self._hl_line.setVisible(False)
        if self._impulse_mode:
            self.exit_impulse_mode(clear_marker=True)
        else:
            self._impulse_pick = None
            self._set_impulse_marker(None, None)
        self.plot.setTitle(msg or "")

    def show_gather(
        self,
        data: np.ndarray,
        *,
        d1: float = 0.004,
        o1: float = 0.0,
        title: str = "",
        pclip: float = 98.0,
        dscale: float = 1.0,
        display_mode: str = "wiggle",
        display_vred: float = 0.0,
        x_coords: Optional[np.ndarray] = None,
        x_label: str = "Trace",
        apply_mute_preview: bool = True,
        highlight_idx: Optional[int] = None,
        reset_view: bool = False,
        trace_labels: Optional[Sequence[str]] = None,
        nominal_dx: Optional[float] = None,
        view_x_range: Optional[Tuple[float, float]] = None,
        x_reduce_origin: float = 0.0,
    ) -> None:
        if self.plot is None:
            return
        # 保留当前缩放/平移（增益刷新、换炮等不要弹回全图）
        keep_view = bool(self._view_initialized) and not reset_view
        # clear(reset_view=False) 已保存视窗时勿在空图上再 capture
        if keep_view and self._data is not None:
            self._capture_view()
        self._data = np.asarray(data, dtype=np.float32).copy()
        self._d1 = float(d1)
        self._o1 = float(o1)
        self._pclip = float(pclip)
        self._dscale = max(0.05, float(dscale))
        self._display_vred = max(0.0, float(display_vred))
        try:
            self._x_reduce_origin = float(x_reduce_origin)
        except (TypeError, ValueError):
            self._x_reduce_origin = 0.0
        mode = str(display_mode or "density").lower()
        if mode not in ("wiggle", "fill+", "fill-", "density"):
            mode = "density"
        self._display_mode = mode
        self._title = title or ""
        self._x_label = x_label
        try:
            nd = float(nominal_dx) if nominal_dx is not None else 0.0
        except (TypeError, ValueError):
            nd = 0.0
        self._nominal_dx = nd if nd > 1e-12 else None
        if view_x_range is not None and len(view_x_range) >= 2:
            try:
                a, b = float(view_x_range[0]), float(view_x_range[1])
                self._view_x_range = (min(a, b), max(a, b))
            except (TypeError, ValueError):
                self._view_x_range = None
        else:
            self._view_x_range = None
        ntr = self._data.shape[1]
        if x_coords is None:
            self._x_coords = np.arange(ntr, dtype=float)
        else:
            self._x_coords = np.asarray(x_coords, dtype=float)
            if self._x_coords.size < ntr:
                self._x_coords = np.pad(
                    self._x_coords, (0, ntr - self._x_coords.size), mode="edge"
                )
        if trace_labels is not None:
            self._trace_labels = [str(s) for s in list(trace_labels)[:ntr]]
            if len(self._trace_labels) < ntr:
                self._trace_labels.extend(
                    [""] * (ntr - len(self._trace_labels))
                )
        else:
            self._trace_labels = []
        self._highlight_idx = (
            int(highlight_idx)
            if highlight_idx is not None and 0 <= int(highlight_idx) < ntr
            else None
        )
        self.plot.setLabel("bottom", self._x_label)
        self._update_time_label()
        self._auto_range = not keep_view
        self._last_apply_mute = bool(apply_mute_preview)
        self._redraw_image(apply_mute=apply_mute_preview)
        self._refresh_overlay()
        self._refresh_highlight()
        if keep_view:
            self._restore_view()
        else:
            self._view_initialized = True

    def set_highlight_idx(self, idx: Optional[int]) -> None:
        """仅更新选中道高亮（不重载数据 / 不重算增益 / 不改视窗）。

        Density 模式下叠画该道真实 wiggle（非竖线），换炮只改这一条曲线。
        """
        ntr = 0 if self._data is None else self._data.shape[1]
        new_idx = int(idx) if idx is not None and 0 <= int(idx) < ntr else None
        if new_idx == self._highlight_idx:
            return
        self._highlight_idx = new_idx
        try:
            self._update_highlight_wiggle()
        except Exception:
            pass
        self._refresh_highlight()

    def update_display_style(
        self,
        *,
        dscale: Optional[float] = None,
        pclip: Optional[float] = None,
        display_mode: Optional[str] = None,
        display_vred: Optional[float] = None,
        title: Optional[str] = None,
    ) -> None:
        """轻量改显示参数并重绘（保留视窗）。"""
        if self._data is None:
            return
        self._capture_view()
        if dscale is not None:
            self._dscale = max(0.05, float(dscale))
        if pclip is not None:
            self._pclip = float(pclip)
        if display_vred is not None:
            self._display_vred = max(0.0, float(display_vred))
        if display_mode is not None:
            mode = str(display_mode).lower()
            if mode in ("wiggle", "fill+", "fill-", "density"):
                self._display_mode = mode
        if title is not None:
            self._title = title
        self._update_time_label()
        self._auto_range = False
        self._redraw_image(apply_mute=self._last_apply_mute)
        self._refresh_highlight()
        self._restore_view()

    def shift_model_x(self, delta: float, *, shift_polygon: bool = True) -> None:
        """
        OBS x 平移后热更新横轴：道振幅不变，只平移 model x / 折合原点 / 视窗。
        ``shift_polygon``：多边形以 model x 为横轴时一并平移顶点。
        """
        try:
            d = float(delta)
        except (TypeError, ValueError):
            return
        if abs(d) < 1e-15:
            return
        self._capture_view()
        if self._saved_view is not None:
            (x0, x1), (y0, y1) = self._saved_view
            self._saved_view = ((x0 + d, x1 + d), (y0, y1))
        if self._x_coords is not None:
            self._x_coords = np.asarray(self._x_coords, dtype=float) + d
        self._x_reduce_origin = float(self._x_reduce_origin) + d
        if self._view_x_range is not None:
            a, b = self._view_x_range
            self._view_x_range = (float(a) + d, float(b) + d)
        if shift_polygon and self._mute_points:
            self._mute_points = [
                (float(px) + d, float(py)) for px, py in self._mute_points
            ]
        if self._data is None:
            self._refresh_overlay()
            return
        self._auto_range = False
        self._redraw_image(apply_mute=self._last_apply_mute)
        self._refresh_highlight()
        self._refresh_overlay()
        self._restore_view()

    def _capture_view(self) -> None:
        if self.plot is None:
            self._saved_view = None
            return
        try:
            xr, yr = self.plot.getViewBox().viewRange()
            self._saved_view = (
                (float(xr[0]), float(xr[1])),
                (float(yr[0]), float(yr[1])),
            )
        except Exception:
            self._saved_view = None

    def _restore_view(self) -> None:
        if self.plot is None or self._saved_view is None:
            return
        try:
            (x0, x1), (y0, y1) = self._saved_view
            self.plot.setXRange(x0, x1, padding=0)
            self.plot.setYRange(y0, y1, padding=0)
        except Exception:
            pass

    def _update_time_label(self) -> None:
        if self.plot is None:
            return
        if self._display_vred > 0:
            self.plot.setLabel(
                "left",
                "t-|x-xobs|/%.3g" % self._display_vred,
                units="s",
            )
        else:
            self.plot.setLabel("left", "Time", units="s")

    def _reduction_tshift(self, x: float) -> float:
        """折合：t' = t - |x - x_obs|/vred（横轴为 model distance 时相对 OBS）。"""
        if self._display_vred <= 0:
            return 0.0
        return -abs(float(x) - float(self._x_reduce_origin)) / float(
            self._display_vred
        )

    def _trace_x_span(self, idx: int) -> Tuple[float, float, float]:
        """返回 (x_left, x_center, x_right) 用于高亮一条道。"""
        assert self._data is not None and self._x_coords is not None
        ntr = self._data.shape[1]
        xs = self._x_coords[:ntr]
        xc = float(xs[idx])
        if ntr == 1:
            x0, w, x1 = self._x_extent()
            return x0, xc, x1
        if idx == 0:
            half = 0.5 * abs(float(xs[1] - xs[0]))
            return xc - half, xc, xc + half
        if idx == ntr - 1:
            half = 0.5 * abs(float(xs[-1] - xs[-2]))
            return xc - half, xc, xc + half
        left = 0.5 * (float(xs[idx - 1]) + xc)
        right = 0.5 * (xc + float(xs[idx + 1]))
        return left, xc, right

    def set_selected_shot_ids(self, ids: Optional[Sequence[int]]) -> None:
        """统一炮集炮号 → 剖面竖线高亮（按 trace_labels 匹配）。"""
        out: List[int] = []
        seen = set()
        for i in ids or []:
            try:
                v = int(i)
            except (TypeError, ValueError):
                continue
            if v not in seen:
                seen.add(v)
                out.append(v)
        self._selected_shot_ids = out
        self._refresh_selection_marks()

    def selected_shot_ids(self) -> List[int]:
        return list(self._selected_shot_ids)

    def _trace_indices_for_shot_ids(self, ids: Sequence[int]) -> List[int]:
        """shot 炮号 → 道下标列表（montage 每炮一道时通常 1:1）。"""
        import re

        want = {int(i) for i in ids}
        if not want or not self._trace_labels:
            return []
        hit: List[int] = []
        for j, lab in enumerate(self._trace_labels):
            if not lab:
                continue
            m = re.match(r"shot_(\d+)\.rsf$", os.path.basename(lab), re.I)
            if m and int(m.group(1)) in want:
                hit.append(j)
        return hit

    def _wiggle_xy_for_trace(self, j: int, *, red: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """单道叠画 wiggle 的 (x, t)；与密度折合口径一致。"""
        assert self._data is not None and self._x_coords is not None
        nt = self._data.shape[0]
        t_step = 1 if nt <= 2500 else (2 if nt <= 6000 else 3)
        base = float(self._x_coords[j])
        tr = self._muted_one_trace(j)
        use_roll = self._display_mode == "density" and self._display_vred > 0 and self._d1 > 0
        if use_roll:
            sh = int(round(-self._reduction_tshift(base) / float(self._d1)))
            if sh:
                tr = np.roll(tr, -sh)
                if sh > 0:
                    tr[-sh:] = 0.0
                else:
                    tr[:-sh] = 0.0
            times = float(self._o1) + np.arange(0, nt, t_step, dtype=float) * float(
                self._d1
            )
        else:
            times = float(self._o1) + np.arange(0, nt, t_step, dtype=float) * float(
                self._d1
            )
            times = times + self._reduction_tshift(base)
        tr_s = tr[::t_step]
        flat = np.abs(tr_s)
        if flat.size > 50_000:
            flat = flat[:: max(1, flat.size // 40_000)]
        p98 = float(np.percentile(flat, 98)) if flat.size else 0.0
        if p98 < 1e-20:
            p98 = float(np.max(flat)) if flat.size else 1.0
        if p98 < 1e-20:
            p98 = 1.0
        spacing = self._median_trace_spacing()
        # 手选红波形略宽；浏览琥珀稍细
        frac = (1.65 if red else 1.55) if self._display_mode == "density" else (
            0.55 if red else 0.45
        )
        scale = (frac * spacing * float(self._dscale)) / p98
        xw = base + tr_s * scale
        return (
            np.asarray(xw, dtype=float),
            np.asarray(times[: xw.size], dtype=float),
        )

    def _plot_shot_wiggly(
        self, curve, shot_ids: Sequence[int], *, red: bool
    ) -> None:
        if curve is None:
            return
        if (
            not self.overlays_enabled()
            or self._data is None
            or self._x_coords is None
            or not shot_ids
        ):
            curve.setData([], [])
            return
        idxs = self._trace_indices_for_shot_ids(shot_ids)
        if not idxs:
            curve.setData([], [])
            return
        if len(idxs) > 120:
            step = int(np.ceil(len(idxs) / 120.0))
            idxs = idxs[::step]
        xs_parts: List[np.ndarray] = []
        ys_parts: List[np.ndarray] = []
        for j in idxs:
            if j < 0 or j >= self._data.shape[1]:
                continue
            xw, tw = self._wiggle_xy_for_trace(int(j), red=red)
            xs_parts.append(xw)
            ys_parts.append(tw)
            xs_parts.append(np.array([np.nan], dtype=float))
            ys_parts.append(np.array([np.nan], dtype=float))
        if not xs_parts:
            curve.setData([], [])
            return
        curve.setData(
            np.concatenate(xs_parts),
            np.concatenate(ys_parts),
            connect="finite",
        )

    def _refresh_selection_marks(self) -> None:
        """已追加手选：红色波形。"""
        self._plot_shot_wiggly(self._sel_curve, self._selected_shot_ids, red=True)

    def _refresh_pending_marks(self) -> None:
        """临时多选：琥珀波形（可多道）。"""
        # 单道时仍走 highlight_idx 的细曲线；多道用同一琥珀笔
        if len(self._pending_shot_ids) <= 1:
            self._update_highlight_wiggle()
            return
        self._plot_shot_wiggly(self._hl_curve, self._pending_shot_ids, red=False)

    def _refresh_highlight(self) -> None:
        """临时琥珀 + 已追加红波形；mute/预处理不叠画。"""
        if self.plot is None:
            return
        if self._hl_region is not None:
            self._hl_region.setVisible(False)
        if self._hl_line is not None:
            self._hl_line.setVisible(False)
        if not self.overlays_enabled():
            if self._hl_curve is not None:
                self._hl_curve.setData([], [])
            if self._sel_curve is not None:
                self._sel_curve.setData([], [])
            return
        self._refresh_selection_marks()
        self._refresh_pending_marks()

    def _x_extent(self) -> Tuple[float, float, float]:
        """返回 (x0, width, x_right) 用于 ImageItem.setRect。"""
        assert self._data is not None and self._x_coords is not None
        ntr = self._data.shape[1]
        xs = self._x_coords[:ntr]
        if ntr == 1:
            # 单道：用典型道间距，勿用 |offset| 估宽
            if self._nominal_dx is not None and self._nominal_dx > 1e-12:
                w = float(self._nominal_dx)
            else:
                w = 2.0 if self._x_label != "Trace" else 1.0
            xc = float(xs[0])
            return xc - 0.5 * w, w, xc + 0.5 * w
        # 用相邻中点扩展
        x0 = float(xs[0] - 0.5 * (xs[1] - xs[0]))
        x1 = float(xs[-1] + 0.5 * (xs[-1] - xs[-2]))
        return x0, x1 - x0, x1

    def _display_x_limits(self) -> Tuple[float, float]:
        """图窗 X 范围：少道时加宽两侧空白，避免一道横跨整窗。"""
        assert self._data is not None and self._x_coords is not None
        if self._view_x_range is not None:
            return float(self._view_x_range[0]), float(self._view_x_range[1])
        x0, w, x1 = self._x_extent()
        ntr = self._data.shape[1]
        if ntr <= 3:
            xc = 0.5 * (x0 + x1)
            half_col = 0.5 * abs(w) if w != 0 else 1.0
            # 道宽约占视窗 ~10%
            view_half = max(half_col * 5.0, 8.0 if self._x_label != "Trace" else 4.0)
            return xc - view_half, xc + view_half
        return x0, x1

    def _local_trace_spacing(self) -> float:
        """当前图中邻道中位间距（稀疏手选时常为大空隙）。"""
        assert self._x_coords is not None and self._data is not None
        xs = np.asarray(self._x_coords[: self._data.shape[1]], dtype=float)
        if xs.size > 1:
            d = np.diff(np.sort(xs))
            d = d[np.isfinite(d) & (d > 1e-12)]
            if d.size:
                med = float(np.median(d))
                small = d[d <= max(med * 3.0, 1e-9)]
                spacing = float(np.median(small)) if small.size else med
                if spacing > 0:
                    return spacing
        return 2.0 if self._x_label != "Trace" else 1.0

    def _median_trace_spacing(self) -> float:
        """wiggle 定宽用的道间距：稀疏子集回退到全炮典型间距。"""
        local = self._local_trace_spacing()
        nom = self._nominal_dx
        if nom is not None and nom > 1e-12 and local > float(nom) * 2.0:
            return float(nom)
        return local

    def _prepared_data(self, apply_mute: bool = True) -> np.ndarray:
        assert self._data is not None
        from ..services.polygon_mute import apply_polygon_mute
        from ..services.preprocess import time_axis

        data = self._data
        if apply_mute and self.mute_enabled():
            nt = data.shape[0]
            times = time_axis(nt, self._d1, self._o1)
            data = apply_polygon_mute(
                data,
                times,
                self._x_coords if self._x_coords is not None else np.arange(data.shape[1]),
                self._mute_points,
                enabled=True,
                invert=self._mute_invert,
                display_vred=float(self._display_vred),
                x_reduce_origin=float(self._x_reduce_origin),
                tp=float(self._mute_tp),
            )
        return np.asarray(data, dtype=np.float32)

    def _wiggle_scale(self, data: Optional[np.ndarray] = None) -> float:
        """
        横向尺度：使 p98(|amp|) * scale ≈ 0.4 * 道间距。
        关闭增益时原始振幅很大，必须按数据归一，否则波形横扫整窗。
        """
        spacing = self._median_trace_spacing()
        target = 0.40 * spacing * float(self._dscale)
        arr = self._data if data is None else data
        if arr is None or arr.size == 0:
            return target
        flat = np.abs(np.asarray(arr, dtype=np.float32))
        if flat.size > 200_000:
            flat = flat.ravel()[:: max(1, flat.size // 200_000)]
        else:
            flat = flat.ravel()
        p98 = float(np.percentile(flat, 98)) if flat.size else 0.0
        if p98 < 1e-20:
            p98 = float(np.max(flat)) if flat.size else 1.0
        if p98 < 1e-20:
            p98 = 1.0
        return target / p98

    def _trace_indices_for_wiggle(self, ntr: int, max_traces: int = 320) -> np.ndarray:
        if ntr <= max_traces:
            return np.arange(ntr, dtype=int)
        step = int(np.ceil(ntr / float(max_traces)))
        idx = np.arange(0, ntr, step, dtype=int)
        hl = self._highlight_idx
        if hl is not None and int(hl) not in set(idx.tolist()):
            idx = np.sort(np.append(idx, int(hl)))
        return idx

    def _muted_one_trace(self, j: int) -> np.ndarray:
        """只对一道做多边形 mute（换炮高亮用）。禁止对整幅 gather 重算。"""
        assert self._data is not None and self._x_coords is not None
        tr = np.asarray(self._data[:, j], dtype=np.float64)
        if not (self._last_apply_mute and self.mute_enabled()):
            return tr
        from ..services.polygon_mute import apply_polygon_mute
        from ..services.preprocess import time_axis

        nt = int(self._data.shape[0])
        times = time_axis(nt, self._d1, self._o1)
        col = apply_polygon_mute(
            self._data[:, j : j + 1],
            times,
            [float(self._x_coords[j])],
            self._mute_points,
            enabled=True,
            invert=self._mute_invert,
            display_vred=float(self._display_vred),
            x_reduce_origin=float(self._x_reduce_origin),
            tp=float(self._mute_tp),
        )
        return np.asarray(col[:, 0], dtype=np.float64)

    def _update_highlight_wiggle(self) -> None:
        """临时单选琥珀波形；多选走 _refresh_pending_marks。"""
        if (
            self.plot is None
            or self._data is None
            or self._x_coords is None
            or self._hl_curve is None
        ):
            return
        if not self.overlays_enabled():
            self._hl_curve.setData([], [])
            return
        if len(self._pending_shot_ids) > 1:
            return
        j = self._highlight_idx
        if j is None and self._pending_shot_ids:
            idxs = self._trace_indices_for_shot_ids(self._pending_shot_ids[:1])
            j = idxs[0] if idxs else None
        if j is None or j < 0 or j >= self._data.shape[1]:
            self._hl_curve.setData([], [])
            return
        xw, tw = self._wiggle_xy_for_trace(int(j), red=False)
        self._hl_curve.setData(xw, tw, connect="finite")

    def _redraw_wiggle(self, data: np.ndarray) -> None:
        assert self.plot is not None and self._x_coords is not None
        nt, ntr = data.shape
        xs = np.asarray(self._x_coords[:ntr], dtype=float)
        scale = self._wiggle_scale(data)
        t_step = 2 if nt > 4000 else (1 if nt <= 2500 else 2)
        if ntr > 800:
            t_step = max(t_step, 3)
        times0 = float(self._o1) + np.arange(0, nt, t_step, dtype=np.float64) * float(
            self._d1
        )
        ns = int(times0.size)
        idx = self._trace_indices_for_wiggle(ntr)
        self._img.clear()

        # 合并为单条曲线：每道后插 NaN；折合：每道时间轴平移
        nseg = int(idx.size)
        chunk = ns + 1
        all_x = np.full(nseg * chunk, np.nan, dtype=np.float32)
        all_t = np.full(nseg * chunk, np.nan, dtype=np.float32)
        sub = data[::t_step, :]
        for k, j in enumerate(idx):
            jj = int(j)
            sl = slice(k * chunk, k * chunk + ns)
            all_x[sl] = np.float32(xs[jj]) + sub[:ns, jj].astype(np.float32) * np.float32(
                scale
            )
            tshift = self._reduction_tshift(float(xs[jj]))
            all_t[sl] = (times0[:ns] + tshift).astype(np.float32)

        if self._wiggle_item is not None:
            self._wiggle_item.setData(all_x, all_t, connect="finite")

        if self._hl_curve is not None:
            if self._highlight_idx is not None and 0 <= int(self._highlight_idx) < ntr:
                j = int(self._highlight_idx)
                tshift = self._reduction_tshift(float(xs[j]))
                self._hl_curve.setData(
                    np.float32(xs[j]) + sub[:ns, j].astype(np.float32) * np.float32(scale),
                    (times0[:ns] + tshift).astype(np.float32),
                    connect="finite",
                )
            else:
                self._hl_curve.setData([], [])

        fill_mode = self._display_mode
        if self._shade_item is not None:
            if fill_mode in ("fill+", "fill-") and nseg:
                fill_pos = fill_mode == "fill+"
                row_step = 2 if ns > 1500 else 1
                parts_x: List[np.ndarray] = []
                parts_y: List[np.ndarray] = []
                fill_idx = idx[:: max(1, int(np.ceil(nseg / 160.0)))]
                for j in fill_idx:
                    jj = int(j)
                    xw = np.float64(xs[jj]) + sub[:ns, jj].astype(np.float64) * scale
                    amp = xw - float(xs[jj])
                    mask = amp > 0 if fill_pos else amp < 0
                    ii = np.flatnonzero(mask)[::row_step]
                    if ii.size == 0:
                        continue
                    bx = np.full(ii.size, float(xs[jj]))
                    by = times0[ii] + self._reduction_tshift(float(xs[jj]))
                    parts_x.append(np.column_stack([bx, xw[ii]]).ravel())
                    parts_y.append(np.column_stack([by, by]).ravel())
                if parts_x:
                    self._shade_item.setData(
                        np.concatenate(parts_x), np.concatenate(parts_y), connect="pairs"
                    )
                else:
                    self._shade_item.setData([], [])
            else:
                self._shade_item.setData([], [])

        self.plot.setTitle(self._title)
        if self._auto_range:
            vx0, vx1 = self._display_x_limits()
            # wiggle 摆动不超过视窗；少道时 _display_x_limits 已留白
            pad = 0.02 * abs(vx1 - vx0)
            h = max(nt, 1) * float(self._d1)
            t0 = float(self._o1)
            t1 = t0 + h
            if self._display_vred > 0 and xs.size:
                shifts = np.array([self._reduction_tshift(float(x)) for x in xs])
                t0 = float(np.min(times0[0] + shifts)) if ns else t0
                t1 = float(np.max(times0[-1] + shifts)) if ns else t1
            self.plot.setXRange(vx0 - pad, vx1 + pad, padding=0)
            self.plot.setYRange(t0, t1, padding=0)

    def _density_raster_dx(self, xs: np.ndarray) -> float:
        xs = np.asarray(xs, dtype=float)
        if xs.size <= 1:
            fallback = 2.0 if self._x_label != "Trace" else 1.0
            nom = self._nominal_dx
            return float(nom) if nom is not None and nom > 1e-12 else fallback
        d = np.diff(np.sort(xs))
        d = d[np.isfinite(d) & (d > 1e-12)]
        if d.size == 0:
            fallback = 2.0 if self._x_label != "Trace" else 1.0
            nom = self._nominal_dx
            return float(nom) if nom is not None and nom > 1e-12 else fallback
        med = float(np.median(d))
        small = d[d <= max(med * 3.0, 1e-9)]
        local = float(np.median(small)) if small.size else med
        nom = self._nominal_dx
        # 手选稀疏：邻道空隙很大时按全炮典型 dx 栅格化，道才是细条而非拉满空隙
        if nom is not None and nom > 1e-12 and local > float(nom) * 2.0:
            return float(nom)
        return local

    def _redraw_density(self, data: np.ndarray) -> None:
        """按真实 x 坐标栅格化；空隙填 0，避免 ImageItem 把邻道拉宽填空白。"""
        assert self.plot is not None and self._x_coords is not None
        if self._wiggle_item is not None:
            self._wiggle_item.setData([], [])
        if self._shade_item is not None:
            self._shade_item.setData([], [])
        view = np.asarray(data, dtype=np.float32)
        ntr = view.shape[1]
        xs = np.asarray(self._x_coords[:ntr], dtype=float)
        # density：按采样点整数平移近似折合
        if self._display_vred > 0 and self._d1 > 0:
            view = view.copy()
            for j in range(ntr):
                sh = int(round(-self._reduction_tshift(float(xs[j])) / self._d1))
                if sh:
                    view[:, j] = np.roll(view[:, j], -sh)
                    if sh > 0:
                        view[-sh:, j] = 0
                    else:
                        view[:-sh, j] = 0
        if abs(self._dscale - 1.0) > 1e-6:
            view = np.asarray(view, dtype=np.float32) * float(self._dscale)

        order = np.argsort(xs)
        xs_s = xs[order]
        cols = view[:, order]
        dx_med = self._density_raster_dx(xs_s)
        nt = int(cols.shape[0])
        if ntr == 1:
            x0 = float(xs_s[0]) - 0.5 * dx_med
            x1 = float(xs_s[0]) + 0.5 * dx_med
            raster = cols
        else:
            x0 = float(xs_s[0]) - 0.5 * dx_med
            x1 = float(xs_s[-1]) + 0.5 * dx_med
            span = max(x1 - x0, dx_med)
            nx = int(min(max(int(np.ceil(span / dx_med)), ntr), 8000))
            dx = span / float(nx)
            raster = np.zeros((nt, nx), dtype=np.float32)
            bin_amp = np.zeros(nx, dtype=np.float32)
            for k in range(ntr):
                j = int(np.floor((float(xs_s[k]) - x0) / dx))
                j = int(min(max(j, 0), nx - 1))
                amp = float(np.max(np.abs(cols[:, k])))
                if amp >= float(bin_amp[j]):
                    raster[:, j] = cols[:, k]
                    bin_amp[j] = amp

        occupied = np.any(np.abs(raster) > 0, axis=0)
        flat = np.abs(raster[:, occupied]) if np.any(occupied) else np.abs(raster)
        if flat.size > 250_000:
            step = max(1, flat.size // 200_000)
            sample = flat.ravel()[::step]
        else:
            sample = flat.ravel()
        clim = float(np.percentile(sample, self._pclip)) if sample.size else 1.0
        if clim < 1e-20:
            clim = float(np.max(flat)) or 1.0
        h = max(nt, 1) * float(self._d1)
        self._img.setImage(raster, autoLevels=False)
        self._img.setLevels((-clim, clim))
        self._img.setRect(QRectF(x0, float(self._o1), float(x1 - x0), h))
        self.plot.setTitle(self._title)
        if self._auto_range:
            vx0, vx1 = self._display_x_limits()
            self.plot.setXRange(vx0, vx1, padding=0.02)
            self.plot.setYRange(float(self._o1), float(self._o1) + h, padding=0)
        # 选中炮：在 density 上叠真实 wiggle
        self._update_highlight_wiggle()
        self._refresh_highlight()

    def _redraw_image(self, apply_mute: bool = True) -> None:
        if self.plot is None or self._data is None:
            return
        data = self._prepared_data(apply_mute=apply_mute)
        if self._display_mode == "density":
            self._redraw_density(data)
        else:
            self._redraw_wiggle(data)

    def _set_pan(self, enabled: bool) -> None:
        """开关拖拽平移。Mute 编辑时关平移以免抢左键；滚轮缩放见 eventFilter。"""
        if self.plot is None:
            return
        self.plot.getViewBox().setMouseEnabled(x=bool(enabled), y=bool(enabled))

    def _zoom_by_wheel(self, event) -> None:
        """Mute 编辑时 ViewBox 鼠标关闭，自行做滚轮缩放（相对光标）。"""
        if self.plot is None:
            return
        try:
            dy = float(event.angleDelta().y())
        except Exception:
            return
        if dy == 0:
            return
        vb = self.plot.getViewBox()
        # 与常见查看器一致：滚轮向上放大
        scale = 0.9 if dy > 0 else (1.0 / 0.9)
        try:
            center = vb.mapSceneToView(self.plot.mapToScene(event.position().toPoint()))
        except Exception:
            try:
                center = vb.mapSceneToView(self.plot.mapToScene(event.pos()))
            except Exception:
                return
        vb.scaleBy((scale, scale), center=center)

    def _find_near_vertex(self, x: float, y: float) -> Optional[int]:
        if not self._mute_points or self.plot is None:
            return None
        xr, yr = self.plot.getViewBox().viewRange()
        x_span = max(1e-6, abs(float(xr[1]) - float(xr[0])))
        y_span = max(1e-6, abs(float(yr[1]) - float(yr[0])))
        tol_x = max(1e-6, 0.03 * x_span)
        tol_y = max(1e-6, 0.03 * y_span)
        best_i: Optional[int] = None
        best = 1e99
        for i, (px, py) in enumerate(self._mute_points):
            dx = (x - float(px)) / tol_x
            dy = (y - float(py)) / tol_y
            s = dx * dx + dy * dy
            if s < best:
                best = s
                best_i = i
        return best_i if best <= 1.0 else None

    def _refresh_overlay(self) -> None:
        if self.plot is None:
            return
        # 未启用且非绘制中：不画多边形/顶点（点仍留在内存，勾选后再画）
        if not self._mute_points or (
            not self._mute_enabled and not self._mute_edit
        ):
            self._poly_item.setData([], [])
            self._vtx_item.setData(spots=[])
            return
        pts = np.asarray(self._mute_points, dtype=float)
        if pts.shape[0] == 1:
            self._poly_item.setData([pts[0, 0]], [pts[0, 1]])
        elif self._mute_enabled and pts.shape[0] >= 3:
            closed = np.vstack([pts, pts[0]])
            self._poly_item.setData(closed[:, 0], closed[:, 1])
        else:
            # 绘制中：折线 + 顶点
            self._poly_item.setData(pts[:, 0], pts[:, 1])
        spots = []
        for i in range(pts.shape[0]):
            sel = self._sel_idx is not None and int(self._sel_idx) == i
            spots.append(
                {
                    "pos": (float(pts[i, 0]), float(pts[i, 1])),
                    "size": 11.0 if sel else 8.5,
                    "pen": pg.mkPen("#ffffff", width=1.5),
                    "brush": pg.mkBrush("#f59e0b" if sel else "#38bdf8"),
                    "symbol": "o",
                }
            )
        self._vtx_item.setData(spots=spots)

    def _map_scene_to_view(self, scene_pos) -> Optional[Tuple[float, float]]:
        if self.plot is None:
            return None
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _on_click(self, ev) -> None:
        if self.plot is None or self._data is None:
            return
        # pyqtgraph: MouseClickEvent
        try:
            button = ev.button()
        except Exception:
            return
        mapped = self._map_scene_to_view(ev.scenePos())
        if mapped is None:
            return
        x, y = mapped

        # 脉冲拾取模式：左键点选并显示标记
        if self._impulse_mode:
            if button == Qt.MouseButton.LeftButton:
                pick = self.sample_at_xy(float(x), float(y))
                if pick is None:
                    self.status.emit("脉冲拾取失败：无有效道")
                    return
                self._impulse_pick = pick
                # 标记用显示时间，落在鼠标点击的纵坐标上（折合坐标系）
                self._set_impulse_marker(float(pick["x"]), float(pick["t_disp"]))
                sid = pick.get("shot_idx")
                self.status.emit(
                    "脉冲点: shot=%s  t=%.4fs  amp=%.4g"
                    % (
                        sid if sid is not None else "?",
                        float(pick["t_true"]),
                        float(pick["amp"]),
                    )
                )
                self.impulse_point_picked.emit(pick)
            elif button == Qt.MouseButton.RightButton:
                self.exit_impulse_mode(clear_marker=True)
            return

        # 闭合后点顶点 → 再进编辑（编辑期间关闭 mute，闭合后才重新 mute）
        if (not self._mute_edit) and len(self._mute_points) >= 3:
            near = self._find_near_vertex(x, y)
            if near is not None and button == Qt.MouseButton.LeftButton:
                self._mute_edit = True
                self._mute_enabled = False
                self._set_pan(False)
                self._drag_idx = int(near)
                self._sel_idx = int(near)
                self._refresh_overlay()
                # 编辑时暂不整图 unmute（太慢）；闭合后再 mute 刷新
                if self._hl_curve is not None:
                    self._hl_curve.setData([], [])
                if self._sel_curve is not None:
                    self._sel_curve.setData([], [])
                self.status.emit("Mute 编辑：选中顶点 #%d（闭合后才 mute）" % (near + 1))
                self.mute_changed.emit()
                return

        if not self._mute_edit:
            if not self._hand_pick_enabled:
                return
            j = self.nearest_trace_index(x)
            if j is None:
                return
            lab = ""
            if 0 <= j < len(self._trace_labels):
                lab = os.path.basename(self._trace_labels[j] or "")
            if button == Qt.MouseButton.LeftButton:
                # 仅左键按落点做琥珀浏览选
                self.set_highlight_idx(j)
                mode = "browse"
                tip = "浏览"
            elif button == Qt.MouseButton.RightButton:
                # 右键不按落点选道/高亮，只追加已有临时选
                if not self._pending_shot_ids:
                    self.status.emit("请先左键/Shift 选道，再右键追加")
                    return
                mode = "append"
                tip = "追加已选"
            elif button == Qt.MouseButton.MiddleButton:
                mode = "cancel"
                tip = "取消选择"
            else:
                return
            self.trace_picked.emit(int(j), mode)
            self.status.emit(
                "%s道 %d%s" % (tip, j, (" · %s" % lab) if lab else "")
            )
            return

        if button == Qt.MouseButton.RightButton:
            near = self._find_near_vertex(x, y)
            if near is not None:
                self._sel_idx = int(near)
                self.delete_selected_vertex()
            else:
                self.finalize_polygon()
            return

        if button == Qt.MouseButton.MiddleButton and self._mute_points:
            self._mute_points.pop()
            self._drag_idx = None
            self._sel_idx = (len(self._mute_points) - 1) if self._mute_points else None
            self._refresh_overlay()
            self.status.emit("撤销一点，剩余 %d" % len(self._mute_points))
            return

        if button == Qt.MouseButton.LeftButton:
            near = self._find_near_vertex(x, y)
            if near is not None:
                self._drag_idx = int(near)
                self._sel_idx = int(near)
                self._refresh_overlay()
                self.status.emit("选中顶点 #%d，拖动调整" % (near + 1))
                return
            self._mute_points.append((x, y))
            self._drag_idx = len(self._mute_points) - 1
            self._sel_idx = self._drag_idx
            self._refresh_overlay()
            # 绘制中只改叠层，不 emit（避免主窗口同步/扫炮拖慢加点）
            self.status.emit(
                "已添加 %d 个顶点（右键闭合后才 mute）" % len(self._mute_points)
            )

    def nearest_trace_index(self, x: Optional[float] = None) -> Optional[int]:
        """按 X 找最近道下标（拼图/单炮通用）。"""
        if self._data is None or self._x_coords is None:
            return None
        ntr = int(self._data.shape[1])
        if ntr <= 0:
            return None
        xs = np.asarray(self._x_coords[:ntr], dtype=float)
        if x is None:
            x = self.mouse_x
        if x is None:
            try:
                xr, _ = self.plot.getViewBox().viewRange()
                x = 0.5 * (float(xr[0]) + float(xr[1]))
            except Exception:
                return 0
        return int(np.argmin(np.abs(xs - float(x))))

    def shot_path_at_trace(self, j: Optional[int] = None) -> Optional[str]:
        """道 j（默认高亮/光标最近道）对应的 shot 路径（拼图 trace_labels）。"""
        if self._data is None:
            return None
        if j is None:
            j = self._highlight_idx
        if j is None:
            j = self.nearest_trace_index()
        if j is None:
            return None
        j = int(j)
        if 0 <= j < len(self._trace_labels) and self._trace_labels[j]:
            return str(self._trace_labels[j])
        return None

    def shot_index_at_trace(self, j: Optional[int] = None) -> Optional[int]:
        """从道标签解析 shot_NNN 炮号。"""
        import re

        path = self.shot_path_at_trace(j)
        if not path:
            return None
        m = re.match(r"shot_(\d+)\.rsf$", os.path.basename(path), re.I)
        return int(m.group(1)) if m else None

    def _focus_in_text_input(self) -> bool:
        try:
            from ..dialog_utils import focus_is_text_input

            return bool(focus_is_text_input())
        except Exception:
            fw = QApplication.focusWidget()
            return isinstance(
                fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
            )

    def _emit_nav_if_idle(self, delta: int, shift_multi: bool) -> None:
        """左右键换炮：输入框/SpinBox 编辑时不抢光标键。"""
        if self._focus_in_text_input():
            return
        self.navigate_shot.emit(int(delta), bool(shift_multi))

    def _on_m_shortcut(self) -> None:
        if self._focus_in_text_input():
            return
        self.toggle_mute_mode()

    def _on_shift_m_shortcut(self) -> None:
        if self._focus_in_text_input():
            return
        self.toggle_invert()

    def _on_del_shortcut(self) -> None:
        if self._focus_in_text_input():
            return
        self.delete_selected_vertex()

    def _on_i_shortcut(self) -> None:
        """避免在参数框输入时误触发 I。"""
        if self._focus_in_text_input():
            return
        self.show_trace_info()

    def _on_impulse_shortcut(self) -> None:
        """Shift+P：切换脉冲拾取模式（输入框聚焦时忽略）。"""
        if self._focus_in_text_input():
            return
        if self._impulse_mode:
            self.exit_impulse_mode(clear_marker=True)
        else:
            self.enter_impulse_mode()

    def sample_at_xy(self, xref: float, yref: float) -> Optional[dict]:
        """
        在 (x,y) 取样（真实时间，扣除折合）。

        返回 dict: j, shot_idx, shot_path, x, t_disp, t_true, it, amp
        """
        if self.plot is None or self._data is None or self._x_coords is None:
            return None
        j = self.nearest_trace_index(float(xref))
        if j is None:
            return None
        ntr = int(self._data.shape[1])
        nt = int(self._data.shape[0])
        if ntr <= 0 or nt <= 0:
            return None
        j = int(max(0, min(j, ntr - 1)))
        off = float(self._x_coords[j])
        t_disp = float(yref)
        t_true = t_disp - self._reduction_tshift(off)
        d1 = float(self._d1) if float(self._d1) > 1e-12 else 0.004
        o1 = float(self._o1)
        it = int(round((t_true - o1) / d1))
        it = int(max(0, min(it, nt - 1)))
        # 标记纵坐标对齐到样点对应的显示时间
        t_true_snap = float(o1 + it * d1)
        t_disp_snap = t_true_snap + self._reduction_tshift(off)
        col = np.asarray(self._data[:, j], dtype=float)
        amp = float(col[it]) if col.size else 0.0
        path = self.shot_path_at_trace(j)
        shot_idx = self.shot_index_at_trace(j)
        return {
            "j": j,
            "shot_idx": shot_idx,
            "shot_path": path,
            "x": off,
            "t_disp": float(t_disp_snap),
            "t_true": t_true_snap,
            "it": it,
            "amp": amp,
            "d1": d1,
            "o1": o1,
            "nt": nt,
        }

    def sample_at_cursor(self) -> Optional[dict]:
        """光标处取样（真实时间，扣除折合）。"""
        xref = self.mouse_x
        yref = self.mouse_y
        if xref is None or yref is None:
            try:
                xr, yr = self.plot.getViewBox().viewRange()
                xref = 0.5 * (float(xr[0]) + float(xr[1]))
                yref = 0.5 * (float(yr[0]) + float(yr[1]))
            except Exception:
                return None
        return self.sample_at_xy(float(xref), float(yref))

    def show_trace_info(self) -> None:
        """I 键：显示鼠标最近道信息（对齐 zplotpy）。"""
        if self.plot is None or self._data is None or self._x_coords is None:
            self.status.emit("道信息：请先加载道集")
            return
        xref = self.mouse_x
        yref = self.mouse_y
        if xref is None or yref is None:
            try:
                xr, yr = self.plot.getViewBox().viewRange()
                xref = 0.5 * (float(xr[0]) + float(xr[1]))
                yref = 0.5 * (float(yr[0]) + float(yr[1]))
            except Exception:
                self.status.emit("道信息：无法读取光标位置")
                return
        j = self.nearest_trace_index(float(xref))
        if j is None:
            self.status.emit("道信息：无道")
            return
        ntr = int(self._data.shape[1])
        off = float(self._x_coords[j])
        t_disp = float(yref)
        t_true = t_disp - self._reduction_tshift(off)
        label = ""
        if j < len(self._trace_labels):
            label = self._trace_labels[j]
        shot_name = os.path.basename(label) if label else ""
        shot_idx = self.shot_index_at_trace(j)
        # 振幅（当前显示数据）
        col = np.asarray(self._data[:, j], dtype=float)
        amp = float(col[min(max(int(round((t_true - self._o1) / self._d1)), 0), col.size - 1)]) if col.size else 0.0
        p98 = float(np.percentile(np.abs(col), 98)) if col.size else 0.0
        msg = (
            "道索引(拼图列): %d / %d\n"
            "  ※ 按 model x 排序后的显示列号，≠ 炮号\n"
            "炮号: %s\n"
            "炮文件: %s\n"
            "X (%s): %.4f\n"
            "光标时间(显示): %.4f s\n"
            "光标时间(真实): %.4f s\n"
            "折合 vred: %s\n"
            "样点振幅≈ %.4g\n"
            "道 |amp| p98: %.4g\n"
            "显示模式: %s\n"
            "高亮道: %s"
            % (
                j,
                ntr - 1,
                str(shot_idx) if shot_idx is not None else "N/A",
                shot_name or (label or "N/A"),
                self._x_label,
                off,
                t_disp,
                t_true,
                ("%.3g km/s" % self._display_vred) if self._display_vred > 0 else "关",
                amp,
                p98,
                self._display_mode,
                str(self._highlight_idx) if self._highlight_idx is not None else "-",
            )
        )
        if label and label != shot_name:
            msg += "\n路径: %s" % label
        self.status.emit(
            "道 %d: %s=%.3f, t=%.3f s%s"
            % (
                j,
                self._x_label,
                off,
                t_disp,
                (", %s" % shot_name) if shot_name else "",
            )
        )
        self._show_trace_info_dialog(msg)

    def _show_trace_info_dialog(self, msg: str) -> None:
        from ..dialog_utils import show_modeless_dialog

        if self._trace_info_dlg is not None:
            try:
                if self._trace_info_dlg.isVisible():
                    # 复用窗口，只更新文本
                    edit = self._trace_info_dlg.findChild(QPlainTextEdit)
                    if edit is not None:
                        edit.setPlainText(msg)
                    self._trace_info_dlg.raise_()
                    self._trace_info_dlg.activateWindow()
                    return
            except RuntimeError:
                self._trace_info_dlg = None

        dlg = QDialog(self)
        dlg.setWindowTitle("道信息 (I)")
        dlg.setModal(False)
        lay = QVBoxLayout(dlg)
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(msg)
        edit.setMinimumSize(360, 260)
        lay.addWidget(edit)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.close)
        lay.addWidget(btn)
        self._trace_info_dlg = dlg
        show_modeless_dialog(dlg)

    def _on_moved(self, pos) -> None:
        if self.plot is None:
            return
        mapped = self._map_scene_to_view(pos)
        if mapped is not None:
            self.mouse_x, self.mouse_y = float(mapped[0]), float(mapped[1])
        if not self._mute_edit:
            # Shift+悬停：加入临时多选（琥珀），右键才追加
            if self._hand_pick_enabled and mapped is not None:
                mods = QApplication.keyboardModifiers()
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    j = self.nearest_trace_index(mapped[0])
                    if j is not None:
                        sid = self.shot_index_at_trace(int(j))
                        if sid is not None and sid != self._last_shift_shot:
                            self._last_shift_shot = int(sid)
                            self.set_highlight_idx(j)
                            self.trace_picked.emit(int(j), "pending_add")
                else:
                    self._last_shift_shot = None
            return
        # 仅在按住左键拖拽时更新 mute 顶点
        try:
            buttons = QApplication.mouseButtons()
        except Exception:
            buttons = Qt.MouseButton.NoButton
        if buttons != Qt.MouseButton.LeftButton:
            if self._dragging:
                self._dragging = False
                self._drag_idx = None
                # 编辑中只动顶点叠层；闭合后拖顶点松开才重 mute
                if self._mute_enabled and not self._mute_edit:
                    self._redraw_image(apply_mute=True)
            return
        if mapped is None:
            return
        x, y = mapped
        if self._drag_idx is None and self._mute_points:
            near = self._find_near_vertex(x, y)
            if near is not None:
                self._drag_idx = int(near)
                self._sel_idx = int(near)
        if self._drag_idx is None:
            return
        if not (0 <= int(self._drag_idx) < len(self._mute_points)):
            return
        self._mute_points[int(self._drag_idx)] = (x, y)
        self._sel_idx = int(self._drag_idx)
        self._dragging = True
        self._refresh_overlay()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent

        if self.plot is not None:
            # mute 编辑关掉了 ViewBox 鼠标 → 补上回滚轮缩放
            if obj is self.plot.viewport() and event.type() == QEvent.Type.Wheel:
                if self._mute_edit:
                    self._zoom_by_wheel(event)
                    event.accept()
                    return True
            if obj is self.plot.scene():
                if event.type() == QEvent.Type.GraphicsSceneMouseRelease:
                    if self._dragging:
                        self._dragging = False
                        self._drag_idx = None
                        # 绘制中不整图重绘；已启用 mute 时松手再 mute 一次
                        if self._mute_enabled and not self._mute_edit:
                            self._last_apply_mute = True
                            self._redraw_image(apply_mute=True)
                            self.mute_changed.emit()
        return super().eventFilter(obj, event)
