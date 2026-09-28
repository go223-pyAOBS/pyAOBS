"""反演结果分析：原生 pyqtgraph（曲线 / 散点 / 条形）。速度场仍走 Matplotlib。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGraphicsRectItem,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
    PARAM_INFLUENCE_AXES,
    decimal_log_tick_values,
    format_decimal_log_tick,
    overlay_curve_series,
    param_influence_data,
    pareto_score_data,
    single_log_curve_data,
    summary_table_ranked,
)

from ..dialog_utils import show_modeless_dialog
from .inv_monitor_pg import (
    _PgHintBar,
    _hex_brush,
    _hex_pen,
    _make_glw,
    save_glw_png,
    style_plot_ink,
)

_TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def series_color(i: int) -> str:
    return _TAB10[int(i) % len(_TAB10)]


def _spot_index(spot, fallback: int | None = None) -> int | None:
    for getter in ("data", "index"):
        try:
            val = getattr(spot, getter, None)
            val = val() if callable(val) else val
            if val is not None:
                return int(val)
        except (TypeError, ValueError):
            continue
    return fallback


def _scatter_item(rec) -> pg.ScatterPlotItem | None:
    item = rec.get("item")
    if isinstance(item, pg.ScatterPlotItem):
        return item
    sc = getattr(item, "scatter", None)
    return sc if isinstance(sc, pg.ScatterPlotItem) else None


def _click_scene_pos(ev) -> QPointF | None:
    for name in ("scenePos", "scenePosition"):
        fn = getattr(ev, name, None)
        if callable(fn):
            try:
                sp = fn()
                return QPointF(float(sp.x()), float(sp.y()))
            except Exception:
                continue
    return None


def _scatter_nearest(rec: dict, ev, *, max_px: float = 18.0) -> tuple[int | None, float]:
    """返回 (下标, 场景距离平方)；未命中为 (None, inf)。"""
    scatter = _scatter_item(rec)
    click = _click_scene_pos(ev)
    if scatter is None or click is None:
        return None, float("inf")
    lim = max_px * max_px
    try:
        spots = scatter.pointsAt(scatter.mapFromScene(click))
        if spots:
            spot = spots[0]
            scene = scatter.mapToScene(spot.pos())
            d2 = (float(scene.x()) - click.x()) ** 2 + (float(scene.y()) - click.y()) ** 2
            return _spot_index(spot, 0), d2
    except Exception:
        pass
    best_i: int | None = None
    best = lim
    try:
        spots = list(scatter.points())
    except Exception:
        spots = []
    for i, spot in enumerate(spots):
        try:
            pos = spot.pos()
            if not (np.isfinite(float(pos.x())) and np.isfinite(float(pos.y()))):
                continue
            scene = scatter.mapToScene(pos)
        except Exception:
            continue
        d2 = (float(scene.x()) - click.x()) ** 2 + (float(scene.y()) - click.y()) ** 2
        if d2 <= best:
            best = d2
            best_i = _spot_index(spot, i)
    return best_i, best


def _scatter_hit_index(rec: dict, ev, *, max_px: float = 18.0) -> int | None:
    """用散点自己的屏幕位置命中（对齐 Matplotlib contains），避免 log 轴换算偏差。"""
    idx, _d2 = _scatter_nearest(rec, ev, max_px=max_px)
    return idx


def _pass_mouse_through(*items) -> None:
    """不要让图元吃掉 Qt press，否则 pyqtgraph 不会再派发 click/drag。"""
    for item in items:
        if item is None:
            continue
        try:
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        except Exception:
            continue


def _wire_spot_clicks(win: "AnalysisPgWindow", rec: dict) -> None:
    """圆点命中对齐 Matplotlib contains：用 Spot 屏幕位置，半径约 18px。"""
    scatter = _scatter_item(rec)
    if scatter is None:
        return
    item = rec.get("item")
    _pass_mouse_through(scatter, item, getattr(item, "curve", None))
    try:
        scatter.setZValue(20)
    except Exception:
        pass

    def _click(ev) -> None:
        if ev.button() not in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            ev.ignore()
            return
        hit = _scatter_hit_index(rec, ev, max_px=18.0)
        if hit is None:
            ev.ignore()
            return
        idx = int(rec["series"]) if rec.get("kind") == "line" else hit
        ev.accept()
        win._emit_pick(
            "click",
            index=idx,
            button=ev.button(),
            ctrl=_ev_ctrl(ev),
            shift=_ev_shift(ev),
            visual_order=win._visual_order_for_vb(rec.get("vb")),
        )

    scatter.mouseClickEvent = _click


class DecimalLogAxis(pg.AxisItem):
    """对数轴刻度用 600 而不是 6×10²，并标出数据点（如 150）。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._linear_data: np.ndarray = np.array([], dtype=float)

    def set_linear_data(self, xs) -> None:
        arr = np.asarray(list(xs), dtype=float).ravel()
        self._linear_data = np.unique(arr[np.isfinite(arr) & (arr > 0)])
        try:
            self.picture = None
        except Exception:
            pass
        self.update()

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        lo, hi = float(minVal), float(maxVal)
        if hi < lo:
            lo, hi = hi, lo
        vmin, vmax = 10.0 ** lo, 10.0 ** hi
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= 0:
            return super().logTickValues(minVal, maxVal, size, stdTicks)
        linear = decimal_log_tick_values(vmin, vmax, self._linear_data)
        logs = [float(np.log10(t)) for t in linear if t > 0 and np.isfinite(t)]
        if not logs:
            return super().logTickValues(minVal, maxVal, size, stdTicks)
        return [(1.0, logs)]

    def logTickStrings(self, values, scale, spacing):
        out: list[str] = []
        for v in values:
            try:
                out.append(format_decimal_log_tick(10.0 ** float(v)))
            except Exception:
                out.append("")
        return out


class AnalysisViewBox(pg.ViewBox):
    """Shift+左拖框选；单击交给 host；其余仍为平移 / 右拖缩放。"""

    def __init__(self, host=None, **kwargs) -> None:
        kwargs.setdefault("enableMenu", False)
        super().__init__(**kwargs)
        self._host = host
        self._p0 = None
        self._rubber: QGraphicsRectItem | None = None

    def mouseClickEvent(self, ev) -> None:  # noqa: N802
        if ev.double():
            super().mouseClickEvent(ev)
            return
        if self._host is not None and ev.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            self._host.on_vb_click(self, ev)
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None) -> None:  # noqa: N802
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if ev.button() == Qt.MouseButton.LeftButton and shift:
            pos = self.mapToView(ev.pos())
            if ev.isStart():
                self._p0 = QPointF(pos)
                self._ensure_rubber()
            if self._p0 is not None:
                self._set_rubber(QRectF(self._p0, QPointF(pos)).normalized())
            if ev.isFinish():
                rect = (
                    QRectF(self._p0, QPointF(pos)).normalized()
                    if self._p0 is not None
                    else QRectF()
                )
                self._clear_rubber()
                self._p0 = None
                if self._host is not None:
                    self._host.on_vb_box(self, rect, ev)
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            if ev.isStart():
                try:
                    self._drag0 = QPointF(ev.scenePos())
                except Exception:
                    self._drag0 = None
            super().mouseDragEvent(ev, axis=axis)
            if ev.isFinish():
                p0 = getattr(self, "_drag0", None)
                self._drag0 = None
                try:
                    sp = ev.scenePos()
                    dx = float(sp.x()) - float(p0.x()) if p0 is not None else 99.0
                    dy = float(sp.y()) - float(p0.y()) if p0 is not None else 99.0
                except Exception:
                    dx = dy = 99.0
                if dx * dx + dy * dy <= 64.0 and self._host is not None:
                    self._host.on_vb_click(self, ev)
            return
        super().mouseDragEvent(ev, axis=axis)

    def _ensure_rubber(self) -> None:
        if self._rubber is not None:
            return
        item = QGraphicsRectItem()
        item.setPen(QPen(QColor("#1d4ed8"), 0))
        item.setBrush(QBrush(QColor(147, 197, 253, 70)))
        item.setZValue(100)
        self.addItem(item)
        self._rubber = item

    def _set_rubber(self, rect: QRectF) -> None:
        if self._rubber is not None:
            self._rubber.setRect(rect)

    def _clear_rubber(self) -> None:
        if self._rubber is None:
            return
        try:
            self.removeItem(self._rubber)
        except Exception:
            pass
        self._rubber = None


class AnalysisPgWindow(QWidget):
    """分析图窗公共壳：提示条、保存、Reset、点选回调。"""

    def __init__(self, title: str, *, save_dir: str | None = None, default_name: str = "analysis.png") -> None:
        super().__init__(None)
        self.setWindowTitle(title)
        self.resize(980, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._save_dir = save_dir or ""
        self._default_name = default_name
        self.names: list[str] = []
        self._pickers: list[dict] = []
        self._home: list[tuple] | None = None
        self.user_pick = None
        self._analysis_pg = True
        self.hint: QLabel | None = None

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        btn_save = QPushButton("保存图像…")
        btn_save.setToolTip("保存为 PNG / JPEG / TIFF / PDF / PS / EPS / SVG")
        btn_save.clicked.connect(self.save_png)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        top.addWidget(btn_save)
        top.addStretch(1)
        top.addWidget(btn_close)
        lay.addLayout(top)

        bar = _PgHintBar(self.reset_view, None, self)
        self.hint = bar.hint
        lay.addWidget(bar)

        self._glw = _make_glw(self)
        lay.addWidget(self._glw, stretch=1)

    def _add_plot(
        self,
        row: int,
        col: int,
        *,
        title: str = "",
        log_x: bool = False,
        log_xs=None,
    ):
        vb = AnalysisViewBox(host=self)
        kw: dict = {"row": row, "col": col, "viewBox": vb}
        if log_x:
            kw["axisItems"] = {"bottom": DecimalLogAxis(orientation="bottom")}
        plot = self._glw.addPlot(**kw)
        if title:
            plot.setTitle(title)
        style_plot_ink(plot)
        plot.showGrid(x=True, y=True, alpha=0.3)
        try:
            plot.setMenuEnabled(False)
        except Exception:
            pass
        if log_x:
            plot.setLogMode(x=True, y=False)
            ax = plot.getAxis("bottom")
            if isinstance(ax, DecimalLogAxis) and log_xs is not None:
                ax.set_linear_data(log_xs)
        return plot, vb

    def _remember_home(self) -> None:
        self._home = []
        for item in self._glw.ci.items.keys():
            if isinstance(item, pg.PlotItem):
                item.enableAutoRange()
                self._home.append((item, item.viewRange()))

    def reset_view(self) -> None:
        if not self._home:
            for item in self._glw.ci.items.keys():
                if isinstance(item, pg.PlotItem):
                    item.enableAutoRange()
            return
        for plot, (xr, yr) in self._home:
            try:
                plot.setXRange(xr[0], xr[1], padding=0)
                plot.setYRange(yr[0], yr[1], padding=0)
            except Exception:
                plot.enableAutoRange()

    def save_png(self) -> None:
        save_glw_png(
            self, self._glw, start_dir=self._save_dir, default_name=self._default_name
        )

    def set_save_dir(self, path: str | Path | None) -> None:
        self._save_dir = str(path) if path else ""

    def on_vb_click(self, vb: AnalysisViewBox, ev) -> None:
        idx = self.hit_index(vb, ev)
        self._emit_pick(
            "click",
            index=idx,
            button=ev.button(),
            ctrl=_ev_ctrl(ev),
            shift=_ev_shift(ev),
            visual_order=self._visual_order_for_vb(vb),
        )

    def on_vb_box(self, vb: AnalysisViewBox, rect: QRectF, ev) -> None:
        if _rect_too_small(vb, rect):
            self.on_vb_click(vb, ev)
            return
        indices = self.box_indices(vb, rect)
        self._emit_pick(
            "box",
            indices=indices,
            button=Qt.MouseButton.LeftButton,
            ctrl=_ev_ctrl(ev),
            shift=True,
        )

    def _emit_pick(self, kind: str, **kw) -> None:
        fn = self.user_pick
        if callable(fn):
            fn(kind, **kw)

    def _visual_order_for_vb(self, vb) -> list[str] | None:
        """条形图按屏幕从上到下的排序做 Shift 连选（与汇总表一致）。"""
        for rec in self._pickers:
            if rec.get("vb") is not vb or rec.get("kind") != "bar":
                continue
            order = rec.get("index_map") or []
            names = list(self.names)
            out = [
                names[int(j)] for j in order if 0 <= int(j) < len(names)
            ]
            return out or None
        return None

    def hit_index(self, vb, ev, *, max_px: float = 18.0) -> int | None:
        click = _click_scene_pos(ev)
        if click is None:
            return None
        best_i: int | None = None
        best_d = max_px * max_px
        bar_hit: int | None = None
        for rec in self._pickers:
            if rec.get("vb") is not vb:
                continue
            kind = rec["kind"]
            if kind == "scatter":
                i, d2 = _scatter_nearest(rec, ev, max_px=max_px)
                if i is not None and d2 <= best_d:
                    best_d = d2
                    best_i = i
            elif kind == "line":
                i, d2 = _scatter_nearest(rec, ev, max_px=max_px)
                if i is not None and d2 <= best_d:
                    best_d = d2
                    best_i = int(rec["series"])
                elif best_i is None and _line_hits(rec, click, vb, max_px):
                    best_i = int(rec["series"])
                    best_d = max_px * max_px
            elif kind == "bar":
                bar_hit = _bar_hit(rec, vb, ev)
        if best_i is not None:
            return best_i
        return bar_hit

    def box_indices(self, vb, rect: QRectF) -> list[int]:
        hit: set[int] = set()
        for rec in self._pickers:
            if rec.get("vb") is not vb:
                continue
            kind = rec["kind"]
            if kind == "scatter":
                scatter = _scatter_item(rec)
                if scatter is not None:
                    try:
                        for i, spot in enumerate(scatter.points()):
                            scene = scatter.mapToScene(spot.pos())
                            view_pt = vb.mapSceneToView(scene)
                            if rect.contains(view_pt):
                                idx = _spot_index(spot, i)
                                if idx is not None:
                                    hit.add(int(idx))
                        continue
                    except Exception:
                        pass
                for i, (x, y) in enumerate(zip(rec["x"], rec["y"])):
                    xy = _to_view_xy(vb, x, y)
                    if xy is not None and rect.contains(QPointF(xy[0], xy[1])):
                        hit.add(int(i))
            elif kind == "line":
                xs, ys = rec["x"], rec["y"]
                pts: list[tuple[float, float]] = []
                inside = False
                for x, y in zip(xs, ys):
                    xy = _to_view_xy(vb, x, y)
                    if xy is None:
                        continue
                    pts.append(xy)
                    if rect.contains(QPointF(xy[0], xy[1])):
                        inside = True
                        break
                if not inside:
                    for j in range(1, len(pts)):
                        if _seg_hits_rect(
                            pts[j - 1][0],
                            pts[j - 1][1],
                            pts[j][0],
                            pts[j][1],
                            rect,
                        ):
                            inside = True
                            break
                if inside:
                    hit.add(int(rec["series"]))
            elif kind == "bar":
                for i, (y0, h, w) in enumerate(
                    zip(rec["y0"], rec["height"], rec["width"])
                ):
                    br = QRectF(0.0, float(y0), float(w), float(h))
                    if rect.intersects(br.normalized()):
                        hit.add(int(rec["index_map"][i]))
        return sorted(hit)

    def set_highlight_names(self, names: list[str] | None) -> None:
        sel = set(names or [])
        nsel = len(sel)
        for rec in self._pickers:
            kind = rec["kind"]
            if kind == "scatter":
                n = len(rec["x"])
                sizes = []
                brushes = []
                pens = []
                for i in range(n):
                    col = rec["colors"][i]
                    on = (not nsel) or (self.names[i] in sel if i < len(self.names) else False)
                    if not nsel:
                        sizes.append(11)
                        brushes.append(_hex_brush(col, 0.9))
                        pens.append(_hex_pen("#222222", 0.4))
                    elif on:
                        sizes.append(18)
                        brushes.append(_hex_brush(col, 1.0))
                        pens.append(_hex_pen("#111111", 2.2))
                    else:
                        sizes.append(11)
                        brushes.append(_hex_brush(col, 0.38))
                        pens.append(_hex_pen(col, 0.5, alpha=0.32))
                rec["item"].setSize(sizes)
                rec["item"].setBrush(brushes)
                rec["item"].setPen(pens)
            elif kind == "line":
                name = self.names[int(rec["series"])] if rec["series"] < len(self.names) else ""
                col = rec["color"]
                if not nsel:
                    rec["item"].setPen(_hex_pen(col, 1.2))
                    rec["item"].setSymbolSize(6)
                    rec["item"].setZValue(3)
                elif name in sel:
                    rec["item"].setPen(_hex_pen(col, 3.2))
                    rec["item"].setSymbolSize(11)
                    rec["item"].setZValue(8)
                else:
                    rec["item"].setPen(_hex_pen(col, 1.0, alpha=0.32))
                    rec["item"].setSymbolSize(6)
                    rec["item"].setZValue(2)
            elif kind == "bar":
                brushes = []
                pens = []
                for i, si in enumerate(rec["index_map"]):
                    col = rec["colors"][i]
                    name = self.names[int(si)] if int(si) < len(self.names) else ""
                    on = (not nsel) or name in sel
                    if not nsel:
                        brushes.append(_hex_brush(col, 0.85))
                        pens.append(_hex_pen("#334155", 0.4))
                    elif on:
                        brushes.append(_hex_brush(col, 1.0))
                        pens.append(_hex_pen("#111111", 2.4))
                    else:
                        brushes.append(_hex_brush(col, 0.32))
                        pens.append(_hex_pen("#94a3b8", 0.35, alpha=0.4))
                rec["item"].setOpts(brushes=brushes, pens=pens)


def _ev_ctrl(ev) -> bool:
    m = ev.modifiers()
    return bool(
        m
        & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.MetaModifier
        )
    )


def _ev_shift(ev) -> bool:
    return bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)


def _event_scene_pos(ev):
    return _click_scene_pos(ev)


def _rect_too_small(vb: pg.ViewBox, rect: QRectF, *, min_px: float = 8.0) -> bool:
    try:
        a = vb.mapFromView(rect.topLeft())
        b = vb.mapFromView(rect.bottomRight())
        dx = float(b.x()) - float(a.x())
        dy = float(b.y()) - float(a.y())
        return dx * dx + dy * dy <= min_px * min_px
    except Exception:
        return True


def _xy_in_view(vb, x, y) -> tuple[np.ndarray, np.ndarray]:
    """ScatterPlotItem 不会随 PlotItem 对数轴变换，需自己放到 View 坐标。"""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    try:
        lm = vb.state.get("logMode") or (False, False)
    except Exception:
        lm = (False, False)
    if lm[0]:
        xs = np.where(np.isfinite(xs) & (xs > 0.0), np.log10(xs), np.nan)
    if lm[1]:
        ys = np.where(np.isfinite(ys) & (ys > 0.0), np.log10(ys), np.nan)
    return xs, ys


def _to_view_xy(vb, x, y) -> tuple[float, float] | None:
    try:
        lm = vb.state.get("logMode") or (False, False)
    except Exception:
        lm = (False, False)
    xv, yv = float(x), float(y)
    if lm[0]:
        if not np.isfinite(xv) or xv <= 0.0:
            return None
        xv = float(np.log10(xv))
    if lm[1]:
        if not np.isfinite(yv) or yv <= 0.0:
            return None
        yv = float(np.log10(yv))
    return xv, yv


def _nearest_xy(rec: dict, click: QPointF, vb: pg.ViewBox) -> tuple[int | None, float]:
    best_i = None
    best = 1e18
    for i, (x, y) in enumerate(zip(rec["x"], rec["y"])):
        xy = _to_view_xy(vb, x, y)
        if xy is None:
            continue
        try:
            sp = vb.mapViewToScene(QPointF(xy[0], xy[1]))
        except Exception:
            continue
        d2 = (float(sp.x()) - click.x()) ** 2 + (float(sp.y()) - click.y()) ** 2
        if d2 < best:
            best = d2
            best_i = int(i)
    return best_i, best


def _line_hits(rec: dict, click: QPointF, vb: pg.ViewBox, max_px: float) -> bool:
    xs, ys = rec["x"], rec["y"]
    if len(xs) == 0:
        return False
    lim = max_px * max_px
    pts = []
    for x, y in zip(xs, ys):
        xy = _to_view_xy(vb, x, y)
        if xy is None:
            continue
        try:
            sp = vb.mapViewToScene(QPointF(xy[0], xy[1]))
        except Exception:
            continue
        pts.append((float(sp.x()), float(sp.y())))
        if (sp.x() - click.x()) ** 2 + (sp.y() - click.y()) ** 2 <= lim:
            return True
    for j in range(1, len(pts)):
        if _seg_dist2(pts[j - 1], pts[j], (click.x(), click.y())) <= lim:
            return True
    return False


def _seg_dist2(p0, p1, p) -> float:
    x0, y0 = p0
    x1, y1 = p1
    px, py = p
    dx, dy = x1 - x0, y1 - y0
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return (px - x0) ** 2 + (py - y0) ** 2
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / den))
    qx, qy = x0 + t * dx, y0 + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def _seg_hits_rect(x0, y0, x1, y1, rect: QRectF) -> bool:
    if rect.contains(QPointF(x0, y0)) or rect.contains(QPointF(x1, y1)):
        return True
    r = rect.normalized()
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - r.left(), r.right() - x0, y0 - r.top(), r.bottom() - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        t = qi / pi
        if pi < 0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return False
    return True


def _bar_hit(rec: dict, vb: pg.ViewBox, ev) -> int | None:
    try:
        pos = vb.mapSceneToView(ev.scenePos())
        x, y = float(pos.x()), float(pos.y())
    except Exception:
        return None
    for i, (y0, h, w) in enumerate(zip(rec["y0"], rec["height"], rec["width"])):
        if 0.0 <= x <= float(w) and float(y0) <= y <= float(y0) + float(h):
            return int(rec["index_map"][i])
    return None


def _style_plot(plot: pg.PlotItem, ylabel: str, xlabel: str = "") -> None:
    plot.setLabel("left", ylabel)
    if xlabel:
        plot.setLabel("bottom", xlabel)
    style_plot_ink(plot)


def show_single_log_window(
    rows,
    *,
    title: str = "",
    save_dir: str | None = None,
) -> AnalysisPgWindow:
    data = single_log_curve_data(rows)
    win = AnalysisPgWindow(
        f"单日志迭代：{title}" if title else "单日志迭代",
        save_dir=save_dir,
        default_name="tt_inverse_single.png",
    )
    win.resize(920, 720)
    it = data["iter"]
    p0, _ = win._add_plot(0, 0)
    p1, _ = win._add_plot(1, 0)
    p2, _ = win._add_plot(2, 0)
    p1.setXLink(p0)
    p2.setXLink(p0)
    _style_plot(p0, "RMS 折射 (Pg)")
    _style_plot(p1, "RMS 反射 (PmP)")
    _style_plot(p2, "χ²", "iteration")
    p0.plot(it, data["rms_pg"], pen=_hex_pen("#1f77b4", 1.5), symbol="o", symbolSize=6)
    p1.plot(it, data["rms_pmp"], pen=_hex_pen("#d62728", 1.5), symbol="s", symbolSize=6)
    p2.addLegend(offset=(8, 8))
    p2.plot(
        it,
        data["chi_tot"],
        pen=_hex_pen("#2ca02c", 1.5),
        symbol="s",
        symbolSize=6,
        name="initial χ² (合并)",
    )
    p2.plot(
        it,
        data["pred_chi"],
        pen=_hex_pen("#9467bd", 1.5),
        symbol="t",
        symbolSize=6,
        name="pred χ² (LSQR)",
    )
    win._glw.ci.layout.setRowStretchFactor(0, 1)
    win._glw.ci.layout.setRowStretchFactor(1, 1)
    win._glw.ci.layout.setRowStretchFactor(2, 1)
    win._remember_home()
    show_modeless_dialog(win, activate=True)
    return win


def show_overlay_window(
    series: dict,
    *,
    title: str = "",
    save_dir: str | None = None,
) -> AnalysisPgWindow:
    curves = overlay_curve_series(series)
    win = AnalysisPgWindow(
        title or "多日志：折射/反射 RMS 叠画",
        save_dir=save_dir,
        default_name="tt_inverse_overlay.png",
    )
    win.names = [c["name"] for c in curves]
    p0, vb0 = win._add_plot(0, 0)
    p1, vb1 = win._add_plot(1, 0)
    p2, vb2 = win._add_plot(2, 0)
    p1.setXLink(p0)
    p2.setXLink(p0)
    _style_plot(p0, "RMS 折射 (Pg)")
    _style_plot(p1, "RMS 反射 (PmP)")
    _style_plot(p2, "pred χ² (LSQR)", "iteration")
    symbols = ("o", "s", "t")
    for i, c in enumerate(curves):
        col = series_color(i)
        for plot, vb, ykey, sym in (
            (p0, vb0, "rms_pg", symbols[0]),
            (p1, vb1, "rms_pmp", symbols[1]),
            (p2, vb2, "pred_chi", symbols[2]),
        ):
            item = plot.plot(
                c["iter"],
                c[ykey],
                pen=_hex_pen(col, 1.2),
                symbol=sym,
                symbolSize=6,
                symbolBrush=_hex_brush(col),
                symbolPen=_hex_pen("#222222", 0.4),
            )
            win._pickers.append(
                {
                    "kind": "line",
                    "item": item,
                    "vb": vb,
                    "series": i,
                    "x": np.asarray(c["iter"], dtype=float),
                    "y": np.asarray(c[ykey], dtype=float),
                    "color": col,
                }
            )
            _wire_spot_clicks(win, win._pickers[-1])
    win._glw.ci.layout.setRowStretchFactor(0, 1)
    win._glw.ci.layout.setRowStretchFactor(1, 1)
    win._glw.ci.layout.setRowStretchFactor(2, 1)
    win._remember_home()
    show_modeless_dialog(win, activate=True)
    return win


def show_pareto_window(
    series: dict,
    *,
    rough_weight: float = 0.001,
    title: str = "",
    save_dir: str | None = None,
) -> AnalysisPgWindow:
    data = pareto_score_data(series, rough_weight)
    win = AnalysisPgWindow(
        title or f"多日志：Pareto 与综合得分（w={rough_weight:g}）",
        save_dir=save_dir,
        default_name="tt_inverse_pareto.png",
    )
    win.resize(1100, 520)
    win.names = list(data["names"])
    n = len(win.names)
    colors = [series_color(i) for i in range(n)]
    p0, vb0 = win._add_plot(0, 0, title="Pareto 式：左下通常更优")
    p1, vb1 = win._add_plot(0, 1, title="综合得分排序（启发式）")
    _style_plot(p0, "pred χ² (LSQR, 末步)", "R = |Lmvh|+|Lmvv|+|Lmd|")
    _style_plot(p1, "", f"score = pred_χ² × (1 + {rough_weight:g} × R)（越小越好）")
    sc = pg.ScatterPlotItem(
        x=data["rough"],
        y=data["pred_chi"],
        size=11,
        data=list(range(n)),
        brush=[_hex_brush(c, 0.9) for c in colors],
        pen=[_hex_pen("#222222", 0.5) for _ in colors],
    )
    p0.addItem(sc)
    win._pickers.append(
        {
            "kind": "scatter",
            "item": sc,
            "vb": vb0,
            "x": np.asarray(data["rough"], dtype=float),
            "y": np.asarray(data["pred_chi"], dtype=float),
            "colors": colors,
        }
    )
    _wire_spot_clicks(win, win._pickers[-1])
    order = list(data["bar_order"])
    y0 = np.arange(n, dtype=float) - 0.35
    widths = np.asarray([data["scores"][i] for i in order], dtype=float)
    heights = np.full(n, 0.7)
    bar_colors = [colors[i] for i in order]
    bars = pg.BarGraphItem(
        x0=np.zeros(n),
        y0=y0,
        width=widths,
        height=heights,
        brushes=[_hex_brush(c, 0.85) for c in bar_colors],
        pens=[_hex_pen("#334155", 0.4) for _ in bar_colors],
    )
    p1.addItem(bars)
    ticks = [
        (float(i), f"{win.names[si]}\n{data['params'][si]}")
        for i, si in enumerate(order)
    ]
    p1.getAxis("left").setTicks([ticks])
    p1.setYRange(-0.6, n - 0.4, padding=0)
    win._pickers.append(
        {
            "kind": "bar",
            "item": bars,
            "vb": vb1,
            "y0": y0,
            "height": heights,
            "width": widths,
            "index_map": order,
            "colors": bar_colors,
        }
    )
    win._glw.ci.layout.setColumnStretchFactor(0, 1)
    win._glw.ci.layout.setColumnStretchFactor(1, 1)
    win._remember_home()
    show_modeless_dialog(win, activate=True)
    return win


def show_param_influence_window(
    series: dict,
    *,
    title: str = "",
    save_dir: str | None = None,
) -> AnalysisPgWindow:
    data = param_influence_data(series)
    win = AnalysisPgWindow(
        title or "多日志：反演参数与指标（平滑/阻尼）",
        save_dir=save_dir,
        default_name="tt_inverse_params.png",
    )
    win.resize(1080, 920)
    win.names = list(data["names"])
    n = len(win.names)
    colors = [series_color(i) for i in range(n)]
    pred = np.array([m["pred_chi"] for m in data["metrics"]], dtype=float)
    rough = np.array([m["roughness_sum"] for m in data["metrics"]], dtype=float)
    for i, (key, xlab) in enumerate(PARAM_INFLUENCE_AXES):
        w_raw = np.array([float(m[key]) for m in data["metrics"]], dtype=float)
        w_plot = np.where(np.isfinite(w_raw) & (w_raw > 0.0), w_raw, np.nan)
        pl, vbl = win._add_plot(
            i, 0, title="pred χ²" if i == 0 else "", log_x=True, log_xs=w_plot
        )
        pr, vbr = win._add_plot(
            i, 1, title="粗糙度 R" if i == 0 else "", log_x=True, log_xs=w_plot
        )
        _style_plot(pl, "pred χ² (LSQR)", xlab)
        _style_plot(pr, "R = |Lmvh|+|Lmvv|+|Lmd|", xlab)
        for plot, vb, ys in ((pl, vbl, pred), (pr, vbr, rough)):
            xv, yv = _xy_in_view(vb, w_plot, ys)
            sc = pg.ScatterPlotItem(
                x=xv,
                y=yv,
                size=11,
                data=list(range(n)),
                brush=[_hex_brush(c, 0.9) for c in colors],
                pen=[_hex_pen("#222222", 0.5) for _ in colors],
            )
            plot.addItem(sc)
            win._pickers.append(
                {
                    "kind": "scatter",
                    "item": sc,
                    "vb": vb,
                    "x": np.asarray(w_plot, dtype=float),
                    "y": np.asarray(ys, dtype=float),
                    "colors": colors,
                }
            )
            _wire_spot_clicks(win, win._pickers[-1])
        win._glw.ci.layout.setRowStretchFactor(i, 1)
    win._glw.ci.layout.setColumnStretchFactor(0, 1)
    win._glw.ci.layout.setColumnStretchFactor(1, 1)
    win._remember_home()
    show_modeless_dialog(win, activate=True)
    return win


class SummaryTableWindow(QWidget):
    """末步汇总表（QTableWidget，原生多选）。"""

    def __init__(
        self,
        series: dict,
        *,
        rough_weight: float = 0.001,
        title: str = "",
        save_dir: str | None = None,
    ) -> None:
        super().__init__(None)
        self.setWindowTitle(title or "多日志：末步参数与指标汇总表")
        self.resize(1240, 520)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._analysis_pg = True
        self.user_pick = None
        self._syncing = False
        ranked = summary_table_ranked(series, rough_weight)
        self.names = [nm for _sc, nm, _m in ranked]
        self.hint = QLabel()
        self.hint.setStyleSheet("color:#666;font-size:11px;")
        self.table = QTableWidget(len(ranked), 9, self)
        self.table.setHorizontalHeaderLabels(
            [
                "Run",
                "s_v",
                "s_d",
                "dv",
                "dd",
                "pred χ²",
                "R",
                f"score↑ (w={rough_weight:g})",
                "RMS",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        for r, (sc, nm, m) in enumerate(ranked):
            vals = [
                nm,
                f"{m['w_sv']:.5g}",
                f"{m['w_sd']:.5g}",
                f"{m['w_dv']:.5g}",
                f"{m['w_dd']:.5g}",
                f"{m['pred_chi']:.5g}",
                f"{m['roughness_sum']:.5g}",
                f"{sc:.5g}",
                f"{m['rms_total']:.5g}",
            ]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, nm)
                    item.setToolTip(nm)
                self.table.setItem(r, c, item)
        hdr = self.table.horizontalHeader()
        self.table.resizeColumnsToContents()
        for c in range(self.table.columnCount()):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        # Run 按文件名加宽，尽量显示全名；过长才中间省略（悬停仍看全文）
        fm = self.table.fontMetrics()
        need = fm.horizontalAdvance("Run") + 32
        for nm in self.names:
            need = max(need, fm.horizontalAdvance(nm) + 32)
        need = max(need, int(self.table.columnWidth(0)))
        run_w = min(max(need, 200), 560)
        self.table.setColumnWidth(0, run_w)
        hdr.setStretchLastSection(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        lay = QVBoxLayout(self)
        cap = QLabel(
            f"按 score 升序（越小越好）。Ctrl/Shift 多选 · 右击菜单。w={rough_weight:g}"
        )
        cap.setStyleSheet("color:#111111;")
        lay.addWidget(cap)
        lay.addWidget(self.hint)
        lay.addWidget(self.table, stretch=1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton("关闭")
        btn.clicked.connect(self.close)
        row.addWidget(btn)
        lay.addLayout(row)
        self.table.itemSelectionChanged.connect(self._on_table_sel)
        self.table.customContextMenuRequested.connect(self._on_table_menu)

    def _on_table_sel(self) -> None:
        if self._syncing or not callable(self.user_pick):
            return
        names = self._selected_names()
        self.user_pick("table", names=names)

    def _on_table_menu(self, _pos) -> None:
        names = self._selected_names()
        if not names:
            row = self.table.currentRow()
            if 0 <= row < len(self.names):
                names = [self.names[row]]
        if names and callable(self.user_pick):
            self.user_pick("menu", names=names)

    def _selected_names(self) -> list[str]:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        return [self.names[r] for r in rows if 0 <= r < len(self.names)]

    def set_highlight_names(self, names: list[str] | None) -> None:
        want = set(names or [])
        self._syncing = True
        try:
            from PySide6.QtCore import QItemSelectionModel

            self.table.clearSelection()
            mode = self.table.selectionModel()
            for r, nm in enumerate(self.names):
                if nm in want:
                    idx = self.table.model().index(r, 0)
                    mode.select(
                        idx,
                        QItemSelectionModel.SelectionFlag.Select
                        | QItemSelectionModel.SelectionFlag.Rows,
                    )
        finally:
            self._syncing = False


def show_summary_table_window(
    series: dict,
    *,
    rough_weight: float = 0.001,
    title: str = "",
    save_dir: str | None = None,
) -> SummaryTableWindow:
    win = SummaryTableWindow(
        series, rough_weight=rough_weight, title=title, save_dir=save_dir
    )
    show_modeless_dialog(win, activate=True)
    return win


def show_mc_profile_window(
    zs,
    mean_v,
    std_v,
    *,
    title: str = "蒙特卡洛 — 平均速度剖面",
    save_dir: str | None = None,
) -> AnalysisPgWindow:
    z = np.asarray(zs, dtype=float)
    m = np.asarray(mean_v, dtype=float)
    s = np.asarray(std_v, dtype=float)
    win = AnalysisPgWindow(title, save_dir=save_dir, default_name="mc_profile.png")
    win.resize(560, 720)
    plot, _vb = win._add_plot(0, 0, title="Monte Carlo mean profile")
    _style_plot(plot, "z relative (km)", "V (km/s)")
    lo = plot.plot(m - s, z, pen=None)
    hi = plot.plot(m + s, z, pen=None)
    fill = pg.FillBetweenItem(lo, hi, brush=_hex_brush("#1f4e79", 0.28))
    plot.addItem(fill)
    plot.plot(m, z, pen=_hex_pen("#1f4e79", 2.0), name="mean V")
    plot.invertY(True)
    plot.addLegend(offset=(8, 8))
    win._remember_home()
    show_modeless_dialog(win, activate=True)
    return win
