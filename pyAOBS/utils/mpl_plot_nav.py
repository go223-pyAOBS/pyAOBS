"""Matplotlib canvas：pyqtgraph ViewBox.PanMode 式交互（跨 GUI 共享）。

- 左键拖：平移
- 右键拖：连续缩放（非框选）
- 滚轮：以光标为中心缩放
- 双击 / Reset View：复位到 **home**（首次完整显示的数据范围）

完整重绘后请调用 ``nav.remember_home_views()``（或 ``notify_plot_updated(canvas)``），
以便 Reset 对应当前数据范围。
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def install_plot_nav_bar(
    parent_layout: QVBoxLayout | QHBoxLayout,
    canvas,
    *,
    parent: QWidget | None = None,
    claim_left: Optional[Callable[[], bool]] = None,
    claim_right: Optional[Callable[[], bool]] = None,
    insert_index: int | None = None,
) -> "PyqtgraphStyleNav":
    """在 layout 中加入提示条 + Reset View，并挂上 ``PyqtgraphStyleNav``。"""
    try:
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    except Exception:
        pass

    bar = QWidget(parent)
    row = QHBoxLayout(bar)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    hint = QLabel("滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位")
    hint.setStyleSheet("color: #666; font-size: 11px;")
    row.addWidget(hint)
    row.addStretch(1)
    btn = QPushButton("Reset View")
    btn.setToolTip("复位到数据范围（等同双击空白）")
    row.addWidget(btn)

    if insert_index is None:
        parent_layout.addWidget(bar)
    else:
        parent_layout.insertWidget(insert_index, bar)

    nav = PyqtgraphStyleNav(canvas, claim_left=claim_left, claim_right=claim_right)
    btn.clicked.connect(nav.reset_view)
    return nav


def is_colorbar_axes(ax) -> bool:
    """色标轴不参与平移/缩放/视窗记忆，否则换 CPT 后右侧范围会被旧 ylim 盖住。"""
    if ax is None:
        return False
    if getattr(ax, "_colorbar", None) is not None:
        return True
    try:
        lab = str(ax.get_label() or "")
    except Exception:
        lab = ""
    return lab in {"<colorbar>", "cax"}


def notify_plot_updated(canvas) -> None:
    """完整重绘后调用：把当前视图锁为 Reset 的 home。"""
    nav = getattr(canvas, "_mpl_plot_nav", None)
    if nav is not None:
        nav.remember_home_views()
        nav.remember_current_views()


class PyqtgraphStyleNav:
    """挂到已有 ``FigureCanvasQTAgg`` 上。"""

    def __init__(
        self,
        canvas,
        *,
        claim_left: Optional[Callable[[], bool]] = None,
        claim_right: Optional[Callable[[], bool]] = None,
        on_right_click: Optional[Callable] = None,
        auto_home_on_draw: bool = True,
    ) -> None:
        self.canvas = canvas
        self.claim_left = claim_left
        self.claim_right = claim_right
        self.on_right_click = on_right_click
        self._home_views: Optional[list[tuple[tuple[float, float], tuple[float, float]]]] = None
        self._saved_views: Optional[list[tuple[tuple[float, float], tuple[float, float]]]] = None
        self._user_view_active = False
        self._drag_ax = None
        self._drag_mode: Optional[str] = None
        self._drag_last_px: Optional[tuple[float, float]] = None
        self._drag_center: Optional[tuple[float, float]] = None
        self._click_px: Optional[tuple[float, float]] = None
        self._click_moved = False
        self._refresh_home_on_draw = False

        try:
            canvas._mpl_plot_nav = self  # type: ignore[attr-defined]
        except Exception:
            pass

        self._cid_scroll = canvas.mpl_connect("scroll_event", self._on_scroll)
        self._cid_press = canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_release = canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion = canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_draw = None
        if auto_home_on_draw:
            self._cid_draw = canvas.mpl_connect("draw_event", self._on_draw)

    @property
    def busy(self) -> bool:
        return self._drag_mode is not None

    @staticmethod
    def _claim(fn, event) -> bool:
        if fn is None:
            return False
        try:
            return bool(fn(event))
        except TypeError:
            return bool(fn())

    def disconnect(self) -> None:
        for cid in (
            getattr(self, "_cid_scroll", None),
            getattr(self, "_cid_press", None),
            getattr(self, "_cid_release", None),
            getattr(self, "_cid_motion", None),
            getattr(self, "_cid_draw", None),
        ):
            if cid is not None:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass

    def _nav_axes(self):
        fig = getattr(self.canvas, "figure", None)
        if fig is None:
            return []
        return [ax for ax in fig.axes if not is_colorbar_axes(ax)]

    def capture_views(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        out: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for ax in self._nav_axes():
            try:
                out.append((tuple(ax.get_xlim()), tuple(ax.get_ylim())))  # type: ignore[arg-type]
            except Exception:
                continue
        return out

    def remember_home_views(self) -> None:
        """记录 Reset / 双击的目标范围（不被缩放覆盖）。"""
        views = self.capture_views()
        if views:
            self._home_views = views

    def schedule_home_refresh(self) -> None:
        """标记：下一次 draw 后把当前视图锁为 home（用于完整重绘）。"""
        self._refresh_home_on_draw = True
        self._home_views = None

    def remember_current_views(self) -> None:
        views = self.capture_views()
        if views:
            self._saved_views = views
            self._user_view_active = True

    def clear_saved_views(self) -> None:
        self._saved_views = None
        self._user_view_active = False
        self._home_views = None

    def restore_saved_views(self) -> bool:
        if not self._user_view_active or not self._saved_views:
            return False
        axes = self._nav_axes()
        if len(axes) != len(self._saved_views):
            return False
        for ax, (xlim, ylim) in zip(axes, self._saved_views):
            try:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            except Exception:
                return False
        return True

    def reset_view(self) -> None:
        """复位到 home；若无 home 则 relim+autoscale 并记为 home。"""
        axes = self._nav_axes()
        if self._home_views and len(self._home_views) == len(axes):
            for ax, (xlim, ylim) in zip(axes, self._home_views):
                try:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                except Exception:
                    pass
        else:
            for ax in axes:
                try:
                    ax.relim()
                    ax.autoscale()
                except Exception:
                    pass
            self.remember_home_views()
        self.remember_current_views()
        self.canvas.draw_idle()

    def snapshot_before_clear(self) -> None:
        if self._user_view_active:
            views = self.capture_views()
            if views:
                self._saved_views = views

    def _on_draw(self, _event) -> None:
        if self._home_views is None or self._refresh_home_on_draw:
            self.remember_home_views()
            self._refresh_home_on_draw = False

    def _reset_drag(self) -> None:
        self._drag_ax = None
        self._drag_mode = None
        self._drag_last_px = None
        self._drag_center = None
        self._click_px = None
        self._click_moved = False

    @staticmethod
    def _is_log_axis(ax, which: str) -> bool:
        try:
            scale = ax.get_xscale() if which == "x" else ax.get_yscale()
        except Exception:
            return False
        return str(scale).lower() in ("log", "logit")

    @staticmethod
    def _zoom_span(
        lo: float, hi: float, c: float, s: float, *, log_scale: bool
    ) -> tuple[float, float]:
        """以 c 为中心按比例 s 缩放区间；log 轴在对数空间缩放以免出现 ≤0。"""
        if log_scale:
            if lo <= 0 or hi <= 0:
                return lo, hi
            from math import exp, log

            if c <= 0:
                c = (lo * hi) ** 0.5
            l0, l1, lc = log(lo), log(hi), log(c)
            nlo, nhi = exp(lc - (lc - l0) * s), exp(lc + (l1 - lc) * s)
            if nlo <= 0 or nhi <= 0 or nlo == nhi:
                return lo, hi
            return nlo, nhi
        lo2 = c - (c - lo) * s
        hi2 = c + (hi - c) * s
        if lo2 == hi2:
            return lo, hi
        return lo2, hi2

    @staticmethod
    def _pan_span(
        lo: float, hi: float, frac: float, *, log_scale: bool
    ) -> tuple[float, float]:
        """``frac`` 为视窗宽度比例（与线性平移 ``Δ = frac * span`` 同号）。"""
        if log_scale:
            if lo <= 0 or hi <= 0:
                return lo, hi
            from math import exp, log

            l0, l1 = log(lo), log(hi)
            shift = frac * (l1 - l0)
            nlo, nhi = exp(l0 - shift), exp(l1 - shift)
            if nlo <= 0 or nhi <= 0:
                return lo, hi
            return nlo, nhi
        span = hi - lo
        return lo - frac * span, hi - frac * span

    def _on_scroll(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if is_colorbar_axes(event.inaxes):
            return
        if self._drag_mode is not None:
            return
        ax = event.inaxes
        scale = 0.85 if event.button == "up" else 1.0 / 0.85
        try:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
        except Exception:
            return
        xc, yc = float(event.xdata), float(event.ydata)
        ax.set_xlim(
            self._zoom_span(x0, x1, xc, scale, log_scale=self._is_log_axis(ax, "x"))
        )
        ax.set_ylim(
            self._zoom_span(y0, y1, yc, scale, log_scale=self._is_log_axis(ax, "y"))
        )
        self.remember_current_views()
        self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes is None:
            return
        if is_colorbar_axes(event.inaxes):
            return

        if getattr(event, "dblclick", False) and event.button == 1:
            if self._claim(self.claim_left, event):
                return
            self._reset_drag()
            self.reset_view()
            return

        if event.x is None or event.y is None:
            return

        if event.button == 1:
            if self._claim(self.claim_left, event):
                return
            mode = "pan"
        elif event.button == 3:
            if self._claim(self.claim_right, event):
                return
            mode = "scale"
        else:
            return

        self._drag_ax = event.inaxes
        self._drag_mode = mode
        self._drag_last_px = (float(event.x), float(event.y))
        self._click_px = self._drag_last_px
        self._click_moved = False
        if mode == "scale":
            if event.xdata is not None and event.ydata is not None:
                self._drag_center = (float(event.xdata), float(event.ydata))
            else:
                xl, yl = event.inaxes.get_xlim(), event.inaxes.get_ylim()
                ax = event.inaxes
                if self._is_log_axis(ax, "x") and xl[0] > 0 and xl[1] > 0:
                    cx = (xl[0] * xl[1]) ** 0.5
                else:
                    cx = 0.5 * (xl[0] + xl[1])
                if self._is_log_axis(ax, "y") and yl[0] > 0 and yl[1] > 0:
                    cy = (yl[0] * yl[1]) ** 0.5
                else:
                    cy = 0.5 * (yl[0] + yl[1])
                self._drag_center = (cx, cy)

    def _on_motion(self, event) -> None:
        if self._drag_ax is None or self._drag_mode is None or self._drag_last_px is None:
            return
        if event.x is None or event.y is None:
            return

        ax = self._drag_ax
        x_px, y_px = float(event.x), float(event.y)
        dx_px = x_px - self._drag_last_px[0]
        dy_px = y_px - self._drag_last_px[1]
        self._drag_last_px = (x_px, y_px)
        if self._click_px is not None:
            adx = x_px - self._click_px[0]
            ady = y_px - self._click_px[1]
            if adx * adx + ady * ady > 64.0:
                self._click_moved = True

        try:
            bbox = ax.get_window_extent()
            w = max(float(bbox.width), 1.0)
            h = max(float(bbox.height), 1.0)
            xl = list(ax.get_xlim())
            yl = list(ax.get_ylim())
        except Exception:
            return

        if self._drag_mode == "pan":
            ax.set_xlim(
                *self._pan_span(
                    xl[0],
                    xl[1],
                    dx_px / w,
                    log_scale=self._is_log_axis(ax, "x"),
                )
            )
            ax.set_ylim(
                *self._pan_span(
                    yl[0],
                    yl[1],
                    dy_px / h,
                    log_scale=self._is_log_axis(ax, "y"),
                )
            )
            self.canvas.draw_idle()
            return

        if self._drag_mode == "scale":
            sx = 1.02 ** (-dx_px * 0.2)
            sy = 1.02 ** (dy_px * 0.2)
            cx, cy = self._drag_center or (0.5 * (xl[0] + xl[1]), 0.5 * (yl[0] + yl[1]))
            ax.set_xlim(
                *self._zoom_span(
                    xl[0], xl[1], cx, sx, log_scale=self._is_log_axis(ax, "x")
                )
            )
            ax.set_ylim(
                *self._zoom_span(
                    yl[0], yl[1], cy, sy, log_scale=self._is_log_axis(ax, "y")
                )
            )
            self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        btn = getattr(event, "button", None)
        btn_n = getattr(btn, "value", btn)
        click = (
            btn_n == 3
            and not self._click_moved
            and callable(self.on_right_click)
        )
        if self._drag_mode is not None:
            self.remember_current_views()
        self._reset_drag()
        if click:
            try:
                self.on_right_click(event)
            except Exception:
                pass
