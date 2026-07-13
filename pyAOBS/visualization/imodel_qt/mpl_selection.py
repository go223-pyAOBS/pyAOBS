"""Matplotlib 点选 / PolygonSelector —— 行为对齐 Tk InteractionMixin（基于 imodel.PointSelector）。"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon

from pyAOBS.visualization.imodel import PointSelector


class MplWorkbenchSelection:
    """管理主图上点选与多边形状态。"""

    def __init__(
        self,
        ax,
        canvas,
        *,
        redraw_plot: Callable[[], None],
    ) -> None:
        self.ax = ax
        self.canvas = canvas
        self._redraw_plot = redraw_plot
        self.point_selector: Optional[PointSelector] = None
        self.polygon_selector: Optional[PolygonSelector] = None
        self.selected_polygons: List[Any] = []
        self.polygon_patches: List[Polygon] = []
        self.polygon_sample_artists: List[Any] = []

    def forget_polygon_sample_artists(self) -> None:
        """主图 ax.clear() 后调用：丢弃引用（与子图Tk polygon_sample_artists一致）。"""
        self.polygon_sample_artists.clear()

    def remove_polygon_sample_artists(self) -> None:
        """从轴上移除多边形采样标记（对齐 Tk clear_selections）。"""
        for artist in self.polygon_sample_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.polygon_sample_artists.clear()

    def add_polygon_sample_markers(
        self, xs: Sequence[float], zs: Sequence[float], *, redraw: bool = True
    ) -> None:
        if not xs or not zs:
            return
        sc = self.ax.scatter(
            list(xs),
            list(zs),
            c="orange",
            marker="s",
            s=50,
            edgecolors="darkorange",
            linewidths=1,
            alpha=0.7,
            zorder=5,
            label="Polygon Samples",
        )
        self.polygon_sample_artists.append(sc)
        if redraw:
            self.canvas.draw()

    def enable_point_selection(
        self,
        on_pick: Callable[[float, float, int], None],
        log_fn: Callable[[str], None],
    ) -> None:
        self._disconnect_point_selector()
        self.point_selector = PointSelector(self.ax, callback=on_pick)
        log_fn(
            "Point selection enabled: Left-click add, Right-click remove, "
            "D delete last, C clear all"
        )

    def enable_polygon_selection(self, log_fn: Callable[[str], None]) -> None:
        self._disconnect_polygon_selector_gui()
        self.polygon_selector = None

        def on_poly(vertices):
            verts = vertices
            if len(verts) > 0:
                first_point = verts[0]
                last_point = verts[-1]
                if (
                    len(verts) > 1
                    and abs(first_point[0] - last_point[0]) < 1e-6
                    and abs(first_point[1] - last_point[1]) < 1e-6
                ):
                    verts = verts[:-1]
            self.selected_polygons.append(verts)
            poly = Polygon(
                verts,
                closed=True,
                fill=False,
                edgecolor="red",
                linewidth=2,
                alpha=0.7,
            )
            poly.set_label("_imodel_polygon")
            self.ax.add_patch(poly)
            self.polygon_patches.append(poly)
            self.canvas.draw()
            log_fn(f"Polygon selected with {len(verts)} vertices")

        self.polygon_selector = PolygonSelector(
            self.ax,
            on_poly,
            useblit=True,
            props=dict(color="red", linestyle="-", linewidth=2, alpha=0.5),
        )
        log_fn("Polygon selection enabled: Left-click vertices, Right-click to complete")

    def clear_everything(self, log_fn: Callable[[str], None]) -> None:
        self.remove_polygon_sample_artists()
        if self.point_selector is not None:
            self.point_selector.clear_points()
        self._disconnect_point_selector()
        self._disconnect_polygon_selector_gui()
        for poly_patch in self.polygon_patches:
            try:
                poly_patch.remove()
            except Exception:
                pass
        self.polygon_patches.clear()
        self.selected_polygons.clear()
        self.point_selector = None
        self.polygon_selector = None
        try:
            self._redraw_plot()
        except Exception:
            pass
        log_fn("All selections cleared")

    def selected_points_xy(self) -> List[Tuple[float, float]]:
        if self.point_selector is None:
            return []
        return list(self.point_selector.get_points())

    def _disconnect_point_selector(self) -> None:
        if self.point_selector is None:
            return
        if hasattr(self.point_selector, "cid_click"):
            self.ax.figure.canvas.mpl_disconnect(self.point_selector.cid_click)
        if hasattr(self.point_selector, "cid_key"):
            self.ax.figure.canvas.mpl_disconnect(self.point_selector.cid_key)
        self.point_selector = None

    def _disconnect_polygon_selector_gui(self) -> None:
        sel = self.polygon_selector
        if not sel:
            return
        try:
            sel.disconnect_events()
        except Exception:
            pass
        for attr_name in ("_line", "_lines", "line", "lines"):
            if hasattr(sel, attr_name):
                attr = getattr(sel, attr_name)
                if attr is None:
                    continue
                try:
                    if isinstance(attr, list):
                        for item in attr:
                            if hasattr(item, "remove"):
                                item.remove()
                    elif hasattr(attr, "remove"):
                        attr.remove()
                except Exception:
                    pass
        try:
            sel.vertices = []
        except Exception:
            pass
        try:
            if hasattr(sel, "set_active"):
                sel.set_active(False)
        except Exception:
            pass
