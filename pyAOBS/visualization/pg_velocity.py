"""速度云图的 pyqtgraph 画法。

vedit ContourPlotWidget、obs_rtm VelCanvas 用 ``ImageItem``。
监视窗与「绘制 smesh」在 Windows 上改用 Matplotlib ``imshow``（见 inv_monitor_model / smesh_plot）。

- ``ImageItem(axisOrder="row-major")``，数据 ``(nz, nx)``
- ``ColorBarItem.setImageItem`` **只在创建时绑一次**，刷新不要再绑
- 刷新走 ``set_velocity_image``（uint8 LUT 最后套上）
- 速度场单独一块 ``GraphicsLayoutWidget``，不要和折线挤在同一 scene
- 不要 ``useOpenGL(False)``：内部 ``setViewport(QWidget())`` 会把位图画黑
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

pg.setConfigOptions(imageAxisOrder="row-major")


def lut_uint8(cmap) -> np.ndarray:
    """``(256, 3)`` uint8。浮点 0–1 LUT 直接 astype 会整表变黑。"""
    try:
        lut = np.asarray(cmap.getLookupTable(nPts=256, alpha=False, mode="byte"))
    except TypeError:
        lut = np.asarray(cmap.getLookupTable(nPts=256, alpha=False))
    if lut.ndim == 1:
        lut = np.repeat(lut[:, None], 3, axis=1)
    if lut.shape[-1] > 3:
        lut = lut[..., :3]
    if lut.dtype != np.uint8:
        mx = float(np.nanmax(lut)) if lut.size else 0.0
        lutf = lut.astype(np.float64, copy=False)
        if mx <= 1.5:
            lutf = lutf * 255.0
        lut = np.clip(np.round(lutf), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(lut, dtype=np.uint8)


def colormap_from_spec(spec: str | None):
    """返回 ``(pg.ColorMap, optional (vmin, vmax))``。空则 seismic。"""
    name = (spec or "seismic").strip() or "seismic"
    if name.lower().endswith(".cpt") and Path(name).is_file():
        try:
            from pyAOBS.visualization.show_model import parse_gmt_cpt_for_matplotlib

            mpl_cmap, zmin, zmax = parse_gmt_cpt_for_matplotlib(name)
            pos = np.linspace(0.0, 1.0, 256)
            colors = np.clip(mpl_cmap(pos), 0.0, 1.0)
            return pg.ColorMap(pos, colors), (float(zmin), float(zmax))
        except Exception:
            name = "seismic"
    elif name.lower().endswith(".cpt"):
        name = "seismic"
    for getter in (
        lambda: pg.colormap.getFromMatplotlib(name),
        lambda: pg.colormap.getFromMatplotlib("seismic"),
        lambda: pg.colormap.getFromMatplotlib("jet"),
        lambda: pg.colormap.get("CET-R4"),
    ):
        try:
            cmap = getter()
            if cmap is not None:
                return cmap, None
        except Exception:
            continue
    pos = np.array([0.0, 0.35, 0.7, 1.0])
    colors = np.array(
        [[0, 0, 128, 255], [0, 255, 255, 255], [255, 255, 0, 255], [128, 0, 0, 255]],
        dtype=np.uint8,
    )
    return pg.ColorMap(pos, colors), None


def robust_levels(data: np.ndarray, *, lo: float = 1.5, hi: float = 8.0) -> tuple[float, float]:
    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(lo), float(hi)
    a = float(np.nanpercentile(arr[finite], 1))
    b = float(np.nanpercentile(arr[finite], 99))
    if b <= a:
        b = a + 1e-6
    return a, b


def _index_uint8(data: np.ndarray, lo: float, hi: float, finite: np.ndarray) -> np.ndarray:
    span = float(hi) - float(lo)
    if span <= 0:
        span = 1e-6
    idx = np.zeros(data.shape, dtype=np.uint8)
    if np.any(finite):
        norm = np.clip((data[finite] - float(lo)) / span, 0.0, 1.0)
        idx[finite] = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return idx


def colorize_rgb(data: np.ndarray, cmap, lo: float, hi: float) -> np.ndarray:
    """标量 → uint8 RGB，不经过 ImageItem。"""
    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    lut = lut_uint8(cmap)
    idx = _index_uint8(arr, lo, hi, finite)
    return np.ascontiguousarray(lut[idx], dtype=np.uint8)


def qimage_rgb32(rgb: np.ndarray) -> QImage:
    """写入 Windows 原生 RGB32（BGRA），避开 Format_RGBA8888。

    用独立 numpy 缓冲再 ``copy()``，不写 ``QImage.bits()``（PySide6 上可能不是可写视图）。
    """
    src = np.ascontiguousarray(rgb, dtype=np.uint8)
    if src.ndim != 3 or src.shape[2] < 3:
        raise ValueError("rgb 应为 (h, w, 3)")
    h, w = int(src.shape[0]), int(src.shape[1])
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[..., 0] = src[..., 2]
    bgra[..., 1] = src[..., 1]
    bgra[..., 2] = src[..., 0]
    bgra[..., 3] = 255
    qimg = QImage(bgra.data, w, h, int(bgra.strides[0]), QImage.Format.Format_RGB32)
    if qimg.isNull():
        raise RuntimeError("QImage 分配失败")
    return qimg.copy()


class VelocityRasterItem(pg.GraphicsObject):
    """把已上色的 QImage 画到数据坐标。监视窗不用 ImageItem（Windows 上会整幅黑）。"""

    def __init__(self) -> None:
        super().__init__()
        self._qimg = QImage()
        self._rect = QRectF()
        self.setZValue(-100)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        try:
            self.setCacheMode(self.CacheMode.NoCache)
        except Exception:
            pass

    def set_field(
        self,
        data: np.ndarray,
        cmap,
        lo: float,
        hi: float,
        rect: QRectF,
    ) -> None:
        rgb = colorize_rgb(data, cmap, lo, hi)
        self.prepareGeometryChange()
        self._qimg = qimage_rgb32(rgb)
        self._rect = QRectF(rect)
        self.update()

    def clear(self) -> None:
        self.prepareGeometryChange()
        self._qimg = QImage()
        self._rect = QRectF()
        self.update()

    def boundingRect(self):
        return QRectF(self._rect)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._rect.isEmpty():
            return None, None
        if ax == 0:
            return float(self._rect.left()), float(self._rect.right())
        return float(self._rect.top()), float(self._rect.bottom())

    def paint(self, painter, _option, _widget=None) -> None:
        if self._qimg.isNull() or self._rect.isEmpty():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(self._rect, self._qimg)


def add_colorbar_plot(glw: pg.GraphicsLayoutWidget, *, row: int, col: int, label: str = "km/s"):
    """独立色标列：PlotItem + VelocityRasterItem，不用 ColorBarItem。"""
    plot = glw.addPlot(row=row, col=col)
    plot.hideAxis("bottom")
    plot.hideAxis("left")
    plot.showAxis("right")
    plot.getAxis("right").setLabel(label)
    plot.setMouseEnabled(x=False, y=False)
    try:
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.getViewBox().setMenuEnabled(False)
    except Exception:
        pass
    try:
        plot.getViewBox().setBackgroundColor((255, 255, 255, 255))
    except Exception:
        pass
    plot.setMaximumWidth(84)
    plot.setMinimumWidth(64)
    item = VelocityRasterItem()
    plot.addItem(item)
    return plot, item


def set_colorbar_raster(item: VelocityRasterItem, plot, cmap, lo: float, hi: float) -> None:
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1e-6
    ramp = np.linspace(lo_f, hi_f, 256, dtype=np.float32)[:, None]
    ramp = np.repeat(ramp, 24, axis=1)
    item.set_field(ramp, cmap, lo_f, hi_f, QRectF(0.0, lo_f, 1.0, hi_f - lo_f))
    plot.setYRange(lo_f, hi_f, padding=0)
    plot.setXRange(0.0, 1.0, padding=0)
    plot.disableAutoRange()


def attach_colorbar(
    plot: pg.PlotItem,
    glw: pg.GraphicsLayoutWidget,
    *,
    row: int,
    col: int,
    cmap,
    label: str = "km/s",
    values: tuple[float, float] = (1.5, 8.0),
) -> tuple[pg.ImageItem, Any]:
    """创建 ImageItem，并把 ColorBarItem **绑一次**。"""
    item = pg.ImageItem(axisOrder="row-major")
    item.setZValue(-100)
    plot.addItem(item)
    cbar = pg.ColorBarItem(
        values=values,
        colorMap=cmap,
        width=18,
        interactive=False,
    )
    try:
        cbar.setLabel(label)
    except Exception:
        pass
    glw.addItem(cbar, row=row, col=col)
    cbar.setImageItem(item)
    try:
        cbar.setMaximumWidth(84)
        cbar.setMinimumWidth(64)
    except Exception:
        pass
    return item, cbar


def set_velocity_image(
    item: pg.ImageItem,
    data: np.ndarray,
    *,
    levels: tuple[float, float],
    rect: QRectF,
    cmap,
    cbar=None,
) -> None:
    """刷新速度栅格。禁止在此调用 ``cbar.setImageItem``。"""
    if cmap is None:
        cmap, _ = colormap_from_spec("viridis")
    lo, hi = float(levels[0]), float(levels[1])
    if hi <= lo:
        hi = lo + 1e-6
    img = np.ascontiguousarray(data, dtype=np.float32)
    item.setImage(img, levels=(lo, hi))
    try:
        item.setColorMap(cmap)
    except Exception:
        pass
    item.setRect(rect)
    item.setZValue(-100)
    if cbar is not None:
        try:
            cbar.setLevels((lo, hi))
        except Exception:
            pass
        try:
            cbar.setColorMap(cmap)
        except Exception:
            pass
    # ColorBarItem.setColorMap 会改写 LUT；uint8 必须最后套
    try:
        item.setLookupTable(lut_uint8(cmap))
    except Exception:
        pass


class VelocityFieldWidget(QWidget):
    """单幅速度场 + 色标（绘制 smesh 等工具窗）。"""

    def __init__(
        self,
        parent=None,
        *,
        cmap_spec: str = "seismic",
        cbar_label: str = "Velocity (km/s)",
        on_save=None,
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        bar = QHBoxLayout()
        hint = QLabel("滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位")
        hint.setStyleSheet("color:#666;font-size:11px;")
        bar.addWidget(hint)
        bar.addStretch(1)
        if on_save is not None:
            btn_save = QPushButton("保存图像…")
            btn_save.clicked.connect(on_save)
            bar.addWidget(btn_save)
        btn = QPushButton("Reset View")
        btn.clicked.connect(self.reset_view)
        bar.addWidget(btn)
        lay.addLayout(bar)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        try:
            self.glw.setViewportUpdateMode(self.glw.ViewportUpdateMode.FullViewportUpdate)
            self.glw.setCacheMode(self.glw.CacheModeFlag.CacheNone)
        except Exception:
            pass
        lay.addWidget(self.glw, stretch=1)
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "模型距离", units="km")
        self.plot.setLabel("left", "深度", units="km")
        self.plot.invertY(True)
        try:
            self.plot.setMenuEnabled(False)
            self.plot.getViewBox().setMenuEnabled(False)
        except Exception:
            pass
        cmap0, _ = colormap_from_spec(cmap_spec)
        self._img = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self._img)
        self._cbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap0, width=18)
        try:
            self._cbar.setLabel(cbar_label)
        except Exception:
            pass
        self.glw.addItem(self._cbar, row=0, col=1)
        self._cbar.setImageItem(self._img)
        try:
            self._cbar.setMaximumWidth(84)
            self._cbar.setMinimumWidth(64)
        except Exception:
            pass
        self.glw.ci.layout.setColumnStretchFactor(0, 100)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)
        self._home: Optional[tuple[float, float, float, float]] = None
        self._overlay: list = []

    def reset_view(self) -> None:
        if self._home is None:
            self.plot.enableAutoRange()
            return
        x0, x1, z0, z1 = self._home
        self.plot.setXRange(x0, x1, padding=0)
        self.plot.setYRange(z0, z1, padding=0)

    def clear_overlays(self) -> None:
        for it in self._overlay:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._overlay.clear()

    def set_field(
        self,
        vel: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
        *,
        cmap_spec: str = "seismic",
        title: str = "",
        overlays: Sequence[dict] | None = None,
    ) -> None:
        vel = np.asarray(vel, dtype=float)
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        if vel.ndim == 2 and vel.shape == (x.size, z.size):
            vel = vel.T
        cmap, cpt_lv = colormap_from_spec(cmap_spec)
        lo, hi = cpt_lv if cpt_lv is not None else robust_levels(vel)
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
        self.clear_overlays()
        set_velocity_image(
            self._img,
            vel,
            levels=(lo, hi),
            rect=QRectF(xmin, zmin, xmax - xmin, zmax - zmin),
            cmap=cmap,
            cbar=self._cbar,
        )
        if overlays:
            for ov in overlays:
                it = pg.PlotCurveItem(
                    np.asarray(ov["x"], dtype=float),
                    np.asarray(ov["z"], dtype=float),
                    pen=pg.mkPen(
                        ov.get("color", "k"),
                        width=float(ov.get("width", 1.0)),
                        style=ov.get("style", Qt.PenStyle.SolidLine),
                    ),
                )
                self.plot.addItem(it)
                self._overlay.append(it)
        self.plot.setTitle(title)
        self.plot.setXRange(xmin, xmax, padding=0)
        self.plot.setYRange(zmin, zmax, padding=0)
        self._home = (xmin, xmax, zmin, zmax)
