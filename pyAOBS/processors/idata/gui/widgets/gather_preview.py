# -*- coding: utf-8 -*-
"""道集预览：全道密度图；按 trid 选分量；横轴 fldr/cdp/ep；点击高亮。"""

from __future__ import annotations

from typing import List, Optional, Sequence, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

try:
    from pyAOBS.utils.qt_combo import connect_combo_deferred
except ImportError:
    def connect_combo_deferred(combo, slot):  # type: ignore
        combo.currentIndexChanged.connect(slot)

try:
    import pyqtgraph as pg

    _HAS_PG = True
except Exception:
    pg = None  # type: ignore
    _HAS_PG = False

if TYPE_CHECKING:
    from ..services.segy_dataset import SegyDataset

_X_KEYS = ("fldr", "cdp", "ep", "offset", "gx", "sx")


class GatherPreviewWidget(QWidget):
    """全道道集预览；点击发出 trace_clicked(row)。"""

    trace_clicked = Signal(int)
    log_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ds: Optional["SegyDataset"] = None
        self._row_map: List[int] = []
        self._x_vals: List[float] = []
        self._dt = 0.004
        self._cache_key: Optional[tuple] = None
        self._highlight_row: Optional[int] = None
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("分量"))
        self.trid_combo = QComboBox()
        self.trid_combo.setMinimumWidth(100)
        connect_combo_deferred(self.trid_combo, self._on_controls_changed)
        bar.addWidget(self.trid_combo)

        bar.addWidget(QLabel("横轴"))
        self.x_combo = QComboBox()
        for key in _X_KEYS:
            self.x_combo.addItem(key, key)
        connect_combo_deferred(self.x_combo, self._on_controls_changed)
        bar.addWidget(self.x_combo)
        bar.addStretch(1)
        root.addLayout(bar)

        self._info = QLabel("未加载道集")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color:#334155;")
        root.addWidget(self._info)

        self._plot = None
        self._img = None
        self._hl_line = None
        if _HAS_PG:
            self._plot = pg.PlotWidget()
            self._plot.setBackground("w")
            self._plot.showGrid(x=True, y=True, alpha=0.25)
            self._plot.setLabel("bottom", "fldr")
            self._plot.setLabel("left", "time", units="s")
            self._plot.invertY(True)
            self._img = pg.ImageItem()
            self._plot.addItem(self._img)
            self._hl_line = pg.InfiniteLine(
                pos=0, angle=90, pen=pg.mkPen("#16a34a", width=2), movable=False
            )
            self._hl_line.hide()
            self._plot.addItem(self._hl_line)
            self._plot.scene().sigMouseClicked.connect(self._on_click)
            root.addWidget(self._plot, stretch=1)
        else:
            root.addWidget(QLabel("pyqtgraph 未安装，无法预览道集"), stretch=1)

    # ---- public API ----
    def clear(self) -> None:
        self._ds = None
        self._row_map = []
        self._x_vals = []
        self._cache_key = None
        self._highlight_row = None
        self._info.setText("未加载道集")
        self.trid_combo.blockSignals(True)
        self.trid_combo.clear()
        self.trid_combo.blockSignals(False)
        if self._img is not None:
            self._img.clear()
        if self._hl_line is not None:
            self._hl_line.hide()
        if self._plot is not None:
            self._plot.getAxis("bottom").setTicks(None)

    def set_dataset(self, ds: Optional["SegyDataset"]) -> None:
        self.clear()
        self._ds = ds
        if ds is None or not ds.is_open:
            return
        self._populate_trid()
        self.reload(force=True)

    def reload(self, *, force: bool = False, highlight: Optional[int] = None) -> None:
        if self._ds is None or not self._ds.is_open:
            return
        if highlight is not None:
            self._highlight_row = int(highlight)
        trid = self.trid_combo.currentData()
        xkey = str(self.x_combo.currentData() or "fldr")
        key = (trid, xkey, self._ds.ntraces)
        if not force and key == self._cache_key and self._row_map:
            self.set_highlight(self._highlight_row)
            return
        self._loading = True
        try:
            self._info.setText("正在加载道集…")
            rows = self._collect_rows(trid)
            if not rows:
                self._row_map = []
                self._x_vals = []
                self._cache_key = key
                if self._img is not None:
                    self._img.clear()
                self._info.setText("无匹配道（检查 trid）")
                return
            # 按横轴字段排序；同值保持原序（gx/sx 用 scalco 后物理坐标）
            keyed = []
            for i in rows:
                xv = self._x_value(i, xkey)
                keyed.append((xv, i))
            keyed.sort(key=lambda t: (t[0], t[1]))
            rows = [i for _, i in keyed]
            x_vals = [xv for xv, _ in keyed]

            dt_us = int(self._ds.get_header(rows[0]).get("dt", 0) or 0)
            # 批量读样本（单次打开文件）
            if hasattr(self._ds, "read_samples_many"):
                raw_cols = self._ds.read_samples_many(rows)
            else:
                raw_cols = [self._ds.read_samples(i) for i in rows]
            cols: List[np.ndarray] = []
            ns_ref = 0
            for y0 in raw_cols:
                y = np.asarray(y0, dtype=np.float32).ravel()
                if ns_ref == 0:
                    ns_ref = int(y.size)
                if y.size != ns_ref:
                    if y.size > ns_ref:
                        y = y[:ns_ref]
                    else:
                        y = np.pad(y, (0, ns_ref - y.size))
                cols.append(y)
            data = np.column_stack(cols)  # (ns, ntr)
            dt_s = float(dt_us) * 1e-6 if dt_us > 0 else 0.004
            # 显示抽稀：长道只画约 2000 点，加快开图
            ns_full = int(data.shape[0])
            max_pts = 2000
            if ns_full > max_pts:
                step = max(1, ns_full // max_pts)
                data = data[::step, :]
                dt_s *= float(step)
            self._row_map = rows
            self._x_vals = x_vals
            self._dt = dt_s
            self._cache_key = key
            self._show_image(data, xkey=xkey)
            self.set_highlight(self._highlight_row)
            self._update_info(self._highlight_row, None)
        except Exception as exc:
            self._info.setText(f"道集加载失败: {exc}")
            self.log_message.emit(f"[道集] 加载失败: {exc}")
        finally:
            self._loading = False

    def ensure_row_visible(self, row: int) -> None:
        """若当前 trid 过滤不含该道，切换到该道的 trid 并重载。"""
        if self._ds is None or not self._ds.is_open:
            return
        if row < 0 or row >= self._ds.ntraces:
            return
        if row in self._row_map:
            return
        trid = int(self._ds.get_header(row).get("trid", 0) or 0)
        idx = self.trid_combo.findData(trid)
        if idx < 0:
            # 退回「全部」
            idx = self.trid_combo.findData(None)
        if idx >= 0 and self.trid_combo.currentIndex() != idx:
            self.trid_combo.blockSignals(True)
            self.trid_combo.setCurrentIndex(idx)
            self.trid_combo.blockSignals(False)
            self.reload(force=True, highlight=row)

    def set_highlight(self, row: Optional[int]) -> None:
        self._highlight_row = int(row) if row is not None else None
        if self._hl_line is None:
            return
        if row is not None and row in self._row_map:
            col = self._row_map.index(row)
            self._hl_line.setPos(float(col))
            self._hl_line.show()
            self._update_info(row, col)
        else:
            self._hl_line.hide()
            self._update_info(None, None)

    # ---- internals ----
    def _populate_trid(self) -> None:
        assert self._ds is not None
        uniq = sorted(
            {int(th.get("trid", 0) or 0) for th in self._ds.headers}
        )
        self.trid_combo.blockSignals(True)
        self.trid_combo.clear()
        self.trid_combo.addItem("全部", None)
        for t in uniq:
            self.trid_combo.addItem(f"trid={t}", t)
        # 默认：优先 trid=1（地震道），否则第一个非空分量，再否则「全部」
        default_idx = 0
        if 1 in uniq:
            default_idx = self.trid_combo.findData(1)
        elif len(uniq) == 1:
            default_idx = self.trid_combo.findData(uniq[0])
        elif uniq:
            default_idx = self.trid_combo.findData(uniq[0])
        if default_idx < 0:
            default_idx = 0
        self.trid_combo.setCurrentIndex(default_idx)
        self.trid_combo.blockSignals(False)

    def _collect_rows(self, trid) -> List[int]:
        assert self._ds is not None
        if trid is None:
            return list(range(self._ds.ntraces))
        t = int(trid)
        return [
            i
            for i, th in enumerate(self._ds.headers)
            if int(th.get("trid", 0) or 0) == t
        ]

    def _x_value(self, row: int, xkey: str) -> float:
        """横轴取值：offset/gx/sx 优先物理量（scalco 后）。"""
        assert self._ds is not None
        if xkey in ("gx", "sx"):
            phy = self._ds.physical_xy(row)
            return float(phy.get(xkey, 0.0))
        th = self._ds.get_header(row)
        if xkey == "offset":
            # offset 不受 scalco；直接用道头
            return float(th.get("offset", 0) or 0)
        return float(th.get(xkey, 0) or 0)

    def _show_image(self, data: np.ndarray, *, xkey: str) -> None:
        arr = np.asarray(data, dtype=np.float32)
        ns, nt = arr.shape
        flat = arr[np.isfinite(arr)]
        if flat.size == 0 or self._img is None or self._plot is None:
            return
        lo, hi = np.percentile(flat, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        clipped = np.clip(arr, lo, hi)
        self._img.setImage(clipped.T, autoLevels=True)
        self._img.setRect(QRectF(-0.5, 0.0, float(nt), float(ns) * self._dt))
        self._plot.setLabel("bottom", xkey)
        self._apply_x_ticks(self._x_vals)
        # 视野
        self._plot.setXRange(-0.5, float(nt) - 0.5, padding=0)
        self._plot.setYRange(0.0, float(ns) * self._dt, padding=0)

    def _format_x_tick(self, v: float) -> str:
        xkey = str(self.x_combo.currentData() or "fldr")
        if xkey in ("gx", "sx", "offset"):
            if abs(v) >= 1000:
                return f"{v:.0f}"
            if abs(v) >= 10:
                return f"{v:.1f}"
            return f"{v:.2f}"
        return str(int(round(v)))

    def _apply_x_ticks(self, x_vals: Sequence[float]) -> None:
        if self._plot is None:
            return
        n = len(x_vals)
        if n == 0:
            self._plot.getAxis("bottom").setTicks(None)
            return
        step = max(1, n // 10)
        ticks = [(float(i), self._format_x_tick(x_vals[i])) for i in range(0, n, step)]
        if ticks[-1][0] != float(n - 1):
            ticks.append((float(n - 1), self._format_x_tick(x_vals[n - 1])))
        self._plot.getAxis("bottom").setTicks([ticks])

    def _update_info(self, row: Optional[int], col: Optional[int]) -> None:
        n = len(self._row_map)
        xkey = str(self.x_combo.currentData() or "fldr")
        trid_lab = self.trid_combo.currentText()
        base = f"{trid_lab}  traces={n}  横轴={xkey}"
        if row is None or self._ds is None or not self._ds.is_open:
            self._info.setText(base + "  |  点击道集选道")
            return
        th = self._ds.get_header(row)
        xv = self._x_vals[col] if col is not None and 0 <= col < len(self._x_vals) else th.get(xkey)
        self._info.setText(
            f"{base}  |  选中 trace={row}  "
            f"trid={th.get('trid')}  fldr={th.get('fldr')}  "
            f"cdp={th.get('cdp')}  ep={th.get('ep')}  "
            f"offset={th.get('offset')}  {xkey}={xv}"
        )

    def _on_controls_changed(self, *_args) -> None:
        if self._loading or self._ds is None:
            return
        self.reload(force=True, highlight=self._highlight_row)

    def _on_click(self, ev) -> None:
        if self._plot is None or not self._row_map:
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if not self._plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        # 仅响应绘图区点击
        vb = self._plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = vb.mapSceneToView(ev.scenePos())
        col = int(round(float(pos.x())))
        if 0 <= col < len(self._row_map):
            row = self._row_map[col]
            self.set_highlight(row)
            self.trace_clicked.emit(row)
