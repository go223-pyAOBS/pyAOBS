# -*- coding: utf-8 -*-
"""速度剖面预览：pyqtgraph ImageItem + Contours / Interfaces（v.in）。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import connect_combo_deferred

try:
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover
    pg = None  # type: ignore
    _PG_ERR = exc
else:
    _PG_ERR = None


def _iface_idx_from_combo(text: str) -> Optional[int]:
    t = str(text or "").strip()
    if t.startswith("Interface "):
        try:
            return int(t.replace("Interface ", "")) - 1
        except ValueError:
            return None
    return None


def _dataset_to_vel_rect(ds) -> Tuple[np.ndarray, float, float, float, float]:
    """xr.Dataset → vel(nz,nx), ox, dx, oz, dz。"""
    from pyAOBS.visualization.obs_rtm_qt.services.model_import import dataset_to_vel_meta

    vel, meta = dataset_to_vel_meta(ds)
    return (
        vel,
        float(meta["o2"]),
        float(meta["d2"]),
        float(meta["o1"]),
        float(meta["d1"]),
    )


def _robust_levels(vel: np.ndarray, n: int = 10) -> Optional[np.ndarray]:
    flat = np.asarray(vel, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    if flat.size > 200_000:
        flat = flat[:: max(1, flat.size // 200_000)]
    vmin = float(np.percentile(flat, 2))
    vmax = float(np.percentile(flat, 98))
    if vmax <= vmin:
        return None
    return np.linspace(vmin, vmax, int(n))


def _data_vrange(vel: np.ndarray) -> Tuple[float, float]:
    flat = np.asarray(vel, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.5, 8.0
    if flat.size > 200_000:
        flat = flat[:: max(1, flat.size // 200_000)]
    vmin = float(np.percentile(flat, 1))
    vmax = float(np.percentile(flat, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _nice_contour_levels(vel: np.ndarray) -> List[float]:
    """常用速度等值线（含 5/6/7/8），落在数据范围内。"""
    vmin, vmax = _data_vrange(vel)
    # 优先整 km/s；水速附近加 1.5
    candidates = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)
    levels = [float(c) for c in candidates if vmin < float(c) < vmax]
    # 至少保证 5–8 中落在范围内的都画
    for t in (5.0, 6.0, 7.0, 8.0):
        if vmin < t < vmax and t not in levels:
            levels.append(t)
    levels = sorted(set(levels))
    if levels:
        return levels
    lv = _robust_levels(vel, 8)
    return [] if lv is None else [float(x) for x in lv]


def _contour_polylines(
    vel: np.ndarray,
    ox: float,
    dx: float,
    oz: float,
    dz: float,
    levels: Sequence[float],
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """等值线折线列表 [(x, z, level), ...]，世界坐标 km。大图先抽稀。"""
    vel = np.asarray(vel, dtype=float)
    nz, nx = vel.shape
    # Contours 对全分辨率极慢：限制到约 400×600
    sx = max(1, int(np.ceil(nx / 600.0)))
    sz = max(1, int(np.ceil(nz / 400.0)))
    if sx > 1 or sz > 1:
        vel = vel[::sz, ::sx]
        dx = float(dx) * sx
        dz = float(dz) * sz
    lines: List[Tuple[np.ndarray, np.ndarray, float]] = []

    try:
        from skimage.measure import find_contours

        for lev in levels:
            lev_f = float(lev)
            for c in find_contours(vel, lev_f):
                if c.shape[0] < 2:
                    continue
                xx = ox + c[:, 1] * dx
                zz = oz + c[:, 0] * dz
                lines.append((xx, zz, lev_f))
        return lines
    except Exception:
        pass

    # matplotlib 离屏求等值线（不弹窗）
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nz, nx = vel.shape
        x = ox + np.arange(nx, dtype=float) * dx
        z = oz + np.arange(nz, dtype=float) * dz
        fig, ax = plt.subplots()
        level_list = [float(lv) for lv in levels]
        cs = ax.contour(x, z, vel, levels=level_list)
        if hasattr(cs, "allsegs"):
            for lev, segs in zip(level_list, cs.allsegs):
                for seg in segs:
                    if seg is None or len(seg) < 2:
                        continue
                    lines.append(
                        (
                            np.asarray(seg[:, 0]),
                            np.asarray(seg[:, 1]),
                            float(lev),
                        )
                    )
        elif hasattr(cs, "collections"):
            for i, lev in enumerate(level_list):
                if i >= len(cs.collections):
                    break
                for path in cs.collections[i].get_paths():
                    v = path.vertices
                    if v is None or len(v) < 2:
                        continue
                    lines.append(
                        (
                            np.asarray(v[:, 0], dtype=float),
                            np.asarray(v[:, 1], dtype=float),
                            float(lev),
                        )
                    )
        plt.close(fig)
    except Exception:
        pass
    return lines


def _label_anchor_on_polyline(
    xx: np.ndarray, zz: np.ndarray
) -> Optional[Tuple[float, float]]:
    """取折线中段一点作为标签锚点。"""
    xx = np.asarray(xx, dtype=float)
    zz = np.asarray(zz, dtype=float)
    n = int(xx.size)
    if n < 2:
        return None
    # 用累积弧长找中点，避免端点贴边
    d = np.hypot(np.diff(xx), np.diff(zz))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-12:
        return float(xx[n // 2]), float(zz[n // 2])
    target = 0.5 * float(s[-1])
    j = int(np.searchsorted(s, target))
    j = min(max(j, 0), n - 1)
    return float(xx[j]), float(zz[j])


class VelCanvas(QWidget):
    """速度/成像/波场叠层画布。``iface_changed``：B/S/M 选择变更。"""

    iface_changed = Signal(object)  # dict basement/seafloor/moho

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._vel: Optional[np.ndarray] = None  # 当前 ImageItem 数据（速度或成像）
        self._ox = self._dx = self._oz = self._dz = 0.0
        # Contours 始终用速度体（勿对成像/波场振幅求 5/6/7/8 km/s 线）
        self._vel_contour: Optional[np.ndarray] = None
        self._vc_ox = self._vc_dx = self._vc_oz = self._vc_dz = 0.0
        self._zelt_model = None
        self._bath_1d: Optional[np.ndarray] = None
        self._shots_xz: Optional[List[Tuple[float, float]]] = None
        self._obs_xz: Optional[List[Tuple[float, float]]] = None
        self._hl_shot_idx: Optional[List[int]] = None
        self._contour_items: List = []
        self._contour_labels: List = []
        self._iface_items: List = []
        self._basement_idx: Optional[int] = None
        self._seafloor_idx: Optional[int] = None  # None = Auto（不单独高亮）
        self._moho_idx: Optional[int] = None
        self._cbar = None
        self._cbar_label = "Velocity (km/s)"
        self._vmin = 1.5
        self._vmax = 8.0
        self._wfl_img = None  # 正传/单波场 RGBA 叠层
        self._wfl_img_r = None  # 反传叠层（正反同步时）
        self._wfl_mode = False
        self._amp_overlay_state: Optional[dict] = None  # 供「叠加速度模型」切换重绘

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if pg is None:
            self.plot = None
            self._img = None
            self.chk_contours = None
            self.chk_interfaces = None
            self.chk_vel_underlay = None
            self.cmb_basement = None
            self.cmb_seafloor = None
            self.cmb_moho = None
            lay.addWidget(QLabel("需要 pyqtgraph: pip install pyqtgraph\n%s" % _PG_ERR))
            return

        opt = QHBoxLayout()
        self.chk_contours = QCheckBox("Contours")
        self.chk_contours.setChecked(False)
        self.chk_contours.setToolTip(
            "叠加速度等值线并标注 5.0/6.0/7.0/8.0（km/s）；"
            "始终基于速度模型，不对成像振幅/波场求等值线；大模型可能稍慢"
        )
        self.chk_interfaces = QCheckBox("Interfaces")
        self.chk_interfaces.setChecked(True)
        self.chk_interfaces.setToolTip(
            "Zelt v.in 层界面（叠在速度几何上，与成像振幅无关）；"
            "仅当速度源为 v.in 时可勾选；.rsf/.grd 无界面节点。"
            "右侧 B/S/M 着色高亮（对齐 imodel）"
        )
        self.chk_vel_underlay = QCheckBox("叠加速度模型")
        self.chk_vel_underlay.setChecked(True)
        self.chk_vel_underlay.setToolTip(
            "默认开：viridis 速度底图 + 波场/成像叠层。\n"
            "取消：不画速度色填，只保留波场/成像、地形 bath（必有）与 Interfaces（若勾选）"
        )
        self.chk_contours.toggled.connect(self._refresh_overlays)
        self.chk_interfaces.toggled.connect(self._refresh_overlays)
        self.chk_vel_underlay.toggled.connect(self._on_vel_underlay_toggled)
        opt.addWidget(self.chk_contours)
        opt.addWidget(self.chk_interfaces)
        opt.addWidget(self.chk_vel_underlay)

        self.cmb_basement = QComboBox()
        self.cmb_seafloor = QComboBox()
        self.cmb_moho = QComboBox()
        self.cmb_basement.setToolTip("B — Basement 沉积基底界面（红）")
        self.cmb_seafloor.setToolTip(
            "S — Seafloor 地形海底（蓝）。\n"
            "选定层 → 生成 vel/bath1d、填海水、预览压水柱均用该界面；\n"
            "载入 v.in 未选时默认 Interface 2（第 2 个界面）。"
        )
        self.cmb_moho.setToolTip("M — Moho 界面（绿）")
        for cmb in (self.cmb_basement, self.cmb_seafloor, self.cmb_moho):
            cmb.setEnabled(False)
            cmb.setMaximumWidth(110)
        for label, cmb in (
            ("B:", self.cmb_basement),
            ("S:", self.cmb_seafloor),
            ("M:", self.cmb_moho),
        ):
            opt.addWidget(QLabel(label))
            opt.addWidget(cmb)
        connect_combo_deferred(self.cmb_basement, self._on_basement_changed)
        connect_combo_deferred(self.cmb_seafloor, self._on_seafloor_changed)
        connect_combo_deferred(self.cmb_moho, self._on_moho_changed)
        opt.addStretch(1)
        lay.addLayout(opt)

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "Distance", units="km")
        self.plot.setLabel("left", "Depth", units="km")
        self.plot.invertY(True)
        self._img = pg.ImageItem()
        self.plot.addItem(self._img)
        self._wfl_img = pg.ImageItem()
        self._wfl_img.setZValue(10)
        self._wfl_img_r = pg.ImageItem()
        self._wfl_img_r.setZValue(11)
        # 叠层仅吃 RGBA，禁止继承速度/振幅 LUT
        for _witem in (self._wfl_img, self._wfl_img_r):
            try:
                _witem.setLookupTable(None)
            except Exception:
                pass
            try:
                _witem.setLevels(None)
            except Exception:
                pass
            self.plot.addItem(_witem)
        self._cmap = None
        try:
            self._cmap = pg.colormap.get("viridis")
            self._img.setLookupTable(self._cmap.getLookupTable(nPts=256))
        except Exception:
            self._cmap = None
        # 色标：插在主图右侧
        self._cbar = None
        try:
            ColorBarItem = getattr(pg, "ColorBarItem", None)
            if ColorBarItem is not None:
                self._cbar = ColorBarItem(
                    values=(1.5, 8.0),
                    width=20,
                    colorMap=self._cmap if self._cmap is not None else "viridis",
                    label="Velocity (km/s)",
                    interactive=False,
                )
                self._cbar.setImageItem(
                    self._img, insert_in=self.plot.getPlotItem()
                )
        except Exception:
            self._cbar = None
        self._bath_curve = self.plot.plot(pen=pg.mkPen("#00bcd4", width=2))
        try:
            self._bath_curve.setZValue(19)
        except Exception:
            pass
        self._shot_scatter = pg.ScatterPlotItem(
            size=6, brush=pg.mkBrush("#ef4444"), pen=None, symbol="t"
        )
        # RTM 当前成像对应炮：更大、描边，叠在普通炮点之上
        self._shot_hl_scatter = pg.ScatterPlotItem(
            size=14,
            brush=pg.mkBrush("#fbbf24"),
            pen=pg.mkPen("#b45309", width=2),
            symbol="star",
        )
        # OBS：大号菱形 + 亮绿填充 + 深描边，避免在 viridis 底图上被淹没
        self._obs_scatter = pg.ScatterPlotItem(
            size=18,
            brush=pg.mkBrush("#39ff14"),
            pen=pg.mkPen("#14532d", width=2.5),
            symbol="d",
        )
        # 须高于不透明波场叠层(z=10/11)，否则动画预览时 OBS/炮点被盖住
        self._shot_scatter.setZValue(20)
        self._shot_hl_scatter.setZValue(21)
        self._obs_scatter.setZValue(22)
        self.plot.addItem(self._shot_scatter)
        self.plot.addItem(self._shot_hl_scatter)
        self.plot.addItem(self._obs_scatter)
        lay.addWidget(self.plot)

    def clear(self, msg: str = "") -> None:
        if self.plot is None:
            return
        self._vel = None
        self._vel_contour = None
        self._zelt_model = None
        self._basement_idx = self._seafloor_idx = self._moho_idx = None
        self._hl_shot_idx = None
        self._clear_dyn_items(self._contour_items)
        self._clear_dyn_items(self._contour_labels)
        self._clear_dyn_items(self._iface_items)
        self._clear_wfl_overlay()
        self._restore_vel_colormap()
        self._img.clear()
        self._bath_curve.setData([], [])
        self._shot_scatter.setData([], [])
        self._shot_hl_scatter.setData([], [])
        self._obs_scatter.setData([], [])
        self.plot.setTitle(msg or "")
        if self.chk_interfaces is not None:
            self.chk_interfaces.setEnabled(False)
        self._refresh_iface_combos(None)

    def _downsample_vel_rect(
        self, vel: np.ndarray, ox: float, dx: float, oz: float, dz: float
    ):
        """与 show_vel 相同的显示抽稀，供 Contours 底速度。"""
        from pyAOBS.visualization.obs_rtm_qt.services.model_import import (
            downsample_vel_meta,
        )

        vel = np.asarray(vel, dtype=np.float32)
        meta0 = {
            "o1": float(oz),
            "d1": float(dz),
            "n1": float(vel.shape[0]),
            "o2": float(ox),
            "d2": float(dx),
            "n2": float(vel.shape[1]),
        }
        vel_disp, meta_d = downsample_vel_meta(vel, meta0)
        return (
            vel_disp,
            float(meta_d["o2"]),
            float(meta_d["d2"]),
            float(meta_d["o1"]),
            float(meta_d["d1"]),
        )

    def _store_contour_vel(
        self,
        vel: np.ndarray,
        *,
        ox: float,
        dx: float,
        oz: float,
        dz: float,
        already_downsampled: bool = False,
    ) -> None:
        if already_downsampled:
            self._vel_contour = np.asarray(vel, dtype=np.float32)
            self._vc_ox, self._vc_dx = float(ox), float(dx)
            self._vc_oz, self._vc_dz = float(oz), float(dz)
            return
        vd, ox2, dx2, oz2, dz2 = self._downsample_vel_rect(vel, ox, dx, oz, dz)
        self._vel_contour = vd
        self._vc_ox, self._vc_dx, self._vc_oz, self._vc_dz = ox2, dx2, oz2, dz2

    @staticmethod
    def _is_velocity_cbar(cbar_label: str) -> bool:
        s = str(cbar_label or "").lower()
        return "veloc" in s or "km/s" in s

    def show_dataset(
        self,
        ds,
        *,
        title: str = "Velocity Model",
        zelt_model=None,
        cbar_label: str = "Velocity (km/s)",
        bath_1d: Optional[np.ndarray] = None,
        bath_ox: float = 0.0,
        bath_dx: float = 0.0,
        shots_xz: Optional[List[Tuple[float, float]]] = None,
        obs_xz: Optional[List[Tuple[float, float]]] = None,
        highlight_shot_idx: Optional[List[int]] = None,
        keep_clim: bool = False,
    ) -> None:
        del keep_clim, bath_ox, bath_dx
        vel, ox, dx, oz, dz = _dataset_to_vel_rect(ds)
        self.show_vel(
            vel,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=title,
            bath_1d=bath_1d,
            shots_xz=shots_xz,
            obs_xz=obs_xz,
            highlight_shot_idx=highlight_shot_idx,
            cbar_label=cbar_label,
            zelt_model=zelt_model,
        )

    def show_vel(
        self,
        vel: np.ndarray,
        *,
        ox: float,
        dx: float,
        oz: float,
        dz: float,
        title: str = "",
        bath_1d: Optional[np.ndarray] = None,
        shots_xz: Optional[List[Tuple[float, float]]] = None,
        obs_xz: Optional[List[Tuple[float, float]]] = None,
        highlight_shot_idx: Optional[List[int]] = None,
        cbar_label: str = "Velocity (km/s)",
        zelt_model=None,
        vel_for_contours=None,
        vel_for_contours_meta: Optional[dict] = None,
    ) -> None:
        """
        显示速度或成像振幅。

        Contours 始终基于速度：显示速度时用本图；显示成像时请传
        ``vel_for_contours``（完整 vel 数组），否则保留上次速度 Contours。
        """
        if self.plot is None:
            return
        # 防护：后台线程绝不能改 pyqtgraph（否则 QTextDocument 段错误）
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            raise RuntimeError("show_vel 必须在 GUI 线程调用")
        self._clear_wfl_overlay()
        self._cbar_label = str(cbar_label or "Velocity (km/s)")
        if self._is_velocity_cbar(self._cbar_label):
            self._restore_vel_colormap()
        else:
            self._apply_amplitude_colormap()
        vel = np.asarray(vel, dtype=np.float32)
        n2_full = int(vel.shape[1])
        vel_disp, ox, dx, oz, dz = self._downsample_vel_rect(vel, ox, dx, oz, dz)
        # bath 与显示 nx 对齐
        if bath_1d is not None:
            bath_1d = np.asarray(bath_1d, float)
            if bath_1d.size == n2_full and vel_disp.shape[1] != bath_1d.size:
                step = max(1, int(round(n2_full / max(vel_disp.shape[1], 1))))
                bath_1d = bath_1d[::step][: vel_disp.shape[1]]

        self._vel = vel_disp
        self._ox, self._dx, self._oz, self._dz = ox, dx, oz, dz
        # Contours：速度显示用本图；成像显示用显式速度底图
        if vel_for_contours is not None:
            vm = vel_for_contours_meta or {}
            self._store_contour_vel(
                np.asarray(vel_for_contours, dtype=np.float32),
                ox=float(vm.get("o2", ox)),
                dx=float(vm.get("d2", dx)),
                oz=float(vm.get("o1", oz)),
                dz=float(vm.get("d1", dz)),
            )
        elif self._is_velocity_cbar(self._cbar_label):
            self._store_contour_vel(
                vel_disp,
                ox=ox,
                dx=dx,
                oz=oz,
                dz=dz,
                already_downsampled=True,
            )
        self._zelt_model = zelt_model
        self._bath_1d = None if bath_1d is None else np.asarray(bath_1d, float)
        self._shots_xz = shots_xz
        self._obs_xz = obs_xz
        self._hl_shot_idx = list(highlight_shot_idx) if highlight_shot_idx else None

        nz, nx = vel_disp.shape
        vmin, vmax = _data_vrange(vel_disp)
        self._vmin, self._vmax = float(vmin), float(vmax)
        rect = QRectF(
            float(ox),
            float(oz),
            float(nx * dx),
            float(nz * dz),
        )
        if self._is_velocity_cbar(self._cbar_label):
            from pyAOBS.visualization.pg_velocity import set_velocity_image

            set_velocity_image(
                self._img,
                vel_disp,
                levels=(self._vmin, self._vmax),
                rect=rect,
                cmap=self._cmap,
                cbar=self._cbar,
            )
        else:
            self._img.setImage(vel_disp, autoLevels=False)
            self._img.setLevels((self._vmin, self._vmax))
            self._img.setRect(rect)
        self._update_colorbar()
        self.plot.setTitle(title or "Velocity (km/s)")
        self.plot.setXRange(ox, ox + nx * dx, padding=0.02)
        self.plot.setYRange(oz, oz + nz * dz, padding=0.02)

        if self._bath_1d is not None and len(self._bath_1d) == nx:
            x = ox + np.arange(nx) * dx
            self._bath_curve.setData(x, self._bath_1d)
        else:
            self._bath_curve.setData([], [])

        if shots_xz:
            self._shot_scatter.setData(
                x=[p[0] for p in shots_xz], y=[p[1] for p in shots_xz]
            )
        else:
            self._shot_scatter.setData([], [])
            self._shot_hl_scatter.setData([], [])
        if obs_xz:
            self._obs_scatter.setData(
                x=[p[0] for p in obs_xz], y=[p[1] for p in obs_xz]
            )
        else:
            self._obs_scatter.setData([], [])
        self._apply_shot_highlight()

        if self.chk_interfaces is not None:
            self.chk_interfaces.setEnabled(zelt_model is not None)
            if zelt_model is None:
                self.chk_interfaces.setChecked(False)
            else:
                self.chk_interfaces.blockSignals(True)
                self.chk_interfaces.setChecked(True)
                self.chk_interfaces.blockSignals(False)
        # 保留已选 B/S/M（重载 v.in / 切到偏移页时勿清空）
        self._refresh_iface_combos(
            zelt_model, prefer=self.interface_selection()
        )

        self._apply_vel_underlay_chrome()
        self._refresh_overlays()

    def _apply_shot_highlight(self) -> None:
        """按 _hl_shot_idx 刷新黄星（不重绘速度图）。"""
        if self._shot_hl_scatter is None:
            return
        shots = self._shots_xz
        if not shots or not self._hl_shot_idx:
            self._shot_hl_scatter.setData([], [])
            return
        hl_x, hl_y = [], []
        nsh = len(shots)
        for i in self._hl_shot_idx:
            if 0 <= int(i) < nsh:
                hl_x.append(float(shots[int(i)][0]))
                hl_y.append(float(shots[int(i)][1]))
        self._shot_hl_scatter.setData(x=hl_x, y=hl_y)

    def set_highlight_shot_idx(
        self, highlight_shot_idx: Optional[List[int]] = None
    ) -> None:
        """道集范围变更后及时更新黄星，无需重载速度/成像。"""
        self._hl_shot_idx = (
            list(highlight_shot_idx) if highlight_shot_idx else None
        )
        self._apply_shot_highlight()

    def set_shot_obs(
        self,
        shots_xz: Optional[List[Tuple[float, float]]] = None,
        obs_xz: Optional[List[Tuple[float, float]]] = None,
        *,
        highlight_shot_idx: Optional[List[int]] = None,
    ) -> None:
        """仅更新炮点/OBS 散点（几何变更后热刷新，不重绘速度体）。"""
        if self.plot is None:
            return
        self._shots_xz = shots_xz
        self._obs_xz = obs_xz
        if highlight_shot_idx is not None:
            self._hl_shot_idx = (
                list(highlight_shot_idx) if highlight_shot_idx else None
            )
        if shots_xz:
            self._shot_scatter.setData(
                x=[p[0] for p in shots_xz], y=[p[1] for p in shots_xz]
            )
        else:
            self._shot_scatter.setData([], [])
            self._shot_hl_scatter.setData([], [])
        if obs_xz:
            self._obs_scatter.setData(
                x=[p[0] for p in obs_xz], y=[p[1] for p in obs_xz]
            )
        else:
            self._obs_scatter.setData([], [])
        self._apply_shot_highlight()

    @staticmethod
    def _resample_2d(arr: np.ndarray, nz: int, nx: int) -> np.ndarray:
        """最近邻抽稀/拉伸到 (nz, nx)。"""
        a = np.asarray(arr, dtype=np.float32)
        if a.shape == (nz, nx):
            return a
        zi = np.linspace(0, a.shape[0] - 1, nz)
        xi = np.linspace(0, a.shape[1] - 1, nx)
        zi = np.clip(np.rint(zi).astype(int), 0, a.shape[0] - 1)
        xi = np.clip(np.rint(xi).astype(int), 0, a.shape[1] - 1)
        return np.ascontiguousarray(a[zi][:, xi])

    @staticmethod
    def _amplitude_gray_lut_u8() -> np.ndarray:
        """与原先 Amplitude 预览相同：CET-L1（否则线性灰）→ (256,3) uint8。"""
        if pg is not None:
            for name in ("CET-L1", "CET-L2", "gray"):
                try:
                    cm = pg.colormap.get(name)
                    if cm is None:
                        continue
                    try:
                        lut = cm.getLookupTable(nPts=256, mode="byte")
                    except TypeError:
                        lut = cm.getLookupTable(nPts=256)
                    return VelCanvas._lut_as_u8_rgb(lut)
                except Exception:
                    continue
        g = np.linspace(0, 255, 256, dtype=np.uint8)
        return np.column_stack([g, g, g])

    @staticmethod
    def wfl_to_rgba(
        wfl: np.ndarray,
        *,
        pclip: float = 98.0,
        alpha_max: float = 0.72,
        floor: float = 0.012,
        tone: str = "gray",
        gamma: float = 0.55,
    ) -> np.ndarray:
        """
        波场 → 半透明 RGBA uint8。

        ``tone``: gray（单场，CET-L1 有符号灰阶，同旧 Amplitude 预览）/
        warm（正传）/ cool（反传）。
        必须用 uint8：float 进 ImageItem 会触发 levels required。

        OBS 源在海底时源旁能量极大；用较低 floor + gamma 压缩，避免水柱弱场
        被 pclip 阈值后整片透明（误以为「从海底才开始」、与压水柱混淆）。
        """
        w = np.asarray(wfl, dtype=np.float32)
        nz, nx = w.shape
        flat = w.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return np.zeros((nz, nx, 4), dtype=np.uint8)
        if flat.size > 300_000:
            flat = flat[:: max(1, flat.size // 300_000)]
        clip = float(np.percentile(np.abs(flat), float(pclip)))
        if not np.isfinite(clip) or clip < 1e-30:
            clip = float(np.max(np.abs(w))) + 1e-30
        amp = np.clip(np.abs(w) / clip, 0.0, 1.0)
        # 压低强源、抬高弱场（水柱上行波）
        gpow = float(gamma) if float(gamma) > 1e-6 else 1.0
        amp_v = np.power(amp, gpow)
        a = np.where(amp < float(floor), 0.0, amp_v * float(alpha_max)) * 255.0
        rgba = np.empty((nz, nx, 4), dtype=np.uint8)
        t = str(tone or "gray").lower()
        if t == "warm":
            # 正传：琥珀/金（与黄星区分仍可读）
            rgba[..., 0] = np.clip(180 + 75 * amp_v, 0, 255).astype(np.uint8)
            rgba[..., 1] = np.clip(120 + 100 * amp_v, 0, 255).astype(np.uint8)
            rgba[..., 2] = np.clip(40 + 40 * amp_v, 0, 255).astype(np.uint8)
        elif t == "cool":
            # 反传：青蓝
            rgba[..., 0] = np.clip(40 + 40 * amp_v, 0, 255).astype(np.uint8)
            rgba[..., 1] = np.clip(140 + 90 * amp_v, 0, 255).astype(np.uint8)
            rgba[..., 2] = np.clip(200 + 55 * amp_v, 0, 255).astype(np.uint8)
        else:
            # 原始 Amplitude：有符号对称灰阶（-clip→黑，0→中灰，+clip→白）+ CET-L1
            lut = VelCanvas._amplitude_gray_lut_u8()
            signed = np.clip(w / clip, -1.0, 1.0)
            idx = ((signed + 1.0) * 0.5 * 255.0).astype(np.uint8)
            rgb = lut[idx]
            rgba[..., 0] = rgb[..., 0]
            rgba[..., 1] = rgb[..., 1]
            rgba[..., 2] = rgb[..., 2]
        rgba[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
        return rgba

    def _clear_wfl_overlay(self) -> None:
        self._wfl_mode = False
        self._amp_overlay_state = None
        for item in (self._wfl_img, self._wfl_img_r):
            if item is None:
                continue
            try:
                item.clear()
            except Exception:
                pass

    def vel_underlay_enabled(self) -> bool:
        """是否叠加速度色填底图（默认 True）。"""
        chk = getattr(self, "chk_vel_underlay", None)
        if chk is None:
            return True
        return bool(chk.isChecked())

    def _neutral_base_rgb(self) -> Optional[np.ndarray]:
        """无速度底图时的浅底，便于只看波场/成像。"""
        if self._vel is None:
            return None
        nz, nx = self._vel.shape[:2]
        return np.full((nz, nx, 3), 250, dtype=np.uint8)

    def _base_rgb_for_overlay(self) -> Optional[np.ndarray]:
        if self.vel_underlay_enabled():
            return self._velocity_rgb_u8()
        return self._neutral_base_rgb()

    def _apply_vel_underlay_chrome(self) -> None:
        """按勾选显示/隐藏速度 ImageItem 与色标；地形与界面不受影响。"""
        show = self.vel_underlay_enabled()
        if self._img is not None:
            try:
                self._img.setVisible(show)
            except Exception:
                try:
                    self._img.setOpacity(1.0 if show else 0.0)
                except Exception:
                    pass
        if self._cbar is not None:
            try:
                self._cbar.setVisible(show)
            except Exception:
                pass

    def _on_vel_underlay_toggled(self, *_args) -> None:
        """勾选变化：有叠层则按新底色重合成；否则只改速度底图可见性。"""
        self._apply_vel_underlay_chrome()
        st = self._amp_overlay_state
        if self._wfl_mode and isinstance(st, dict):
            try:
                if st.get("dual"):
                    self.set_wfl_dual_overlay(
                        st["wfl_s"],
                        st["wfl_r"],
                        pclip=float(st.get("pclip", 98.0)),
                        title=st.get("title"),
                    )
                else:
                    self.set_wfl_overlay(
                        st["wfl"],
                        pclip=float(st.get("pclip", 98.0)),
                        title=st.get("title"),
                        tone=str(st.get("tone") or "gray"),
                        alpha_max=float(st.get("alpha_max", 0.55)),
                    )
            except Exception:
                pass
        self._refresh_overlays()

    def _ensure_viridis_cmap(self):
        if self._cmap is None and pg is not None:
            try:
                self._cmap = pg.colormap.get("viridis")
            except Exception:
                self._cmap = None
        return self._cmap

    def _restore_vel_colormap(self) -> None:
        """速度底图强制回 viridis（成像灰度 / 波场叠层后易丢 LUT 或色标错绑）。"""
        if self._img is None:
            return
        cmap = self._ensure_viridis_cmap()
        try:
            if cmap is not None:
                if hasattr(self._img, "setColorMap"):
                    try:
                        self._img.setColorMap(cmap)
                    except Exception:
                        pass
                try:
                    lut = cmap.getLookupTable(nPts=256, mode="byte")
                except TypeError:
                    lut = self._lut_as_u8_rgb(cmap.getLookupTable(nPts=256))
                else:
                    lut = self._lut_as_u8_rgb(lut)
                self._img.setLookupTable(lut)
            else:
                g = np.linspace(0, 255, 256, dtype=np.ubyte)
                self._img.setLookupTable(np.column_stack([g, g, g]))
        except Exception:
            pass
        # ColorBar 与 ImageItem 重新对齐（Amplitude 后常见色标与底图不一致）
        if self._cbar is not None:
            try:
                if cmap is not None and hasattr(self._cbar, "setColorMap"):
                    self._cbar.setColorMap(cmap)
            except Exception:
                pass
            try:
                self._cbar.setImageItem(
                    self._img, insert_in=self.plot.getPlotItem()
                )
            except Exception:
                try:
                    self._cbar.setImageItem(self._img)
                except Exception:
                    pass

    def pin_velocity_base_display(self) -> None:
        """
        波场叠层后钉死速度底图：viridis LUT + 色标绑回 _img + 速度 levels。
        预览 / 动画逐帧共用（避免 Amplitude 灰度色标残留或 ColorBar 误绑 RGBA）。
        「叠加速度模型」关闭时隐藏色填与色标，仍保留 bath / Interfaces。
        """
        if self._img is None:
            return
        if not self._is_velocity_cbar(getattr(self, "_cbar_label", "")):
            self._apply_vel_underlay_chrome()
            return
        # 重新写回标量速度图，防止叠层流程把底图弄成灰度幅值
        if self._vel is not None:
            try:
                nz, nx = self._vel.shape
                self._img.setImage(
                    np.asarray(self._vel, dtype=np.float32), autoLevels=False
                )
                self._img.setRect(
                    QRectF(
                        float(self._ox),
                        float(self._oz),
                        float(nx * self._dx),
                        float(nz * self._dz),
                    )
                )
            except Exception:
                pass
        self._restore_vel_colormap()
        try:
            self._img.setLevels((float(self._vmin), float(self._vmax)))
        except Exception:
            pass
        if self._cbar is not None:
            try:
                self._cbar.setLevels((float(self._vmin), float(self._vmax)))
            except Exception:
                pass
            try:
                if hasattr(self._cbar, "setLabel"):
                    self._cbar.setLabel(self._cbar_label)
                elif hasattr(self._cbar, "axis") and hasattr(
                    self._cbar.axis, "setLabel"
                ):
                    self._cbar.axis.setLabel(self._cbar_label)
            except Exception:
                pass
        self._apply_vel_underlay_chrome()

    def _apply_amplitude_colormap(self) -> None:
        """成像振幅：灰度 LUT，与速度 viridis 区分。"""
        if self._img is None:
            return
        try:
            g = np.linspace(0, 255, 256, dtype=np.ubyte)
            lut = np.column_stack([g, g, g])
            self._img.setLookupTable(lut)
            if self._cbar is not None and hasattr(self._cbar, "setColorMap"):
                try:
                    cm = pg.colormap.get("CET-L1") if pg is not None else None
                    if cm is not None:
                        self._cbar.setColorMap(cm)
                except Exception:
                    pass
        except Exception:
            pass

    @staticmethod
    def _lut_as_u8_rgb(lut) -> np.ndarray:
        """
        ColorMap LUT → (256, 3) uint8。

        pyqtgraph 默认 ``getLookupTable`` 常为 float 0–1；直接 ``astype(uint8)``
        会截成全 0，波场合成底图变成黑/灰。
        """
        a = np.asarray(lut)
        if a.ndim == 1:
            a = np.column_stack([a, a, a])
        rgb = np.asarray(a[..., :3], dtype=np.float64)
        if rgb.size == 0:
            g = np.linspace(0, 255, 256, dtype=np.uint8)
            return np.column_stack([g, g, g])
        mx = float(np.nanmax(rgb))
        if mx <= 1.0 + 1e-3:
            rgb = rgb * 255.0
        return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

    def _velocity_rgb_u8(self) -> Optional[np.ndarray]:
        """当前速度底图 → viridis RGB uint8，供与波场 alpha 合成。"""
        if self._vel is None:
            return None
        vel = np.asarray(self._vel, dtype=np.float32)
        vn, vx = float(self._vmin), float(self._vmax)
        t = (vel - vn) / max(vx - vn, 1e-12)
        idx = (np.clip(t, 0.0, 1.0) * 255.0).astype(np.uint8)
        cmap = self._ensure_viridis_cmap()
        if cmap is not None:
            try:
                try:
                    lut = cmap.getLookupTable(nPts=256, mode="byte")
                except TypeError:
                    lut = cmap.getLookupTable(nPts=256)
                lut_u8 = self._lut_as_u8_rgb(lut)
                return np.ascontiguousarray(lut_u8[idx, :3])
            except Exception:
                pass
        return np.stack([idx, idx, idx], axis=-1)

    def _blend_wfl_rgba_onto_rgb(
        self, base_rgb: np.ndarray, wfl: np.ndarray, *, pclip: float, tone: str, alpha_max: float
    ) -> np.ndarray:
        nz, nx = base_rgb.shape[:2]
        w = self._resample_2d(wfl, nz, nx)
        rgba = self.wfl_to_rgba(w, pclip=pclip, tone=tone, alpha_max=alpha_max)
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        out = base_rgb.astype(np.float32) * (1.0 - a) + rgba[..., :3].astype(np.float32) * a
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def _show_rgb_overlay(self, item, rgb: np.ndarray) -> None:
        """不透明 RGB 叠层（已含速度底色+波场）；ColorBar 仍绑底层速度标量图。"""
        if item is None:
            return
        nz, nx = rgb.shape[:2]
        disp = np.empty((nz, nx, 4), dtype=np.uint8)
        disp[..., :3] = rgb
        disp[..., 3] = 255
        try:
            if hasattr(item, "setLookupTable"):
                item.setLookupTable(None)
        except Exception:
            pass
        try:
            item.setLevels(None)
        except Exception:
            pass
        try:
            item.clear()
        except Exception:
            pass
        item.setImage(disp, autoLevels=False)
        try:
            item.setLevels(None)
        except Exception:
            pass
        item.setRect(
            QRectF(
                float(self._ox),
                float(self._oz),
                float(nx * self._dx),
                float(nz * self._dz),
            )
        )

    def _set_overlay_item(
        self,
        item,
        wfl: np.ndarray,
        *,
        pclip: float,
        tone: str,
        alpha_max: float,
        base_rgb: Optional[np.ndarray] = None,
    ) -> None:
        """
        将波场 alpha 合成到速度 RGB 后以不透明图显示。

        不用半透明 ImageItem：部分环境下 RGBA alpha 被忽略，会整屏盖成波场灰，
        速度色标底图消失（预览尤甚；动画逐帧 pin 看似正常）。
        """
        if item is None or self._vel is None:
            return
        if base_rgb is None:
            base_rgb = self._base_rgb_for_overlay()
        if base_rgb is None:
            return
        # 无速度底时略提高不透明度，避免浅底上波场过淡
        amax = float(alpha_max)
        if not self.vel_underlay_enabled():
            amax = max(amax, 0.85)
        rgb = self._blend_wfl_rgba_onto_rgb(
            base_rgb, wfl, pclip=pclip, tone=tone, alpha_max=amax
        )
        self._show_rgb_overlay(item, rgb)

    def show_amp_on_vel(
        self,
        vel: np.ndarray,
        amp: np.ndarray,
        *,
        ox: float,
        dx: float,
        oz: float,
        dz: float,
        title: str = "",
        shots_xz: Optional[List[Tuple[float, float]]] = None,
        obs_xz: Optional[List[Tuple[float, float]]] = None,
        highlight_shot_idx: Optional[List[int]] = None,
        pclip: float = 98.0,
        zelt_model=None,
        bath_1d: Optional[np.ndarray] = None,
        alpha_max: float = 0.55,
        tone: str = "gray",
    ) -> None:
        """
        速度色标底图 + 振幅/波场 alpha 合成叠层。

        成像预览与波场快照共用此路径（色标仍为 Velocity）。
        """
        if self.plot is None:
            return
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            raise RuntimeError("show_amp_on_vel 必须在 GUI 线程调用")

        self.show_vel(
            vel,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=title,
            bath_1d=bath_1d,
            shots_xz=shots_xz,
            obs_xz=obs_xz,
            highlight_shot_idx=highlight_shot_idx,
            cbar_label="Velocity (km/s)",
            zelt_model=zelt_model,
        )
        self.set_wfl_overlay(
            amp, pclip=pclip, title=title or None, tone=tone, alpha_max=alpha_max
        )

    def show_wfl_on_vel(
        self,
        vel: np.ndarray,
        wfl: np.ndarray,
        *,
        ox: float,
        dx: float,
        oz: float,
        dz: float,
        title: str = "",
        shots_xz: Optional[List[Tuple[float, float]]] = None,
        obs_xz: Optional[List[Tuple[float, float]]] = None,
        highlight_shot_idx: Optional[List[int]] = None,
        pclip: float = 98.0,
        zelt_model=None,
        bath_1d: Optional[np.ndarray] = None,
    ) -> None:
        """速度底图 + 波场叠层（委托 show_amp_on_vel）。"""
        self.show_amp_on_vel(
            vel,
            wfl,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=title,
            shots_xz=shots_xz,
            obs_xz=obs_xz,
            highlight_shot_idx=highlight_shot_idx,
            pclip=pclip,
            zelt_model=zelt_model,
            bath_1d=bath_1d,
            alpha_max=0.55,
            tone="gray",
        )

    def show_wfl_dual_on_vel(
        self,
        vel: np.ndarray,
        wfl_s: np.ndarray,
        wfl_r: np.ndarray,
        *,
        ox: float,
        dx: float,
        oz: float,
        dz: float,
        title: str = "",
        shots_xz: Optional[List[Tuple[float, float]]] = None,
        obs_xz: Optional[List[Tuple[float, float]]] = None,
        highlight_shot_idx: Optional[List[int]] = None,
        pclip: float = 98.0,
        zelt_model=None,
        bath_1d: Optional[np.ndarray] = None,
    ) -> None:
        """同一速度底图上叠正传(暖色)+反传(冷色)。"""
        if self.plot is None:
            return
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            raise RuntimeError("show_wfl_dual_on_vel 必须在 GUI 线程调用")

        self.show_vel(
            vel,
            ox=ox,
            dx=dx,
            oz=oz,
            dz=dz,
            title=title,
            bath_1d=bath_1d,
            shots_xz=shots_xz,
            obs_xz=obs_xz,
            highlight_shot_idx=highlight_shot_idx,
            cbar_label="Velocity (km/s)",
            zelt_model=zelt_model,
        )
        self.set_wfl_dual_overlay(wfl_s, wfl_r, pclip=pclip, title=title or None)

    def set_wfl_overlay(
        self,
        wfl: np.ndarray,
        *,
        pclip: float = 98.0,
        title: Optional[str] = None,
        tone: str = "gray",
        alpha_max: float = 0.55,
    ) -> None:
        """仅更新单波场/成像振幅叠层（动画逐帧）；清掉第二叠层。"""
        if self.plot is None or self._wfl_img is None or self._vel is None:
            return
        if self._wfl_img_r is not None:
            try:
                self._wfl_img_r.clear()
            except Exception:
                pass
        self._amp_overlay_state = {
            "dual": False,
            "wfl": np.ascontiguousarray(wfl, dtype=np.float32),
            "pclip": float(pclip),
            "tone": str(tone or "gray"),
            "alpha_max": float(alpha_max),
            "title": title,
        }
        self._set_overlay_item(
            self._wfl_img,
            wfl,
            pclip=pclip,
            tone=tone,
            alpha_max=float(alpha_max),
        )
        self._wfl_mode = True
        self.pin_velocity_base_display()
        if title is not None:
            self.plot.setTitle(title)

    def set_wfl_dual_overlay(
        self,
        wfl_s: np.ndarray,
        wfl_r: np.ndarray,
        *,
        pclip: float = 98.0,
        title: Optional[str] = None,
    ) -> None:
        """同帧更新正传+反传叠层（动画用）。"""
        if self.plot is None or self._vel is None or self._wfl_img is None:
            return
        base = self._base_rgb_for_overlay()
        if base is None:
            return
        a_s = 0.50 if self.vel_underlay_enabled() else 0.80
        a_r = 0.50 if self.vel_underlay_enabled() else 0.80
        # 同一张 RGB：底色 → 正传暖 → 反传冷
        rgb = self._blend_wfl_rgba_onto_rgb(
            base, wfl_s, pclip=pclip, tone="warm", alpha_max=a_s
        )
        rgb = self._blend_wfl_rgba_onto_rgb(
            rgb, wfl_r, pclip=pclip, tone="cool", alpha_max=a_r
        )
        self._amp_overlay_state = {
            "dual": True,
            "wfl_s": np.ascontiguousarray(wfl_s, dtype=np.float32),
            "wfl_r": np.ascontiguousarray(wfl_r, dtype=np.float32),
            "pclip": float(pclip),
            "title": title,
        }
        self._show_rgb_overlay(self._wfl_img, rgb)
        if self._wfl_img_r is not None:
            try:
                self._wfl_img_r.clear()
            except Exception:
                pass
        self._wfl_mode = True
        self.pin_velocity_base_display()
        if title is not None:
            self.plot.setTitle(title)

    def _refresh_iface_combos(self, zelt_model, *, prefer: Optional[dict] = None) -> None:
        """按 v.in depth_nodes 填充 B/S/M；``prefer`` 用于恢复已选索引。"""
        combos = (self.cmb_basement, self.cmb_seafloor, self.cmb_moho)
        if any(c is None for c in combos):
            return
        keep = prefer if prefer is not None else self.interface_selection()
        for c in combos:
            c.blockSignals(True)
        try:
            self.cmb_basement.clear()
            self.cmb_seafloor.clear()
            self.cmb_moho.clear()
            dn = getattr(zelt_model, "depth_nodes", None) if zelt_model else None
            if dn is not None and len(dn) > 0:
                self.cmb_basement.addItem("None")
                self.cmb_moho.addItem("None")
                self.cmb_seafloor.addItem("Auto")
                for i in range(len(dn) - 1):
                    label = "Interface %d" % (i + 1)
                    self.cmb_basement.addItem(label)
                    self.cmb_seafloor.addItem(label)
                    self.cmb_moho.addItem(label)
                for c in combos:
                    c.setEnabled(True)
                self._apply_iface_selection_to_combos(keep)
            else:
                self.cmb_basement.addItem("None")
                self.cmb_seafloor.addItem("Auto")
                self.cmb_moho.addItem("None")
                self._basement_idx = self._seafloor_idx = self._moho_idx = None
                self.cmb_basement.setCurrentIndex(0)
                self.cmb_seafloor.setCurrentIndex(0)
                self.cmb_moho.setCurrentIndex(0)
                for c in combos:
                    c.setEnabled(False)
        finally:
            for c in combos:
                c.blockSignals(False)

    @staticmethod
    def _combo_set_iface(cmb: QComboBox, idx: Optional[int], *, none_text: str) -> None:
        if idx is None:
            i = cmb.findText(none_text)
            cmb.setCurrentIndex(i if i >= 0 else 0)
            return
        text = "Interface %d" % (int(idx) + 1)
        i = cmb.findText(text)
        cmb.setCurrentIndex(i if i >= 0 else 0)

    def _apply_iface_selection_to_combos(self, sel: Optional[dict]) -> None:
        """写入下拉；若目标层尚未出现在列表中，仍保留索引供随后 show_vel 恢复。"""
        sel = sel or {}

        def _one(cmb, key: str, none_text: str) -> Optional[int]:
            want = sel.get(key)
            if cmb is None:
                return want if want is None else int(want)
            self._combo_set_iface(cmb, want, none_text=none_text)
            t = cmb.currentText()
            if want is not None and t == none_text:
                return int(want)
            if t == none_text:
                return None
            return _iface_idx_from_combo(t)

        self._basement_idx = _one(self.cmb_basement, "basement", "None")
        self._seafloor_idx = _one(self.cmb_seafloor, "seafloor", "Auto")
        self._moho_idx = _one(self.cmb_moho, "moho", "None")

    def apply_interface_selection(
        self, sel: Optional[dict], *, emit: bool = False
    ) -> None:
        """应用 B/S/M（含下拉与着色）；用于速度页→偏移页同步。"""
        if self.cmb_basement is None:
            return
        for c in (self.cmb_basement, self.cmb_seafloor, self.cmb_moho):
            if c is not None:
                c.blockSignals(True)
        try:
            self._apply_iface_selection_to_combos(sel)
        finally:
            for c in (self.cmb_basement, self.cmb_seafloor, self.cmb_moho):
                if c is not None:
                    c.blockSignals(False)
        self._refresh_overlays()
        if emit:
            self.iface_changed.emit(self.interface_selection())

    def _emit_iface_changed(self) -> None:
        self.iface_changed.emit(self.interface_selection())

    def _on_basement_changed(self, *_args) -> None:
        if self.cmb_basement is None:
            return
        t = self.cmb_basement.currentText()
        self._basement_idx = None if t == "None" else _iface_idx_from_combo(t)
        self._refresh_overlays()
        self._emit_iface_changed()

    def _on_seafloor_changed(self, *_args) -> None:
        if self.cmb_seafloor is None:
            return
        t = self.cmb_seafloor.currentText()
        self._seafloor_idx = None if t == "Auto" else _iface_idx_from_combo(t)
        self._refresh_overlays()
        self._emit_iface_changed()

    def _on_moho_changed(self, *_args) -> None:
        if self.cmb_moho is None:
            return
        t = self.cmb_moho.currentText()
        self._moho_idx = None if t == "None" else _iface_idx_from_combo(t)
        self._refresh_overlays()
        self._emit_iface_changed()

    def interface_selection(self) -> dict:
        """供后续 bath / 导出使用：B/S/M 0-based 索引（S=None 表示 Auto）。"""
        return {
            "basement": self._basement_idx,
            "seafloor": self._seafloor_idx,
            "moho": self._moho_idx,
        }

    def _clear_dyn_items(self, items: List) -> None:
        if self.plot is None:
            return
        for it in items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        items.clear()

    def _update_colorbar(self) -> None:
        if self._cbar is None:
            return
        try:
            self._cbar.setLevels((float(self._vmin), float(self._vmax)))
        except Exception:
            pass
        try:
            # 部分版本用 setLabel / axis label
            if hasattr(self._cbar, "setLabel"):
                self._cbar.setLabel(self._cbar_label)
            elif hasattr(self._cbar, "axis") and hasattr(self._cbar.axis, "setLabel"):
                self._cbar.axis.setLabel(self._cbar_label)
        except Exception:
            pass

    def _refresh_overlays(self) -> None:
        if self.plot is None or self._vel is None:
            return
        self._clear_dyn_items(self._contour_items)
        self._clear_dyn_items(self._contour_labels)
        self._clear_dyn_items(self._iface_items)

        if self.chk_contours is not None and self.chk_contours.isChecked():
            self._draw_contours()
        if (
            self.chk_interfaces is not None
            and self.chk_interfaces.isChecked()
            and self._zelt_model is not None
        ):
            self._draw_interfaces()

    def _draw_contours(self) -> None:
        assert self.plot is not None
        # 优先速度底图；无底图且当前就是速度显示时才退回 _vel
        if self._vel_contour is not None:
            vel = self._vel_contour
            ox, dx, oz, dz = self._vc_ox, self._vc_dx, self._vc_oz, self._vc_dz
        elif self._is_velocity_cbar(self._cbar_label) and self._vel is not None:
            vel = self._vel
            ox, dx, oz, dz = self._ox, self._dx, self._oz, self._dz
        else:
            return
        levels = _nice_contour_levels(vel)
        if not levels:
            return
        pen = pg.mkPen((255, 255, 255, 200), width=1.2)
        # 每个速度值只在最长折线上标一次；5/6/7/8 必标，其它整档也标
        label_ok = {5.0, 6.0, 7.0, 8.0} | {
            float(lv) for lv in levels if abs(float(lv) - round(float(lv))) < 1e-6
        }
        best_by_lev: dict = {}
        for xx, zz, lev in _contour_polylines(vel, ox, dx, oz, dz, levels):
            curve = self.plot.plot(xx, zz, pen=pen)
            try:
                curve.setZValue(18)  # 高于波场叠层(10)，低于炮点/OBS(20+)
            except Exception:
                pass
            self._contour_items.append(curve)
            lev_f = float(lev)
            if lev_f not in label_ok:
                continue
            length = float(np.sum(np.hypot(np.diff(xx), np.diff(zz))))
            prev = best_by_lev.get(lev_f)
            if prev is None or length > prev[0]:
                best_by_lev[lev_f] = (length, xx, zz, lev_f)

        prefer = (5.0, 6.0, 7.0, 8.0)
        ordered = [lv for lv in prefer if lv in best_by_lev]
        ordered += sorted(lv for lv in best_by_lev if lv not in prefer)
        for lev in ordered:
            _length, xx, zz, lev_f = best_by_lev[lev]
            anchor = _label_anchor_on_polyline(xx, zz)
            if anchor is None:
                continue
            txt = "%.1f" % lev_f
            try:
                item = pg.TextItem(
                    text=txt,
                    color=(20, 20, 20),
                    fill=pg.mkBrush(255, 255, 255, 210),
                    anchor=(0.5, 0.5),
                )
            except TypeError:
                item = pg.TextItem(text=txt, color=(255, 255, 255), anchor=(0.5, 0.5))
            item.setPos(anchor[0], anchor[1])
            try:
                from PySide6.QtGui import QFont

                f = QFont()
                f.setPointSize(9)
                f.setBold(True)
                item.setFont(f)
            except Exception:
                pass
            self.plot.addItem(item)
            self._contour_labels.append(item)

    def _draw_interfaces(self) -> None:
        """层界面着色：底界虚线；M 绿 / B 红 / S 蓝（对齐 imodel）。"""
        assert self.plot is not None
        zm = self._zelt_model
        if zm is None:
            return
        try:
            n = len(zm.depth_nodes)
        except Exception:
            return
        for i in range(n):
            try:
                x_coords, z_coords = zm.get_layer_geometry(i)
                # 先 asarray：部分 numpy 对 list 直接 isnan 会异常/静默失败
                xv0 = np.asarray(x_coords, dtype=float)
                zv0 = np.asarray(z_coords, dtype=float)
                vm = ~(np.isnan(xv0) | np.isnan(zv0))
                if not np.any(vm):
                    continue
                xv = xv0[vm]
                zv = zv0[vm]
                is_b = self._basement_idx is not None and i == int(self._basement_idx)
                is_s = self._seafloor_idx is not None and i == int(self._seafloor_idx)
                is_m = self._moho_idx is not None and i == int(self._moho_idx)
                if i == n - 1:
                    pen = pg.mkPen("#111111", width=2, style=Qt.PenStyle.DashLine)
                elif is_m:
                    pen = pg.mkPen("#16a34a", width=2.6)  # Moho 绿
                elif is_b:
                    pen = pg.mkPen("#dc2626", width=2.6)  # Basement 红
                elif is_s:
                    pen = pg.mkPen("#2563eb", width=2.6)  # Seafloor 蓝
                else:
                    pen = pg.mkPen("#333333", width=1.4)
                curve = self.plot.plot(xv, zv, pen=pen)
                try:
                    curve.setZValue(18)  # 与 Contours 同层：高于波场叠层，低于炮点/OBS
                except Exception:
                    pass
                self._iface_items.append(curve)
            except Exception:
                continue
