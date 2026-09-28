# -*- coding: utf-8 -*-
"""输入页工区剖面预览：海底地形 + OBS 位置（范围跟 OBS 分布）。"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from matplotlib.figure import Figure


def survey_xlim_from_obs(
    obs_x: Optional[np.ndarray],
    *,
    seafloor_x: Optional[np.ndarray] = None,
    pad_frac: float = 0.08,
    pad_km_min: float = 5.0,
) -> Optional[tuple[float, float]]:
    """
    按 OBS 分布定横轴范围；无 OBS 时退回海底全段。

    边距 = max(跨度 * pad_frac, pad_km_min)。
    """
    if obs_x is not None and np.size(obs_x) > 0:
        xs = np.asarray(obs_x, dtype=float)
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return survey_xlim_from_obs(None, seafloor_x=seafloor_x)
        xmin = float(np.min(xs))
        xmax = float(np.max(xs))
        span = max(xmax - xmin, 1.0)
        pad = max(span * float(pad_frac), float(pad_km_min))
        return xmin - pad, xmax + pad
    if seafloor_x is not None and np.size(seafloor_x) > 0:
        sx = np.asarray(seafloor_x, dtype=float)
        sx = sx[np.isfinite(sx)]
        if sx.size >= 2:
            return float(np.min(sx)), float(np.max(sx))
    return None


class SurveyMapWidget(QWidget):
    """X–Z 剖面小图：海底曲线 + OBS 三角标注。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("IphaseSurveyMap")
        self.setMinimumHeight(200)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._hint = QLabel("工区剖面：指定海底地形与 OBS/炮点深度后自动刷新（范围按 OBS）")
        self._hint.setObjectName("IphaseCaption")
        self._hint.setWordWrap(True)
        lay.addWidget(self._hint)

        self.fig = Figure(figsize=(6.5, 2.2), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas, stretch=1)
        self._ax = self.fig.add_subplot(111)
        self._draw_empty("No bathymetry / OBS yet")

    def _draw_empty(self, msg: str) -> None:
        ax = self._ax
        ax.clear()
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#666")
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()

    def update_map(
        self,
        *,
        seafloor_x: Optional[np.ndarray] = None,
        seafloor_z: Optional[np.ndarray] = None,
        obs_x: Optional[np.ndarray] = None,
        obs_z: Optional[np.ndarray] = None,
        obs_labels: Optional[Sequence[str]] = None,
    ) -> None:
        has_sea = (
            seafloor_x is not None
            and seafloor_z is not None
            and np.size(seafloor_x) >= 2
            and np.size(seafloor_z) >= 2
        )
        has_obs = obs_x is not None and np.size(obs_x) > 0
        if not has_sea and not has_obs:
            self._hint.setText("工区剖面：请指定海底地形和/或 OBS/炮点深度")
            self._draw_empty("No bathymetry / OBS yet")
            return

        ax = self._ax
        ax.clear()
        xlim = survey_xlim_from_obs(
            np.asarray(obs_x, dtype=float) if has_obs else None,
            seafloor_x=np.asarray(seafloor_x, dtype=float) if has_sea else None,
        )

        if has_sea:
            sx = np.asarray(seafloor_x, dtype=float)
            sz = np.asarray(seafloor_z, dtype=float)
            v = np.isfinite(sx) & np.isfinite(sz)
            sx, sz = sx[v], sz[v]
            if xlim is not None:
                # 略扩一点再裁，避免边缘断线
                lo, hi = xlim[0] - 1.0, xlim[1] + 1.0
                m = (sx >= lo) & (sx <= hi)
                if np.count_nonzero(m) >= 2:
                    sx, sz = sx[m], sz[m]
            ax.plot(sx, sz, "-", color="#1f77b4", linewidth=1.4, label="seafloor")

        n_obs = 0
        if has_obs:
            ox = np.asarray(obs_x, dtype=float)
            if obs_z is not None and np.size(obs_z) == np.size(ox):
                oz = np.asarray(obs_z, dtype=float)
            elif has_sea:
                oz = np.interp(
                    ox,
                    np.asarray(seafloor_x, dtype=float),
                    np.asarray(seafloor_z, dtype=float),
                    left=np.nan,
                    right=np.nan,
                )
            else:
                oz = np.zeros_like(ox)
            v = np.isfinite(ox) & np.isfinite(oz)
            ox, oz = ox[v], oz[v]
            n_obs = int(ox.size)
            if n_obs:
                ax.scatter(
                    ox,
                    oz,
                    marker="^",
                    s=42,
                    c="#d62728",
                    edgecolors="k",
                    linewidths=0.5,
                    zorder=5,
                    label="OBS",
                )
                labels = list(obs_labels) if obs_labels is not None else []
                for i, (xx, zz) in enumerate(zip(ox, oz)):
                    lab = ""
                    if i < len(labels) and str(labels[i]).strip():
                        lab = str(labels[i]).strip()
                    if lab:
                        ax.annotate(
                            lab,
                            (xx, zz),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="#a00000",
                        )

        if xlim is not None:
            ax.set_xlim(xlim)
        ax.invert_yaxis()
        ax.set_xlabel("Model distance (km)", fontsize=8)
        ax.set_ylabel("Depth (km)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if has_sea or n_obs:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        self.fig.tight_layout(pad=0.35)

        if xlim is not None:
            self._hint.setText(
                f"工区剖面：X∈[{xlim[0]:.1f}, {xlim[1]:.1f}] km"
                f"（按 OBS 分布留边）；OBS={n_obs}"
            )
        else:
            self._hint.setText(f"工区剖面已更新；OBS={n_obs}")
        self.canvas.draw_idle()
