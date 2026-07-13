"""Fig.12a background (digitized KKHS02 standard H–Vp diagram)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_HVP_DIGITIZED = (
    Path(__file__).resolve().parents[1] / "data" / "standard H-Vp diagram.txt"
)

H_LIM = (3.0, 35.0)
VP_LIM = (6.9, 7.36)

CHI_SOLID = (1.0, 2.0, 4.0, 8.0)
CHI1_B_DASHED = (10.0, 20.0, 30.0)
TP_CONTOURS = (1300, 1350, 1400, 1450, 1500, 1550)


@dataclass(frozen=True)
class CurveSeries:
    label: str
    kind: str
    h_km: np.ndarray
    vp_km_s: np.ndarray
    chi: float | None = None
    b_km: float | None = None
    tp_c: float | None = None


def parse_hvp_digitized(path: Path) -> list[CurveSeries]:
    series: list[CurveSeries] = []
    key: str | None = None
    pts: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal key, pts
        if key is None or not pts:
            key = None
            pts = []
            return
        h = np.array([p[0] for p in pts], dtype=float)
        v = np.array([p[1] for p in pts], dtype=float)
        order = np.argsort(h)
        h, v = h[order], v[order]
        uniq_h, idx = np.unique(h, return_index=True)
        h, v = uniq_h, v[idx]

        m_chi = re.match(r"^[xX]=(\d+(?:\.\d+)?)\s*,\s*b=(\d+(?:\.\d+)?)$", key.strip())
        m_tp = re.match(r"^T=(\d+(?:\.\d+)?)$", key.strip(), re.I)
        if m_chi:
            chi = float(m_chi.group(1))
            b = float(m_chi.group(2))
            series.append(
                CurveSeries(
                    label=f"χ={chi:g}, b={b:g} km",
                    kind="track",
                    h_km=h,
                    vp_km_s=v,
                    chi=chi,
                    b_km=b,
                )
            )
        elif m_tp:
            tp = float(m_tp.group(1))
            series.append(
                CurveSeries(
                    label=f"{tp:.0f}°C",
                    kind="tp",
                    h_km=h,
                    vp_km_s=v,
                    tp_c=tp,
                )
            )
        key = None
        pts = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("Generated"):
            continue
        if re.match(r"^[xX]=\s*\d", line) or re.match(r"^T=\s*\d", line, re.I):
            flush()
            key = line
            continue
        if key is None:
            continue
        parts = line.split()
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    flush()
    return series


def smooth_spline(h: np.ndarray, v: np.ndarray, *, n_eval: int = 200) -> tuple[np.ndarray, np.ndarray]:
    from scipy.interpolate import PchipInterpolator

    if len(h) < 2:
        return h, v
    h_fine = np.linspace(float(h.min()), float(h.max()), n_eval)
    return h_fine, PchipInterpolator(h, v)(h_fine)


def find_series(
    all_series: list[CurveSeries],
    *,
    chi: float | None = None,
    b_km: float | None = None,
    tp_c: float | None = None,
) -> CurveSeries | None:
    for s in all_series:
        if tp_c is not None and s.kind == "tp" and s.tp_c is not None and abs(s.tp_c - tp_c) < 0.5:
            return s
        if chi is not None and b_km is not None and s.kind == "track":
            if s.chi is not None and s.b_km is not None:
                if abs(s.chi - chi) < 1e-9 and abs(s.b_km - b_km) < 1e-9:
                    return s
    return None


def draw_fig12a_background(
    ax,
    all_series: list[CurveSeries],
    *,
    h_lim: tuple[float, float] = H_LIM,
    vp_lim: tuple[float, float] = VP_LIM,
    show_legend: bool = True,
) -> None:
    """Draw digitized Fig.12a model curves on ``ax``."""
    import matplotlib.pyplot as plt

    for tp in TP_CONTOURS:
        s = find_series(all_series, tp_c=float(tp))
        if s is None:
            continue
        hf, vf = smooth_spline(s.h_km, s.vp_km_s)
        ax.plot(hf, vf, color="0.72", lw=0.9, ls="--", alpha=0.95, zorder=1)
        i_lab = int(0.55 * (len(hf) - 1))
        x_lab = float(np.clip(hf[i_lab], h_lim[0] + 0.4, h_lim[1] - 1.2))
        y_lab = float(np.clip(vf[i_lab], vp_lim[0] + 0.008, vp_lim[1] - 0.012))
        ax.text(x_lab, y_lab, f"{tp:.0f}°C", fontsize=7.5, color="0.42", ha="center")

    cmap = plt.cm.viridis
    for i, chi in enumerate(CHI_SOLID):
        s = find_series(all_series, chi=chi, b_km=0.0)
        if s is None:
            continue
        hf, vf = smooth_spline(s.h_km, s.vp_km_s)
        col = cmap(i / max(len(CHI_SOLID) - 1, 1))
        ax.plot(hf, vf, color=col, lw=1.6, zorder=3, label=f"χ={chi:g}, b=0")

    for j, b_km in enumerate(CHI1_B_DASHED):
        s = find_series(all_series, chi=1.0, b_km=b_km)
        if s is None:
            continue
        hf, vf = smooth_spline(s.h_km, s.vp_km_s)
        shade = 0.12 + 0.22 * (j / max(len(CHI1_B_DASHED) - 1, 1))
        ax.plot(
            hf,
            vf,
            color=str(shade),
            lw=1.2,
            ls="--",
            dashes=(4.5, 2.5),
            zorder=2,
            label=f"χ=1, b={b_km:g} km",
        )

    ax.set_xlim(h_lim)
    ax.set_ylim(vp_lim)
    if show_legend:
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9, ncol=1)
